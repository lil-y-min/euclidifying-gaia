#!/usr/bin/env python3
"""
Embed 8D Gaia features into 2D (UMAP preferred, TSNE fallback), then
analyze where close double-star candidates (< threshold arcsec) sit in
feature space and their local neighborhoods.

Inputs:
- output/dataset_npz/*/metadata.csv (8D features + source_id)
- output/crossmatch/wds/wds_xmatch/*_wds_best.csv (WDS nearest match)

Outputs:
- plots/qa/embeddings/double_stars_8d/
- output/experiments/embeddings/double_stars_8d/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Avoid OpenMP shared-memory/threading issues in constrained environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_schema import get_feature_cols, normalize_feature_set


FEATURE_COLS_8D = [
    "feat_log10_snr",
    "feat_ruwe",
    "feat_astrometric_excess_noise",
    "feat_parallax_over_error",
    "feat_visibility_periods_used",
    "feat_ipd_frac_multi_peak",
    "feat_c_star",
    "feat_pm_significance",
]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def infer_field_tag(path_str: str) -> str:
    try:
        p = Path(path_str)
        return p.parent.name if p.parent.name else "unknown"
    except Exception:
        return "unknown"


def load_metadata(dataset_root: Path, feature_cols: List[str], metadata_name: str = "metadata.csv") -> pd.DataFrame:
    usecols = ["source_id", "phot_g_mean_mag", "fits_path", *feature_cols]
    dfs = []

    for meta_path in sorted(dataset_root.glob(f"*/{metadata_name}")):
        if not meta_path.exists():
            continue
        df = pd.read_csv(meta_path, usecols=lambda c: c in set(usecols), low_memory=False)
        # Numeric coercion for robust merge/embedding.
        for c in ["source_id", "phot_g_mean_mag", *feature_cols]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "fits_path" in df.columns:
            df["field_tag"] = df["fits_path"].astype(str).map(infer_field_tag)
        else:
            df["field_tag"] = meta_path.parent.name

        dfs.append(df[["source_id", "phot_g_mean_mag", "field_tag", *feature_cols]])

    if not dfs:
        raise RuntimeError(f"No {metadata_name} found under {dataset_root}")

    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=["source_id", *feature_cols]).copy()
    out["source_id"] = out["source_id"].astype(np.int64)

    # Keep one row per Gaia source (same source can appear across fields).
    out = out.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return out


def load_wds_labels(wds_dir: Path, threshold_arcsec: float, threshold_mode: str = "le") -> pd.DataFrame:
    usecols = ["source_id", "wds_dist_arcsec", "wds_lstsep", "wds_comp", "wds_disc", "field_tag"]
    dfs = []

    for p in sorted(wds_dir.glob("*_wds_best.csv")):
        df = pd.read_csv(p, usecols=lambda c: c in set(usecols), low_memory=False)
        if "source_id" not in df.columns:
            continue

        for c in ["source_id", "wds_dist_arcsec", "wds_lstsep"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["source_id"]).copy()
        df["source_id"] = df["source_id"].astype(np.int64)

        # Mark doubles by WDS match distance threshold.
        if threshold_mode == "lt":
            sel = df["wds_dist_arcsec"] < threshold_arcsec
        else:
            sel = df["wds_dist_arcsec"] <= threshold_arcsec
        df["is_double"] = df["wds_dist_arcsec"].notna() & sel
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No *_wds_best.csv found under {wds_dir}")

    w = pd.concat(dfs, ignore_index=True)
    # For each source, keep the closest WDS distance if repeated.
    w = w.sort_values(["source_id", "wds_dist_arcsec"], ascending=[True, True], na_position="last")
    w = w.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return w[["source_id", "is_double", "wds_dist_arcsec", "wds_lstsep", "wds_comp", "wds_disc"]]


def choose_backend(method: str) -> str:
    method = method.lower().strip()
    if method != "umap":
        raise ValueError("Only UMAP is supported now. Use --method umap.")

    try:
        import umap  # noqa: F401
        return "umap"
    except ModuleNotFoundError:
        if method == "umap":
            raise RuntimeError("Requested --method umap but umap-learn is not installed.")
        return "tsne"
    except Exception as e:
        if method == "umap":
            raise RuntimeError(f"Requested --method umap but UMAP failed to initialize: {e}") from e
        raise RuntimeError(f"UMAP failed to initialize: {e}") from e


def build_embedding(X: np.ndarray, backend: str, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    if backend != "umap":
        raise ValueError("Only UMAP backend is supported.")

    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        init="spectral",
    )
    emb = reducer.fit_transform(X)
    return emb


def scale_features(X: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower().strip()
    if mode == "standard":
        return StandardScaler().fit_transform(X)
    if mode == "minmax":
        return MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(X)
    if mode == "none":
        return X
    raise ValueError("scale mode must be one of: standard, minmax, none")


def parse_seed_list(text: str) -> List[int]:
    t = (text or "").strip()
    if not t:
        return []
    out = []
    for part in t.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    # preserve order, drop duplicates
    seen = set()
    uniq = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def knn_overlap_fraction(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    k: int = 50,
    sample_n: int = 15000,
    seed: int = 123,
) -> float:
    n = emb_a.shape[0]
    if n < 3:
        return float("nan")
    k_eff = int(max(2, min(k, n - 1)))
    if sample_n is not None and n > int(sample_n):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=int(sample_n), replace=False))
        a = emb_a[idx]
        b = emb_b[idx]
    else:
        a = emb_a
        b = emb_b
    nn_a = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(a)
    nn_b = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(b)
    ia = nn_a.kneighbors(a, return_distance=False)[:, 1:]
    ib = nn_b.kneighbors(b, return_distance=False)[:, 1:]
    frac = []
    for xa, xb in zip(ia, ib):
        sa = set(xa.tolist())
        sb = set(xb.tolist())
        frac.append(len(sa.intersection(sb)) / float(k_eff))
    return float(np.mean(frac))


def compute_local_double_fraction(emb: np.ndarray, is_double: np.ndarray, k: int = 50) -> np.ndarray:
    k_eff = int(max(5, min(k, len(emb) - 1)))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(emb)
    idx = nn.kneighbors(emb, return_distance=False)
    neigh = idx[:, 1:]
    return is_double[neigh].mean(axis=1)


def cluster_doubles_dbscan(emb: np.ndarray, is_double: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.full(len(emb), -1, dtype=int)
    d_idx = np.where(is_double)[0]
    if len(d_idx) < 8:
        return labels

    xd = emb[d_idx]
    # Data-driven eps: median distance to 5th neighbor among doubles.
    nn = NearestNeighbors(n_neighbors=min(6, len(xd)), metric="euclidean").fit(xd)
    dists, _ = nn.kneighbors(xd)
    kth = dists[:, -1]
    eps = float(np.quantile(kth, 0.60))
    eps = max(eps, 1e-3)

    min_samples = int(np.clip(round(np.sqrt(len(xd)) / 2), 4, 20))
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    cl = model.fit_predict(xd)

    labels[d_idx] = cl
    return labels


def plot_main_map(df: pd.DataFrame, out_png: Path, title: str, threshold_text: str = '<=1.0"') -> None:
    plt.figure(figsize=(10, 8))
    plt.hexbin(df["x"], df["y"], gridsize=170, bins="log", cmap="Greys", mincnt=1)
    d = df[df["is_double"]]
    plt.scatter(d["x"], d["y"], s=18, c="#e63946", alpha=0.95, edgecolors="white", linewidths=0.25, label=f"WDS double ({threshold_text})")
    plt.xlabel("Embedding X")
    plt.ylabel("Embedding Y")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Background density (log counts)")
    plt.legend(loc="best", frameon=True)
    savefig(out_png)


def plot_local_enrichment(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        df["x"],
        df["y"],
        c=df["local_double_frac_k50"],
        s=6,
        cmap="viridis",
        alpha=0.8,
        linewidths=0,
    )
    d = df[df["is_double"]]
    plt.scatter(d["x"], d["y"], s=16, facecolors="none", edgecolors="#f94144", linewidths=0.5, label="doubles")
    plt.xlabel("Embedding X")
    plt.ylabel("Embedding Y")
    plt.title("Local enrichment of doubles (k=50 neighbors in 2D embedding)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Fraction doubles in local neighborhood")
    plt.legend(loc="best", frameon=True)
    savefig(out_png)


def plot_doubles_by_wds_dist(df: pd.DataFrame, out_png: Path) -> None:
    d = df[df["is_double"] & df["wds_dist_arcsec"].notna()].copy()
    plt.figure(figsize=(9, 7))
    if len(d) == 0:
        plt.text(0.5, 0.5, "No doubles found under threshold", ha="center", va="center")
        plt.axis("off")
        savefig(out_png)
        return

    sc = plt.scatter(d["x"], d["y"], c=d["wds_dist_arcsec"], s=24, cmap="magma_r", alpha=0.95, edgecolors="white", linewidths=0.2)
    plt.xlabel("Embedding X")
    plt.ylabel("Embedding Y")
    plt.title('Double-star subset colored by WDS match distance (arcsec)')
    cbar = plt.colorbar(sc)
    cbar.set_label("wds_dist_arcsec")
    savefig(out_png)


def plot_double_clusters(df: pd.DataFrame, out_png: Path) -> None:
    d = df[df["is_double"]].copy()
    plt.figure(figsize=(9, 7))
    if len(d) == 0:
        plt.text(0.5, 0.5, "No doubles found under threshold", ha="center", va="center")
        plt.axis("off")
        savefig(out_png)
        return

    noise = d[d["double_cluster"] < 0]
    core = d[d["double_cluster"] >= 0]

    if len(noise) > 0:
        plt.scatter(noise["x"], noise["y"], s=22, c="lightgray", alpha=0.9, label="noise")

    if len(core) > 0:
        sc = plt.scatter(core["x"], core["y"], s=26, c=core["double_cluster"], cmap="tab20", alpha=0.95, edgecolors="white", linewidths=0.2)
        cbar = plt.colorbar(sc)
        cbar.set_label("DBSCAN cluster id")

    plt.xlabel("Embedding X")
    plt.ylabel("Embedding Y")
    plt.title("DBSCAN clusters within double-star subset")
    plt.legend(loc="best", frameon=True)
    savefig(out_png)


def plot_neighborhood_examples(df: pd.DataFrame, out_png: Path, n_examples: int = 6) -> None:
    d = df[df["is_double"]].copy()
    if len(d) == 0:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No doubles found", ha="center", va="center")
        plt.axis("off")
        savefig(out_png)
        return

    # Pick doubles in the most enriched neighborhoods.
    d = d.sort_values("local_double_frac_k50", ascending=False).head(n_examples)

    emb = df[["x", "y"]].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=min(121, len(df)), metric="euclidean").fit(emb)

    n = len(d)
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(5 * int(np.ceil(n / 2)), 8))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, d.iterrows()):
        center = np.array([[row["x"], row["y"]]], dtype=float)
        idx = nn.kneighbors(center, return_distance=False)[0]
        neigh = df.iloc[idx]

        ax.scatter(neigh["x"], neigh["y"], s=10, c="0.75", alpha=0.7)
        nd = neigh[neigh["is_double"]]
        if len(nd) > 0:
            ax.scatter(nd["x"], nd["y"], s=18, c="#e63946", alpha=0.95)
        ax.scatter([row["x"]], [row["y"]], s=70, marker="*", c="gold", edgecolors="black", linewidths=0.6)

        ax.set_title(f"src {int(row['source_id'])}\nlocal_frac={row['local_double_frac_k50']:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Neighborhood snapshots around high-enrichment doubles", fontsize=13)
    savefig(out_png)


def plot_feature_color_panels(df: pd.DataFrame, feature_cols: List[str], out_png: Path) -> None:
    n = len(feature_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, c in zip(axes, feature_cols):
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if np.any(mask):
            lo, hi = np.nanpercentile(vals[mask], [1, 99])
            vals = np.clip(vals, lo, hi)
        sc = ax.scatter(df["x"], df["y"], c=vals, s=5, cmap="viridis", alpha=0.75, linewidths=0)
        d = df[df["is_double"]]
        ax.scatter(d["x"], d["y"], s=12, facecolors="none", edgecolors="#f94144", linewidths=0.4)
        ax.set_title(c, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Embedding color-coded by Gaia features", fontsize=13)
    savefig(out_png)

def select_prefixed_cols(df: pd.DataFrame, prefixes: List[str], max_cols: int = 12) -> List[str]:
    out: List[str] = []
    for c in df.columns:
        if any(c.startswith(p) for p in prefixes):
            out.append(c)
    return out[:max_cols]


def plot_field_tag_colors(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 8))
    tags = sorted(df["field_tag"].dropna().astype(str).unique().tolist())
    cmap = plt.get_cmap("tab20", max(len(tags), 1))
    for i, tag in enumerate(tags):
        sub = df[df["field_tag"] == tag]
        plt.scatter(sub["x"], sub["y"], s=5, color=cmap(i), alpha=0.6, linewidths=0, label=tag)
    d = df[df["is_double"]]
    plt.scatter(d["x"], d["y"], s=16, facecolors="none", edgecolors="black", linewidths=0.5, label="doubles")
    plt.xlabel("Embedding X")
    plt.ylabel("Embedding Y")
    plt.title("Embedding color-coded by field")
    if len(tags) <= 20:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
    savefig(out_png)


def analyze_cluster_neighbors(df: pd.DataFrame, n_per_cluster: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d_core = df[(df["is_double"]) & (df["double_cluster"] >= 0)].copy()
    if d_core.empty:
        return pd.DataFrame(), pd.DataFrame()

    cent = (
        d_core.groupby("double_cluster")[["x", "y"]]
        .mean()
        .reset_index()
        .rename(columns={"x": "cluster_cx", "y": "cluster_cy"})
    )
    sizes = d_core.groupby("double_cluster").size().reset_index(name="n_double_in_cluster")
    cent = cent.merge(sizes, on="double_cluster", how="left")

    non = df[~df["is_double"]].copy()
    rows = []
    for _, r in cent.iterrows():
        c = int(r["double_cluster"])
        dx = non["x"].to_numpy(dtype=float) - float(r["cluster_cx"])
        dy = non["y"].to_numpy(dtype=float) - float(r["cluster_cy"])
        dist = np.sqrt(dx * dx + dy * dy)
        ord_idx = np.argsort(dist)[: int(n_per_cluster)]
        sub = non.iloc[ord_idx].copy()
        sub["double_cluster"] = c
        sub["dist_to_cluster_centroid"] = dist[ord_idx]
        sub["rank_nearest_to_cluster"] = np.arange(1, len(sub) + 1)
        rows.append(
            sub[
                [
                    "double_cluster",
                    "rank_nearest_to_cluster",
                    "dist_to_cluster_centroid",
                    "source_id",
                    "field_tag",
                    "phot_g_mean_mag",
                    "local_double_frac_k50",
                    "x",
                    "y",
                    *FEATURE_COLS_8D,
                ]
            ]
        )

    neigh = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return cent, neigh


def plot_cluster_neighbor_map(df: pd.DataFrame, centroids: pd.DataFrame, neigh: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 8))
    plt.scatter(df["x"], df["y"], s=4, c="0.85", alpha=0.5, linewidths=0)
    d = df[df["is_double"]]
    plt.scatter(d["x"], d["y"], s=18, c="#e63946", alpha=0.95, linewidths=0, label="doubles")

    if not neigh.empty:
        plt.scatter(
            neigh["x"],
            neigh["y"],
            s=20,
            c=neigh["double_cluster"],
            cmap="tab20",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.2,
            label="near-cluster non-doubles",
        )
    if not centroids.empty:
        plt.scatter(
            centroids["cluster_cx"],
            centroids["cluster_cy"],
            s=120,
            marker="X",
            c=centroids["double_cluster"],
            cmap="tab20",
            edgecolors="black",
            linewidths=0.5,
            label="cluster centroids",
        )
    plt.xlabel("Embedding X")
    plt.ylabel("Embedding Y")
    plt.title("Non-double sources nearest to double-star clusters")
    plt.legend(loc="best", frameon=True)
    savefig(out_png)


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    n = int(len(df))
    nd = int(df["is_double"].sum())
    frac = float(nd / max(n, 1))
    stats = {
        "n_points_embedded": n,
        "n_double_selected": nd,
        "frac_double": frac,
        "local_double_frac_k50_mean_double": float(df.loc[df["is_double"], "local_double_frac_k50"].mean()) if nd else 0.0,
        "local_double_frac_k50_mean_non_double": float(df.loc[~df["is_double"], "local_double_frac_k50"].mean()) if n - nd else 0.0,
    }
    return stats


def plot_scaling_comparison(
    embed_by_scaling: Dict[str, pd.DataFrame],
    out_png: Path,
    threshold_text: str,
) -> None:
    order = ["none", "minmax", "standard"]
    keys = [k for k in order if k in embed_by_scaling]
    if len(keys) != 3:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=False, sharey=False)
    for ax, s in zip(axes, keys):
        d = embed_by_scaling[s]
        ax.hexbin(d["x"], d["y"], gridsize=100, bins="log", cmap="Greys", mincnt=1)
        ds = d[d["is_double"]]
        ax.scatter(ds["x"], ds["y"], s=12, c="#e63946", alpha=0.95, edgecolors="white", linewidths=0.25)
        ax.set_title(f"UMAP with scaling={s}")
        ax.set_xlabel("Embedding X")
        ax.set_ylabel("Embedding Y")
        ax.text(0.02, 0.02, f"doubles: {int(ds.shape[0])} ({threshold_text})", transform=ax.transAxes, fontsize=9)
    fig.suptitle("Effect of feature scaling on UMAP geometry", fontsize=14)
    savefig(out_png)


def run_single_scaling(
    dfe_base: pd.DataFrame,
    backend: str,
    scaling: str,
    feature_cols: List[str],
    feature_label: str,
    args,
    out_plot: Path,
    out_tab: Path,
) -> pd.DataFrame:
    dfe = dfe_base.copy()
    X = dfe[feature_cols].to_numpy(dtype=np.float32)
    X = scale_features(X, mode=scaling)
    if bool(args.shuffle_before_umap):
        rng_shuffle = np.random.default_rng(int(args.shuffle_seed))
        perm = rng_shuffle.permutation(len(dfe))
        dfe = dfe.iloc[perm].reset_index(drop=True)
        X = X[perm]
        print(f"[SHUFFLE] permuted rows before UMAP using shuffle_seed={args.shuffle_seed}")

    print(f"Fitting {backend.upper()} on {len(dfe):,} points with scaling={scaling}...")
    emb = build_embedding(X, backend=backend, seed=args.seed, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    dfe["x"] = emb[:, 0]
    dfe["y"] = emb[:, 1]

    print("Computing neighborhood enrichment + clustering...")
    dfe["local_double_frac_k50"] = compute_local_double_fraction(
        dfe[["x", "y"]].to_numpy(dtype=np.float32),
        dfe["is_double"].to_numpy(dtype=bool),
        k=50,
    )
    dfe["double_cluster"] = cluster_doubles_dbscan(
        dfe[["x", "y"]].to_numpy(dtype=np.float32),
        dfe["is_double"].to_numpy(dtype=bool),
        seed=args.seed,
    )

    emb_csv = out_tab / f"embedding_{backend}.csv"
    dfe.to_csv(emb_csv, index=False)

    d_only = dfe[dfe["is_double"]].copy()
    ranked_cols = [
        "source_id",
        "field_tag",
        "phot_g_mean_mag",
        "wds_dist_arcsec",
        "wds_lstsep",
        "local_double_frac_k50",
        "double_cluster",
        "x",
        "y",
    ]
    d_only.sort_values("local_double_frac_k50", ascending=False)[ranked_cols].to_csv(
        out_tab / f"double_candidates_ranked_by_local_enrichment_{backend}.csv",
        index=False,
    )

    cl_tab = (
        d_only.groupby("double_cluster", dropna=False)
        .size()
        .reset_index(name="n_points")
        .sort_values("n_points", ascending=False)
    )
    cl_tab.to_csv(out_tab / f"double_cluster_summary_{backend}.csv", index=False)

    cent, neigh = analyze_cluster_neighbors(dfe, n_per_cluster=int(args.cluster_neighbor_n))
    if not cent.empty:
        cent.to_csv(out_tab / f"double_cluster_centroids_{backend}.csv", index=False)
    if not neigh.empty:
        neigh.to_csv(out_tab / f"non_double_neighbors_near_clusters_{backend}.csv", index=False)

    stats = summarize(dfe)
    stats.update(
        {
            "backend": backend,
            "double_threshold_arcsec": float(args.double_threshold_arcsec),
            "double_condition": str(args.double_condition),
            "scaling": str(scaling),
            "wds_dir": str(args.wds_dir_resolved),
            "feature_cols": feature_cols,
            "feature_set": feature_label,
            "shuffle_before_umap": bool(args.shuffle_before_umap),
            "shuffle_seed": int(args.shuffle_seed),
        }
    )
    if args.stability_seeds_parsed:
        rows = []
        for s in args.stability_seeds_parsed:
            if int(s) == int(args.seed):
                continue
            print(f"[STABILITY] fitting additional seed={s} ...")
            emb_s = build_embedding(
                X,
                backend=backend,
                seed=int(s),
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
            )
            ov = knn_overlap_fraction(
                emb,
                emb_s,
                k=int(args.stability_k),
                sample_n=int(args.stability_sample_n),
                seed=int(args.seed),
            )
            rows.append({"seed_ref": int(args.seed), "seed_alt": int(s), "k": int(args.stability_k), "sample_n": int(args.stability_sample_n), "mean_knn_overlap": ov})
            print(f"[STABILITY] seed {s}: mean_kNN_overlap@{args.stability_k}={ov:.4f}")
        if rows:
            stab_csv = out_tab / f"umap_seed_stability_{backend}.csv"
            pd.DataFrame(rows).to_csv(stab_csv, index=False)
            stats["seed_stability_csv"] = str(stab_csv)
            stats["seed_stability_n_runs"] = int(len(rows))
            stats["seed_stability_k"] = int(args.stability_k)
            stats["seed_stability_sample_n"] = int(args.stability_sample_n)
            print("[STABILITY] wrote:", stab_csv)

    with open(out_tab / f"summary_{backend}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    threshold_text = f"{'<' if args.double_condition == 'lt' else '<='}{args.double_threshold_arcsec:.3g}\""
    title = f"{backend.upper()} projection of Gaia {feature_label} features with close doubles overlay"
    plot_main_map(dfe, out_plot / f"01_{backend}_main_map_density_plus_doubles.png", title, threshold_text=threshold_text)
    plot_local_enrichment(dfe, out_plot / f"02_{backend}_local_double_enrichment_k50.png")
    plot_doubles_by_wds_dist(dfe, out_plot / f"03_{backend}_doubles_colored_by_wds_dist.png")
    plot_double_clusters(dfe, out_plot / f"04_{backend}_double_clusters_dbscan.png")
    plot_neighborhood_examples(dfe, out_plot / f"05_{backend}_double_neighborhood_examples.png", n_examples=6)
    plot_feature_color_panels(dfe, feature_cols, out_plot / f"06_{backend}_features_colored_panels.png")
    plot_field_tag_colors(dfe, out_plot / f"07_{backend}_color_by_field_tag.png")
    plot_cluster_neighbor_map(dfe, cent, neigh, out_plot / f"08_{backend}_cluster_neighbor_map.png")

    # Optional: color Gaia-feature embedding by extra (e.g. morph/pixel-derived) columns.
    if args.extra_features_csv:
        p = Path(args.extra_features_csv)
        if p.exists():
            ex = pd.read_csv(p, low_memory=False)
            if "source_id" in ex.columns:
                ex["source_id"] = pd.to_numeric(ex["source_id"], errors="coerce")
                ex = ex.dropna(subset=["source_id"]).copy()
                ex["source_id"] = ex["source_id"].astype(np.int64)
                ex = ex.drop_duplicates(subset=["source_id"], keep="first")
                dfe_ex = dfe.merge(ex, on="source_id", how="left", suffixes=("", "_extra"))
                prefs = [s.strip() for s in str(args.extra_feature_prefixes).split(",") if s.strip()]
                cols = select_prefixed_cols(dfe_ex, prefs, max_cols=int(args.extra_feature_max))
                if cols:
                    plot_feature_color_panels(dfe_ex, cols, out_plot / f"09_{backend}_extra_features_colored_panels.png")
        else:
            print(f"[WARN] extra_features_csv not found: {p}")
    return dfe


def main() -> None:
    ap = argparse.ArgumentParser(description="2D embedding of 8D Gaia features for double-star clustering checks")
    ap.add_argument("--feature_set", default="8D", help="Gaia feature set: 8D/10D/16D (17D alias -> 16D)")
    ap.add_argument("--metadata_name", default="metadata.csv", help="Per-field metadata filename (e.g. metadata_16d.csv)")
    ap.add_argument("--out_tag", default="", help="Optional output subfolder tag (e.g. gaia_16d_phase3_20260216)")
    ap.add_argument("--extra_features_csv", default="", help="Optional CSV with source_id + extra features for color panels")
    ap.add_argument("--extra_feature_prefixes", default="morph_", help="Comma-separated column prefixes to plot from extra CSV")
    ap.add_argument("--extra_feature_max", type=int, default=12, help="Max number of extra features to panel-plot")
    ap.add_argument("--method", default="umap", choices=["umap"], help="Embedding backend")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--double_threshold_arcsec", type=float, default=1.0)
    ap.add_argument("--double_condition", default="le", choices=["le", "lt"], help="Double selection: <= threshold (le) or < threshold (lt)")
    ap.add_argument("--scaling", default="standard", choices=["standard", "minmax", "none"], help="Feature scaling before embedding")
    ap.add_argument("--compare_scaling", action="store_true", help="Run UMAP for none/minmax/standard and save comparison plots")
    ap.add_argument("--n_neighbors", type=int, default=30, help="UMAP n_neighbors")
    ap.add_argument("--min_dist", type=float, default=0.08, help="UMAP min_dist")
    ap.add_argument("--max_points", type=int, default=120000, help="Max points for UMAP")
    ap.add_argument("--cluster_neighbor_n", type=int, default=30, help="Nearest non-double sources exported per double cluster")
    ap.add_argument("--wds_dir", default=None, help="Folder containing *_wds_best.csv")
    ap.add_argument("--shuffle_before_umap", action="store_true", help="Shuffle row order before fitting UMAP (sanity check for order effects)")
    ap.add_argument("--shuffle_seed", type=int, default=123, help="Seed used for optional pre-UMAP shuffle")
    ap.add_argument("--stability_seeds", default="", help="Comma-separated extra UMAP seeds for stability checks, e.g. '7,42,123'")
    ap.add_argument("--stability_k", type=int, default=50, help="k for kNN-overlap stability metric")
    ap.add_argument("--stability_sample_n", type=int, default=15000, help="Max sample size for stability kNN-overlap")
    args = ap.parse_args()
    args.stability_seeds_parsed = parse_seed_list(args.stability_seeds)

    base = Path(__file__).resolve().parents[1]
    dataset_root = base / "output" / "dataset_npz"
    wds_dir = Path(args.wds_dir) if args.wds_dir else (base / "output" / "crossmatch" / "wds" / "wds_xmatch")
    args.wds_dir_resolved = wds_dir

    feature_set_norm = normalize_feature_set(args.feature_set)
    feature_cols = get_feature_cols(feature_set_norm)
    feature_label = feature_set_norm.lower()

    if str(args.out_tag).strip():
        out_plot_root = base / "plots" / "qa" / "embeddings" / str(args.out_tag).strip()
        out_tab_root = base / "output" / "experiments" / "embeddings" / str(args.out_tag).strip()
    else:
        out_plot_root = base / "plots" / "qa" / "embeddings" / f"double_stars_gaia_{feature_label}"
        out_tab_root = base / "output" / "experiments" / "embeddings" / f"double_stars_gaia_{feature_label}"

    out_plot = ensure_dir(out_plot_root)
    out_tab = ensure_dir(out_tab_root)

    print("Loading metadata...")
    m = load_metadata(dataset_root, feature_cols, metadata_name=str(args.metadata_name))
    print(f"  metadata unique sources: {len(m):,}")

    print("Loading WDS labels...")
    w = load_wds_labels(
        wds_dir,
        threshold_arcsec=float(args.double_threshold_arcsec),
        threshold_mode=str(args.double_condition),
    )
    print(f"  WDS unique sources: {len(w):,}")

    df = m.merge(w, on="source_id", how="left")
    df["is_double"] = df["is_double"].fillna(False).astype(bool)

    backend = choose_backend(args.method)
    print(f"Embedding backend: {backend}")

    rng = np.random.default_rng(args.seed)
    idx_double = np.where(df["is_double"].to_numpy())[0]
    idx_non = np.where(~df["is_double"].to_numpy())[0]

    max_points = int(args.max_points)
    n_keep_non = max(0, min(len(idx_non), max_points - len(idx_double)))

    if n_keep_non < len(idx_non):
        keep_non = rng.choice(idx_non, size=n_keep_non, replace=False)
    else:
        keep_non = idx_non

    keep = np.concatenate([idx_double, keep_non])
    keep = np.unique(keep)
    dfe_base = df.iloc[keep].copy().reset_index(drop=True)
    print(f"Sampled {len(dfe_base):,} points (doubles={int(dfe_base['is_double'].sum()):,})")

    if args.compare_scaling:
        all_embeds: Dict[str, pd.DataFrame] = {}
        for scaling in ["none", "minmax", "standard"]:
            out_plot_s = ensure_dir(out_plot / f"umap_{scaling}")
            out_tab_s = ensure_dir(out_tab / f"umap_{scaling}")
            all_embeds[scaling] = run_single_scaling(
                dfe_base=dfe_base,
                backend=backend,
                scaling=scaling,
                feature_cols=feature_cols,
                feature_label=feature_label,
                args=args,
                out_plot=out_plot_s,
                out_tab=out_tab_s,
            )
            print("  Outputs:", out_plot_s, "|", out_tab_s)

        threshold_text = f"{'<' if args.double_condition == 'lt' else '<='}{args.double_threshold_arcsec:.3g}\""
        plot_scaling_comparison(
            embed_by_scaling=all_embeds,
            out_png=out_plot / "umap_scaling_comparison_main_map.png",
            threshold_text=threshold_text,
        )
        print("Done.")
        print("  Comparison plot:", out_plot / "umap_scaling_comparison_main_map.png")
        return

    dfe = run_single_scaling(
        dfe_base=dfe_base,
        backend=backend,
        scaling=str(args.scaling),
        feature_cols=feature_cols,
        feature_label=feature_label,
        args=args,
        out_plot=out_plot,
        out_tab=out_tab,
    )
    print("Done.")
    print("  Table outputs:", out_tab)
    print("  Plot outputs :", out_plot)


if __name__ == "__main__":
    main()
