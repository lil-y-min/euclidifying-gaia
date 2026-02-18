#!/usr/bin/env python3
"""
UMAP exploratory analysis pack for Gaia 8D embedding outputs.

Inputs:
- output/experiments/embeddings/double_stars_8d/umap_standard/embedding_umap.csv

Outputs:
- output/experiments/embeddings/double_stars_8d/umap_standard/umap_exploration/
- plots/qa/embeddings/double_stars_8d/umap_exploration/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


FEATURE_COLS_DEFAULT = [
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


def robust_limits(values: np.ndarray, p_lo: float = 2, p_hi: float = 98) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [p_lo, p_hi])
    if hi <= lo:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def load_embedding(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    required = ["source_id", "x", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in embedding CSV: {missing}")

    for c in ["source_id", "x", "y", "local_double_frac_k50", "phot_g_mean_mag", "double_cluster", *FEATURE_COLS_DEFAULT]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["source_id", "x", "y"]).copy()
    df["source_id"] = df["source_id"].astype(np.int64)
    return df.reset_index(drop=True)


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def balanced_sample_by_field(df: pd.DataFrame, per_field_n: int, seed: int) -> pd.DataFrame:
    if "field_tag" not in df.columns:
        return df.copy()
    if per_field_n <= 0:
        return df.copy()

    rng = np.random.default_rng(seed)
    out = []
    for field, sub in df.groupby("field_tag", sort=True):
        n_take = min(len(sub), int(per_field_n))
        if n_take <= 0:
            continue
        if len(sub) == n_take:
            out.append(sub)
        else:
            idx = rng.choice(len(sub), size=n_take, replace=False)
            out.append(sub.iloc[idx])
    if not out:
        return df.copy()
    return pd.concat(out, ignore_index=True).reset_index(drop=True)


def plot_density_hex(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(9.0, 7.0))
    hb = plt.hexbin(df["x"], df["y"], gridsize=180, mincnt=1, bins="log", cmap="magma")
    cbar = plt.colorbar(hb)
    cbar.set_label("log10(count)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP density map")
    savefig(out_png)


def plot_feature_panels(df: pd.DataFrame, features: Iterable[str], out_png: Path) -> None:
    feats = [f for f in features if f in df.columns]
    if not feats:
        return
    n = len(feats)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for i, f in enumerate(feats):
        ax = axes.flat[i]
        vals = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)
        lo, hi = robust_limits(vals)
        sc = ax.scatter(
            df["x"],
            df["y"],
            c=vals,
            s=5,
            cmap="viridis",
            vmin=lo,
            vmax=hi,
            linewidths=0,
            alpha=0.72,
        )
        ax.set_title(f)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n, nrows * ncols):
        axes.flat[j].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("UMAP-1")
    for ax in axes[:, 0]:
        ax.set_ylabel("UMAP-2")
    fig.suptitle("UMAP colored by feature values", fontsize=14)
    savefig(out_png)


def plot_top_fields(df: pd.DataFrame, out_png: Path, top_n: int = 12) -> None:
    if "field_tag" not in df.columns:
        return
    counts = df["field_tag"].astype(str).value_counts(dropna=False)
    top = counts.head(top_n).index.tolist()
    sub = df[df["field_tag"].astype(str).isin(top)].copy()
    if sub.empty:
        return

    plt.figure(figsize=(9.2, 7.2))
    for t in top:
        s = sub[sub["field_tag"].astype(str) == t]
        plt.scatter(s["x"], s["y"], s=6, alpha=0.6, linewidths=0, label=f"{t} (n={len(s)})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(f"UMAP by field tag (top {top_n})")
    plt.legend(frameon=True, fontsize=8, loc="best")
    savefig(out_png)


def run_dbscan_labels(df: pd.DataFrame, eps: float, min_samples: int) -> np.ndarray:
    xy = df[["x", "y"]].to_numpy(dtype=np.float32)
    return DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=-1).fit_predict(xy)


def summarize_clusters(df: pd.DataFrame, labels: np.ndarray, features: list[str]) -> pd.DataFrame:
    d = df.copy()
    d["cluster_dbscan"] = labels
    d = d[d["cluster_dbscan"] >= 0].copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    for c, sub in d.groupby("cluster_dbscan", sort=True):
        row = {"cluster_dbscan": int(c), "n": int(len(sub))}
        for f in features:
            if f in sub.columns:
                vals = pd.to_numeric(sub[f], errors="coerce").dropna().to_numpy(dtype=float)
                if vals.size > 0:
                    row[f"{f}_median"] = float(np.median(vals))
                    row[f"{f}_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                else:
                    row[f"{f}_median"] = np.nan
                    row[f"{f}_iqr"] = np.nan
        if "phot_g_mean_mag" in sub.columns:
            v = pd.to_numeric(sub["phot_g_mean_mag"], errors="coerce").dropna().to_numpy(dtype=float)
            row["phot_g_mean_mag_median"] = float(np.median(v)) if v.size else np.nan
        if "field_tag" in sub.columns:
            vc = sub["field_tag"].astype(str).value_counts()
            top_field = vc.index[0] if len(vc) else ""
            row["top_field_tag"] = str(top_field)
            row["top_field_frac"] = float(vc.iloc[0] / len(sub)) if len(vc) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def plot_dbscan_map(df: pd.DataFrame, labels: np.ndarray, out_png: Path) -> None:
    d = df.copy()
    d["cluster_dbscan"] = labels
    noise = d[d["cluster_dbscan"] < 0]
    core = d[d["cluster_dbscan"] >= 0]

    plt.figure(figsize=(9.0, 7.0))
    plt.scatter(noise["x"], noise["y"], s=4, c="0.85", alpha=0.4, linewidths=0, label=f"noise (n={len(noise)})")
    if not core.empty:
        sc = plt.scatter(
            core["x"],
            core["y"],
            s=8,
            c=core["cluster_dbscan"],
            cmap="tab20",
            alpha=0.85,
            linewidths=0,
            label=f"clusters (n={len(core)})",
        )
        cbar = plt.colorbar(sc)
        cbar.set_label("DBSCAN cluster id")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("DBSCAN structure on UMAP")
    plt.legend(frameon=True, fontsize=9, loc="best")
    savefig(out_png)


def plot_cluster_feature_heatmap(summary: pd.DataFrame, features: list[str], out_png: Path) -> None:
    if summary.empty:
        return
    med_cols = [f"{f}_median" for f in features if f"{f}_median" in summary.columns]
    if not med_cols:
        return
    mat = summary[med_cols].to_numpy(dtype=float)

    # z-score by feature column for contrast between clusters
    col_mean = np.nanmean(mat, axis=0, keepdims=True)
    col_std = np.nanstd(mat, axis=0, keepdims=True)
    col_std[col_std < 1e-12] = 1.0
    z = (mat - col_mean) / col_std

    plt.figure(figsize=(0.9 * len(med_cols) + 4, 0.45 * len(summary) + 3))
    im = plt.imshow(z, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    plt.colorbar(im, label="cluster median z-score")
    plt.yticks(np.arange(len(summary)), [f"c{int(c)} (n={int(n)})" for c, n in zip(summary["cluster_dbscan"], summary["n"])])
    plt.xticks(np.arange(len(med_cols)), [c.replace("_median", "") for c in med_cols], rotation=45, ha="right")
    plt.title("DBSCAN cluster fingerprints (feature medians)")
    savefig(out_png)


def compute_axis_trend_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        sub = df[["x", "y", f]].copy()
        sub[f] = pd.to_numeric(sub[f], errors="coerce")
        sub = sub.dropna()
        if len(sub) < 10:
            continue
        # pandas spearman: rank-correlation without scipy dependency.
        rho_x = float(sub["x"].corr(sub[f], method="spearman"))
        rho_y = float(sub["y"].corr(sub[f], method="spearman"))
        rows.append(
            {
                "feature": f,
                "spearman_vs_x": rho_x,
                "spearman_vs_y": rho_y,
                "axis_strength_l2": float(np.sqrt(rho_x * rho_x + rho_y * rho_y)),
                "n_used": int(len(sub)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("axis_strength_l2", ascending=False).reset_index(drop=True)
    return out


def compute_knn_smoothness(df: pd.DataFrame, features: list[str], k: int, sample_n: int, seed: int) -> pd.DataFrame:
    d = sample_df(df, n=sample_n, seed=seed).copy()
    xy = d[["x", "y"]].to_numpy(dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=int(k) + 1, algorithm="auto")
    nn.fit(xy)
    idx = nn.kneighbors(xy, return_distance=False)[:, 1:]

    rows = []
    for f in features:
        if f not in d.columns:
            continue
        vals = pd.to_numeric(d[f], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(vals)
        if ok.sum() < 100:
            continue
        global_std = float(np.nanstd(vals))
        if global_std < 1e-12:
            continue

        # Mean absolute neighbor difference, normalized by global std.
        # Lower => smoother manifold for that feature.
        diffs = []
        for i in range(len(vals)):
            if not np.isfinite(vals[i]):
                continue
            neigh = idx[i]
            neigh_vals = vals[neigh]
            neigh_vals = neigh_vals[np.isfinite(neigh_vals)]
            if neigh_vals.size == 0:
                continue
            diffs.append(float(np.mean(np.abs(neigh_vals - vals[i]))))
        if not diffs:
            continue
        mad_knn = float(np.mean(diffs))
        rows.append(
            {
                "feature": f,
                "mean_abs_neighbor_diff": mad_knn,
                "global_std": global_std,
                "smoothness_ratio": mad_knn / global_std,
                "n_used": int(ok.sum()),
                "k": int(k),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("smoothness_ratio", ascending=True).reset_index(drop=True)
    return out


def plot_knn_smoothness(smooth_df: pd.DataFrame, out_png: Path) -> None:
    if smooth_df.empty:
        return
    plt.figure(figsize=(8.8, 5.2))
    y = np.arange(len(smooth_df))
    plt.barh(y, smooth_df["smoothness_ratio"], color="#5e81ac", alpha=0.85)
    plt.yticks(y, smooth_df["feature"])
    plt.gca().invert_yaxis()
    plt.xlabel("kNN smoothness ratio (lower = smoother)")
    plt.title("Feature continuity on UMAP manifold")
    savefig(out_png)


def scan_dbscan_eps(df: pd.DataFrame, eps_values: list[float], min_samples: int) -> pd.DataFrame:
    rows = []
    labels_ref = None
    eps_ref = None
    for eps in eps_values:
        labels = run_dbscan_labels(df, eps=eps, min_samples=min_samples)
        n_noise = int((labels < 0).sum())
        n_clustered = int((labels >= 0).sum())
        n_clusters = int(len(set(labels[labels >= 0])))
        frac_noise = float(n_noise / len(labels))
        ari_to_prev = np.nan
        if labels_ref is not None:
            ari_to_prev = float(adjusted_rand_score(labels_ref, labels))
        rows.append(
            {
                "eps": float(eps),
                "min_samples": int(min_samples),
                "n_clusters": n_clusters,
                "n_clustered": n_clustered,
                "n_noise": n_noise,
                "frac_noise": frac_noise,
                "ari_vs_prev_eps": ari_to_prev,
                "prev_eps": eps_ref if eps_ref is not None else np.nan,
            }
        )
        labels_ref = labels
        eps_ref = float(eps)
    return pd.DataFrame(rows)


def plot_dbscan_scan(scan_df: pd.DataFrame, out_png: Path) -> None:
    if scan_df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(8.8, 5.2))
    ax1.plot(scan_df["eps"], scan_df["n_clusters"], marker="o", color="#d62828", label="n_clusters")
    ax1.set_xlabel("DBSCAN eps")
    ax1.set_ylabel("n_clusters", color="#d62828")
    ax1.tick_params(axis="y", labelcolor="#d62828")

    ax2 = ax1.twinx()
    ax2.plot(scan_df["eps"], scan_df["frac_noise"], marker="s", color="#1d3557", label="frac_noise")
    ax2.set_ylabel("frac_noise", color="#1d3557")
    ax2.tick_params(axis="y", labelcolor="#1d3557")
    plt.title("DBSCAN sensitivity across eps")
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory analysis pack for UMAP embeddings.")
    ap.add_argument(
        "--embedding_csv",
        default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/embedding_umap.csv",
    )
    ap.add_argument(
        "--out_tab_dir",
        default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/umap_exploration",
    )
    ap.add_argument(
        "--out_plot_dir",
        default="/data/yn316/Codes/plots/qa/embeddings/double_stars_8d/umap_exploration",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot_sample_n", type=int, default=80000)
    ap.add_argument("--balance_by_field", action="store_true", help="Use field-balanced sampling for analysis.")
    ap.add_argument("--balance_per_field", type=int, default=3500, help="Max rows per field when --balance_by_field is used.")
    ap.add_argument("--dbscan_eps", type=float, default=0.28)
    ap.add_argument("--dbscan_min_samples", type=int, default=80)
    ap.add_argument("--knn_k", type=int, default=40)
    ap.add_argument("--knn_sample_n", type=int, default=30000)
    args = ap.parse_args()

    out_tab = ensure_dir(Path(args.out_tab_dir))
    out_plot = ensure_dir(Path(args.out_plot_dir))

    df_raw = load_embedding(Path(args.embedding_csv))
    if bool(args.balance_by_field):
        df = balanced_sample_by_field(df_raw, per_field_n=int(args.balance_per_field), seed=int(args.seed))
    else:
        df = df_raw.copy()

    dplot = sample_df(df, n=int(args.plot_sample_n), seed=int(args.seed))

    print(f"Loaded rows raw: {len(df_raw):,} | working rows: {len(df):,} | plot sample: {len(dplot):,}")
    print(f"Field-balanced mode: {bool(args.balance_by_field)} | balance_per_field={int(args.balance_per_field)}")

    features = [f for f in FEATURE_COLS_DEFAULT if f in df.columns]
    extra = [c for c in ["phot_g_mean_mag", "local_double_frac_k50"] if c in df.columns]
    feature_panels = features + extra

    plot_density_hex(dplot, out_plot / "01_umap_density_hexbin.png")
    plot_feature_panels(dplot, feature_panels, out_plot / "02_umap_feature_color_panels.png")
    plot_top_fields(dplot, out_plot / "03_umap_top_fields_overlay.png", top_n=12)

    labels = run_dbscan_labels(dplot, eps=float(args.dbscan_eps), min_samples=int(args.dbscan_min_samples))
    plot_dbscan_map(dplot, labels, out_plot / "04_umap_dbscan_clusters.png")

    cluster_summary = summarize_clusters(dplot, labels, features=features)
    cluster_summary.to_csv(out_tab / "cluster_fingerprints_dbscan.csv", index=False)
    plot_cluster_feature_heatmap(cluster_summary, features=features, out_png=out_plot / "05_cluster_feature_heatmap.png")

    trend_stats = compute_axis_trend_stats(df, features=feature_panels)
    trend_stats.to_csv(out_tab / "feature_axis_trend_stats.csv", index=False)

    smooth_df = compute_knn_smoothness(df, features=feature_panels, k=int(args.knn_k), sample_n=int(args.knn_sample_n), seed=int(args.seed))
    smooth_df.to_csv(out_tab / "feature_knn_smoothness.csv", index=False)
    plot_knn_smoothness(smooth_df, out_plot / "06_feature_knn_smoothness.png")

    scan_eps = [0.20, 0.24, 0.28, 0.32, 0.36, 0.40]
    scan_df = scan_dbscan_eps(dplot, eps_values=scan_eps, min_samples=int(args.dbscan_min_samples))
    scan_df.to_csv(out_tab / "dbscan_eps_sensitivity.csv", index=False)
    plot_dbscan_scan(scan_df, out_plot / "07_dbscan_eps_sensitivity.png")

    settings = pd.DataFrame(
        [
            {
                "embedding_csv": str(args.embedding_csv),
                "balance_by_field": bool(args.balance_by_field),
                "balance_per_field": int(args.balance_per_field),
                "rows_raw": int(len(df_raw)),
                "rows_working": int(len(df)),
                "rows_plot_sample": int(len(dplot)),
                "seed": int(args.seed),
                "dbscan_eps": float(args.dbscan_eps),
                "dbscan_min_samples": int(args.dbscan_min_samples),
                "knn_k": int(args.knn_k),
                "knn_sample_n": int(args.knn_sample_n),
            }
        ]
    )
    settings.to_csv(out_tab / "run_settings.csv", index=False)

    print("Done.")
    print("  Tables:", out_tab)
    print("  Plots :", out_plot)


if __name__ == "__main__":
    main()
