#!/usr/bin/env python3
"""
Rank near-cluster non-double sources for follow-up using:
1) proximity to double-cluster centroid
2) local enrichment around doubles
3) morphology-based double-likeness from stamp images

Inputs (default):
- output/experiments/embeddings/double_stars_8d/umap_standard/
  - double_candidates_ranked_by_local_enrichment_umap.csv
  - non_double_neighbors_near_clusters_umap.csv
  - embedding_umap.csv
- output/dataset_npz/*/metadata.csv and NPZ stamps

Outputs:
- output/experiments/embeddings/double_stars_8d/umap_standard/candidate_ranking/
- plots/qa/embeddings/double_stars_8d/candidate_ranking/
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MORPH_COLS = [
    "m_flux_sum",
    "m_flux_peak",
    "m_peak_ratio_2over1",
    "m_peak_sep_pix",
    "m_ellipticity",
    "m_size_a_pix",
    "m_size_b_pix",
    "m_concentration_r2_r6",
    "m_edge_flux_frac",
    "m_asymmetry_180",
]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def infer_field_tag_from_fits(path_str: str) -> str:
    try:
        p = Path(path_str)
        return p.parent.name if p.parent.name else "unknown"
    except Exception:
        return "unknown"


def build_source_lookup(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(dataset_root.glob("*/metadata.csv")):
        field_tag = meta_path.parent.name
        usecols = ["source_id", "npz_file", "index_in_file", "fits_path", "phot_g_mean_mag"]
        df = pd.read_csv(meta_path, usecols=lambda c: c in set(usecols), low_memory=False)

        for c in ["source_id", "index_in_file", "phot_g_mean_mag"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["source_id", "index_in_file", "npz_file"]).copy()
        df["source_id"] = df["source_id"].astype(np.int64)
        df["index_in_file"] = df["index_in_file"].astype(np.int64)

        if "fits_path" in df.columns:
            ft = df["fits_path"].astype(str).map(infer_field_tag_from_fits)
            df["field_tag"] = np.where(ft == "unknown", field_tag, ft)
        else:
            df["field_tag"] = field_tag

        df["npz_path"] = df.apply(lambda r: str(dataset_root / str(r["field_tag"]) / str(r["npz_file"])), axis=1)
        rows.append(df[["source_id", "field_tag", "npz_path", "index_in_file", "phot_g_mean_mag"]])

    if not rows:
        raise RuntimeError(f"No metadata.csv found under {dataset_root}")

    all_meta = pd.concat(rows, ignore_index=True)
    all_meta = all_meta.sort_values(["source_id"]).drop_duplicates(subset=["source_id"], keep="first")
    return all_meta.reset_index(drop=True)


def fetch_stamps(records: pd.DataFrame, lookup: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rec = records.copy()
    rec["source_id"] = pd.to_numeric(rec["source_id"], errors="coerce")
    rec = rec.dropna(subset=["source_id"]).copy()
    rec["source_id"] = rec["source_id"].astype(np.int64)

    merged = rec.merge(lookup, on="source_id", how="left", suffixes=("", "_meta"))

    cache: Dict[str, np.ndarray] = {}
    stamps: List[np.ndarray] = []
    miss = []

    for _, r in merged.iterrows():
        npz_path = r.get("npz_path")
        idx = r.get("index_in_file")

        if pd.isna(npz_path) or pd.isna(idx):
            miss.append({"source_id": int(r["source_id"]), "reason": "missing_lookup"})
            stamps.append(None)
            continue

        npz_path = str(npz_path)
        idx = int(idx)
        p = Path(npz_path)

        if not p.exists():
            miss.append({"source_id": int(r["source_id"]), "reason": f"npz_not_found:{npz_path}"})
            stamps.append(None)
            continue

        try:
            if npz_path not in cache:
                arr = np.load(npz_path)
                if "X" not in arr.files:
                    miss.append({"source_id": int(r["source_id"]), "reason": f"missing_X_key:{npz_path}"})
                    stamps.append(None)
                    continue
                cache[npz_path] = arr["X"]

            x = cache[npz_path]
            if idx < 0 or idx >= x.shape[0]:
                miss.append({"source_id": int(r["source_id"]), "reason": f"index_out_of_range:{idx}"})
                stamps.append(None)
                continue

            stamps.append(np.array(x[idx], dtype=np.float32))
        except Exception as e:
            miss.append({"source_id": int(r["source_id"]), "reason": f"load_error:{e}"})
            stamps.append(None)

    merged["stamp"] = stamps
    miss_df = pd.DataFrame(miss, columns=["source_id", "reason"])
    return merged, miss_df


def _safe_percentile(a: np.ndarray, q: float) -> float:
    vals = a[np.isfinite(a)]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, q))


def extract_morphology(stamp: np.ndarray) -> Dict[str, float]:
    img = np.array(stamp, dtype=np.float64)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # Background subtraction then positive-part features.
    bg = _safe_percentile(img, 10.0)
    ip = img - bg
    ip[ip < 0] = 0.0

    flux_sum = float(np.sum(ip))
    peak = float(np.max(ip)) if ip.size else 0.0

    h, w = ip.shape
    yy, xx = np.indices((h, w))

    if flux_sum <= 0:
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        qxx = qyy = qxy = 0.0
        a = b = 0.0
        ell = 0.0
    else:
        cx = float(np.sum(ip * xx) / flux_sum)
        cy = float(np.sum(ip * yy) / flux_sum)

        dx = xx - cx
        dy = yy - cy
        qxx = float(np.sum(ip * dx * dx) / flux_sum)
        qyy = float(np.sum(ip * dy * dy) / flux_sum)
        qxy = float(np.sum(ip * dx * dy) / flux_sum)

        tr = qxx + qyy
        det = qxx * qyy - qxy * qxy
        disc = max(tr * tr - 4.0 * det, 0.0)
        lam1 = max((tr + math.sqrt(disc)) * 0.5, 0.0)
        lam2 = max((tr - math.sqrt(disc)) * 0.5, 0.0)
        a = math.sqrt(lam1)
        b = math.sqrt(lam2)
        ell = (a - b) / (a + b + 1e-9)

    flat = ip.ravel()
    if flat.size >= 2:
        idx2 = np.argpartition(flat, -2)[-2:]
        vals2 = flat[idx2]
        ord2 = np.argsort(vals2)[::-1]
        i1 = idx2[ord2[0]]
        i2 = idx2[ord2[1]]
        y1, x1 = divmod(int(i1), w)
        y2, x2 = divmod(int(i2), w)
        p1 = float(flat[i1])
        p2 = float(flat[i2])
        peak_ratio = p2 / (p1 + 1e-9)
        peak_sep = float(math.hypot(x2 - x1, y2 - y1))
    else:
        peak_ratio = 0.0
        peak_sep = 0.0

    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    f_r2 = float(np.sum(ip[rr <= 2.0]))
    f_r6 = float(np.sum(ip[rr <= 6.0]))
    concentration = f_r2 / (f_r6 + 1e-9)

    border = np.zeros_like(ip, dtype=bool)
    border[:2, :] = True
    border[-2:, :] = True
    border[:, :2] = True
    border[:, -2:] = True
    edge_flux_frac = float(np.sum(ip[border]) / (flux_sum + 1e-9))

    # 180-degree asymmetry around center pixel approximation.
    rot180 = np.flipud(np.fliplr(ip))
    asym = float(np.sum(np.abs(ip - rot180)) / (np.sum(np.abs(ip)) + 1e-9))

    return {
        "m_flux_sum": flux_sum,
        "m_flux_peak": peak,
        "m_peak_ratio_2over1": float(peak_ratio),
        "m_peak_sep_pix": float(peak_sep),
        "m_ellipticity": float(ell),
        "m_size_a_pix": float(a),
        "m_size_b_pix": float(b),
        "m_concentration_r2_r6": float(concentration),
        "m_edge_flux_frac": float(edge_flux_frac),
        "m_asymmetry_180": float(asym),
    }


def add_morph_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        stamp = r.get("stamp", None)
        if isinstance(stamp, np.ndarray):
            feats = extract_morphology(stamp)
        else:
            feats = {k: np.nan for k in MORPH_COLS}
        rows.append(feats)
    fdf = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), fdf], axis=1)
    return out


def fit_morph_classifier(pos: pd.DataFrame, neg: pd.DataFrame, seed: int = 42) -> Tuple[Pipeline, Dict[str, float]]:
    p = pos.dropna(subset=MORPH_COLS).copy()
    n = neg.dropna(subset=MORPH_COLS).copy()
    p["y"] = 1
    n["y"] = 0
    all_df = pd.concat([p, n], ignore_index=True)

    X = all_df[MORPH_COLS].to_numpy(dtype=np.float64)
    y = all_df["y"].to_numpy(dtype=int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
            ),
        ]
    )

    auc = np.nan
    if len(np.unique(y)) == 2 and np.sum(y == 1) >= 10 and np.sum(y == 0) >= 30:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        aucs = []
        for tr, te in skf.split(X, y):
            model.fit(X[tr], y[tr])
            pte = model.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], pte))
        auc = float(np.mean(aucs))

    model.fit(X, y)
    return model, {"cv_auc": auc, "n_pos": int(np.sum(y == 1)), "n_neg": int(np.sum(y == 0))}


def robust_minmax(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    lo = np.nanpercentile(v, 5) if np.isfinite(v).any() else 0.0
    hi = np.nanpercentile(v, 95) if np.isfinite(v).any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(v)), index=v.index)
    z = (v - lo) / (hi - lo)
    return z.clip(0, 1)


def plot_feature_distributions(groups: Dict[str, pd.DataFrame], out_png: Path) -> None:
    feats = ["m_peak_sep_pix", "m_ellipticity", "m_peak_ratio_2over1", "m_concentration_r2_r6", "m_asymmetry_180"]
    fig, axes = plt.subplots(1, len(feats), figsize=(3.7 * len(feats), 4.2))

    for ax, f in zip(axes, feats):
        for name, color in [("doubles", "#d62828"), ("near_non_double", "#1d3557"), ("background", "#6c757d")]:
            df = groups[name]
            v = pd.to_numeric(df[f], errors="coerce").dropna().to_numpy(dtype=float)
            if len(v) == 0:
                continue
            v = np.clip(v, np.percentile(v, 1), np.percentile(v, 99))
            ax.hist(v, bins=35, density=True, histtype="step", linewidth=1.8, alpha=0.95, color=color, label=name)
        ax.set_title(f, fontsize=9)
        ax.tick_params(labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=8, frameon=True)
    fig.suptitle("Morphology feature distributions", fontsize=13)
    savefig(out_png)


def plot_score_diagnostics(near: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    axes[0].scatter(near["dist_to_cluster_centroid"], near["double_like_prob"], s=24, c=near["candidate_score"], cmap="viridis", alpha=0.9)
    axes[0].set_xlabel("Distance to cluster centroid")
    axes[0].set_ylabel("Double-like probability")
    axes[0].set_title("Morph prob vs cluster distance")

    axes[1].scatter(near["local_double_frac_k50"], near["double_like_prob"], s=24, c=near["candidate_score"], cmap="viridis", alpha=0.9)
    axes[1].set_xlabel("Local double fraction (k50)")
    axes[1].set_ylabel("Double-like probability")
    axes[1].set_title("Morph prob vs local enrichment")

    sc = axes[2].scatter(near["x"], near["y"], s=22, c=near["candidate_score"], cmap="plasma", alpha=0.9)
    axes[2].set_xlabel("Embedding X")
    axes[2].set_ylabel("Embedding Y")
    axes[2].set_title("Candidate score over UMAP")
    cbar = plt.colorbar(sc, ax=axes[2])
    cbar.set_label("candidate_score")

    fig.suptitle("Near-cluster non-double candidate diagnostics", fontsize=13)
    savefig(out_png)


def robust_limits(img: np.ndarray) -> Tuple[float, float]:
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if lo == hi:
            hi = lo + 1e-6
    return float(lo), float(hi)


def plot_top_candidate_gallery(near_ranked: pd.DataFrame, out_png: Path, n_show: int = 36, ncols: int = 6) -> None:
    vis = near_ranked[near_ranked["stamp"].notna()].head(n_show).copy().reset_index(drop=True)
    n = len(vis)
    if n == 0:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No resolved near-candidate stamps.", ha="center", va="center")
        plt.axis("off")
        savefig(out_png)
        return

    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, r) in zip(axes, vis.iterrows()):
        img = r["stamp"]
        lo, hi = robust_limits(img)
        ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(
            f"{int(r['source_id'])}\nscore={float(r['candidate_score']):.3f} p={float(r['double_like_prob']):.2f}",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Top near-cluster non-double candidates", fontsize=13)
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank and inspect near-cluster non-double candidates")
    ap.add_argument("--dataset_root", default="/data/yn316/Codes/output/dataset_npz")
    ap.add_argument("--embedding_dir", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard")
    ap.add_argument("--out_tab_dir", default=None)
    ap.add_argument("--out_plot_dir", default="/data/yn316/Codes/plots/qa/embeddings/double_stars_8d/candidate_ranking")
    ap.add_argument("--n_neg_train", type=int, default=1500)
    ap.add_argument("--n_top_gallery", type=int, default=36)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    embedding_dir = Path(args.embedding_dir)
    out_tab_dir = Path(args.out_tab_dir) if args.out_tab_dir else (embedding_dir / "candidate_ranking")
    out_plot_dir = Path(args.out_plot_dir)
    ensure_dir(out_tab_dir)
    ensure_dir(out_plot_dir)

    doubles_csv = embedding_dir / "double_candidates_ranked_by_local_enrichment_umap.csv"
    near_csv = embedding_dir / "non_double_neighbors_near_clusters_umap.csv"
    emb_csv = embedding_dir / "embedding_umap.csv"

    if not doubles_csv.exists() or not near_csv.exists() or not emb_csv.exists():
        raise FileNotFoundError("Missing expected embedding outputs in embedding_dir")

    print("Building lookup from dataset metadata...")
    lookup = build_source_lookup(dataset_root)

    print("Loading embedding tables...")
    doubles = pd.read_csv(doubles_csv)
    near = pd.read_csv(near_csv)
    emb = pd.read_csv(emb_csv, usecols=["source_id", "is_double", "x", "y", "local_double_frac_k50"])
    emb["source_id"] = pd.to_numeric(emb["source_id"], errors="coerce").astype("Int64")
    emb = emb.dropna(subset=["source_id"]).copy()
    emb["source_id"] = emb["source_id"].astype(np.int64)

    rng = np.random.default_rng(args.seed)

    print("Selecting background non-doubles for classifier training...")
    exclude = set(pd.to_numeric(near["source_id"], errors="coerce").dropna().astype(np.int64).tolist())
    exclude |= set(pd.to_numeric(doubles["source_id"], errors="coerce").dropna().astype(np.int64).tolist())

    bg_pool = emb[(~emb["is_double"]) & (~emb["source_id"].isin(exclude))].copy()
    if len(bg_pool) == 0:
        raise RuntimeError("No background non-double pool available")

    n_bg = min(int(args.n_neg_train), len(bg_pool))
    bg_idx = rng.choice(len(bg_pool), size=n_bg, replace=False)
    bg = bg_pool.iloc[bg_idx].copy().reset_index(drop=True)

    print("Resolving stamps for groups...")
    doubles_r, miss_d = fetch_stamps(doubles, lookup)
    near_r, miss_n = fetch_stamps(near, lookup)
    bg_r, miss_b = fetch_stamps(bg, lookup)

    print("Extracting morphology features...")
    doubles_f = add_morph_features(doubles_r)
    near_f = add_morph_features(near_r)
    bg_f = add_morph_features(bg_r)

    print("Training morphology-based double-likeness model...")
    model, info = fit_morph_classifier(doubles_f, bg_f, seed=args.seed)

    for df in [doubles_f, near_f, bg_f]:
        X = df[MORPH_COLS].to_numpy(dtype=np.float64)
        ok = np.all(np.isfinite(X), axis=1)
        p = np.full(len(df), np.nan)
        if np.any(ok):
            p[ok] = model.predict_proba(X[ok])[:, 1]
        df["double_like_prob"] = p

    print("Scoring near-cluster non-doubles...")
    near_sc = near_f.copy()
    near_sc["dist_score"] = np.exp(-pd.to_numeric(near_sc["dist_to_cluster_centroid"], errors="coerce") / (np.nanmedian(near_sc["dist_to_cluster_centroid"]) + 1e-9))
    near_sc["enrich_score"] = robust_minmax(near_sc["local_double_frac_k50"])
    near_sc["morph_score"] = robust_minmax(near_sc["double_like_prob"])

    near_sc["candidate_score"] = (
        0.50 * near_sc["morph_score"]
        + 0.30 * near_sc["dist_score"].clip(0, 1)
        + 0.20 * near_sc["enrich_score"]
    )

    near_ranked = near_sc.sort_values("candidate_score", ascending=False).reset_index(drop=True)

    # Group summary table for quick comparisons.
    def summarize_group(name: str, df: pd.DataFrame) -> Dict[str, float]:
        out = {"group": name, "n": int(len(df))}
        for c in MORPH_COLS + ["double_like_prob", "local_double_frac_k50"]:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                out[f"{c}_median"] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
        return out

    grp_summary = pd.DataFrame(
        [
            summarize_group("doubles", doubles_f),
            summarize_group("near_non_double", near_f),
            summarize_group("background", bg_f),
        ]
    )

    print("Writing outputs...")
    near_ranked.to_csv(out_tab_dir / "near_cluster_non_double_ranked_candidates.csv", index=False)
    doubles_f.to_csv(out_tab_dir / "doubles_with_morph_features.csv", index=False)
    near_f.to_csv(out_tab_dir / "near_non_doubles_with_morph_features.csv", index=False)
    bg_f.to_csv(out_tab_dir / "background_non_doubles_with_morph_features.csv", index=False)
    grp_summary.to_csv(out_tab_dir / "morph_group_summary.csv", index=False)

    miss_all = pd.concat([miss_d, miss_n, miss_b], ignore_index=True)
    miss_all.to_csv(out_tab_dir / "missing_stamp_sources.csv", index=False)

    model_info = pd.DataFrame([info])
    model_info.to_csv(out_tab_dir / "morph_classifier_info.csv", index=False)

    plot_feature_distributions(
        groups={
            "doubles": doubles_f,
            "near_non_double": near_f,
            "background": bg_f,
        },
        out_png=out_plot_dir / "01_morph_feature_distributions.png",
    )
    plot_score_diagnostics(near_ranked, out_plot_dir / "02_near_candidate_score_diagnostics.png")
    plot_top_candidate_gallery(near_ranked, out_plot_dir / "03_top_near_candidates_stamp_gallery.png", n_show=int(args.n_top_gallery))

    print("Done.")
    print("  Ranked candidates:", out_tab_dir / "near_cluster_non_double_ranked_candidates.csv")
    print("  Plots:", out_plot_dir)


if __name__ == "__main__":
    main()
