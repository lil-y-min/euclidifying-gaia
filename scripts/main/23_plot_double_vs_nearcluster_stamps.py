#!/usr/bin/env python3
"""
Build visual stamp galleries for:
1) double-star candidates
2) near-cluster non-doubles
3) paired comparison within each detected double cluster

Default inputs are the UMAP standard-scaling outputs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


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

        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
        df["index_in_file"] = pd.to_numeric(df["index_in_file"], errors="coerce")
        df["phot_g_mean_mag"] = pd.to_numeric(df.get("phot_g_mean_mag", np.nan), errors="coerce")

        df = df.dropna(subset=["source_id", "index_in_file", "npz_file"]).copy()
        df["source_id"] = df["source_id"].astype(np.int64)
        df["index_in_file"] = df["index_in_file"].astype(np.int64)

        if "fits_path" in df.columns:
            ft = df["fits_path"].astype(str).map(infer_field_tag_from_fits)
            df["field_tag"] = np.where(ft == "unknown", field_tag, ft)
        else:
            df["field_tag"] = field_tag

        df["npz_path"] = df.apply(lambda r: str(dataset_root / str(r["field_tag"]) / str(r["npz_file"])), axis=1)

        rows.append(df[["source_id", "field_tag", "npz_file", "npz_path", "index_in_file", "phot_g_mean_mag"]])

    if not rows:
        raise RuntimeError(f"No metadata.csv found under {dataset_root}")

    all_meta = pd.concat(rows, ignore_index=True)
    all_meta = all_meta.sort_values(["source_id"]).drop_duplicates(subset=["source_id"], keep="first")
    return all_meta.reset_index(drop=True)


def fetch_stamps(records: pd.DataFrame, lookup: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    need = records.copy()
    need["source_id"] = pd.to_numeric(need["source_id"], errors="coerce")
    need = need.dropna(subset=["source_id"]).copy()
    need["source_id"] = need["source_id"].astype(np.int64)

    merged = need.merge(lookup, on="source_id", how="left", suffixes=("", "_meta"))

    cache: Dict[str, np.ndarray] = {}
    stamps: List[np.ndarray] = []
    missing: List[dict] = []

    for _, r in merged.iterrows():
        npz_path = r.get("npz_path")
        idx = r.get("index_in_file")

        if pd.isna(npz_path) or pd.isna(idx):
            missing.append({"source_id": int(r["source_id"]), "reason": "missing_lookup"})
            stamps.append(None)
            continue

        npz_path = str(npz_path)
        idx = int(idx)
        p = Path(npz_path)
        if not p.exists():
            missing.append({"source_id": int(r["source_id"]), "reason": f"npz_not_found:{npz_path}"})
            stamps.append(None)
            continue

        try:
            if npz_path not in cache:
                arr = np.load(npz_path)
                if "X" not in arr.files:
                    missing.append({"source_id": int(r["source_id"]), "reason": f"missing_X_key:{npz_path}"})
                    stamps.append(None)
                    continue
                cache[npz_path] = arr["X"]

            x = cache[npz_path]
            if idx < 0 or idx >= x.shape[0]:
                missing.append({"source_id": int(r["source_id"]), "reason": f"index_out_of_range:{idx}"})
                stamps.append(None)
                continue

            stamps.append(np.array(x[idx], dtype=np.float32))
        except Exception as e:
            missing.append({"source_id": int(r["source_id"]), "reason": f"load_error:{e}"})
            stamps.append(None)

    merged["stamp"] = stamps
    return merged, missing


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


def plot_stamp_grid(df: pd.DataFrame, out_png: Path, title: str, ncols: int = 6) -> None:
    vis = df[df["stamp"].notna()].copy().reset_index(drop=True)
    n = len(vis)

    if n == 0:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No resolved stamps.", ha="center", va="center")
        plt.axis("off")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        return

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, r) in zip(axes, vis.iterrows()):
        img = r["stamp"]
        lo, hi = robust_limits(img)
        ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi)

        sid = int(r["source_id"])
        cluster = r.get("double_cluster", np.nan)
        cluster_txt = f"cl={int(cluster)}" if pd.notna(cluster) else ""
        if "wds_dist_arcsec" in vis.columns and pd.notna(r.get("wds_dist_arcsec", np.nan)):
            extra = f"wds={float(r['wds_dist_arcsec']):.3f}\""
        elif "dist_to_cluster_centroid" in vis.columns and pd.notna(r.get("dist_to_cluster_centroid", np.nan)):
            extra = f"d={float(r['dist_to_cluster_centroid']):.3f}"
        else:
            extra = ""

        ax.set_title(f"{sid}\n{cluster_txt} {extra}".strip(), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_cluster_pairs(doubles: pd.DataFrame, nond: pd.DataFrame, out_png: Path, pairs_per_cluster: int = 8) -> None:
    d = doubles[doubles["stamp"].notna()].copy()
    n = nond[nond["stamp"].notna()].copy()

    if d.empty or n.empty:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "Insufficient stamps for pair view.", ha="center", va="center")
        plt.axis("off")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        return

    rows = []
    for cl in sorted(set(d["double_cluster"].dropna().astype(int).unique()) & set(n["double_cluster"].dropna().astype(int).unique())):
        dcl = d[d["double_cluster"] == cl].sort_values("local_double_frac_k50", ascending=False).head(pairs_per_cluster)
        ncl = n[n["double_cluster"] == cl].sort_values("rank_nearest_to_cluster", ascending=True).head(pairs_per_cluster)
        m = min(len(dcl), len(ncl))
        for i in range(m):
            rows.append((int(cl), dcl.iloc[i], ncl.iloc[i]))

    if not rows:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No cluster overlaps found for pair view.", ha="center", va="center")
        plt.axis("off")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        return

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 2, figsize=(7.0, max(2.5 * nrows, 4)))
    axes = np.array(axes).reshape(nrows, 2)

    for r, (cl, dr, nr) in enumerate(rows):
        for c, rr, lbl in [(0, dr, "double"), (1, nr, "near non-double")]:
            ax = axes[r, c]
            img = rr["stamp"]
            lo, hi = robust_limits(img)
            ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi)
            if c == 0:
                ttl = f"cl {cl} | {lbl}\n{int(rr['source_id'])} wds={float(rr['wds_dist_arcsec']):.3f}\""
            else:
                ttl = f"cl {cl} | {lbl}\n{int(rr['source_id'])} rank={int(rr['rank_nearest_to_cluster'])} d={float(rr['dist_to_cluster_centroid']):.3f}"
            ax.set_title(ttl, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Cluster-wise stamp pairs: doubles vs near-cluster non-doubles", fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def add_gaia_single_multiple_flag(
    near_df: pd.DataFrame,
    metric_col: str = "feat_ipd_frac_multi_peak",
    threshold: float = 0.0,
) -> pd.DataFrame:
    out = near_df.copy()
    vals = pd.to_numeric(out.get(metric_col, np.nan), errors="coerce")
    out["gaia_multiple_metric"] = vals
    out["gaia_is_multiple"] = vals > float(threshold)
    out["gaia_detection_label"] = np.where(out["gaia_is_multiple"], "multiple", "single")
    return out


def plot_single_vs_multiple_pairs(
    near_df: pd.DataFrame,
    out_png: Path,
    pairs_per_cluster: int = 8,
) -> None:
    n = near_df[near_df["stamp"].notna()].copy()
    if n.empty:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No resolved near-cluster stamps.", ha="center", va="center")
        plt.axis("off")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        return

    rows = []
    clusters = sorted(n["double_cluster"].dropna().astype(int).unique().tolist())
    for cl in clusters:
        sub = n[n["double_cluster"] == cl]
        m = sub[sub["gaia_is_multiple"]].sort_values("rank_nearest_to_cluster", ascending=True).head(pairs_per_cluster)
        s = sub[~sub["gaia_is_multiple"]].sort_values("rank_nearest_to_cluster", ascending=True).head(pairs_per_cluster)
        k = min(len(m), len(s))
        for i in range(k):
            rows.append((cl, s.iloc[i], m.iloc[i]))

    if not rows:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No cluster has both Gaia-single and Gaia-multiple near-cluster sources.", ha="center", va="center")
        plt.axis("off")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        return

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 2, figsize=(7.0, max(2.5 * nrows, 4)))
    axes = np.array(axes).reshape(nrows, 2)

    for r, (cl, sr, mr) in enumerate(rows):
        for c, rr, lbl in [(0, sr, "single"), (1, mr, "multiple")]:
            ax = axes[r, c]
            img = rr["stamp"]
            lo, hi = robust_limits(img)
            ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi)
            ttl = (
                f"cl {cl} | {lbl}\n"
                f"{int(rr['source_id'])} rank={int(rr['rank_nearest_to_cluster'])} "
                f"d={float(rr['dist_to_cluster_centroid']):.3f} "
                f"ipd={float(rr['gaia_multiple_metric']):.1f}"
            )
            ax.set_title(ttl, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Near-cluster non-doubles: Gaia single vs multiple (cluster-wise pairs)", fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot stamp galleries for doubles and near-cluster non-doubles")
    ap.add_argument("--dataset_root", default="/data/yn316/Codes/output/dataset_npz")
    ap.add_argument("--embedding_dir", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard")
    ap.add_argument("--out_plot_dir", default="/data/yn316/Codes/plots/qa/embeddings/double_stars_8d/stamp_galleries")
    ap.add_argument("--n_doubles", type=int, default=36)
    ap.add_argument("--n_near", type=int, default=36)
    ap.add_argument("--pairs_per_cluster", type=int, default=8)
    ap.add_argument("--gaia_multiple_col", default="feat_ipd_frac_multi_peak", help="Column used to split near-cluster sources into single/multiple")
    ap.add_argument("--gaia_multiple_threshold", type=float, default=0.0, help="Value above which source is treated as Gaia-multiple")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    embedding_dir = Path(args.embedding_dir)
    out_plot_dir = ensure_dir(Path(args.out_plot_dir))

    doubles_csv = embedding_dir / "double_candidates_ranked_by_local_enrichment_umap.csv"
    near_csv = embedding_dir / "non_double_neighbors_near_clusters_umap.csv"

    if not doubles_csv.exists() or not near_csv.exists():
        raise FileNotFoundError(
            f"Missing expected input CSVs in {embedding_dir}: {doubles_csv.name}, {near_csv.name}"
        )

    print("Building source lookup...")
    lookup = build_source_lookup(dataset_root)

    print("Loading candidate tables...")
    doubles = pd.read_csv(doubles_csv).head(int(args.n_doubles)).copy()
    near = pd.read_csv(near_csv).head(int(args.n_near)).copy()

    print("Resolving stamps for doubles...")
    doubles_resolved, miss_d = fetch_stamps(doubles, lookup)

    print("Resolving stamps for near-cluster non-doubles...")
    near_resolved, miss_n = fetch_stamps(near, lookup)
    near_resolved = add_gaia_single_multiple_flag(
        near_resolved,
        metric_col=str(args.gaia_multiple_col),
        threshold=float(args.gaia_multiple_threshold),
    )

    print("Plotting galleries...")
    plot_stamp_grid(
        doubles_resolved,
        out_plot_dir / "01_doubles_top_stamps.png",
        title="Top double-star candidates (by local enrichment)",
        ncols=6,
    )
    plot_stamp_grid(
        near_resolved,
        out_plot_dir / "02_near_cluster_non_doubles_top_stamps.png",
        title="Nearest non-doubles around double-star cluster centroids",
        ncols=6,
    )
    plot_cluster_pairs(
        doubles_resolved,
        near_resolved,
        out_plot_dir / "03_cluster_pairs_doubles_vs_non_doubles.png",
        pairs_per_cluster=int(args.pairs_per_cluster),
    )
    plot_stamp_grid(
        near_resolved[~near_resolved["gaia_is_multiple"]].copy(),
        out_plot_dir / "04_near_cluster_gaia_single_stamps.png",
        title="Near-cluster non-doubles classified as Gaia single",
        ncols=6,
    )
    plot_stamp_grid(
        near_resolved[near_resolved["gaia_is_multiple"]].copy(),
        out_plot_dir / "05_near_cluster_gaia_multiple_stamps.png",
        title="Near-cluster non-doubles classified as Gaia multiple",
        ncols=6,
    )
    plot_single_vs_multiple_pairs(
        near_resolved,
        out_plot_dir / "06_cluster_pairs_gaia_single_vs_multiple.png",
        pairs_per_cluster=int(args.pairs_per_cluster),
    )

    miss_df = pd.DataFrame(miss_d + miss_n, columns=["source_id", "reason"])
    miss_path = out_plot_dir / "missing_stamp_sources.csv"
    miss_df.to_csv(miss_path, index=False)

    doubles_resolved.to_csv(out_plot_dir / "doubles_resolved_records.csv", index=False)
    near_resolved.to_csv(out_plot_dir / "near_non_doubles_resolved_records.csv", index=False)
    pd.DataFrame(
        {
            "metric_col": [str(args.gaia_multiple_col)],
            "multiple_if_value_gt": [float(args.gaia_multiple_threshold)],
            "n_near_total": [int(len(near_resolved))],
            "n_near_gaia_single": [int((~near_resolved["gaia_is_multiple"]).sum())],
            "n_near_gaia_multiple": [int((near_resolved["gaia_is_multiple"]).sum())],
        }
    ).to_csv(out_plot_dir / "gaia_single_multiple_split_summary.csv", index=False)

    print("Done.")
    print("  Output directory:", out_plot_dir)
    print("  Missing source records:", miss_path)


if __name__ == "__main__":
    main()
