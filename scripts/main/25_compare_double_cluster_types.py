#!/usr/bin/env python3
"""
Compare stamp morphology between two sets of double-star clusters.
Default comparison:
- Type A: clusters 0 and 1
- Type B: cluster 2

Inputs:
- output/experiments/embeddings/double_stars_8d/umap_standard/double_candidates_ranked_by_local_enrichment_umap.csv
- output/dataset_npz/*/metadata.csv + NPZ files

Outputs:
- plots/qa/embeddings/double_stars_8d/cluster_type_comparison/
- output/experiments/embeddings/double_stars_8d/umap_standard/cluster_type_comparison/
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


def savefig(path: Path, dpi: int = 240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def infer_field_tag(path_str: str) -> str:
    try:
        p = Path(path_str)
        return p.parent.name if p.parent.name else "unknown"
    except Exception:
        return "unknown"


def build_source_lookup(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(dataset_root.glob("*/metadata.csv")):
        default_tag = meta_path.parent.name
        usecols = ["source_id", "npz_file", "index_in_file", "fits_path", "phot_g_mean_mag"]
        df = pd.read_csv(meta_path, usecols=lambda c: c in set(usecols), low_memory=False)

        for c in ["source_id", "index_in_file", "phot_g_mean_mag"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["source_id", "index_in_file", "npz_file"]).copy()
        df["source_id"] = df["source_id"].astype(np.int64)
        df["index_in_file"] = df["index_in_file"].astype(np.int64)

        if "fits_path" in df.columns:
            tags = df["fits_path"].astype(str).map(infer_field_tag)
            df["field_tag"] = np.where(tags == "unknown", default_tag, tags)
        else:
            df["field_tag"] = default_tag

        df["npz_path"] = df.apply(lambda r: str(dataset_root / str(r["field_tag"]) / str(r["npz_file"])), axis=1)
        rows.append(df[["source_id", "field_tag", "npz_path", "index_in_file", "phot_g_mean_mag"]])

    all_meta = pd.concat(rows, ignore_index=True)
    all_meta = all_meta.sort_values("source_id").drop_duplicates(subset=["source_id"], keep="first")
    return all_meta.reset_index(drop=True)


def fetch_stamps(df: pd.DataFrame, lookup: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rec = df.copy()
    rec["source_id"] = pd.to_numeric(rec["source_id"], errors="coerce")
    rec = rec.dropna(subset=["source_id"]).copy()
    rec["source_id"] = rec["source_id"].astype(np.int64)

    m = rec.merge(lookup, on="source_id", how="left", suffixes=("", "_meta"))
    cache: Dict[str, np.ndarray] = {}
    stamps = []
    miss = []

    for _, r in m.iterrows():
        p = r.get("npz_path")
        idx = r.get("index_in_file")
        sid = int(r["source_id"])

        if pd.isna(p) or pd.isna(idx):
            miss.append({"source_id": sid, "reason": "missing_lookup"})
            stamps.append(None)
            continue

        p = str(p)
        idx = int(idx)
        pp = Path(p)
        if not pp.exists():
            miss.append({"source_id": sid, "reason": f"npz_not_found:{p}"})
            stamps.append(None)
            continue

        try:
            if p not in cache:
                arr = np.load(p)
                cache[p] = arr["X"]
            x = cache[p]
            if idx < 0 or idx >= x.shape[0]:
                miss.append({"source_id": sid, "reason": f"index_out_of_range:{idx}"})
                stamps.append(None)
                continue
            stamps.append(np.array(x[idx], dtype=np.float32))
        except Exception as e:
            miss.append({"source_id": sid, "reason": f"load_error:{e}"})
            stamps.append(None)

    m["stamp"] = stamps
    miss_df = pd.DataFrame(miss, columns=["source_id", "reason"])
    return m, miss_df


def robust_limits(img: np.ndarray) -> Tuple[float, float]:
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [1, 99])
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1e-6
    return float(lo), float(hi)


def normalize_stamp(img: np.ndarray) -> np.ndarray:
    a = np.array(img, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    bg = np.percentile(a, 10)
    a = a - bg
    a[a < 0] = 0
    s = np.sum(a)
    if s > 0:
        a = a / s
    return a.astype(np.float32)


def morph_features(img: np.ndarray) -> Dict[str, float]:
    a = normalize_stamp(img)
    h, w = a.shape
    yy, xx = np.indices((h, w))
    s = float(np.sum(a))
    if s <= 0:
        return {"peak_sep_pix": 0.0, "peak_ratio_2over1": 0.0, "ellipticity": 0.0}

    cx = float(np.sum(a * xx) / s)
    cy = float(np.sum(a * yy) / s)
    dx = xx - cx
    dy = yy - cy
    qxx = float(np.sum(a * dx * dx) / s)
    qyy = float(np.sum(a * dy * dy) / s)
    qxy = float(np.sum(a * dx * dy) / s)
    tr = qxx + qyy
    det = qxx * qyy - qxy * qxy
    disc = max(tr * tr - 4 * det, 0.0)
    lam1 = max((tr + math.sqrt(disc)) * 0.5, 0.0)
    lam2 = max((tr - math.sqrt(disc)) * 0.5, 0.0)
    a1 = math.sqrt(lam1)
    b1 = math.sqrt(lam2)
    ell = (a1 - b1) / (a1 + b1 + 1e-9)

    flat = a.ravel()
    idx2 = np.argpartition(flat, -2)[-2:]
    vals = flat[idx2]
    ord2 = np.argsort(vals)[::-1]
    i1, i2 = idx2[ord2[0]], idx2[ord2[1]]
    y1, x1 = divmod(int(i1), w)
    y2, x2 = divmod(int(i2), w)
    p1, p2 = float(flat[i1]), float(flat[i2])

    return {
        "peak_sep_pix": float(math.hypot(x2 - x1, y2 - y1)),
        "peak_ratio_2over1": float(p2 / (p1 + 1e-9)),
        "ellipticity": float(ell),
    }


def plot_dual_gallery(a_df: pd.DataFrame, b_df: pd.DataFrame, out_png: Path, n_show: int = 18, ncols: int = 6) -> None:
    aa = a_df[a_df["stamp"].notna()].sort_values("local_double_frac_k50", ascending=False).head(n_show).reset_index(drop=True)
    bb = b_df[b_df["stamp"].notna()].sort_values("local_double_frac_k50", ascending=False).head(n_show).reset_index(drop=True)

    n = max(len(aa), len(bb), n_show)
    nrows = int(math.ceil(n / ncols))

    fig = plt.figure(figsize=(2.6 * ncols * 2, 2.8 * nrows + 1.5))
    gs = fig.add_gridspec(nrows, ncols * 2)

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)

        ax1 = fig.add_subplot(gs[r, c])
        if i < len(aa):
            im = aa.loc[i, "stamp"]
            lo, hi = robust_limits(im)
            ax1.imshow(im, origin="lower", cmap="gray", vmin=lo, vmax=hi)
            sid = int(aa.loc[i, "source_id"])
            cl = int(aa.loc[i, "double_cluster"])
            ax1.set_title(f"{sid}\ncl={cl}", fontsize=6)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[r, c + ncols])
        if i < len(bb):
            im = bb.loc[i, "stamp"]
            lo, hi = robust_limits(im)
            ax2.imshow(im, origin="lower", cmap="gray", vmin=lo, vmax=hi)
            sid = int(bb.loc[i, "source_id"])
            cl = int(bb.loc[i, "double_cluster"])
            ax2.set_title(f"{sid}\ncl={cl}", fontsize=6)
        ax2.set_xticks([])
        ax2.set_yticks([])

    fig.suptitle("Left: Type A (clusters 0+1)   |   Right: Type B (cluster 2)", fontsize=14)
    savefig(out_png)


def plot_stacks_and_diff(a_df: pd.DataFrame, b_df: pd.DataFrame, out_png: Path) -> None:
    aa = a_df[a_df["stamp"].notna()].copy()
    bb = b_df[b_df["stamp"].notna()].copy()

    a_stack = np.stack([normalize_stamp(x) for x in aa["stamp"].to_list()], axis=0)
    b_stack = np.stack([normalize_stamp(x) for x in bb["stamp"].to_list()], axis=0)

    a_mean = np.mean(a_stack, axis=0)
    b_mean = np.mean(b_stack, axis=0)
    a_med = np.median(a_stack, axis=0)
    b_med = np.median(b_stack, axis=0)
    diff_mean = a_mean - b_mean
    diff_med = a_med - b_med

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for ax, img, ttl in [
        (axes[0, 0], a_mean, "Type A mean (norm)"),
        (axes[0, 1], b_mean, "Type B mean (norm)"),
        (axes[0, 2], diff_mean, "Mean difference A-B"),
        (axes[1, 0], a_med, "Type A median (norm)"),
        (axes[1, 1], b_med, "Type B median (norm)"),
        (axes[1, 2], diff_med, "Median difference A-B"),
    ]:
        if "difference" in ttl:
            vmax = np.max(np.abs(img)) + 1e-9
            ax.imshow(img, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        else:
            ax.imshow(img, origin="lower", cmap="magma")
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Stacked stamp comparison between cluster types", fontsize=13)
    savefig(out_png)


def plot_morph_hist(a_df: pd.DataFrame, b_df: pd.DataFrame, out_png: Path) -> pd.DataFrame:
    recs = []
    for gname, df in [("A_0plus1", a_df), ("B_2", b_df)]:
        for _, r in df.iterrows():
            if not isinstance(r.get("stamp"), np.ndarray):
                continue
            f = morph_features(r["stamp"])
            f["group"] = gname
            f["source_id"] = int(r["source_id"])
            f["double_cluster"] = int(r["double_cluster"])
            recs.append(f)

    mdf = pd.DataFrame(recs)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, col, ttl in [
        (axes[0], "peak_sep_pix", "Peak separation [pix]"),
        (axes[1], "peak_ratio_2over1", "Second/first peak ratio"),
        (axes[2], "ellipticity", "Ellipticity"),
    ]:
        for grp, color in [("A_0plus1", "#d62828"), ("B_2", "#1d3557")]:
            v = mdf.loc[mdf["group"] == grp, col].to_numpy(dtype=float)
            if len(v) == 0:
                continue
            v = np.clip(v, np.percentile(v, 1), np.percentile(v, 99))
            ax.hist(v, bins=16, density=True, histtype="step", linewidth=2.0, color=color, label=grp)
        ax.set_title(ttl, fontsize=10)
    axes[0].legend(frameon=True, fontsize=9)
    fig.suptitle("Morphology proxy comparison: type A vs type B", fontsize=13)
    savefig(out_png)
    return mdf


def parse_cluster_list(s: str) -> List[int]:
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare stamp types between selected double-cluster groups")
    ap.add_argument("--dataset_root", default="/data/yn316/Codes/output/dataset_npz")
    ap.add_argument("--doubles_csv", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/double_candidates_ranked_by_local_enrichment_umap.csv")
    ap.add_argument("--group_a", default="0,1", help="Comma-separated cluster ids for type A")
    ap.add_argument("--group_b", default="2", help="Comma-separated cluster ids for type B")
    ap.add_argument("--n_show", type=int, default=18)
    ap.add_argument("--out_plot_dir", default="/data/yn316/Codes/plots/qa/embeddings/double_stars_8d/cluster_type_comparison")
    ap.add_argument("--out_tab_dir", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/cluster_type_comparison")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    doubles_csv = Path(args.doubles_csv)
    out_plot = ensure_dir(Path(args.out_plot_dir))
    out_tab = ensure_dir(Path(args.out_tab_dir))

    a_clusters = parse_cluster_list(args.group_a)
    b_clusters = parse_cluster_list(args.group_b)

    df = pd.read_csv(doubles_csv)
    df["double_cluster"] = pd.to_numeric(df["double_cluster"], errors="coerce")
    df = df.dropna(subset=["double_cluster", "source_id"]).copy()
    df["double_cluster"] = df["double_cluster"].astype(int)

    a_df = df[df["double_cluster"].isin(a_clusters)].copy()
    b_df = df[df["double_cluster"].isin(b_clusters)].copy()

    if len(a_df) == 0 or len(b_df) == 0:
        raise RuntimeError("One cluster group is empty. Check --group_a/--group_b values.")

    print(f"Type A clusters {a_clusters}: {len(a_df)} sources")
    print(f"Type B clusters {b_clusters}: {len(b_df)} sources")

    print("Building source lookup...")
    lookup = build_source_lookup(dataset_root)

    print("Resolving stamps...")
    a_res, miss_a = fetch_stamps(a_df, lookup)
    b_res, miss_b = fetch_stamps(b_df, lookup)

    print("Plotting comparisons...")
    plot_dual_gallery(a_res, b_res, out_plot / "01_typeA_vs_typeB_stamp_galleries.png", n_show=int(args.n_show), ncols=6)
    plot_stacks_and_diff(a_res, b_res, out_plot / "02_typeA_vs_typeB_stacks_and_diff.png")
    morph = plot_morph_hist(a_res, b_res, out_plot / "03_typeA_vs_typeB_morph_proxy_hist.png")

    # Save tables
    a_res.to_csv(out_tab / "typeA_resolved_records.csv", index=False)
    b_res.to_csv(out_tab / "typeB_resolved_records.csv", index=False)
    morph.to_csv(out_tab / "typeA_vs_typeB_morph_features.csv", index=False)

    summary = []
    for name, d in [("A", morph[morph["group"] == "A_0plus1"]), ("B", morph[morph["group"] == "B_2"])]:
        summary.append(
            {
                "group": name,
                "n": int(len(d)),
                "peak_sep_pix_median": float(np.nanmedian(d["peak_sep_pix"])),
                "peak_ratio_2over1_median": float(np.nanmedian(d["peak_ratio_2over1"])),
                "ellipticity_median": float(np.nanmedian(d["ellipticity"])),
            }
        )
    pd.DataFrame(summary).to_csv(out_tab / "typeA_vs_typeB_summary.csv", index=False)

    miss = pd.concat([miss_a, miss_b], ignore_index=True)
    miss.to_csv(out_tab / "missing_stamp_sources.csv", index=False)

    print("Done.")
    print("  Plots:", out_plot)
    print("  Tables:", out_tab)


if __name__ == "__main__":
    main()
