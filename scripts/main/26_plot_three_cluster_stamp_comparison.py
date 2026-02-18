#!/usr/bin/env python3
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


def build_lookup(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for meta in sorted(dataset_root.glob("*/metadata.csv")):
        default_tag = meta.parent.name
        usecols = ["source_id", "npz_file", "index_in_file", "fits_path"]
        df = pd.read_csv(meta, usecols=lambda c: c in set(usecols), low_memory=False)
        for c in ["source_id", "index_in_file"]:
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
        rows.append(df[["source_id", "field_tag", "npz_path", "index_in_file"]])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("source_id").drop_duplicates(subset=["source_id"], keep="first")
    return out.reset_index(drop=True)


def fetch_stamps(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    m = df.copy()
    m["source_id"] = pd.to_numeric(m["source_id"], errors="coerce")
    m = m.dropna(subset=["source_id"]).copy()
    m["source_id"] = m["source_id"].astype(np.int64)
    m = m.merge(lookup, on="source_id", how="left", suffixes=("", "_meta"))

    cache: Dict[str, np.ndarray] = {}
    stamps: List[np.ndarray] = []

    for _, r in m.iterrows():
        p = r.get("npz_path")
        idx = r.get("index_in_file")
        if pd.isna(p) or pd.isna(idx):
            stamps.append(None)
            continue
        p = str(p)
        idx = int(idx)
        pp = Path(p)
        if not pp.exists():
            stamps.append(None)
            continue
        try:
            if p not in cache:
                cache[p] = np.load(p)["X"]
            x = cache[p]
            if idx < 0 or idx >= x.shape[0]:
                stamps.append(None)
                continue
            stamps.append(np.array(x[idx], dtype=np.float32))
        except Exception:
            stamps.append(None)

    m["stamp"] = stamps
    return m


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


def pick_rows(df: pd.DataFrame, n: int, mode: str, seed: int) -> pd.DataFrame:
    vis = df[df["stamp"].notna()].copy()
    if mode == "top":
        vis = vis.sort_values("local_double_frac_k50", ascending=False)
        return vis.head(n).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if len(vis) <= n:
        return vis.reset_index(drop=True)
    idx = rng.choice(len(vis), size=n, replace=False)
    return vis.iloc[idx].reset_index(drop=True)


def plot_three_clusters(c0: pd.DataFrame, c1: pd.DataFrame, c2: pd.DataFrame, out_png: Path, title: str) -> None:
    n = max(len(c0), len(c1), len(c2))
    if n == 0:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No resolved stamps for clusters 0/1/2", ha="center", va="center")
        plt.axis("off")
        savefig(out_png)
        return

    fig, axes = plt.subplots(n, 3, figsize=(10.5, 2.35 * n))
    if n == 1:
        axes = np.array([axes])

    cols = [c0, c1, c2]
    labels = ["Cluster 0", "Cluster 1", "Cluster 2"]

    for r in range(n):
        for c in range(3):
            ax = axes[r, c]
            df = cols[c]
            if r < len(df):
                row = df.iloc[r]
                img = row["stamp"]
                lo, hi = robust_limits(img)
                ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi)
                sid = int(row["source_id"])
                lf = float(row.get("local_double_frac_k50", np.nan))
                ax.set_title(f"{labels[c]}\n{sid} | local={lf:.2f}", fontsize=7)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title, fontsize=14)
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side stamp comparison for clusters 0/1/2")
    ap.add_argument("--dataset_root", default="/data/yn316/Codes/output/dataset_npz")
    ap.add_argument("--doubles_csv", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/double_candidates_ranked_by_local_enrichment_umap.csv")
    ap.add_argument("--n_rows", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_plot_dir", default="/data/yn316/Codes/plots/qa/embeddings/double_stars_8d/cluster_type_comparison")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    doubles_csv = Path(args.doubles_csv)
    out_plot = ensure_dir(Path(args.out_plot_dir))

    df = pd.read_csv(doubles_csv)
    df["double_cluster"] = pd.to_numeric(df["double_cluster"], errors="coerce")
    df = df.dropna(subset=["double_cluster", "source_id"]).copy()
    df["double_cluster"] = df["double_cluster"].astype(int)

    d0 = df[df["double_cluster"] == 0].copy()
    d1 = df[df["double_cluster"] == 1].copy()
    d2 = df[df["double_cluster"] == 2].copy()

    lookup = build_lookup(dataset_root)
    d0 = fetch_stamps(d0, lookup)
    d1 = fetch_stamps(d1, lookup)
    d2 = fetch_stamps(d2, lookup)

    c0_top = pick_rows(d0, args.n_rows, mode="top", seed=args.seed)
    c1_top = pick_rows(d1, args.n_rows, mode="top", seed=args.seed)
    c2_top = pick_rows(d2, args.n_rows, mode="top", seed=args.seed)

    c0_rnd = pick_rows(d0, args.n_rows, mode="random", seed=args.seed + 1)
    c1_rnd = pick_rows(d1, args.n_rows, mode="random", seed=args.seed + 2)
    c2_rnd = pick_rows(d2, args.n_rows, mode="random", seed=args.seed + 3)

    plot_three_clusters(
        c0_top,
        c1_top,
        c2_top,
        out_plot / "04_clusters_0_1_2_side_by_side_top_local.png",
        title="Clusters 0 vs 1 vs 2 (top by local enrichment)",
    )
    plot_three_clusters(
        c0_rnd,
        c1_rnd,
        c2_rnd,
        out_plot / "05_clusters_0_1_2_side_by_side_random.png",
        title="Clusters 0 vs 1 vs 2 (random samples)",
    )

    print("Done.")
    print("Saved:", out_plot / "04_clusters_0_1_2_side_by_side_top_local.png")
    print("Saved:", out_plot / "05_clusters_0_1_2_side_by_side_random.png")


if __name__ == "__main__":
    main()
