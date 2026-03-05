#!/usr/bin/env python3
"""
Side-by-side comparison of 20×20 vs 10×10 pixel UMAP.

Diagnostic: do field clusters weaken when the background border is removed?
If yes, the outer pixels were driving field structure, not morphology.

Outputs: report/model_decision/2026-03-05_umap_improvements/04_20x20_vs_10x10_field/
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "report" / "model_decision" / "2026-03-05_umap_improvements" / "04_20x20_vs_10x10_field"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_20 = BASE / "output" / "experiments" / "embeddings" / "double_stars_pixels" / \
    "email_pixels_all_filtered" / "umap_standard" / "embedding_umap.csv"
CSV_10 = BASE / "output" / "experiments" / "embeddings" / "double_stars_pixels" / \
    "crop10x10_all_filtered" / "umap_standard" / "embedding_umap.csv"

LOAD_COLS = ["source_id", "field_tag", "x", "y", "phot_g_mean_mag",
             "morph_asym_180", "morph_ellipticity_e2"]


def savefig(path: Path, dpi: int = 200) -> None:
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


def load(csv: Path) -> pd.DataFrame:
    usecols = [c for c in LOAD_COLS
               if c in pd.read_csv(csv, nrows=0).columns]
    df = pd.read_csv(csv, usecols=usecols, low_memory=False)
    df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).copy()
    return df


def make_field_colormap(tags_a, tags_b):
    all_tags = sorted(set(list(tags_a) + list(tags_b)))
    cmap = plt.cm.get_cmap("tab20", max(len(all_tags), 1))
    return {tag: cmap(i) for i, tag in enumerate(all_tags)}, all_tags


def plot_field_side_by_side(df20, df10, field_cmap, all_tags, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, df, label, pix in zip(axes, [df20, df10], ["20×20 pixels", "10×10 pixels (border cropped)"], [20, 10]):
        tags = sorted(df["field_tag"].dropna().astype(str).unique())
        top_tags = df["field_tag"].value_counts().head(17).index.tolist()
        for tag in tags:
            sub = df[df["field_tag"] == tag]
            color = field_cmap.get(str(tag), "#aaaaaa")
            if tag in top_tags:
                ax.scatter(sub["x"], sub["y"], s=2, color=color, alpha=0.55,
                           linewidths=0, label=tag, rasterized=True)
            else:
                ax.scatter(sub["x"], sub["y"], s=2, color="#cccccc", alpha=0.25,
                           linewidths=0, rasterized=True)
        ax.set_title(f"Pixel UMAP — {label}\nn={len(df):,}", fontsize=12)
        ax.set_xlabel("UMAP-1", fontsize=10)
        ax.set_ylabel("UMAP-2", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Shared legend on the right
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=field_cmap.get(t, "#aaa"), markersize=6, label=t)
               for t in all_tags if t in df20["field_tag"].values or t in df10["field_tag"].values]
    fig.legend(handles=handles[:20], loc="center right", bbox_to_anchor=(1.01, 0.5),
               frameon=True, fontsize=7, ncol=1)
    fig.suptitle("20×20 vs 10×10 Pixel UMAP — Field Clustering Comparison", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    savefig(out_png)


def plot_continuous_side_by_side(df20, df10, col, label, out_png, cmap="viridis"):
    # Shared vmin/vmax across both panels
    vals_all = pd.concat([
        pd.to_numeric(df20[col], errors="coerce"),
        pd.to_numeric(df10[col], errors="coerce"),
    ]).dropna()
    if len(vals_all) < 10:
        print(f"  [SKIP] not enough data for {col}")
        return
    vmin, vmax = float(np.percentile(vals_all, 1)), float(np.percentile(vals_all, 99))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, df, title in zip(axes,
                              [df20, df10],
                              ["20×20 pixels", "10×10 pixels (border cropped)"]):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        sc = ax.scatter(df["x"], df["y"], c=np.clip(vals, vmin, vmax),
                        s=2, cmap=cmap, alpha=0.65, linewidths=0,
                        vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(f"{title}\nn={len(df):,}", fontsize=12)
        ax.set_xlabel("UMAP-1", fontsize=10)
        ax.set_ylabel("UMAP-2", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label=label)
    fig.suptitle(f"20×20 vs 10×10 — {label} (shared scale)", fontsize=13)
    savefig(out_png)


def main():
    print("[1/3] Loading embeddings...")
    df20 = load(CSV_20)
    df10 = load(CSV_10)
    print(f"  20×20: {len(df20):,} rows")
    print(f"  10×10: {len(df10):,} rows")

    tags20 = df20["field_tag"].dropna().astype(str).unique().tolist()
    tags10 = df10["field_tag"].dropna().astype(str).unique().tolist()
    field_cmap, all_tags = make_field_colormap(tags20, tags10)
    print(f"  {len(all_tags)} unique fields across both runs")

    print("[2/3] Plotting...")
    plot_field_side_by_side(df20, df10, field_cmap, all_tags,
                            OUT_DIR / "01_field_side_by_side.png")

    plot_continuous_side_by_side(df20, df10, "phot_g_mean_mag", "G magnitude",
                                 OUT_DIR / "02_gmag_side_by_side.png", cmap="magma_r")

    plot_continuous_side_by_side(df20, df10, "morph_asym_180", "Asymmetry 180",
                                 OUT_DIR / "03_asym_side_by_side.png", cmap="viridis")

    plot_continuous_side_by_side(df20, df10, "morph_ellipticity_e2", "Ellipticity e2",
                                 OUT_DIR / "04_ellipticity_side_by_side.png", cmap="plasma")

    print("[3/3] Saving summary...")
    summary = {
        "20x20": {"csv": str(CSV_20), "n_points": len(df20), "pixel_dim": 400},
        "10x10": {"csv": str(CSV_10), "n_points": len(df10), "pixel_dim": 100},
        "n_fields": len(all_tags),
        "fields": all_tags,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== DONE ===")
    print("Out dir:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("*.png")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
