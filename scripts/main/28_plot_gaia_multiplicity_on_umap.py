#!/usr/bin/env python3
"""
Plot Gaia multiplicity status of doubles on top of the UMAP 2D map.

Inputs:
- output/experiments/embeddings/double_stars_8d/umap_standard/embedding_umap.csv
- output/experiments/embeddings/double_stars_8d/umap_standard/gaia_multiplicity_check/gaia_multiplicity_per_source_clear.csv

Outputs:
- plots/qa/embeddings/double_stars_8d/gaia_multiplicity_check/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def load_inputs(embedding_csv: Path, mult_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    emb = pd.read_csv(embedding_csv, usecols=["source_id", "x", "y"])
    mul = pd.read_csv(mult_csv)
    for c in ["source_id", "x", "y"]:
        if c in emb.columns:
            emb[c] = pd.to_numeric(emb[c], errors="coerce")
        if c in mul.columns:
            mul[c] = pd.to_numeric(mul[c], errors="coerce")
    emb = emb.dropna(subset=["source_id", "x", "y"]).copy()
    mul = mul.dropna(subset=["source_id", "x", "y"]).copy()
    emb["source_id"] = emb["source_id"].astype(np.int64)
    mul["source_id"] = mul["source_id"].astype(np.int64)
    return emb, mul


def _status_palette() -> dict[str, str]:
    return {
        "single_detected": "#1f77b4",
        "multiple_detected": "#d62728",
    }


def plot_status_overlay(
    emb: pd.DataFrame,
    mul: pd.DataFrame,
    status_col: str,
    out_png: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10.5, 8.5))
    plt.scatter(emb["x"], emb["y"], s=3, c="0.85", alpha=0.45, linewidths=0, label="all sources")

    colors = _status_palette()
    markers = {"A_0plus1": "o", "B_2": "s"}

    for typ in ["A_0plus1", "B_2"]:
        for status in ["single_detected", "multiple_detected"]:
            sub = mul[(mul["cluster_type"] == typ) & (mul[status_col] == status)]
            if sub.empty:
                continue
            label = f"{typ} | {status} (n={len(sub)})"
            plt.scatter(
                sub["x"],
                sub["y"],
                s=50,
                c=colors[status],
                marker=markers.get(typ, "o"),
                edgecolors="white",
                linewidths=0.5,
                alpha=0.95,
                label=label,
            )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.legend(frameon=True, fontsize=9, loc="best")
    savefig(out_png)


def plot_status_panels(
    emb: pd.DataFrame,
    mul: pd.DataFrame,
    status_col: str,
    out_png: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True, sharey=True)
    colors = _status_palette()

    for ax, typ in zip(axes, ["A_0plus1", "B_2"]):
        ax.scatter(emb["x"], emb["y"], s=2, c="0.88", alpha=0.35, linewidths=0)
        sub = mul[mul["cluster_type"] == typ].copy()
        for status in ["single_detected", "multiple_detected"]:
            ss = sub[sub[status_col] == status]
            if ss.empty:
                continue
            ax.scatter(
                ss["x"],
                ss["y"],
                s=55,
                c=colors[status],
                edgecolors="white",
                linewidths=0.5,
                alpha=0.95,
                label=f"{status} (n={len(ss)})",
            )
        ax.set_title(typ)
        ax.set_xlabel("UMAP-1")
        ax.legend(frameon=True, fontsize=9, loc="best")
    axes[0].set_ylabel("UMAP-2")
    fig.suptitle(title, fontsize=13)
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Gaia multiplicity status on UMAP map")
    ap.add_argument(
        "--embedding_csv",
        default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/embedding_umap.csv",
    )
    ap.add_argument(
        "--multiplicity_csv",
        default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/gaia_multiplicity_check/gaia_multiplicity_per_source_clear.csv",
    )
    ap.add_argument(
        "--out_plot_dir",
        default="/data/yn316/Codes/plots/qa/embeddings/double_stars_8d/gaia_multiplicity_check",
    )
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_plot_dir))
    emb, mul = load_inputs(Path(args.embedding_csv), Path(args.multiplicity_csv))

    plot_status_overlay(
        emb,
        mul,
        status_col="gaia_status_r10",
        out_png=out_dir / "03_umap_gaia_single_vs_multiple_r10_overlay.png",
        title="Gaia multiplicity on UMAP (<=1.0 arcsec criterion)",
    )
    plot_status_overlay(
        emb,
        mul,
        status_col="gaia_status_r20",
        out_png=out_dir / "04_umap_gaia_single_vs_multiple_r20_overlay.png",
        title="Gaia multiplicity on UMAP (<=2.0 arcsec criterion)",
    )
    plot_status_panels(
        emb,
        mul,
        status_col="gaia_status_r10",
        out_png=out_dir / "05_umap_gaia_single_vs_multiple_r10_panels_by_type.png",
        title="Gaia multiplicity by cluster type (<=1.0 arcsec)",
    )

    print("Done.")
    print("  Plots:", out_dir)


if __name__ == "__main__":
    main()
