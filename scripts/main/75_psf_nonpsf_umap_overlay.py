#!/usr/bin/env python3
"""
PSF / non-PSF overlay on the pixel UMAP.

Merges the PSF classification labels (label=0 PSF-like, label=1 non-PSF-like)
into the existing pixel embedding by source_id and produces:
  - Density map with PSF / non-PSF highlighted
  - Score (probability) gradient
  - Side-by-side PSF vs non-PSF density
  - Morphology panels split by class
  - Gaia feature panels split by class

Outputs: report/model_decision/2026-03-05_umap_improvements/05_psf_nonpsf_overlay/
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE    = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "report" / "model_decision" / "2026-03-05_umap_improvements" / "05_psf_nonpsf_overlay"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_EMB_CSV = BASE / "output" / "experiments" / "embeddings" / "double_stars_pixels" / \
    "email_pixels_all_filtered" / "umap_standard" / "embedding_umap.csv"
LABELS_CSV = BASE / "output" / "ml_runs" / "nn_psf_labels" / "labels_psf_weak.csv"

MORPH_COLS = [
    "morph_asym_180", "morph_concentration_r80r20", "morph_ellipticity_e2",
    "morph_peakedness_kurtosis", "morph_gini", "morph_smoothness",
    "morph_edge_asym_180", "morph_peak_to_total",
]
MORPH_LABELS = {
    "morph_asym_180": "Asymmetry 180", "morph_concentration_r80r20": "Concentration r80/r20",
    "morph_ellipticity_e2": "Ellipticity e2", "morph_peakedness_kurtosis": "Kurtosis",
    "morph_gini": "Gini", "morph_smoothness": "Smoothness",
    "morph_edge_asym_180": "Edge Asymmetry", "morph_peak_to_total": "Peak/Total",
}
FEAT_COLS = [
    "feat_log10_snr", "feat_ruwe", "feat_astrometric_excess_noise",
    "feat_ipd_frac_multi_peak", "feat_c_star", "feat_pm_significance",
]
FEAT_LABELS = {
    "feat_log10_snr": "log10 SNR", "feat_ruwe": "RUWE",
    "feat_astrometric_excess_noise": "Astrometric Excess Noise",
    "feat_ipd_frac_multi_peak": "IPD Frac Multi-Peak",
    "feat_c_star": "C*", "feat_pm_significance": "PM Significance",
}

PSF_COLOR    = "#1d3557"   # dark blue  — PSF-like
NONPSF_COLOR = "#e63946"   # red        — non-PSF-like


def savefig(path: Path, dpi: int = 200) -> None:
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


def plot_density_with_overlay(df, out_png):
    """Hexbin background + PSF / non-PSF scatter overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, label_val, color, title in [
        (axes[0], 0, PSF_COLOR,    "PSF-like  (label=0)"),
        (axes[1], 1, NONPSF_COLOR, "non-PSF-like  (label=1)"),
    ]:
        ax.hexbin(df["x"], df["y"], gridsize=160, bins="log", cmap="Greys", mincnt=1)
        sub = df[df["label"] == label_val]
        ax.scatter(sub["x"], sub["y"], s=4, color=color, alpha=0.5,
                   linewidths=0, rasterized=True, label=f"{title}  n={len(sub):,}")
        ax.set_title(f"Pixel UMAP — {title}\nn={len(sub):,} / {len(df):,} total", fontsize=11)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Pixel UMAP — PSF-like vs non-PSF-like overlay", fontsize=13)
    fig.tight_layout()
    savefig(out_png)


def plot_label_colored(df, out_png):
    """Single scatter coloured by label (PSF=0 blue, non-PSF=1 red)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for label_val, color, name in [
        (0, PSF_COLOR,    "PSF-like"),
        (1, NONPSF_COLOR, "non-PSF-like"),
    ]:
        sub = df[df["label"] == label_val]
        ax.scatter(sub["x"], sub["y"], s=3, color=color, alpha=0.45,
                   linewidths=0, rasterized=True, label=f"{name}  n={len(sub):,}")
    ax.set_xlabel("UMAP-1", fontsize=11); ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title("Pixel UMAP — PSF-like (blue) vs non-PSF-like (red)", fontsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    savefig(out_png)


def plot_score_gradient(df, out_png):
    """Colour by non-PSF score (continuous probability)."""
    if "score_non_psf_like" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    vals = pd.to_numeric(df["score_non_psf_like"], errors="coerce").to_numpy(float)
    sc = ax.scatter(df["x"], df["y"], c=vals, s=3, cmap="RdBu_r", alpha=0.65,
                    linewidths=0, vmin=0, vmax=1, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("P(non-PSF-like)", fontsize=10)
    ax.set_xlabel("UMAP-1", fontsize=11); ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title("Pixel UMAP — non-PSF probability gradient", fontsize=12)
    savefig(out_png)


def plot_panels_by_class(df, cols, labels, out_png, suptitle, cmap="viridis"):
    cols_ok = [c for c in cols if c in df.columns]
    if not cols_ok: return
    n = len(cols_ok)
    ncols = 2   # PSF / non-PSF columns
    nrows = n
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    if nrows == 1: axes = axes[np.newaxis, :]

    for i, col in enumerate(cols_ok):
        vals_all = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals_all) < 10: continue
        vmin, vmax = float(np.percentile(vals_all, 1)), float(np.percentile(vals_all, 99))

        for j, (label_val, class_name, color) in enumerate([
            (0, "PSF-like",     PSF_COLOR),
            (1, "non-PSF-like", NONPSF_COLOR),
        ]):
            ax = axes[i, j]
            sub = df[df["label"] == label_val]
            vals = np.clip(pd.to_numeric(sub[col], errors="coerce").to_numpy(float), vmin, vmax)
            sc = ax.scatter(sub["x"], sub["y"], c=vals, s=2, cmap=cmap, alpha=0.6,
                            linewidths=0, vmin=vmin, vmax=vmax, rasterized=True)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
            ax.set_title(f"{class_name} — {labels.get(col, col)}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig(out_png)


def main():
    print("[1/3] Loading data...")
    emb = pd.read_csv(PIXEL_EMB_CSV, low_memory=False)
    emb["source_id"] = pd.to_numeric(emb["source_id"], errors="coerce")
    emb = emb.dropna(subset=["source_id", "x", "y"]).copy()
    emb["source_id"] = emb["source_id"].astype(np.int64)

    lab = pd.read_csv(LABELS_CSV, low_memory=False,
                      usecols=["source_id", "label", "score_psf_like", "score_non_psf_like"])
    lab["source_id"] = pd.to_numeric(lab["source_id"], errors="coerce")
    lab = lab.dropna(subset=["source_id"]).copy()
    lab["source_id"] = lab["source_id"].astype(np.int64)
    lab = lab.drop_duplicates(subset=["source_id"], keep="first")

    df = emb.merge(lab, on="source_id", how="inner")
    n_psf    = int((df["label"] == 0).sum())
    n_nonpsf = int((df["label"] == 1).sum())
    print(f"  Matched {len(df):,} sources  |  PSF-like: {n_psf:,}  |  non-PSF: {n_nonpsf:,}")

    print("[2/3] Plotting...")
    plot_density_with_overlay(df,  OUT_DIR / "01_density_psf_nonpsf_overlay.png")
    plot_label_colored(df,         OUT_DIR / "02_label_colored.png")
    plot_score_gradient(df,        OUT_DIR / "03_nonpsf_score_gradient.png")
    plot_panels_by_class(df, MORPH_COLS, MORPH_LABELS,
                         OUT_DIR / "04_morphology_by_class.png",
                         "Pixel UMAP morphology metrics — PSF vs non-PSF")
    plot_panels_by_class(df, FEAT_COLS, FEAT_LABELS,
                         OUT_DIR / "05_gaia_features_by_class.png",
                         "Pixel UMAP Gaia features — PSF vs non-PSF", cmap="plasma")

    print("[3/3] Summary...")
    summary = {
        "n_matched": len(df), "n_psf_like": n_psf, "n_non_psf_like": n_nonpsf,
        "frac_non_psf": round(n_nonpsf / len(df), 4),
        "pixel_emb_csv": str(PIXEL_EMB_CSV), "labels_csv": str(LABELS_CSV),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== DONE ===")
    print("Out dir:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("*.png")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
