#!/usr/bin/env python3
"""
87_thesis_umap_figures.py
=========================

Generate two thesis UMAP figures:

  Figure A — 08_umap_features_colored_panels.png (8 panels, 4×2)
    The 16D Gaia UMAP coloured by the 8 most physically informative features.
    Removes the 8 redundant/uninformative panels from the original 16-panel version.

  Figure B — 25_umap_morphology_panels.png (6 panels, 3×2)
    The same UMAP coloured by 6 morphological summary statistics,
    using predicted values from the morph_metrics_16d run.

Outputs:
  report/phd-thesis-template-2.4/Chapter4/Figs/Raster/08_umap_features_colored_panels.png
  report/phd-thesis-template-2.4/Chapter4/Figs/Raster/25_umap_morphology_panels.png
"""
from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
UMAP_CSV = BASE / "output" / "experiments" / "embeddings" / \
    "umap16d_manualv8_filtered" / "embedding_umap.csv"
PIX_UMAP_CSV = BASE / "output" / "experiments" / "embeddings" / \
    "double_stars_pixels" / "email_pixels_all_filtered" / \
    "umap_standard" / "embedding_umap.csv"
MORPH_CSV = BASE / "output" / "ml_runs" / "morph_metrics_16d" / "full_gaia_predictions.csv"
OUT_DIR = BASE / "report" / "phd-thesis-template-2.4" / "Chapter4" / "Figs" / "Raster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load embedding ────────────────────────────────────────────────────────────
print("Loading UMAP embedding …")
df = pd.read_csv(UMAP_CSV)
x = df["x"].values
y = df["y"].values

# ── Plotting helpers ──────────────────────────────────────────────────────────
DOT_SIZE = 1.0
ALPHA     = 0.6
DPI       = 220

def clip_percentile(vals: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    lo_v, hi_v = np.nanpercentile(vals, lo), np.nanpercentile(vals, hi)
    return np.clip(vals, lo_v, hi_v)

def scatter_panel(ax, vals, label, cmap="viridis"):
    finite = np.isfinite(vals)
    v = clip_percentile(vals[finite])
    sc = ax.scatter(x[finite], y[finite], c=v, s=DOT_SIZE, alpha=ALPHA,
                    cmap=cmap, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, fontsize=14, pad=5)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=12)
    # Overplot NaN sources in dark grey to keep density faithful
    nan_mask = ~finite
    if nan_mask.sum() > 0:
        ax.scatter(x[nan_mask], y[nan_mask], c="#cccccc", s=DOT_SIZE * 0.5,
                   alpha=0.3, linewidths=0, rasterized=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure A — 8 selected Gaia features (4 rows × 2 cols)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    ("feat_log10_snr",             "SNR  ($\\log_{10}$)"),
    ("feat_ruwe",                  "RUWE"),
    ("feat_astrometric_excess_noise", "Astrometric excess noise"),
    ("feat_pm_significance",       "PM significance"),
    ("feat_ipd_frac_multi_peak",   "IPD multi-peak fraction"),
    ("feat_ipd_gof_harmonic_amplitude", "IPD harmonic amplitude"),
    ("feat_ipd_gof_harmonic_phase","IPD harmonic phase"),
    ("feat_c_star",                "$c^\\star$  (BP+RP flux excess)"),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, (col, label) in zip(axes.flat, FEATURES):
    vals = df[col].values
    scatter_panel(ax, vals, label)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
out_a = OUT_DIR / "08_umap_features_colored_panels.png"
fig.savefig(out_a, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_a}")

# ─────────────────────────────────────────────────────────────────────────────
# Figures B & C — morphology metrics panels
# ─────────────────────────────────────────────────────────────────────────────
print("Loading morph predictions …")
MORPH_COLS_ALL = [
    "pred_gini", "pred_kurtosis", "pred_smoothness", "pred_ellipticity",
    "pred_multipeak_ratio", "pred_asymmetry_180", "pred_concentration",
    "pred_roundness", "pred_mirror_asymmetry", "pred_m20", "pred_hf_artifacts",
]
morph = pd.read_csv(MORPH_CSV, usecols=["source_id"] + MORPH_COLS_ALL)
merged = df[["source_id", "x", "y"]].merge(morph, on="source_id", how="left")
xm = merged["x"].values
ym = merged["y"].values

# Figure B — 6 selected panels for synthesis (25_umap_morphology_panels.png)
MORPH_PANELS_6 = [
    ("pred_gini",           "Gini coefficient"),
    ("pred_kurtosis",       "Kurtosis"),
    ("pred_smoothness",     "Smoothness"),
    ("pred_ellipticity",    "Ellipticity"),
    ("pred_multipeak_ratio","Multi-peak ratio"),
    ("pred_asymmetry_180",  "Asymmetry (180°)"),
]

fig2, axes2 = plt.subplots(2, 3, figsize=(13, 8))
for ax, (col, label) in zip(axes2.flat, MORPH_PANELS_6):
    vals = merged[col].values
    finite = np.isfinite(vals)
    v = clip_percentile(vals[finite])
    sc = ax.scatter(xm[finite], ym[finite], c=v, s=DOT_SIZE, alpha=ALPHA,
                    cmap="viridis", linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, fontsize=14, pad=5)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=12)

fig2.tight_layout(h_pad=1.5, w_pad=1.0)
out_b = OUT_DIR / "25_umap_morphology_panels.png"
fig2.savefig(out_b, dpi=DPI, bbox_inches="tight")
plt.close(fig2)
print(f"Saved {out_b}")

# Figure C — pixel UMAP coloured by true morphology stats (18_pixel_umap_morphology_panels.png)
print("Loading pixel UMAP …")
pix = pd.read_csv(PIX_UMAP_CSV, low_memory=False)
pix = pix.dropna(subset=["x", "y"])
xp = pix["x"].values
yp = pix["y"].values

PIX_MORPH_PANELS = [
    ("morph_gini",                "Gini coefficient"),
    ("morph_peakedness_kurtosis", "Kurtosis"),
    ("morph_smoothness",          "Smoothness"),
    ("morph_concentration_r80r20","Concentration"),
    ("morph_asym_180",            "Asymmetry (180°)"),
    ("morph_mirror_asym_lr",      "Mirror asymmetry"),
    ("morph_m20",                 "$M_{20}$"),
    ("morph_peak_to_total",       "Peak-to-total flux"),
    ("morph_ellipticity_e1",      "Ellipticity $e_1$"),
]

fig3, axes3 = plt.subplots(3, 3, figsize=(13, 12))
for ax, (col, label) in zip(axes3.flat, PIX_MORPH_PANELS):
    vals = pd.to_numeric(pix[col], errors="coerce").values
    finite = np.isfinite(vals)
    v = clip_percentile(vals[finite])
    sc = ax.scatter(xp[finite], yp[finite], c=v, s=DOT_SIZE, alpha=ALPHA,
                    cmap="viridis", linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, fontsize=14, pad=5)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=12)

fig3.tight_layout(h_pad=1.5, w_pad=1.0)
out_c = OUT_DIR / "18_pixel_umap_morphology_panels.png"
fig3.savefig(out_c, dpi=DPI, bbox_inches="tight")
plt.close(fig3)
print(f"Saved {out_c}")
print("Done.")
