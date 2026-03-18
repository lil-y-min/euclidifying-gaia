#!/usr/bin/env python3
"""
88_thesis_border_montage.py
============================

Regenerate 16_border_median_montage.png for the thesis.

Left panel  — per-stamp border-pixel median distribution, zoomed (0–50 ADU).
Right panel — full stamp pixel intensity distribution (all 400 pixels), full range.

Both panels use consistent field colours and a shared legend.

Inputs:
  plots/intensity_bg_zoom_raw/per_stamp_bg_stats.csv   (border_median per stamp)
  plots/intensity_by_field_smoke/intensity_hists_raw.csv  (full stamp histograms)

Output:
  report/phd-thesis-template-2.4/Chapter4/Figs/Raster/16_border_median_montage.png
"""
from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE    = Path(__file__).resolve().parents[2]
BG_CSV  = BASE / "plots" / "intensity_bg_zoom_raw" / "per_stamp_bg_stats.csv"
INT_CSV = BASE / "plots" / "intensity_by_field_smoke" / "intensity_hists_raw.csv"
OUT     = BASE / "report" / "phd-thesis-template-2.4" / "Chapter4" / "Figs" / "Raster" / \
          "16_border_median_montage.png"

# ── Field colour palette (matches UMAP field-coloured figure) ─────────────────
FIELD_COLOURS = {
    "ERO-Abell2390":  "#e41a1c",
    "ERO-Abell2764":  "#ff7f00",
    "ERO-Barnard30":  "#4daf4a",
    "ERO-Dorado":     "#984ea3",
    "ERO-Fornax":     "#a65628",
    "ERO-HolmbergII": "#f781bf",
    "ERO-Horsehead":  "#999999",
    "ERO-IC10":       "#377eb8",
    "ERO-IC342":      "#a6cee3",
    "ERO-Messier78":  "#b2df8a",
    "ERO-NGC2403":    "#fb9a99",
    "ERO-NGC6254":    "#fdbf6f",
    "ERO-NGC6397":    "#cab2d6",
    "ERO-NGC6744":    "#ffff99",
    "ERO-NGC6822":    "#6a3d9a",
    "ERO-Perseus":    "#1f78b4",
    "ERO-Taurus":     "#33a02c",
}

SHORT = {k: k.replace("ERO-", "") for k in FIELD_COLOURS}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading border median data …")
bg = pd.read_csv(BG_CSV)
print("Loading full-stamp intensity histograms …")
ints = pd.read_csv(INT_CSV)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

BW_ZOOM = 1.0   # ADU bin width for zoom panel
ZOOM_MAX = 50.0

for field, colour in FIELD_COLOURS.items():
    sub = bg[bg["field"] == field]["border_median"].dropna().values
    if len(sub) == 0:
        continue
    bins = np.arange(-10, ZOOM_MAX + BW_ZOOM, BW_ZOOM)
    counts, edges = np.histogram(sub, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax_left.plot(centres, counts, color=colour, lw=0.8, alpha=0.85,
                 label=SHORT[field])

ax_left.set_yscale("log")
ax_left.set_xlim(-10, ZOOM_MAX)
ax_left.set_xlabel("Border-pixel median per stamp [ADU]", fontsize=14)
ax_left.set_ylabel("Count (log scale)", fontsize=14)
ax_left.set_title("Border-pixel median — zoomed (0 to 50 ADU)", fontsize=15)
ax_left.tick_params(labelsize=13)

for field, colour in FIELD_COLOURS.items():
    sub = ints[ints["field"] == field]
    if sub.empty:
        continue
    bc = sub["bin_center"].values
    cn = sub["count_norm"].values
    order = np.argsort(bc)
    ax_right.plot(bc[order], cn[order], color=colour, lw=0.8, alpha=0.85,
                  label=SHORT[field])

ax_right.set_yscale("log")
ax_right.set_xlim(-500, 21000)
ax_right.set_xlabel("Pixel intensity per stamp [ADU]", fontsize=14)
ax_right.set_ylabel("Normalised count (log scale)", fontsize=14)
ax_right.set_title("Full stamp pixel distribution — all pixels", fontsize=15)
ax_right.tick_params(labelsize=13)
ax_right.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, integer=True))

# shared legend on the right of the right panel
handles, labels = ax_right.get_legend_handles_labels()
fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.02, 0.5),
           fontsize=11, frameon=True, ncol=1)

fig.tight_layout(rect=[0, 0, 0.91, 1])
fig.savefig(OUT, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT}")
