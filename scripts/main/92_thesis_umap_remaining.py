#!/usr/bin/env python3
"""
92_thesis_umap_remaining.py
============================

Generate the three remaining UMAP thesis figures with consistent fonts:

  Figure 07 — 07_umap_color_by_field_tag.png
    16D Gaia UMAP coloured by ERO field tag.

  Figure 17 — 17_pixel_umap_comparison.png
    Side-by-side comparison: 20×20 vs 10×10 pixel UMAP, coloured by field.

  Figure 24 — 24_umap_classifier_overlay.png
    2×2 panels: one per classifier (galaxy disk, WISE quasar, Quaia, DESI),
    gray background + colored overlay above score threshold.

All figures use the same font constants to ensure thesis consistency.

Outputs: report/phd-thesis-template-2.4/Chapter4/Figs/Raster/
"""
from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

BASE    = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "report" / "phd-thesis-template-2.4" / "Chapter4" / "Figs" / "Raster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared font constants (consistent across all UMAP thesis figures) ─────────
FS_TITLE   = 14   # panel titles
FS_SUPTITLE= 14   # figure-level suptitle
FS_LABEL   = 13   # axis labels
FS_TICK    = 12   # tick / colorbar tick labels
FS_LEGEND  = 11   # legend entries
DPI        = 220
DOT_SIZE   = 1.0
ALPHA      = 0.25

# ── Paths ─────────────────────────────────────────────────────────────────────
UMAP_16D_CSV = BASE / "output" / "experiments" / "embeddings" / \
    "umap16d_manualv8_filtered" / "embedding_umap.csv"

PIX_20_CSV = BASE / "output" / "experiments" / "embeddings" / \
    "double_stars_pixels" / "email_pixels_all_filtered" / \
    "umap_standard" / "embedding_umap.csv"
PIX_10_CSV = BASE / "output" / "experiments" / "embeddings" / \
    "double_stars_pixels" / "crop10x10_all_filtered" / \
    "umap_standard" / "embedding_umap.csv"

GALAXY_MODEL = BASE / "output" / "ml_runs" / "galaxy_env_clf" / \
    "model_inside_galaxy_15d.json"
WISE_SCORES  = BASE / "output" / "ml_runs" / "quasar_wise_clf" / "scores_all.csv"
QUAIA_SCORES = BASE / "output" / "ml_runs" / "quaia_clf"        / "scores_all.csv"
DESI_SCORES  = BASE / "output" / "ml_runs" / "galaxy_desi_clf"  / "scores_all.csv"

GALAXY_FEATURES = [
    "feat_log10_snr", "feat_ruwe", "feat_astrometric_excess_noise",
    "feat_parallax_over_error", "feat_ipd_frac_multi_peak",
    "feat_c_star", "feat_pm_significance",
    "feat_astrometric_excess_noise_sig", "feat_ipd_gof_harmonic_amplitude",
    "feat_ipd_gof_harmonic_phase", "feat_ipd_frac_odd_win",
    "feat_phot_bp_n_contaminated_transits", "feat_phot_bp_n_blended_transits",
    "feat_phot_rp_n_contaminated_transits", "feat_phot_rp_n_blended_transits",
]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 07 — 16D UMAP coloured by field tag
# ─────────────────────────────────────────────────────────────────────────────

def make_fig07():
    print("Figure 07: loading 16D UMAP …")
    df = pd.read_csv(UMAP_16D_CSV, usecols=["source_id", "field_tag", "x", "y"])
    df = df.dropna(subset=["x", "y"])

    tags = sorted(df["field_tag"].dropna().astype(str).unique().tolist())
    cmap = plt.get_cmap("tab20", max(len(tags), 1))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, tag in enumerate(tags):
        sub = df[df["field_tag"] == tag]
        ax.scatter(sub["x"], sub["y"], s=DOT_SIZE, color=cmap(i),
                   alpha=0.85, linewidths=0, label=tag, rasterized=True)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP-1", fontsize=FS_LABEL)
    ax.set_ylabel("UMAP-2", fontsize=FS_LABEL)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=FS_LEGEND,
        markerscale=5,
    )

    fig.tight_layout()
    out = OUT_DIR / "07_umap_color_by_field_tag.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 17 — 20×20 vs 10×10 pixel UMAP side-by-side
# ─────────────────────────────────────────────────────────────────────────────

def make_fig17():
    print("Figure 17: loading pixel UMAPs …")
    load_cols = ["source_id", "field_tag", "x", "y"]

    def load_pix(csv: Path) -> pd.DataFrame:
        cols = pd.read_csv(csv, nrows=0).columns.tolist()
        use = [c for c in load_cols if c in cols]
        df = pd.read_csv(csv, usecols=use, low_memory=False)
        return df.dropna(subset=["x", "y"])

    df20 = load_pix(PIX_20_CSV)
    df10 = load_pix(PIX_10_CSV)
    print(f"  20×20: {len(df20):,}  |  10×10: {len(df10):,}")

    all_tags = sorted(set(
        df20["field_tag"].dropna().astype(str).tolist() +
        df10["field_tag"].dropna().astype(str).tolist()
    ))
    cmap = plt.get_cmap("tab20", max(len(all_tags), 1))
    tag_color = {t: cmap(i) for i, t in enumerate(all_tags)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    titles = ["Pixel UMAP — 20×20 pixels", "Pixel UMAP — 10×10 pixels (border cropped)"]

    for ax, df, title in zip(axes, [df20, df10], titles):
        top_tags = df["field_tag"].value_counts().head(17).index.tolist()
        for tag in sorted(df["field_tag"].dropna().astype(str).unique()):
            sub = df[df["field_tag"] == tag]
            if tag in top_tags:
                ax.scatter(sub["x"], sub["y"], s=DOT_SIZE + 0.5,
                           color=tag_color.get(tag, "#aaaaaa"),
                           alpha=0.5, linewidths=0, label=tag, rasterized=True)
            else:
                ax.scatter(sub["x"], sub["y"], s=DOT_SIZE,
                           color="#cccccc", alpha=0.25, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_xlabel("UMAP-1", fontsize=FS_LABEL)
        ax.set_ylabel("UMAP-2", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xticks([]); ax.set_yticks([])

    # Shared legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=tag_color.get(t, "#aaa"),
                   markersize=6, label=t)
        for t in all_tags
        if t in df20["field_tag"].values or t in df10["field_tag"].values
    ]
    fig.legend(handles=handles[:20], loc="center right",
               bbox_to_anchor=(1.01, 0.5), frameon=True,
               fontsize=FS_LEGEND, ncol=1)

    fig.suptitle("20×20 vs 10×10 pixel UMAP — field clustering comparison",
                 fontsize=FS_SUPTITLE)
    fig.tight_layout(rect=[0, 0, 0.87, 1])

    out = OUT_DIR / "17_pixel_umap_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 24 — 2×2 classifier overlay UMAP
# ─────────────────────────────────────────────────────────────────────────────

def make_fig24():
    print("Figure 24: loading 16D UMAP for classifier overlay …")
    df = pd.read_csv(UMAP_16D_CSV)
    df = df.dropna(subset=["x", "y"])
    x = df["x"].values
    y = df["y"].values
    sid = df["source_id"].astype(np.int64).values

    # Galaxy disk: predict from model
    print("  Predicting galaxy disk scores …")
    feat_ok = [f for f in GALAXY_FEATURES if f in df.columns]
    X_gal = df[feat_ok].values.astype(np.float32)
    gal_model = xgb.Booster()
    gal_model.load_model(str(GALAXY_MODEL))
    dmatrix = xgb.DMatrix(X_gal, feature_names=feat_ok)
    gal_scores = gal_model.predict(dmatrix)

    # Load scores from CSV, merge on source_id
    def load_scores(csv_path: Path) -> np.ndarray:
        sc = pd.read_csv(csv_path, usecols=["source_id", "xgb_score"])
        sc["source_id"] = sc["source_id"].astype(np.int64)
        sc = sc.drop_duplicates(subset="source_id", keep="first")
        score_map = dict(zip(sc["source_id"], sc["xgb_score"]))
        return np.array([score_map.get(s, 0.0) for s in sid], dtype=np.float32)

    print("  Loading WISE, Quaia, DESI scores …")
    wise_scores  = load_scores(WISE_SCORES)
    quaia_scores = load_scores(QUAIA_SCORES)
    desi_scores  = load_scores(DESI_SCORES)

    panels = [
        ("Galaxy disk",   gal_scores,   "#7B2D8B",  0.40),  # purple
        ("WISE quasar",   wise_scores,  "#FF6600",  0.45),  # orange
        ("Quaia quasar",  quaia_scores, "#1565C0",  0.85),  # blue
        ("DESI galaxy",   desi_scores,  "#1B7837",  0.18),  # green
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (label, scores, color, thresh) in zip(axes.flat, panels):
        above = scores >= thresh
        # Gray background
        ax.scatter(x[~above], y[~above], s=DOT_SIZE * 0.8,
                   c="#cccccc", alpha=0.2, linewidths=0, rasterized=True)
        # Colored overlay
        ax.scatter(x[above], y[above], s=DOT_SIZE * 3.0,
                   c=color, alpha=0.8, linewidths=0, rasterized=True)

        n_above = int(above.sum())
        ax.set_title(f"{label}  (n={n_above:,})", fontsize=FS_TITLE, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

        # Score threshold annotation
        ax.text(0.02, 0.02, f"threshold = {thresh:.2f}",
                transform=ax.transAxes, fontsize=FS_TICK - 1,
                color="#444444", va="bottom")

    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    out = OUT_DIR / "24_umap_classifier_overlay.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_fig07()
    make_fig17()
    make_fig24()
    print("Done.")
