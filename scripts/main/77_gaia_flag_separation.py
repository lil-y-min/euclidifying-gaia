"""
77_gaia_flag_separation.py
==========================

Cross-validation: does predicted morphology separate Gaia quality flag
distributions? If high-predicted-gini sources have systematically elevated
RUWE / ipd_frac_multi_peak / astrometric_excess_noise, the regressors encode
physically real information — not just noise.

Three Gaia quality flags (un-scaled from 16D feature array):
  RUWE                     (threshold: > 1.4 → problematic astrometry)
  ipd_frac_multi_peak      (threshold: > 10% → likely double/crowded)
  astrometric_excess_noise (threshold: > 1.0 → non-point-source excess)

Test set is binned by predicted gini into tertiles (low / mid / high).

Plots:
  18_flag_separation_overall.png  — violin plots overall (3 flags × 3 gini bins)
  19_flag_separation_by_mag.png   — same split by magnitude bin (3 mag × 3 flags)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import xgboost as xgb

import importlib
_mod = importlib.import_module("69_morph_metrics_xgb")
CFG          = _mod.CFG
METRIC_NAMES = _mod.METRIC_NAMES

from feature_schema import get_feature_cols


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR    = CFG.base_dir
MODELS_DIR  = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "models"
MAG_CACHE   = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "mag_cache"
SCALER_PATH = CFG.gaia_scaler_npz
PLOTS_DIR   = BASE_DIR / "plots" / "ml_runs" / "morph_metrics_16d"

# Gaia flag feature names and known "bad" thresholds
FLAG_SPECS = [
    ("feat_ruwe",                    "RUWE",                      1.4),
    ("feat_ipd_frac_multi_peak",     "ipd_frac_multi_peak (%)",   10.0),
    ("feat_astrometric_excess_noise","astrometric_excess_noise",   1.0),
]

GINI_IDX = METRIC_NAMES.index("gini")


# ======================================================================
# HELPERS
# ======================================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def load_cache() -> tuple:
    """Load X, Y, splits, mag from mag_cache (built by script 75)."""
    keys = ["X", "Y", "splits", "mag"]
    missing = [k for k in keys if not (MAG_CACHE / f"{k}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"Cache missing: {missing} — run script 75 first."
        )
    return tuple(np.load(MAG_CACHE / f"{k}.npy") for k in keys)


def unscale_features(
    X_scaled: np.ndarray, feature_cols: List[str]
) -> dict[str, np.ndarray]:
    """Un-scale selected Gaia flag columns from IQR-scaled X."""
    d = np.load(SCALER_PATH, allow_pickle=True)
    names   = [str(x) for x in d["feature_names"].tolist()]
    y_min   = d["y_min"].astype(np.float64)
    y_iqr   = d["y_iqr"].astype(np.float64)
    name_to_scaler = {n: (y_min[i], y_iqr[i]) for i, n in enumerate(names)}

    raw = {}
    for feat_name, label, _ in FLAG_SPECS:
        if feat_name not in feature_cols:
            continue
        col_idx = feature_cols.index(feat_name)
        mn, iqr = name_to_scaler[feat_name]
        raw[feat_name] = X_scaled[:, col_idx] * iqr + mn
    return raw


def tertile_labels(values: np.ndarray) -> tuple[np.ndarray, List[str]]:
    """Bin values into three equal-population tertiles. Returns (bin_idx, labels)."""
    p33 = np.percentile(values, 33.33)
    p67 = np.percentile(values, 66.67)
    bins = np.zeros(len(values), dtype=int)
    bins[values > p33] = 1
    bins[values > p67] = 2
    labels = [
        f"Low  (≤{p33:.3f})",
        f"Mid  ({p33:.3f}–{p67:.3f})",
        f"High (>{p67:.3f})",
    ]
    return bins, labels


# ======================================================================
# PLOT — violin grid
# ======================================================================

COLORS = ["#4393c3", "#f4a582", "#d6604d"]   # low / mid / high gini


def _violin_panel(
    ax: plt.Axes,
    data_by_bin: List[np.ndarray],
    bin_labels: List[str],
    flag_label: str,
    threshold: float,
    gini_bin_labels: List[str],
) -> None:
    # Clip extreme outliers for display (p99.5)
    all_vals = np.concatenate(data_by_bin)
    clip_hi = np.percentile(all_vals[np.isfinite(all_vals)], 99.5)

    parts = ax.violinplot(
        [np.clip(d[np.isfinite(d)], 0, clip_hi) for d in data_by_bin],
        positions=[0, 1, 2],
        showmedians=True,
        showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.axhline(threshold, color="red", lw=1.2, ls="--", alpha=0.8,
               label=f"threshold = {threshold}")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f"Low\n(n={len(d)})" for d in data_by_bin], fontsize=8)
    ax.set_ylabel(flag_label, fontsize=8)
    ax.set_xlabel("Predicted gini bin", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    # Annotate medians
    for i, d in enumerate(data_by_bin):
        med = np.nanmedian(d)
        ax.text(i, clip_hi * 0.97, f"med={med:.2f}",
                ha="center", va="top", fontsize=7, color="black")


def plot_flag_separation(
    raw_flags: dict,
    gini_pred: np.ndarray,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    gini_bins, gini_labels = tertile_labels(gini_pred)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (feat_name, flag_label, threshold) in zip(axes, FLAG_SPECS):
        flag_vals = raw_flags[feat_name]
        data_by_bin = [flag_vals[gini_bins == b] for b in range(3)]
        _violin_panel(ax, data_by_bin, gini_labels, flag_label, threshold,
                      gini_labels)

    fig.suptitle(
        f"Gaia quality flags by predicted gini bin{title_suffix}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, out_path)


def plot_flag_separation_by_mag(
    raw_flags: dict,
    gini_pred: np.ndarray,
    mag: np.ndarray,
    out_path: Path,
) -> None:
    # Magnitude tertiles
    mp33 = np.percentile(mag, 33.33)
    mp67 = np.percentile(mag, 66.67)
    mag_masks = [mag <= mp33, (mag > mp33) & (mag <= mp67), mag > mp67]
    mag_lo, mag_hi = mag.min(), mag.max()
    mag_labels = [
        f"Bright {mag_lo:.1f}–{mp33:.1f}",
        f"Mid {mp33:.1f}–{mp67:.1f}",
        f"Faint {mp67:.1f}–{mag_hi:.1f}",
    ]

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 13))

    for row, (mmask, mlabel) in enumerate(zip(mag_masks, mag_labels)):
        gp = gini_pred[mmask]
        gini_bins, gini_labels = tertile_labels(gp)

        for col, (feat_name, flag_label, threshold) in enumerate(FLAG_SPECS):
            ax = axes[row, col]
            flag_vals = raw_flags[feat_name][mmask]
            data_by_bin = [flag_vals[gini_bins == b] for b in range(3)]
            _violin_panel(ax, data_by_bin, gini_labels, flag_label, threshold,
                          gini_labels)
            if row == 0:
                ax.set_title(flag_label, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{mlabel}\n{flag_label}", fontsize=8)

    fig.suptitle(
        "Gaia quality flags by predicted gini bin — split by magnitude",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    feature_cols = get_feature_cols(CFG.feature_set)

    # ---- Load data ----
    print("Loading cached data ...")
    X, Y, splits, mag = load_cache()
    X_test   = X[splits == 2]
    mag_test = mag[splits == 2]
    print(f"Test set: {len(X_test)} objects")

    # ---- Un-scale Gaia flag columns ----
    raw_flags = unscale_features(X_test, feature_cols)
    print(f"Un-scaled flags: {list(raw_flags.keys())}")

    # ---- Load gini booster and predict ----
    print("Predicting gini ...")
    fnames = [f"f{i}" for i in range(len(feature_cols))]
    b = xgb.Booster()
    b.load_model(str(MODELS_DIR / "booster_gini.json"))
    b.feature_names = fnames
    gini_pred = b.predict(xgb.DMatrix(X_test, feature_names=fnames))
    print(f"  gini pred range: [{gini_pred.min():.4f}, {gini_pred.max():.4f}]")

    # ---- Plot overall ----
    print("\nPlot 18: Overall flag separation ...")
    plot_flag_separation(
        raw_flags, gini_pred,
        PLOTS_DIR / "18_flag_separation_overall.png",
    )

    # ---- Plot by magnitude ----
    print("Plot 19: Flag separation by magnitude bin ...")
    plot_flag_separation_by_mag(
        raw_flags, gini_pred, mag_test,
        PLOTS_DIR / "19_flag_separation_by_mag.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
