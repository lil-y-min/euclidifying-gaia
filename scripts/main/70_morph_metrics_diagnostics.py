"""
70_morph_metrics_diagnostics.py
================================

Diagnostic plots for the 11-metric morphology regressors trained in script 69.

Plots generated:
  01_feature_importance.png       — gain importance per metric (11 panels)
  02_distribution_overlay.png     — true vs predicted distribution per metric
  03_corr_matrix_true.png         — Pearson correlation matrix of true metrics
  04_corr_matrix_predicted.png    — Pearson correlation matrix of predicted metrics

Requires: trained boosters from output/ml_runs/morph_metrics_16d/models/
Re-runs data loading to obtain X_test / Y_test.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xgboost as xgb

# Re-use everything from script 69
import importlib, sys
_mod = importlib.import_module("69_morph_metrics_xgb")
CFG = _mod.CFG
METRIC_NAMES = _mod.METRIC_NAMES
load_all_data = _mod.load_all_data

from feature_schema import get_feature_cols


# ======================================================================
# HELPERS
# ======================================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def load_boosters(models_dir: Path, feature_names: List[str]) -> List[xgb.Booster]:
    boosters = []
    for name in METRIC_NAMES:
        path = models_dir / f"booster_{name}.json"
        b = xgb.Booster()
        b.load_model(str(path))
        b.feature_names = [f"f{i}" for i in range(len(feature_names))]
        boosters.append(b)
    return boosters


def get_predictions(
    boosters: List[xgb.Booster], X_test: np.ndarray, feature_names: List[str]
) -> np.ndarray:
    fnames = [f"f{i}" for i in range(len(feature_names))]
    dtest = xgb.DMatrix(X_test, feature_names=fnames)
    return np.column_stack([b.predict(dtest) for b in boosters])


def short_feat_name(name: str) -> str:
    """Strip feat_ prefix and shorten for display."""
    return name.replace("feat_", "").replace("phot_", "").replace("astrometric_", "astro_")


# ======================================================================
# PLOT 1 — Feature importance
# ======================================================================

def plot_feature_importance(
    boosters: List[xgb.Booster],
    feature_names: List[str],
    out_path: Path,
) -> None:
    short_names = [short_feat_name(n) for n in feature_names]
    n_feat = len(feature_names)
    ncols = 4
    nrows = (len(METRIC_NAMES) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, (booster, metric) in enumerate(zip(boosters, METRIC_NAMES)):
        ax = axes[i]
        scores = booster.get_score(importance_type="gain")
        # Map f0, f1, ... to feature names
        vals = np.array([scores.get(f"f{j}", 0.0) for j in range(n_feat)])
        # Normalize to sum to 1
        total = vals.sum()
        if total > 0:
            vals = vals / total
        order = np.argsort(vals)[::-1]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_feat))[::-1]
        ax.barh(
            range(n_feat),
            vals[order],
            color=colors,
            edgecolor="none",
        )
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels([short_names[j] for j in order], fontsize=7)
        ax.set_title(metric, fontsize=9, fontweight="bold")
        ax.set_xlabel("Gain (normalized)", fontsize=7)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=7)

    for j in range(len(METRIC_NAMES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Feature importance (gain) per morphology regressor", fontsize=13, y=1.01
    )
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# PLOT 2 — Distribution overlay (true vs predicted)
# ======================================================================

def plot_distribution_overlay(
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    out_path: Path,
) -> None:
    ncols = 4
    nrows = (len(METRIC_NAMES) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, name in enumerate(METRIC_NAMES):
        ax = axes[i]
        y_t = Y_test[:, i]
        y_p = Y_pred[:, i]
        ok = np.isfinite(y_t) & np.isfinite(y_p)
        y_t, y_p = y_t[ok], y_p[ok]

        lo = min(np.percentile(y_t, 1), np.percentile(y_p, 1))
        hi = max(np.percentile(y_t, 99), np.percentile(y_p, 99))
        bins = np.linspace(lo, hi, 60)

        ax.hist(y_t, bins=bins, density=True, alpha=0.5, color="steelblue",
                label="True", edgecolor="none")
        ax.hist(y_p, bins=bins, density=True, alpha=0.5, color="tomato",
                label="Predicted", edgecolor="none")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(METRIC_NAMES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "True vs Predicted distributions (test set)", fontsize=13, y=1.01
    )
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# PLOTS 3 & 4 — Correlation matrices
# ======================================================================

def plot_corr_matrix(
    Y: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    # Drop rows with any NaN
    ok = np.all(np.isfinite(Y), axis=1)
    Y_clean = Y[ok]
    C = np.corrcoef(Y_clean.T)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(len(METRIC_NAMES)))
    ax.set_yticks(range(len(METRIC_NAMES)))
    ax.set_xticklabels(METRIC_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(METRIC_NAMES, fontsize=8)

    # Annotate cells
    for row in range(len(METRIC_NAMES)):
        for col in range(len(METRIC_NAMES)):
            val = C[row, col]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title(title, fontsize=12, pad=12)
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# PLOT — Density scatter (raw + column-norm + row-norm)
# ======================================================================

def _density_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    mode: str,   # "raw" | "col" | "row"
    n_bins: int = 60,
) -> None:
    """
    Draw one 2D density panel.
      mode="raw"  → log-scaled counts
      mode="col"  → column-normalize (P(pred|true)): normalize each x-bin
      mode="row"  → row-normalize    (P(true|pred)): normalize each y-bin
    """
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]

    # Use 1st–99th percentile range to avoid extreme outliers dominating
    lo_x, hi_x = np.percentile(yt, 1), np.percentile(yt, 99)
    lo_y, hi_y = np.percentile(yp, 1), np.percentile(yp, 99)
    # Expand to common range so 1:1 line is meaningful
    lo = min(lo_x, lo_y)
    hi = max(hi_x, hi_y)

    H, xedges, yedges = np.histogram2d(yt, yp, bins=n_bins,
                                        range=[[lo, hi], [lo, hi]])
    # H shape: (n_bins_x, n_bins_y) — x=true, y=pred

    if mode == "col":
        col_sum = H.sum(axis=1, keepdims=True)
        col_sum[col_sum == 0] = 1
        H = H / col_sum
        cmap, label = "viridis", "P(pred | true)"
    elif mode == "row":
        row_sum = H.sum(axis=0, keepdims=True)
        row_sum[row_sum == 0] = 1
        H = H / row_sum
        cmap, label = "viridis", "P(true | pred)"
    else:
        cmap, label = "viridis", "log(1+count)"

    # Apply log1p to compress dynamic range for all modes
    H = np.log1p(H)

    # imshow: H is (x_bins, y_bins) but imshow plots (row=y, col=x)
    vmax = np.percentile(H[H > 0], 95) if np.any(H > 0) else H.max()
    im = ax.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[lo, hi, lo, hi],
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label=f"log(1+{label})", pad=0.02,
                 fraction=0.046, shrink=0.8)
    ax.plot([lo, hi], [lo, hi], "r--", lw=0.8, alpha=0.7)
    ax.set_xlabel("True", fontsize=7)
    ax.set_ylabel("Predicted", fontsize=7)
    ax.tick_params(labelsize=6)


def plot_density_scatter(
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    mode: str,
    out_path: Path,
) -> None:
    titles = {
        "raw": "2D density (log scale)",
        "col": "Column-normalized  P(pred | true)",
        "row": "Row-normalized  P(true | pred)",
    }
    ncols = 4
    nrows = (len(METRIC_NAMES) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, name in enumerate(METRIC_NAMES):
        ax = axes[i]
        _density_panel(ax, Y_test[:, i], Y_pred[:, i], name, mode=mode)
        ax.set_title(name, fontsize=9, fontweight="bold")

    for j in range(len(METRIC_NAMES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Morphology regressors — {titles[mode]} (test set)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    cfg = CFG
    feature_cols = get_feature_cols(cfg.feature_set)
    models_dir = cfg.output_root / "models"
    plots_root = cfg.plots_root

    print("Loading boosters ...")
    boosters = load_boosters(models_dir, feature_cols)

    print("Re-loading data (needed for test arrays) ...")
    X, Y, splits, _ = load_all_data(cfg, feature_cols)
    X_test = X[splits == 2]
    Y_test = Y[splits == 2]
    print(f"Test set: {len(X_test)} rows")

    print("Computing predictions ...")
    Y_pred = get_predictions(boosters, X_test, feature_cols)

    # ---- Plot 1: Feature importance ----
    print("\nPlot 1: Feature importance ...")
    plot_feature_importance(
        boosters, feature_cols,
        plots_root / "04_feature_importance.png",
    )

    # ---- Plot 2: Distribution overlay ----
    print("Plot 2: Distribution overlay ...")
    plot_distribution_overlay(
        Y_test, Y_pred,
        plots_root / "05_distribution_overlay.png",
    )

    # ---- Plot 3: Correlation matrix (true) ----
    print("Plot 3: Correlation matrix (true) ...")
    plot_corr_matrix(
        Y_test,
        title="Pearson correlation matrix — True metrics (test set)",
        out_path=plots_root / "06_corr_matrix_true.png",
    )

    # ---- Plot 4: Correlation matrix (predicted) ----
    print("Plot 4: Correlation matrix (predicted) ...")
    plot_corr_matrix(
        Y_pred,
        title="Pearson correlation matrix — Predicted metrics (test set)",
        out_path=plots_root / "07_corr_matrix_predicted.png",
    )

    # ---- Density scatter plots ----
    print("Plot 5: Density scatter (raw) ...")
    plot_density_scatter(Y_test, Y_pred, mode="raw",
                         out_path=plots_root / "08_density_scatter_raw.png")

    print("Plot 6: Density scatter (column-normalized, P(pred|true)) ...")
    plot_density_scatter(Y_test, Y_pred, mode="col",
                         out_path=plots_root / "09_density_scatter_col_norm.png")

    print("Plot 7: Density scatter (row-normalized, P(true|pred)) ...")
    plot_density_scatter(Y_test, Y_pred, mode="row",
                         out_path=plots_root / "10_density_scatter_row_norm.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
