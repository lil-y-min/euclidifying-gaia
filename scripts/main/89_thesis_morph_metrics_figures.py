#!/usr/bin/env python3
"""
89_thesis_morph_metrics_figures.py
====================================

Generate thesis-quality P(pred|true) and P(true|pred) density scatter
figures for the 11 morphology regressors (Figure 4.1 and companion).

Reuses data loading + booster infrastructure from scripts 69/70.

Outputs (saved directly to thesis Figs directory):
  Chapter4/Figs/Raster/09_morph_metrics_scatter.png
      Column-normalized density: P(pred | true)
  Chapter4/Figs/Raster/09b_morph_metrics_scatter_row_norm.png
      Row-normalized density:    P(true | pred)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[2]
THESIS   = BASE / "report" / "phd-thesis-template-2.4"
OUT_DIR  = THESIS / "Chapter4" / "Figs" / "Raster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── reuse infrastructure from scripts 69 / 70 ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
_s69 = importlib.import_module("69_morph_metrics_xgb")
_s70 = importlib.import_module("70_morph_metrics_diagnostics")

CFG          = _s69.CFG
METRIC_NAMES = _s69.METRIC_NAMES
load_all_data = _s69.load_all_data

from feature_schema import get_feature_cols

# ── typography ────────────────────────────────────────────────────────────────
FS_TITLE = 20   # panel titles
FS_LABEL = 18   # axis labels
FS_TICK  = 16   # tick labels
FS_CBAR  = 15   # colorbar label
FS_SUP   = 20   # figure suptitle


# ── helpers ───────────────────────────────────────────────────────────────────

def load_boosters(models_dir: Path, n_features: int) -> list[xgb.Booster]:
    boosters = []
    for name in METRIC_NAMES:
        b = xgb.Booster()
        b.load_model(str(models_dir / f"booster_{name}.json"))
        b.feature_names = [f"f{i}" for i in range(n_features)]
        boosters.append(b)
    return boosters


def get_predictions(boosters: list[xgb.Booster], X: np.ndarray) -> np.ndarray:
    dm = xgb.DMatrix(X, feature_names=[f"f{i}" for i in range(X.shape[1])])
    return np.column_stack([b.predict(dm) for b in boosters])


def _density_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mode: str,
    n_bins: int = 60,
) -> None:
    """
    mode='col': P(pred|true)  — column-normalize each true bin
    mode='row': P(true|pred)  — row-normalize   each pred bin
    """
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]

    lo = min(np.percentile(yt, 1), np.percentile(yp, 1))
    hi = max(np.percentile(yt, 99), np.percentile(yp, 99))

    H, _, _ = np.histogram2d(yt, yp, bins=n_bins, range=[[lo, hi], [lo, hi]])

    if mode == "col":
        s = H.sum(axis=1, keepdims=True); s[s == 0] = 1
        H = H / s
        cbar_label = "log(1 + P(pred | true))"
    else:
        s = H.sum(axis=0, keepdims=True); s[s == 0] = 1
        H = H / s
        cbar_label = "log(1 + P(true | pred))"

    H = np.log1p(H)
    vmax = np.percentile(H[H > 0], 99) if np.any(H > 0) else H.max()

    im = ax.imshow(
        H.T, origin="lower", aspect="auto",
        extent=[lo, hi, lo, hi], cmap="viridis",
        interpolation="nearest", vmin=0, vmax=vmax,
    )
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046, shrink=0.85)
    cb.set_label(cbar_label, fontsize=FS_CBAR)
    cb.ax.tick_params(labelsize=FS_CBAR)

    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, alpha=0.7)
    ax.set_xlabel("True", fontsize=FS_LABEL)
    ax.set_ylabel("Predicted", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)


def make_figure(
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    mode: str,
    out_path: Path,
) -> None:
    ncols = 4
    nrows = (len(METRIC_NAMES) + ncols - 1) // ncols   # 3 rows for 11 metrics
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5.0 * nrows))
    axes = axes.flatten()

    for i, name in enumerate(METRIC_NAMES):
        _density_panel(axes[i], Y_test[:, i], Y_pred[:, i], mode=mode)
        axes[i].set_title(name, fontsize=FS_TITLE, fontweight="bold")

    for j in range(len(METRIC_NAMES), len(axes)):
        axes[j].set_visible(False)

    suptitles = {
        "col": "Morphology regressors — column-normalised  P(pred | true)  (test set)",
        "row": "Morphology regressors — row-normalised  P(true | pred)  (test set)",
    }
    fig.suptitle(suptitles[mode], fontsize=FS_SUP, y=1.01)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    feature_cols = get_feature_cols(CFG.feature_set)
    models_dir   = CFG.output_root / "models"

    print("Loading boosters …")
    boosters = load_boosters(models_dir, len(feature_cols))

    print("Loading test data …")
    X, Y, splits, _ = load_all_data(CFG, feature_cols)
    X_test = X[splits == 2]
    Y_test = Y[splits == 2]
    print(f"  Test set: {len(X_test):,} rows")

    print("Computing predictions …")
    Y_pred = get_predictions(boosters, X_test)

    print("\nFigure 1: P(pred | true) …")
    make_figure(
        Y_test, Y_pred, mode="col",
        out_path=OUT_DIR / "09_morph_metrics_scatter.png",
    )

    print("Figure 2: P(true | pred) …")
    make_figure(
        Y_test, Y_pred, mode="row",
        out_path=OUT_DIR / "09b_morph_metrics_scatter_row_norm.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
