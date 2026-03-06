"""
74_val_test_closure.py
======================

Closure test: compare R² on the validation set vs the test set for all 11
morphology regressors. The validation set was used for early-stopping signal
during training; the test set was completely unseen. If val R² ≈ test R² the
model generalises and there is no early-stopping data leakage.

Loads cached arrays from morph_sweep/arrays/ (no recomputation needed).

Outputs:
  plots/ml_runs/morph_metrics_16d/
    11_val_test_closure_bar.png   — grouped bar chart val vs test R²
    12_val_test_closure_scatter.png — scatter val R² vs test R² per metric
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score

import importlib
_mod = importlib.import_module("69_morph_metrics_xgb")
CFG          = _mod.CFG
METRIC_NAMES = _mod.METRIC_NAMES

from feature_schema import get_feature_cols


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR   = CFG.base_dir
ARRAY_DIR  = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "mag_cache"
MODELS_DIR = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "models"
PLOTS_DIR  = BASE_DIR / "plots" / "ml_runs" / "morph_metrics_16d"


# ======================================================================
# HELPERS
# ======================================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def load_cached_split(split_code: int, kind: str) -> np.ndarray:
    """Load X or Y for a given split from the mag_cache (built by script 75)."""
    px = ARRAY_DIR / "X.npy"
    py = ARRAY_DIR / "Y.npy"
    ps = ARRAY_DIR / "splits.npy"
    if not (px.exists() and py.exists() and ps.exists()):
        raise FileNotFoundError(
            f"Cache missing in {ARRAY_DIR} — run script 75 first to build it."
        )
    splits = np.load(ps)
    mask = splits == split_code
    arr = np.load(px if kind == "X" else py)
    return arr[mask]


def load_boosters(feature_cols: List[str]) -> List[xgb.Booster]:
    fnames = [f"f{i}" for i in range(len(feature_cols))]
    boosters = []
    for name in METRIC_NAMES:
        b = xgb.Booster()
        b.load_model(str(MODELS_DIR / f"booster_{name}.json"))
        b.feature_names = fnames
        boosters.append(b)
    return boosters


def evaluate_r2(
    boosters: List[xgb.Booster],
    X: np.ndarray,
    Y: np.ndarray,
    feature_cols: List[str],
) -> dict[str, float]:
    fnames = [f"f{i}" for i in range(len(feature_cols))]
    results = {}
    for i, (b, name) in enumerate(zip(boosters, METRIC_NAMES)):
        y_true = Y[:, i]
        ok = np.isfinite(y_true)
        if not np.any(ok):
            results[name] = np.nan
            continue
        y_pred = b.predict(xgb.DMatrix(X[ok], feature_names=fnames))
        results[name] = float(r2_score(y_true[ok], y_pred))
    return results


# ======================================================================
# PLOTS
# ======================================================================

def plot_grouped_bar(
    r2_val: dict, r2_test: dict, out_path: Path
) -> None:
    metrics = METRIC_NAMES
    n = len(metrics)
    x = np.arange(n)
    w = 0.35

    val_vals  = [r2_val[m]  for m in metrics]
    test_vals = [r2_test[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(16, 5))
    b1 = ax.bar(x - w / 2, val_vals,  w, label="Validation", color="steelblue",  alpha=0.85)
    b2 = ax.bar(x + w / 2, test_vals, w, label="Test",       color="darkorange", alpha=0.85)

    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=6.5, rotation=90)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_ylim(min(0, min(val_vals + test_vals) - 0.05),
                max(val_vals + test_vals) * 1.22)
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(fontsize=10)
    ax.set_title(
        "Closure test: Validation vs Test R²  (val used for early-stopping only)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_path)


def plot_scatter_closure(
    r2_val: dict, r2_test: dict, out_path: Path
) -> None:
    vals  = np.array([r2_val[m]  for m in METRIC_NAMES])
    tests = np.array([r2_test[m] for m in METRIC_NAMES])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(vals, tests, s=60, zorder=3, color="steelblue")

    for name, v, t in zip(METRIC_NAMES, vals, tests):
        ax.annotate(name, (v, t), fontsize=7,
                    xytext=(4, 2), textcoords="offset points")

    lo = min(vals.min(), tests.min()) - 0.02
    hi = max(vals.max(), tests.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, alpha=0.7, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Validation R²", fontsize=10)
    ax.set_ylabel("Test R²", fontsize=10)
    ax.set_title("Closure: val R² vs test R² per metric", fontsize=11)
    ax.legend(fontsize=8)

    # Pearson r between val and test R² vectors
    r = float(np.corrcoef(vals, tests)[0, 1])
    ax.text(0.05, 0.92, f"Pearson r = {r:.3f}", transform=ax.transAxes,
            fontsize=9, color="dimgrey")

    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    feature_cols = get_feature_cols(CFG.feature_set)

    print("Loading cached arrays ...")
    X_val  = load_cached_split(1, "X")
    Y_val  = load_cached_split(1, "Y")
    X_test = load_cached_split(2, "X")
    Y_test = load_cached_split(2, "Y")
    print(f"  val={len(X_val)}  test={len(X_test)}")

    print("Loading boosters ...")
    boosters = load_boosters(feature_cols)

    print("Evaluating on val ...")
    r2_val = evaluate_r2(boosters, X_val, Y_val, feature_cols)

    print("Evaluating on test ...")
    r2_test = evaluate_r2(boosters, X_test, Y_test, feature_cols)

    # Summary table
    df = pd.DataFrame({"val": r2_val, "test": r2_test})
    df["delta"] = df["test"] - df["val"]
    df.index.name = "metric"
    print("\n=== R² CLOSURE SUMMARY ===")
    print(df.round(4).to_string())
    print(f"\nMean val R²  = {df['val'].mean():.4f}")
    print(f"Mean test R² = {df['test'].mean():.4f}")
    print(f"Mean |delta| = {df['delta'].abs().mean():.4f}")

    print("\nGenerating plots ...")
    plot_grouped_bar(r2_val, r2_test, PLOTS_DIR / "11_val_test_closure_bar.png")
    plot_scatter_closure(r2_val, r2_test, PLOTS_DIR / "12_val_test_closure_scatter.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
