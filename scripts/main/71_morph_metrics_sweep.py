"""
71_morph_metrics_sweep.py
=========================

Ablation sweep over XGBoost hyperparameters for the 11-metric morphology
regressors. Each config changes one parameter from the baseline to isolate
its contribution, plus a combined config with all best settings.

Configs:
  baseline     — reg:squarederror, depth=6, lr=0.05, alpha=0.0
  huber        — reg:pseudohubererror, depth=6, lr=0.05, alpha=0.0
  depth8       — reg:squarederror,    depth=8, lr=0.05, alpha=0.0
  lr001        — reg:squarederror,    depth=6, lr=0.01, alpha=0.0
  alpha01      — reg:squarederror,    depth=6, lr=0.05, alpha=0.1
  all_combined — reg:pseudohubererror, depth=8, lr=0.01, alpha=0.1

Data arrays are cached to disk after the first load to avoid recomputing
morphology metrics on every run.

Outputs:
  output/ml_runs/morph_sweep/
    arrays/          — cached X/Y arrays (written once)
    <config_name>/   — boosters per config
    sweep_results.csv
  plots/ml_runs/morph_sweep/
    01_delta_r2_heatmap.png   — ΔR² vs baseline per config × metric
    02_mean_r2_bar.png        — mean R² across metrics per config
    03_r2_per_metric.png      — full R² table as grouped bar chart
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score

import importlib
_mod = importlib.import_module("69_morph_metrics_xgb")
CFG_BASE      = _mod.CFG
METRIC_NAMES  = _mod.METRIC_NAMES
load_all_data = _mod.load_all_data

from feature_schema import get_feature_cols


# ======================================================================
# SWEEP DEFINITION
# ======================================================================

SWEEP_CONFIGS: Dict[str, Dict] = {
    "baseline": {
        "objective":     "reg:squarederror",
        "max_depth":     6,
        "learning_rate": 0.05,
        "reg_alpha":     0.0,
    },
    "huber": {
        "objective":     "reg:pseudohubererror",
        "max_depth":     6,
        "learning_rate": 0.05,
        "reg_alpha":     0.0,
    },
    "depth8": {
        "objective":     "reg:squarederror",
        "max_depth":     8,
        "learning_rate": 0.05,
        "reg_alpha":     0.0,
    },
    "lr001": {
        "objective":     "reg:squarederror",
        "max_depth":     6,
        "learning_rate": 0.01,
        "reg_alpha":     0.0,
    },
    "alpha01": {
        "objective":     "reg:squarederror",
        "max_depth":     6,
        "learning_rate": 0.05,
        "reg_alpha":     0.1,
    },
    "all_combined": {
        "objective":     "reg:pseudohubererror",
        "max_depth":     8,
        "learning_rate": 0.01,
        "reg_alpha":     0.1,
    },
}

# Shared fixed params (same as existing baseline)
FIXED_PARAMS = {
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
    "seed":             123,
    "eval_metric":      "rmse",
}

NUM_BOOST_ROUND      = 3000   # higher ceiling gives lr=0.01 room to converge
EARLY_STOPPING_ROUND = 100

BASE_DIR   = CFG_BASE.base_dir
OUTPUT_DIR = BASE_DIR / "output" / "ml_runs" / "morph_sweep"
PLOTS_DIR  = BASE_DIR / "plots"  / "ml_runs" / "morph_sweep"
ARRAY_DIR  = OUTPUT_DIR / "arrays"


# ======================================================================
# DATA — load once, cache to disk
# ======================================================================

def load_or_cache_arrays(feature_cols: List[str]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray
]:
    """Return X_train, Y_train, X_val, Y_val, X_test, Y_test.
    Computes metrics and caches to .npy files on first call.
    """
    ARRAY_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "X_train": ARRAY_DIR / "X_train.npy",
        "Y_train": ARRAY_DIR / "Y_train.npy",
        "X_val":   ARRAY_DIR / "X_val.npy",
        "Y_val":   ARRAY_DIR / "Y_val.npy",
        "X_test":  ARRAY_DIR / "X_test.npy",
        "Y_test":  ARRAY_DIR / "Y_test.npy",
    }
    if all(f.exists() for f in files.values()):
        print("Loading cached arrays from disk ...")
        return tuple(np.load(files[k]) for k in
                     ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"])

    print("Cache not found — computing metrics (this takes a few minutes) ...")
    X, Y, splits, _ = load_all_data(CFG_BASE, feature_cols)
    arrs = {
        "X_train": X[splits == 0], "Y_train": Y[splits == 0],
        "X_val":   X[splits == 1], "Y_val":   Y[splits == 1],
        "X_test":  X[splits == 2], "Y_test":  Y[splits == 2],
    }
    for k, arr in arrs.items():
        np.save(files[k], arr)
        print(f"  Saved {files[k].name}  shape={arr.shape}")
    return tuple(arrs[k] for k in
                 ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"])


# ======================================================================
# TRAINING
# ======================================================================

def train_one_config(
    config_name: str,
    sweep_params: Dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    feature_cols: List[str],
) -> List[xgb.Booster]:
    models_dir = OUTPUT_DIR / config_name / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    fnames = [f"f{i}" for i in range(len(feature_cols))]
    params = {**FIXED_PARAMS, **sweep_params}

    boosters = []
    for i, metric in enumerate(METRIC_NAMES):
        y_tr = Y_train[:, i]
        y_vl = Y_val[:, i]
        ok_tr = np.isfinite(y_tr)
        ok_vl = np.isfinite(y_vl)
        dtrain = xgb.DMatrix(X_train[ok_tr], label=y_tr[ok_tr],
                             feature_names=fnames)
        dval   = xgb.DMatrix(X_val[ok_vl],   label=y_vl[ok_vl],
                             feature_names=fnames)
        t0 = time.time()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUND,
            verbose_eval=False,
        )
        elapsed = time.time() - t0
        print(f"    [{i+1:2d}/11] {metric:<20s}  "
              f"best_round={booster.best_iteration:<5d}  {elapsed:.1f}s")
        booster.save_model(str(models_dir / f"booster_{metric}.json"))
        boosters.append(booster)
    return boosters


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate_config(
    boosters: List[xgb.Booster],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    feature_cols: List[str],
) -> Dict[str, float]:
    """Returns dict metric -> R² on test set."""
    fnames = [f"f{i}" for i in range(len(feature_cols))]
    dtest  = xgb.DMatrix(X_test, feature_names=fnames)
    results = {}
    for i, (booster, metric) in enumerate(zip(boosters, METRIC_NAMES)):
        y_true = Y_test[:, i]
        ok = np.isfinite(y_true)
        if not np.any(ok):
            results[metric] = np.nan
            continue
        y_pred = booster.predict(xgb.DMatrix(
            X_test[ok], feature_names=fnames))
        results[metric] = float(r2_score(y_true[ok], y_pred))
    return results


# ======================================================================
# PLOTS
# ======================================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_delta_r2_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """ΔR² relative to baseline, configs × metrics."""
    baseline = df.loc["baseline"]
    delta = df.drop("baseline").subtract(baseline)

    fig, ax = plt.subplots(figsize=(13, 4))
    vmax = max(abs(delta.values.max()), abs(delta.values.min()), 0.01)
    im = ax.imshow(delta.values, aspect="auto", cmap="RdBu",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="ΔR²  (config − baseline)")

    ax.set_xticks(range(len(METRIC_NAMES)))
    ax.set_xticklabels(METRIC_NAMES, rotation=40, ha="right", fontsize=8)
    config_labels = list(delta.index)
    ax.set_yticks(range(len(config_labels)))
    ax.set_yticklabels(config_labels, fontsize=9)

    for row in range(len(config_labels)):
        for col in range(len(METRIC_NAMES)):
            val = delta.values[row, col]
            color = "white" if abs(val) > 0.5 * vmax else "black"
            ax.text(col, row, f"{val:+.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title("ΔR²  relative to baseline  (blue = better)", fontsize=11)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_mean_r2_bar(df: pd.DataFrame, out_path: Path) -> None:
    """Mean R² across all metrics per config."""
    means = df.mean(axis=1).sort_values(ascending=False)
    colors = ["steelblue" if c != "baseline" else "grey" for c in means.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(means.index, means.values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_ylabel("Mean R²  (across 11 metrics)")
    ax.set_title("Mean R² per sweep config", fontsize=11)
    ax.set_ylim(0, min(1.0, means.max() * 1.15))
    ax.axhline(means["baseline"], color="grey", lw=1, ls="--", alpha=0.6)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_r2_per_metric(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: R² per metric, one group per metric, bars = configs."""
    configs = list(df.index)
    n_configs = len(configs)
    n_metrics = len(METRIC_NAMES)
    x = np.arange(n_metrics)
    width = 0.8 / n_configs

    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(16, 5))
    for i, cfg in enumerate(configs):
        offset = (i - n_configs / 2 + 0.5) * width
        vals = df.loc[cfg].values
        ax.bar(x + offset, vals, width=width * 0.9,
               label=cfg, color=cmap(i / n_configs), edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("R² per metric per sweep config", fontsize=11)
    ax.legend(fontsize=8, ncol=n_configs)
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    feature_cols = get_feature_cols(CFG_BASE.feature_set)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"SWEEP: {list(SWEEP_CONFIGS.keys())}")
    print(f"Metrics: {METRIC_NAMES}")
    print(f"Arrays cache: {ARRAY_DIR}")
    print("=" * 70)

    # ---- Data ----
    X_train, Y_train, X_val, Y_val, X_test, Y_test = \
        load_or_cache_arrays(feature_cols)
    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}\n")

    # ---- Sweep ----
    all_results: Dict[str, Dict[str, float]] = {}

    for cfg_name, sweep_params in SWEEP_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"CONFIG: {cfg_name}  |  {sweep_params}")
        print(f"{'='*60}")
        t0 = time.time()
        boosters = train_one_config(
            cfg_name, sweep_params,
            X_train, Y_train, X_val, Y_val,
            feature_cols,
        )
        r2s = evaluate_config(boosters, X_test, Y_test, feature_cols)
        all_results[cfg_name] = r2s
        mean_r2 = np.nanmean(list(r2s.values()))
        print(f"  mean R²={mean_r2:.4f}   total elapsed={time.time()-t0:.1f}s")

    # ---- Results table ----
    df = pd.DataFrame(all_results).T          # rows=configs, cols=metrics
    df = df[METRIC_NAMES]
    df.index.name = "config"

    results_path = OUTPUT_DIR / "sweep_results.csv"
    df.to_csv(results_path)
    print(f"\nSaved: {results_path}")

    print("\n=== R² SUMMARY ===")
    print(df.round(4).to_string())
    print("\n=== MEAN R² PER CONFIG ===")
    print(df.mean(axis=1).round(4).sort_values(ascending=False).to_string())

    # ---- Plots ----
    print("\n--- Generating plots ---")
    plot_delta_r2_heatmap(df, PLOTS_DIR / "01_delta_r2_heatmap.png")
    plot_mean_r2_bar(df,      PLOTS_DIR / "02_mean_r2_bar.png")
    plot_r2_per_metric(df,    PLOTS_DIR / "03_r2_per_metric.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
