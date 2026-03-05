"""
73_morph_lr_huber_sweep.py
==========================

Tests two targeted fixes for the morphology regressors:
  1. Lower learning rate (lr=0.005, NUM_BOOST_ROUND=6000)
  2. Per-metric tuned Huber loss (huber_slope = std(y_train) per metric)
  3. Combined: both together

Configs:
  baseline       — reg:squarederror,      lr=0.05,  huber_slope=N/A
  lr_low         — reg:squarederror,      lr=0.005  (more rounds)
  huber_tuned    — reg:pseudohubererror,  lr=0.05,  huber_slope=std(y_train)
  huber_lr_low   — reg:pseudohubererror,  lr=0.005, huber_slope=std(y_train)

Loads cached arrays from morph_sweep/arrays/ (no metric recomputation).

Outputs:
  output/ml_runs/morph_lr_huber/
    <config>/models/booster_<metric>.json
    sweep_results.csv
  plots/ml_runs/morph_lr_huber/
    01_delta_r2_heatmap.png
    02_mean_r2_bar.png
    03_r2_per_metric.png
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

from feature_schema import get_feature_cols


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR   = CFG_BASE.base_dir
ARRAY_DIR  = BASE_DIR / "output" / "ml_runs" / "morph_sweep" / "arrays"
OUTPUT_DIR = BASE_DIR / "output" / "ml_runs" / "morph_lr_huber"
PLOTS_DIR  = BASE_DIR / "plots"  / "ml_runs" / "morph_lr_huber"


# ======================================================================
# FIXED PARAMS (shared across all configs)
# ======================================================================

FIXED_PARAMS = {
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.0,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
    "seed":             123,
    "eval_metric":      "rmse",
}

EARLY_STOPPING_ROUND = 150


# ======================================================================
# DATA — load from cache
# ======================================================================

def load_cached_arrays() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray
]:
    keys = ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"]
    arrays = []
    for k in keys:
        p = ARRAY_DIR / f"{k}.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Cache missing: {p}\n"
                "Run 71_morph_metrics_sweep.py first to generate cached arrays."
            )
        arrays.append(np.load(p))
    return tuple(arrays)


# ======================================================================
# TRAINING
# ======================================================================

def train_one_config(
    config_name: str,
    base_params: Dict,
    num_boost_round: int,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    feature_cols: List[str],
    huber_tuned: bool = False,
) -> List[xgb.Booster]:
    """
    Train 11 boosters for one config.
    If huber_tuned=True, set huber_slope=std(y_train_i) per metric.
    """
    models_dir = OUTPUT_DIR / config_name / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    fnames = [f"f{i}" for i in range(len(feature_cols))]
    boosters = []

    for i, metric in enumerate(METRIC_NAMES):
        y_tr = Y_train[:, i]
        y_vl = Y_val[:, i]
        ok_tr = np.isfinite(y_tr)
        ok_vl = np.isfinite(y_vl)

        params = dict(base_params)

        if huber_tuned:
            slope = float(np.std(y_tr[ok_tr]))
            slope = max(slope, 1e-2)   # guard against degenerate metrics
            params["huber_slope"] = slope

        dtrain = xgb.DMatrix(X_train[ok_tr], label=y_tr[ok_tr],
                             feature_names=fnames)
        dval   = xgb.DMatrix(X_val[ok_vl],   label=y_vl[ok_vl],
                             feature_names=fnames)

        t0 = time.time()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUND,
            verbose_eval=False,
        )
        elapsed = time.time() - t0
        slope_str = f"  slope={params.get('huber_slope', 'N/A'):.3g}" if huber_tuned else ""
        print(f"    [{i+1:2d}/11] {metric:<22s}  "
              f"best_round={booster.best_iteration:<5d}  {elapsed:.1f}s{slope_str}")

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
    fnames = [f"f{i}" for i in range(len(feature_cols))]
    results = {}
    for i, (booster, metric) in enumerate(zip(boosters, METRIC_NAMES)):
        y_true = Y_test[:, i]
        ok = np.isfinite(y_true)
        if not np.any(ok):
            results[metric] = np.nan
            continue
        y_pred = booster.predict(xgb.DMatrix(X_test[ok], feature_names=fnames))
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
    baseline = df.loc["baseline"]
    delta = df.drop("baseline").subtract(baseline)

    fig, ax = plt.subplots(figsize=(13, 3.5))
    vmax = max(abs(delta.values.max()), abs(delta.values.min()), 0.005)
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
    ax.set_title("R² per metric — lr & Huber sweep", fontsize=11)
    ax.legend(fontsize=8, ncol=n_configs)
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# SWEEP DEFINITION
# ======================================================================

# Each entry: (base_params, num_boost_round, huber_tuned)
CONFIGS: Dict[str, Tuple[Dict, int, bool]] = {
    "baseline": (
        {**FIXED_PARAMS, "objective": "reg:squarederror", "learning_rate": 0.05},
        3000, False,
    ),
    "lr_low": (
        {**FIXED_PARAMS, "objective": "reg:squarederror", "learning_rate": 0.005},
        6000, False,
    ),
    "huber_tuned": (
        {**FIXED_PARAMS, "objective": "reg:pseudohubererror", "learning_rate": 0.05},
        3000, True,
    ),
    "huber_lr_low": (
        {**FIXED_PARAMS, "objective": "reg:pseudohubererror", "learning_rate": 0.005},
        6000, True,
    ),
}


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    feature_cols = get_feature_cols(CFG_BASE.feature_set)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"LR & HUBER SWEEP — configs: {list(CONFIGS.keys())}")
    print(f"Metrics: {METRIC_NAMES}")
    print(f"Loading arrays from: {ARRAY_DIR}")
    print("=" * 70)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cached_arrays()
    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}\n")

    all_results: Dict[str, Dict[str, float]] = {}

    for cfg_name, (base_params, nbr, huber_tuned) in CONFIGS.items():
        print(f"\n{'='*60}")
        obj = base_params.get("objective", "?")
        lr  = base_params.get("learning_rate", "?")
        print(f"CONFIG: {cfg_name}  |  obj={obj}  lr={lr}  "
              f"rounds={nbr}  huber_tuned={huber_tuned}")
        print(f"{'='*60}")
        t0 = time.time()
        boosters = train_one_config(
            cfg_name, base_params, nbr,
            X_train, Y_train, X_val, Y_val,
            feature_cols, huber_tuned=huber_tuned,
        )
        r2s = evaluate_config(boosters, X_test, Y_test, feature_cols)
        all_results[cfg_name] = r2s
        mean_r2 = np.nanmean(list(r2s.values()))
        print(f"  mean R²={mean_r2:.4f}   total elapsed={time.time()-t0:.1f}s")
        for metric, r2 in r2s.items():
            print(f"    {metric:<22s}  R²={r2:.4f}")

    # ---- Results table ----
    df = pd.DataFrame(all_results).T
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
