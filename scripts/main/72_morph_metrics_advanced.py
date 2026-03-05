"""
72_morph_metrics_advanced.py
============================

Advanced improvements for the 11-metric morphology regressors.
Tests three orthogonal strategies on top of the baseline XGBoost config:

  target_transform  — Yeo-Johnson transform per metric target before training,
                      invert predictions before evaluating (R² on original scale)
  sample_weights    — inverse histogram-density weights per metric to fight
                      regression-to-mean on rare/extreme sources
  feat_engineering  — expand 16D → 28D with cross-products and polynomials
                      of the most informative Gaia features
  all_advanced      — all three combined

Configs:
  baseline          — squarederror, 16D, no transform, no weights
  target_transform  — yeo-johnson targets, else baseline
  sample_weights    — inverse density weights, else baseline
  feat_engineering  — 28D features, else baseline
  all_advanced      — transform + weights + 28D features

Uses cached arrays from script 71 (output/ml_runs/morph_sweep/arrays/).
Outputs:
  output/ml_runs/morph_advanced/
    <config>/models/booster_<metric>.json
    advanced_results.csv
  plots/ml_runs/morph_advanced/
    01_delta_r2_heatmap.png
    02_mean_r2_bar.png
    03_r2_per_metric.png
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer

import importlib
_mod        = importlib.import_module("69_morph_metrics_xgb")
METRIC_NAMES = _mod.METRIC_NAMES
from feature_schema import get_feature_cols


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR     = _mod.CFG.base_dir
ARRAY_DIR    = BASE_DIR / "output" / "ml_runs" / "morph_sweep" / "arrays"
OUTPUT_DIR   = BASE_DIR / "output" / "ml_runs" / "morph_advanced"
PLOTS_DIR    = BASE_DIR / "plots"  / "ml_runs" / "morph_advanced"

FEATURE_COLS_16D = get_feature_cols("16D")   # 16 base features

# ======================================================================
# FIXED XGB PARAMS (same as baseline throughout)
# ======================================================================

BASE_XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_lambda":       1.0,
    "reg_alpha":        0.0,
    "tree_method":      "hist",
    "seed":             123,
}

NUM_BOOST_ROUND      = 3000
EARLY_STOPPING_ROUND = 100


# ======================================================================
# FEATURE ENGINEERING  (16D → 28D)
# ======================================================================
# Feature index map (after IQR scaling):
#   0  log10_snr          5  ipd_frac_multi_peak   10 ipd_gof_harmonic_phase
#   1  ruwe               6  c_star                11 ipd_frac_odd_win
#   2  astro_excess_noise 7  pm_significance       12 bp_n_contaminated
#   3  parallax_over_err  8  astro_excess_noise_sig 13 bp_n_blended
#   4  visibility_periods 9  ipd_gof_harmonic_amp  14 rp_n_contaminated
#                                                   15 rp_n_blended

NEW_FEATURE_NAMES = [
    "ruwe_sq",
    "ruwe_x_ipd_multi",
    "ruwe_x_c_star",
    "snr_x_ruwe",
    "harmonic_amp_x_ipd_multi",
    "ipd_multi_sq",
    "ipd_odd_x_ipd_multi",
    "c_star_sq",
    "astro_noise_x_ruwe",
    "total_contamination",
    "total_blended",
    "pm_sig_x_ruwe",
]


def engineer_features(X: np.ndarray) -> np.ndarray:
    """Expand 16D → 28D feature matrix."""
    new_cols = np.column_stack([
        X[:, 1] ** 2,                    # ruwe²
        X[:, 1] * X[:, 5],               # ruwe × ipd_frac_multi_peak
        X[:, 1] * X[:, 6],               # ruwe × c_star
        X[:, 0] * X[:, 1],               # log10_snr × ruwe
        X[:, 9] * X[:, 5],               # harmonic_amp × ipd_frac_multi_peak
        X[:, 5] ** 2,                    # ipd_frac_multi_peak²
        X[:, 11] * X[:, 5],              # ipd_frac_odd_win × ipd_frac_multi_peak
        X[:, 6] ** 2,                    # c_star²
        X[:, 2] * X[:, 1],               # astro_excess_noise × ruwe
        X[:, 12] + X[:, 14],             # total BP+RP contaminated transits
        X[:, 13] + X[:, 15],             # total BP+RP blended transits
        X[:, 7] * X[:, 1],               # pm_significance × ruwe
    ])
    return np.concatenate([X, new_cols], axis=1).astype(np.float32)


def make_feature_names(use_eng: bool) -> List[str]:
    if use_eng:
        return FEATURE_COLS_16D + NEW_FEATURE_NAMES
    return FEATURE_COLS_16D


# ======================================================================
# SAMPLE WEIGHTS  (inverse histogram density per metric)
# ======================================================================

def compute_sample_weights(
    y: np.ndarray,
    n_bins: int = 100,
    clip_pct: float = 99,
) -> np.ndarray:
    """
    Inverse frequency weights: rare values get higher weight.
    Uses histogram density (fast, O(n)) rather than KDE.
    Weights are normalized to mean=1 and clipped at clip_pct percentile.
    """
    ok = np.isfinite(y)
    w = np.ones(len(y), dtype=np.float32)
    if not np.any(ok):
        return w
    y_ok = y[ok]
    counts, bin_edges = np.histogram(y_ok, bins=n_bins)
    bin_idx = np.digitize(y_ok, bin_edges[:-1]) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    density = np.maximum(counts[bin_idx].astype(float), 1.0)
    w_ok = 1.0 / density
    w_ok /= w_ok.mean()
    clip_val = np.percentile(w_ok, clip_pct)
    w_ok = np.clip(w_ok, 0.0, clip_val)
    w[ok] = w_ok.astype(np.float32)
    return w


# ======================================================================
# TARGET TRANSFORM  (Yeo-Johnson per metric)
# ======================================================================

def fit_target_transformer(y_tr: np.ndarray) -> Optional[PowerTransformer]:
    ok = np.isfinite(y_tr)
    if ok.sum() < 10:
        return None
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(y_tr[ok].reshape(-1, 1))
    return pt


def apply_transform(pt: Optional[PowerTransformer], y: np.ndarray) -> np.ndarray:
    if pt is None:
        return y.copy()
    ok = np.isfinite(y)
    out = y.copy()
    if np.any(ok):
        out[ok] = pt.transform(y[ok].reshape(-1, 1)).flatten()
    return out


def invert_transform(pt: Optional[PowerTransformer], y: np.ndarray) -> np.ndarray:
    if pt is None:
        return y.copy()
    return pt.inverse_transform(y.reshape(-1, 1)).flatten()


# ======================================================================
# TRAINING
# ======================================================================

def train_one_config(
    config_name: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    feature_names: List[str],
    use_target_transform: bool = False,
    use_sample_weights: bool = False,
) -> Tuple[List[xgb.Booster], List[Optional[PowerTransformer]]]:
    models_dir = OUTPUT_DIR / config_name / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    fnames = [f"f{i}" for i in range(len(feature_names))]
    boosters, transformers = [], []

    for i, metric in enumerate(METRIC_NAMES):
        y_tr = Y_train[:, i]
        y_vl = Y_val[:, i]
        ok_tr = np.isfinite(y_tr)
        ok_vl = np.isfinite(y_vl)

        # Target transform
        pt = fit_target_transformer(y_tr[ok_tr]) if use_target_transform else None
        y_tr_fit = apply_transform(pt, y_tr[ok_tr]) if pt else y_tr[ok_tr]
        y_vl_fit = apply_transform(pt, y_vl[ok_vl]) if pt else y_vl[ok_vl]

        # Sample weights
        w_tr = compute_sample_weights(y_tr[ok_tr]) if use_sample_weights else None

        dtrain = xgb.DMatrix(X_train[ok_tr], label=y_tr_fit,
                             feature_names=fnames, weight=w_tr)
        dval   = xgb.DMatrix(X_val[ok_vl],   label=y_vl_fit,
                             feature_names=fnames)

        t0 = time.time()
        booster = xgb.train(
            BASE_XGB_PARAMS,
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
        transformers.append(pt)

    return boosters, transformers


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate_config(
    boosters: List[xgb.Booster],
    transformers: List[Optional[PowerTransformer]],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    fnames = [f"f{i}" for i in range(len(feature_names))]
    results = {}
    for i, (booster, pt, metric) in enumerate(zip(boosters, transformers, METRIC_NAMES)):
        y_true = Y_test[:, i]
        ok = np.isfinite(y_true)
        if not np.any(ok):
            results[metric] = np.nan
            continue
        dtest = xgb.DMatrix(X_test[ok], feature_names=fnames)
        y_pred_transformed = booster.predict(dtest)
        # Invert transform before computing R² (always on original scale)
        y_pred = invert_transform(pt, y_pred_transformed)
        # Filter NaN that can arise from inverse transform on extreme predictions
        finite = np.isfinite(y_pred)
        if not np.any(finite):
            results[metric] = np.nan
            continue
        results[metric] = float(r2_score(y_true[ok][finite], y_pred[finite]))
    return results


# ======================================================================
# PLOTS  (same as sweep script)
# ======================================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_delta_r2_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    baseline = df.loc["baseline"]
    delta = df.drop("baseline").subtract(baseline)
    vmax = max(abs(delta.values.max()), abs(delta.values.min()), 0.01)

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(delta.values, aspect="auto", cmap="RdBu",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="ΔR²  (config − baseline)")
    ax.set_xticks(range(len(METRIC_NAMES)))
    ax.set_xticklabels(METRIC_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(delta.index)))
    ax.set_yticklabels(list(delta.index), fontsize=9)
    for row in range(len(delta.index)):
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
    ax.set_title("Mean R² per advanced config", fontsize=11)
    ax.set_ylim(0, min(1.0, means.max() * 1.15))
    ax.axhline(means.get("baseline", 0), color="grey", lw=1, ls="--", alpha=0.6)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_r2_per_metric(df: pd.DataFrame, out_path: Path) -> None:
    configs = list(df.index)
    n_configs = len(configs)
    x = np.arange(len(METRIC_NAMES))
    width = 0.8 / n_configs
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(16, 5))
    for i, cfg in enumerate(configs):
        offset = (i - n_configs / 2 + 0.5) * width
        ax.bar(x + offset, df.loc[cfg].values, width=width * 0.9,
               label=cfg, color=cmap(i / n_configs), edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("R² per metric — advanced configs", fontsize=11)
    ax.legend(fontsize=8, ncol=n_configs)
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load cached arrays ----
    print("Loading cached arrays ...")
    X_train = np.load(ARRAY_DIR / "X_train.npy")
    Y_train = np.load(ARRAY_DIR / "Y_train.npy")
    X_val   = np.load(ARRAY_DIR / "X_val.npy")
    Y_val   = np.load(ARRAY_DIR / "Y_val.npy")
    X_test  = np.load(ARRAY_DIR / "X_test.npy")
    Y_test  = np.load(ARRAY_DIR / "Y_test.npy")
    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # Feature-engineered versions
    X_train_eng = engineer_features(X_train)
    X_val_eng   = engineer_features(X_val)
    X_test_eng  = engineer_features(X_test)
    feat_names_16d = FEATURE_COLS_16D
    feat_names_28d = make_feature_names(use_eng=True)
    print(f"Feature dims: 16D={X_train.shape[1]}  28D={X_train_eng.shape[1]}\n")

    # ---- Config definitions ----
    CONFIGS = {
        "baseline": dict(
            X_tr=X_train,     X_vl=X_val,     X_te=X_test,
            feat=feat_names_16d,
            use_target_transform=False,
            use_sample_weights=False,
        ),
        "target_transform": dict(
            X_tr=X_train,     X_vl=X_val,     X_te=X_test,
            feat=feat_names_16d,
            use_target_transform=True,
            use_sample_weights=False,
        ),
        "sample_weights": dict(
            X_tr=X_train,     X_vl=X_val,     X_te=X_test,
            feat=feat_names_16d,
            use_target_transform=False,
            use_sample_weights=True,
        ),
        "feat_engineering": dict(
            X_tr=X_train_eng, X_vl=X_val_eng, X_te=X_test_eng,
            feat=feat_names_28d,
            use_target_transform=False,
            use_sample_weights=False,
        ),
        "all_advanced": dict(
            X_tr=X_train_eng, X_vl=X_val_eng, X_te=X_test_eng,
            feat=feat_names_28d,
            use_target_transform=True,
            use_sample_weights=True,
        ),
    }

    # ---- Sweep ----
    all_results: Dict[str, Dict[str, float]] = {}

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"CONFIG: {cfg_name}  |  "
              f"feat={len(cfg['feat'])}D  "
              f"transform={cfg['use_target_transform']}  "
              f"weights={cfg['use_sample_weights']}")
        print(f"{'='*60}")
        t0 = time.time()
        boosters, transformers = train_one_config(
            cfg_name,
            cfg["X_tr"], Y_train,
            cfg["X_vl"], Y_val,
            cfg["feat"],
            use_target_transform=cfg["use_target_transform"],
            use_sample_weights=cfg["use_sample_weights"],
        )
        r2s = evaluate_config(
            boosters, transformers,
            cfg["X_te"], Y_test,
            cfg["feat"],
        )
        all_results[cfg_name] = r2s
        mean_r2 = np.nanmean(list(r2s.values()))
        print(f"  mean R²={mean_r2:.4f}   elapsed={time.time()-t0:.1f}s")

    # ---- Results ----
    df = pd.DataFrame(all_results).T[METRIC_NAMES]
    df.index.name = "config"

    results_path = OUTPUT_DIR / "advanced_results.csv"
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
