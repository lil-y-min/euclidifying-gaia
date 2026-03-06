"""
80_psf_morph_classification.py
===============================

Two analyses using PSF/non-PSF labels from nn_psf_labels/labels_psf_weak.csv:

PART A — Morphology metric space coloured by PSF/non-PSF label
  Scatter plots: ellipticity vs gini, kurtosis vs asymmetry_180, etc.
  Shows that our computed morphology metrics separate the two classes.

PART B — Classification AUC comparison
  Three classifiers on the TEST set:
    1. Gaia 16D direct         → PSF label  (existing approach, baseline)
    2. Predicted 11 metrics    → PSF label  (via morphology bottleneck)
    3. True 11 metrics         → PSF label  (oracle upper bound)
  Plots ROC curves for all three + reports AUC.

Outputs:
  plots/ml_runs/morph_metrics_16d/
    24_psf_morph_scatter.png         — scatter colored by PSF label
    25_psf_classification_roc.png    — ROC curves comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.model_selection import cross_val_score

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
PRED_CSV    = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "full_gaia_predictions.csv"
PSF_LABELS  = BASE_DIR / "output" / "ml_runs" / "nn_psf_labels" / "labels_psf_weak.csv"
PLOTS_DIR   = BASE_DIR / "plots"  / "ml_runs" / "morph_metrics_16d"


# ======================================================================
# HELPERS
# ======================================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def load_psf_labels() -> pd.DataFrame:
    df = pd.read_csv(PSF_LABELS)
    df["source_id"] = df["source_id"].astype(str)
    return df


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PRED_CSV)
    df["source_id"] = df["source_id"].astype(str)
    return df


# ======================================================================
# PART A — Morphology scatter colored by PSF label
# ======================================================================

def plot_psf_morph_scatter(
    psf_df: pd.DataFrame,
    Y_test: np.ndarray,
    labels_test: np.ndarray,
    out_path: Path,
) -> None:
    """
    4-panel scatter: pairs of our true morphology metrics, colored by PSF label.
    """
    pairs = [
        ("gini",       "ellipticity"),
        ("kurtosis",   "asymmetry_180"),
        ("gini",       "multipeak_ratio"),
        ("smoothness", "hf_artifacts"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = {1: "#2166ac", 0: "#d6604d"}
    labels_map = {1: "PSF-like", 0: "Non-PSF"}
    alpha = 0.15
    s = 3

    for ax, (mx, my) in zip(axes, pairs):
        ix = METRIC_NAMES.index(mx)
        iy = METRIC_NAMES.index(my)

        for lbl in [1, 0]:
            mask = (labels_test == lbl)
            x = Y_test[mask, ix]
            y = Y_test[mask, iy]
            ok = np.isfinite(x) & np.isfinite(y)
            # Clip outliers for display
            xp = np.clip(x[ok], *np.percentile(x[ok], [0.5, 99.5]))
            yp = np.clip(y[ok], *np.percentile(y[ok], [0.5, 99.5]))
            ax.scatter(xp, yp, s=s, alpha=alpha, color=colors[lbl],
                       label=f"{labels_map[lbl]} (n={ok.sum()})", rasterized=True)

        ax.set_xlabel(mx, fontsize=9)
        ax.set_ylabel(my, fontsize=9)
        ax.legend(fontsize=8, markerscale=3)
        ax.set_title(f"{mx} vs {my}", fontsize=10, fontweight="bold")

    fig.suptitle(
        "True morphology metrics: PSF-like (blue) vs Non-PSF (red)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# PART B — ROC comparison
# ======================================================================

def train_xgb_classifier(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    label: str,
) -> tuple[np.ndarray, float]:
    """Train simple XGBoost binary classifier, return (y_score, auc)."""
    params = {
        "objective":     "binary:logistic",
        "eval_metric":   "auc",
        "max_depth":     5,
        "learning_rate": 0.05,
        "subsample":     0.8,
        "colsample_bytree": 0.8,
        "seed":          42,
        "tree_method":   "hist",
    }
    ok_tr = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    ok_te = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)

    dtrain = xgb.DMatrix(X_train[ok_tr], label=y_train[ok_tr])
    dval   = xgb.DMatrix(X_test[ok_te],  label=y_test[ok_te])

    booster = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    scores = booster.predict(xgb.DMatrix(X_test[ok_te]))
    fpr, tpr, _ = roc_curve(y_test[ok_te], scores)
    auc_val = sk_auc(fpr, tpr)
    print(f"  {label:<35s}  AUC = {auc_val:.4f}  (n_test={ok_te.sum()})")
    return (fpr, tpr, auc_val, label)


def plot_roc_comparison(results: list, out_path: Path) -> None:
    colors = ["#1a9850", "#4393c3", "#d73027"]
    fig, ax = plt.subplots(figsize=(7, 7))

    for (fpr, tpr, auc_val, label), color in zip(results, colors):
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{label}  (AUC = {auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        "ROC: PSF/non-PSF classification via different feature sets",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# LOAD GAIA 16D FEATURES WITH SOURCE_ID
# ======================================================================

def load_gaia_features_with_source_id(feature_cols: List[str]) -> pd.DataFrame:
    """
    Read metadata_16d.csv for each field, apply mag filter + feature validity,
    scale with IQR scaler. Returns DataFrame with source_id + scaled gaia_0..gaia_N cols.
    """
    gaia_min, gaia_iqr = _mod.load_gaia_scaler(CFG.gaia_scaler_npz, feature_cols)
    field_dirs = _mod.list_field_dirs(CFG.dataset_root)
    rows = []
    for fdir in field_dirs:
        meta_path = fdir / _mod.RAW_META_16D
        if not meta_path.exists():
            continue
        meta = pd.read_csv(meta_path, engine="python", on_bad_lines="skip")
        g = pd.to_numeric(meta["phot_g_mean_mag"], errors="coerce").to_numpy()
        feat_raw = meta[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid = (
            np.isfinite(g) & (g >= CFG.gaia_g_mag_min)
            & np.all(np.isfinite(feat_raw), axis=1)
        )
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        X = ((feat_raw[idx] - gaia_min) / np.where(gaia_iqr > 0, gaia_iqr, 1.0)).astype(np.float32)
        sub = pd.DataFrame(X, columns=[f"gaia_{i}" for i in range(len(feature_cols))])
        sub["source_id"] = meta["source_id"].iloc[idx].astype(str).values
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    feature_cols = get_feature_cols(CFG.feature_set)

    # ---- Load PSF labels (all splits) ----
    print("Loading PSF labels ...")
    psf_df = load_psf_labels()
    print(f"  Total PSF labels: {len(psf_df)}  "
          f"(PSF={psf_df['label'].sum()}, nonPSF={(psf_df['label']==0).sum()})")

    # ---- Load predictions CSV (has pred_* cols + source_id) ----
    print("Loading predictions ...")
    pred_df = load_predictions()

    # ---- Load proper scaled 16D Gaia features keyed by source_id ----
    print("Loading Gaia 16D features from metadata ...")
    gaia_df = load_gaia_features_with_source_id(feature_cols)
    gaia_cols = [f"gaia_{i}" for i in range(len(feature_cols))]
    print(f"  Loaded {len(gaia_df)} sources with valid Gaia features")

    # ---- Three-way merge: PSF labels + gaia_df + pred_df ----
    merged = psf_df.merge(gaia_df, on="source_id", how="inner")
    merged = merged.merge(pred_df, on="source_id", how="inner")
    print(f"  Three-way matched: {len(merged)} sources")
    print(f"  Split breakdown: " +
          str(merged["split_code"].value_counts().sort_index().to_dict()))

    if len(merged) == 0:
        print("  ERROR: No matches found.")
        return

    test_set = merged[merged["split_code"] == 2]
    print(f"  Test set: {len(test_set)}")

    labels_matched   = merged["label"].to_numpy(dtype=np.float32)
    labels_test_only = test_set["label"].to_numpy(dtype=np.float32)
    matched_splits   = merged["split_code"].to_numpy()
    test_mask        = matched_splits == 2

    # Feature matrices
    X_gaia  = merged[gaia_cols].to_numpy(dtype=np.float32)
    pred_cols = [f"pred_{m}" for m in METRIC_NAMES]
    X_pred  = merged[pred_cols].to_numpy(dtype=np.float32)

    proxy_morph_cols = [
        "m_concentration_r2_r6", "m_asymmetry_180", "m_ellipticity",
        "m_peak_sep_pix", "m_edge_flux_frac", "m_peak_ratio_2over1",
    ]
    X_proxy = merged[proxy_morph_cols].to_numpy(dtype=np.float32)

    def get_split(arr, code_train=0, code_test=2):
        tr = matched_splits == code_train
        te = matched_splits == code_test
        return arr[tr], arr[te]

    # ---- PART A: Scatter plot ----
    print("\nPart A: Morphology scatter ...")
    proxy_map = {
        "ellipticity":    "m_ellipticity",
        "asymmetry_180":  "m_asymmetry_180",
        "concentration":  "m_concentration_r2_r6",
        "multipeak_ratio":"m_peak_ratio_2over1",
    }
    Y_for_scatter = np.zeros((len(merged), len(METRIC_NAMES)), dtype=np.float32)
    for mname, pcol in proxy_map.items():
        midx = METRIC_NAMES.index(mname)
        Y_for_scatter[:, midx] = pd.to_numeric(merged[pcol], errors="coerce").to_numpy()
    for mname in ["gini", "smoothness", "hf_artifacts", "kurtosis"]:
        if mname not in proxy_map:
            midx = METRIC_NAMES.index(mname)
            Y_for_scatter[:, midx] = merged[f"pred_{mname}"].to_numpy()

    plot_psf_morph_scatter(
        psf_df, Y_for_scatter[test_mask], labels_test_only,
        PLOTS_DIR / "24_psf_morph_scatter.png",
    )

    # ---- PART B: ROC comparison ----
    print("\nPart B: ROC comparison ...")

    y_tr, y_te         = get_split(labels_matched)
    X_tr_gaia, X_te_gaia   = get_split(X_gaia)
    X_tr_pred, X_te_pred   = get_split(X_pred)
    X_tr_proxy, X_te_proxy = get_split(X_proxy)

    print(f"  Train: {y_tr.sum():.0f} PSF / {(y_tr==0).sum()} non-PSF")
    print(f"  Test:  {y_te.sum():.0f} PSF / {(y_te==0).sum()} non-PSF")

    results = []
    results.append(train_xgb_classifier(
        X_tr_gaia,  y_tr, X_te_gaia,  y_te, "Gaia 16D direct",
    ))
    results.append(train_xgb_classifier(
        X_tr_pred,  y_tr, X_te_pred,  y_te, "Predicted morphology (11 metrics)",
    ))
    results.append(train_xgb_classifier(
        X_tr_proxy, y_tr, X_te_proxy, y_te, "True morphology proxy (6 metrics)",
    ))

    plot_roc_comparison(results, PLOTS_DIR / "25_psf_classification_roc.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
