#!/usr/bin/env python3
"""
76d_retrain_15d.py
==================
Retrain the galaxy disk membership classifier with 15D features
(16D minus feat_visibility_periods_used, which is a scanning-law confound).

Saves:
  output/ml_runs/galaxy_env_clf/model_inside_galaxy_15d.json
  output/ml_runs/galaxy_env_clf/metrics_15d.json
  output/ml_runs/galaxy_env_clf/feature_importance_15d.csv
  report/model_decision/20260306_galaxy_disk_membership/optimal_threshold_15d.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
)
import sys

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts" / "main"))
from feature_schema import get_feature_cols, scaler_stem

DATASET_ROOT = BASE / "output" / "dataset_npz"
OUT_DIR      = BASE / "output" / "ml_runs" / "galaxy_env_clf"
REPORT_DIR   = BASE / "report" / "model_decision" / "20260306_galaxy_disk_membership"
META_NAME    = "metadata_16d.csv"
FEATURE_SET  = "15D"

GALAXIES = [
    ("IC342",       56.6988,   68.0961,  9.976, 9.527,   0.0, 2.7, "ERO-IC342"),
    ("NGC6744",    287.4421,  -63.8575,  7.744, 4.775,  13.7, 2.0, "ERO-NGC6744"),
    ("NGC2403",    114.2142,   65.6031,  9.976, 5.000, 126.3, 3.0, "ERO-NGC2403"),
    ("NGC6822",    296.2404,  -14.8031,  7.744, 7.396,   0.0, 3.0, "ERO-NGC6822"),
    ("HolmbergII", 124.7708,   70.7219,  3.972, 2.812,  15.2, 1.2, "ERO-HolmbergII"),
    ("IC10",         5.0721,   59.3039,  3.380, 3.013, 129.0, 1.0, "ERO-IC10"),
]
OUTSIDE_FIELDS = {
    "ERO-Abell2390", "ERO-Abell2764", "ERO-Barnard30",
    "ERO-Horsehead", "ERO-Messier78", "ERO-Taurus",
    "ERO-NGC6397", "ERO-NGC6254", "ERO-Perseus",
}

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "seed": 123,
    "verbosity": 0,
}


def elliptical_radius(ra_s, dec_s, ra0, dec0, a, b, pa_deg):
    pa = np.radians(pa_deg)
    da = (ra_s - ra0) * np.cos(np.radians(dec0)) * 60.0
    dd = (dec_s - dec0) * 60.0
    x  = da * np.sin(pa) + dd * np.cos(pa)
    y  = da * np.cos(pa) - dd * np.sin(pa)
    return np.sqrt((x / a)**2 + (y / b)**2)


def main():
    feature_cols = get_feature_cols(FEATURE_SET)
    print(f"Feature set: {FEATURE_SET}  ({len(feature_cols)} features)")
    print("  Dropped vs 16D: feat_visibility_periods_used")

    # Load scaler
    stem  = scaler_stem(FEATURE_SET)
    npz   = BASE / "output" / "scalers" / f"{stem}.npz"
    d     = np.load(npz, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    idx   = np.array([names.index(c) for c in feature_cols], dtype=int)
    feat_min = d["y_min"].astype(np.float32)[idx]
    feat_iqr = d["y_iqr"].astype(np.float32)[idx]

    # Load all fields
    galaxy_tags = {g[7] for g in GALAXIES}
    all_tags    = galaxy_tags | OUTSIDE_FIELDS
    frames = []
    for fdir in sorted(DATASET_ROOT.iterdir()):
        if not (fdir / META_NAME).exists() or fdir.name not in all_tags:
            continue
        df = pd.read_csv(fdir / META_NAME, low_memory=False)
        df["field_tag"] = fdir.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=feature_cols + ["ra", "dec"]).copy()

    # Labels
    df["label"] = 0
    for name, ra0, dec0, a, b, pa, r_eff, tag in GALAXIES:
        m = df["field_tag"] == tag
        if not m.any():
            continue
        ell = elliptical_radius(
            df.loc[m, "ra"].to_numpy(), df.loc[m, "dec"].to_numpy(),
            ra0, dec0, a, b, pa)
        df.loc[m, "label"] = (ell <= 1.0).astype(int)

    print(f"  Total rows: {len(df):,}  inside={df['label'].sum():,}")

    # Stratified split (same seed as script 76)
    y   = df["label"].to_numpy(dtype=np.float32)
    idx_all = np.arange(len(y))
    idx_tmp, idx_te = tts(idx_all, test_size=0.20, stratify=y, random_state=42)
    idx_tr, idx_vl  = tts(idx_tmp, test_size=0.125, stratify=y[idx_tmp], random_state=42)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    X = (X - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)

    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_vl, y_vl = X[idx_vl], y[idx_vl]
    X_te, y_te = X[idx_te], y[idx_te]
    print(f"  Train={len(X_tr)} val={len(X_vl)} test={len(X_te)}")

    pos_ratio = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    params = {**PARAMS, "scale_pos_weight": pos_ratio}

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_vl, label=y_vl, feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_te, label=y_te, feature_names=feature_cols)

    print("Training 15D classifier...")
    booster = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    booster.save_model(str(OUT_DIR / "model_inside_galaxy_15d.json"))

    # Metrics
    prob_te = booster.predict(dtest)
    auc = roc_auc_score(y_te, prob_te)
    ap  = average_precision_score(y_te, prob_te)
    print(f"\nTest  AUC={auc:.4f}  AP={ap:.4f}")

    # Youden-J threshold on val set
    prob_vl = booster.predict(dval)
    fpr, tpr, thrs = roc_curve(y_vl, prob_vl)
    j_idx = np.argmax(tpr - fpr)
    thr_youden = float(thrs[j_idx])
    print(f"Youden threshold (val): {thr_youden:.4f}")

    metrics = {
        "feature_set": FEATURE_SET,
        "auc": auc, "ap": ap,
        "n_test": int(len(y_te)),
        "n_inside_test": int(y_te.sum()),
        "best_iteration": booster.best_iteration,
    }
    (OUT_DIR / "metrics_15d.json").write_text(json.dumps(metrics, indent=2))

    thr_data = {
        "thr_youden": thr_youden,
        "auc": auc, "ap": ap,
        "note": "15D model — visibility_periods_used removed",
    }
    (REPORT_DIR / "optimal_threshold.json").write_text(json.dumps(thr_data, indent=2))
    print(f"  Updated optimal_threshold.json → thr_youden={thr_youden:.4f}")

    fi = booster.get_score(importance_type="gain")
    fi_df = pd.DataFrame({"feature": list(fi.keys()), "gain": list(fi.values())})
    fi_df = fi_df.sort_values("gain", ascending=False)
    fi_df.to_csv(OUT_DIR / "feature_importance_15d.csv", index=False)

    print("\nTop features (15D):")
    print(fi_df.head(10).to_string(index=False))
    print(f"\nSaved model → {OUT_DIR / 'model_inside_galaxy_15d.json'}")


if __name__ == "__main__":
    main()
