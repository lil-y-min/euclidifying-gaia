#!/usr/bin/env python3
"""
83_galaxy_clf_roc.py
====================
Re-evaluate the galaxy environment classifier (script 76) and produce:
  - ROC curve with Youden-J optimal threshold marked
  - Precision-Recall curve with F1-optimal threshold marked
  - Saves optimal threshold to optimal_threshold.json (for script 82)

Outputs: report/model_decision/20260306_galaxy_disk_membership/
    17_roc_prc_galaxy_clf.png
    optimal_threshold.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split as tts
from astropy.coordinates import SkyCoord
import astropy.units as u

import sys
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts" / "main"))
from feature_schema import get_feature_cols, scaler_stem

DATASET_ROOT = BASE / "output" / "dataset_npz"
MODEL_PATH   = BASE / "output" / "ml_runs" / "galaxy_env_clf" / "model_inside_galaxy.json"
PLOT_DIR     = BASE / "report" / "model_decision" / "20260306_galaxy_disk_membership"
META_NAME    = "metadata_16d.csv"
FEATURE_SET  = "16D"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

BIG_GALAXY_TABLE = [
    ("IC342",       56.6988,    68.0961,   9.976,  9.527,   0.0, 2.7, "ERO-IC342"),
    ("NGC6744",    287.4421,   -63.8575,   7.744,  4.775,  13.7, 2.0, "ERO-NGC6744"),
    ("NGC2403",    114.2142,    65.6031,   9.976,  5.000, 126.3, 3.0, "ERO-NGC2403"),
    ("NGC6822",    296.2404,   -14.8031,   7.744,  7.396,   0.0, 3.0, "ERO-NGC6822"),
    ("HolmbergII", 124.7708,    70.7219,   3.972,  2.812,  15.2, 1.2, "ERO-HolmbergII"),
    ("IC10",         5.0721,    59.3039,   3.380,  3.013, 129.0, 1.0, "ERO-IC10"),
]
OUTSIDE_FIELDS = {
    "ERO-Abell2390", "ERO-Abell2764", "ERO-Barnard30",
    "ERO-Horsehead", "ERO-Messier78", "ERO-Taurus",
    "ERO-NGC6397",   "ERO-NGC6254",   "ERO-Perseus",
}


def load_scaler(feature_cols):
    stem = scaler_stem(FEATURE_SET)
    d = np.load(BASE / "output" / "scalers" / f"{stem}.npz", allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    idx = np.array([names.index(c) for c in feature_cols], dtype=int)
    return d["y_min"].astype(np.float32)[idx], d["y_iqr"].astype(np.float32)[idx]


def elliptical_radius(ra_s, dec_s, ra0, dec0, a, b, pa_deg):
    pa = np.radians(pa_deg)
    da = (ra_s - ra0) * np.cos(np.radians(dec0)) * 60.0
    dd = (dec_s - dec0) * 60.0
    x  = da * np.sin(pa) + dd * np.cos(pa)
    y  = da * np.cos(pa) - dd * np.sin(pa)
    return np.sqrt((x / a)**2 + (y / b)**2)


def build_dataset(feature_cols):
    galaxy_tags = {row[7] for row in BIG_GALAXY_TABLE}
    keep_tags   = galaxy_tags | OUTSIDE_FIELDS

    frames = []
    for fdir in sorted(DATASET_ROOT.iterdir()):
        if not fdir.is_dir(): continue
        meta = fdir / META_NAME
        if not meta.exists(): continue
        if fdir.name not in keep_tags: continue
        df = pd.read_csv(meta, low_memory=False)
        df["field_tag"] = fdir.name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["label_inside"] = 0

    for name, ra0, dec0, a, b, pa, r_eff, tag in BIG_GALAXY_TABLE:
        m = df["field_tag"] == tag
        if not m.any(): continue
        ell = elliptical_radius(df.loc[m,"ra"].to_numpy(), df.loc[m,"dec"].to_numpy(),
                                ra0, dec0, a, b, pa)
        df.loc[m, "label_inside"] = (ell <= 1.0).astype(int)

    df = df.dropna(subset=feature_cols + ["ra", "dec"]).copy()
    return df


def main():
    feature_cols = get_feature_cols(FEATURE_SET)
    feat_min, feat_iqr = load_scaler(feature_cols)

    print("Building dataset...")
    df = build_dataset(feature_cols)
    print(f"  Total: {len(df):,}  Inside: {df['label_inside'].sum():,}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    X = (X - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)
    y = df["label_inside"].to_numpy(dtype=np.float32)

    # Reproduce same split as script 76
    idx_all = np.arange(len(y))
    idx_tmp, idx_te = tts(idx_all, test_size=0.20, stratify=y, random_state=42)
    idx_tr, idx_vl  = tts(idx_tmp, test_size=0.125, stratify=y[idx_tmp], random_state=42)

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    # Evaluate on validation set for threshold selection
    dval  = xgb.DMatrix(X[idx_vl], feature_names=feature_cols)
    dtest = xgb.DMatrix(X[idx_te], feature_names=feature_cols)

    prob_vl = booster.predict(dval)
    prob_te = booster.predict(dtest)
    y_vl, y_te = y[idx_vl], y[idx_te]

    # Youden-J threshold from validation ROC
    fpr_v, tpr_v, thr_v = roc_curve(y_vl, prob_vl)
    j_idx  = np.argmax(tpr_v - fpr_v)
    thr_youden = float(thr_v[j_idx])

    # F1-optimal threshold from validation PR
    prec_v, rec_v, thr_pr_v = precision_recall_curve(y_vl, prob_vl)
    f1_v = 2 * prec_v[:-1] * rec_v[:-1] / np.maximum(prec_v[:-1] + rec_v[:-1], 1e-9)
    thr_f1 = float(thr_pr_v[np.argmax(f1_v)])

    print(f"  Youden-J threshold (val): {thr_youden:.4f}")
    print(f"  F1-optimal threshold (val): {thr_f1:.4f}")

    # Test metrics
    auc = roc_auc_score(y_te, prob_te)
    ap  = average_precision_score(y_te, prob_te)
    fpr_t, tpr_t, _ = roc_curve(y_te, prob_te)
    prec_t, rec_t, _ = precision_recall_curve(y_te, prob_te)

    print(f"  Test AUC={auc:.4f}  AP={ap:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax = axes[0]
    ax.plot(fpr_t, tpr_t, color="#1d3557", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    # Mark Youden-J on test ROC
    j_idx_t = np.argmax(tpr_t - fpr_t)
    ax.scatter([fpr_t[j_idx_t]], [tpr_t[j_idx_t]], s=80, color="#e63946", zorder=5,
               label=f"Youden-J  thr={thr_youden:.3f}")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC — galaxy disk membership classifier", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # PRC
    ax = axes[1]
    ax.plot(rec_t, prec_t, color="#457b9d", lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(y_te.mean(), color="gray", lw=0.8, linestyle="--",
               label=f"Baseline  ({y_te.mean():.3f})")
    # Mark F1-optimal on test PRC
    f1_t = 2 * prec_t[:-1] * rec_t[:-1] / np.maximum(prec_t[:-1] + rec_t[:-1], 1e-9)
    best_t = np.argmax(f1_t)
    ax.scatter([rec_t[best_t]], [prec_t[best_t]], s=80, color="#e63946", zorder=5,
               label=f"F1-opt  thr={thr_f1:.3f}  F1={f1_t[best_t]:.3f}")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall — galaxy disk membership classifier", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle(
        f"Galaxy environment classifier  |  Test AUC={auc:.3f}  AP={ap:.3f}  "
        f"n_test={len(y_te):,}  n_inside={int(y_te.sum())}",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = PLOT_DIR / "17_roc_prc_galaxy_clf.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png.name}")

    # Save thresholds
    out_json = PLOT_DIR / "optimal_threshold.json"
    out_json.write_text(json.dumps({
        "thr_youden": thr_youden,
        "thr_f1":     thr_f1,
        "auc":        auc,
        "ap":         ap,
        "note":       "thresholds selected on val set; AUC/AP on test set",
    }, indent=2))
    print(f"  saved {out_json.name}")
    print(f"\n  Use thr_youden={thr_youden:.4f} or thr_f1={thr_f1:.4f} in script 82.")


if __name__ == "__main__":
    main()
