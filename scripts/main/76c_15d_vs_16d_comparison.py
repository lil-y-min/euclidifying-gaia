"""
76c_15d_vs_16d_comparison.py
==============================

Trains both 16D and 15D (no visibility_periods_used) classifiers for the
inside/outside galaxy task and produces comparison plots.

Outputs (plots/ml_runs/galaxy_env_clf/):
  20_roc_comparison.png          16D vs 15D ROC on same axes
  21_pr_comparison.png           16D vs 15D PR on same axes
  22_feature_importance_15d.png  15D feature importance bar chart
  23_feature_importance_comparison.png  side-by-side 16D vs 15D
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema import get_feature_cols, scaler_stem

# ======================================================================
BASE         = Path(__file__).resolve().parents[2]
DATASET_ROOT = BASE / "output" / "dataset_npz"
OUT_DIR      = BASE / "output" / "ml_runs" / "galaxy_env_clf"
PLOT_DIR     = BASE / "plots"  / "ml_runs" / "galaxy_env_clf"
META_NAME    = "metadata_16d.csv"

BIG_GALAXY_TABLE = [
    ("IC342",       56.6988,    68.0961,   10.7,  "ERO-IC342"),
    ("NGC6744",    287.4421,   -63.8575,    7.75, "ERO-NGC6744"),
    ("NGC2403",    114.2142,    65.6031,   10.95, "ERO-NGC2403"),
    ("NGC6822",    296.2404,   -14.8031,    7.75, "ERO-NGC6822"),
    ("HolmbergII", 124.7708,    70.7219,    3.95, "ERO-HolmbergII"),
    ("IC10",         5.0721,    59.3039,    3.15, "ERO-IC10"),
]
OUTSIDE_FIELDS = {
    "ERO-Abell2390","ERO-Abell2764","ERO-Barnard30",
    "ERO-Horsehead","ERO-Messier78","ERO-Taurus",
    "ERO-NGC6397","ERO-NGC6254","ERO-Perseus",
}

PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "learning_rate": 0.05, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 1.0, "tree_method": "hist",
    "seed": 123, "verbosity": 0,
}

# ======================================================================

def load_scaler(feature_cols):
    stem = scaler_stem("16D")
    d    = np.load(BASE / "output" / "scalers" / f"{stem}.npz", allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    idx  = np.array([names.index(c) for c in feature_cols], dtype=int)
    return d["y_min"].astype(np.float32)[idx], d["y_iqr"].astype(np.float32)[idx]


def load_and_label():
    frames = []
    for fdir in sorted(p for p in DATASET_ROOT.iterdir()
                       if p.is_dir() and (p / META_NAME).exists()):
        df = pd.read_csv(fdir / META_NAME, low_memory=False)
        if "field_tag" not in df.columns:
            df["field_tag"] = fdir.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    galaxy_tags = {g[4] for g in BIG_GALAXY_TABLE}
    df = df[df["field_tag"].isin(galaxy_tags | OUTSIDE_FIELDS)].copy()
    df["label"] = 0

    src = SkyCoord(ra=df["ra"].to_numpy()*u.deg,
                   dec=df["dec"].to_numpy()*u.deg, frame="icrs")
    for name, ra, dec, r_d25, tag in BIG_GALAXY_TABLE:
        gc   = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
        mask = df["field_tag"] == tag
        if not mask.any(): continue
        seps = src[mask.to_numpy()].separation(gc).to(u.arcmin).value
        df.loc[df.index[mask], "label"] = (seps < r_d25).astype(int)
    return df


def train_and_eval(df, feature_cols):
    fm, fi = load_scaler(feature_cols)
    df = df.dropna(subset=feature_cols).copy()
    X  = (df[feature_cols].to_numpy(dtype=np.float32) - fm) / np.where(fi > 0, fi, 1.0)
    y  = df["label"].to_numpy(dtype=np.float32)
    sp = df["split_code"].to_numpy(dtype=int) if "split_code" in df.columns else np.zeros(len(df), dtype=int)

    X_tr,y_tr = X[sp==0],y[sp==0]
    X_vl,y_vl = X[sp==1],y[sp==1]
    X_te,y_te = X[sp==2],y[sp==2]

    spw    = float((y_tr==0).sum()) / max(float((y_tr==1).sum()), 1.0)
    params = {**PARAMS, "scale_pos_weight": spw}

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_vl, label=y_vl, feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_te, label=y_te, feature_names=feature_cols)

    booster = xgb.train(
        params, dtrain, num_boost_round=1000,
        evals=[(dtrain,"train"),(dval,"val")],
        early_stopping_rounds=50, verbose_eval=100,
    )

    prob   = booster.predict(dtest)
    auc    = roc_auc_score(y_te, prob)
    ap     = average_precision_score(y_te, prob)
    fpr, tpr, _ = roc_curve(y_te, prob)
    prec, rec, _ = precision_recall_curve(y_te, prob)
    fi_scores = booster.get_score(importance_type="gain")

    return dict(auc=auc, ap=ap, fpr=fpr, tpr=tpr,
                prec=prec, rec=rec, fi=fi_scores,
                y_te=y_te, booster=booster)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cols_16d = get_feature_cols("16D")
    cols_15d = [c for c in cols_16d if c != "feat_visibility_periods_used"]

    print("Loading data...")
    df = load_and_label()
    print(f"  {len(df)} sources, {df['label'].sum()} inside-galaxy")

    print("\nTraining 16D model...")
    r16 = train_and_eval(df, cols_16d)
    print(f"  16D: AUC={r16['auc']:.4f}  AP={r16['ap']:.4f}")

    print("\nTraining 15D model (no visibility_periods_used)...")
    r15 = train_and_eval(df, cols_15d)
    print(f"  15D: AUC={r15['auc']:.4f}  AP={r15['ap']:.4f}")

    C16 = "#2196F3"   # blue  for 16D
    C15 = "#e8671b"   # orange for 15D

    # ------------------------------------------------------------------
    # 20  ROC comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r16["fpr"], r16["tpr"], color=C16, lw=2,
            label=f"16D  AUC={r16['auc']:.3f}")
    ax.plot(r15["fpr"], r15["tpr"], color=C15, lw=2, linestyle="--",
            label=f"15D (no visibility_periods_used)  AUC={r15['auc']:.3f}")
    ax.plot([0,1],[0,1], "k--", lw=0.7)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC — inside-galaxy classifier: 16D vs 15D", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "20_roc_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved 20_roc_comparison.png")

    # ------------------------------------------------------------------
    # 21  PR comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r16["rec"], r16["prec"], color=C16, lw=2,
            label=f"16D  AP={r16['ap']:.3f}")
    ax.plot(r15["rec"], r15["prec"], color=C15, lw=2, linestyle="--",
            label=f"15D (no visibility_periods_used)  AP={r15['ap']:.3f}")
    prevalence = r16["y_te"].mean()
    ax.axhline(prevalence, color="k", linestyle=":", lw=0.9,
               label=f"Baseline (prevalence={prevalence:.3f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("PR — inside-galaxy classifier: 16D vs 15D", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "21_pr_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved 21_pr_comparison.png")

    # ------------------------------------------------------------------
    # 22  15D feature importance
    # ------------------------------------------------------------------
    fi15 = pd.DataFrame({"feature": list(r15["fi"].keys()),
                          "gain":    list(r15["fi"].values())}).sort_values("gain", ascending=False)
    top = fi15.head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([f.replace("feat_","") for f in top["feature"]][::-1],
            top["gain"].values[::-1], color=C15)
    ax.set_xlabel("XGBoost gain")
    ax.set_title("Feature importance — 15D classifier\n(visibility_periods_used removed)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "22_feature_importance_15d.png", dpi=150)
    plt.close(fig)
    print("  saved 22_feature_importance_15d.png")

    # ------------------------------------------------------------------
    # 23  Side-by-side feature importance: 16D vs 15D
    # ------------------------------------------------------------------
    fi16 = pd.DataFrame({"feature": list(r16["fi"].keys()),
                          "gain":    list(r16["fi"].values())}).sort_values("gain", ascending=False)
    top16 = fi16.head(16)
    top15 = fi15.head(15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.barh([f.replace("feat_","") for f in top16["feature"]][::-1],
            top16["gain"].values[::-1], color=C16)
    ax.set_xlabel("XGBoost gain")
    ax.set_title(f"16D  (AUC={r16['auc']:.3f})", fontsize=11)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    # Highlight features that changed rank significantly
    feats15 = [f.replace("feat_","") for f in top15["feature"]]
    colors15 = []
    rank16   = {f.replace("feat_",""): i for i, f in enumerate(top16["feature"])}
    for f in feats15:
        r_old = rank16.get(f, 99)
        r_new = feats15.index(f)
        if f == "visibility_periods_used":
            colors15.append("#cccccc")          # removed — grey
        elif r_new < r_old - 1:
            colors15.append("#2ecc71")          # moved up — green
        else:
            colors15.append(C15)
    ax.barh(feats15[::-1], top15["gain"].values[::-1], color=colors15[::-1])
    ax.set_xlabel("XGBoost gain")
    ax.set_title(f"15D — no visibility_periods_used  (AUC={r15['auc']:.3f})", fontsize=11)
    ax.grid(axis="x", alpha=0.25)

    # Legend for colours
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=C15,      label="same rank"),
        Patch(color="#2ecc71", label="moved up without visibility"),
    ], fontsize=8, loc="lower right")

    fig.suptitle("Feature importance: 16D vs 15D inside-galaxy classifier", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "23_feature_importance_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved 23_feature_importance_comparison.png")

    print(f"\nAll plots -> {PLOT_DIR}")
    print(f"\nSummary:")
    print(f"  16D: AUC={r16['auc']:.4f}  AP={r16['ap']:.4f}")
    print(f"  15D: AUC={r15['auc']:.4f}  AP={r15['ap']:.4f}")
    print(f"  Cost of removing visibility_periods_used: "
          f"ΔAUC={r15['auc']-r16['auc']:+.4f}  ΔAP={r15['ap']-r16['ap']:+.4f}")


if __name__ == "__main__":
    main()
