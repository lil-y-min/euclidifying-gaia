"""
76_galaxy_environment_classifier.py
=====================================

Binary XGBoost classifier: is a Gaia source physically inside a big
nearby galaxy (label=1) or not (label=0)?

Labels are computed purely from sky geometry: each source is matched to the
nearest galaxy in a hardcoded table of big ERO targets.  If the angular
separation from the galaxy nucleus is less than R_D25 (half the D25 optical
diameter), the source is labelled "inside".  Sources in ERO fields that do NOT
host a big galaxy are used as the "outside" negative population.

Features: 16D Gaia feat_ columns (same as stamp reconstruction pipeline).
Split: reuses existing split_code (0=train, 1=val, 2=test) from metadata_16d.

Outputs:
  output/ml_runs/galaxy_env_clf/
    model_inside_galaxy.json
    metrics.json
    feature_importance.csv
  plots/ml_runs/galaxy_env_clf/
    01_roc_pr.png
    02_feature_importance.png
    03_separation_distribution.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema import get_feature_cols, scaler_stem


# ======================================================================
# CONFIG
# ======================================================================

BASE = Path(__file__).resolve().parents[2]
DATASET_ROOT = BASE / "output" / "dataset_npz"
RUN_NAME = "galaxy_env_clf"
OUT_DIR = BASE / "output" / "ml_runs" / RUN_NAME
PLOT_DIR = BASE / "plots" / "ml_runs" / RUN_NAME
META_NAME = "metadata_16d.csv"
FEATURE_SET = "16D"

# Galaxy nuclei: (RA_deg, Dec_deg, R_D25_arcmin, r_eff_arcmin, field_tag)
# R_D25 = half the optical D25 diameter (from NED/HyperLeda).
# r_eff = effective (half-light) radius in arcmin.
BIG_GALAXY_TABLE = [
    # name           RA          Dec        R_D25   r_eff   field_tag
    ("IC342",       56.6988,    68.0961,   10.7,   2.7,   "ERO-IC342"),
    ("NGC6744",    287.4421,   -63.8575,    7.75,  2.0,   "ERO-NGC6744"),
    ("NGC2403",    114.2142,    65.6031,   10.95,  3.0,   "ERO-NGC2403"),
    ("NGC6822",    296.2404,   -14.8031,    7.75,  3.0,   "ERO-NGC6822"),
    ("HolmbergII", 124.7708,    70.7219,    3.95,  1.2,   "ERO-HolmbergII"),
    ("IC10",         5.0721,    59.3039,    3.15,  1.0,   "ERO-IC10"),
]

# Fields that serve purely as "outside" negatives (no big galaxy host).
# Abell clusters, nebulae, globular clusters — sources there are not
# inside any nearby galaxy disk.
OUTSIDE_FIELDS = {
    "ERO-Abell2390", "ERO-Abell2764", "ERO-Barnard30",
    "ERO-Horsehead", "ERO-Messier78", "ERO-Taurus",
    "ERO-NGC6397", "ERO-NGC6254", "ERO-Perseus",
}

# XGB
NUM_BOOST_ROUND = 1000
EARLY_STOP = 50
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


# ======================================================================
# DATA LOADING
# ======================================================================

def load_scaler(feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    stem = scaler_stem(FEATURE_SET)
    npz = BASE / "output" / "scalers" / f"{stem}.npz"
    d = np.load(npz, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    y_min = d["y_min"].astype(np.float32)
    y_iqr = d["y_iqr"].astype(np.float32)
    idx = np.array([names.index(c) for c in feature_cols], dtype=int)
    return y_min[idx], y_iqr[idx]


def build_galaxy_coords() -> list[tuple]:
    """Return list of (name, SkyCoord, R_D25_arcmin, r_eff_arcmin, field_tag)."""
    out = []
    for name, ra, dec, r_d25, r_eff, tag in BIG_GALAXY_TABLE:
        sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        out.append((name, sc, r_d25, r_eff, tag))
    return out


def load_all_fields(feature_cols: list[str]) -> pd.DataFrame:
    """Load metadata_16d from all field dirs, attach field_tag."""
    field_dirs = sorted([
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir() and (p / META_NAME).exists()
    ])
    frames = []
    for fdir in field_dirs:
        df = pd.read_csv(fdir / META_NAME, low_memory=False)
        if "field_tag" not in df.columns:
            df["field_tag"] = fdir.name
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No field dirs found under {DATASET_ROOT}")
    return pd.concat(frames, ignore_index=True)


def assign_labels(df: pd.DataFrame, galaxies: list) -> pd.DataFrame:
    """
    For each row:
    - If in a big-galaxy field: compute sep from nucleus; inside=1 if sep<R_D25.
    - If in a pure-outside field: label=0.
    - Else: drop (field not in either category).
    """
    galaxy_tags = {g[4] for g in galaxies}
    keep_mask = df["field_tag"].isin(galaxy_tags | OUTSIDE_FIELDS)
    df = df[keep_mask].copy()

    df["label_inside"] = 0
    df["sep_arcmin"] = np.nan
    df["r_norm"] = np.nan  # sep / r_eff (for use in script 77)

    src_coords = SkyCoord(
        ra=df["ra"].to_numpy() * u.deg,
        dec=df["dec"].to_numpy() * u.deg,
        frame="icrs",
    )

    for name, gal_coord, r_d25, r_eff, tag in galaxies:
        in_field = df["field_tag"] == tag
        if not in_field.any():
            continue
        seps = src_coords[in_field.to_numpy()].separation(gal_coord).to(u.arcmin).value
        idx = df.index[in_field]
        df.loc[idx, "sep_arcmin"] = seps
        df.loc[idx, "r_norm"] = seps / r_eff
        df.loc[idx, "label_inside"] = (seps < r_d25).astype(int)

    return df


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols(FEATURE_SET)
    feat_min, feat_iqr = load_scaler(feature_cols)

    print("Loading metadata from all fields...")
    df_raw = load_all_fields(feature_cols)
    print(f"  Total rows loaded: {len(df_raw)}")

    galaxies = build_galaxy_coords()
    df = assign_labels(df_raw, galaxies)
    print(f"  Rows after field filter: {len(df)}")
    print(f"  Inside (label=1): {df['label_inside'].sum()}")
    print(f"  Outside (label=0): {(df['label_inside'] == 0).sum()}")

    # Drop rows missing features or ra/dec
    df = df.dropna(subset=feature_cols + ["ra", "dec"]).copy()
    print(f"  Rows after NaN drop: {len(df)}")

    # Scale features
    X = df[feature_cols].to_numpy(dtype=np.float32)
    X = (X - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)
    y = df["label_inside"].to_numpy(dtype=np.float32)
    splits = df["split_code"].to_numpy(dtype=int) if "split_code" in df.columns else np.zeros(len(df), dtype=int)

    X_tr, y_tr = X[splits == 0], y[splits == 0]
    X_vl, y_vl = X[splits == 1], y[splits == 1]
    X_te, y_te = X[splits == 2], y[splits == 2]
    print(f"  Train={len(X_tr)}, Val={len(X_vl)}, Test={len(X_te)}")

    pos_ratio = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    params = {**PARAMS, "scale_pos_weight": pos_ratio}
    print(f"  scale_pos_weight={pos_ratio:.2f}")

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_vl, label=y_vl, feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_te, label=y_te, feature_names=feature_cols)

    print("Training XGB classifier...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=100,
    )

    booster.save_model(str(OUT_DIR / "model_inside_galaxy.json"))

    # ---- Evaluate ----
    prob_te = booster.predict(dtest)
    auc = roc_auc_score(y_te, prob_te)
    ap  = average_precision_score(y_te, prob_te)
    print(f"\nTest AUC={auc:.4f}  AP={ap:.4f}")

    metrics = {"auc": auc, "ap": ap, "n_test": int(len(y_te)),
               "n_inside_test": int(y_te.sum()),
               "best_iteration": booster.best_iteration}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Feature importance
    fi = booster.get_score(importance_type="gain")
    fi_df = pd.DataFrame({"feature": list(fi.keys()), "gain": list(fi.values())})
    fi_df = fi_df.sort_values("gain", ascending=False)
    fi_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    # ---- Plots ----

    # ROC + PR
    fpr, tpr, _ = roc_curve(y_te, prob_te)
    prec, rec, _ = precision_recall_curve(y_te, prob_te)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC  AUC={auc:.3f}")
    axes[1].plot(rec, prec)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR  AP={ap:.3f}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_roc_pr.png", dpi=150)
    plt.close(fig)

    # Feature importance
    top = fi_df.head(16)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"][::-1], top["gain"][::-1])
    ax.set_xlabel("Gain")
    ax.set_title("Feature importance (gain) — inside-galaxy classifier")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_feature_importance.png", dpi=150)
    plt.close(fig)

    # Separation distribution coloured by field
    df_plot = df.dropna(subset=["sep_arcmin"]).copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, _, r_d25, _, tag in galaxies:
        sub = df_plot[df_plot["field_tag"] == tag]["sep_arcmin"]
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=60, histtype="step", label=tag.replace("ERO-", ""))
        ax.axvline(r_d25, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xlabel("Angular separation from nucleus (arcmin)")
    ax.set_ylabel("Sources")
    ax.set_title("Source distribution vs. D25 boundary (dashed)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_separation_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\nOutputs -> {OUT_DIR}")
    print(f"Plots   -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
