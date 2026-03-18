"""
77_galaxy_radius_regression.py
================================

XGBoost regressor: predict r_norm = angular_sep / r_eff for Gaia sources
inside big nearby galaxies, using the 16D Gaia feature set.

Scientific question: how much of the Gaia quality-degradation signal
(RUWE, AEN, IPD metrics, blended transits) is radially ordered within a
galaxy disk? A high R² means the contamination is geometrically encoded —
implying the Gaia features carry spatial provenance information.

Only sources labelled "inside" (sep < R_D25) are used for regression.
The target r_norm = sep_arcmin / r_eff_arcmin (so r_norm=1 ~ half-light radius).

Features: 16D Gaia feat_ columns.
Split: reuses split_code from metadata_16d.

Outputs:
  output/ml_runs/galaxy_radius_reg/
    model_radius.json
    metrics.json
    feature_importance.csv
  plots/ml_runs/galaxy_radius_reg/
    01_scatter_pred_vs_true.png
    02_residuals_by_field.png
    03_feature_importance.png
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
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema import get_feature_cols, scaler_stem


# ======================================================================
# CONFIG (mirrors 76 — same galaxy table, same ERO fields)
# ======================================================================

BASE = Path(__file__).resolve().parents[2]
DATASET_ROOT = BASE / "output" / "dataset_npz"
RUN_NAME = "galaxy_radius_reg"
OUT_DIR = BASE / "output" / "ml_runs" / RUN_NAME
PLOT_DIR = BASE / "report" / "model_decision" / "20260311_galaxy_radius_regression"
META_NAME = "metadata_16d.csv"
FEATURE_SET = "16D"

BIG_GALAXY_TABLE = [
    ("IC342",       56.6988,    68.0961,   10.7,   2.7,   "ERO-IC342"),
    ("NGC6744",    287.4421,   -63.8575,    7.75,  2.0,   "ERO-NGC6744"),
    ("NGC2403",    114.2142,    65.6031,   10.95,  3.0,   "ERO-NGC2403"),
    ("NGC6822",    296.2404,   -14.8031,    7.75,  3.0,   "ERO-NGC6822"),
    ("HolmbergII", 124.7708,    70.7219,    3.95,  1.2,   "ERO-HolmbergII"),
    ("IC10",         5.0721,    59.3039,    3.15,  1.0,   "ERO-IC10"),
]

NUM_BOOST_ROUND = 1000
EARLY_STOP = 50
PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
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
# HELPERS
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


def load_all_fields() -> pd.DataFrame:
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
    return pd.concat(frames, ignore_index=True)


def assign_radius_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only sources inside big galaxy fields; compute r_norm and sep."""
    galaxy_tags = {row[5] for row in BIG_GALAXY_TABLE}
    df = df[df["field_tag"].isin(galaxy_tags)].copy()

    df["r_norm"]    = np.nan
    df["sep_arcmin"] = np.nan
    df["galaxy"]    = ""

    src_coords = SkyCoord(
        ra=df["ra"].to_numpy() * u.deg,
        dec=df["dec"].to_numpy() * u.deg,
        frame="icrs",
    )

    for name, ra, dec, r_d25, r_eff, tag in BIG_GALAXY_TABLE:
        gal_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        in_field  = df["field_tag"] == tag
        if not in_field.any():
            continue
        seps = src_coords[in_field.to_numpy()].separation(gal_coord).to(u.arcmin).value
        idx  = df.index[in_field]
        df.loc[idx, "sep_arcmin"] = seps
        df.loc[idx, "r_norm"]    = seps / r_eff
        df.loc[idx, "galaxy"]    = name

    # Keep only sources inside D25 (where signal should exist)
    inside_mask = pd.Series(False, index=df.index)
    for name, ra, dec, r_d25, r_eff, tag in BIG_GALAXY_TABLE:
        inside_mask |= (df["galaxy"] == name) & (df["sep_arcmin"] < r_d25)
    df = df[inside_mask].copy()
    return df


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols(FEATURE_SET)
    feat_min, feat_iqr = load_scaler(feature_cols)

    print("Loading metadata...")
    df_raw = load_all_fields()
    df = assign_radius_labels(df_raw)
    df = df.dropna(subset=feature_cols + ["r_norm", "ra", "dec"]).copy()
    print(f"  Inside-galaxy sources: {len(df)}")
    print(f"  r_norm range: {df['r_norm'].min():.2f} – {df['r_norm'].max():.2f}")
    print(f"  By galaxy:\n{df.groupby('galaxy').size().to_string()}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    X = (X - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)
    y = df["r_norm"].to_numpy(dtype=np.float32)
    splits = df["split_code"].to_numpy(dtype=int) if "split_code" in df.columns else np.zeros(len(df), dtype=int)

    X_tr, y_tr = X[splits == 0], y[splits == 0]
    X_vl, y_vl = X[splits == 1], y[splits == 1]
    X_te, y_te = X[splits == 2], y[splits == 2]
    gal_te      = df["galaxy"].to_numpy()[splits == 2]
    print(f"  Train={len(X_tr)}, Val={len(X_vl)}, Test={len(X_te)}")

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_vl, label=y_vl, feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_te, label=y_te, feature_names=feature_cols)

    print("Training XGB regressor...")
    booster = xgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=100,
    )

    booster.save_model(str(OUT_DIR / "model_radius.json"))

    # ---- Evaluate ----
    y_pred = booster.predict(dtest)
    r2   = r2_score(y_te, y_pred)
    rmse = float(np.sqrt(np.mean((y_pred - y_te) ** 2)))
    r, p = pearsonr(y_te, y_pred)
    print(f"\nTest  R²={r2:.4f}  RMSE={rmse:.4f}  Pearson r={r:.4f}  (p={p:.2e})")

    # Per-galaxy breakdown
    per_gal = {}
    for gname in np.unique(gal_te):
        m = gal_te == gname
        if m.sum() < 5:
            continue
        per_gal[gname] = {
            "n": int(m.sum()),
            "r2": float(r2_score(y_te[m], y_pred[m])),
            "rmse": float(np.sqrt(np.mean((y_pred[m] - y_te[m]) ** 2))),
        }
        print(f"  {gname}: n={per_gal[gname]['n']}  R²={per_gal[gname]['r2']:.3f}  RMSE={per_gal[gname]['rmse']:.3f}")

    metrics = {
        "r2": float(r2), "rmse": float(rmse),
        "pearson_r": float(r), "pearson_p": float(p),
        "n_test": int(len(y_te)),
        "best_iteration": int(booster.best_iteration),
        "per_galaxy": per_gal,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Feature importance
    fi = booster.get_score(importance_type="gain")
    fi_df = pd.DataFrame({"feature": list(fi.keys()), "gain": list(fi.values())})
    fi_df = fi_df.sort_values("gain", ascending=False)
    fi_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    # ---- Plots ----

    # Scatter: predicted vs true r_norm, coloured by galaxy
    fig, ax = plt.subplots(figsize=(6, 6))
    galaxies_present = np.unique(gal_te)
    cmap = plt.get_cmap("tab10")
    for i, gname in enumerate(galaxies_present):
        m = gal_te == gname
        ax.scatter(y_te[m], y_pred[m], s=4, alpha=0.4,
                   color=cmap(i % 10), label=gname)
    lim = max(y_te.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("True $r / r_{\\rm eff}$")
    ax.set_ylabel("Predicted $r / r_{\\rm eff}$")
    ax.set_title(f"Galaxy radius regression  $R^2={r2:.3f}$  $r={r:.3f}$")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_scatter_pred_vs_true.png", dpi=150)
    plt.close(fig)

    # Residuals by galaxy
    fig, ax = plt.subplots(figsize=(7, 4))
    resid = y_pred - y_te
    for i, gname in enumerate(galaxies_present):
        m = gal_te == gname
        ax.hist(resid[m], bins=40, histtype="step",
                label=f"{gname} (n={m.sum()})", color=cmap(i % 10))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Residual (pred − true) $r / r_{\\rm eff}$")
    ax.set_title("Radius prediction residuals by galaxy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_residuals_by_field.png", dpi=150)
    plt.close(fig)

    # Feature importance
    top = fi_df.head(16)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"][::-1], top["gain"][::-1])
    ax.set_xlabel("Gain")
    ax.set_title("Feature importance — galaxy radius regressor")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_feature_importance.png", dpi=150)
    plt.close(fig)

    print(f"\nOutputs -> {OUT_DIR}")
    print(f"Plots   -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
