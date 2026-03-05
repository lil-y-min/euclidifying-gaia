"""
78_compact_galaxy_classifier.py
=================================

Binary XGBoost classifier: Gaia source is a compact/dwarf galaxy (label=1)
vs. stellar point source (label=0), using Gaia 16D features.

Labels are derived from Gaia DR3's own DSC (Discrete Source Classifier)
combined-model probabilities, pulled from WSDB:
  classprob_dsc_combmod_galaxy > GALAXY_THRESH  -> label=1 (galaxy)
  classprob_dsc_combmod_star   > STAR_THRESH    -> label=0 (star)
  middle zone                                   -> dropped (ambiguous)

This uses Gaia's own internal classifier as a silver label, then asks:
"Can we do better / can we characterise what Gaia encodes?" using only the
quality-degradation features (RUWE, AEN, IPD, blended transits, etc).

WSDB query target:
  gaiadr3.gaia_source  (columns: source_id, classprob_dsc_combmod_galaxy,
                                  classprob_dsc_combmod_star)

Run the WSDB pull first:
  python 78_compact_galaxy_classifier.py --pull_wsdb

Then train:
  python 78_compact_galaxy_classifier.py --train

Or both in one go:
  python 78_compact_galaxy_classifier.py --pull_wsdb --train

Outputs:
  output/ml_runs/compact_galaxy_clf/
    dsc_labels.csv           (source_id, label, gal_prob, star_prob)
    model_compact_galaxy.json
    metrics.json
    feature_importance.csv
  plots/ml_runs/compact_galaxy_clf/
    01_roc_pr.png
    02_feature_importance.png
    03_prob_calibration.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    calibration_curve,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema import get_feature_cols, scaler_stem


# ======================================================================
# CONFIG
# ======================================================================

BASE = Path(__file__).resolve().parents[2]
DATASET_ROOT = BASE / "output" / "dataset_npz"
RUN_NAME = "compact_galaxy_clf"
OUT_DIR = BASE / "output" / "ml_runs" / RUN_NAME
PLOT_DIR = BASE / "plots" / "ml_runs" / RUN_NAME
META_NAME = "metadata_16d.csv"
FEATURE_SET = "16D"
DSC_LABELS_CSV = OUT_DIR / "dsc_labels.csv"

# Label thresholds on Gaia DSC probabilities
GALAXY_THRESH = 0.80   # classprob_dsc_combmod_galaxy > this -> label=1
STAR_THRESH   = 0.95   # classprob_dsc_combmod_star   > this -> label=0

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
# WSDB PULL
# ======================================================================

def pull_dsc_labels(source_ids: np.ndarray) -> pd.DataFrame:
    """
    Query WSDB gaiadr3.gaia_source for DSC classifier probabilities.
    Auth from environment: WSDB_HOST, WSDB_DB, WSDB_USER, WSDB_PASS, WSDB_PORT.
    """
    try:
        import psycopg
    except ImportError:
        raise RuntimeError("psycopg not available. Install with: pip install psycopg[binary]")

    host = os.environ.get("WSDB_HOST", "wsdb.ast.cam.ac.uk")
    db   = os.environ.get("WSDB_DB",   "wsdb")
    user = os.environ.get("WSDB_USER", "")
    pw   = os.environ.get("WSDB_PASS", "")
    port = int(os.environ.get("WSDB_PORT", "5432"))
    if not user or not pw:
        raise RuntimeError("Set WSDB_USER and WSDB_PASS environment variables.")

    ids_unique = np.unique(source_ids).tolist()
    print(f"  Querying WSDB for {len(ids_unique)} source_ids...")

    CHUNK = 10_000
    frames = []
    with psycopg.connect(host=host, dbname=db, user=user, password=pw,
                         port=port, autocommit=True) as conn:
        for i in range(0, len(ids_unique), CHUNK):
            chunk = ids_unique[i : i + CHUNK]
            ids_str = ",".join(str(x) for x in chunk)
            q = (
                "SELECT source_id, "
                "classprob_dsc_combmod_galaxy, "
                "classprob_dsc_combmod_star "
                f"FROM gaiadr3.gaia_source "
                f"WHERE source_id IN ({ids_str})"
            )
            rows = conn.execute(q).fetchall()
            frames.append(pd.DataFrame(rows, columns=[
                "source_id", "classprob_dsc_combmod_galaxy", "classprob_dsc_combmod_star"
            ]))
            if (i // CHUNK) % 5 == 0:
                print(f"  ... fetched {i + len(chunk)} / {len(ids_unique)}")

    return pd.concat(frames, ignore_index=True)


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


def load_all_fields(feature_cols: list[str]) -> pd.DataFrame:
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


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull_wsdb", action="store_true",
                    help="Pull DSC probabilities from WSDB and save dsc_labels.csv")
    ap.add_argument("--train", action="store_true",
                    help="Train XGB classifier (requires dsc_labels.csv)")
    args = ap.parse_args()

    if not args.pull_wsdb and not args.train:
        ap.print_help()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols(FEATURE_SET)

    # ---- Step 1: Pull WSDB ----
    if args.pull_wsdb:
        print("Loading all metadata to collect source_ids...")
        df_meta = load_all_fields(feature_cols)
        src_ids = pd.to_numeric(df_meta["source_id"], errors="coerce").dropna().astype(np.int64).to_numpy()
        print(f"  Unique source_ids: {len(np.unique(src_ids))}")

        dsc = pull_dsc_labels(src_ids)
        dsc.to_csv(DSC_LABELS_CSV, index=False)
        print(f"  Saved DSC labels -> {DSC_LABELS_CSV}  ({len(dsc)} rows)")

    # ---- Step 2: Train ----
    if args.train:
        if not DSC_LABELS_CSV.exists():
            raise RuntimeError(f"DSC labels not found at {DSC_LABELS_CSV}. Run --pull_wsdb first.")

        feat_min, feat_iqr = load_scaler(feature_cols)

        print("Loading metadata and DSC labels...")
        df_meta = load_all_fields(feature_cols)
        dsc = pd.read_csv(DSC_LABELS_CSV, low_memory=False)
        dsc["source_id"] = pd.to_numeric(dsc["source_id"], errors="coerce").astype("Int64")
        df_meta["source_id"] = pd.to_numeric(df_meta["source_id"], errors="coerce").astype("Int64")

        df = df_meta.merge(dsc, on="source_id", how="inner")
        print(f"  Rows after merge: {len(df)}")

        gal_p  = pd.to_numeric(df["classprob_dsc_combmod_galaxy"], errors="coerce")
        star_p = pd.to_numeric(df["classprob_dsc_combmod_star"],   errors="coerce")

        is_galaxy = gal_p  > GALAXY_THRESH
        is_star   = star_p > STAR_THRESH
        keep = is_galaxy | is_star
        df = df[keep].copy()

        df["label"] = is_galaxy[keep].astype(int)
        print(f"  Galaxy (label=1): {df['label'].sum()}")
        print(f"  Star   (label=0): {(df['label']==0).sum()}")
        print(f"  Total kept: {len(df)}")

        df = df.dropna(subset=feature_cols).copy()
        X = df[feature_cols].to_numpy(dtype=np.float32)
        X = (X - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)
        y = df["label"].to_numpy(dtype=np.float32)
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

        booster.save_model(str(OUT_DIR / "model_compact_galaxy.json"))

        # ---- Evaluate ----
        prob_te = booster.predict(dtest)
        auc = roc_auc_score(y_te, prob_te)
        ap  = average_precision_score(y_te, prob_te)
        print(f"\nTest AUC={auc:.4f}  AP={ap:.4f}")

        metrics = {
            "auc": auc, "ap": ap,
            "n_test": int(len(y_te)),
            "n_galaxy_test": int(y_te.sum()),
            "galaxy_thresh": GALAXY_THRESH,
            "star_thresh": STAR_THRESH,
            "best_iteration": booster.best_iteration,
        }
        (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

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
        ax.set_title("Feature importance — compact galaxy classifier")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "02_feature_importance.png", dpi=150)
        plt.close(fig)

        # Calibration: XGB prob vs DSC gal_prob (on test set)
        gal_p_te = pd.to_numeric(
            df["classprob_dsc_combmod_galaxy"].iloc[splits == 2], errors="coerce"
        ).to_numpy()
        valid_cal = np.isfinite(gal_p_te)
        if valid_cal.sum() > 50:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(gal_p_te[valid_cal], prob_te[valid_cal], s=3, alpha=0.3)
            ax.set_xlabel("Gaia DSC galaxy probability")
            ax.set_ylabel("XGB predicted galaxy probability")
            ax.set_title("XGB vs Gaia DSC probability (test set)")
            fig.tight_layout()
            fig.savefig(PLOT_DIR / "03_prob_calibration.png", dpi=150)
            plt.close(fig)

        print(f"\nOutputs -> {OUT_DIR}")
        print(f"Plots   -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
