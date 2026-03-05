"""
79_quasar_wise_locus_classifier.py
====================================

Binary XGBoost classifier: Gaia source lies in the WISE quasar locus
(label=1) vs not (label=0), trained on the 16D Gaia feature set.

Labels are derived from AllWISE W1/W2/W3 Vega magnitudes pulled from WSDB.
The selection box replicates the dense quasar blob in Fig. 5 of
arxiv:2512.08803 (Euclid Q1 quasar spectroscopy paper):

  W1 − W2 (Vega):  WISE_W12_LO < w1−w2 < WISE_W12_HI
  W2 − W3 (Vega):  WISE_W23_LO < w2−w3 < WISE_W23_HI

Sources outside the box with reliable photometry are used as negatives.
Sources with missing or flagged WISE photometry are dropped.

Scientific question: can Gaia quality-degradation features (RUWE, AEN, IPD,
blended transits, PM significance, parallax S/N) predict WISE quasar colours?
A high AUC means Gaia encodes AGN-related information independently of IR.

Pipeline:
  --pull_wsdb    Crossmatch ERO sources to AllWISE via WSDB, save wise_phot.csv
  --train        Load wise_phot.csv + 16D metadata, assign labels, train XGB

Usage:
  python 79_quasar_wise_locus_classifier.py --pull_wsdb
  python 79_quasar_wise_locus_classifier.py --train
  python 79_quasar_wise_locus_classifier.py --pull_wsdb --train

Outputs:
  output/ml_runs/quasar_wise_clf/
    wise_phot.csv              (source_id, ra, dec, w1, w2, w3, field_tag)
    model_quasar_wise.json
    metrics.json
    feature_importance.csv
  plots/ml_runs/quasar_wise_clf/
    01_wise_color_color.png    (reproduce Fig. 5 style with our sources)
    02_roc_pr.png
    03_feature_importance.png
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
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema import get_feature_cols, scaler_stem


# ======================================================================
# CONFIG
# ======================================================================

BASE = Path(__file__).resolve().parents[2]
DATASET_ROOT = BASE / "output" / "dataset_npz"
RUN_NAME = "quasar_wise_clf"
OUT_DIR  = BASE / "output" / "ml_runs" / RUN_NAME
PLOT_DIR = BASE / "plots"   / "ml_runs" / RUN_NAME
META_NAME   = "metadata_16d.csv"
FEATURE_SET = "16D"
WISE_CSV    = OUT_DIR / "wise_phot.csv"

# WSDB AllWISE table
WSDB_SCHEMA = "allwise"
WSDB_TABLE  = "main"
# Columns: w1mpro, w2mpro, w3mpro are Vega mags; w?snr are S/N; ext_flg is
# extended-source flag; cc_flags encodes contamination artefacts.
WISE_COLS = "ra, dec, w1mpro, w2mpro, w3mpro, w1snr, w2snr, w3snr, ext_flg, cc_flags"

# Match radius for AllWISE crossmatch (arcsec).
# AllWISE positional accuracy ~0.5", so 2" is generous but avoids misses.
WISE_MATCH_RADIUS_ARCSEC = 2.0

# Minimum S/N per band to use photometry
WISE_MIN_SNR = 3.0

# Quasar locus selection box in Vega magnitudes (from Fig. 5 visual inspection)
# W1 - W2 vs W2 - W3; adjust these to refine purity/completeness trade-off.
WISE_W12_LO =  0.3   # W1 - W2 lower bound
WISE_W12_HI =  2.0   # W1 - W2 upper bound
WISE_W23_LO =  1.0   # W2 - W3 lower bound
WISE_W23_HI =  4.5   # W2 - W3 upper bound

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
# WSDB PULL — AllWISE crossmatch by sky position
# ======================================================================

def pull_wise_photometry(df_sources: pd.DataFrame) -> pd.DataFrame:
    """
    For each source (ra, dec), find the nearest AllWISE match within
    WISE_MATCH_RADIUS_ARCSEC.  Queries are batched spatially via a
    temporary table upload to avoid per-row round-trips.

    Requires: WSDB_USER and WSDB_PASS environment variables.
    """
    try:
        import psycopg
    except ImportError:
        raise RuntimeError("psycopg not available. pip install psycopg[binary]")

    host = os.environ.get("WSDB_HOST", "wsdb.ast.cam.ac.uk")
    db   = os.environ.get("WSDB_DB",   "wsdb")
    user = os.environ.get("WSDB_USER", "")
    pw   = os.environ.get("WSDB_PASS", "")
    port = int(os.environ.get("WSDB_PORT", "5432"))
    if not user or not pw:
        raise RuntimeError("Set WSDB_USER and WSDB_PASS env variables.")

    r_deg = WISE_MATCH_RADIUS_ARCSEC / 3600.0
    results = []
    n_src = len(df_sources)

    with psycopg.connect(host=host, dbname=db, user=user, password=pw,
                         port=port, autocommit=True) as conn:

        CHUNK = 5_000
        for i in range(0, n_src, CHUNK):
            batch = df_sources.iloc[i : i + CHUNK]

            # Build VALUES with explicit PostgreSQL type casts on the first row.
            rows_sql = []
            for j, row in enumerate(batch.itertuples()):
                tag_esc = str(row.field_tag).replace("'", "''")
                if j == 0:
                    rows_sql.append(
                        f"({int(row.source_id)}::bigint, "
                        f"{float(row.ra)}::double precision, "
                        f"{float(row.dec)}::double precision, "
                        f"'{tag_esc}'::text)"
                    )
                else:
                    rows_sql.append(
                        f"({int(row.source_id)}, {float(row.ra)}, {float(row.dec)}, '{tag_esc}')"
                    )
            vals = ", ".join(rows_sql)

            q = f"""
                SELECT
                    s.source_id,
                    s.field_tag,
                    s.ra        AS gaia_ra,
                    s.dec       AS gaia_dec,
                    w.ra        AS wise_ra,
                    w.dec       AS wise_dec,
                    w.w1mpro, w.w2mpro, w.w3mpro,
                    w.w1snr, w.w2snr, w.w3snr,
                    w.ext_flg, w.cc_flags,
                    q3c_dist(s.ra, s.dec, w.ra, w.dec) * 3600 AS sep_arcsec
                FROM (VALUES {vals}) AS s(source_id, ra, dec, field_tag)
                JOIN LATERAL (
                    SELECT {WISE_COLS}
                    FROM {WSDB_SCHEMA}.{WSDB_TABLE} AS w2
                    WHERE q3c_radial_query(w2.ra, w2.dec, s.ra, s.dec, {r_deg})
                    ORDER BY q3c_dist(s.ra, s.dec, w2.ra, w2.dec) ASC
                    LIMIT 1
                ) w ON TRUE
            """
            rows = conn.execute(q).fetchall()
            if rows:
                df_batch = pd.DataFrame(rows, columns=[
                    "source_id", "field_tag", "gaia_ra", "gaia_dec",
                    "wise_ra", "wise_dec",
                    "w1mpro", "w2mpro", "w3mpro",
                    "w1snr", "w2snr", "w3snr",
                    "ext_flg", "cc_flags", "sep_arcsec",
                ])
                results.append(df_batch)
            if (i // CHUNK) % 5 == 0:
                print(f"  ... processed {min(i + CHUNK, n_src)} / {n_src}")

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


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


def assign_quasar_labels(df_wise: pd.DataFrame) -> pd.DataFrame:
    """
    Compute W1-W2 and W2-W3 colours, apply S/N cuts and selection box.
    Returns df with 'label' column (1=quasar locus, 0=outside locus).
    Rows with insufficient S/N are dropped.
    """
    df = df_wise.copy()
    for col in ["w1mpro", "w2mpro", "w3mpro", "w1snr", "w2snr", "w3snr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # S/N quality cut
    snr_ok = (
        (df["w1snr"] >= WISE_MIN_SNR) &
        (df["w2snr"] >= WISE_MIN_SNR) &
        (df["w3snr"] >= WISE_MIN_SNR)
    )
    df = df[snr_ok].copy()

    df["w1_w2"] = df["w1mpro"] - df["w2mpro"]
    df["w2_w3"] = df["w2mpro"] - df["w3mpro"]

    in_locus = (
        (df["w1_w2"] >= WISE_W12_LO) & (df["w1_w2"] <= WISE_W12_HI) &
        (df["w2_w3"] >= WISE_W23_LO) & (df["w2_w3"] <= WISE_W23_HI)
    )
    df["label"] = in_locus.astype(int)
    return df


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull_wsdb", action="store_true",
                    help="Pull AllWISE photometry from WSDB.")
    ap.add_argument("--train", action="store_true",
                    help="Train classifier (requires wise_phot.csv).")
    args = ap.parse_args()

    if not args.pull_wsdb and not args.train:
        ap.print_help()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols(FEATURE_SET)

    # ---- Step 1: Pull AllWISE ----
    if args.pull_wsdb:
        print("Loading metadata to get source positions...")
        df_meta = load_all_fields(feature_cols)
        needed = ["source_id", "ra", "dec", "field_tag"]
        df_src = df_meta[needed].dropna(subset=["ra", "dec"]).drop_duplicates("source_id").copy()
        df_src["source_id"] = pd.to_numeric(df_src["source_id"], errors="coerce").astype(np.int64)
        print(f"  Sources to match: {len(df_src)}")

        df_wise = pull_wise_photometry(df_src)
        df_wise.to_csv(WISE_CSV, index=False)
        print(f"  WISE matches saved -> {WISE_CSV}  ({len(df_wise)} rows)")
        n_matched = df_wise["source_id"].nunique()
        print(f"  Unique sources with WISE match: {n_matched} / {len(df_src)}")

    # ---- Step 2: Train ----
    if args.train:
        if not WISE_CSV.exists():
            raise RuntimeError(f"WISE photometry not found at {WISE_CSV}. Run --pull_wsdb first.")

        feat_min, feat_iqr = load_scaler(feature_cols)

        print("Loading WISE photometry and assigning labels...")
        df_wise_raw = pd.read_csv(WISE_CSV, low_memory=False)
        df_wise = assign_quasar_labels(df_wise_raw)

        n_qso  = int(df_wise["label"].sum())
        n_tot  = len(df_wise)
        print(f"  Quasar locus (label=1): {n_qso}")
        print(f"  Outside locus (label=0): {n_tot - n_qso}")
        print(f"  Total with S/N cuts:     {n_tot}")

        print("Merging with Gaia 16D features...")
        df_meta = load_all_fields(feature_cols)
        df_meta["source_id"] = pd.to_numeric(df_meta["source_id"], errors="coerce").astype("Int64")
        df_wise["source_id"] = pd.to_numeric(df_wise["source_id"], errors="coerce").astype("Int64")

        df = df_wise[["source_id", "label", "w1_w2", "w2_w3", "field_tag"]].merge(
            df_meta[["source_id", "split_code"] + feature_cols],
            on="source_id", how="inner",
        )
        df = df.dropna(subset=feature_cols).copy()
        print(f"  Rows after merge + NaN drop: {len(df)}")

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

        booster.save_model(str(OUT_DIR / "model_quasar_wise.json"))

        prob_te = booster.predict(dtest)
        auc = roc_auc_score(y_te, prob_te)
        ap  = average_precision_score(y_te, prob_te)
        print(f"\nTest AUC={auc:.4f}  AP={ap:.4f}")

        metrics = {
            "auc": auc, "ap": ap,
            "n_test": int(len(y_te)),
            "n_quasar_test": int(y_te.sum()),
            "selection_box": {
                "w1_w2": [WISE_W12_LO, WISE_W12_HI],
                "w2_w3": [WISE_W23_LO, WISE_W23_HI],
            },
            "best_iteration": booster.best_iteration,
        }
        (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

        fi = booster.get_score(importance_type="gain")
        fi_df = pd.DataFrame({"feature": list(fi.keys()), "gain": list(fi.values())})
        fi_df = fi_df.sort_values("gain", ascending=False)
        fi_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)

        # ---- Plots ----

        # Fig. 5 style: W1-W2 vs W2-W3 with quasar locus box
        df_plot = df[["w1_w2", "w2_w3", "label"]].dropna()
        fig, ax = plt.subplots(figsize=(7, 6))
        outside = df_plot["label"] == 0
        inside  = df_plot["label"] == 1
        ax.scatter(df_plot.loc[outside, "w2_w3"], df_plot.loc[outside, "w1_w2"],
                   s=3, alpha=0.2, color="gray", label="Outside locus")
        ax.scatter(df_plot.loc[inside, "w2_w3"], df_plot.loc[inside, "w1_w2"],
                   s=8, alpha=0.6, color="tab:orange", label="Quasar locus")
        # Draw selection box
        rect_x = [WISE_W23_LO, WISE_W23_HI, WISE_W23_HI, WISE_W23_LO, WISE_W23_LO]
        rect_y = [WISE_W12_LO, WISE_W12_LO, WISE_W12_HI, WISE_W12_HI, WISE_W12_LO]
        ax.plot(rect_x, rect_y, "b--", lw=1.2, label="Selection box")
        ax.set_xlabel("W2 − W3 [Vega mag]")
        ax.set_ylabel("W1 − W2 [Vega mag]")
        ax.set_title("WISE colour-colour: quasar locus selection (ERO sources)")
        ax.legend(markerscale=3, fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "01_wise_color_color.png", dpi=150)
        plt.close(fig)

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
        fig.savefig(PLOT_DIR / "02_roc_pr.png", dpi=150)
        plt.close(fig)

        # Feature importance
        top = fi_df.head(16)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top["feature"][::-1], top["gain"][::-1])
        ax.set_xlabel("Gain")
        ax.set_title("Feature importance — WISE quasar locus classifier")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "03_feature_importance.png", dpi=150)
        plt.close(fig)

        print(f"\nOutputs -> {OUT_DIR}")
        print(f"Plots   -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
