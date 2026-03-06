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
WISE_W12_LO =  0.6   # W1 - W2 lower bound (Vega)
WISE_W12_HI =  1.6   # W1 - W2 upper bound (Vega)
WISE_W23_LO =  2.0   # W2 - W3 lower bound (Vega)
WISE_W23_HI =  4.2   # W2 - W3 upper bound (Vega)

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
        import warnings

        label_arr  = df["label"].to_numpy(dtype=int)
        # Colours used throughout: quasar=orange, non-quasar=steelblue
        C_QSO  = "#e8671b"
        C_NON  = "#4b8ec4"
        C_GREY = "#aaaaaa"

        # ------------------------------------------------------------------
        # 01  WISE colour-colour diagram (Fig. 5 style)
        # ------------------------------------------------------------------
        df_cc = df[["w1_w2", "w2_w3", "label"]].dropna()
        fig, ax = plt.subplots(figsize=(7, 6))
        mask_out = df_cc["label"] == 0
        mask_in  = df_cc["label"] == 1
        ax.scatter(df_cc.loc[mask_out, "w2_w3"], df_cc.loc[mask_out, "w1_w2"],
                   s=4, alpha=0.25, color=C_GREY, label=f"Outside locus  (n={mask_out.sum()})")
        ax.scatter(df_cc.loc[mask_in,  "w2_w3"], df_cc.loc[mask_in,  "w1_w2"],
                   s=10, alpha=0.7, color=C_QSO,  label=f"Quasar locus   (n={mask_in.sum()})")
        rect_x = [WISE_W23_LO, WISE_W23_HI, WISE_W23_HI, WISE_W23_LO, WISE_W23_LO]
        rect_y = [WISE_W12_LO, WISE_W12_LO, WISE_W12_HI, WISE_W12_HI, WISE_W12_LO]
        ax.plot(rect_x, rect_y, "b--", lw=1.5, label="Selection box")
        ax.set_xlabel("W2 − W3 [Vega mag]", fontsize=12)
        ax.set_ylabel("W1 − W2 [Vega mag]", fontsize=12)
        ax.set_title("WISE colour-colour: ERO sources (AllWISE Vega)", fontsize=12)
        ax.legend(markerscale=2.5, fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "01_wise_color_color.png", dpi=150)
        plt.close(fig)
        print("  saved 01_wise_color_color.png")

        # ------------------------------------------------------------------
        # 02  ROC + PR curves
        # ------------------------------------------------------------------
        fpr, tpr, _ = roc_curve(y_te, prob_te)
        prec, rec, _ = precision_recall_curve(y_te, prob_te)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(fpr, tpr, color=C_QSO, lw=1.8)
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
        axes[0].set_title(f"ROC  AUC = {auc:.3f}")
        axes[1].plot(rec, prec, color=C_QSO, lw=1.8)
        axes[1].axhline(y_te.mean(), color="k", linestyle="--", lw=0.8,
                        label=f"Baseline (prevalence={y_te.mean():.3f})")
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        axes[1].set_title(f"Precision-Recall  AP = {ap:.3f}")
        axes[1].legend(fontsize=8)
        fig.suptitle("Quasar locus classifier — test set", fontsize=11)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "02_roc_pr.png", dpi=150)
        plt.close(fig)
        print("  saved 02_roc_pr.png")

        # ------------------------------------------------------------------
        # 03  Feature importance (gain)
        # ------------------------------------------------------------------
        top = fi_df.head(16)
        clean_names = [f.replace("feat_", "") for f in top["feature"]]
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(clean_names[::-1], top["gain"].values[::-1], color=C_QSO)
        ax.set_xlabel("XGBoost gain")
        ax.set_title("Feature importance — WISE quasar locus classifier")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "03_feature_importance.png", dpi=150)
        plt.close(fig)
        print("  saved 03_feature_importance.png")

        # ------------------------------------------------------------------
        # 04  Feature distributions: quasar vs non-quasar (test set)
        # ------------------------------------------------------------------
        X_te_df = pd.DataFrame(
            X_te / np.where(feat_iqr > 0, 1.0, 1.0),  # already scaled
            columns=feature_cols
        )
        short = [f.replace("feat_", "") for f in feature_cols]
        n_feat = len(feature_cols)
        ncols = 4
        nrows = int(np.ceil(n_feat / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.2))
        axes = axes.flatten()
        qso_mask  = y_te == 1
        non_mask  = y_te == 0
        for k, (fc, name) in enumerate(zip(feature_cols, short)):
            ax = axes[k]
            vals_q = X_te_df[fc].values[qso_mask]
            vals_n = X_te_df[fc].values[non_mask]
            lo = np.nanpercentile(np.concatenate([vals_q, vals_n]), 1)
            hi = np.nanpercentile(np.concatenate([vals_q, vals_n]), 99)
            bins = np.linspace(lo, hi, 35)
            ax.hist(vals_n, bins=bins, density=True, alpha=0.5, color=C_NON, label="non-QSO")
            ax.hist(vals_q, bins=bins, density=True, alpha=0.7, color=C_QSO, label="QSO locus")
            ax.set_title(name, fontsize=8)
            ax.tick_params(labelsize=6)
        for k in range(n_feat, len(axes)):
            axes[k].set_visible(False)
        axes[0].legend(fontsize=7)
        fig.suptitle("Gaia 16D feature distributions: QSO locus vs non-QSO (test, scaled)", fontsize=10)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "04_feature_distributions.png", dpi=150)
        plt.close(fig)
        print("  saved 04_feature_distributions.png")

        # ------------------------------------------------------------------
        # 05  XGB predicted probability distribution
        # ------------------------------------------------------------------
        prob_tr_full = booster.predict(dtrain)
        fig, ax = plt.subplots(figsize=(7, 4))
        bins_p = np.linspace(0, 1, 50)
        ax.hist(prob_te[y_te == 0], bins=bins_p, density=True, alpha=0.5,
                color=C_NON, label="non-QSO (test)")
        ax.hist(prob_te[y_te == 1], bins=bins_p, density=True, alpha=0.7,
                color=C_QSO, label="QSO locus (test)")
        ax.set_xlabel("XGB predicted probability")
        ax.set_ylabel("Density")
        ax.set_title("Score distribution — quasar locus classifier")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "05_score_distribution.png", dpi=150)
        plt.close(fig)
        print("  saved 05_score_distribution.png")

        # ------------------------------------------------------------------
        # Save XGB scores for all sources (for UMAP overlay and future reuse)
        # ------------------------------------------------------------------
        X_all = df[feature_cols].to_numpy(dtype=np.float32)
        X_all_scaled = (X_all - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)
        prob_all = booster.predict(xgb.DMatrix(X_all_scaled, feature_names=feature_cols))
        scores_df = pd.DataFrame({
            "source_id": df["source_id"].to_numpy(),
            "label":     label_arr,
            "xgb_score": prob_all,
            "w1_w2":     df["w1_w2"].to_numpy(),
            "w2_w3":     df["w2_w3"].to_numpy(),
        })
        scores_df.to_csv(OUT_DIR / "scores_all.csv", index=False)
        print(f"  saved scores_all.csv ({len(scores_df)} rows)")

        # ------------------------------------------------------------------
        # 06  Overlay on existing 16D UMAP
        # ------------------------------------------------------------------
        EMB_16D = (BASE / "output" / "experiments" / "embeddings"
                   / "umap16d_manualv8_filtered" / "embedding_umap.csv")
        print(f"  loading 16D UMAP from {EMB_16D.name}...")
        umap16 = pd.read_csv(EMB_16D, usecols=["source_id", "x", "y"], low_memory=False)
        umap16["source_id"] = pd.to_numeric(umap16["source_id"], errors="coerce").astype("Int64")
        scores_df["source_id"] = scores_df["source_id"].astype("Int64")
        merged16 = umap16.merge(scores_df, on="source_id", how="left")

        in_locus = merged16["label"] == 1
        has_wise = merged16["label"].notna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # (a) label
        ax = axes[0]
        ax.scatter(merged16.loc[~in_locus, "x"], merged16.loc[~in_locus, "y"],
                   s=0.5, alpha=0.06, color="#555555", rasterized=True)
        ax.scatter(merged16.loc[in_locus, "x"], merged16.loc[in_locus, "y"],
                   s=25, alpha=1.0, color="#ff007f", rasterized=True, zorder=5,
                   edgecolors="white", linewidths=0.4,
                   label=f"WISE QSO locus (n={in_locus.sum()})")
        ax.set_title("(a) WISE quasar locus label", fontsize=10)
        ax.legend(markerscale=1.5, fontsize=8, loc="best",
                  framealpha=0.85, edgecolor="none")
        ax.set_xticks([]); ax.set_yticks([])

        # (b) XGB score — grey for unmatched, plasma for WISE-matched
        ax = axes[1]
        bg = merged16["xgb_score"].isna()
        ax.scatter(merged16.loc[bg, "x"], merged16.loc[bg, "y"],
                   s=0.5, alpha=0.06, color="#555555", rasterized=True)
        sc = ax.scatter(merged16.loc[~bg, "x"], merged16.loc[~bg, "y"],
                        s=3, c=merged16.loc[~bg, "xgb_score"],
                        cmap="RdPu", alpha=0.8, vmin=0, vmax=1,
                        rasterized=True, zorder=4)
        plt.colorbar(sc, ax=ax, label="XGB P(quasar)")
        ax.set_title("(b) XGB predicted P(quasar)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle("16D Gaia UMAP — WISE quasar locus overlay", fontsize=12)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "06_umap16d_quasar_overlay.png", dpi=150)
        plt.close(fig)
        print("  saved 06_umap16d_quasar_overlay.png")

        # ------------------------------------------------------------------
        # 07  Overlay on existing pixel UMAP
        # ------------------------------------------------------------------
        EMB_PIX = (BASE / "output" / "experiments" / "embeddings"
                   / "double_stars_pixels"
                   / "pixels_umap_standard_manualv8_filtered"
                   / "umap_standard" / "embedding_umap.csv")
        print(f"  loading pixel UMAP from {EMB_PIX.parent.parent.name}/...")
        umap_pix = pd.read_csv(EMB_PIX, usecols=["source_id", "x", "y"], low_memory=False)
        umap_pix["source_id"] = pd.to_numeric(umap_pix["source_id"], errors="coerce").astype("Int64")
        merged_pix = umap_pix.merge(scores_df, on="source_id", how="left")

        in_locus_p = merged_pix["label"] == 1

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(merged_pix.loc[~in_locus_p, "x"], merged_pix.loc[~in_locus_p, "y"],
                   s=0.5, alpha=0.06, color="#555555", rasterized=True)
        ax.scatter(merged_pix.loc[in_locus_p, "x"], merged_pix.loc[in_locus_p, "y"],
                   s=25, alpha=1.0, color="#ff007f", rasterized=True, zorder=5,
                   edgecolors="white", linewidths=0.4,
                   label=f"WISE QSO locus (n={in_locus_p.sum()})")
        ax.set_title("(a) WISE quasar locus label", fontsize=10)
        ax.legend(markerscale=1.5, fontsize=8, loc="best",
                  framealpha=0.85, edgecolor="none")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1]
        bg_p = merged_pix["xgb_score"].isna()
        ax.scatter(merged_pix.loc[bg_p, "x"], merged_pix.loc[bg_p, "y"],
                   s=0.5, alpha=0.06, color="#555555", rasterized=True)
        sc = ax.scatter(merged_pix.loc[~bg_p, "x"], merged_pix.loc[~bg_p, "y"],
                        s=3, c=merged_pix.loc[~bg_p, "xgb_score"],
                        cmap="RdPu", alpha=0.8, vmin=0, vmax=1,
                        rasterized=True, zorder=4)
        plt.colorbar(sc, ax=ax, label="XGB P(quasar)")
        ax.set_title("(b) XGB predicted P(quasar)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle("Pixel UMAP — WISE quasar locus overlay", fontsize=12)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "07_umap_pixel_quasar_overlay.png", dpi=150)
        plt.close(fig)
        print("  saved 07_umap_pixel_quasar_overlay.png")

        print(f"\nOutputs -> {OUT_DIR}")
        print(f"Plots   -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
