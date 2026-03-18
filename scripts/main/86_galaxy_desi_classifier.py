"""
86_galaxy_desi_classifier.py
=============================

Binary XGBoost classifier: Gaia source is a DESI-confirmed galaxy (label=1)
vs everything else (label=0), using 15D Gaia features.

Ground truth comes from two DESI × Euclid crossmatched files:
  data/with_desi/desi_euclid_ero_galaxies.fits   (1,844 galaxies, ERO fields)
  data/with_desi/desi_euclid_q1_galaxies.fits    (44,667 galaxies, Q1 fields)

Steps:
  --pull_wsdb   Query WSDB: for each DESI galaxy position, pull the nearest
                Gaia DR3 source within WSDB_RADIUS_ARCSEC.
                Covers both ERO (1,844 positions) and Q1 (44,667 positions).
                Saves:
                  desi_gaia_direct_ero.csv   — DESI ERO galaxies with Gaia features
                  desi_gaia_direct_q1.csv    — DESI Q1 galaxies with Gaia features

  --train       Build labelled dataset and train XGB galaxy classifier.
                Positives:  desi_gaia_direct_ero.csv
                Negatives:  ERO metadata sources whose source_id is not in positives
                Transfer:   apply to desi_gaia_direct_q1 + q1_gaia_features.csv

Outputs:
  output/ml_runs/galaxy_desi_clf/
    desi_gaia_direct_ero.csv
    desi_gaia_direct_q1.csv
    model_galaxy_desi.json
    metrics_ero.json
    metrics_q1_transfer.json
    feature_importance.csv
    scores_all.csv
  plots/ml_runs/galaxy_desi_clf/
    01_roc_pr.png
    02_feature_importance.png
    03_score_distribution.png
    04_umap16d_galaxy_overlay.png
    05_galaxy_stamp_catalog_p{N}.png

Usage:
  python 86_galaxy_desi_classifier.py --pull_wsdb
  python 86_galaxy_desi_classifier.py --train
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
from astropy.io import fits
from scipy.spatial import cKDTree
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema import get_feature_cols, scaler_stem


# ======================================================================
# CONFIG
# ======================================================================

BASE         = Path(__file__).resolve().parents[2]
DATASET_ROOT = BASE / "output" / "dataset_npz"
RUN_NAME     = "galaxy_desi_clf"
OUT_DIR      = BASE / "output" / "ml_runs" / RUN_NAME
PLOT_DIR     = BASE / "report" / "model_decision" / "20260311_desi_confirmed_galaxy_clf"
META_NAME    = "metadata_16d.csv"
FEATURE_SET  = "15D"

DESI_ERO_FITS = BASE / "data" / "with_desi" / "desi_euclid_ero_galaxies.fits"
DESI_Q1_FITS  = BASE / "data" / "with_desi" / "desi_euclid_q1_galaxies.fits"
Q1_GAIA_CSV   = BASE / "output" / "ml_runs" / "quasar_q1" / "q1_gaia_features.csv"

DIRECT_ERO_CSV = OUT_DIR / "desi_gaia_direct_ero.csv"
DIRECT_Q1_CSV  = OUT_DIR / "desi_gaia_direct_q1.csv"

EMB_16D = (BASE / "output" / "experiments" / "embeddings"
           / "umap16d_manualv8_filtered" / "embedding_umap.csv")

WSDB_RADIUS_ARCSEC = 2.0   # cone radius for Gaia match around each DESI galaxy
WSDB_CHUNK         = 500   # positions per SQL batch

# Stamp catalog
STAMP_PIX       = 20
STAMPS_PER_PAGE = 80
NCOLS           = 8

# XGB
NUM_BOOST_ROUND = 1000
EARLY_STOP      = 50
PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
    "seed":             123,
    "verbosity":        0,
}

# Visual
C_GAL  = "#2ecc71"
C_GREY = "#555555"

# Raw Gaia columns to pull from WSDB
GAIA_RAW_COLS = [
    "source_id",
    "ra", "dec",
    "phot_g_mean_mag",
    "phot_g_mean_flux_over_error",
    "ruwe",
    "astrometric_excess_noise",
    "astrometric_excess_noise_sig",
    "parallax", "parallax_error",
    "visibility_periods_used",
    "ipd_frac_multi_peak",
    "ipd_gof_harmonic_amplitude",
    "ipd_gof_harmonic_phase",
    "ipd_frac_odd_win",
    "phot_bp_rp_excess_factor", "bp_rp",
    "pmra", "pmdec", "pmra_error", "pmdec_error",
    "phot_bp_n_contaminated_transits", "phot_bp_n_blended_transits",
    "phot_rp_n_contaminated_transits", "phot_rp_n_blended_transits",
]


# ======================================================================
# HELPERS
# ======================================================================

def native(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    return a.astype(a.dtype.newbyteorder("="), copy=False)


def load_scaler(feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    stem = scaler_stem(FEATURE_SET)
    npz  = BASE / "output" / "scalers" / f"{stem}.npz"
    d    = np.load(npz, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    y_min = d["y_min"].astype(np.float32)
    y_iqr = d["y_iqr"].astype(np.float32)
    idx   = np.array([names.index(c) for c in feature_cols], dtype=int)
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
    full = pd.concat(frames, ignore_index=True)
    return full.dropna(subset=feature_cols).reset_index(drop=True)


def load_desi_fits(path: Path) -> pd.DataFrame:
    print(f"  Reading {path.name} ...")
    with fits.open(path, memmap=True) as hdul:
        d = hdul[1].data
        df = {col: native(d[col]) for col in d.names}
    df = pd.DataFrame(df)
    print(f"  Loaded {len(df):,} rows")
    return df


def compute_gaia_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 15D feat_* columns from raw Gaia DR3 quantities."""
    df = df.copy()
    snr = pd.to_numeric(df.get("phot_g_mean_flux_over_error", np.nan), errors="coerce")
    df["feat_log10_snr"] = np.log10(np.maximum(snr, 1e-6))
    df["feat_ruwe"] = pd.to_numeric(df.get("ruwe", np.nan), errors="coerce")
    df["feat_astrometric_excess_noise"] = pd.to_numeric(
        df.get("astrometric_excess_noise", np.nan), errors="coerce")
    plx   = pd.to_numeric(df.get("parallax",       np.nan), errors="coerce")
    plxe  = pd.to_numeric(df.get("parallax_error", np.nan), errors="coerce")
    # NaN → 0: no parallax solution means SNR = 0 (physically correct for extragalactic sources)
    df["feat_parallax_over_error"] = (plx / plxe.replace(0, np.nan)).fillna(0.0)
    df["feat_ipd_frac_multi_peak"] = pd.to_numeric(
        df.get("ipd_frac_multi_peak", np.nan), errors="coerce")
    C_exc = pd.to_numeric(df.get("phot_bp_rp_excess_factor", np.nan), errors="coerce")
    bp_rp = pd.to_numeric(df.get("bp_rp", np.nan), errors="coerce")
    poly  = 1.154360 + 0.033772 * bp_rp + 0.032277 * bp_rp**2
    df["feat_c_star"] = C_exc - poly
    pmra  = pd.to_numeric(df.get("pmra",        np.nan), errors="coerce")
    pmdec = pd.to_numeric(df.get("pmdec",       np.nan), errors="coerce")
    pmrae = pd.to_numeric(df.get("pmra_error",  np.nan), errors="coerce")
    pmdece= pd.to_numeric(df.get("pmdec_error", np.nan), errors="coerce")
    # NaN → 0: no PM solution means significance = 0 (physically correct for extragalactic sources)
    df["feat_pm_significance"] = np.sqrt(
        (pmra / pmrae.replace(0, np.nan))**2 +
        (pmdec / pmdece.replace(0, np.nan))**2
    ).fillna(0.0)
    for col in ["astrometric_excess_noise_sig", "ipd_gof_harmonic_amplitude",
                "ipd_gof_harmonic_phase", "ipd_frac_odd_win",
                "phot_bp_n_contaminated_transits", "phot_bp_n_blended_transits",
                "phot_rp_n_contaminated_transits", "phot_rp_n_blended_transits"]:
        df["feat_" + col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
    return df


def wsdb_connect():
    import psycopg
    return psycopg.connect(
        host    =os.environ.get("WSDB_HOST", "wsdb.ast.cam.ac.uk"),
        dbname  =os.environ.get("WSDB_DB",   "wsdb"),
        user    =os.environ.get("WSDB_USER",  ""),
        password=os.environ.get("WSDB_PASS",  ""),
        port    =int(os.environ.get("WSDB_PORT", "5432")),
        autocommit=True,
    )


# ======================================================================
# STEP 1: WSDB PULL
# ======================================================================

def pull_gaia_for_positions(
    conn,
    ra_arr: np.ndarray,
    dec_arr: np.ndarray,
    gal_ids: np.ndarray,
    label: str,
) -> pd.DataFrame:
    """
    For each (ra, dec) position, find the nearest Gaia DR3 source within
    WSDB_RADIUS_ARCSEC using a batched LATERAL cone search.
    Returns one row per matched galaxy (nearest Gaia source per position).
    """
    radius_deg = WSDB_RADIUS_ARCSEC / 3600.0
    n          = len(ra_arr)
    all_rows   = []

    # Outer SELECT references the LATERAL alias "g"
    outer_cols_sql = ", ".join(f"g.{c}" for c in GAIA_RAW_COLS)
    # Inner SELECT uses bare column names (no table prefix needed inside LATERAL)
    inner_cols_sql = ", ".join(GAIA_RAW_COLS)

    print(f"  [{label}] Querying WSDB for {n:,} positions "
          f"(radius={WSDB_RADIUS_ARCSEC}\", chunk={WSDB_CHUNK}) ...")

    for i in range(0, n, WSDB_CHUNK):
        i1      = min(i + WSDB_CHUNK, n)
        batch_ra  = ra_arr[i:i1]
        batch_dec = dec_arr[i:i1]
        batch_ids = gal_ids[i:i1]

        # Build VALUES clause: (gal_id, ra, dec)
        rows_sql = []
        for j in range(len(batch_ra)):
            if j == 0:
                rows_sql.append(
                    f"({int(batch_ids[j])}::bigint, "
                    f"{float(batch_ra[j])}::double precision, "
                    f"{float(batch_dec[j])}::double precision)"
                )
            else:
                rows_sql.append(
                    f"({int(batch_ids[j])}, "
                    f"{float(batch_ra[j])}, "
                    f"{float(batch_dec[j])})"
                )
        vals = ",\n        ".join(rows_sql)

        q = f"""
            WITH positions(gal_id, gal_ra, gal_dec) AS (
                VALUES {vals}
            )
            SELECT
                p.gal_id,
                q3c_dist(g.ra, g.dec, p.gal_ra, p.gal_dec) * 3600.0 AS sep_arcsec,
                {outer_cols_sql}
            FROM positions p
            CROSS JOIN LATERAL (
                SELECT {inner_cols_sql}
                FROM gaia_dr3.gaia_source
                WHERE q3c_radial_query(ra, dec, p.gal_ra, p.gal_dec, {radius_deg})
                ORDER BY q3c_dist(ra, dec, p.gal_ra, p.gal_dec)
                LIMIT 1
            ) g
        """
        rows = conn.execute(q).fetchall()
        all_rows.extend(rows)

        if (i // WSDB_CHUNK) % 10 == 0:
            print(f"    chunk {i1:,}/{n:,} — cumulative matches: {len(all_rows):,}")

    col_names = ["gal_id", "sep_arcsec"] + GAIA_RAW_COLS
    df = pd.DataFrame(all_rows, columns=col_names)
    print(f"  [{label}] Total matched: {len(df):,} / {n:,} positions")
    return df


def pull_wsdb() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = wsdb_connect()

    # ---- ERO ----
    df_ero = load_desi_fits(DESI_ERO_FITS)
    ra_ero  = df_ero["euclid_ra"].values.astype(float)
    dec_ero = df_ero["euclid_dec"].values.astype(float)
    ids_ero = np.arange(len(df_ero), dtype=np.int64)   # internal row index as gal_id

    df_gaia_ero = pull_gaia_for_positions(conn, ra_ero, dec_ero, ids_ero, "ERO")
    df_gaia_ero = compute_gaia_features(df_gaia_ero)

    # Attach DESI redshift using gal_id (row index into df_ero)
    df_gaia_ero["desi_z"] = df_ero["desi_z"].values[df_gaia_ero["gal_id"].values]

    df_gaia_ero.to_csv(DIRECT_ERO_CSV, index=False)
    print(f"  Saved {len(df_gaia_ero):,} rows → {DIRECT_ERO_CSV.name}")

    # ---- Q1 ----
    df_q1 = load_desi_fits(DESI_Q1_FITS)
    ra_q1   = df_q1["euclid_ra"].values.astype(float)
    dec_q1  = df_q1["euclid_dec"].values.astype(float)
    ids_q1  = np.arange(len(df_q1), dtype=np.int64)

    df_gaia_q1 = pull_gaia_for_positions(conn, ra_q1, dec_q1, ids_q1, "Q1")
    df_gaia_q1 = compute_gaia_features(df_gaia_q1)
    df_gaia_q1["desi_z"] = df_q1["desi_z"].values[df_gaia_q1["gal_id"].values]

    df_gaia_q1.to_csv(DIRECT_Q1_CSV, index=False)
    print(f"  Saved {len(df_gaia_q1):,} rows → {DIRECT_Q1_CSV.name}")

    conn.close()


# ======================================================================
# STEP 2: TRAIN
# ======================================================================

def build_training_data(
    df_meta: pd.DataFrame,
    df_pos: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Positives: DESI galaxies in df_pos (directly pulled from WSDB, all label=1).
    Negatives: ERO metadata sources whose source_id is NOT in df_pos.
    Returns a combined DataFrame with a 'label' column.
    """
    pos_ids = set(df_pos["source_id"].dropna().astype(np.int64).values)

    df_neg = df_meta[~df_meta["source_id"].isin(pos_ids)].copy()
    df_neg["label"]  = 0
    df_neg["desi_z"] = np.nan

    df_pos_full = df_pos.copy()
    df_pos_full["label"] = 1
    # Align columns: keep only what's needed
    keep = ["source_id", "label", "desi_z"] + feature_cols
    keep_neg = [c for c in keep if c in df_neg.columns]
    keep_pos = [c for c in keep if c in df_pos_full.columns]

    df_combined = pd.concat(
        [df_neg[keep_neg], df_pos_full[keep_pos]], ignore_index=True
    )
    # Only drop rows where source_id is missing — NaN features pass through to XGBoost.
    df_combined = df_combined.dropna(subset=["source_id"]).reset_index(drop=True)

    n_pos = (df_combined["label"] == 1).sum()
    n_neg = (df_combined["label"] == 0).sum()
    print(f"  Positives (DESI galaxy, WSDB-matched): {n_pos:,}")
    print(f"  Negatives (non-galaxy from ERO):        {n_neg:,}")
    print(f"  Imbalance ratio: 1 : {n_neg / max(n_pos, 1):.1f}")
    return df_combined


def run_train(feature_cols: list[str]) -> None:
    if not DIRECT_ERO_CSV.exists():
        raise FileNotFoundError(
            f"{DIRECT_ERO_CSV} not found — run --pull_wsdb first."
        )
    if not DIRECT_Q1_CSV.exists():
        raise FileNotFoundError(
            f"{DIRECT_Q1_CSV} not found — run --pull_wsdb first."
        )

    print("\n[data] Loading ERO metadata ...")
    df_meta = load_all_fields(feature_cols)
    print(f"  ERO sources: {len(df_meta):,}")

    print("\n[data] Loading WSDB-pulled ERO galaxy features ...")
    df_pos = pd.read_csv(DIRECT_ERO_CSV, low_memory=False)
    # Impute PM and parallax significance NaNs with 0: physically correct for
    # extragalactic sources where Gaia finds no astrometric solution.
    for _f in ("feat_pm_significance", "feat_parallax_over_error"):
        if _f in df_pos.columns:
            n_nan = df_pos[_f].isna().sum()
            if n_nan:
                df_pos[_f] = df_pos[_f].fillna(0.0)
                print(f"  Imputed {n_nan} NaN → 0 in {_f}")
    # For remaining NaNs (feat_ruwe, feat_c_star, contamination counts), pass
    # NaN through to XGBoost, which learns the optimal missing-value direction
    # at each split — no imputation needed.
    # Only drop rows missing source_id or sep_arcsec (unusable rows).
    df_pos = df_pos.dropna(subset=["source_id"]).reset_index(drop=True)
    n_any_nan = df_pos[feature_cols].isna().any(axis=1).sum()
    print(f"  DESI ERO galaxies with Gaia source_id: {len(df_pos):,} "
          f"({n_any_nan} with at least one NaN feature, passed to XGBoost natively)")

    print("\n[label] Building training dataset ...")
    df_labeled = build_training_data(df_meta, df_pos, feature_cols)

    # ---- Train ----
    print("\n[train] Training XGB galaxy classifier ...")
    y_min, y_iqr = load_scaler(feature_cols)
    X = df_labeled[feature_cols].values.astype(np.float32)
    X = (X - y_min) / np.where(y_iqr > 0, y_iqr, 1.0)
    y = df_labeled["label"].values.astype(np.float32)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    spw    = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    params = {**PARAMS, "scale_pos_weight": spw}
    print(f"  scale_pos_weight = {spw:.1f}")

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    booster = xgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
    )
    print(f"  Best iteration: {booster.best_iteration}")

    scores_va = booster.predict(dval)
    ero_auc   = float(roc_auc_score(y_va, scores_va))
    ero_ap    = float(average_precision_score(y_va, scores_va))
    print(f"  Val AUC={ero_auc:.4f}  AP={ero_ap:.4f}")

    booster.save_model(str(OUT_DIR / "model_galaxy_desi.json"))

    n_pos_va = int((y_va == 1).sum())
    n_neg_va = int((y_va == 0).sum())
    ero_metrics = {
        "n_train": int(len(X_tr)),
        "n_val":   int(len(X_va)),
        "n_pos_val": n_pos_va, "n_neg_val": n_neg_va,
        "imbalance_ratio": float(n_neg_va / max(n_pos_va, 1)),
        "scale_pos_weight": spw,
        "auc": ero_auc, "ap": ero_ap,
        "best_iteration": int(booster.best_iteration),
    }
    with open(OUT_DIR / "metrics_ero.json", "w") as f:
        json.dump(ero_metrics, f, indent=2)

    # Feature importance
    fi = booster.get_score(importance_type="gain")
    fi_rows = [{"feature": col, "gain": fi.get(f"f{i}", 0.0)}
               for i, col in enumerate(feature_cols)]
    pd.DataFrame(fi_rows).sort_values("gain", ascending=False).to_csv(
        OUT_DIR / "feature_importance.csv", index=False
    )

    # Score all ERO sources
    X_all   = df_labeled[feature_cols].values.astype(np.float32)
    X_all   = (X_all - y_min) / np.where(y_iqr > 0, y_iqr, 1.0)
    scores_all = booster.predict(xgb.DMatrix(X_all))
    df_scores  = df_labeled[["source_id", "label"]].copy()
    df_scores["xgb_score"] = scores_all
    if "desi_z" in df_labeled.columns:
        df_scores["desi_z"] = df_labeled["desi_z"].values
    # Merge back metadata for stamps
    meta_cols = ["source_id", "phot_g_mean_mag", "field_tag", "npz_file", "index_in_file"]
    avail = [c for c in meta_cols if c in df_meta.columns]
    df_scores = df_scores.merge(df_meta[avail], on="source_id", how="left")
    df_scores.to_csv(OUT_DIR / "scores_all.csv", index=False)

    # ---- Q1 transfer test ----
    print("\n[Q1 transfer] Loading WSDB-pulled Q1 galaxy features ...")
    df_q1_pos = pd.read_csv(DIRECT_Q1_CSV, low_memory=False)
    for _f in ("feat_pm_significance", "feat_parallax_over_error"):
        if _f in df_q1_pos.columns:
            df_q1_pos[_f] = df_q1_pos[_f].fillna(0.0)
    df_q1_pos = df_q1_pos.dropna(subset=["source_id"]).reset_index(drop=True)
    print(f"  DESI Q1 galaxies with Gaia source_id: {len(df_q1_pos):,}")

    # Combine with the broader Q1 Gaia pool for negatives
    df_q1_pool = pd.read_csv(Q1_GAIA_CSV, low_memory=False)
    df_q1_pool = df_q1_pool.dropna(subset=feature_cols).reset_index(drop=True)
    q1_pos_ids = set(df_q1_pos["source_id"].dropna().astype(np.int64).values)
    df_q1_pool["label"] = df_q1_pool["source_id"].isin(q1_pos_ids).astype(int)

    X_q1   = df_q1_pool[feature_cols].values.astype(np.float32)
    X_q1   = (X_q1 - y_min) / np.where(y_iqr > 0, y_iqr, 1.0)
    s_q1   = booster.predict(xgb.DMatrix(X_q1))
    q1_auc = float(roc_auc_score(df_q1_pool["label"], s_q1))
    q1_ap  = float(average_precision_score(df_q1_pool["label"], s_q1))
    n_q1_pos = int(df_q1_pool["label"].sum())
    n_q1_neg = int((df_q1_pool["label"] == 0).sum())
    print(f"  Q1 transfer AUC={q1_auc:.4f}  AP={q1_ap:.4f}  "
          f"(n_pos={n_q1_pos:,}, n_neg={n_q1_neg:,})")

    q1_metrics = {
        "n_pos": n_q1_pos, "n_neg": n_q1_neg,
        "imbalance_ratio": float(n_q1_neg / max(n_q1_pos, 1)),
        "auc": q1_auc, "ap": q1_ap,
    }
    with open(OUT_DIR / "metrics_q1_transfer.json", "w") as f:
        json.dump(q1_metrics, f, indent=2)

    q1_fpr, q1_tpr, _ = roc_curve(df_q1_pool["label"], s_q1)
    q1_prec, q1_rec, _= precision_recall_curve(df_q1_pool["label"], s_q1)

    # ---- Plots ----
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_roc_pr(y_va, scores_va, q1_metrics,
                 np.array(q1_fpr), np.array(q1_tpr),
                 np.array(q1_prec), np.array(q1_rec),
                 ero_auc, ero_ap)
    _plot_feature_importance(booster, feature_cols)
    _plot_score_distribution(df_scores)
    _plot_umap_overlay(df_scores)

    # Stamp catalog for confirmed ERO galaxy hits (positives in df_scores)
    df_hits = df_scores[df_scores["label"] == 1].copy()
    _stamp_catalog(df_hits)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"ERO val      AUC={ero_auc:.4f}  AP={ero_ap:.4f}  "
          f"(n_pos={n_pos_va}, n_neg={n_neg_va})")
    print(f"Q1 transfer  AUC={q1_auc:.4f}  AP={q1_ap:.4f}  "
          f"(n_pos={n_q1_pos:,}, n_neg={n_q1_neg:,})")
    print(f"Outputs: {OUT_DIR}")
    print(f"Plots:   {PLOT_DIR}")


# ======================================================================
# PLOTS
# ======================================================================

def _plot_roc_pr(y_va, scores_va, q1_metrics,
                 q1_fpr, q1_tpr, q1_prec, q1_rec,
                 ero_auc, ero_ap):
    print("\n[plot] 01_roc_pr.png ...")
    fpr, tpr, _  = roc_curve(y_va, scores_va)
    prec, rec, _ = precision_recall_curve(y_va, scores_va)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(fpr, tpr, lw=2, color="#2196F3",
            label=f"ERO val  AUC={ero_auc:.3f}")
    ax.plot(q1_fpr, q1_tpr, lw=2, color=C_GAL, linestyle="--",
            label=f"Q1 transfer  AUC={q1_metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — DESI galaxy classifier (WSDB-matched)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(rec, prec, lw=2, color="#2196F3",
            label=f"ERO val  AP={ero_ap:.3f}")
    ax.plot(q1_rec, q1_prec, lw=2, color=C_GAL, linestyle="--",
            label=f"Q1 transfer  AP={q1_metrics['ap']:.3f}")
    baseline = (y_va == 1).sum() / len(y_va)
    ax.axhline(baseline, color="k", lw=0.8, linestyle="--", alpha=0.4,
               label=f"Random ({baseline:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_roc_pr.png", dpi=150)
    plt.close(fig)
    print("  saved 01_roc_pr.png")


def _plot_feature_importance(booster, feature_cols):
    print("[plot] 02_feature_importance.png ...")
    fi = booster.get_score(importance_type="gain")
    labels = [c.replace("feat_", "") for c in feature_cols]
    values = [fi.get(f"f{i}", 0.0) for i in range(len(feature_cols))]
    order  = np.argsort(values)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([labels[i] for i in order], [values[i] for i in order],
            color="#2196F3", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title("XGB feature importance — DESI galaxy classifier (15D, WSDB-matched)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_feature_importance.png", dpi=150)
    plt.close(fig)
    print("  saved 02_feature_importance.png")


def _plot_score_distribution(df_scores):
    print("[plot] 03_score_distribution.png ...")
    gal  = df_scores.loc[df_scores["label"] == 1, "xgb_score"]
    star = df_scores.loc[df_scores["label"] == 0, "xgb_score"]
    bins = np.linspace(0, 1, 51)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, log in zip(axes, [False, True]):
        ax.hist(star, bins=bins, density=True, color=C_GREY, alpha=0.6,
                label=f"Non-galaxy (n={len(star):,})")
        ax.hist(gal,  bins=bins, density=True, color=C_GAL, alpha=0.8,
                label=f"DESI galaxy (n={len(gal):,})")
        if log:
            ax.set_yscale("log"); ax.set_ylim(bottom=1e-3)
            ax.set_title("Score distribution (log scale)")
        else:
            ax.set_title("Score distribution (linear)")
        ax.set_xlabel("XGB score"); ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_score_distribution.png", dpi=150)
    plt.close(fig)
    print("  saved 03_score_distribution.png")


def _plot_umap_overlay(df_scores):
    if not EMB_16D.exists():
        print("[plot] UMAP embedding not found, skipping.")
        return
    print("[plot] 04_umap16d_galaxy_overlay.png ...")
    umap = pd.read_csv(EMB_16D, usecols=["source_id", "x", "y"], low_memory=False)
    umap["source_id"] = pd.to_numeric(umap["source_id"], errors="coerce").astype("Int64")
    merged = umap.merge(df_scores, on="source_id", how="left")
    in_gal = merged["label"] == 1
    has_s  = merged["xgb_score"].notna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(merged.loc[~in_gal, "x"], merged.loc[~in_gal, "y"],
               s=0.5, alpha=0.06, color=C_GREY, rasterized=True)
    ax.scatter(merged.loc[in_gal, "x"], merged.loc[in_gal, "y"],
               s=25, alpha=1.0, color=C_GAL, rasterized=True, zorder=5,
               edgecolors="white", linewidths=0.4,
               label=f"DESI galaxy (n={in_gal.sum()})")
    ax.set_title("(a) DESI confirmed galaxies", fontsize=10)
    ax.legend(markerscale=1.5, fontsize=8, framealpha=0.85, edgecolor="none")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.scatter(merged.loc[~has_s, "x"], merged.loc[~has_s, "y"],
               s=0.5, alpha=0.06, color=C_GREY, rasterized=True)
    sc = ax.scatter(merged.loc[has_s, "x"], merged.loc[has_s, "y"],
                    s=3, c=merged.loc[has_s, "xgb_score"],
                    cmap="RdPu", alpha=0.8, vmin=0, vmax=1,
                    rasterized=True, zorder=4)
    plt.colorbar(sc, ax=ax, label="XGB P(DESI galaxy)")
    ax.set_title("(b) XGB predicted probability", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("15D UMAP — DESI galaxy classifier overlay (WSDB-matched)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_umap16d_galaxy_overlay.png", dpi=150)
    plt.close(fig)
    print("  saved 04_umap16d_galaxy_overlay.png")


def _stamp_catalog(df_hits: pd.DataFrame) -> None:
    need = {"npz_file", "index_in_file", "field_tag"}
    if not need.issubset(df_hits.columns):
        print("[plot] Missing stamp columns, skipping stamp catalog.")
        return
    df_hits = df_hits.dropna(subset=["npz_file", "index_in_file"]).copy()
    df_hits["index_in_file"] = df_hits["index_in_file"].astype(int)
    df_hits = df_hits.sort_values("xgb_score", ascending=False).reset_index(drop=True)
    print(f"\n[plot] Generating stamp catalog for {len(df_hits)} galaxies ...")

    # Load stamps
    stamp_map: dict[int, np.ndarray] = {}
    for (ft, nf), grp in df_hits.groupby(["field_tag", "npz_file"]):
        path = DATASET_ROOT / str(ft) / str(nf)
        if not path.exists():
            continue
        with np.load(path) as d:
            if "X" not in d:
                continue
            Xall = d["X"]
        for row in grp.itertuples(index=False):
            ii = int(row.index_in_file)
            if not (0 <= ii < Xall.shape[0]):
                continue
            stamp = np.asarray(Xall[ii], dtype=np.float32)
            if stamp.ndim == 1:
                s = int(round(stamp.shape[0] ** 0.5))
                stamp = stamp.reshape(s, s)
            stamp_map[int(row.source_id)] = stamp

    n_total = len(df_hits)
    nrows   = int(np.ceil(STAMPS_PER_PAGE / NCOLS))
    n_pages = max(1, int(np.ceil(n_total / STAMPS_PER_PAGE)))

    for page in range(n_pages):
        i0, i1 = page * STAMPS_PER_PAGE, min((page + 1) * STAMPS_PER_PAGE, n_total)
        batch  = df_hits.iloc[i0:i1]

        fig = plt.figure(figsize=(NCOLS * 1.8, nrows * 2.1), facecolor="black")
        fig.suptitle(
            f"Euclid VIS stamps — DESI confirmed galaxies  "
            f"(page {page+1}/{n_pages}, sources {i0+1}–{i1}, ranked by XGB score)",
            color="white", fontsize=9, y=0.995,
        )
        gs = gridspec.GridSpec(nrows, NCOLS, figure=fig,
                               hspace=0.55, wspace=0.08,
                               top=0.965, bottom=0.01, left=0.01, right=0.99)

        for k, row in enumerate(batch.itertuples(index=False)):
            ax = fig.add_subplot(gs[k // NCOLS, k % NCOLS])
            ax.set_facecolor("black")
            sid   = int(row.source_id)
            stamp = stamp_map.get(sid)
            if stamp is not None:
                lo = np.nanpercentile(stamp, 1.0)
                hi = np.nanpercentile(stamp, 99.5)
                img = np.clip((stamp - lo) / (hi - lo + 1e-9), 0, 1)
                ax.imshow(img, cmap="gray", origin="lower",
                          interpolation="nearest", vmin=0, vmax=1)
            else:
                ax.imshow(np.zeros((STAMP_PIX, STAMP_PIX)), cmap="gray",
                          origin="lower", vmin=0, vmax=1)
                ax.text(0.5, 0.5, "N/A", color="red", fontsize=6,
                        ha="center", va="center", transform=ax.transAxes)
            field = str(getattr(row, "field_tag", "?")).replace("ERO-", "")
            gmag  = getattr(row, "phot_g_mean_mag", np.nan)
            score = float(row.xgb_score)
            dz    = getattr(row, "desi_z", np.nan)
            gstr  = f"G={float(gmag):.1f}" if pd.notna(gmag) else ""
            zstr  = f"z={float(dz):.3f}"   if pd.notna(dz)   else ""
            ax.set_title(f"{field}  {gstr}\np={score:.2f}  {zstr}",
                         color="white", fontsize=4.5, pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

        for k in range(len(batch), nrows * NCOLS):
            fig.add_subplot(gs[k // NCOLS, k % NCOLS]).set_visible(False)

        out_png = PLOT_DIR / f"05_galaxy_stamp_catalog_p{page+1:02d}.png"
        fig.savefig(out_png, dpi=180, facecolor="black")
        plt.close(fig)
        print(f"  saved {out_png.name}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull_wsdb", action="store_true",
                        help="Pull Gaia features for DESI galaxy positions from WSDB")
    parser.add_argument("--train", action="store_true",
                        help="Train XGB classifier and generate plots")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols(FEATURE_SET)
    print(f"Feature set: {FEATURE_SET} ({len(feature_cols)} features)")

    if args.pull_wsdb:
        pull_wsdb()

    if args.train:
        run_train(feature_cols)

    if not args.pull_wsdb and not args.train:
        parser.print_help()


if __name__ == "__main__":
    main()
