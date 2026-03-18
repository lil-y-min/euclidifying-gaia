"""
84_q1_quasar_pipeline.py
=========================

Full quasar classification pipeline on the Euclid Q1 dataset, mirroring the
ERO analysis (scripts 79 + 81) to enable a direct ERO↔Q1 comparison.

Q1 source catalog is in WSDB as euclid_q1.mer_final_cat (29.9M sources),
which has a gaia_id column allowing a direct JOIN with gaia_dr3.gaia_source
to obtain the 15D Gaia features.

Steps:
  --pull_wsdb   Pull Q1 × Gaia DR3 features + AllWISE photometry from WSDB
  --quaia       Crossmatch with Quaia catalog (Gaia source_id join)
  --train       Train 15D XGB, apply ERO model (transfer test), compare
  --plots       Comparison plots: ERO vs Q1 performance, populations, redshift

Outputs:
  output/ml_runs/quasar_q1/
    q1_gaia_features.csv       (Q1 sources with 15D Gaia features)
    q1_wise_phot.csv           (AllWISE photometry for Q1 sources)
    q1_quaia_hits.csv          (Quaia matches in Q1)
    model_q1_quasar.json       (Q1-trained XGB booster)
    metrics_q1.json            (Q1 metrics)
    metrics_comparison.json    (Q1 vs ERO comparison)
  plots/ml_runs/quasar_q1/
    01_roc_pr_comparison.png   ERO vs Q1 ROC and PR on same axes
    02_feature_importance_comparison.png  side-by-side feature importance
    03_wise_color_color_q1.png WISE colour-colour for Q1 sources
    04_quaia_redshift_comparison.png  Quaia quasar redshift dist ERO vs Q1
    05_transfer_score_dist.png ERO model applied to Q1: score distribution

Usage:
  python 84_q1_quasar_pipeline.py --pull_wsdb
  python 84_q1_quasar_pipeline.py --quaia
  python 84_q1_quasar_pipeline.py --train --plots
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
from astropy.io import fits
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
RUN_NAME     = "quasar_q1"
OUT_DIR      = BASE / "output" / "ml_runs" / RUN_NAME
PLOT_DIR     = BASE / "plots"  / "ml_runs" / RUN_NAME
FEATURE_SET  = "15D"

QUAIA_FITS   = BASE / "data" / "quaia_G20.5.fits"

# ERO results for comparison (from scripts 79 + 81)
ERO_WISE_METRICS  = BASE / "output" / "ml_runs" / "quasar_wise_clf" / "metrics.json"
ERO_QUAIA_METRICS = BASE / "output" / "ml_runs" / "quaia_clf"       / "metrics.json"
ERO_WISE_FI       = BASE / "output" / "ml_runs" / "quasar_wise_clf" / "feature_importance.csv"
ERO_QUAIA_FI      = BASE / "output" / "ml_runs" / "quaia_clf"       / "feature_importance.csv"
ERO_QUAIA_HITS    = BASE / "output" / "ml_runs" / "quaia_clf"       / "quaia_hits.csv"
ERO_WISE_SCORES   = BASE / "output" / "ml_runs" / "quasar_wise_clf" / "scores_all.csv"
ERO_QUAIA_BOOSTER = BASE / "output" / "ml_runs" / "quaia_clf"       / "model_quaia.json"

# WISE selection box (same as script 79)
WISE_W12_LO, WISE_W12_HI = 0.6, 1.6
WISE_W23_LO, WISE_W23_HI = 2.0, 4.2
WISE_MIN_SNR = 3.0
WISE_MATCH_RADIUS_ARCSEC = 2.0

# Gaia quality filter for Q1 pull (matching typical ERO source quality)
GAIA_GMAG_MAX = 21.0   # only pull Gaia sources bright enough for reliable features

# XGB params (same as ERO)
PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "learning_rate": 0.05, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 1.0, "tree_method": "hist",
    "seed": 123, "verbosity": 0,
}
NUM_BOOST_ROUND = 1000
EARLY_STOP      = 50

# Visual
C_ERO   = "#2196F3"   # blue  for ERO
C_Q1    = "#7b2d8b"   # purple for Q1
C_QSO   = "#ff007f"   # quasar points


# ======================================================================
# WSDB HELPERS
# ======================================================================

def wsdb_connect():
    import psycopg
    return psycopg.connect(
        host=os.environ.get("WSDB_HOST", "wsdb.ast.cam.ac.uk"),
        dbname=os.environ.get("WSDB_DB", "wsdb"),
        user=os.environ.get("WSDB_USER", ""),
        password=os.environ.get("WSDB_PASS", ""),
        port=int(os.environ.get("WSDB_PORT", "5432")),
        autocommit=True,
    )


def compute_gaia_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 15D Gaia feature columns from raw Gaia DR3 quantities.
    Mirrors the preprocessing done for the ERO dataset.
    """
    df = df.copy()

    # log10(G-band S/N)
    snr = pd.to_numeric(df.get("phot_g_mean_flux_over_error", np.nan), errors="coerce")
    df["feat_log10_snr"] = np.log10(np.maximum(snr, 1e-6))

    # RUWE
    df["feat_ruwe"] = pd.to_numeric(df.get("ruwe", np.nan), errors="coerce")

    # Astrometric excess noise
    df["feat_astrometric_excess_noise"] = pd.to_numeric(
        df.get("astrometric_excess_noise", np.nan), errors="coerce")

    # Parallax over error
    plx = pd.to_numeric(df.get("parallax", np.nan), errors="coerce")
    plx_err = pd.to_numeric(df.get("parallax_error", np.nan), errors="coerce")
    df["feat_parallax_over_error"] = plx / plx_err.replace(0, np.nan)

    # IPD fraction multi-peak
    df["feat_ipd_frac_multi_peak"] = pd.to_numeric(
        df.get("ipd_frac_multi_peak", np.nan), errors="coerce")

    # C* (corrected BP/RP flux excess factor, Riello et al. 2021)
    C_exc = pd.to_numeric(df.get("phot_bp_rp_excess_factor", np.nan), errors="coerce")
    bp_rp = pd.to_numeric(df.get("bp_rp", np.nan), errors="coerce")
    # Polynomial baseline for 5-param astrometric solution (Riello+21, Table 3)
    poly = 1.154360 + 0.033772 * bp_rp + 0.032277 * bp_rp**2
    df["feat_c_star"] = C_exc - poly

    # PM significance: sqrt((pmra/pmra_err)^2 + (pmdec/pmdec_err)^2)
    pmra  = pd.to_numeric(df.get("pmra",       np.nan), errors="coerce")
    pmdec = pd.to_numeric(df.get("pmdec",      np.nan), errors="coerce")
    pmra_e  = pd.to_numeric(df.get("pmra_error",  np.nan), errors="coerce")
    pmdec_e = pd.to_numeric(df.get("pmdec_error", np.nan), errors="coerce")
    pm_sig = np.sqrt((pmra / pmra_e.replace(0, np.nan))**2 +
                     (pmdec / pmdec_e.replace(0, np.nan))**2)
    df["feat_pm_significance"] = pm_sig

    # Extended features
    for col in ["astrometric_excess_noise_sig", "ipd_gof_harmonic_amplitude",
                "ipd_gof_harmonic_phase", "ipd_frac_odd_win",
                "phot_bp_n_contaminated_transits", "phot_bp_n_blended_transits",
                "phot_rp_n_contaminated_transits", "phot_rp_n_blended_transits"]:
        feat = "feat_" + col
        df[feat] = pd.to_numeric(df.get(col, np.nan), errors="coerce")

    return df


# ======================================================================
# STEP 1: PULL Q1 × GAIA + WISE FROM WSDB
# ======================================================================

def pull_q1_data() -> None:
    """
    Pull Q1 sources with Gaia features and AllWISE photometry from WSDB.
    Saves:
      q1_gaia_features.csv   — all Q1 sources with 15D Gaia features
      q1_wise_phot.csv       — subset also matched to AllWISE
    """
    conn = wsdb_connect()

    gaia_raw_cols = [
        "g.source_id",
        "q.right_ascension AS ra", "q.declination AS dec",
        "g.phot_g_mean_mag",
        "g.phot_g_mean_flux_over_error",
        "g.ruwe",
        "g.astrometric_excess_noise",
        "g.astrometric_excess_noise_sig",
        "g.parallax", "g.parallax_error",
        "g.visibility_periods_used",
        "g.ipd_frac_multi_peak",
        "g.ipd_gof_harmonic_amplitude",
        "g.ipd_gof_harmonic_phase",
        "g.ipd_frac_odd_win",
        "g.phot_bp_rp_excess_factor", "g.bp_rp",
        "g.pmra", "g.pmdec", "g.pmra_error", "g.pmdec_error",
        "g.phot_bp_n_contaminated_transits", "g.phot_bp_n_blended_transits",
        "g.phot_rp_n_contaminated_transits", "g.phot_rp_n_blended_transits",
        "q.point_like_prob", "q.extended_prob",
        "q.flux_vis_psf",
    ]
    cols_sql = ",\n    ".join(gaia_raw_cols)

    print("Pulling Q1 × Gaia DR3 features (G < 21, ruwe not null)...")
    q = f"""
        SELECT
            {cols_sql}
        FROM euclid_q1.mer_final_cat q
        JOIN gaia_dr3.gaia_source g ON g.source_id = q.gaia_id
        WHERE q.gaia_id IS NOT NULL
          AND g.phot_g_mean_mag < {GAIA_GMAG_MAX}
          AND g.ruwe IS NOT NULL
    """
    rows = conn.execute(q).fetchall()
    col_names = [
        "source_id", "ra", "dec", "phot_g_mean_mag", "phot_g_mean_flux_over_error",
        "ruwe", "astrometric_excess_noise", "astrometric_excess_noise_sig",
        "parallax", "parallax_error", "visibility_periods_used",
        "ipd_frac_multi_peak", "ipd_gof_harmonic_amplitude",
        "ipd_gof_harmonic_phase", "ipd_frac_odd_win",
        "phot_bp_rp_excess_factor", "bp_rp",
        "pmra", "pmdec", "pmra_error", "pmdec_error",
        "phot_bp_n_contaminated_transits", "phot_bp_n_blended_transits",
        "phot_rp_n_contaminated_transits", "phot_rp_n_blended_transits",
        "point_like_prob", "extended_prob", "flux_vis_psf",
    ]
    df = pd.DataFrame(rows, columns=col_names)
    print(f"  Q1 Gaia sources pulled: {len(df):,}")

    df = compute_gaia_features(df)
    df.to_csv(OUT_DIR / "q1_gaia_features.csv", index=False)
    print(f"  Saved q1_gaia_features.csv")

    # ---- AllWISE crossmatch ----
    print("Pulling AllWISE photometry for Q1 sources (chunked)...")
    r_deg = WISE_MATCH_RADIUS_ARCSEC / 3600.0
    wise_cols = "w.ra AS wise_ra, w.dec AS wise_dec, w.w1mpro, w.w2mpro, w.w3mpro, w.w1snr, w.w2snr, w.w3snr, w.ext_flg, w.cc_flags"

    results = []
    CHUNK = 5_000
    n_src = len(df)
    for i in range(0, n_src, CHUNK):
        batch = df.iloc[i : i + CHUNK]
        rows_sql = []
        for j, row in enumerate(batch.itertuples()):
            if j == 0:
                rows_sql.append(
                    f"({int(row.source_id)}::bigint, "
                    f"{float(row.ra)}::double precision, "
                    f"{float(row.dec)}::double precision)"
                )
            else:
                rows_sql.append(
                    f"({int(row.source_id)}, {float(row.ra)}, {float(row.dec)})"
                )
        vals = ", ".join(rows_sql)
        q_wise = f"""
            SELECT s.source_id, {wise_cols},
                   q3c_dist(s.ra, s.dec, w.ra, w.dec)*3600 AS sep_arcsec
            FROM (VALUES {vals}) AS s(source_id, ra, dec)
            JOIN LATERAL (
                SELECT {', '.join(['w.'+c for c in ['ra','dec','w1mpro','w2mpro','w3mpro','w1snr','w2snr','w3snr','ext_flg','cc_flags']])}
                FROM allwise.main w
                WHERE q3c_radial_query(w.ra, w.dec, s.ra, s.dec, {r_deg})
                ORDER BY q3c_dist(s.ra, s.dec, w.ra, w.dec) ASC
                LIMIT 1
            ) w ON TRUE
        """
        try:
            wrows = conn.execute(q_wise).fetchall()
        except Exception as e:
            print(f"  WARN chunk {i}: {e}")
            continue
        if wrows:
            results.append(pd.DataFrame(wrows, columns=[
                "source_id", "wise_ra", "wise_dec",
                "w1mpro", "w2mpro", "w3mpro",
                "w1snr", "w2snr", "w3snr",
                "ext_flg", "cc_flags", "sep_arcsec",
            ]))
        if (i // CHUNK) % 20 == 0:
            print(f"  ... processed {min(i + CHUNK, n_src):,} / {n_src:,}")

    if results:
        wise_df = pd.concat(results, ignore_index=True)
        wise_df.to_csv(OUT_DIR / "q1_wise_phot.csv", index=False)
        print(f"  Saved q1_wise_phot.csv ({len(wise_df):,} WISE matches)")
    else:
        print("  WARNING: no WISE matches found")

    conn.close()


# ======================================================================
# STEP 2: QUAIA CROSSMATCH
# ======================================================================

def pull_quaia() -> None:
    """Crossmatch Q1 Gaia sources with Quaia by source_id."""
    if not QUAIA_FITS.exists():
        raise FileNotFoundError(f"Quaia FITS not found at {QUAIA_FITS}")

    df_q1 = pd.read_csv(OUT_DIR / "q1_gaia_features.csv", low_memory=False)
    df_q1["source_id"] = pd.to_numeric(df_q1["source_id"], errors="coerce").astype("Int64")
    q1_ids = set(df_q1["source_id"].dropna().tolist())

    print(f"Loading Quaia catalog from {QUAIA_FITS.name}...")
    with fits.open(QUAIA_FITS, memmap=True) as hdul:
        data = hdul[1].data
        def native(arr):
            a = np.asarray(arr)
            return a.astype(a.dtype.newbyteorder("="), copy=False)
        quaia = pd.DataFrame({"source_id": native(data["source_id"]).astype(np.int64)})
        for col in ["ra", "dec", "redshift_quaia", "redshift_source",
                    "phot_g_mean_mag", "w1_mag", "w2_mag"]:
            if col in data.names:
                quaia[col] = native(data[col])
    quaia["source_id"] = quaia["source_id"].astype("Int64")
    print(f"  Quaia total: {len(quaia):,}")

    hits = quaia[quaia["source_id"].isin(q1_ids)].copy()
    print(f"  Quaia hits in Q1 dataset: {len(hits):,}")

    hits.to_csv(OUT_DIR / "q1_quaia_hits.csv", index=False)
    print(f"  Saved q1_quaia_hits.csv")


# ======================================================================
# HELPERS: SCALER + LABEL ASSIGNMENT
# ======================================================================

def load_scaler(feature_cols: list[str]):
    stem = scaler_stem(FEATURE_SET)
    npz = BASE / "output" / "scalers" / f"{stem}.npz"
    d = np.load(npz, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    y_min = d["y_min"].astype(np.float32)
    y_iqr = d["y_iqr"].astype(np.float32)
    idx = np.array([names.index(c) for c in feature_cols], dtype=int)
    return y_min[idx], y_iqr[idx]


def assign_wise_labels(df_gaia: pd.DataFrame, df_wise: pd.DataFrame) -> pd.DataFrame:
    """Merge WISE photometry, apply S/N cuts, assign quasar-locus label."""
    df_wise = df_wise.copy()
    for col in ["w1mpro", "w2mpro", "w3mpro", "w1snr", "w2snr", "w3snr"]:
        df_wise[col] = pd.to_numeric(df_wise[col], errors="coerce")
    df_wise["source_id"] = pd.to_numeric(df_wise["source_id"], errors="coerce").astype("Int64")

    snr_ok = ((df_wise["w1snr"] >= WISE_MIN_SNR) &
              (df_wise["w2snr"] >= WISE_MIN_SNR) &
              (df_wise["w3snr"] >= WISE_MIN_SNR))
    df_wise = df_wise[snr_ok].copy()
    df_wise["w1_w2"] = df_wise["w1mpro"] - df_wise["w2mpro"]
    df_wise["w2_w3"] = df_wise["w2mpro"] - df_wise["w3mpro"]
    in_locus = (
        (df_wise["w1_w2"] >= WISE_W12_LO) & (df_wise["w1_w2"] <= WISE_W12_HI) &
        (df_wise["w2_w3"] >= WISE_W23_LO) & (df_wise["w2_w3"] <= WISE_W23_HI)
    )
    df_wise["label_wise"] = in_locus.astype(int)

    df = df_gaia.merge(
        df_wise[["source_id", "w1_w2", "w2_w3", "label_wise"]],
        on="source_id", how="inner",
    )
    df = df.rename(columns={"label_wise": "label"})
    return df


# ======================================================================
# STEP 3: TRAIN + COMPARE
# ======================================================================

def train_and_compare(run_wise: bool = True) -> None:
    """Train 15D classifier on Q1, apply ERO model, generate comparison plots."""
    feature_cols = get_feature_cols(FEATURE_SET)
    feat_min, feat_iqr = load_scaler(feature_cols)

    # ---- Load Q1 data ----
    df_gaia = pd.read_csv(OUT_DIR / "q1_gaia_features.csv", low_memory=False)
    df_gaia["source_id"] = pd.to_numeric(df_gaia["source_id"], errors="coerce").astype("Int64")

    # ---- WISE labels ----
    wise_path = OUT_DIR / "q1_wise_phot.csv"
    if not wise_path.exists():
        raise FileNotFoundError("Run --pull_wsdb first to generate q1_wise_phot.csv")
    df_wise = pd.read_csv(wise_path, low_memory=False)
    df_wise["source_id"] = pd.to_numeric(df_wise["source_id"], errors="coerce").astype("Int64")

    df = assign_wise_labels(df_gaia, df_wise)
    df = df.dropna(subset=feature_cols).copy()

    n_qso = int(df["label"].sum())
    print(f"Q1 WISE dataset: {len(df):,} sources, {n_qso} quasar-locus")

    # ---- Quaia labels (for Quaia classifier) ----
    quaia_path = OUT_DIR / "q1_quaia_hits.csv"
    if quaia_path.exists():
        q1_quaia = pd.read_csv(quaia_path, low_memory=False)
        q1_quaia["source_id"] = pd.to_numeric(q1_quaia["source_id"], errors="coerce").astype("Int64")
        quaia_ids = set(q1_quaia["source_id"].dropna().tolist())
        df_all_gaia = df_gaia.dropna(subset=feature_cols).copy()
        df_all_gaia["label_quaia"] = df_all_gaia["source_id"].isin(quaia_ids).astype(int)
        n_quaia = int(df_all_gaia["label_quaia"].sum())
        print(f"Q1 Quaia dataset: {len(df_all_gaia):,} sources, {n_quaia} Quaia quasars")
    else:
        print("  quaia_hits not found — run --quaia; Quaia comparison will be skipped")
        q1_quaia = None

    # ---- Train/val/test split (random 70/15/15) ----
    rng = np.random.RandomState(42)
    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42,
                                           stratify=df["label"].to_numpy())
    val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, random_state=42,
                                           stratify=df["label"].to_numpy()[temp_idx])

    X = (df[feature_cols].to_numpy(dtype=np.float32) - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)
    y = df["label"].to_numpy(dtype=np.float32)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_vl, y_vl = X[val_idx],   y[val_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    pos_ratio = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    params = {**PARAMS, "scale_pos_weight": pos_ratio}

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_vl, label=y_vl, feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_te, label=y_te, feature_names=feature_cols)

    print("Training Q1 15D XGBoost classifier...")
    booster = xgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=100,
    )
    booster.save_model(str(OUT_DIR / "model_q1_quasar.json"))

    prob_q1 = booster.predict(dtest)
    auc_q1  = roc_auc_score(y_te, prob_q1)
    ap_q1   = average_precision_score(y_te, prob_q1)
    print(f"Q1 trained model: AUC={auc_q1:.4f}  AP={ap_q1:.4f}")

    # ---- Apply ERO-trained model to Q1 test set ----
    ero_booster = xgb.Booster()
    ero_booster.load_model(str(ERO_WISE_SCORES.parent / "model_quasar_wise.json"))
    prob_transfer = ero_booster.predict(dtest)
    auc_transfer  = roc_auc_score(y_te, prob_transfer)
    ap_transfer   = average_precision_score(y_te, prob_transfer)
    print(f"ERO model on Q1:  AUC={auc_transfer:.4f}  AP={ap_transfer:.4f}")

    # ---- Load ERO metrics for comparison ----
    ero_auc, ero_ap = np.nan, np.nan
    if ERO_WISE_METRICS.exists():
        m = json.loads(ERO_WISE_METRICS.read_text())
        ero_auc, ero_ap = m.get("auc", np.nan), m.get("ap", np.nan)

    metrics = {
        "q1_auc": auc_q1, "q1_ap": ap_q1,
        "ero_model_on_q1_auc": auc_transfer, "ero_model_on_q1_ap": ap_transfer,
        "ero_auc": ero_auc, "ero_ap": ero_ap,
        "q1_n_test": int(len(y_te)), "q1_n_quasar_test": int(y_te.sum()),
    }
    (OUT_DIR / "metrics_comparison.json").write_text(json.dumps(metrics, indent=2))

    fi_q1 = pd.DataFrame({
        "feature": list(booster.get_score(importance_type="gain").keys()),
        "gain":    list(booster.get_score(importance_type="gain").values()),
    }).sort_values("gain", ascending=False)
    fi_q1.to_csv(OUT_DIR / "feature_importance_q1.csv", index=False)

    return booster, prob_q1, prob_transfer, y_te, df, fi_q1, q1_quaia, df_gaia


# ======================================================================
# STEP 4: PLOTS
# ======================================================================

def make_plots(booster, prob_q1, prob_transfer, y_te,
               df_wise_q1, fi_q1, q1_quaia_df, df_gaia) -> None:
    feature_cols = get_feature_cols(FEATURE_SET)
    feat_min, feat_iqr = load_scaler(feature_cols)

    # ---- ROC curves ----
    fpr_q1, tpr_q1, _ = roc_curve(y_te, prob_q1)
    fpr_tr, tpr_tr, _ = roc_curve(y_te, prob_transfer)
    prec_q1, rec_q1, _ = precision_recall_curve(y_te, prob_q1)
    prec_tr, rec_tr, _ = precision_recall_curve(y_te, prob_transfer)

    metrics = json.loads((OUT_DIR / "metrics_comparison.json").read_text())

    # Load ERO ROC/PR if booster available
    ero_roc_data = {}
    ero_wise_model = BASE / "output" / "ml_runs" / "quasar_wise_clf" / "model_quasar_wise.json"
    ero_wise_scores = ERO_WISE_SCORES
    if ero_wise_scores.exists():
        ero_s = pd.read_csv(ero_wise_scores)
        # We can compute ROC from stored scores
        ero_prob = ero_s["xgb_score"].to_numpy()
        ero_lab  = ero_s["label"].to_numpy()
        fpr_ero, tpr_ero, _ = roc_curve(ero_lab, ero_prob)
        prec_ero, rec_ero, _ = precision_recall_curve(ero_lab, ero_prob)
        ero_roc_data = {"fpr": fpr_ero, "tpr": tpr_ero,
                        "prec": prec_ero, "rec": rec_ero}

    # ---- Plot 01: ROC + PR comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(fpr_q1, tpr_q1, color=C_Q1, lw=2,
            label=f"Q1 model on Q1  AUC={metrics['q1_auc']:.3f}")
    ax.plot(fpr_tr, tpr_tr, color=C_Q1, lw=1.5, linestyle="--",
            label=f"ERO model on Q1  AUC={metrics['ero_model_on_q1_auc']:.3f}")
    if ero_roc_data:
        ax.plot(ero_roc_data["fpr"], ero_roc_data["tpr"], color=C_ERO, lw=2,
                label=f"ERO model on ERO  AUC={metrics['ero_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC — WISE quasar locus classifier", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(rec_q1, prec_q1, color=C_Q1, lw=2,
            label=f"Q1 model on Q1  AP={metrics['q1_ap']:.3f}")
    ax.plot(rec_tr, prec_tr, color=C_Q1, lw=1.5, linestyle="--",
            label=f"ERO model on Q1  AP={metrics['ero_model_on_q1_ap']:.3f}")
    if ero_roc_data:
        ax.plot(ero_roc_data["rec"], ero_roc_data["prec"], color=C_ERO, lw=2,
                label=f"ERO model on ERO  AP={metrics['ero_ap']:.3f}")
    ax.axhline(y_te.mean(), color="gray", lw=0.8, linestyle=":",
               label=f"Q1 prevalence={y_te.mean():.4f}")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision-Recall — WISE quasar locus classifier", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("ERO vs Q1: quasar classifier performance (15D Gaia features)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_roc_pr_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved 01_roc_pr_comparison.png")

    # ---- Plot 02: Feature importance comparison ERO vs Q1 ----
    fi_ero = None
    if ERO_WISE_FI.exists():
        fi_ero = pd.read_csv(ERO_WISE_FI).head(15)

    if fi_ero is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        top_q1 = fi_q1.head(15)

        ax = axes[0]
        ax.barh([f.replace("feat_","") for f in fi_ero["feature"]][::-1],
                fi_ero["gain"].values[::-1], color=C_ERO, alpha=0.85)
        ax.set_xlabel("XGBoost gain")
        ax.set_title(f"ERO  (AUC={metrics['ero_auc']:.3f})", fontsize=10)
        ax.grid(axis="x", alpha=0.25)

        ax = axes[1]
        ax.barh([f.replace("feat_","") for f in top_q1["feature"]][::-1],
                top_q1["gain"].values[::-1], color=C_Q1, alpha=0.85)
        ax.set_xlabel("XGBoost gain")
        ax.set_title(f"Q1  (AUC={metrics['q1_auc']:.3f})", fontsize=10)
        ax.grid(axis="x", alpha=0.25)

        fig.suptitle("Feature importance: ERO vs Q1 WISE quasar classifier (15D)", fontsize=11)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "02_feature_importance_comparison.png", dpi=150)
        plt.close(fig)
        print("  saved 02_feature_importance_comparison.png")

    # ---- Plot 03: WISE colour-colour for Q1 ----
    df_cc = df_wise_q1[["w1_w2", "w2_w3", "label"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 6))
    m0 = df_cc["label"] == 0
    m1 = df_cc["label"] == 1
    ax.scatter(df_cc.loc[m0, "w2_w3"], df_cc.loc[m0, "w1_w2"],
               s=4, alpha=0.2, color="#aaaaaa", label=f"Outside locus (n={m0.sum()})")
    ax.scatter(df_cc.loc[m1, "w2_w3"], df_cc.loc[m1, "w1_w2"],
               s=12, alpha=0.8, color=C_Q1, label=f"Quasar locus (n={m1.sum()})")
    rect_x = [WISE_W23_LO, WISE_W23_HI, WISE_W23_HI, WISE_W23_LO, WISE_W23_LO]
    rect_y = [WISE_W12_LO, WISE_W12_LO, WISE_W12_HI, WISE_W12_HI, WISE_W12_LO]
    ax.plot(rect_x, rect_y, "b--", lw=1.5, label="Selection box")
    ax.set_xlabel("W2 − W3 [Vega mag]", fontsize=11)
    ax.set_ylabel("W1 − W2 [Vega mag]", fontsize=11)
    ax.set_title("WISE colour-colour: Q1 sources (AllWISE Vega)", fontsize=11)
    ax.legend(markerscale=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_wise_color_color_q1.png", dpi=150)
    plt.close(fig)
    print("  saved 03_wise_color_color_q1.png")

    # ---- Plot 04: Quaia redshift distribution ERO vs Q1 ----
    if q1_quaia_df is not None and ERO_QUAIA_HITS.exists():
        ero_quaia = pd.read_csv(ERO_QUAIA_HITS, low_memory=False)
        z_ero = pd.to_numeric(ero_quaia.get("redshift_quaia", pd.Series(dtype=float)),
                              errors="coerce").dropna()
        z_q1  = pd.to_numeric(q1_quaia_df.get("redshift_quaia", pd.Series(dtype=float)),
                              errors="coerce").dropna()

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 5, 35)
        ax.hist(z_ero, bins=bins, density=True, alpha=0.6, color=C_ERO,
                label=f"ERO Quaia quasars (n={len(z_ero)})")
        ax.hist(z_q1,  bins=bins, density=True, alpha=0.6, color=C_Q1,
                label=f"Q1 Quaia quasars (n={len(z_q1)})")
        ax.set_xlabel("Photometric redshift (Quaia)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Quaia quasar redshift distribution: ERO vs Q1", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "04_quaia_redshift_comparison.png", dpi=150)
        plt.close(fig)
        print("  saved 04_quaia_redshift_comparison.png")

    # ---- Plot 05: ERO model applied to Q1 — score distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bins_p = np.linspace(0, 1, 40)
    for ax, prob, title, c in [
        (axes[0], prob_q1,      "Q1-trained model on Q1 test set", C_Q1),
        (axes[1], prob_transfer, "ERO-trained model on Q1 test set", C_ERO),
    ]:
        ax.hist(prob[y_te == 0], bins=bins_p, density=True, alpha=0.5,
                color="#aaaaaa", label="Non-quasar")
        ax.hist(prob[y_te == 1], bins=bins_p, density=True, alpha=0.8,
                color=c, label="WISE quasar locus")
        ax.axvline(0.5, color="k", lw=0.8, linestyle="--")
        ax.set_xlabel("XGB P(quasar)")
        ax.set_ylabel("Density")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Score distributions: model generalization ERO → Q1", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_transfer_score_dist.png", dpi=150)
    plt.close(fig)
    print("  saved 05_transfer_score_dist.png")

    print(f"\nAll plots -> {PLOT_DIR}")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull_wsdb", action="store_true",
                    help="Pull Q1 × Gaia + WISE from WSDB")
    ap.add_argument("--quaia", action="store_true",
                    help="Crossmatch with Quaia catalog")
    ap.add_argument("--train", action="store_true",
                    help="Train Q1 classifier and run transfer test")
    ap.add_argument("--plots", action="store_true",
                    help="Generate comparison plots (requires --train)")
    args = ap.parse_args()

    if not any([args.pull_wsdb, args.quaia, args.train, args.plots]):
        ap.print_help()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    user = os.environ.get("WSDB_USER", "")
    pw   = os.environ.get("WSDB_PASS", "")
    if (args.pull_wsdb or args.quaia) and (not user or not pw):
        raise RuntimeError("Set WSDB_USER and WSDB_PASS environment variables.")

    if args.pull_wsdb:
        pull_q1_data()

    if args.quaia:
        pull_quaia()

    if args.train or args.plots:
        results = train_and_compare()
        if args.plots:
            make_plots(*results)


if __name__ == "__main__":
    main()
