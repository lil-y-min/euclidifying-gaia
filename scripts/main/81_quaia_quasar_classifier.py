"""
81_quaia_quasar_classifier.py
===============================

Binary XGBoost classifier: Gaia source is a confirmed quasar from the
Quaia catalog (label=1) vs non-quasar (label=0), using 16D Gaia features.

Quaia (Storey-Fisher et al. 2024, zenodo.org/records/10403370) is a catalog
of ~1.3M quasars selected from Gaia DR3 + unWISE photometry.  It contains
Gaia source_id directly, so the crossmatch is a simple integer join.

Labels:
  label=1  : source_id present in Quaia G20.5
  label=0  : all other sources in our ERO dataset
  (Sources in Quaia but outside our ERO footprint are simply absent from
   the merge and do not appear in the training set.)

Also reports:
  - How many Quaia quasars fall inside the WISE color-cut box (script 79)
    => measures completeness/purity of that photometric selection.
  - Feature importance comparison against script 79 (WISE color cut).

Outputs:
  output/ml_runs/quaia_clf/
    quaia_hits.csv               (matched Quaia sources in our dataset)
    model_quaia.json
    metrics.json
    feature_importance.csv
  plots/ml_runs/quaia_clf/
    01_roc_pr.png
    02_feature_importance.png
    03_wise_crosscheck.png       (Quaia sources on W1-W2 vs W2-W3 diagram)
    04_umap16d_quaia_overlay.png
    05_umap_pixel_quaia_overlay.png
    06_quaia_stamp_catalog_p{N}.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
from astropy.io import fits
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema import get_feature_cols, scaler_stem


# ======================================================================
# CONFIG
# ======================================================================

BASE         = Path(__file__).resolve().parents[2]
DATASET_ROOT = BASE / "output" / "dataset_npz"
RUN_NAME     = "quaia_clf"
OUT_DIR      = BASE / "output" / "ml_runs" / RUN_NAME
PLOT_DIR     = BASE / "plots"  / "ml_runs" / RUN_NAME
META_NAME    = "metadata_16d.csv"
FEATURE_SET  = "15D"

QUAIA_FITS   = BASE / "data" / "quaia_G20.5.fits"

# WISE color cut from script 79 (for cross-check plot)
WISE_CSV     = BASE / "output" / "ml_runs" / "quasar_wise_clf" / "wise_phot.csv"
WISE_W12_LO, WISE_W12_HI = 0.6, 1.6
WISE_W23_LO, WISE_W23_HI = 2.0, 4.2
WISE_MIN_SNR = 3.0

# Existing UMAP embeddings
EMB_16D = (BASE / "output" / "experiments" / "embeddings"
           / "umap16d_manualv8_filtered" / "embedding_umap.csv")
EMB_PIX = (BASE / "output" / "experiments" / "embeddings"
           / "double_stars_pixels"
           / "pixels_umap_standard_manualv8_filtered"
           / "umap_standard" / "embedding_umap.csv")

# Stamp catalog
STAMP_PIX        = 20
STAMPS_PER_PAGE  = 80
NCOLS            = 8

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

# Visual
C_QSO  = "#ff007f"
C_GREY = "#555555"


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


def load_quaia_source_ids(fits_path: Path) -> pd.DataFrame:
    """Read Quaia FITS and return DataFrame with source_id and key columns."""
    print(f"  Reading {fits_path.name}...")
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        cols = data.names
        print(f"  Quaia columns: {cols[:20]} ...")

        def native(arr):
            """Convert FITS big-endian array to native byte order."""
            a = np.asarray(arr)
            return a.astype(a.dtype.newbyteorder("="), copy=False)

        source_id = native(data["source_id"]).astype(np.int64)
        df = pd.DataFrame({"source_id": source_id})

        # Add optional columns if present
        for col in ["ra", "dec", "redshift_quaia", "redshift_source",
                    "phot_g_mean_mag", "w1_mag", "w2_mag"]:
            if col in cols:
                df[col] = native(data[col])
    print(f"  Quaia total sources: {len(df):,}")
    return df


def load_stamps(df_sel: pd.DataFrame) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for (field_tag, npz_file), grp in df_sel.groupby(["field_tag", "npz_file"]):
        npz_path = DATASET_ROOT / str(field_tag) / str(npz_file)
        if not npz_path.exists():
            continue
        with np.load(npz_path) as d:
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
            out[int(row.source_id)] = stamp
    return out


def norm_stamp(stamp: np.ndarray) -> np.ndarray:
    lo = np.nanpercentile(stamp, 1.0)
    hi = np.nanpercentile(stamp, 99.5)
    if hi <= lo:
        return np.zeros_like(stamp)
    return np.clip((stamp - lo) / (hi - lo), 0, 1)


def umap_overlay(emb_path: Path, scores_df: pd.DataFrame,
                 title: str, out_png: Path) -> None:
    """Overlay Quaia label + XGB score on an existing UMAP embedding."""
    umap = pd.read_csv(emb_path, usecols=["source_id", "x", "y"], low_memory=False)
    umap["source_id"] = pd.to_numeric(umap["source_id"], errors="coerce").astype("Int64")
    merged = umap.merge(scores_df, on="source_id", how="left")

    in_q  = merged["label"] == 1
    has_s = merged["xgb_score"].notna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(merged.loc[~in_q, "x"], merged.loc[~in_q, "y"],
               s=0.5, alpha=0.06, color=C_GREY, rasterized=True)
    ax.scatter(merged.loc[in_q, "x"], merged.loc[in_q, "y"],
               s=25, alpha=1.0, color=C_QSO, rasterized=True, zorder=5,
               edgecolors="white", linewidths=0.4,
               label=f"Quaia quasar (n={in_q.sum()})")
    ax.set_title("(a) Quaia confirmed quasars", fontsize=10)
    ax.legend(markerscale=1.5, fontsize=8, framealpha=0.85, edgecolor="none")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.scatter(merged.loc[~has_s, "x"], merged.loc[~has_s, "y"],
               s=0.5, alpha=0.06, color=C_GREY, rasterized=True)
    sc = ax.scatter(merged.loc[has_s, "x"], merged.loc[has_s, "y"],
                    s=3, c=merged.loc[has_s, "xgb_score"],
                    cmap="RdPu", alpha=0.8, vmin=0, vmax=1,
                    rasterized=True, zorder=4)
    plt.colorbar(sc, ax=ax, label="XGB P(Quaia quasar)")
    ax.set_title("(b) XGB predicted probability", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  saved {out_png.name}")


def stamp_catalog(df_hits: pd.DataFrame, out_prefix: Path) -> None:
    """Produce paginated stamp gallery of Quaia quasars."""
    df_hits = df_hits.dropna(subset=["npz_file", "index_in_file"]).copy()
    df_hits["index_in_file"] = df_hits["index_in_file"].astype(int)
    df_hits = df_hits.sort_values("xgb_score", ascending=False).reset_index(drop=True)

    stamp_map = load_stamps(df_hits)
    n_total   = len(df_hits)
    nrows     = int(np.ceil(STAMPS_PER_PAGE / NCOLS))
    n_pages   = max(1, int(np.ceil(n_total / STAMPS_PER_PAGE)))
    print(f"  Generating {n_pages} stamp page(s) for {n_total} Quaia hits...")

    for page in range(n_pages):
        i0, i1 = page * STAMPS_PER_PAGE, min((page + 1) * STAMPS_PER_PAGE, n_total)
        batch = df_hits.iloc[i0:i1]

        fig = plt.figure(figsize=(NCOLS * 1.8, nrows * 2.1), facecolor="black")
        fig.suptitle(
            f"Euclid VIS stamps — Quaia confirmed quasars  "
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
                ax.imshow(norm_stamp(stamp), cmap="gray",
                          origin="lower", interpolation="nearest", vmin=0, vmax=1)
            else:
                ax.imshow(np.zeros((STAMP_PIX, STAMP_PIX)), cmap="gray",
                          origin="lower", vmin=0, vmax=1)
                ax.text(0.5, 0.5, "N/A", color="red", fontsize=6,
                        ha="center", va="center", transform=ax.transAxes)

            field = str(getattr(row, "field_tag", "?")).replace("ERO-", "")
            gmag  = getattr(row, "phot_g_mean_mag", np.nan)
            score = float(row.xgb_score) if hasattr(row, "xgb_score") else np.nan
            z     = getattr(row, "redshift_quaia", np.nan)
            gmag_s = f"G={gmag:.1f}" if np.isfinite(float(gmag)) else ""
            z_s    = f"z={z:.2f}"    if np.isfinite(float(z))    else ""
            ax.set_title(f"{field}  {gmag_s}\np={score:.2f}  {z_s}",
                         color="white", fontsize=4.5, pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

        for k in range(len(batch), nrows * NCOLS):
            fig.add_subplot(gs[k // NCOLS, k % NCOLS]).set_visible(False)

        out_png = out_prefix.parent / f"{out_prefix.name}_p{page+1:02d}.png"
        fig.savefig(out_png, dpi=180, facecolor="black")
        plt.close(fig)
        print(f"  saved {out_png.name}")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not QUAIA_FITS.exists():
        raise FileNotFoundError(
            f"Quaia catalog not found at {QUAIA_FITS}. "
            "Download from https://zenodo.org/records/10403370"
        )

    feature_cols = get_feature_cols(FEATURE_SET)
    feat_min, feat_iqr = load_scaler(feature_cols)

    # ---- Load Quaia ----
    print("Loading Quaia catalog...")
    quaia = load_quaia_source_ids(QUAIA_FITS)
    quaia_ids = set(quaia["source_id"].tolist())

    # ---- Load ERO metadata ----
    print("Loading ERO metadata...")
    df_meta = load_all_fields(feature_cols)
    df_meta["source_id"] = pd.to_numeric(df_meta["source_id"], errors="coerce").astype("Int64")
    print(f"  ERO sources: {len(df_meta):,}")

    # ---- Crossmatch by source_id ----
    df_meta["label"] = df_meta["source_id"].isin(
        pd.array(list(quaia_ids), dtype="Int64")
    ).astype(int)

    n_hits = int(df_meta["label"].sum())
    print(f"  Quaia hits in ERO dataset: {n_hits}")

    # Save matched sources
    hits = df_meta[df_meta["label"] == 1].merge(
        quaia, on="source_id", how="left", suffixes=("", "_quaia")
    )
    hits.to_csv(OUT_DIR / "quaia_hits.csv", index=False)
    print(f"  Saved quaia_hits.csv ({len(hits)} rows)")

    # ---- Train XGB ----
    df = df_meta.dropna(subset=feature_cols).copy()
    X = df[feature_cols].to_numpy(dtype=np.float32)
    X = (X - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)
    y = df["label"].to_numpy(dtype=np.float32)
    splits = df["split_code"].to_numpy(dtype=int) if "split_code" in df.columns else np.zeros(len(df), dtype=int)

    X_tr, y_tr = X[splits == 0], y[splits == 0]
    X_vl, y_vl = X[splits == 1], y[splits == 1]
    X_te, y_te = X[splits == 2], y[splits == 2]
    print(f"\nSplit — Train={len(X_tr)}, Val={len(X_vl)}, Test={len(X_te)}")
    print(f"  Quaia in test: {int(y_te.sum())}")

    pos_ratio = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    params = {**PARAMS, "scale_pos_weight": pos_ratio}
    print(f"  scale_pos_weight={pos_ratio:.1f}")

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_vl, label=y_vl, feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_te, label=y_te, feature_names=feature_cols)

    print("Training XGB classifier...")
    booster = xgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=100,
    )
    booster.save_model(str(OUT_DIR / "model_quaia.json"))

    prob_te = booster.predict(dtest)
    auc = roc_auc_score(y_te, prob_te)
    ap  = average_precision_score(y_te, prob_te)
    print(f"\nTest AUC={auc:.4f}  AP={ap:.4f}")

    # Feature importance
    fi = booster.get_score(importance_type="gain")
    fi_df = pd.DataFrame({"feature": list(fi.keys()), "gain": list(fi.values())})
    fi_df = fi_df.sort_values("gain", ascending=False)
    fi_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    # XGB scores on all sources (for UMAP overlay)
    X_all_scaled = (df[feature_cols].to_numpy(dtype=np.float32) - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)
    prob_all = booster.predict(xgb.DMatrix(X_all_scaled, feature_names=feature_cols))
    scores_df = pd.DataFrame({
        "source_id": df["source_id"].to_numpy(),
        "label":     y,
        "xgb_score": prob_all,
    })
    scores_df["source_id"] = scores_df["source_id"].astype("Int64")
    scores_df.to_csv(OUT_DIR / "scores_all.csv", index=False)

    # ---- WISE cross-check ----
    wise_available = WISE_CSV.exists()
    if wise_available:
        wise = pd.read_csv(WISE_CSV, low_memory=False)
        for col in ["w1mpro", "w2mpro", "w3mpro", "w1snr", "w2snr", "w3snr"]:
            wise[col] = pd.to_numeric(wise[col], errors="coerce")
        snr_ok = ((wise["w1snr"] >= WISE_MIN_SNR) &
                  (wise["w2snr"] >= WISE_MIN_SNR) &
                  (wise["w3snr"] >= WISE_MIN_SNR))
        wise = wise[snr_ok].copy()
        wise["w1_w2"] = wise["w1mpro"] - wise["w2mpro"]
        wise["w2_w3"] = wise["w2mpro"] - wise["w3mpro"]
        wise["source_id"] = pd.to_numeric(wise["source_id"], errors="coerce").astype("Int64")

        # Tag Quaia hits in WISE
        wise["is_quaia"] = wise["source_id"].isin(
            scores_df.loc[scores_df["label"] == 1, "source_id"]
        )
        in_box = (
            (wise["w1_w2"] >= WISE_W12_LO) & (wise["w1_w2"] <= WISE_W12_HI) &
            (wise["w2_w3"] >= WISE_W23_LO) & (wise["w2_w3"] <= WISE_W23_HI)
        )
        n_quaia_wise = wise["is_quaia"].sum()
        n_quaia_in_box = (wise["is_quaia"] & in_box).sum()
        print(f"\nWISE cross-check:")
        print(f"  Quaia hits with WISE S/N≥3: {n_quaia_wise}")
        print(f"  Quaia hits inside color box: {n_quaia_in_box} / {n_quaia_wise} "
              f"({100*n_quaia_in_box/max(n_quaia_wise,1):.0f}% completeness)")
        print(f"  Color-box sources that are Quaia: "
              f"{n_quaia_in_box} / {in_box.sum()} "
              f"({100*n_quaia_in_box/max(in_box.sum(),1):.0f}% purity)")

    # ---- Save metrics ----
    metrics = {
        "auc": auc, "ap": ap,
        "n_quaia_hits": n_hits,
        "n_test": int(len(y_te)),
        "n_quaia_test": int(y_te.sum()),
        "scale_pos_weight": pos_ratio,
        "best_iteration": booster.best_iteration,
    }
    if wise_available:
        metrics["wise_crosscheck"] = {
            "n_quaia_with_wise": int(n_quaia_wise),
            "n_quaia_in_box": int(n_quaia_in_box),
            "color_box_completeness_pct": float(100 * n_quaia_in_box / max(n_quaia_wise, 1)),
            "color_box_purity_pct": float(100 * n_quaia_in_box / max(int(in_box.sum()), 1)),
        }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ================================================================
    # PLOTS
    # ================================================================

    # 01 ROC + PR
    fpr, tpr, _ = roc_curve(y_te, prob_te)
    prec, rec, _ = precision_recall_curve(y_te, prob_te)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color=C_QSO, lw=1.8)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC  AUC = {auc:.3f}")
    axes[1].plot(rec, prec, color=C_QSO, lw=1.8)
    axes[1].axhline(y_te.mean(), color="k", linestyle="--", lw=0.8,
                    label=f"Baseline (prevalence={y_te.mean():.4f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR  AP = {ap:.3f}")
    axes[1].legend(fontsize=8)
    fig.suptitle("Quaia quasar classifier (15D) — test set", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_roc_pr.png", dpi=150)
    plt.close(fig)
    print("  saved 01_roc_pr.png")

    # 02 Feature importance
    top = fi_df.head(16)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([f.replace("feat_", "") for f in top["feature"]][::-1],
            top["gain"].values[::-1], color=C_QSO)
    ax.set_xlabel("XGBoost gain")
    ax.set_title("Feature importance — Quaia quasar classifier (15D)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_feature_importance.png", dpi=150)
    plt.close(fig)
    print("  saved 02_feature_importance.png")

    # 03 WISE color-color cross-check
    if wise_available:
        fig, ax = plt.subplots(figsize=(7, 6))
        not_quaia = ~wise["is_quaia"]
        ax.scatter(wise.loc[not_quaia, "w2_w3"], wise.loc[not_quaia, "w1_w2"],
                   s=3, alpha=0.2, color="#aaaaaa", label="ERO (WISE S/N≥3)")
        ax.scatter(wise.loc[wise["is_quaia"], "w2_w3"],
                   wise.loc[wise["is_quaia"], "w1_w2"],
                   s=25, alpha=1.0, color=C_QSO, zorder=5,
                   edgecolors="white", linewidths=0.4,
                   label=f"Quaia hit (n={int(wise['is_quaia'].sum())})")
        rect_x = [WISE_W23_LO, WISE_W23_HI, WISE_W23_HI, WISE_W23_LO, WISE_W23_LO]
        rect_y = [WISE_W12_LO, WISE_W12_LO, WISE_W12_HI, WISE_W12_HI, WISE_W12_LO]
        ax.plot(rect_x, rect_y, "b--", lw=1.5, label="Script-79 color box")
        ax.set_xlabel("W2 − W3 [Vega mag]", fontsize=11)
        ax.set_ylabel("W1 − W2 [Vega mag]", fontsize=11)
        ax.set_title("Quaia quasars on WISE colour-colour diagram", fontsize=11)
        ax.legend(fontsize=9, markerscale=2)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "03_wise_crosscheck.png", dpi=150)
        plt.close(fig)
        print("  saved 03_wise_crosscheck.png")

    # 04 + 05  UMAP overlays
    umap_overlay(EMB_16D, scores_df,
                 "15D Gaia UMAP — Quaia confirmed quasars",
                 PLOT_DIR / "04_umap16d_quaia_overlay.png")
    umap_overlay(EMB_PIX, scores_df,
                 "Pixel UMAP — Quaia confirmed quasars",
                 PLOT_DIR / "05_umap_pixel_quaia_overlay.png")

    # 06  Stamp catalog
    hits_for_stamps = hits.merge(
        df_meta[["source_id", "npz_file", "index_in_file",
                 "phot_g_mean_mag", "field_tag"]],
        on="source_id", how="left", suffixes=("", "_m"),
    ).merge(scores_df[["source_id", "xgb_score"]], on="source_id", how="left")
    if "field_tag_m" in hits_for_stamps.columns:
        hits_for_stamps["field_tag"] = hits_for_stamps["field_tag_m"].combine_first(
            hits_for_stamps["field_tag"]
        )
    stamp_catalog(hits_for_stamps, PLOT_DIR / "06_quaia_stamp_catalog")

    print(f"\nOutputs -> {OUT_DIR}")
    print(f"Plots   -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
