"""
76b_galaxy_environment_deep_analysis.py
=========================================

Three follow-up analyses for the inside/outside galaxy classifier (script 76):

  1. Radial XGB score profile — mean P(inside) vs angular separation per galaxy.
     If the model learned genuine environment, score should decay with radius
     and drop at the D25 boundary. Tests whether signal is real or a scanning-law
     confound.

  2. Within-field retrain — classifier using ONLY sources inside big-galaxy fields
     (inside-D25 vs outside-D25 within the same field). Removes the field-identity
     confound from Gaia's scanning law (visibility_periods_used, ipd_gof_harmonic_phase).

  3. UMAP overlay — inside-galaxy labels + XGB scores on existing 16D and pixel UMAPs.

Outputs (all in plots/ml_runs/galaxy_env_clf/):
  10_radial_score_profile.png
  11_within_field_roc_pr.png
  12_within_field_feature_importance.png
  13_umap16d_galaxy_env_overlay.png
  14_umap_pixel_galaxy_env_overlay.png
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
# CONFIG — must match script 76
# ======================================================================

BASE         = Path(__file__).resolve().parents[2]
DATASET_ROOT = BASE / "output" / "dataset_npz"
OUT_DIR      = BASE / "output" / "ml_runs" / "galaxy_env_clf"
PLOT_DIR     = BASE / "plots"  / "ml_runs" / "galaxy_env_clf"
META_NAME    = "metadata_16d.csv"
FEATURE_SET  = "16D"

BIG_GALAXY_TABLE = [
    ("IC342",       56.6988,    68.0961,   10.7,   2.7,   "ERO-IC342"),
    ("NGC6744",    287.4421,   -63.8575,    7.75,  2.0,   "ERO-NGC6744"),
    ("NGC2403",    114.2142,    65.6031,   10.95,  3.0,   "ERO-NGC2403"),
    ("NGC6822",    296.2404,   -14.8031,    7.75,  3.0,   "ERO-NGC6822"),
    ("HolmbergII", 124.7708,    70.7219,    3.95,  1.2,   "ERO-HolmbergII"),
    ("IC10",         5.0721,    59.3039,    3.15,  1.0,   "ERO-IC10"),
]

OUTSIDE_FIELDS = {
    "ERO-Abell2390", "ERO-Abell2764", "ERO-Barnard30",
    "ERO-Horsehead", "ERO-Messier78", "ERO-Taurus",
    "ERO-NGC6397", "ERO-NGC6254", "ERO-Perseus",
}

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

EMB_16D = (BASE / "output" / "experiments" / "embeddings"
           / "umap16d_manualv8_filtered" / "embedding_umap.csv")
EMB_PIX = (BASE / "output" / "experiments" / "embeddings"
           / "double_stars_pixels"
           / "pixels_umap_standard_manualv8_filtered"
           / "umap_standard" / "embedding_umap.csv")

C_IN   = "#e8671b"
C_OUT  = "#4b8ec4"
C_GREY = "#555555"


# ======================================================================
# HELPERS
# ======================================================================

def load_scaler(feature_cols):
    stem = scaler_stem(FEATURE_SET)
    npz  = BASE / "output" / "scalers" / f"{stem}.npz"
    d    = np.load(npz, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    idx  = np.array([names.index(c) for c in feature_cols], dtype=int)
    return d["y_min"].astype(np.float32)[idx], d["y_iqr"].astype(np.float32)[idx]


def load_all_fields(feature_cols):
    frames = []
    for fdir in sorted(p for p in DATASET_ROOT.iterdir()
                       if p.is_dir() and (p / META_NAME).exists()):
        df = pd.read_csv(fdir / META_NAME, low_memory=False)
        if "field_tag" not in df.columns:
            df["field_tag"] = fdir.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def assign_labels(df):
    galaxy_tags = {g[5] for g in BIG_GALAXY_TABLE}
    keep = df["field_tag"].isin(galaxy_tags | OUTSIDE_FIELDS)
    df = df[keep].copy()
    df["label"]       = 0
    df["sep_arcmin"]  = np.nan
    df["galaxy"]      = ""

    src_coords = SkyCoord(ra=df["ra"].to_numpy() * u.deg,
                          dec=df["dec"].to_numpy() * u.deg, frame="icrs")
    for name, ra, dec, r_d25, r_eff, tag in BIG_GALAXY_TABLE:
        gc   = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        mask = df["field_tag"] == tag
        if not mask.any():
            continue
        seps = src_coords[mask.to_numpy()].separation(gc).to(u.arcmin).value
        idx  = df.index[mask]
        df.loc[idx, "sep_arcmin"] = seps
        df.loc[idx, "galaxy"]     = name
        df.loc[idx, "label"]      = (seps < r_d25).astype(int)
    return df


# ======================================================================
# 1. RADIAL XGB SCORE PROFILE
# ======================================================================

def plot_radial_profile(df, booster, X_scaled, feat_min, feat_iqr, feature_cols):
    """Plot mean XGB P(inside) vs angular separation, one line per galaxy."""
    df = df.copy()
    df["xgb_score"] = booster.predict(xgb.DMatrix(X_scaled, feature_names=feature_cols))

    galaxy_only = df[df["galaxy"] != ""].copy()
    galaxies    = [g for g in galaxy_only["galaxy"].unique() if galaxy_only[galaxy_only["galaxy"]==g]["sep_arcmin"].notna().any()]

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (name, ra, dec, r_d25, r_eff, tag) in enumerate(BIG_GALAXY_TABLE):
        ax  = axes[i]
        sub = galaxy_only[galaxy_only["galaxy"] == name].dropna(subset=["sep_arcmin"])
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        # Bin by separation
        max_sep  = sub["sep_arcmin"].quantile(0.98)
        bins     = np.linspace(0, max_sep, 25)
        bin_mid  = 0.5 * (bins[:-1] + bins[1:])
        bin_idx  = np.digitize(sub["sep_arcmin"], bins) - 1
        bin_idx  = np.clip(bin_idx, 0, len(bins) - 2)

        mean_score = np.full(len(bin_mid), np.nan)
        std_score  = np.full(len(bin_mid), np.nan)
        counts     = np.zeros(len(bin_mid), dtype=int)
        for b in range(len(bin_mid)):
            sel = sub["xgb_score"].values[bin_idx == b]
            if len(sel) >= 3:
                mean_score[b] = sel.mean()
                std_score[b]  = sel.std() / np.sqrt(len(sel))
                counts[b]     = len(sel)

        valid = np.isfinite(mean_score)
        color = cmap(i % 10)
        ax.plot(bin_mid[valid], mean_score[valid], color=color, lw=2)
        ax.fill_between(bin_mid[valid],
                        mean_score[valid] - std_score[valid],
                        mean_score[valid] + std_score[valid],
                        alpha=0.25, color=color)
        ax.axvline(r_d25, color="red", linestyle="--", lw=1.5, label=f"D25 = {r_d25:.1f}'")
        ax.axvline(r_eff, color="gray", linestyle=":", lw=1.2, label=f"r_eff = {r_eff:.1f}'")
        ax.set_xlabel("Angular separation from nucleus (arcmin)", fontsize=9)
        ax.set_ylabel("Mean XGB P(inside)", fontsize=9)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Radial XGB score profile — does P(inside) decay with separation?\n"
        "Red dashed = D25 boundary, gray dotted = r_eff",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "10_radial_score_profile.png", dpi=150)
    plt.close(fig)
    print("  saved 10_radial_score_profile.png")


# ======================================================================
# 2. WITHIN-FIELD RETRAIN
# ======================================================================

def within_field_analysis(df, feat_min, feat_iqr, feature_cols):
    """
    Retrain using ONLY sources in big-galaxy fields (inside vs outside D25
    within the same field). Eliminates scanning-law confound.
    """
    galaxy_tags = {g[5] for g in BIG_GALAXY_TABLE}
    df_wf = df[df["field_tag"].isin(galaxy_tags)].copy()
    df_wf = df_wf.dropna(subset=feature_cols + ["sep_arcmin"]).copy()

    print(f"  Within-field: {len(df_wf)} sources")
    print(f"  Inside (label=1): {df_wf['label'].sum()}")
    print(f"  Outside (label=0): {(df_wf['label']==0).sum()}")

    X = df_wf[feature_cols].to_numpy(dtype=np.float32)
    X = (X - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)
    y = df_wf["label"].to_numpy(dtype=np.float32)
    splits = df_wf["split_code"].to_numpy(dtype=int) if "split_code" in df_wf.columns else np.zeros(len(df_wf), dtype=int)

    X_tr, y_tr = X[splits == 0], y[splits == 0]
    X_vl, y_vl = X[splits == 1], y[splits == 1]
    X_te, y_te = X[splits == 2], y[splits == 2]
    print(f"  Train={len(X_tr)}, Val={len(X_vl)}, Test={len(X_te)}")

    if y_tr.sum() < 5 or y_te.sum() < 3:
        print("  [SKIP] too few positives after within-field split.")
        return None, None

    pos_ratio = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    params    = {**PARAMS, "scale_pos_weight": pos_ratio}

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_vl, label=y_vl, feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_te, label=y_te, feature_names=feature_cols)

    booster_wf = xgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=100,
    )
    booster_wf.save_model(str(OUT_DIR / "model_within_field.json"))

    prob_te = booster_wf.predict(dtest)
    auc = roc_auc_score(y_te, prob_te)
    ap  = average_precision_score(y_te, prob_te)
    print(f"  Within-field AUC={auc:.4f}  AP={ap:.4f}")

    # ROC + PR
    fpr, tpr, _ = roc_curve(y_te, prob_te)
    prec, rec, _ = precision_recall_curve(y_te, prob_te)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color=C_IN, lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC  AUC = {auc:.3f}")
    axes[1].plot(rec, prec, color=C_IN, lw=2)
    axes[1].axhline(y_te.mean(), color="k", linestyle="--", lw=0.8,
                    label=f"Baseline = {y_te.mean():.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR  AP = {ap:.3f}")
    axes[1].legend(fontsize=8)
    fig.suptitle("Within-field classifier (no scanning-law confound)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "11_within_field_roc_pr.png", dpi=150)
    plt.close(fig)
    print("  saved 11_within_field_roc_pr.png")

    # Feature importance
    fi = booster_wf.get_score(importance_type="gain")
    fi_df = pd.DataFrame({"feature": list(fi.keys()), "gain": list(fi.values())})
    fi_df = fi_df.sort_values("gain", ascending=False)
    top = fi_df.head(16)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([f.replace("feat_", "") for f in top["feature"]][::-1],
            top["gain"].values[::-1], color=C_IN)
    ax.set_xlabel("XGBoost gain")
    ax.set_title("Feature importance — within-field classifier\n"
                 "(scanning-law confound removed)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "12_within_field_feature_importance.png", dpi=150)
    plt.close(fig)
    print("  saved 12_within_field_feature_importance.png")

    return booster_wf, auc


# ======================================================================
# 3. UMAP OVERLAYS
# ======================================================================

def umap_overlay(emb_path, scores_df, title, out_png):
    umap = pd.read_csv(emb_path, usecols=["source_id", "x", "y"], low_memory=False)
    umap["source_id"] = pd.to_numeric(umap["source_id"], errors="coerce").astype("Int64")
    merged = umap.merge(scores_df, on="source_id", how="left")

    in_gal = merged["label"] == 1
    has_s  = merged["xgb_score"].notna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(merged.loc[~in_gal, "x"], merged.loc[~in_gal, "y"],
               s=0.5, alpha=0.06, color=C_GREY, rasterized=True)
    ax.scatter(merged.loc[in_gal, "x"], merged.loc[in_gal, "y"],
               s=12, alpha=0.9, color=C_IN, rasterized=True, zorder=5,
               edgecolors="white", linewidths=0.3,
               label=f"Inside galaxy disk (n={in_gal.sum()})")
    ax.set_title("(a) Inside-galaxy label", fontsize=10)
    ax.legend(markerscale=2, fontsize=8, framealpha=0.85, edgecolor="none")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.scatter(merged.loc[~has_s, "x"], merged.loc[~has_s, "y"],
               s=0.5, alpha=0.06, color=C_GREY, rasterized=True)
    sc = ax.scatter(merged.loc[has_s, "x"], merged.loc[has_s, "y"],
                    s=2, c=merged.loc[has_s, "xgb_score"],
                    cmap="RdPu", alpha=0.8, vmin=0, vmax=1,
                    rasterized=True, zorder=4)
    plt.colorbar(sc, ax=ax, label="XGB P(inside galaxy)")
    ax.set_title("(b) XGB predicted P(inside galaxy)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  saved {out_png.name}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols(FEATURE_SET)
    feat_min, feat_iqr = load_scaler(feature_cols)

    print("Loading metadata and assigning labels...")
    df_raw = load_all_fields(feature_cols)
    df     = assign_labels(df_raw)
    df     = df.dropna(subset=feature_cols + ["ra", "dec"]).copy()
    print(f"  Total: {len(df)}, inside={df['label'].sum()}, outside={(df['label']==0).sum()}")

    # Load original booster from script 76
    booster_path = OUT_DIR / "model_inside_galaxy.json"
    if not booster_path.exists():
        raise FileNotFoundError("Run script 76 first to generate model_inside_galaxy.json")
    booster = xgb.Booster()
    booster.load_model(str(booster_path))

    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    X_all_scaled = (X_all - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)

    # Scores DataFrame for UMAP overlay
    df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype("Int64")
    scores_df = pd.DataFrame({
        "source_id": df["source_id"].to_numpy(),
        "label":     df["label"].to_numpy(dtype=int),
        "xgb_score": booster.predict(xgb.DMatrix(X_all_scaled, feature_names=feature_cols)),
    })
    scores_df["source_id"] = scores_df["source_id"].astype("Int64")

    # ---- 1. Radial profile ----
    print("\n[1] Radial XGB score profile...")
    plot_radial_profile(df, booster, X_all_scaled, feat_min, feat_iqr, feature_cols)

    # ---- 2. Within-field retrain ----
    print("\n[2] Within-field retrain (confound-free)...")
    within_field_analysis(df, feat_min, feat_iqr, feature_cols)

    # ---- 3. UMAP overlays ----
    print("\n[3] UMAP overlays...")
    umap_overlay(EMB_16D, scores_df,
                 "16D Gaia UMAP — inside-galaxy label overlay",
                 PLOT_DIR / "13_umap16d_galaxy_env_overlay.png")
    umap_overlay(EMB_PIX, scores_df,
                 "Pixel UMAP — inside-galaxy label overlay",
                 PLOT_DIR / "14_umap_pixel_galaxy_env_overlay.png")

    print(f"\nDone. Plots -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
