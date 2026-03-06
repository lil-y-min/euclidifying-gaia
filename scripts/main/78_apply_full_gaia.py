"""
78_apply_full_gaia.py
=====================

Apply the trained morphology regressors to ALL Gaia sources in the 17 ERO
fields that have valid 16D features and phot_g_mean_mag >= 17 — regardless
of whether they have a Euclid stamp or a split_code.

This turns the regressors into a practical tool: predicted morphology metrics
for every eligible Gaia source in the survey footprint.

A composite "morphological weirdness" score is computed as:
  weirdness = 0.5 * gini_rank + 0.3 * multipeak_rank + 0.2 * ellipticity_rank
  (each rank is the percentile rank within the full predicted set, 0–1)

Outputs:
  output/ml_runs/morph_metrics_16d/
    full_gaia_predictions.csv        — all sources with predicted metrics + score
    full_gaia_top500_weird.csv       — top 500 by weirdness score

  plots/ml_runs/morph_metrics_16d/
    20_full_gaia_pred_distributions.png  — predicted vs labeled test set distributions
    21_full_gaia_weirdness_scatter.png   — predicted gini vs multipeak, colored by field
    22_full_gaia_top_weird_flags.png     — Gaia flag values for top 1% weirdest sources
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xgboost as xgb
from scipy.stats import rankdata

import importlib
_mod = importlib.import_module("69_morph_metrics_xgb")
CFG          = _mod.CFG
METRIC_NAMES = _mod.METRIC_NAMES

from feature_schema import get_feature_cols


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR    = CFG.base_dir
MODELS_DIR  = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "models"
MAG_CACHE   = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "mag_cache"
OUTPUT_DIR  = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d"
PLOTS_DIR   = BASE_DIR / "plots"  / "ml_runs" / "morph_metrics_16d"

GAIA_DISPLAY_COLS = [
    "ruwe", "ipd_frac_multi_peak", "astrometric_excess_noise",
    "phot_g_mean_mag",
]


# ======================================================================
# HELPERS
# ======================================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def load_boosters(feature_cols: List[str]) -> List[xgb.Booster]:
    fnames = [f"f{i}" for i in range(len(feature_cols))]
    boosters = []
    for name in METRIC_NAMES:
        b = xgb.Booster()
        b.load_model(str(MODELS_DIR / f"booster_{name}.json"))
        b.feature_names = fnames
        boosters.append(b)
    return boosters


# ======================================================================
# FULL GAIA LOADING (no stamp / split_code requirement)
# ======================================================================

def load_full_gaia(
    cfg, feature_cols: List[str]
) -> pd.DataFrame:
    """
    Load ALL Gaia sources in ERO fields with valid 16D features and mag >= 17.
    No stamp requirement. Returns a DataFrame with scaled X features + metadata.
    """
    gaia_min, gaia_iqr = _mod.load_gaia_scaler(cfg.gaia_scaler_npz, feature_cols)
    field_dirs = _mod.list_field_dirs(cfg.dataset_root)
    print(f"Found {len(field_dirs)} field directories.")

    rows = []
    for fdir in field_dirs:
        field_tag = fdir.name
        meta_path = fdir / _mod.RAW_META_16D
        meta = pd.read_csv(meta_path, engine="python", on_bad_lines="skip")
        nrows = len(meta)

        # Mag filter
        g = pd.to_numeric(meta["phot_g_mean_mag"], errors="coerce").to_numpy()
        mag_ok = np.isfinite(g) & (g >= cfg.gaia_g_mag_min)

        # Feature validity
        feat_raw = meta[feature_cols].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        feat_ok = np.all(np.isfinite(feat_raw), axis=1)

        valid = mag_ok & feat_ok
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            continue

        # Scale features
        X_field = (
            (feat_raw[valid_idx].astype(np.float32) - gaia_min)
            / np.where(gaia_iqr > 0, gaia_iqr, 1.0)
        )

        # Gather metadata row
        sub = meta.iloc[valid_idx].copy()
        sub["field_name"] = field_tag
        for j, col in enumerate(feature_cols):
            sub[f"_X{j}"] = X_field[:, j]

        # Keep useful Gaia display columns
        for col in GAIA_DISPLAY_COLS:
            if col not in sub.columns:
                sub[col] = np.nan

        rows.append(sub)
        print(f"  [{field_tag}]  {len(valid_idx)} sources")

    df = pd.concat(rows, ignore_index=True)
    print(f"\nTotal: {len(df)} Gaia sources with valid 16D features & mag >= 17")
    return df, feature_cols, gaia_min, gaia_iqr


# ======================================================================
# PLOTS
# ======================================================================

def plot_pred_distributions(
    Y_pred_full: np.ndarray,
    Y_test_labeled: np.ndarray,
    out_path: Path,
) -> None:
    ncols = 4
    nrows = (len(METRIC_NAMES) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, name in enumerate(METRIC_NAMES):
        ax = axes[i]
        yf = Y_pred_full[:, i]
        yl = Y_test_labeled[:, i]
        ok_f = np.isfinite(yf)
        ok_l = np.isfinite(yl)

        lo = min(np.percentile(yf[ok_f], 1), np.percentile(yl[ok_l], 1))
        hi = max(np.percentile(yf[ok_f], 99), np.percentile(yl[ok_l], 99))
        bins = np.linspace(lo, hi, 60)

        ax.hist(yl[ok_l], bins=bins, density=True, alpha=0.55,
                color="steelblue", label="Labeled test set", edgecolor="none")
        ax.hist(yf[ok_f], bins=bins, density=True, alpha=0.55,
                color="darkorange", label="Full Gaia (predicted)", edgecolor="none")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted value", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(METRIC_NAMES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Predicted morphology: full Gaia set vs labeled test set",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    savefig(fig, out_path)


def plot_weirdness_scatter(
    df_pred: pd.DataFrame,
    out_path: Path,
) -> None:
    fields = df_pred["field_name"].unique()
    cmap   = cm.get_cmap("tab20", len(fields))
    field_color = {f: cmap(i) for i, f in enumerate(fields)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: gini vs multipeak_ratio
    ax = axes[0]
    for field in fields:
        sub = df_pred[df_pred["field_name"] == field]
        ax.scatter(sub["pred_gini"], sub["pred_multipeak_ratio"],
                   s=2, alpha=0.3, color=field_color[field], rasterized=True)
    ax.set_xlabel("Predicted gini", fontsize=10)
    ax.set_ylabel("Predicted multipeak_ratio", fontsize=10)
    ax.set_title("All Gaia sources: gini vs multipeak", fontsize=10)

    # Legend (fields)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=field_color[f], markersize=6, label=f.replace("ERO-", ""))
        for f in fields
    ]
    ax.legend(handles=handles, fontsize=6, ncol=2, loc="upper right")

    # Right: gini vs ellipticity, colored by weirdness
    ax2 = axes[1]
    sc = ax2.scatter(
        df_pred["pred_gini"], df_pred["pred_ellipticity"],
        c=df_pred["weirdness_score"], cmap="hot_r",
        s=2, alpha=0.4, rasterized=True,
    )
    plt.colorbar(sc, ax=ax2, label="Weirdness score")
    ax2.set_xlabel("Predicted gini", fontsize=10)
    ax2.set_ylabel("Predicted ellipticity", fontsize=10)
    ax2.set_title("Weirdness score (gini × multipeak × ellipticity rank)", fontsize=10)

    fig.tight_layout()
    savefig(fig, out_path)


def plot_top_weird_flags(
    df_top: pd.DataFrame,
    out_path: Path,
) -> None:
    flag_cols = ["ruwe", "ipd_frac_multi_peak", "astrometric_excess_noise"]
    flag_labels = ["RUWE", "ipd_frac_multi_peak (%)", "AEN (mas)"]
    thresholds  = [1.4, 10.0, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col, label, thresh in zip(axes, flag_cols, flag_labels, thresholds):
        vals = pd.to_numeric(df_top[col], errors="coerce").dropna()
        clip = np.percentile(vals, 99) if len(vals) > 0 else vals.max()
        ax.hist(np.clip(vals, 0, clip), bins=40, color="tomato",
                edgecolor="none", alpha=0.8)
        ax.axvline(thresh, color="black", lw=1.5, ls="--",
                   label=f"threshold = {thresh}")
        frac_above = (vals > thresh).mean()
        ax.set_title(f"{label}\n{frac_above:.1%} above threshold", fontsize=9)
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Gaia quality flags for top 1% weirdest sources (n={len(df_top)})",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    feature_cols = get_feature_cols(CFG.feature_set)

    # ---- Load full Gaia set ----
    print("Loading full Gaia catalog from ERO fields ...")
    df_full, feature_cols, gaia_min, gaia_iqr = load_full_gaia(CFG, feature_cols)

    # Reconstruct X matrix
    X_cols = [f"_X{j}" for j in range(len(feature_cols))]
    X_full = df_full[X_cols].to_numpy(dtype=np.float32)

    # ---- Load boosters & predict ----
    print("\nLoading boosters ...")
    boosters = load_boosters(feature_cols)
    fnames   = [f"f{i}" for i in range(len(feature_cols))]
    dm       = xgb.DMatrix(X_full, feature_names=fnames)

    print("Predicting all 11 metrics ...")
    Y_pred = np.column_stack([b.predict(dm) for b in boosters])

    # Add predictions to DataFrame
    for i, name in enumerate(METRIC_NAMES):
        df_full[f"pred_{name}"] = Y_pred[:, i]

    # ---- Weirdness score ----
    n = len(df_full)
    gini_rank   = rankdata(df_full["pred_gini"]) / n
    multi_rank  = rankdata(df_full["pred_multipeak_ratio"]) / n
    ell_rank    = rankdata(df_full["pred_ellipticity"]) / n
    df_full["weirdness_score"] = (
        0.5 * gini_rank + 0.3 * multi_rank + 0.2 * ell_rank
    )

    # ---- Save full CSV ----
    save_cols = (
        ["source_id", "field_name", "phot_g_mean_mag"] +
        [c for c in GAIA_DISPLAY_COLS if c != "phot_g_mean_mag"] +
        [f"pred_{m}" for m in METRIC_NAMES] +
        ["weirdness_score"]
    )
    save_cols = [c for c in save_cols if c in df_full.columns]
    out_csv   = OUTPUT_DIR / "full_gaia_predictions.csv"
    df_full[save_cols].to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(df_full)} rows)")

    # ---- Top 500 weirdest ----
    df_top500 = df_full.nlargest(500, "weirdness_score")[save_cols]
    df_top500.to_csv(OUTPUT_DIR / "full_gaia_top500_weird.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'full_gaia_top500_weird.csv'}")

    # Top 1% for flag plot
    n_top1pct = max(1, int(0.01 * n))
    df_top1pct = df_full.nlargest(n_top1pct, "weirdness_score")

    # ---- Load labeled test Y for comparison ----
    print("\nLoading labeled test Y for distribution comparison ...")
    Y_test_labeled = np.load(MAG_CACHE / "Y.npy")[
        np.load(MAG_CACHE / "splits.npy") == 2
    ]

    # ---- Plots ----
    print("\nGenerating plots ...")
    plot_pred_distributions(
        Y_pred, Y_test_labeled,
        PLOTS_DIR / "20_full_gaia_pred_distributions.png",
    )
    plot_weirdness_scatter(
        df_full,
        PLOTS_DIR / "21_full_gaia_weirdness_scatter.png",
    )
    plot_top_weird_flags(
        df_top1pct,
        PLOTS_DIR / "22_full_gaia_top_weird_flags.png",
    )

    # Summary stats
    print("\n=== WEIRDNESS SCORE SUMMARY ===")
    print(f"Total sources: {n}")
    print(f"Top 500 pred_gini range:  [{df_top500['pred_gini'].min():.4f}, {df_top500['pred_gini'].max():.4f}]")
    print(f"Top 500 pred_multipeak range: [{df_top500['pred_multipeak_ratio'].min():.4f}, {df_top500['pred_multipeak_ratio'].max():.4f}]")
    print(f"\nTop 10 weirdest sources:")
    top10_cols = ["field_name", "phot_g_mean_mag", "pred_gini",
                  "pred_multipeak_ratio", "pred_ellipticity", "weirdness_score"]
    top10_cols = [c for c in top10_cols if c in df_full.columns]
    print(df_full.nlargest(10, "weirdness_score")[top10_cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
