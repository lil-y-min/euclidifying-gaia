"""
82_quasar_redshift_analysis.py
================================

Analyses how Gaia 15D features and XGBoost classification scores vary
with redshift for confirmed Quaia quasars.

Scientific hypothesis: at low redshift the host galaxy outshines the AGN,
so Gaia measures extended-source artefacts (elevated RUWE, AEN, IPD).
At high redshift the quasar is a true point source and Gaia sees it cleanly.
If this is true, we expect:
  - RUWE / AEN / ipd_frac_multi_peak elevated at low-z
  - pm_significance elevated at low-z (host galaxy pulls centroid)
  - XGB score lower at low-z (classifier less confident)

Inputs (must exist — run 81_quaia_quasar_classifier.py first):
  output/ml_runs/quaia_clf/quaia_hits.csv     (source_id, redshift_quaia, features)
  output/ml_runs/quaia_clf/scores_all.csv     (source_id, label, xgb_score)
  output/experiments/embeddings/umap16d_manualv8_filtered/embedding_umap.csv

Outputs (plots/ml_runs/quaia_clf/):
  07_features_vs_redshift.png   Running median of key Gaia features vs z
  08_score_vs_redshift.png      XGB score trend vs z with smoothed line
  09_umap_quasars_by_redshift.png  Quasars on UMAP coloured by redshift
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema import get_feature_cols

# ======================================================================
BASE     = Path(__file__).resolve().parents[2]
OUT_DIR  = BASE / "output" / "ml_runs" / "quaia_clf"
PLOT_DIR = BASE / "plots"  / "ml_runs" / "quaia_clf"
META_DIR = BASE / "output" / "dataset_npz"
META_NAME = "metadata_16d.csv"

EMB_16D = (BASE / "output" / "experiments" / "embeddings"
           / "umap16d_manualv8_filtered" / "embedding_umap.csv")

C_QSO = "#ff007f"

# Features to plot vs redshift (most physically interpretable)
FEATURES_TO_PLOT = [
    ("feat_ruwe",                    "RUWE"),
    ("feat_astrometric_excess_noise","Astrometric excess noise"),
    ("feat_ipd_frac_multi_peak",     "IPD frac multi-peak"),
    ("feat_pm_significance",         "PM significance"),
    ("feat_parallax_over_error",     "Parallax / error"),
    ("feat_c_star",                  "C* (crowding)"),
]

SMOOTH_WINDOW = 9   # points for running-median smoothing (sorted by z)


# ======================================================================

def running_stat(z: np.ndarray, v: np.ndarray,
                 n_bins: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_centres, medians, IQR half-width) in equal-count bins."""
    order = np.argsort(z)
    z_s, v_s = z[order], v[order]
    splits = np.array_split(np.arange(len(z_s)), n_bins)
    zc, med, iqr_hw = [], [], []
    for idx in splits:
        if len(idx) < 3:
            continue
        zc.append(np.median(z_s[idx]))
        med.append(np.median(v_s[idx]))
        q25, q75 = np.percentile(v_s[idx], [25, 75])
        iqr_hw.append((q75 - q25) / 2)
    return np.array(zc), np.array(med), np.array(iqr_hw)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load quaia hits (have redshift) ----
    hits_path   = OUT_DIR / "quaia_hits.csv"
    scores_path = OUT_DIR / "scores_all.csv"
    if not hits_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Run 81_quaia_quasar_classifier.py first.")

    hits   = pd.read_csv(hits_path, low_memory=False)
    scores = pd.read_csv(scores_path, low_memory=False)
    hits["source_id"]   = pd.to_numeric(hits["source_id"],   errors="coerce").astype("Int64")
    scores["source_id"] = pd.to_numeric(scores["source_id"], errors="coerce").astype("Int64")

    # ---- Load full metadata to get raw Gaia features ----
    print("Loading ERO metadata for Gaia features...")
    feature_cols = get_feature_cols("15D")
    frames = []
    for fdir in sorted(p for p in META_DIR.iterdir()
                       if p.is_dir() and (p / META_NAME).exists()):
        df = pd.read_csv(fdir / META_NAME,
                         usecols=lambda c: c in {"source_id"} | set(feature_cols),
                         low_memory=False)
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    meta["source_id"] = pd.to_numeric(meta["source_id"], errors="coerce").astype("Int64")
    meta = meta.drop_duplicates("source_id")

    # ---- Build quasar-only dataframe with features + redshift + score ----
    quasars = (
        hits[["source_id", "redshift_quaia"]]
        .merge(scores[["source_id", "xgb_score"]], on="source_id", how="inner")
        .merge(meta, on="source_id", how="inner")
    )
    quasars["redshift_quaia"] = pd.to_numeric(quasars["redshift_quaia"], errors="coerce")
    quasars = quasars.dropna(subset=["redshift_quaia", "xgb_score"]).copy()
    print(f"Quasars with redshift + features: {len(quasars)}")

    z = quasars["redshift_quaia"].to_numpy()
    p = quasars["xgb_score"].to_numpy()

    # ======================================================================
    # Plot 07: Running median of key Gaia features vs redshift
    # ======================================================================
    nf = len(FEATURES_TO_PLOT)
    ncols = 3
    nrows = int(np.ceil(nf / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.2))
    axes = axes.flatten()

    for k, (feat, fname) in enumerate(FEATURES_TO_PLOT):
        ax = axes[k]
        if feat not in quasars.columns:
            ax.set_visible(False)
            continue

        vals = pd.to_numeric(quasars[feat], errors="coerce").to_numpy()
        good = np.isfinite(z) & np.isfinite(vals)
        if good.sum() < 10:
            ax.set_visible(False)
            continue

        zg, vg = z[good], vals[good]

        # Background scatter
        ax.scatter(zg, vg, s=8, alpha=0.35, color=C_QSO, rasterized=True)

        # Running median
        zc, med, hw = running_stat(zg, vg, n_bins=10)
        ax.plot(zc, med, color="black", lw=1.8, zorder=5)
        ax.fill_between(zc, med - hw, med + hw, alpha=0.2, color="black", zorder=4)

        ax.set_xlabel("Redshift (Quaia)", fontsize=9)
        ax.set_ylabel(fname, fontsize=9)
        ax.set_title(fname, fontsize=9, fontweight="bold")
        ax.grid(alpha=0.2)
        # Clip y-axis to p5–p95 to avoid outlier stretch
        ylo, yhi = np.nanpercentile(vg, 2), np.nanpercentile(vg, 98)
        ax.set_ylim(ylo - 0.05 * (yhi - ylo), yhi + 0.1 * (yhi - ylo))

    for k in range(nf, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        "Gaia 15D feature trends vs redshift — Quaia confirmed quasars\n"
        "(black line = running median ± IQR/2 in equal-count bins)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_features_vs_redshift.png", dpi=150)
    plt.close(fig)
    print("  saved 07_features_vs_redshift.png")

    # ======================================================================
    # Plot 08: XGB score vs redshift with smoothed trend
    # ======================================================================
    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter
    ax.scatter(z, p, s=18, alpha=0.5, color=C_QSO, edgecolors="none",
               rasterized=True, label="Quaia quasars")

    # Running median + IQR
    good = np.isfinite(z) & np.isfinite(p)
    zc, med, hw = running_stat(z[good], p[good], n_bins=10)
    ax.plot(zc, med, color="black", lw=2, zorder=5, label="Running median")
    ax.fill_between(zc, med - hw, med + hw, alpha=0.2, color="black",
                    zorder=4, label="IQR/2 band")

    ax.axhline(0.5, color="gray", lw=0.8, linestyle="--", label="p = 0.5")
    ax.set_xlabel("Photometric redshift (Quaia)", fontsize=11)
    ax.set_ylabel("XGB P(quasar) — 15D classifier", fontsize=11)
    ax.set_title("Classifier score vs redshift — does host galaxy contaminate at low z?",
                 fontsize=10)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_score_vs_redshift.png", dpi=150)
    plt.close(fig)
    print("  saved 08_score_vs_redshift.png")

    # ======================================================================
    # Plot 09: Quasars on 15D UMAP coloured by redshift
    # ======================================================================
    if not EMB_16D.exists():
        print(f"  UMAP embedding not found at {EMB_16D} — skipping plot 09")
        return

    umap = pd.read_csv(EMB_16D, usecols=["source_id", "x", "y"], low_memory=False)
    umap["source_id"] = pd.to_numeric(umap["source_id"], errors="coerce").astype("Int64")
    umap = umap.merge(
        quasars[["source_id", "redshift_quaia"]],
        on="source_id", how="left",
    )
    in_q   = umap["redshift_quaia"].notna()
    has_z  = in_q & umap["redshift_quaia"].notna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) all quasars highlighted
    ax = axes[0]
    ax.scatter(umap.loc[~in_q, "x"], umap.loc[~in_q, "y"],
               s=0.5, alpha=0.06, color="#555555", rasterized=True)
    ax.scatter(umap.loc[in_q, "x"], umap.loc[in_q, "y"],
               s=25, alpha=1.0, color=C_QSO, zorder=5,
               edgecolors="white", linewidths=0.4,
               label=f"Quaia quasars (n={in_q.sum()})")
    ax.set_title("(a) Quaia quasar positions", fontsize=10)
    ax.legend(markerscale=1.5, fontsize=8, framealpha=0.85, edgecolor="none")
    ax.set_xticks([]); ax.set_yticks([])

    # (b) coloured by redshift
    ax = axes[1]
    ax.scatter(umap.loc[~in_q, "x"], umap.loc[~in_q, "y"],
               s=0.5, alpha=0.06, color="#555555", rasterized=True)
    z_vals = umap.loc[has_z, "redshift_quaia"].to_numpy()
    sc = ax.scatter(
        umap.loc[has_z, "x"], umap.loc[has_z, "y"],
        s=30, c=z_vals, cmap="plasma",
        vmin=0, vmax=4, alpha=0.95, zorder=5,
        edgecolors="white", linewidths=0.3, rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="Photometric redshift")
    ax.set_title("(b) Coloured by redshift — do low-z/high-z cluster differently?",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("15D Gaia UMAP — Quaia quasars coloured by redshift", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_umap_quasars_by_redshift.png", dpi=150)
    plt.close(fig)
    print("  saved 09_umap_quasars_by_redshift.png")

    print(f"\nOutputs -> {OUT_DIR}")
    print(f"Plots   -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
