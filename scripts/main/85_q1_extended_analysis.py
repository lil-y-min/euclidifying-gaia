"""
85_q1_extended_analysis.py
===========================

Extended analysis of the Euclid Q1 quasar classification results (requires
script 84 to have been run first with --pull_wsdb --quaia --train --plots).

Extensions
----------
--ext1   Redshift stratification
         Quasars exist at different distances (redshifts). Does our Gaia-based
         classifier score degrade for more distant quasars? We use 3,127 Quaia
         confirmed quasars in Q1 (which have known redshifts) to answer this.

--ext2   Euclid morphology vs Gaia score
         Euclid's own pipeline gives each source a point_like_prob (how much
         it looks like a point, i.e. a star or quasar) from VIS imaging. We
         compare this to our XGB score derived purely from Gaia numbers —
         two completely independent measurements of the same thing.

--ext3   UMAP overlay of Q1 on ERO feature space
         UMAP is a dimensionality reduction algorithm that squishes 15 numbers
         per source down to 2 for plotting. We fit the map on ERO data, then
         project Q1 sources onto it to see if Q1 occupies the same regions.

--ext6   Magnitude-dependent performance
         Gaia measures brightness as G magnitude (lower = brighter, range ~17–21).
         Faint sources have noisier measurements. Does classifier performance
         degrade for faint objects?

--all    Run all four extensions.

Usage
-----
  python 85_q1_extended_analysis.py --ext1
  python 85_q1_extended_analysis.py --ext2
  python 85_q1_extended_analysis.py --ext3      # slow: ~5 min for UMAP fit
  python 85_q1_extended_analysis.py --ext6
  python 85_q1_extended_analysis.py --all

Outputs (plots/ml_runs/quasar_q1/)
-----------------------------------
  ext1_redshift_performance.png       Score vs redshift + AUC per z-bin (ERO vs Q1)
  ext2_01_score_vs_pointlike.png      XGB score vs Euclid point_like_prob
  ext2_02_morphology_agreement.png    Binned agreement analysis
  ext3_01_umap_q1_overlay.png         Q1 sources projected onto ERO UMAP
  ext6_01_performance_vs_magnitude.png  AUC/AP vs G magnitude
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema import get_feature_cols, scaler_stem

# ======================================================================
# CONFIG
# ======================================================================

BASE        = Path(__file__).resolve().parents[2]
Q1_DIR      = BASE / "output" / "ml_runs" / "quasar_q1"
ERO_WISE    = BASE / "output" / "ml_runs" / "quasar_wise_clf"
PLOT_DIR    = BASE / "report" / "model_decision" / "20260306_q1_quasar_extended"
EMB_ERO     = (BASE / "output" / "experiments" / "embeddings"
               / "umap16d_manualv8_filtered" / "embedding_umap.csv")

FEATURE_SET = "15D"

# Magnitude bins for extension 6
MAG_BINS    = [17.0, 18.0, 19.0, 20.0, 21.0]

# Redshift bins for extension 1

# Colours
C_Q1    = "#7b2d8b"
C_ERO   = "#2196F3"
C_QSO   = "#ff007f"
C_WISE  = "#e8671b"

MIN_SOURCES_FOR_AUC = 20   # skip a bin if fewer than this many positives


# ======================================================================
# SHARED HELPERS
# ======================================================================

def load_scaler(feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    stem = scaler_stem(FEATURE_SET)
    npz  = BASE / "output" / "scalers" / f"{stem}.npz"
    d    = np.load(npz, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    y_min = d["y_min"].astype(np.float32)
    y_iqr = d["y_iqr"].astype(np.float32)
    idx   = np.array([names.index(c) for c in feature_cols], dtype=int)
    return y_min[idx], y_iqr[idx]


def normalize(X: np.ndarray, feat_min: np.ndarray, feat_iqr: np.ndarray) -> np.ndarray:
    return (X.astype(np.float32) - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)


def load_and_score_q1(feature_cols, feat_min, feat_iqr) -> pd.DataFrame:
    """
    Load all Q1 sources with 15D features and score them with the Q1-trained
    XGB model. Returns a DataFrame with source_id, xgb_score, phot_g_mean_mag,
    point_like_prob, extended_prob.
    """
    print("Loading Q1 Gaia features...")
    df = pd.read_csv(Q1_DIR / "q1_gaia_features.csv", low_memory=False)
    df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype("Int64")

    ok = df[feature_cols].notna().all(axis=1)
    df_ok = df[ok].copy()

    X  = normalize(df_ok[feature_cols].to_numpy(), feat_min, feat_iqr)
    bst = xgb.Booster()
    bst.load_model(str(Q1_DIR / "model_q1_quasar.json"))
    prob = bst.predict(xgb.DMatrix(X, feature_names=feature_cols))

    df_ok["xgb_score"] = prob
    cols_keep = ["source_id", "xgb_score", "phot_g_mean_mag",
                 "point_like_prob", "extended_prob"]
    cols_keep = [c for c in cols_keep if c in df_ok.columns]
    print(f"  Scored {len(df_ok):,} Q1 sources.")
    return df_ok[cols_keep + feature_cols]


def load_wise_labels() -> pd.DataFrame:
    """
    Load the WISE-labelled Q1 subset (sources with AllWISE S/N≥3 photometry).
    Returns source_id, label (1=quasar-locus, 0=not), w1_w2, w2_w3.
    """
    wise = pd.read_csv(Q1_DIR / "q1_wise_phot.csv", low_memory=False)
    wise["source_id"] = pd.to_numeric(wise["source_id"], errors="coerce").astype("Int64")
    for c in ["w1mpro", "w2mpro", "w3mpro", "w1snr", "w2snr", "w3snr"]:
        wise[c] = pd.to_numeric(wise[c], errors="coerce")
    snr = ((wise["w1snr"] >= 3) & (wise["w2snr"] >= 3) & (wise["w3snr"] >= 3))
    wise = wise[snr].copy()
    wise["w1_w2"] = wise["w1mpro"] - wise["w2mpro"]
    wise["w2_w3"] = wise["w2mpro"] - wise["w3mpro"]
    wise["label"] = (
        (wise["w1_w2"] >= 0.6) & (wise["w1_w2"] <= 1.6) &
        (wise["w2_w3"] >= 2.0) & (wise["w2_w3"] <= 4.2)
    ).astype(int)
    return wise[["source_id", "label", "w1_w2", "w2_w3"]]


# ======================================================================
# EXTENSION 1: REDSHIFT STRATIFICATION (ERO vs Q1)
# ======================================================================

# Redshift bins — merge z>3 into one bin (tail too sparse for reliable stats)
Z_BINS_EXT1   = [0.0, 1.0, 2.0, 3.0, 7.0]
Z_LABELS_EXT1 = ["z < 1", "1 ≤ z < 2", "2 ≤ z < 3", "z ≥ 3"]

ERO_QUAIA_HITS  = BASE / "output" / "ml_runs" / "quaia_clf"       / "quaia_hits.csv"
ERO_WISE_SCORES = BASE / "output" / "ml_runs" / "quasar_wise_clf" / "scores_all.csv"


def _optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Find the decision threshold that maximises Youden's J = TPR - FPR.
    This is the principled way to set a threshold for imbalanced data:
    it finds the single operating point that maximises true positives
    while minimising false positives simultaneously.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def run_ext1(df_scored: pd.DataFrame) -> None:
    """
    Does the classifier score distribution change with redshift?

    Binary recall is saturated at ~1 for both surveys (the model is too good),
    so instead we show the raw score distributions per z-bin as box plots.
    Even if all quasars score above the threshold, the distribution may shift
    downward at high-z — meaning the model is less confident, even if it still
    technically classifies them correctly.

    Each z-bin shows two side-by-side box plots: Q1 (purple) and ERO (blue).
    Individual points are jittered on top. The Youden-optimal threshold for
    each survey is drawn as a dashed horizontal line.
    """
    print("\n--- Extension 1: Score distributions by redshift, ERO vs Q1 ---")

    # ---- Q1 ----
    wise_q1    = load_wise_labels()
    df_wise_q1 = df_scored.merge(wise_q1[["source_id", "label"]], on="source_id", how="inner")
    df_wise_q1 = df_wise_q1.dropna(subset=["xgb_score", "label"])
    t_q1 = _optimal_threshold(df_wise_q1["xgb_score"].to_numpy(),
                               df_wise_q1["label"].to_numpy())
    print(f"  Q1  optimal threshold (Youden's J): {t_q1:.4f}")

    q1_quaia = pd.read_csv(Q1_DIR / "q1_quaia_hits.csv", low_memory=False)
    q1_quaia["source_id"]      = pd.to_numeric(q1_quaia["source_id"], errors="coerce").astype("Int64")
    q1_quaia["redshift_quaia"] = pd.to_numeric(q1_quaia["redshift_quaia"], errors="coerce")
    q1_quaia = q1_quaia.merge(df_scored[["source_id", "xgb_score"]], on="source_id", how="inner")
    q1_quaia = q1_quaia.dropna(subset=["redshift_quaia", "xgb_score"])
    print(f"  Q1  Quaia quasars with z + score: {len(q1_quaia)}")

    # ---- ERO ----
    ero_s = pd.read_csv(ERO_WISE_SCORES, low_memory=False)
    ero_s["source_id"] = pd.to_numeric(ero_s["source_id"], errors="coerce").astype("Int64")
    ero_s = ero_s.dropna(subset=["xgb_score", "label"])
    t_ero = _optimal_threshold(ero_s["xgb_score"].to_numpy(), ero_s["label"].to_numpy())
    print(f"  ERO optimal threshold (Youden's J): {t_ero:.4f}")

    ero_quaia = pd.read_csv(ERO_QUAIA_HITS, low_memory=False)
    ero_quaia["source_id"]      = pd.to_numeric(ero_quaia["source_id"], errors="coerce").astype("Int64")
    ero_quaia["redshift_quaia"] = pd.to_numeric(ero_quaia["redshift_quaia"], errors="coerce")
    ero_quaia = ero_quaia.merge(ero_s[["source_id", "xgb_score"]], on="source_id", how="inner")
    ero_quaia = ero_quaia.dropna(subset=["redshift_quaia", "xgb_score"])
    print(f"  ERO Quaia quasars with z + score: {len(ero_quaia)}")

    # Assign z-bins
    for df, name in [(q1_quaia, "q1"), (ero_quaia, "ero")]:
        df["z_bin"] = pd.cut(df["redshift_quaia"], bins=Z_BINS_EXT1,
                             labels=Z_LABELS_EXT1, right=False)

    # ---- Figure: side-by-side box plots per z-bin ----
    n_bins = len(Z_LABELS_EXT1)
    fig, ax = plt.subplots(figsize=(12, 6))

    w     = 0.3    # box width
    gap   = 0.08   # gap between Q1 and ERO box within a bin
    rng   = np.random.default_rng(42)

    x_ticks, x_tick_labels = [], []

    for i, bl in enumerate(Z_LABELS_EXT1):
        x_q1  = i - gap/2 - w/2
        x_ero = i + gap/2 + w/2

        s_q1_bin  = q1_quaia.loc[q1_quaia["z_bin"] == bl, "xgb_score"].dropna().to_numpy()
        s_ero_bin = ero_quaia.loc[ero_quaia["z_bin"] == bl, "xgb_score"].dropna().to_numpy()

        for scores, xpos, color, label in [
            (s_q1_bin,  x_q1,  C_Q1,  f"Q1"),
            (s_ero_bin, x_ero, C_ERO, f"ERO"),
        ]:
            if len(scores) == 0:
                continue
            bp = ax.boxplot(
                scores,
                positions=[xpos], widths=w,
                patch_artist=True, notch=False,
                manage_ticks=False,
                boxprops=dict(facecolor=color, alpha=0.55, linewidth=1.2),
                medianprops=dict(color="white", linewidth=2.5),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.5),
                flierprops=dict(marker="o", markersize=2, alpha=0.3,
                                markerfacecolor=color, markeredgewidth=0),
            )
            # Jittered individual points
            jitter = rng.uniform(-w * 0.35, w * 0.35, len(scores))
            ax.scatter(xpos + jitter, scores,
                       s=5 if color == C_Q1 else 12,
                       alpha=0.35 if color == C_Q1 else 0.55,
                       color=color, zorder=3, linewidths=0)

        n_q1_bin  = len(s_q1_bin)
        n_ero_bin = len(s_ero_bin)
        x_ticks.append(i)
        x_tick_labels.append(f"{bl}\nQ1 n={n_q1_bin}  ERO n={n_ero_bin}")

        # Print per-bin medians
        if len(s_q1_bin):
            print(f"  {bl}  Q1  median={np.median(s_q1_bin):.3f}  n={n_q1_bin}")
        if len(s_ero_bin):
            print(f"  {bl}  ERO median={np.median(s_ero_bin):.3f}  n={n_ero_bin}")

    # Threshold lines
    ax.axhline(t_q1,  color=C_Q1,  lw=1.2, linestyle="--", alpha=0.7,
               label=f"Q1 Youden threshold = {t_q1:.2f}")
    ax.axhline(t_ero, color=C_ERO, lw=1.2, linestyle="--", alpha=0.7,
               label=f"ERO Youden threshold = {t_ero:.2f}")

    # Dummy handles for the legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_Q1,  alpha=0.7, label=f"Q1  (n={len(q1_quaia)})"),
        Patch(facecolor=C_ERO, alpha=0.7, label=f"ERO (n={len(ero_quaia)})"),
        plt.Line2D([0], [0], color=C_Q1,  lw=1.5, linestyle="--",
                   label=f"Q1 threshold = {t_q1:.2f}"),
        plt.Line2D([0], [0], color=C_ERO, lw=1.5, linestyle="--",
                   label=f"ERO threshold = {t_ero:.2f}"),
    ], fontsize=9, framealpha=0.85, loc="lower left")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=9)
    ax.set_ylabel("XGB P(quasar)", fontsize=12)
    ax.set_xlabel("Photometric redshift bin (Quaia)", fontsize=12)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(-0.6, n_bins - 0.4)
    ax.set_title(
        "Classifier score distribution for Quaia quasars by redshift — ERO vs Q1\n"
        "(dashed lines = Youden-optimal decision threshold per survey)",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ext1_score_by_redshift.png", dpi=150)
    plt.close(fig)
    print("  saved ext1_score_by_redshift.png")


# ======================================================================
# EXTENSION 2: EUCLID MORPHOLOGY vs GAIA SCORE
# ======================================================================

def run_ext2(df_scored: pd.DataFrame) -> None:
    """
    Gaia XGB score vs Euclid point_like_prob — two independent measurements.

    The global correlation is near zero (r~0.03) because they measure different
    things: point_like_prob separates stars+quasars from galaxies (image shape),
    while XGB score separates quasars from stars+galaxies (astrometric anomalies).

    The thesis-relevant story: quasars occupy a unique corner (high on BOTH axes)
    that neither classifier alone can isolate. This is shown by splitting the
    population into three physically motivated groups:

      - Extended   : point_like_prob < 0.3  → galaxies
      - Point-like : point_like_prob > 0.7, not Quaia → mostly stars
      - Quaia QSOs : Quaia confirmed quasars

    Left  — Scatter: where each group sits in (XGB score, point_like_prob) space.
    Right — XGB score distributions for each group as overlaid histograms.
            Key result: the XGB score clearly separates quasars FROM stars even
            though both have high point_like_prob. Gaia adds information that
            Euclid morphology alone cannot provide.
    """
    print("\n--- Extension 2: Euclid morphology vs Gaia score ---")

    if "point_like_prob" not in df_scored.columns:
        print("  point_like_prob not in features — skipping ext2")
        return

    df = df_scored[["source_id", "xgb_score", "point_like_prob"]].dropna().copy()
    print(f"  Sources with XGB score + point_like_prob: {len(df):,}")

    # Quaia quasars
    quaia = pd.read_csv(Q1_DIR / "q1_quaia_hits.csv", low_memory=False)
    quaia["source_id"] = pd.to_numeric(quaia["source_id"], errors="coerce").astype("Int64")
    quaia_ids = set(quaia["source_id"].dropna())
    df["is_quaia"] = df["source_id"].isin(quaia_ids)

    # Three populations
    m_ext    = (df["point_like_prob"] < 0.3)  & ~df["is_quaia"]   # extended (galaxies)
    m_point  = (df["point_like_prob"] > 0.7)  & ~df["is_quaia"]   # point-like (stars)
    m_quasar = df["is_quaia"]                                       # confirmed quasars

    df_ext   = df[m_ext]
    df_point = df[m_point]
    df_qso   = df[m_quasar]

    corr = df["xgb_score"].corr(df["point_like_prob"])
    print(f"  Pearson r (global) = {corr:.4f}")
    print(f"  Extended   (plp<0.3, non-Quaia): {len(df_ext):,}")
    print(f"  Point-like (plp>0.7, non-Quaia): {len(df_point):,}")
    print(f"  Quaia quasars:                   {len(df_qso):,}")

    # Subsample large groups for scatter
    rng    = np.random.default_rng(42)
    N_SCAT = 15_000
    s_ext   = df_ext.sample(min(N_SCAT, len(df_ext)),   random_state=42)
    s_point = df_point.sample(min(N_SCAT, len(df_point)), random_state=42)

    C_EXT   = "#999999"   # grey for extended
    C_POINT = "#2196F3"   # blue for point-like (stars)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panel (a): Scatter of three populations ----
    ax = axes[0]
    ax.scatter(s_ext["xgb_score"],   s_ext["point_like_prob"],
               s=2, alpha=0.15, color=C_EXT,   rasterized=True,
               label=f"Extended / galaxies  (n={len(df_ext):,})")
    ax.scatter(s_point["xgb_score"], s_point["point_like_prob"],
               s=2, alpha=0.15, color=C_POINT, rasterized=True,
               label=f"Point-like / stars   (n={len(df_point):,})")
    ax.scatter(df_qso["xgb_score"],  df_qso["point_like_prob"],
               s=14, alpha=0.85, color=C_QSO, zorder=5,
               edgecolors="white", linewidths=0.3,
               label=f"Quaia quasars        (n={len(df_qso):,})")

    # Threshold lines
    ax.axhline(0.7, color=C_POINT, lw=0.8, linestyle=":", alpha=0.7)
    ax.axhline(0.3, color=C_EXT,   lw=0.8, linestyle=":", alpha=0.7)

    # Region annotations
    ax.text(0.02, 0.88, "Stars", fontsize=9, color=C_POINT,
            transform=ax.transAxes, fontweight="bold")
    ax.text(0.02, 0.04, "Galaxies", fontsize=9, color="#777777",
            transform=ax.transAxes, fontweight="bold")
    ax.text(0.72, 0.88, "Quasars", fontsize=9, color=C_QSO,
            transform=ax.transAxes, fontweight="bold")

    ax.set_xlabel("XGB P(quasar) — 15D Gaia features", fontsize=11)
    ax.set_ylabel("Euclid point_like_prob — VIS imaging", fontsize=11)
    ax.set_title("(a) Three populations in classifier space", fontsize=10)
    ax.legend(fontsize=8, markerscale=3, framealpha=0.85,
              edgecolor="none", loc="center right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.98, 0.02, f"Pearson r = {corr:.3f}", ha="right",
            transform=ax.transAxes, fontsize=8, color="gray")
    ax.grid(alpha=0.15)

    # ---- Panel (b): XGB score distributions per group ----
    ax = axes[1]
    bins_h = np.linspace(0, 1, 50)
    ax.hist(df_ext["xgb_score"],   bins=bins_h, density=True, alpha=0.55,
            color=C_EXT,   label=f"Extended / galaxies  (n={len(df_ext):,})",
            edgecolor="none")
    ax.hist(df_point["xgb_score"], bins=bins_h, density=True, alpha=0.55,
            color=C_POINT, label=f"Point-like / stars   (n={len(df_point):,})",
            edgecolor="none")
    ax.hist(df_qso["xgb_score"],   bins=bins_h, density=True, alpha=0.85,
            color=C_QSO,   label=f"Quaia quasars        (n={len(df_qso):,})",
            edgecolor="none")

    ax.set_xlabel("XGB P(quasar) — 15D Gaia features", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(b) XGB score distribution per morphology group\n"
                 "(Gaia discriminates quasars from stars even with same point_like_prob)",
                 fontsize=9)
    ax.legend(fontsize=8, framealpha=0.85, edgecolor="none")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.2)

    fig.suptitle(
        "Gaia 15D classifier vs Euclid VIS morphology — Q1\n"
        "Two independent measurements capturing different physical properties",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ext2_score_vs_morphology.png", dpi=150)
    plt.close(fig)
    print("  saved ext2_score_vs_morphology.png")


# ======================================================================
# EXTENSION 3: UMAP OVERLAY
# ======================================================================

def run_ext3(df_scored: pd.DataFrame, feature_cols: list[str],
             feat_min: np.ndarray, feat_iqr: np.ndarray) -> None:
    """
    Project Q1 sources into the 2D space defined by fitting UMAP on ERO data.

    UMAP (Uniform Manifold Approximation and Projection) finds a 2D layout
    that preserves the neighbourhood structure of the 15D feature space.
    Sources with similar Gaia measurements end up close together.

    We fit the UMAP on ERO data, then call .transform() on Q1 features to
    map them into the same 2D coordinate system — so direct comparison is valid.

    This will take ~3–8 minutes depending on hardware (fitting on 150k points).
    """
    try:
        import umap as umap_lib
    except ImportError:
        print("  umap-learn not installed — skipping ext3. Install with: pip install umap-learn")
        return

    print("\n--- Extension 3: UMAP overlay of Q1 on ERO feature space ---")

    # Load ERO embedding (has source_id, 15D features, x, y from previous run)
    print("  Loading ERO embedding...")
    ero = pd.read_csv(EMB_ERO, usecols=["source_id", "x", "y"] + feature_cols,
                      low_memory=False)
    # feat_visibility_periods_used exists in ERO CSV but not in 15D set, so
    # usecols above selects only the 15D features that exist in both
    ero = ero.dropna(subset=feature_cols)
    print(f"  ERO sources with full 15D features: {len(ero):,}")

    # Q1 scored sources
    df_q1 = df_scored.dropna(subset=feature_cols).copy()
    print(f"  Q1 sources with full 15D features:  {len(df_q1):,}")

    # Normalise features
    X_ero = normalize(ero[feature_cols].to_numpy(), feat_min, feat_iqr)
    X_q1  = normalize(df_q1[feature_cols].to_numpy(), feat_min, feat_iqr)

    # Fit UMAP on a subsample of ERO (fitting on 50k is faster and still captures structure)
    N_FIT  = min(50_000, len(X_ero))
    rng    = np.random.default_rng(42)
    fit_idx = rng.choice(len(X_ero), N_FIT, replace=False)
    X_fit   = X_ero[fit_idx]

    print(f"  Fitting UMAP on {N_FIT:,} ERO sources (15D → 2D)...")
    print("  (This typically takes 3–8 minutes on CPU.)")
    reducer = umap_lib.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        random_state=42, verbose=False,
    )
    reducer.fit(X_fit)

    print("  Projecting all ERO sources...")
    xy_ero = reducer.transform(X_ero)
    print("  Projecting all Q1 sources...")
    xy_q1  = reducer.transform(X_q1)

    # Save Q1 projected coords for future use
    q1_umap_df = pd.DataFrame({
        "source_id": df_q1["source_id"].values,
        "x": xy_q1[:, 0], "y": xy_q1[:, 1],
        "xgb_score": df_q1["xgb_score"].values,
    })
    q1_umap_df.to_csv(Q1_DIR / "q1_umap_coords.csv", index=False)
    print("  Saved q1_umap_coords.csv")

    # Load Quaia IDs for highlighting
    quaia = pd.read_csv(Q1_DIR / "q1_quaia_hits.csv", low_memory=False)
    quaia["source_id"] = pd.to_numeric(quaia["source_id"], errors="coerce").astype("Int64")
    quaia_ids = set(quaia["source_id"].dropna())
    is_quaia_q1 = df_q1["source_id"].isin(quaia_ids).values

    # ---- Plot: 2 panels ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel (a): ERO background, Q1 coloured by XGB score
    ax = axes[0]
    ax.scatter(xy_ero[:, 0], xy_ero[:, 1],
               s=0.3, alpha=0.04, color="#555555", rasterized=True, label="ERO sources")
    sc = ax.scatter(xy_q1[:, 0], xy_q1[:, 1],
                    s=1, c=df_q1["xgb_score"].values,
                    cmap="RdPu", alpha=0.6, vmin=0, vmax=1, rasterized=True)
    plt.colorbar(sc, ax=ax, label="XGB P(quasar)", shrink=0.85)
    ax.set_title("(a) Q1 coloured by Gaia XGB score", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel (b): ERO background, Q1 Quaia quasars highlighted
    ax = axes[1]
    ax.scatter(xy_ero[:, 0], xy_ero[:, 1],
               s=0.3, alpha=0.04, color="#555555", rasterized=True)
    ax.scatter(xy_q1[~is_quaia_q1, 0], xy_q1[~is_quaia_q1, 1],
               s=0.8, alpha=0.08, color="#aaaaaa", rasterized=True,
               label=f"Q1 non-Quaia (n={(~is_quaia_q1).sum():,})")
    ax.scatter(xy_q1[is_quaia_q1, 0], xy_q1[is_quaia_q1, 1],
               s=20, alpha=0.9, color=C_QSO, rasterized=True, zorder=5,
               edgecolors="white", linewidths=0.3,
               label=f"Q1 Quaia quasars (n={is_quaia_q1.sum():,})")
    ax.set_title("(b) Q1 Quaia quasars highlighted", fontsize=10)
    ax.legend(fontsize=8, markerscale=3, framealpha=0.85, edgecolor="none")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "15D Gaia UMAP — Q1 sources projected onto ERO feature space\n"
        f"(UMAP fit on {N_FIT:,} ERO sources, transformed to Q1)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ext3_01_umap_q1_overlay.png", dpi=150)
    plt.close(fig)
    print("  saved ext3_01_umap_q1_overlay.png")


# ======================================================================
# EXTENSION 6: MAGNITUDE-DEPENDENT PERFORMANCE
# ======================================================================

def run_ext6(df_scored: pd.DataFrame) -> None:
    """
    Does the classifier work equally well across G magnitudes?

    G magnitude is Gaia's brightness measurement. Brighter = lower number.
    Our dataset covers G ~ 17–21. At G ~ 20-21 (faint), Gaia measurements
    have larger errors, which may degrade the features we rely on.

    We use the WISE-labelled Q1 subset (ground-truth labels from WISE colours)
    and compute AUC/AP separately for each 1-magnitude-wide bin.
    """
    print("\n--- Extension 6: Magnitude-dependent performance ---")

    wise = load_wise_labels()
    df = df_scored.merge(wise[["source_id", "label"]], on="source_id", how="inner")
    df = df.dropna(subset=["xgb_score", "phot_g_mean_mag", "label"])
    print(f"  WISE-labelled sources with XGB score + magnitude: {len(df):,}")

    bin_edges  = MAG_BINS
    bin_labels = [f"G {bin_edges[i]:.0f}–{bin_edges[i+1]:.0f}"
                  for i in range(len(bin_edges) - 1)]
    df["mag_bin"] = pd.cut(df["phot_g_mean_mag"], bins=bin_edges, labels=bin_labels, right=False)

    results = []
    for label in bin_labels:
        sub = df[df["mag_bin"] == label].dropna(subset=["label", "xgb_score"])
        n_pos = int(sub["label"].sum())
        n_tot = len(sub)
        if n_pos < MIN_SOURCES_FOR_AUC or (n_tot - n_pos) < MIN_SOURCES_FOR_AUC:
            print(f"  Skip {label}: n_pos={n_pos}, n_neg={n_tot - n_pos}")
            results.append(dict(label=label, auc=np.nan, ap=np.nan, n_tot=n_tot, n_pos=n_pos))
            continue
        auc = roc_auc_score(sub["label"], sub["xgb_score"])
        ap  = average_precision_score(sub["label"], sub["xgb_score"])
        prev = n_pos / n_tot
        print(f"  {label}: n={n_tot}, n_qso={n_pos} ({prev*100:.1f}%), AUC={auc:.4f}, AP={ap:.4f}")
        results.append(dict(label=label, auc=auc, ap=ap, n_tot=n_tot, n_pos=n_pos))

    res = pd.DataFrame(results)
    res.to_csv(Q1_DIR / "ext6_performance_by_magnitude.csv", index=False)

    valid = res.dropna(subset=["auc"])
    x = np.arange(len(valid))

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Panel 1: AUC and AP
    ax = axes[0]
    ax.bar(x - 0.2, valid["auc"], width=0.35, color=C_Q1,   alpha=0.85,
           label="AUC (ROC)", edgecolor="white")
    ax.bar(x + 0.2, valid["ap"],  width=0.35, color=C_WISE,  alpha=0.85,
           label="AP (PR curve)", edgecolor="white")
    for i, row in enumerate(valid.itertuples()):
        if not np.isnan(row.auc):
            ax.text(i - 0.2, row.auc + 0.005, f"{row.auc:.3f}", ha="center",
                    va="bottom", fontsize=7.5)
        if not np.isnan(row.ap):
            ax.text(i + 0.2, row.ap + 0.005, f"{row.ap:.3f}", ha="center",
                    va="bottom", fontsize=7.5)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.5, color="gray", lw=0.7, linestyle=":", label="Random baseline")
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title("Classifier performance vs Gaia G magnitude (Q1, WISE labels)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # Panel 2: source counts
    ax = axes[1]
    ax.bar(x - 0.2, valid["n_tot"], width=0.35, color="#aaaaaa",
           alpha=0.85, label="Total WISE sources", edgecolor="white")
    ax.bar(x + 0.2, valid["n_pos"], width=0.35, color=C_Q1,
           alpha=0.85, label="WISE quasar locus", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(valid["label"], fontsize=10)
    ax.set_xlabel("G magnitude bin", fontsize=11)
    ax.set_ylabel("Source count", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ext6_01_performance_vs_magnitude.png", dpi=150)
    plt.close(fig)
    print("  saved ext6_01_performance_vs_magnitude.png")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Extended Q1 quasar analysis (script 85)")
    ap.add_argument("--ext1", action="store_true", help="Redshift stratification")
    ap.add_argument("--ext2", action="store_true", help="Euclid morphology vs Gaia score")
    ap.add_argument("--ext3", action="store_true", help="UMAP overlay (slow ~5 min)")
    ap.add_argument("--ext6", action="store_true", help="Magnitude-dependent performance")
    ap.add_argument("--all",  action="store_true", help="Run all extensions")
    args = ap.parse_args()

    if not any([args.ext1, args.ext2, args.ext3, args.ext6, args.all]):
        ap.print_help()
        return

    run_e1 = args.ext1 or args.all
    run_e2 = args.ext2 or args.all
    run_e3 = args.ext3 or args.all
    run_e6 = args.ext6 or args.all

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    Q1_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols(FEATURE_SET)
    feat_min, feat_iqr = load_scaler(feature_cols)

    # Score all Q1 sources once (shared by all extensions)
    df_scored = load_and_score_q1(feature_cols, feat_min, feat_iqr)

    if run_e1:
        run_ext1(df_scored)
    if run_e2:
        run_ext2(df_scored)
    if run_e3:
        run_ext3(df_scored, feature_cols, feat_min, feat_iqr)
    if run_e6:
        run_ext6(df_scored)

    print(f"\nPlots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
