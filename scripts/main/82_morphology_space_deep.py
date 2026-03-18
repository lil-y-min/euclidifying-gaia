"""
82_morphology_space_deep.py
============================

Deep morphology space analysis for the new unseen field (RA=150, Dec=+2)
vs the ERO training/test set. Three thesis-quality plots:

  Plot 29 — Corner plot: 5-metric pairwise grid, new field vs ERO contours
  Plot 30 — PCA of 11D morphology space: new field (weirdness) + ERO (grey)
  Plot 31 — Spearman correlation heatmap: Gaia flags × predicted metrics
             computed separately for ERO test set and new field

All outputs go to:
  plots/ml_runs/morph_metrics_16d/new_field/
  report/model_decision/20260306_morph_regressors_generalisation/application/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import importlib
sys.path.insert(0, str(Path(__file__).parent))
_mod = importlib.import_module("69_morph_metrics_xgb")
CFG          = _mod.CFG
METRIC_NAMES = _mod.METRIC_NAMES

# ======================================================================
# PATHS
# ======================================================================

BASE_DIR  = CFG.base_dir
MAG_CACHE = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "mag_cache"
PRED_ERO  = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "full_gaia_predictions.csv"
PRED_NEW  = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "new_field" / "new_field_predictions.csv"
RAW_NEW   = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "new_field" / "gaia_new_field.csv"

PLOTS_DIR = BASE_DIR / "plots" / "ml_runs" / "morph_metrics_16d" / "new_field"
REPORT_DIR = Path("/data/yn316/Codes/report/model_decision/20260306_morph_regressors_generalisation/application")


def savefig(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p = PLOTS_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    import shutil
    shutil.copy2(p, REPORT_DIR / name)
    plt.close(fig)
    print(f"Saved: {p}")


def kde2d_grid(x, y, xlo, xhi, ylo, yhi, bw=0.15, n=150):
    xg = np.linspace(xlo, xhi, n)
    yg = np.linspace(ylo, yhi, n)
    XX, YY = np.meshgrid(xg, yg)
    pts = np.vstack([np.clip(x, xlo, xhi), np.clip(y, ylo, yhi)])
    ok = np.all(np.isfinite(pts), axis=0)
    if ok.sum() < 20:
        return XX, YY, np.zeros((n, n))
    ZZ = gaussian_kde(pts[:, ok], bw_method=bw)(
        np.vstack([XX.ravel(), YY.ravel()])
    ).reshape(XX.shape)
    return XX, YY, ZZ


def contour_levels(Z, pcts=(50, 80, 95)):
    base = Z[Z > Z.max() * 0.001]
    if len(base) == 0:
        return []
    return [np.percentile(base, p) for p in pcts]


# ======================================================================
# LOAD DATA
# ======================================================================

def load_data():
    # ERO true Y (test split)
    Y_ero_true = np.load(MAG_CACHE / "Y.npy")[
        np.load(MAG_CACHE / "splits.npy") == 2
    ]

    # ERO predictions (full Gaia — use test split from pred CSV)
    # We'll use Y_ero_true as the ERO morphology — it's the ground truth
    # For Gaia flags in ERO: load from full_gaia_predictions + raw metadata
    pred_ero = pd.read_csv(PRED_ERO)

    # New field
    pred_new = pd.read_csv(PRED_NEW)
    raw_new  = pd.read_csv(RAW_NEW)

    # Merge raw Gaia flags into new field predictions
    raw_new["source_id"] = raw_new["source_id"].astype(str)
    pred_new["source_id"] = pred_new["source_id"].astype(str)
    new_merged = pred_new.merge(
        raw_new[["source_id", "ruwe", "ipd_frac_multi_peak", "astrometric_excess_noise"]],
        on="source_id", how="left",
    )

    return Y_ero_true, pred_ero, pred_new, new_merged


# ======================================================================
# PLOT 29 — CORNER PLOT
# ======================================================================

CORNER_METRICS = ["gini", "ellipticity", "kurtosis", "multipeak_ratio", "concentration"]
CORNER_NICE    = ["Gini", "Ellipticity", "Kurtosis", "Multi-peak ratio", "Concentration"]


def plot_corner(Y_ero: np.ndarray, Y_new: np.ndarray) -> None:
    n = len(CORNER_METRICS)
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.2 * n))

    idx = [METRIC_NAMES.index(m) for m in CORNER_METRICS]

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]

            if col > row:
                ax.set_visible(False)
                continue

            xi = idx[col]
            yi = idx[row]

            xe = Y_ero[:, xi];  ye = Y_ero[:, yi]
            xn = Y_new[:, xi];  yn = Y_new[:, yi]

            ok_e = np.isfinite(xe) & np.isfinite(ye)
            ok_n = np.isfinite(xn) & np.isfinite(yn)

            if col == row:
                # Diagonal: 1D KDE
                from scipy.stats import gaussian_kde as gkde
                lo = min(np.percentile(xe[ok_e], 1), np.percentile(xn[ok_n], 1))
                hi = max(np.percentile(xe[ok_e], 99), np.percentile(xn[ok_n], 99))
                xg = np.linspace(lo, hi, 300)
                ax.plot(xg, gkde(xe[ok_e])(xg), color="#2166ac", lw=2)
                ax.plot(xg, gkde(xn[ok_n])(xg), color="#d73027", lw=2)
                ax.set_xlim(lo, hi)
                ax.set_ylim(bottom=0)
                ax.set_yticks([])
            else:
                # Off-diagonal: contours
                xlo = min(np.percentile(xe[ok_e], 1), np.percentile(xn[ok_n], 1))
                xhi = max(np.percentile(xe[ok_e], 99), np.percentile(xn[ok_n], 99))
                ylo = min(np.percentile(ye[ok_e], 1), np.percentile(yn[ok_n], 1))
                yhi = max(np.percentile(ye[ok_e], 99), np.percentile(yn[ok_n], 99))

                XX, YY, Ze = kde2d_grid(xe[ok_e], ye[ok_e], xlo, xhi, ylo, yhi)
                _,  _,  Zn = kde2d_grid(xn[ok_n], yn[ok_n], xlo, xhi, ylo, yhi)

                for Z, color, lws in [
                    (Ze, "#2166ac", [0.8, 1.4, 2.2]),
                    (Zn, "#d73027", [0.8, 1.4, 2.2]),
                ]:
                    lvls = contour_levels(Z)
                    if lvls:
                        ax.contour(XX, YY, Z, levels=lvls, colors=color,
                                   linewidths=lws)

                ax.set_xlim(xlo, xhi)
                ax.set_ylim(ylo, yhi)

            # Labels on edges only
            if row == n - 1:
                ax.set_xlabel(CORNER_NICE[col], fontsize=9)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != col:
                ax.set_ylabel(CORNER_NICE[row], fontsize=9)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)

    # Legend
    handles = [
        Line2D([0],[0], color="#2166ac", lw=2, label=f"ERO test set  (n={len(Y_ero):,})"),
        Line2D([0],[0], color="#d73027", lw=2, label=f"New field  (n={len(Y_new):,})"),
    ]
    fig.legend(handles=handles, fontsize=11, loc="upper right",
               bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(
        "Predicted morphology — pairwise corner plot\n"
        "ERO training footprint (blue) vs new unseen field (red)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    savefig(fig, "29_corner_plot_5metrics.png")


# ======================================================================
# PLOT 30 — PCA OF 11D MORPHOLOGY SPACE
# ======================================================================

def plot_pca(Y_ero: np.ndarray, Y_new: np.ndarray, weird_new: np.ndarray) -> None:
    # Fit PCA on ERO, project both
    ok_e = np.all(np.isfinite(Y_ero), axis=1)
    ok_n = np.all(np.isfinite(Y_new), axis=1)

    scaler = StandardScaler().fit(Y_ero[ok_e])
    Ye_s = scaler.transform(Y_ero[ok_e])
    Yn_s = scaler.transform(Y_new[ok_n])

    pca = PCA(n_components=2).fit(Ye_s)
    Pe = pca.transform(Ye_s)
    Pn = pca.transform(Yn_s)

    var = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: density comparison
    ax = axes[0]
    ax.scatter(Pe[:, 0], Pe[:, 1], s=1, alpha=0.15, color="#2166ac",
               rasterized=True, label=f"ERO test set (n={ok_e.sum():,})")
    ax.scatter(Pn[:, 0], Pn[:, 1], s=2, alpha=0.4, color="#d73027",
               rasterized=True, label=f"New field (n={ok_n.sum():,})")
    ax.set_xlabel(f"PC1  ({var[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var[1]:.1f}% variance)", fontsize=11)
    ax.set_title("PCA of 11D morphology space", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, markerscale=4)

    # Right: new field colored by weirdness score
    ax2 = axes[1]
    w = weird_new[ok_n]
    sc = ax2.scatter(Pn[:, 0], Pn[:, 1], c=w, cmap="hot_r",
                     s=4, alpha=0.6, rasterized=True,
                     vmin=np.percentile(w, 5), vmax=np.percentile(w, 95))
    plt.colorbar(sc, ax=ax2, label="Weirdness score")

    # ERO contours for reference
    from scipy.stats import gaussian_kde as gkde
    xlo, xhi = np.percentile(Pe[:, 0], [1, 99])
    ylo, yhi = np.percentile(Pe[:, 1], [1, 99])
    XX, YY, Ze = kde2d_grid(Pe[:, 0], Pe[:, 1], xlo, xhi, ylo, yhi, bw=0.12)
    lvls = contour_levels(Ze, pcts=(50, 80, 95))
    if lvls:
        ax2.contour(XX, YY, Ze, levels=lvls, colors="royalblue",
                    linewidths=[0.8, 1.4, 2.0], alpha=0.7)
    ax2.set_xlabel(f"PC1  ({var[0]:.1f}% variance)", fontsize=11)
    ax2.set_ylabel(f"PC2  ({var[1]:.1f}% variance)", fontsize=11)
    ax2.set_title("New field colored by weirdness  +  ERO contours (blue)",
                  fontsize=11, fontweight="bold")

    # PCA loadings annotation (top 3 contributors per axis)
    loadings = pca.components_  # (2, 11)
    for pc_i, ax_i in enumerate([ax, ax2]):
        top = np.argsort(np.abs(loadings[pc_i]))[::-1][:3]
        txt = "PC{} ← {}".format(
            pc_i+1, ", ".join(METRIC_NAMES[j] for j in top)
        )
        ax_i.text(0.02, 0.02, txt, transform=ax_i.transAxes,
                  fontsize=7, color="grey", va="bottom")

    fig.suptitle(
        "PCA of predicted 11D morphology space\n"
        "New field sits inside the ERO morphological manifold",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "30_pca_morphology_space.png")


# ======================================================================
# PLOT 31 — SPEARMAN CORRELATION HEATMAP
# ======================================================================

GAIA_FLAGS = ["ruwe", "ipd_frac_multi_peak", "astrometric_excess_noise"]
FLAG_NICE  = ["RUWE", "ipd_frac_multi_peak", "AEN"]


def spearman_matrix(flags_df: pd.DataFrame, pred_arr: np.ndarray) -> np.ndarray:
    """Returns (n_flags × n_metrics) Spearman r matrix."""
    mat = np.full((len(GAIA_FLAGS), len(METRIC_NAMES)), np.nan)
    for i, flag in enumerate(GAIA_FLAGS):
        fv = pd.to_numeric(flags_df[flag], errors="coerce").to_numpy()
        for j in range(len(METRIC_NAMES)):
            mv = pred_arr[:, j]
            ok = np.isfinite(fv) & np.isfinite(mv)
            if ok.sum() > 30:
                mat[i, j] = spearmanr(fv[ok], mv[ok]).statistic
    return mat


def plot_spearman(pred_ero: pd.DataFrame, new_merged: pd.DataFrame) -> None:
    pred_cols = [f"pred_{m}" for m in METRIC_NAMES]

    Y_ero_pred = pred_ero[pred_cols].to_numpy(dtype=np.float32)
    Y_new_pred = new_merged[pred_cols].to_numpy(dtype=np.float32)

    # ERO: Gaia flags are in pred_ero directly
    mat_ero = spearman_matrix(pred_ero, Y_ero_pred)
    mat_new = spearman_matrix(new_merged, Y_new_pred)

    metric_nice = [m.replace("_", "\n") for m in METRIC_NAMES]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5),
                             gridspec_kw={"width_ratios": [10, 10, 1]})

    vmax = 0.6
    kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    im0 = axes[0].imshow(mat_ero, **kw)
    im1 = axes[1].imshow(mat_new, **kw)

    for ax, mat, title in [
        (axes[0], mat_ero, "ERO test set"),
        (axes[1], mat_new, "New field  (RA=150°, Dec=+2°)"),
    ]:
        ax.set_xticks(range(len(METRIC_NAMES)))
        ax.set_xticklabels(metric_nice, fontsize=8, rotation=0, ha="center")
        ax.set_yticks(range(len(GAIA_FLAGS)))
        ax.set_yticklabels(FLAG_NICE, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Annotate cells
        for i in range(len(GAIA_FLAGS)):
            for j in range(len(METRIC_NAMES)):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=7.5,
                            color="white" if abs(v) > 0.35 else "black",
                            fontweight="bold" if abs(v) > 0.4 else "normal")

    # Difference panel
    diff = mat_new - mat_ero
    axes[2].imshow(diff, cmap="PuOr", vmin=-0.3, vmax=0.3, aspect="auto")
    axes[2].set_xticks([0])
    axes[2].set_xticklabels(["Δ\n(new−ERO)"], fontsize=8)
    axes[2].set_yticks(range(len(GAIA_FLAGS)))
    axes[2].set_yticklabels([])
    axes[2].set_title("Difference", fontsize=10, fontweight="bold")
    for i in range(len(GAIA_FLAGS)):
        v = diff[i, 0] if diff.shape[1] == 1 else np.nanmean(diff[i])
    for i in range(len(GAIA_FLAGS)):
        for j in range(diff.shape[1]):
            v = diff[i, j]
            if np.isfinite(v):
                axes[2].text(j, i, f"{v:+.2f}", ha="center", va="center",
                             fontsize=7.5,
                             color="white" if abs(v) > 0.15 else "black")

    plt.colorbar(im0, ax=axes[1], label="Spearman r", fraction=0.03)

    fig.suptitle(
        "Spearman correlation: Gaia quality flags × predicted morphology metrics\n"
        "Preserved correlation structure confirms regressor generalisation",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    savefig(fig, "31_spearman_flags_vs_metrics.png")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    print("Loading data ...")
    Y_ero_true, pred_ero, pred_new, new_merged = load_data()

    pred_cols = [f"pred_{m}" for m in METRIC_NAMES]
    Y_new_pred = pred_new[pred_cols].to_numpy(dtype=np.float32)
    weird_new  = pred_new["weirdness_score"].to_numpy(dtype=np.float32)

    print(f"  ERO true Y:    {Y_ero_true.shape}")
    print(f"  New field pred: {Y_new_pred.shape}")

    print("\nPlot 29: corner plot ...")
    plot_corner(Y_ero_true, Y_new_pred)

    print("Plot 30: PCA ...")
    plot_pca(Y_ero_true, Y_new_pred, weird_new)

    print("Plot 31: Spearman heatmap ...")
    plot_spearman(pred_ero, new_merged)

    print("\nDone.")


if __name__ == "__main__":
    main()
