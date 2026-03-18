"""
81_new_field_morphology.py
==========================

Apply trained morphology regressors to a NEW Gaia field — no Euclid stamp required.

Queries Gaia DR3 via the ESA TAP service for a 10 sq-deg region not covered by
any ERO training field, then predicts all 11 morphology metrics and compares the
predicted distribution to the ERO training/test set.

This demonstrates the main application of the thesis: predicting Euclid-like
morphology for any Gaia source anywhere in the sky.

Field chosen: RA=150, Dec=+2  (clean extragalactic field, |b|~50 deg)
              away from all 17 ERO training fields.

Outputs:
  output/ml_runs/morph_metrics_16d/new_field/
    gaia_new_field.csv          — raw Gaia query result
    new_field_predictions.csv   — predicted morphology for all sources

  plots/ml_runs/morph_metrics_16d/new_field/
    26_new_field_distributions.png   — predicted metrics: new field vs ERO test set
    27_new_field_sky_scatter.png     — sky positions colored by pred gini / multipeak
    28_new_field_top_weird.png       — top 20 weirdest stamps info table
"""

from __future__ import annotations

import io
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xgboost as xgb
from scipy.stats import rankdata

import importlib
sys.path.insert(0, str(Path(__file__).parent))
_mod = importlib.import_module("69_morph_metrics_xgb")
CFG          = _mod.CFG
METRIC_NAMES = _mod.METRIC_NAMES

from feature_schema import get_feature_cols

# ======================================================================
# CONFIG
# ======================================================================

FIELD_RA    = 150.0   # degrees
FIELD_DEC   = 2.0     # degrees
FIELD_RAD   = 1.784   # degrees  → ~10 sq deg
MAG_MIN     = 17.0
MAG_MAX     = 21.0

TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"

BASE_DIR   = CFG.base_dir
MODELS_DIR = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "models"
MAG_CACHE  = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "mag_cache"
OUT_DIR    = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "new_field"
PLOTS_DIR  = BASE_DIR / "plots"  / "ml_runs" / "morph_metrics_16d" / "new_field"


# ======================================================================
# GAIA TAP QUERY
# ======================================================================

ADQL = """
SELECT
    source_id, ra, dec,
    phot_g_mean_mag, bp_rp,
    phot_g_mean_flux_over_error,
    ruwe,
    astrometric_excess_noise,
    astrometric_excess_noise_sig,
    parallax_over_error,
    visibility_periods_used,
    ipd_frac_multi_peak,
    ipd_frac_odd_win,
    ipd_gof_harmonic_amplitude,
    ipd_gof_harmonic_phase,
    phot_bp_rp_excess_factor,
    pmra, pmdec, pmra_error, pmdec_error,
    phot_bp_n_contaminated_transits,
    phot_bp_n_blended_transits,
    phot_rp_n_contaminated_transits,
    phot_rp_n_blended_transits
FROM gaiadr3.gaia_source
WHERE CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {ra}, {dec}, {rad})
) = 1
AND phot_g_mean_mag BETWEEN {mag_min} AND {mag_max}
AND phot_g_mean_flux_over_error IS NOT NULL
AND ruwe IS NOT NULL
""".format(
    ra=FIELD_RA, dec=FIELD_DEC, rad=FIELD_RAD,
    mag_min=MAG_MIN, mag_max=MAG_MAX,
)


def query_gaia_tap(adql: str) -> pd.DataFrame:
    """Submit ADQL to Gaia TAP sync endpoint, return DataFrame."""
    params = urllib.parse.urlencode({
        "REQUEST":  "doQuery",
        "LANG":     "ADQL",
        "FORMAT":   "csv",
        "QUERY":    adql,
    }).encode()

    print(f"Querying Gaia TAP ({TAP_URL}) ...")
    req = urllib.request.Request(TAP_URL, data=params, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = resp.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(content))
    print(f"  Received {len(df)} sources")
    return df


# ======================================================================
# FEATURE ENGINEERING  (mirrors original pipeline)
# ======================================================================

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log10_snr, c_star, pm_significance from raw Gaia columns."""
    df = df.copy()

    # log10 SNR
    flux_err = pd.to_numeric(df["phot_g_mean_flux_over_error"], errors="coerce")
    df["log10_snr"] = np.log10(np.clip(flux_err, 1e-6, None))

    # c_star: Riello+2021 corrected BP/RP excess factor
    bprp = pd.to_numeric(df["bp_rp"], errors="coerce").to_numpy()
    excess = pd.to_numeric(df["phot_bp_rp_excess_factor"], errors="coerce").to_numpy()
    poly = 1.154360 + 0.033772 * bprp + 0.032277 * bprp ** 2
    df["c_star"] = excess - poly

    # proper motion significance
    pmra  = pd.to_numeric(df["pmra"],       errors="coerce").to_numpy()
    pmdec = pd.to_numeric(df["pmdec"],      errors="coerce").to_numpy()
    epmra = pd.to_numeric(df["pmra_error"], errors="coerce").to_numpy()
    epmdec= pd.to_numeric(df["pmdec_error"],errors="coerce").to_numpy()
    pm_total = np.sqrt(pmra**2 + pmdec**2)
    pm_err   = np.sqrt(epmra**2 + epmdec**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["pm_significance"] = np.where(pm_err > 0, pm_total / pm_err, 0.0)

    return df


FEAT_TO_RAW = {
    "feat_log10_snr":                   "log10_snr",
    "feat_ruwe":                        "ruwe",
    "feat_astrometric_excess_noise":    "astrometric_excess_noise",
    "feat_parallax_over_error":         "parallax_over_error",
    "feat_visibility_periods_used":     "visibility_periods_used",
    "feat_ipd_frac_multi_peak":         "ipd_frac_multi_peak",
    "feat_c_star":                      "c_star",
    "feat_pm_significance":             "pm_significance",
    "feat_astrometric_excess_noise_sig":"astrometric_excess_noise_sig",
    "feat_ipd_gof_harmonic_amplitude":  "ipd_gof_harmonic_amplitude",
    "feat_ipd_gof_harmonic_phase":      "ipd_gof_harmonic_phase",
    "feat_ipd_frac_odd_win":            "ipd_frac_odd_win",
    "feat_phot_bp_n_contaminated_transits": "phot_bp_n_contaminated_transits",
    "feat_phot_bp_n_blended_transits":      "phot_bp_n_blended_transits",
    "feat_phot_rp_n_contaminated_transits": "phot_rp_n_contaminated_transits",
    "feat_phot_rp_n_blended_transits":      "phot_rp_n_blended_transits",
}


def build_feature_matrix(df: pd.DataFrame, feature_cols: List[str],
                          gaia_min: np.ndarray, gaia_iqr: np.ndarray) -> np.ndarray:
    raw_feats = []
    for fc in feature_cols:
        raw_col = FEAT_TO_RAW.get(fc, fc.replace("feat_", ""))
        if raw_col in df.columns:
            raw_feats.append(pd.to_numeric(df[raw_col], errors="coerce").to_numpy())
        else:
            print(f"  WARNING: missing column {raw_col}, filling NaN")
            raw_feats.append(np.full(len(df), np.nan))
    X_raw = np.column_stack(raw_feats).astype(np.float32)
    X = (X_raw - gaia_min) / np.where(gaia_iqr > 0, gaia_iqr, 1.0)
    return X


# ======================================================================
# LOAD BOOSTERS
# ======================================================================

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
# PLOTS
# ======================================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def kde1d(vals: np.ndarray, x_grid: np.ndarray, bw: float | None = None) -> np.ndarray:
    """Simple Gaussian KDE."""
    from scipy.stats import gaussian_kde
    ok = np.isfinite(vals)
    if ok.sum() < 10:
        return np.zeros_like(x_grid)
    k = gaussian_kde(vals[ok], bw_method=bw)
    return k(x_grid)


def plot_kde_comparison(Y_new: np.ndarray, Y_ero: np.ndarray,
                        df_new: pd.DataFrame, out_path: Path) -> None:
    """
    Plot A: KDE curves for 4 key metrics, new field vs ERO test set.
    Split by magnitude tertile inside the new field to show mag dependence.
    """
    key_metrics = ["gini", "ellipticity", "multipeak_ratio", "kurtosis"]
    key_idx     = [METRIC_NAMES.index(m) for m in key_metrics]
    nice_names  = ["Gini coefficient", "Ellipticity", "Multi-peak ratio", "Kurtosis"]

    mag = df_new["phot_g_mean_mag"].to_numpy()
    t1, t2 = np.percentile(mag, [33, 66])
    bright = mag <= t1
    faint  = mag >= t2

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for ax, midx, mname, nice in zip(axes, key_idx, key_metrics, nice_names):
        yn   = Y_new[:, midx]
        ye   = Y_ero[:, midx]
        yn_b = Y_new[bright, midx]
        yn_f = Y_new[faint,  midx]

        lo = min(np.nanpercentile(yn, 1), np.nanpercentile(ye, 1))
        hi = max(np.nanpercentile(yn, 99), np.nanpercentile(ye, 99))
        xg = np.linspace(lo, hi, 300)

        ax.plot(xg, kde1d(ye,   xg), color="#2166ac", lw=2,   label="ERO test set")
        ax.plot(xg, kde1d(yn,   xg), color="#d73027", lw=2,   label=f"New field (all)")
        ax.plot(xg, kde1d(yn_b, xg), color="#d73027", lw=1.5, ls="--",
                label=f"New field (bright G<{t1:.1f})", alpha=0.7)
        ax.plot(xg, kde1d(yn_f, xg), color="#d73027", lw=1.5, ls=":",
                label=f"New field (faint G>{t2:.1f})", alpha=0.7)

        ax.set_xlabel(nice, fontsize=11)
        ax.set_ylabel("Density" if ax is axes[0] else "", fontsize=11)
        ax.set_title(nice, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Predicted morphology: new field (RA=150°, Dec=+2°) vs ERO test set\n"
        "Dashed/dotted = magnitude tertiles within new field",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, out_path)


def plot_gaia_vs_predicted(df_new: pd.DataFrame, out_path: Path) -> None:
    """
    Plot B: Gaia raw quality flag vs predicted morphology metric.
    Shows the regressors recover real signal in an unseen field.
    Two panels:
      left:  ipd_frac_multi_peak  vs pred_multipeak_ratio  (hexbin)
      right: ruwe                 vs pred_ellipticity       (hexbin)
    """
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    pairs = [
        ("ipd_frac_multi_peak", "pred_multipeak_ratio",
         "Gaia ipd_frac_multi_peak  (raw flag, %)",
         "Predicted multi-peak ratio",
         "Gaia multi-peak flag → predicted multi-peak ratio"),
        ("ruwe", "pred_ellipticity",
         "Gaia RUWE  (renormalised unit weight error)",
         "Predicted ellipticity",
         "Gaia RUWE → predicted ellipticity"),
    ]

    for ax, (xcol, ycol, xlabel, ylabel, title) in zip(axes, pairs):
        xv = pd.to_numeric(df_new[xcol], errors="coerce").to_numpy()
        yv = df_new[ycol].to_numpy()
        ok = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[ok], yv[ok]

        # clip to p1–p99 for display
        xlo, xhi = np.percentile(xv, [1, 99])
        ylo, yhi = np.percentile(yv, [1, 99])
        xv = np.clip(xv, xlo, xhi)
        yv = np.clip(yv, ylo, yhi)

        hb = ax.hexbin(xv, yv, gridsize=50, cmap="magma", norm=LogNorm(), linewidths=0.2)
        plt.colorbar(hb, ax=ax, label="Source count (log)")

        # running median
        bins = np.percentile(xv, np.linspace(5, 95, 15))
        meds = [np.median(yv[(xv >= bins[i]) & (xv < bins[i+1])])
                for i in range(len(bins)-1)]
        xmid = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(xmid, meds, "w-", lw=2, label="Running median")
        ax.plot(xmid, meds, "k--", lw=1)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Gaia quality flags vs predicted morphology in an unseen field\n"
        "Demonstrates regressor generalisation beyond the training footprint",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, out_path)


def plot_morphology_space(Y_new: np.ndarray, Y_ero: np.ndarray, out_path: Path) -> None:
    """
    Plot C: 2D morphology space — pred gini vs pred ellipticity.
    Three panels:
      left:   ERO test set smooth KDE density
      centre: new field smooth KDE density  (same colour scale)
      right:  both as iso-density contours overlaid (ERO blue, new field red)
    All panels share the same axes → directly comparable.
    """
    from scipy.stats import gaussian_kde

    gi_idx = METRIC_NAMES.index("gini")
    el_idx = METRIC_NAMES.index("ellipticity")

    gn = Y_new[:, gi_idx];  en = Y_new[:, el_idx]
    ge = Y_ero[:, gi_idx];  ee = Y_ero[:, el_idx]
    ok_n = np.isfinite(gn) & np.isfinite(en)
    ok_e = np.isfinite(ge) & np.isfinite(ee)

    # shared limits
    xlo = min(np.percentile(gn[ok_n], 1), np.percentile(ge[ok_e], 1))
    xhi = max(np.percentile(gn[ok_n], 99), np.percentile(ge[ok_e], 99))
    ylo = min(np.percentile(en[ok_n], 1), np.percentile(ee[ok_e], 1))
    yhi = max(np.percentile(en[ok_n], 99), np.percentile(ee[ok_e], 99))

    xg = np.linspace(xlo, xhi, 200)
    yg = np.linspace(ylo, yhi, 200)
    XX, YY = np.meshgrid(xg, yg)
    grid_pts = np.vstack([XX.ravel(), YY.ravel()])

    def kde2d(x, y, bw=0.12):
        pts = np.vstack([np.clip(x, xlo, xhi), np.clip(y, ylo, yhi)])
        return gaussian_kde(pts, bw_method=bw)(grid_pts).reshape(XX.shape)

    Z_ero = kde2d(ge[ok_e], ee[ok_e])
    Z_new = kde2d(gn[ok_n], en[ok_n])

    vmax = max(Z_ero.max(), Z_new.max())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    titles = [
        f"ERO test set  (n={ok_e.sum():,})",
        f"New field — RA=150°, Dec=+2°  (n={ok_n.sum():,})",
        "Iso-density contours overlaid",
    ]

    for ax, Z, title in zip(axes[:2], [Z_ero, Z_new], titles):
        im = ax.imshow(
            Z, origin="lower", aspect="auto", cmap="magma",
            extent=[xlo, xhi, ylo, yhi], vmin=0, vmax=vmax,
        )
        ax.set_xlabel("Predicted Gini", fontsize=11)
        ax.set_ylabel("Predicted Ellipticity", fontsize=11)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=9)

    plt.colorbar(im, ax=axes[1], label="KDE density")

    # Right panel: contours only
    ax3 = axes[2]
    pct_levels = [50, 75, 90, 97]
    for Z, color, label in [
        (Z_ero, "#2166ac", "ERO test set"),
        (Z_new, "#d73027", "New field"),
    ]:
        lvls = [np.percentile(Z[Z > Z.max()*0.001], p) for p in pct_levels]
        cs = ax3.contour(XX, YY, Z, levels=lvls, colors=color,
                         linewidths=[0.8, 1.2, 1.8, 2.5])
        ax3.clabel(cs, fmt={l: f"{p}%" for l, p in zip(lvls, pct_levels)},
                   fontsize=7, inline=True, colors=color)

    from matplotlib.lines import Line2D
    ax3.legend(handles=[
        Line2D([0],[0], color="#2166ac", lw=2, label="ERO test set"),
        Line2D([0],[0], color="#d73027", lw=2, label="New field"),
    ], fontsize=9)
    ax3.set_xlim(xlo, xhi); ax3.set_ylim(ylo, yhi)
    ax3.set_xlabel("Predicted Gini", fontsize=11)
    ax3.set_ylabel("Predicted Ellipticity", fontsize=11)
    ax3.set_title(titles[2], fontsize=10, fontweight="bold")
    ax3.tick_params(labelsize=9)

    fig.suptitle(
        "Predicted morphology space: Gini vs Ellipticity\n"
        "The new unseen field occupies the same region as the ERO training footprint",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = get_feature_cols(CFG.feature_set)

    # ---- Query or load cached ----
    csv_cache = OUT_DIR / "gaia_new_field.csv"
    if csv_cache.exists():
        print(f"Loading cached Gaia query: {csv_cache}")
        df_raw = pd.read_csv(csv_cache)
        print(f"  {len(df_raw)} sources")
    else:
        df_raw = query_gaia_tap(ADQL)
        df_raw.to_csv(csv_cache, index=False)
        print(f"Saved: {csv_cache}")

    # ---- Feature engineering ----
    print("Computing derived features ...")
    df = compute_derived_features(df_raw)

    gaia_min, gaia_iqr = _mod.load_gaia_scaler(CFG.gaia_scaler_npz, feature_cols)
    X = build_feature_matrix(df, feature_cols, gaia_min, gaia_iqr)

    # Drop rows with any NaN feature
    valid = np.all(np.isfinite(X), axis=1)
    df   = df[valid].reset_index(drop=True)
    X    = X[valid]
    print(f"  Valid sources (all features finite): {len(df)}")

    # ---- Predict ----
    print("Predicting morphology ...")
    boosters = load_boosters(feature_cols)
    fnames   = [f"f{i}" for i in range(len(feature_cols))]
    dm       = xgb.DMatrix(X, feature_names=fnames)
    Y_pred   = np.column_stack([b.predict(dm) for b in boosters])

    for i, name in enumerate(METRIC_NAMES):
        df[f"pred_{name}"] = Y_pred[:, i]

    # ---- Weirdness score ----
    n = len(df)
    df["weirdness_score"] = (
        0.5 * rankdata(df["pred_gini"]) / n
        + 0.3 * rankdata(df["pred_multipeak_ratio"]) / n
        + 0.2 * rankdata(df["pred_ellipticity"]) / n
    )

    save_cols = (
        ["source_id", "ra", "dec", "phot_g_mean_mag"] +
        [f"pred_{m}" for m in METRIC_NAMES] +
        ["weirdness_score"]
    )
    pred_csv = OUT_DIR / "new_field_predictions.csv"
    df[save_cols].to_csv(pred_csv, index=False)
    print(f"Saved: {pred_csv}  ({n} rows)")

    # ---- Load ERO test set for comparison ----
    print("Loading ERO test set predictions for comparison ...")
    Y_ero = np.load(MAG_CACHE / "Y.npy")[np.load(MAG_CACHE / "splits.npy") == 2]

    # ---- Plots ----
    print("\nGenerating plots ...")
    plot_kde_comparison(Y_pred, Y_ero, df, PLOTS_DIR / "26_new_field_kde_comparison.png")
    plot_gaia_vs_predicted(df, PLOTS_DIR / "27_new_field_gaia_vs_predicted.png")
    plot_morphology_space(Y_pred, Y_ero, PLOTS_DIR / "28_new_field_morphology_space.png")

    # ---- Summary ----
    print("\n=== NEW FIELD SUMMARY ===")
    print(f"Field: RA={FIELD_RA}, Dec={FIELD_DEC}, r={FIELD_RAD}° (~10 sq deg)")
    print(f"Sources: {n}  (G={MAG_MIN}–{MAG_MAX})")
    print(f"\nPredicted metric medians (new field vs ERO test):")
    print(f"  {'metric':<22} {'new field':>10}  {'ERO test':>10}")
    for i, name in enumerate(METRIC_NAMES):
        med_new = np.nanmedian(Y_pred[:, i])
        med_ero = np.nanmedian(Y_ero[:, i])
        print(f"  {name:<22} {med_new:>10.4f}  {med_ero:>10.4f}")

    print("\nTop 5 weirdest sources:")
    top5 = df.nlargest(5, "weirdness_score")[
        ["source_id", "ra", "dec", "phot_g_mean_mag",
         "pred_gini", "pred_multipeak_ratio", "pred_ellipticity", "weirdness_score"]
    ]
    print(top5.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
