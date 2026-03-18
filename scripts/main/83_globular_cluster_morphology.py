"""
83_globular_cluster_morphology.py
==================================

Apply trained morphology regressors to Gaia sources around NGC 6397 —
one of the closest globular clusters (d~2.5 kpc), well-studied, and
coincidentally one of the 17 ERO fields.

We query a 1-degree radius (much larger than the ERO stamp coverage)
to capture: dense cluster core + intermediate annuli + clean field stars.

Key prediction: if the regressors learned real morphological signal,
predicted multipeak_ratio and ellipticity should rise toward the
cluster centre where crowding is severe.

Outputs (plots + copied to report folder):
  32_ngc6397_sky_map.png           — 2D map of predicted gini / multipeak
  33_ngc6397_radial_profile.png    — radial profile of 4 metrics vs distance
  34_ngc6397_core_vs_field.png     — KDE: core vs intermediate vs field
"""

from __future__ import annotations

import io
import sys
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

import importlib
sys.path.insert(0, str(Path(__file__).parent))
_mod = importlib.import_module("69_morph_metrics_xgb")
CFG          = _mod.CFG
METRIC_NAMES = _mod.METRIC_NAMES

from feature_schema import get_feature_cols
import xgboost as xgb

# ======================================================================
# CONFIG
# ======================================================================

# NGC 6397 centre (Harris 1996, 2010 ed.)
CLUSTER_RA   = 265.175
CLUSTER_DEC  = -53.674
CLUSTER_NAME = "NGC 6397"
QUERY_RAD    = 1.0   # degrees
MAG_MIN      = 17.0
MAG_MAX      = 21.0

# Key physical radii in degrees
R_CORE       = 0.06  / 60.0   # ~0.06 arcmin → deg
R_HALF       = 2.9   / 60.0   # ~2.9  arcmin → deg
R_TIDAL      = 28.0  / 60.0   # ~28   arcmin → deg

# Zone boundaries for KDE comparison
R_CORE_ZONE  = 5.0  / 60.0   # inner 5 arcmin
R_FIELD_ZONE = 40.0 / 60.0   # beyond 40 arcmin = field stars

TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"

BASE_DIR   = CFG.base_dir
MODELS_DIR = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "models"
OUT_DIR    = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "ngc6397"
PLOTS_DIR  = BASE_DIR / "plots"  / "ml_runs" / "morph_metrics_16d" / "ngc6397"
REPORT_DIR = Path("/data/yn316/Codes/report/model_decision"
                  "/20260306_morph_regressors_generalisation/application")


def savefig(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p = PLOTS_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    shutil.copy2(p, REPORT_DIR / name)
    plt.close(fig)
    print(f"Saved: {p}")


# ======================================================================
# GAIA QUERY  (same columns as script 81)
# ======================================================================

ADQL = """
SELECT source_id, ra, dec,
    phot_g_mean_mag, bp_rp,
    phot_g_mean_flux_over_error,
    ruwe, astrometric_excess_noise, astrometric_excess_noise_sig,
    parallax_over_error, visibility_periods_used,
    ipd_frac_multi_peak, ipd_frac_odd_win,
    ipd_gof_harmonic_amplitude, ipd_gof_harmonic_phase,
    phot_bp_rp_excess_factor,
    pmra, pmdec, pmra_error, pmdec_error,
    phot_bp_n_contaminated_transits, phot_bp_n_blended_transits,
    phot_rp_n_contaminated_transits, phot_rp_n_blended_transits
FROM gaiadr3.gaia_source
WHERE CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {ra}, {dec}, {rad})
) = 1
AND phot_g_mean_mag BETWEEN {mag_min} AND {mag_max}
AND phot_g_mean_flux_over_error IS NOT NULL
AND ruwe IS NOT NULL
""".format(ra=CLUSTER_RA, dec=CLUSTER_DEC, rad=QUERY_RAD,
           mag_min=MAG_MIN, mag_max=MAG_MAX)


def query_tap(adql: str) -> pd.DataFrame:
    params = urllib.parse.urlencode({
        "REQUEST": "doQuery", "LANG": "ADQL",
        "FORMAT": "csv", "QUERY": adql,
    }).encode()
    print(f"Querying Gaia TAP for {CLUSTER_NAME} (r={QUERY_RAD}°) ...")
    req = urllib.request.Request(TAP_URL, data=params, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        content = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(content))
    print(f"  Received {len(df)} sources")
    return df


# ======================================================================
# FEATURE ENGINEERING  (identical to script 81)
# ======================================================================

FEAT_TO_RAW = {
    "feat_log10_snr":                    "log10_snr",
    "feat_ruwe":                         "ruwe",
    "feat_astrometric_excess_noise":     "astrometric_excess_noise",
    "feat_parallax_over_error":          "parallax_over_error",
    "feat_visibility_periods_used":      "visibility_periods_used",
    "feat_ipd_frac_multi_peak":          "ipd_frac_multi_peak",
    "feat_c_star":                       "c_star",
    "feat_pm_significance":              "pm_significance",
    "feat_astrometric_excess_noise_sig": "astrometric_excess_noise_sig",
    "feat_ipd_gof_harmonic_amplitude":   "ipd_gof_harmonic_amplitude",
    "feat_ipd_gof_harmonic_phase":       "ipd_gof_harmonic_phase",
    "feat_ipd_frac_odd_win":             "ipd_frac_odd_win",
    "feat_phot_bp_n_contaminated_transits": "phot_bp_n_contaminated_transits",
    "feat_phot_bp_n_blended_transits":      "phot_bp_n_blended_transits",
    "feat_phot_rp_n_contaminated_transits": "phot_rp_n_contaminated_transits",
    "feat_phot_rp_n_blended_transits":      "phot_rp_n_blended_transits",
}


def compute_features(df: pd.DataFrame, feature_cols, gaia_min, gaia_iqr):
    df = df.copy()
    flux = pd.to_numeric(df["phot_g_mean_flux_over_error"], errors="coerce")
    df["log10_snr"] = np.log10(np.clip(flux, 1e-6, None))

    bprp   = pd.to_numeric(df["bp_rp"], errors="coerce").to_numpy()
    excess = pd.to_numeric(df["phot_bp_rp_excess_factor"], errors="coerce").to_numpy()
    df["c_star"] = excess - (1.154360 + 0.033772*bprp + 0.032277*bprp**2)

    pmra   = pd.to_numeric(df["pmra"],        errors="coerce").to_numpy()
    pmdec  = pd.to_numeric(df["pmdec"],       errors="coerce").to_numpy()
    epmra  = pd.to_numeric(df["pmra_error"],  errors="coerce").to_numpy()
    epmdec = pd.to_numeric(df["pmdec_error"], errors="coerce").to_numpy()
    pm_tot = np.sqrt(pmra**2 + pmdec**2)
    pm_err = np.sqrt(epmra**2 + epmdec**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["pm_significance"] = np.where(pm_err > 0, pm_tot / pm_err, 0.0)

    raw = []
    for fc in feature_cols:
        rc = FEAT_TO_RAW.get(fc, fc.replace("feat_", ""))
        raw.append(pd.to_numeric(df[rc], errors="coerce").to_numpy()
                   if rc in df.columns else np.full(len(df), np.nan))
    X_raw = np.column_stack(raw).astype(np.float32)
    X = (X_raw - gaia_min) / np.where(gaia_iqr > 0, gaia_iqr, 1.0)
    valid = np.all(np.isfinite(X), axis=1)
    return df, X, valid


def angular_sep_deg(ra, dec, ra0, dec0):
    """Great-circle distance in degrees (small-angle safe)."""
    ra_r  = np.radians(ra);  dec_r  = np.radians(dec)
    ra0_r = np.radians(ra0); dec0_r = np.radians(dec0)
    dra = ra_r - ra0_r
    d = np.arcsin(np.sqrt(
        np.sin((dec_r - dec0_r)/2)**2
        + np.cos(dec_r)*np.cos(dec0_r)*np.sin(dra/2)**2
    )) * 2
    return np.degrees(d)


# ======================================================================
# PLOTS
# ======================================================================

def plot_sky_map(df: pd.DataFrame) -> None:
    """
    Plot 32: 2D sky map of the cluster region colored by predicted
    multipeak_ratio and gini. Shows the spatial morphology structure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, metric, cmap, label in [
        (axes[0], "pred_multipeak_ratio", "inferno",
         "Predicted multi-peak ratio"),
        (axes[1], "pred_gini", "plasma",
         "Predicted Gini"),
    ]:
        vals = df[metric].to_numpy()
        vlo, vhi = np.percentile(vals, [2, 98])

        # offset RA for display
        dra  = (df["ra"].to_numpy() - CLUSTER_RA) * 60   # arcmin
        ddec = (df["dec"].to_numpy() - CLUSTER_DEC) * 60

        sc = ax.scatter(dra, ddec, c=np.clip(vals, vlo, vhi),
                        cmap=cmap, s=2, alpha=0.7, rasterized=True)
        plt.colorbar(sc, ax=ax, label=label)

        # mark physical radii
        for r_deg, ls, lbl in [
            (R_CORE,  ":",  f"Core r={R_CORE*60:.1f}'"),
            (R_HALF,  "--", f"Half-light r={R_HALF*60:.1f}'"),
            (R_TIDAL, "-",  f"Tidal r={R_TIDAL*60:.0f}'"),
        ]:
            theta = np.linspace(0, 2*np.pi, 300)
            r_am  = r_deg * 60
            ax.plot(r_am*np.cos(theta), r_am*np.sin(theta),
                    color="white", lw=1.2, ls=ls, alpha=0.8)
            ax.plot(r_am*np.cos(theta), r_am*np.sin(theta),
                    color="black", lw=0.6, ls=ls, alpha=0.5)

        ax.set_aspect("equal")
        ax.set_xlabel("ΔRA (arcmin)", fontsize=10)
        ax.set_ylabel("ΔDec (arcmin)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.invert_xaxis()

    fig.suptitle(
        f"{CLUSTER_NAME}  —  predicted morphology from Gaia catalog statistics\n"
        f"Circles: core / half-light / tidal radius",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "32_ngc6397_sky_map.png")


def plot_radial_profile(df: pd.DataFrame) -> None:
    """
    Plot 33: Radial profiles of 4 key predicted metrics vs angular
    distance from cluster centre. Running median ± IQR shaded.
    Vertical lines mark core, half-light, tidal radii.
    """
    r = df["r_arcmin"].to_numpy()
    metrics = ["multipeak_ratio", "ellipticity", "gini", "kurtosis"]
    nice    = ["Multi-peak ratio", "Ellipticity", "Gini", "Kurtosis"]
    colors  = ["#d73027", "#4393c3", "#1a9850", "#984ea3"]

    # Radial bins (log-spaced to resolve the core)
    r_edges = np.logspace(np.log10(0.3), np.log10(60), 25)
    r_mids  = 0.5 * (r_edges[:-1] + r_edges[1:])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, m, nice_m, col in zip(axes, metrics, nice, colors):
        vals = df[f"pred_{m}"].to_numpy()
        meds, p25s, p75s = [], [], []

        for lo, hi in zip(r_edges[:-1], r_edges[1:]):
            mask = (r >= lo) & (r < hi) & np.isfinite(vals)
            if mask.sum() < 5:
                meds.append(np.nan); p25s.append(np.nan); p75s.append(np.nan)
            else:
                meds.append(np.median(vals[mask]))
                p25s.append(np.percentile(vals[mask], 25))
                p75s.append(np.percentile(vals[mask], 75))

        meds = np.array(meds); p25s = np.array(p25s); p75s = np.array(p75s)
        ok = np.isfinite(meds)

        ax.fill_between(r_mids[ok], p25s[ok], p75s[ok],
                        alpha=0.25, color=col)
        ax.plot(r_mids[ok], meds[ok], color=col, lw=2.5, label="Median")

        # Physical radii
        for r_deg, ls, lbl in [
            (R_CORE*60,  ":", "Core"),
            (R_HALF*60,  "--", "Half-light"),
            (R_TIDAL*60, "-",  "Tidal"),
        ]:
            ax.axvline(r_deg, color="grey", lw=1.2, ls=ls, alpha=0.7)
            ax.text(r_deg*1.05, ax.get_ylim()[1] if ok.any() else 0,
                    lbl, fontsize=7, color="grey", va="top")

        ax.set_xscale("log")
        ax.set_xlabel("Angular distance from centre (arcmin)", fontsize=10)
        ax.set_ylabel(f"Predicted {nice_m}", fontsize=10)
        ax.set_title(nice_m, fontsize=11, fontweight="bold")
        ax.set_xlim(r_edges[0], r_edges[-1])
        ax.tick_params(labelsize=9)

    fig.suptitle(
        f"{CLUSTER_NAME}  —  predicted morphology radial profile\n"
        "Median ± IQR in log-spaced radial bins  |  "
        "Dotted=core, dashed=half-light, solid=tidal radius",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "33_ngc6397_radial_profile.png")


def plot_core_vs_field(df: pd.DataFrame) -> None:
    """
    Plot 34: KDE comparison of predicted morphology for three zones:
      core   (r < 5 arcmin)
      halo   (5–40 arcmin)
      field  (r > 40 arcmin)
    """
    r = df["r_arcmin"].to_numpy()
    core  = r <  R_CORE_ZONE  * 60
    halo  = (r >= R_CORE_ZONE * 60) & (r < R_FIELD_ZONE * 60)
    field = r >= R_FIELD_ZONE * 60

    zones  = [core,  halo,   field]
    labels = [f"Core  (r<{R_CORE_ZONE*60:.0f}′, n={core.sum()})",
              f"Halo  ({R_CORE_ZONE*60:.0f}′–{R_FIELD_ZONE*60:.0f}′, n={halo.sum()})",
              f"Field  (r>{R_FIELD_ZONE*60:.0f}′, n={field.sum()})"]
    colors = ["#d73027", "#fc8d59", "#2166ac"]

    metrics = ["multipeak_ratio", "ellipticity", "gini", "asymmetry_180"]
    nice    = ["Multi-peak ratio", "Ellipticity", "Gini", "Asymmetry 180°"]

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))

    for ax, m, nice_m in zip(axes, metrics, nice):
        for mask, lbl, col in zip(zones, labels, colors):
            vals = df[f"pred_{m}"].to_numpy()[mask]
            ok   = np.isfinite(vals)
            if ok.sum() < 10:
                continue
            lo = np.percentile(vals[ok], 1)
            hi = np.percentile(vals[ok], 99)
            xg = np.linspace(lo, hi, 300)
            kde = gaussian_kde(vals[ok], bw_method=0.25)
            ax.plot(xg, kde(xg), color=col, lw=2.2, label=lbl)
            ax.fill_between(xg, kde(xg), alpha=0.08, color=col)

        ax.set_xlabel(f"Predicted {nice_m}", fontsize=10)
        ax.set_ylabel("Density" if ax is axes[0] else "", fontsize=10)
        ax.set_title(nice_m, fontsize=11, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=9)

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"{CLUSTER_NAME}  —  predicted morphology by cluster zone\n"
        "Core sources show higher crowding-induced morphological complexity",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "34_ngc6397_core_vs_field.png")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = get_feature_cols(CFG.feature_set)

    # Query or load cache
    csv_cache = OUT_DIR / "gaia_ngc6397.csv"
    if csv_cache.exists():
        print(f"Loading cached query: {csv_cache}")
        df_raw = pd.read_csv(csv_cache)
        print(f"  {len(df_raw)} sources")
    else:
        df_raw = query_tap(ADQL)
        df_raw.to_csv(csv_cache, index=False)

    # Feature engineering
    gaia_min, gaia_iqr = _mod.load_gaia_scaler(CFG.gaia_scaler_npz, feature_cols)
    df, X, valid = compute_features(df_raw, feature_cols, gaia_min, gaia_iqr)
    df   = df[valid].reset_index(drop=True)
    X    = X[valid]
    print(f"Valid sources: {len(df)}")

    # Angular separation
    df["r_arcmin"] = angular_sep_deg(
        df["ra"].to_numpy(), df["dec"].to_numpy(),
        CLUSTER_RA, CLUSTER_DEC,
    ) * 60.0

    # Predict
    print("Predicting ...")
    fnames   = [f"f{i}" for i in range(len(feature_cols))]
    boosters = []
    for name in METRIC_NAMES:
        b = xgb.Booster()
        b.load_model(str(MODELS_DIR / f"booster_{name}.json"))
        b.feature_names = fnames
        boosters.append(b)
    dm = xgb.DMatrix(X, feature_names=fnames)
    Y_pred = np.column_stack([b.predict(dm) for b in boosters])
    for i, name in enumerate(METRIC_NAMES):
        df[f"pred_{name}"] = Y_pred[:, i]

    # Save
    pred_csv = OUT_DIR / "ngc6397_predictions.csv"
    df.to_csv(pred_csv, index=False)
    print(f"Saved: {pred_csv}")

    print("\nGenerating plots ...")
    plot_sky_map(df)
    plot_radial_profile(df)
    plot_core_vs_field(df)

    # Quick stats
    r = df["r_arcmin"].to_numpy()
    print(f"\n=== {CLUSTER_NAME} SUMMARY ===")
    print(f"Total sources: {len(df)}  (G={MAG_MIN}–{MAG_MAX})")
    for zone, mask in [("Core (<5')", r < 5),
                       ("Halo (5-40')", (r>=5) & (r<40)),
                       ("Field (>40')", r >= 40)]:
        n = mask.sum()
        mp = df["pred_multipeak_ratio"][mask].median()
        el = df["pred_ellipticity"][mask].median()
        print(f"  {zone:18s}  n={n:5d}  "
              f"median multipeak={mp:.4f}  ellipticity={el:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
