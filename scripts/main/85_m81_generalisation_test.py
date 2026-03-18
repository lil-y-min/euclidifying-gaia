#!/usr/bin/env python3
"""
85_m81_generalisation_test.py
==============================
Out-of-sample generalisation test for the 15D galaxy disk membership classifier.

Target: M81 (NGC 3031) — SA(s)ab, 3.6 Mpc, NOT in training set.
  - Different morphology from training galaxies (Sa vs SABc/IB)
  - Similar distance range (3.6 Mpc, training: 0.5–9.4 Mpc)
  - Extremely well-studied → independent fact-check via NED HII regions

Steps:
  1. --pull_wsdb   Query Gaia DR3 around M81 via WSDB, compute 15D features, save CSV
  2. (default)     Load CSV, score with 15D model, generate diagnostic plots

Fact-checks (all independent of the D25 geometric label):
  A. Radial score profile — should peak at nucleus, decay to D25 boundary
  B. Sky scatter plot     — high-score sources should trace the optical disk
  C. PM-space plot        — predicted members should cluster near (0,0) proper motion
  D. NED HII region overlay — high-score sources should coincide with HII regions

Outputs: report/model_decision/20260306_galaxy_disk_membership/
  m81_20_sky_scatter.png
  m81_21_radial_profile.png
  m81_22_pm_space.png
  m81_23_ned_overlay.png
  m81_gaia_sources.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse
import xgboost as xgb

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts" / "main"))
from feature_schema import get_feature_cols, scaler_stem

FEATURE_SET  = "15D"
MODEL_PATH   = BASE / "output" / "ml_runs" / "galaxy_env_clf" / "model_inside_galaxy_15d.json"
PLOT_DIR     = BASE / "report" / "model_decision" / "20260306_galaxy_disk_membership"
CSV_PATH     = PLOT_DIR / "m81_gaia_sources.csv"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# M81 (NGC 3031) — HyperLEDA / NED
M81 = dict(
    name    = "M81 (NGC 3031)",
    ra      = 148.8882,
    dec     = 69.0653,
    a       = 13.45,    # semi-major axis, arcmin
    b       = 7.04,     # semi-minor axis, arcmin
    pa      = 157.0,    # position angle N→E, degrees
    r_eff   = 3.5,      # effective radius, arcmin
    dist_mpc= 3.6,
    morph   = "SA(s)ab",
    query_r = 0.55,     # cone radius for WSDB query, degrees (≈2×a)
)

# ── thesis style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})

SCORE_CMAP = "RdPu"
HIGH_COL   = "#7b2d8b"


# ── feature engineering (mirrors script 05) ────────────────────────────────────

def corrected_flux_excess_c_star(bp_rp, phot_bp_rp_excess_factor):
    x = np.asarray(bp_rp, dtype=float)
    c = np.asarray(phot_bp_rp_excess_factor, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x) & np.isfinite(c)
    xo, co = x[ok], c[ok]
    f = np.empty_like(xo)
    m1, m2, m3 = xo < 0.5, (xo >= 0.5) & (xo < 4.0), xo >= 4.0
    f[m1] = 1.154360 + 0.033772 * xo[m1] + 0.032277 * xo[m1] ** 2
    f[m2] = 1.162004 + 0.011464 * xo[m2] + 0.049255 * xo[m2] ** 2 - 0.005879 * xo[m2] ** 3
    f[m3] = 1.057572 + 0.140537 * xo[m3]
    out[ok] = co - f
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 15D feat_ columns from raw Gaia DR3 columns."""
    df = df.copy()
    df["feat_log10_snr"] = np.log10(
        np.clip(df["phot_g_mean_flux_over_error"].to_numpy(dtype=float), 1e-6, None))
    df["feat_ruwe"] = df["ruwe"]
    df["feat_astrometric_excess_noise"] = df["astrometric_excess_noise"]
    df["feat_parallax_over_error"] = (
        df["parallax"].to_numpy(dtype=float) /
        np.where(df["parallax_error"].to_numpy(dtype=float) > 0,
                 df["parallax_error"].to_numpy(dtype=float), np.nan))
    df["feat_ipd_frac_multi_peak"] = df["ipd_frac_multi_peak"]
    df["feat_c_star"] = corrected_flux_excess_c_star(
        df["bp_rp"].to_numpy(dtype=float),
        df["phot_bp_rp_excess_factor"].to_numpy(dtype=float))
    pmra  = df["pmra"].to_numpy(dtype=float)
    pmdec = df["pmdec"].to_numpy(dtype=float)
    pmra_e  = df["pmra_error"].to_numpy(dtype=float)
    pmdec_e = df["pmdec_error"].to_numpy(dtype=float)
    pm_tot     = np.hypot(pmra, pmdec)
    pm_tot_err = np.hypot(pmra_e, pmdec_e)
    sig = np.full(len(df), np.nan)
    ok  = np.isfinite(pm_tot) & np.isfinite(pm_tot_err) & (pm_tot_err > 0)
    sig[ok] = pm_tot[ok] / pm_tot_err[ok]
    df["feat_pm_significance"]                 = sig
    df["feat_astrometric_excess_noise_sig"]    = df["astrometric_excess_noise_sig"]
    df["feat_ipd_gof_harmonic_amplitude"]      = df["ipd_gof_harmonic_amplitude"]
    df["feat_ipd_gof_harmonic_phase"]          = df["ipd_gof_harmonic_phase"]
    df["feat_ipd_frac_odd_win"]                = df["ipd_frac_odd_win"]
    df["feat_phot_bp_n_contaminated_transits"] = df["phot_bp_n_contaminated_transits"]
    df["feat_phot_bp_n_blended_transits"]      = df["phot_bp_n_blended_transits"]
    df["feat_phot_rp_n_contaminated_transits"] = df["phot_rp_n_contaminated_transits"]
    df["feat_phot_rp_n_blended_transits"]      = df["phot_rp_n_blended_transits"]
    return df


# ── WSDB pull ──────────────────────────────────────────────────────────────────

WSDB_QUERY = """
SELECT
    g.source_id, g.ra, g.dec, g.phot_g_mean_mag,
    g.phot_g_mean_flux_over_error,
    g.ruwe,
    g.astrometric_excess_noise, g.astrometric_excess_noise_sig,
    g.parallax, g.parallax_error,
    g.pmra, g.pmdec, g.pmra_error, g.pmdec_error,
    g.bp_rp, g.phot_bp_rp_excess_factor,
    g.ipd_frac_multi_peak, g.ipd_gof_harmonic_amplitude,
    g.ipd_gof_harmonic_phase, g.ipd_frac_odd_win,
    g.phot_bp_n_contaminated_transits, g.phot_bp_n_blended_transits,
    g.phot_rp_n_contaminated_transits, g.phot_rp_n_blended_transits
FROM gaia_dr3.gaia_source g
WHERE q3c_radial_query(g.ra, g.dec, {ra}, {dec}, {r})
  AND g.phot_g_mean_mag < 21.0
""".strip()


def pull_wsdb():
    import psycopg2
    pw = os.environ.get("WSDB_PASS", "")
    conn = psycopg2.connect(
        host="wsdb.ast.cam.ac.uk", dbname="wsdb",
        user="yasmine_nourlil_2026", password=pw,
        connect_timeout=30,
    )
    q = WSDB_QUERY.format(ra=M81["ra"], dec=M81["dec"], r=M81["query_r"])
    print(f"  Querying WSDB (cone r={M81['query_r']}°)...")
    df = pd.read_sql(q, conn)
    conn.close()
    print(f"  {len(df):,} sources returned")
    df = compute_features(df)
    df.to_csv(CSV_PATH, index=False)
    print(f"  Saved → {CSV_PATH}")
    return df


# ── helpers ────────────────────────────────────────────────────────────────────

def elliptical_radius(ra_s, dec_s):
    pa  = np.radians(M81["pa"])
    da  = (ra_s - M81["ra"]) * np.cos(np.radians(M81["dec"])) * 60.0
    dd  = (dec_s - M81["dec"]) * 60.0
    x   = da * np.sin(pa) + dd * np.cos(pa)
    y   = da * np.cos(pa) - dd * np.sin(pa)
    return np.sqrt((x / M81["a"]) ** 2 + (y / M81["b"]) ** 2)


def sky_offsets(ra_s, dec_s):
    da = (ra_s - M81["ra"]) * np.cos(np.radians(M81["dec"])) * 60.0
    dd = (dec_s - M81["dec"]) * 60.0
    return da, dd


def draw_d25(ax, scale=1.0, **kw):
    angle = -M81["pa"]
    ax.add_patch(Ellipse(
        (0, 0), width=2 * M81["b"] * scale, height=2 * M81["a"] * scale,
        angle=angle, **kw))


def load_scaler():
    feature_cols = get_feature_cols(FEATURE_SET)
    stem  = scaler_stem(FEATURE_SET)
    npz   = BASE / "output" / "scalers" / f"{stem}.npz"
    d     = np.load(npz, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    idx   = np.array([names.index(c) for c in feature_cols], dtype=int)
    return (feature_cols,
            d["y_min"].astype(np.float32)[idx],
            d["y_iqr"].astype(np.float32)[idx])


def score_df(df):
    feature_cols, feat_min, feat_iqr = load_scaler()
    df = df.dropna(subset=feature_cols + ["ra", "dec"]).copy()
    X  = df[feature_cols].to_numpy(dtype=np.float32)
    X  = (X - feat_min) / np.where(feat_iqr > 0, feat_iqr, 1.0)
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    df["score"]      = booster.predict(dmat).astype(np.float32)
    df["ell_radius"] = elliptical_radius(df["ra"].to_numpy(), df["dec"].to_numpy())
    df["label_geom"] = (df["ell_radius"] <= 1.0).astype(int)
    return df


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_sky(df):
    da, dd = sky_offsets(df["ra"].to_numpy(), df["dec"].to_numpy())
    scores = df["score"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    sc = ax.scatter(da, dd, c=scores, cmap=SCORE_CMAP, vmin=0, vmax=1,
                    s=3, alpha=0.6, linewidths=0, rasterized=True)
    draw_d25(ax, scale=1.0, fill=False, edgecolor="#e63946", lw=1.8,
             linestyle="--", zorder=5, label="D25")
    draw_d25(ax, scale=0.5, fill=False, edgecolor="#e63946", lw=0.8,
             linestyle=":", zorder=5, alpha=0.6)
    ax.add_patch(plt.Circle((0, 0), M81["r_eff"], fill=False,
                              edgecolor="#457b9d", lw=1.2, ls="-.", alpha=0.7,
                              label=f"r_eff = {M81['r_eff']}'"))
    ax.scatter([0], [0], marker="+", s=100, color="black", lw=1.5, zorder=6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("XGB P(inside disk)", fontsize=11)
    pad = M81["a"] * 1.4
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Δα·cos δ  (arcmin)"); ax.set_ylabel("Δδ  (arcmin)")
    n_in = int(df["label_geom"].sum())
    thr  = 0.5
    n_pred = int((scores >= thr).sum())
    ax.set_title(
        f"{M81['name']}  ({M81['morph']}, {M81['dist_mpc']} Mpc)"
        f"\nn_inside (geom)={n_in:,}   n_pred (thr=0.5)={n_pred:,}   "
        f"[OUT-OF-SAMPLE TEST]", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    out = PLOT_DIR / "m81_20_sky_scatter.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_radial(df):
    r     = df["ell_radius"].to_numpy()
    score = df["score"].to_numpy()

    bins  = np.linspace(0, 2.5, 26)
    bc    = (bins[:-1] + bins[1:]) / 2.0
    med, q25, q75 = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() < 5:
            med.append(np.nan); q25.append(np.nan); q75.append(np.nan)
        else:
            med.append(np.median(score[m]))
            q25.append(np.percentile(score[m], 25))
            q75.append(np.percentile(score[m], 75))

    med, q25, q75 = map(np.array, (med, q25, q75))

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.fill_between(bc, q25, q75, alpha=0.25, color=HIGH_COL)
    ax.plot(bc, med, "-o", color=HIGH_COL, ms=5, lw=2)
    ax.axvline(1.0, color="#e63946", lw=1.5, ls="--", label="D25 boundary (r=1)")
    ax.axvline(M81["r_eff"] / M81["a"], color="#457b9d", lw=1.2, ls="-.",
               alpha=0.8, label=f"r_eff / a = {M81['r_eff']/M81['a']:.2f}")
    ax.set_xlabel("Elliptical radius  r / a  (1 = D25 boundary)")
    ax.set_ylabel("XGB P(inside disk)  — median ± IQR")
    ax.set_title(f"Radial score profile — {M81['name']}  [out-of-sample]", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 2.5); ax.set_ylim(-0.02, 1.02)
    out = PLOT_DIR / "m81_21_radial_profile.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_pm_space(df):
    scores = df["score"].to_numpy()
    pmra   = df["pmra"].to_numpy(dtype=float)
    pmdec  = df["pmdec"].to_numpy(dtype=float)
    ok     = np.isfinite(pmra) & np.isfinite(pmdec)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sc = ax.scatter(pmra[ok], pmdec[ok], c=scores[ok], cmap=SCORE_CMAP,
                    vmin=0, vmax=1, s=3, alpha=0.5, linewidths=0, rasterized=True)
    # M81 bulk proper motion (van der Marel & Guhathakurta 2008 / Gaia DR3)
    ax.scatter([-0.081], [0.058], marker="*", s=200, color="#e63946",
               zorder=6, label="M81 bulk PM (Gaia DR3)", edgecolors="white", lw=0.5)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("XGB P(inside disk)", fontsize=11)
    ax.set_xlabel("μ_α* (mas yr⁻¹)"); ax.set_ylabel("μ_δ (mas yr⁻¹)")
    ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
    ax.set_title(
        f"Proper motion space — {M81['name']}  [out-of-sample]\n"
        "High-score sources should cluster near M81 bulk PM", fontsize=11)
    ax.legend(fontsize=9)
    out = PLOT_DIR / "m81_22_pm_space.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_ned_overlay(df):
    """Fetch NED HII regions around M81 and overlay on sky scatter."""
    try:
        from astroquery.ned import Ned
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        print("  Querying NED for HII regions / associations around M81...")
        pos = SkyCoord(ra=M81["ra"] * u.deg, dec=M81["dec"] * u.deg, frame="icrs")
        ned_table = Ned.query_region(pos, radius=M81["a"] * u.arcmin * 1.3)
        # Filter for HII regions and stellar associations
        type_col = "Type" if "Type" in ned_table.colnames else "Object Type"
        hii_mask = np.array([
            str(t).strip() in ("HII", "HII region", "G", "!*", "As*", "As")
            for t in ned_table[type_col]
        ])
        ned_sub = ned_table[hii_mask]
        print(f"  NED objects (HII/associations): {len(ned_sub)}")
    except Exception as e:
        print(f"  NED query failed: {e}  — skipping overlay")
        ned_sub = None

    da, dd = sky_offsets(df["ra"].to_numpy(), df["dec"].to_numpy())
    scores = df["score"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    sc = ax.scatter(da, dd, c=scores, cmap=SCORE_CMAP, vmin=0, vmax=1,
                    s=3, alpha=0.5, linewidths=0, rasterized=True, zorder=1)
    draw_d25(ax, scale=1.0, fill=False, edgecolor="#e63946", lw=1.8,
             linestyle="--", zorder=5, label="D25")

    if ned_sub is not None and len(ned_sub) > 0:
        ned_ra  = np.array(ned_sub["RA"].data, dtype=float)
        ned_dec = np.array(ned_sub["DEC"].data, dtype=float)
        ned_da  = (ned_ra  - M81["ra"])  * np.cos(np.radians(M81["dec"])) * 60.0
        ned_dd  = (ned_dec - M81["dec"]) * 60.0
        ax.scatter(ned_da, ned_dd, marker="x", s=40, color="#ff7f00",
                   lw=1.2, zorder=6, label=f"NED HII/assoc. (n={len(ned_sub)})")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("XGB P(inside disk)", fontsize=11)
    pad = M81["a"] * 1.4
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Δα·cos δ  (arcmin)"); ax.set_ylabel("Δδ  (arcmin)")
    ax.set_title(
        f"{M81['name']} — XGB score + NED HII regions\n"
        "Independent fact-check: high-score sources should overlap HII regions",
        fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    out = PLOT_DIR / "m81_23_ned_overlay.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull_wsdb", action="store_true",
                        help="Query WSDB and save raw CSV (requires WSDB_PASS env var)")
    args = parser.parse_args()

    if args.pull_wsdb:
        pull_wsdb()
        return

    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run with --pull_wsdb first.")
        sys.exit(1)

    print(f"Loading {CSV_PATH.name}...")
    df_raw = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"  {len(df_raw):,} sources")

    print("Scoring with 15D model...")
    df = score_df(df_raw)
    print(f"  Scored {len(df):,} sources (after NaN drop)")
    print(f"  Inside D25 (geom): {df['label_geom'].sum():,}")
    print(f"  Score ≥ 0.5: {(df['score'] >= 0.5).sum():,}")

    print("\nGenerating plots...")
    plot_sky(df)
    plot_radial(df)
    plot_pm_space(df)
    plot_ned_overlay(df)

    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    sys.path.insert(0, "/data/yn316/pylibs")
    main()
