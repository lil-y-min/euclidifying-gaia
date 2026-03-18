#!/usr/bin/env python3
"""
83_galaxy_disk_shap_magnitude.py
=================================
Thesis-quality diagnostic plots for the galaxy disk membership classifier.

Plot 18 — SHAP beeswarm
    SHAP values for all test-set sources.  Shows direction + magnitude of each
    Gaia feature's contribution to P(inside disk).  More interpretable than
    gain importance: reveals whether high RUWE pushes toward inside or outside,
    and how heterogeneous that effect is across sources.

Plot 19 — XGB score vs G magnitude
    Panel A: median XGB score ± IQR binned by G magnitude, split by true label
             (inside / outside disk).  Tests whether the classifier is a trivial
             photometric proxy or genuinely exploits astrometric/morphological
             features.
    Panel B: precision and recall per G-magnitude bin at the Youden threshold.

Outputs:  report/model_decision/20260306_galaxy_disk_membership/
    18_shap_beeswarm.png
    19_score_vs_gmag.png
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xgboost as xgb
from sklearn.model_selection import train_test_split as tts
from astropy.coordinates import SkyCoord
import astropy.units as u

# ── paths ──────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts" / "main"))
from feature_schema import get_feature_cols, scaler_stem

DATASET_ROOT = BASE / "output" / "dataset_npz"
MODEL_PATH   = BASE / "output" / "ml_runs" / "galaxy_env_clf" / "model_inside_galaxy_15d.json"
PLOT_DIR     = BASE / "report" / "model_decision" / "20260306_galaxy_disk_membership"
META_NAME    = "metadata_16d.csv"
FEATURE_SET  = "15D"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

THR_JSON   = PLOT_DIR / "optimal_threshold.json"
THRESHOLD  = json.loads(THR_JSON.read_text())["thr_youden"] if THR_JSON.exists() else 0.5

# Galaxy table (mirrors script 76 / 82)
GALAXIES = [
    ("IC342",       56.6988,   68.0961,  9.976, 9.527,   0.0, 2.7, "ERO-IC342"),
    ("NGC6744",    287.4421,  -63.8575,  7.744, 4.775,  13.7, 2.0, "ERO-NGC6744"),
    ("NGC2403",    114.2142,   65.6031,  9.976, 5.000, 126.3, 3.0, "ERO-NGC2403"),
    ("NGC6822",    296.2404,  -14.8031,  7.744, 7.396,   0.0, 3.0, "ERO-NGC6822"),
    ("HolmbergII", 124.7708,   70.7219,  3.972, 2.812,  15.2, 1.2, "ERO-HolmbergII"),
    ("IC10",         5.0721,   59.3039,  3.380, 3.013, 129.0, 1.0, "ERO-IC10"),
]
OUTSIDE_FIELDS = {
    "ERO-Abell2390", "ERO-Abell2764", "ERO-Barnard30",
    "ERO-Horsehead", "ERO-Messier78", "ERO-Taurus",
    "ERO-NGC6397", "ERO-NGC6254", "ERO-Perseus",
}

# ── thesis style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

INSIDE_COL  = "#7b2d8b"   # purple (high P)
OUTSIDE_COL = "#2166ac"   # blue


# ── helpers ────────────────────────────────────────────────────────────────────

def load_scaler(feature_cols):
    stem  = scaler_stem(FEATURE_SET)
    npz   = BASE / "output" / "scalers" / f"{stem}.npz"
    d     = np.load(npz, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    idx   = np.array([names.index(c) for c in feature_cols], dtype=int)
    return d["y_min"].astype(np.float32)[idx], d["y_iqr"].astype(np.float32)[idx]


def elliptical_radius(ra_s, dec_s, ra0, dec0, a, b, pa_deg):
    pa  = np.radians(pa_deg)
    da  = (ra_s - ra0) * np.cos(np.radians(dec0)) * 60.0
    dd  = (dec_s - dec0) * 60.0
    x   = da * np.sin(pa) + dd * np.cos(pa)
    y   = da * np.cos(pa) - dd * np.sin(pa)
    return np.sqrt((x / a) ** 2 + (y / b) ** 2)


def load_and_label():
    """Load all galaxy + outside fields; attach label, G mag, score."""
    feature_cols = get_feature_cols(FEATURE_SET)
    feat_min, feat_iqr = load_scaler(feature_cols)

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    galaxy_tags = {g[7] for g in GALAXIES}
    all_tags    = galaxy_tags | OUTSIDE_FIELDS

    frames = []
    for fdir in sorted(DATASET_ROOT.iterdir()):
        if not (fdir / META_NAME).exists() or fdir.name not in all_tags:
            continue
        df = pd.read_csv(fdir / META_NAME, low_memory=False)
        df["field_tag"] = fdir.name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=feature_cols + ["ra", "dec"]).copy()

    # G magnitude
    if "phot_g_mean_mag" in df.columns:
        df["g_mag"] = df["phot_g_mean_mag"]
    elif "feat_phot_g_mean_mag" in df.columns:
        df["g_mag"] = df["feat_phot_g_mean_mag"]
    else:
        df["g_mag"] = np.nan

    # Labels
    df["label"] = 0
    for name, ra0, dec0, a, b, pa, r_eff, tag in GALAXIES:
        m = df["field_tag"] == tag
        if not m.any():
            continue
        ell = elliptical_radius(
            df.loc[m, "ra"].to_numpy(), df.loc[m, "dec"].to_numpy(),
            ra0, dec0, a, b, pa
        )
        df.loc[m, "label"] = (ell <= 1.0).astype(int)

    # Scale features + score
    X    = df[feature_cols].to_numpy(dtype=np.float32)
    iqr  = np.where(feat_iqr > 0, feat_iqr, 1.0)
    X_sc = (X - feat_min) / iqr
    dmat = xgb.DMatrix(X_sc, feature_names=feature_cols)
    df["score"] = booster.predict(dmat).astype(np.float32)

    # Reproducible stratified split — same seed as script 76
    y    = df["label"].to_numpy()
    idx  = np.arange(len(y))
    idx_tmp, idx_te = tts(idx, test_size=0.20, stratify=y, random_state=42)
    idx_tr, idx_vl  = tts(idx_tmp, test_size=0.125, stratify=y[idx_tmp], random_state=42)

    return df, X_sc, feature_cols, booster, idx_te, idx_tr


# ── Plot 18: SHAP beeswarm ─────────────────────────────────────────────────────

FEATURE_LABELS = {
    "feat_log10_snr":                       "log₁₀ SNR",
    "feat_ruwe":                            "RUWE",
    "feat_astrometric_excess_noise":        "Astr. excess noise",
    "feat_parallax_over_error":             "Parallax / error",
    "feat_visibility_periods_used":         "Visibility periods",
    "feat_ipd_frac_multi_peak":             "IPD frac. multi-peak",
    "feat_c_star":                          "C*  (flux excess)",
    "feat_pm_significance":                 "PM significance",
    "feat_astrometric_excess_noise_sig":    "Astr. excess noise sig.",
    "feat_ipd_gof_harmonic_amplitude":      "IPD GoF harmonic amp.",
    "feat_ipd_gof_harmonic_phase":          "IPD GoF harmonic phase",
    "feat_ipd_frac_odd_win":                "IPD frac. odd window",
    "feat_phot_bp_n_contaminated_transits": "BP contaminated transits",
    "feat_phot_bp_n_blended_transits":      "BP blended transits",
    "feat_phot_rp_n_contaminated_transits": "RP contaminated transits",
    "feat_phot_rp_n_blended_transits":      "RP blended transits",
}


def plot_shap(df, X_sc, feature_cols, booster, idx_te):
    import shap

    print("  Computing SHAP values on test set...")
    X_te = X_sc[idx_te]
    explainer   = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_te)   # shape (n_test, n_features)

    labels = [FEATURE_LABELS.get(c, c) for c in feature_cols]

    fig, ax = plt.subplots(figsize=(9, 6))

    shap.summary_plot(
        shap_values, X_te,
        feature_names=labels,
        plot_type="dot",
        max_display=16,
        show=False,
        color_bar=True,
        plot_size=None,   # use our own fig
    )

    # summary_plot draws on plt.gcf() — grab and re-style
    gcf = plt.gcf()
    gcf.set_size_inches(9, 6)
    gcf.axes[0].set_xlabel("SHAP value  (impact on log-odds of inside-disk)", fontsize=11)
    gcf.axes[0].set_title(
        "Feature contributions to galaxy disk membership  (SHAP beeswarm, test set)",
        fontsize=12, pad=10,
    )
    gcf.tight_layout()
    out = PLOT_DIR / "18_shap_beeswarm.png"
    gcf.savefig(out, dpi=250, bbox_inches="tight")
    plt.close("all")
    print(f"  saved {out.name}")


# ── Plot 19: score vs G magnitude ─────────────────────────────────────────────

def plot_magnitude(df):
    df_m = df.dropna(subset=["g_mag"]).copy()
    mag_bins = np.arange(14.0, 21.5, 0.5)
    bin_centres = (mag_bins[:-1] + mag_bins[1:]) / 2.0

    inside  = df_m["label"] == 1
    outside = ~inside

    def bin_stats(mask):
        scores = df_m.loc[mask, "score"].to_numpy()
        mags   = df_m.loc[mask, "g_mag"].to_numpy()
        med, q25, q75, ns = [], [], [], []
        for lo, hi in zip(mag_bins[:-1], mag_bins[1:]):
            m = (mags >= lo) & (mags < hi)
            if m.sum() < 5:
                med.append(np.nan); q25.append(np.nan)
                q75.append(np.nan); ns.append(0)
            else:
                med.append(np.median(scores[m]))
                q25.append(np.percentile(scores[m], 25))
                q75.append(np.percentile(scores[m], 75))
                ns.append(m.sum())
        return np.array(med), np.array(q25), np.array(q75), np.array(ns)

    med_in,  q25_in,  q75_in,  n_in  = bin_stats(inside)
    med_out, q25_out, q75_out, n_out = bin_stats(outside)

    # Per-bin precision / recall at THRESHOLD
    prec, rec, n_tot = [], [], []
    for lo, hi in zip(mag_bins[:-1], mag_bins[1:]):
        m = (df_m["g_mag"] >= lo) & (df_m["g_mag"] < hi)
        sub = df_m[m]
        if len(sub) < 5:
            prec.append(np.nan); rec.append(np.nan); n_tot.append(0)
            continue
        pred_pos = sub["score"] >= THRESHOLD
        true_pos = sub["label"] == 1
        tp = (pred_pos & true_pos).sum()
        fp = (pred_pos & ~true_pos).sum()
        fn = (~pred_pos & true_pos).sum()
        prec.append(tp / max(tp + fp, 1))
        rec.append(tp / max(tp + fn, 1))
        n_tot.append(len(sub))
    prec = np.array(prec)
    rec  = np.array(rec)

    # ── figure ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 2]},
                                    constrained_layout=True)

    # Panel A — score vs magnitude
    ax1.fill_between(bin_centres, q25_in,  q75_in,  alpha=0.20, color=INSIDE_COL)
    ax1.fill_between(bin_centres, q25_out, q75_out, alpha=0.20, color=OUTSIDE_COL)
    ax1.plot(bin_centres, med_in,  "-o", color=INSIDE_COL,  ms=5, lw=1.8,
             label="Inside disk  (label = 1)")
    ax1.plot(bin_centres, med_out, "-o", color=OUTSIDE_COL, ms=5, lw=1.8,
             label="Outside disk (label = 0)")
    ax1.axhline(THRESHOLD, color="gray", lw=1.2, ls="--",
                label=f"Threshold = {THRESHOLD:.2f}")
    ax1.axvline(19.5, color="#888", lw=0.9, ls=":", alpha=0.7)
    ax1.text(19.55, 0.05, "G = 19.5\n(Gaia limit)", fontsize=8.5,
             color="#666", va="bottom")
    ax1.set_ylabel("XGB  P(inside disk)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_title(
        "XGB score vs. Gaia G magnitude — galaxy disk membership classifier",
        fontsize=12,
    )
    # Source counts as small bar at top
    ax1b = ax1.twinx()
    ax1b.bar(bin_centres, n_in,  width=0.45, alpha=0.15, color=INSIDE_COL,  align="center")
    ax1b.bar(bin_centres, n_out, width=0.45, alpha=0.10, color=OUTSIDE_COL, align="center",
             bottom=n_in)
    ax1b.set_ylabel("N sources (shaded bars)", fontsize=9, color="#aaa")
    ax1b.tick_params(axis="y", labelcolor="#aaa", labelsize=8)
    ax1b.spines["top"].set_visible(False)

    # Panel B — precision / recall
    ax2.plot(bin_centres, prec, "-s", color="#d62728", ms=5, lw=1.8, label="Precision")
    ax2.plot(bin_centres, rec,  "-^", color="#2ca02c", ms=5, lw=1.8, label="Recall")
    ax2.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax2.axvline(19.5, color="#888", lw=0.9, ls=":", alpha=0.7)
    ax2.set_ylabel(f"Score  (thr = {THRESHOLD:.2f})")
    ax2.set_xlabel("Gaia G magnitude")
    ax2.set_ylim(-0.02, 1.05)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.legend(loc="lower left", framealpha=0.9)
    ax2.set_title("Precision and recall per magnitude bin", fontsize=11)

    out = PLOT_DIR / "19_score_vs_gmag.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df, X_sc, feature_cols, booster, idx_te, idx_tr = load_and_label()
    print(f"  {len(df):,} sources  |  inside={df['label'].sum():,}  |  test set={len(idx_te):,}")

    print("Plot 18 — SHAP beeswarm...")
    plot_shap(df, X_sc, feature_cols, booster, idx_te)

    print("Plot 19 — score vs G magnitude...")
    plot_magnitude(df)

    print("\nDone. Plots in:", PLOT_DIR)


if __name__ == "__main__":
    sys.path.insert(0, "/data/yn316/pylibs")
    main()
