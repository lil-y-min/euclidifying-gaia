#!/usr/bin/env python3
"""
82_galaxy_sky_plots.py
======================
Sky-plane visualisation for the 6 big ERO galaxies:
  - Sources plotted in (Δα·cos δ, Δδ) offset coordinates (arcmin)
  - Colour = XGB P(inside galaxy) from model_inside_galaxy.json
  - D25 ellipse and 0.5·D25 ellipse overlaid
  - True label markers: inside (circle) / outside (dot)
  - One panel per galaxy, arranged in 2×3 grid

Also produces individual per-galaxy figures at higher resolution.

Outputs:  plots/ml_runs/galaxy_env_clf/
    15_galaxy_sky_overview_6panel.png
    16_sky_<galaxy>.png  (×6)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import xgboost as xgb
import astropy.units as u

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts" / "main"))
from feature_schema import get_feature_cols, scaler_stem

DATASET_ROOT = BASE / "output" / "dataset_npz"
MODEL_PATH   = BASE / "output" / "ml_runs" / "galaxy_env_clf" / "model_inside_galaxy_15d.json"
PLOT_DIR     = BASE / "report" / "model_decision" / "20260306_galaxy_disk_membership"
META_NAME    = "metadata_16d.csv"
FEATURE_SET  = "15D"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Optimal threshold (Youden-J on val set, written by script 83)
_thr_json = PLOT_DIR / "optimal_threshold.json"
PRED_THRESHOLD = json.loads(_thr_json.read_text())["thr_youden"] if _thr_json.exists() else 0.5

THR_15B = 0.7

# HyperLEDA elliptical parameters
# (name, RA_deg, Dec_deg, a_arcmin, b_arcmin, pa_deg, r_eff_arcmin, field_tag, morph_type, dist_Mpc)
GALAXIES = [
    ("IC342",       56.6988,   68.0961,  9.976, 9.527,   0.0, 2.7, "ERO-IC342",      "SABc",  3.3),
    ("NGC 6744",   287.4421,  -63.8575,  7.744, 4.775,  13.7, 2.0, "ERO-NGC6744",    "SBbc",  9.4),
    ("NGC 2403",   114.2142,   65.6031,  9.976, 5.000, 126.3, 3.0, "ERO-NGC2403",    "SABc",  3.2),
    ("NGC 6822",   296.2404,  -14.8031,  7.744, 7.396,   0.0, 3.0, "ERO-NGC6822",    "IB",    0.5),
    ("Holmberg II",124.7708,   70.7219,  3.972, 2.812,  15.2, 1.2, "ERO-HolmbergII", "I",     3.4),
    ("IC 10",        5.0721,   59.3039,  3.380, 3.013, 129.0, 1.0, "ERO-IC10",       "IB",    0.7),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_scaler(feature_cols):
    stem = scaler_stem(FEATURE_SET)
    npz  = BASE / "output" / "scalers" / f"{stem}.npz"
    d    = np.load(npz, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    idx  = np.array([names.index(c) for c in feature_cols], dtype=int)
    return d["y_min"].astype(np.float32)[idx], d["y_iqr"].astype(np.float32)[idx]


def elliptical_radius(ra_s, dec_s, ra0, dec0, a, b, pa_deg):
    pa = np.radians(pa_deg)
    da = (ra_s - ra0) * np.cos(np.radians(dec0)) * 60.0   # arcmin East+
    dd = (dec_s - dec0) * 60.0                              # arcmin North+
    x  = da * np.sin(pa) + dd * np.cos(pa)
    y  = da * np.cos(pa) - dd * np.sin(pa)
    return np.sqrt((x / a)**2 + (y / b)**2)


def sky_offsets(ra_s, dec_s, ra0, dec0):
    """Return (Δα·cos δ, Δδ) in arcmin."""
    da = (ra_s - ra0) * np.cos(np.radians(dec0)) * 60.0
    dd = (dec_s - dec0) * 60.0
    return da, dd


def draw_ellipse(ax, a, b, pa_deg, scale=1.0, **kw):
    """Draw an ellipse in (Δα, Δδ) offset space.
    pa_deg: N→E angle of major axis. In offset space, North=+y, East=+x.
    matplotlib Ellipse angle: CCW from +x axis → angle = 90 - pa_deg.
    """
    angle = -pa_deg   # convert PA(N→E) to matplotlib convention (CCW from +x)
    e = Ellipse((0, 0), width=2*b*scale, height=2*a*scale,
                angle=angle, **kw)
    ax.add_patch(e)


def savefig(path, dpi=220):
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


# ── DSS2 sky image ────────────────────────────────────────────────────────────

def fetch_sky_image(ra0, dec0, size_arcmin, survey="DSS2 Red", pixels=600):
    """
    Fetch a DSS2 image centred on (ra0, dec0).

    Returns (data_2d, extent) where extent=(x_left, x_right, y_bot, y_top) in
    offset arcmin (East+, North+), derived from the FITS WCS — so no manual
    flip is needed and the image aligns automatically with the scatter plot axes.
    Returns (None, None) on failure.
    """
    try:
        from astroquery.skyview import SkyView
        from astropy.coordinates import SkyCoord as _SC
        from astropy.wcs import WCS
        pos = _SC(ra=ra0 * u.deg, dec=dec0 * u.deg, frame="icrs")
        imgs = SkyView.get_images(
            position=pos,
            survey=[survey],
            width=size_arcmin * u.arcmin,
            height=size_arcmin * u.arcmin,
            pixels=pixels,
        )
    except Exception as exc:
        print(f"  SkyView fetch failed ({survey}): {exc}")
        return None, None

    if not imgs:
        return None, None

    hdu  = imgs[0][0]
    data = hdu.data.astype(np.float32)

    # Derive extent from WCS so the image aligns with offset coordinates
    from astropy.wcs import WCS
    wcs = WCS(hdu.header)
    ny, nx = data.shape
    # Four corners in pixel coords (0-based): BL, BR, TR, TL
    pix = np.array([[0, 0], [nx - 1, 0], [nx - 1, ny - 1], [0, ny - 1]], dtype=float)
    ra_c, dec_c = wcs.pixel_to_world_values(pix[:, 0], pix[:, 1])
    da_c = (ra_c - ra0) * np.cos(np.radians(dec0)) * 60.0   # East+, arcmin
    dd_c = (dec_c - dec0) * 60.0                              # North+, arcmin

    x_left  = float(np.mean([da_c[0], da_c[3]]))   # da at col 0
    x_right = float(np.mean([da_c[1], da_c[2]]))   # da at col nx-1
    y_bot   = float(np.mean([dd_c[0], dd_c[1]]))   # dd at row 0
    y_top   = float(np.mean([dd_c[2], dd_c[3]]))   # dd at row ny-1

    # Standard FITS: East is on the LEFT (x_left > x_right).
    # Flip image horizontally so East = +x (right), matching the scatter plot.
    if x_left > x_right:
        data = np.fliplr(data)
        x_left, x_right = x_right, x_left

    return data, (x_left, x_right, y_bot, y_top)


def plot_sky_image_panel(ax, data, extent, a, b, pa_deg,
                         survey="DSS2 Red", fontsize=11):
    """Display a DSS2 image with arcsinh stretch and D25 ellipse overlay."""
    lo = np.nanpercentile(data, 2)
    hi = np.nanpercentile(data, 99)
    scale = max(hi - lo, 1e-6)
    img = np.arcsinh((data - lo) / scale * 3.0)
    vmin = np.nanpercentile(img, 1)
    vmax = np.nanpercentile(img, 99)
    im = ax.imshow(
        img, origin="lower", cmap="gray_r",
        vmin=vmin, vmax=vmax,
        extent=extent,
        aspect="equal",
    )
    # D25 ellipse
    from matplotlib.patches import Ellipse as MEllipse
    angle = -pa_deg
    ax.add_patch(MEllipse((0, 0), width=2 * b, height=2 * a, angle=angle,
                           fill=False, edgecolor="#e63946", linewidth=1.8,
                           linestyle="--", zorder=5, label="D25"))
    ax.scatter([0], [0], marker="+", s=100, color="white", linewidths=1.5, zorder=6)
    ax.set_xlabel("Δα·cos δ  (arcmin)", fontsize=fontsize - 1)
    ax.set_ylabel("Δδ  (arcmin)", fontsize=fontsize - 1)
    ax.set_title(f"{survey}  (arcsinh stretch)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    return im


# ── Load data + run inference ─────────────────────────────────────────────────

def load_and_score():
    feature_cols = get_feature_cols(FEATURE_SET)
    feat_min, feat_iqr = load_scaler(feature_cols)

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    all_frames = []
    galaxy_tags = {g[7] for g in GALAXIES}
    for fdir in sorted(DATASET_ROOT.iterdir()):
        meta = fdir / META_NAME
        if not meta.exists():
            continue
        if fdir.name not in galaxy_tags:
            continue
        df = pd.read_csv(meta, low_memory=False)
        df["field_tag"] = fdir.name
        all_frames.append(df)

    df = pd.concat(all_frames, ignore_index=True)
    df = df.dropna(subset=feature_cols + ["ra", "dec"]).copy()

    X = df[feature_cols].to_numpy(dtype=np.float32)
    iqr = np.where(feat_iqr > 0, feat_iqr, 1.0)
    X = (X - feat_min) / iqr

    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    df["score"] = booster.predict(dmat).astype(np.float32)

    # Attach elliptical radius and true label per galaxy
    df["ell_radius"] = np.nan
    df["label_inside"] = 0
    for name, ra0, dec0, a, b, pa, r_eff, tag, *_ in GALAXIES:
        m = df["field_tag"] == tag
        if not m.any():
            continue
        ell = elliptical_radius(
            df.loc[m, "ra"].to_numpy(), df.loc[m, "dec"].to_numpy(),
            ra0, dec0, a, b, pa
        )
        df.loc[m, "ell_radius"] = ell
        df.loc[m, "label_inside"] = (ell <= 1.0).astype(int)

    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

CMAP = "RdPu"

def plot_galaxy_panel(ax, sub, ra0, dec0, a, b, pa_deg, name, morph, dist_mpc, fontsize=9, thr=None):
    da, dd = sky_offsets(sub["ra"].to_numpy(), sub["dec"].to_numpy(), ra0, dec0)

    inside  = sub["label_inside"] == 1
    outside = ~inside
    scores  = sub["score"].to_numpy()

    # Outside sources: coloured by score, small
    ax.scatter(da[outside], dd[outside],
               c=scores[outside], cmap=CMAP, vmin=0, vmax=1,
               s=2, alpha=0.45, linewidths=0, rasterized=True, zorder=1)

    # Inside sources: coloured by score, larger
    sc = ax.scatter(da[inside], dd[inside],
                    c=scores[inside], cmap=CMAP, vmin=0, vmax=1,
                    s=9, alpha=0.9, linewidths=0, rasterized=True, zorder=3)

    # D25 ellipse
    draw_ellipse(ax, a, b, pa_deg, scale=1.0,
                 fill=False, edgecolor="#e63946", linewidth=1.8,
                 linestyle="--", zorder=5, label="D25")
    # 0.5 × D25
    draw_ellipse(ax, a, b, pa_deg, scale=0.5,
                 fill=False, edgecolor="#e63946", linewidth=0.8,
                 linestyle=":", zorder=5, alpha=0.6)

    # Nucleus
    ax.scatter([0], [0], marker="+", s=80, color="white",
               linewidths=1.5, zorder=6)
    ax.scatter([0], [0], marker="+", s=80, color="#333333",
               linewidths=0.8, zorder=7)

    _thr   = thr if thr is not None else PRED_THRESHOLD
    n_in   = int(inside.sum())
    n_pred = int((scores > _thr).sum())
    ax.set_title(
        f"{name}  ({morph}, {dist_mpc} Mpc)\n"
        f"n_inside={n_in:,}   n_pred={n_pred:,}  (thr={_thr:.2f})",
        fontsize=fontsize,
    )
    ax.set_xlabel("Δα·cos δ  (arcmin)", fontsize=fontsize - 1)
    ax.set_ylabel("Δδ  (arcmin)", fontsize=fontsize - 1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=fontsize - 2)

    # Pad axis limits to show full ellipse + buffer
    pad = max(a, b) * 1.35
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)

    return sc


def make_overview(df):
    fig, axes = plt.subplots(2, 3, figsize=(20, 13), constrained_layout=True)
    axes = axes.ravel()

    sc_ref = None
    for i, (name, ra0, dec0, a, b, pa, r_eff, tag, morph, dist) in enumerate(GALAXIES):
        sub = df[df["field_tag"] == tag].copy()
        if sub.empty:
            axes[i].axis("off")
            continue
        sc = plot_galaxy_panel(axes[i], sub, ra0, dec0, a, b, pa, name, morph, dist)
        sc_ref = sc

    fig.suptitle("ERO galaxy fields — Gaia sources coloured by P(inside galaxy disk)",
                 fontsize=14)

    if sc_ref is not None:
        cbar = fig.colorbar(sc_ref, ax=axes.tolist(), shrink=0.6, aspect=30, pad=0.02)
        cbar.set_label("XGB P(inside disk)", fontsize=11)

    # Legend inside last axis (bottom-right)
    leg_elements = [
        mpatches.Patch(facecolor="none", linestyle="--", label="D25 ellipse",
                       edgecolor="#e63946", linewidth=1.8),
        plt.Line2D([0],[0], linestyle="", marker="o", color="#cccccc",
                   markersize=5, label="Outside (grey)"),
        plt.Line2D([0],[0], linestyle="", marker="o", color="#7b2d8b",
                   markersize=5, label="Inside — high P"),
    ]
    axes[-1].legend(handles=leg_elements, loc="lower right", fontsize=8.5,
                    frameon=True, framealpha=0.85)

    savefig(PLOT_DIR / "15_galaxy_sky_overview_6panel.png", dpi=200)


def make_overview_15b(df):
    """Same as make_overview but with threshold=THR_15B, saved as 15b."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13), constrained_layout=True)
    axes = axes.ravel()

    sc_ref = None
    for i, (name, ra0, dec0, a, b, pa, r_eff, tag, morph, dist) in enumerate(GALAXIES):
        sub = df[df["field_tag"] == tag].copy()
        if sub.empty:
            axes[i].axis("off")
            continue
        sc = plot_galaxy_panel(axes[i], sub, ra0, dec0, a, b, pa, name, morph, dist,
                               thr=THR_15B)
        sc_ref = sc

    fig.suptitle(
        f"ERO galaxy fields — Gaia sources coloured by P(inside galaxy disk)  [thr={THR_15B:.2f}]",
        fontsize=14,
    )

    if sc_ref is not None:
        cbar = fig.colorbar(sc_ref, ax=axes.tolist(), shrink=0.6, aspect=30, pad=0.02)
        cbar.set_label("XGB P(inside disk)", fontsize=11)

    leg_elements = [
        mpatches.Patch(facecolor="none", linestyle="--", label="D25 ellipse",
                       edgecolor="#e63946", linewidth=1.8),
        plt.Line2D([0],[0], linestyle="", marker="o", color="#cccccc",
                   markersize=5, label="Outside (grey)"),
        plt.Line2D([0],[0], linestyle="", marker="o", color="#7b2d8b",
                   markersize=5, label="Inside — high P"),
    ]
    axes[-1].legend(handles=leg_elements, loc="lower right", fontsize=8.5,
                    frameon=True, framealpha=0.85)

    savefig(PLOT_DIR / "15b_galaxy_sky_overview_6panel.png", dpi=200)


def make_individual(df):
    for name, ra0, dec0, a, b, pa, r_eff, tag, morph, dist in GALAXIES:
        sub = df[df["field_tag"] == tag].copy()
        if sub.empty:
            continue

        # Fetch DSS2 image — size matches the scatter plot axis limits
        img_size = max(a, b) * 1.35 * 2   # arcmin, same as scatter pad
        sky_data, extent = fetch_sky_image(ra0, dec0, img_size)
        has_image = sky_data is not None

        if has_image:
            fig, (ax_sky, ax_img) = plt.subplots(1, 2, figsize=(17, 8),
                                                   constrained_layout=True)
        else:
            fig, ax_sky = plt.subplots(figsize=(8, 8), constrained_layout=True)

        sc = plot_galaxy_panel(ax_sky, sub, ra0, dec0, a, b, pa, name, morph, dist, fontsize=11)
        cb = fig.colorbar(sc, ax=ax_sky, fraction=0.046, pad=0.04)
        cb.set_label("XGB P(inside disk)", fontsize=10)

        # r_eff circle on sky panel
        circ = plt.Circle((0, 0), r_eff, fill=False, edgecolor="#457b9d",
                           linewidth=1.2, linestyle="-.", alpha=0.7, label=f"r_eff={r_eff}'")
        ax_sky.add_patch(circ)
        ax_sky.legend(fontsize=9, loc="upper right")

        if has_image:
            im = plot_sky_image_panel(ax_img, sky_data, extent, a, b, pa, fontsize=11)
            fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04).set_label(
                "arcsinh(flux)", fontsize=10)

        slug = name.replace(" ", "_").lower()
        savefig(PLOT_DIR / f"16_sky_{slug}.png", dpi=220)


def main():
    print("Loading data and scoring...")
    df = load_and_score()
    print(f"  Total sources: {len(df):,}")
    print(f"  Inside (ell ≤ 1): {df['label_inside'].sum():,}")

    print("Making 6-panel overview...")
    make_overview(df)

    print("Making 6-panel overview (15b, thr=0.7)...")
    make_overview_15b(df)

    print("Making individual galaxy plots (with Euclid flux map)...")
    make_individual(df)

    print("\n=== DONE ===")
    print("Plots dir:", PLOT_DIR)
    for f in sorted(PLOT_DIR.glob("1[56]*.png")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
