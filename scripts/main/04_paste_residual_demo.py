import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales

from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib import colors

# ===========
# EDIT PATHS
# ===========
BASE = Path(__file__).resolve().parents[1]  # .../Codes
FITS_IMAGE = r"D:/NY_IoA/images/ERO-Barnard30/Euclid-VIS-ERO-Barnard30-Flattened.v3.fits"
CSV = BASE / "output" / "crossmatch" / "gaia_euclid" / "euclid_ero_xmatch_gaia_dr3_r0.5.csv"

OUTDIR = BASE / "plots" / "experiments" / "stamp_inspection_outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

STAMP_ARCSEC = 3.2          # 3.2" at 0.1"/pix ~ 32 pix
REGION_PIX = 800            # region cutout size (pixels)
MAX_STAMPS_TO_PASTE = 150   # skip overlaps, so actual pasted may be lower
RANDOM_SEED = 22

# For the 3 panels (recommended)
MAKE_THREE_PANEL = False

# For stamp boxes on the original
DRAW_STAMP_BOXES = True

# Try multiple region centers to get a region with decent number of candidates
MAX_CENTER_TRIES = 50
MIN_CANDIDATES_IN_REGION = 30  # increase for denser plots


def open_fits_first_2d(path):
    """Open FITS and return (hdul, data2d, wcs) from the first 2D HDU."""
    hdul = fits.open(path, memmap=True)  # keep memmap to avoid loading huge array into RAM
    hdu_idx = None
    for i, h in enumerate(hdul):
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            hdu_idx = i
            break
    if hdu_idx is None:
        raise RuntimeError("No 2D image HDU found in FITS file.")
    data = hdul[hdu_idx].data
    wcs = WCS(hdul[hdu_idx].header)
    return hdul, data, wcs


def pixel_scale_arcsec_per_pix(wcs):
    scales_deg = proj_plane_pixel_scales(wcs)
    return float(np.mean(scales_deg) * 3600.0)


def robust_limits(img, mask=None, p_lo=5, p_hi=99.5):
    """Percentile limits, optionally using only masked pixels."""
    if mask is not None:
        vals = img[mask]
    else:
        vals = img[np.isfinite(img)]
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        return np.nanmin(img), np.nanmax(img)
    vmin, vmax = np.percentile(vals, [p_lo, p_hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return np.nanmin(vals), np.nanmax(vals)
    return vmin, vmax


def main():
    np.random.seed(RANDOM_SEED)

    # ----------------------------
    # 1) Open image + WCS
    # ----------------------------
    hdul, data, wcs = open_fits_first_2d(FITS_IMAGE)
    ny, nx = data.shape

    scale = pixel_scale_arcsec_per_pix(wcs)
    stamp_pix = int(round(STAMP_ARCSEC / scale))
    if stamp_pix % 2 == 1:
        stamp_pix += 1
    half = stamp_pix // 2

    print("Pixel scale [arcsec/pix]:", scale)
    print("Stamp size:", STAMP_ARCSEC, "arcsec ~", stamp_pix, "pix")

    # ----------------------------
    # 2) Load RA/Dec from CSV
    # ----------------------------
    usecols = ["v_alpha_j2000", "v_delta_j2000"]
    df = pd.read_csv(CSV, usecols=usecols).dropna()
    print("Loaded matched rows with coords:", len(df))

    # ----------------------------
    # 3) WCS world->pixel for all rows, filter usable centers
    # ----------------------------
    sky_all = SkyCoord(df["v_alpha_j2000"].values * u.deg,
                       df["v_delta_j2000"].values * u.deg,
                       frame="icrs")
    x_all, y_all = wcs.world_to_pixel(sky_all)

    finite = np.isfinite(x_all) & np.isfinite(y_all)

    region_half = REGION_PIX / 2.0
    margin = region_half + half + 2

    inside = (
        (x_all > margin) & (x_all < (nx - 1 - margin)) &
        (y_all > margin) & (y_all < (ny - 1 - margin))
    )

    ok = finite & inside
    df_ok = df.loc[ok].copy()
    df_ok["x"] = x_all[ok]
    df_ok["y"] = y_all[ok]

    print("Rows usable for region-centers (finite + safely inside image):", len(df_ok))
    if len(df_ok) == 0:
        raise RuntimeError("No usable rows after filtering. CSV coords may not match this FITS footprint.")

    # ----------------------------
    # 4) Try centers until we find a region with enough candidates
    # ----------------------------
    best = None  # (n_cand, region, region_data, region_wcs)
    for trial in range(MAX_CENTER_TRIES):
        center = df_ok.sample(1, random_state=RANDOM_SEED + trial).iloc[0]
        cx, cy = float(center["x"]), float(center["y"])

        region = Cutout2D(data, position=(cx, cy), size=(REGION_PIX, REGION_PIX), wcs=wcs, mode="trim")
        region_data = region.data
        region_wcs = region.wcs

        # Candidate count inside region (safe for stamping)
        sky_ok = SkyCoord(df_ok["v_alpha_j2000"].values * u.deg,
                          df_ok["v_delta_j2000"].values * u.deg,
                          frame="icrs")
        rx, ry = region_wcs.world_to_pixel(sky_ok)

        rny, rnx = region_data.shape
        inside_region = (
            (rx > half) & (rx < (rnx - 1 - half)) &
            (ry > half) & (ry < (rny - 1 - half))
        )

        n_cand = int(np.sum(inside_region))
        if best is None or n_cand > best[0]:
            best = (n_cand, region, region_data, region_wcs, rx, ry, inside_region)

        if n_cand >= MIN_CANDIDATES_IN_REGION:
            print(f"Chose trial {trial}: candidates inside region = {n_cand}")
            break

    # Use best region found (even if it didn’t reach the target)
    n_cand, region, region_data, region_wcs, rx, ry, inside_region = best
    print("Region cutout shape:", region_data.shape)
    print("Candidates inside region (safe for stamping):", n_cand)

    cand = df_ok.loc[inside_region].copy()
    cand["rx"] = rx[inside_region]
    cand["ry"] = ry[inside_region]
    cand = cand.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # ----------------------------
    # 5) Paste stamps into blank canvas (skip overlaps)
    # ----------------------------
    canvas = np.zeros_like(region_data, dtype=float)
    mask = np.zeros_like(region_data, dtype=bool)
    pasted_positions = []

    pasted = 0
    for _, r in cand.iterrows():
        x = int(round(r["rx"]))
        y = int(round(r["ry"]))
        x0, x1 = x - half, x + half
        y0, y1 = y - half, y + half

        if mask[y0:y1, x0:x1].any():
            continue

        stamp = region_data[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = stamp
        mask[y0:y1, x0:x1] = True

        pasted_positions.append((x0, y0))
        pasted += 1
        if pasted >= MAX_STAMPS_TO_PASTE:
            break

    print("Pasted stamps:", pasted)
    print("Masked pixels:", int(mask.sum()), f"({mask.mean()*100:.4f}% of region)")

    # ----------------------------
    # 6) Residual on pasted pixels
    # ----------------------------
    resid = np.zeros_like(region_data, dtype=float)
    resid[mask] = region_data[mask] - canvas[mask]
    max_abs = float(np.max(np.abs(resid[mask]))) if mask.any() else float("nan")
    print("Max |residual| on mask:", max_abs)

    # ----------------------------
    # 7) Plot outputs
    # ----------------------------
    if MAKE_THREE_PANEL:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        vmin0, vmax0 = robust_limits(region_data)
        axes[0].imshow(region_data, origin="lower", cmap="gray", vmin=vmin0, vmax=vmax0)
        if DRAW_STAMP_BOXES:
            for (x0, y0) in pasted_positions:
                axes[0].add_patch(Rectangle((x0, y0), stamp_pix, stamp_pix,
                                            fill=False, linewidth=1.0, edgecolor="red"))
        axes[0].set_title(f"Original region (stamp boxes, N={pasted})")
        axes[0].set_xticks([]); axes[0].set_yticks([])

        vmin1, vmax1 = robust_limits(canvas, mask=mask)
        axes[1].imshow(canvas, origin="lower", cmap="gray", vmin=vmin1, vmax=vmax1)
        axes[1].set_title("Pasted canvas")
        axes[1].set_xticks([]); axes[1].set_yticks([])

        abs_resid = np.zeros_like(resid)
        abs_resid[mask] = np.abs(resid[mask])
        vmin2, vmax2 = robust_limits(abs_resid, mask=mask, p_lo=0, p_hi=99.5)
        axes[2].imshow(abs_resid, origin="lower", cmap="gray", vmin=vmin2, vmax=vmax2)
        axes[2].set_title("Abs residual (only pasted pixels)")
        axes[2].set_xticks([]); axes[2].set_yticks([])

        outpath = os.path.join(OUTDIR, f"residual_stamp{STAMP_ARCSEC:.1f}arcsec.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close(fig)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        vmin0, vmax0 = robust_limits(region_data)
        ax.imshow(region_data, origin="lower", cmap="gray", vmin=vmin0, vmax=vmax0)
        for (x0, y0) in pasted_positions:
            ax.add_patch(Rectangle((x0, y0), stamp_pix, stamp_pix,
                                   fill=False, linewidth=1.0, edgecolor="red"))
        ax.set_title(f"Original region (stamp boxes, N={pasted})")
        ax.set_xticks([]); ax.set_yticks([])
        outpath = os.path.join(OUTDIR, f"residual_stamp{STAMP_ARCSEC:.1f}arcsec.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close(fig)

    print("Saved:", outpath)
    hdul.close()


if __name__ == "__main__":
    main()
