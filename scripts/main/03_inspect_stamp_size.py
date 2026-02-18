import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

from matplotlib.lines import Line2D


# =========================
# CONFIG — EDIT THESE PATHS
# =========================
BASE = Path(__file__).resolve().parents[1]  # .../Codes
FITS_IMAGE = r"D:/NY_IoA/images/ERO-Barnard30/Euclid-VIS-ERO-Barnard30-Flattened.v3.fits"  # or .fits
CSV_R05 = BASE / "output" / "crossmatch" / "gaia_euclid" / "euclid_ero_xmatch_gaia_dr3_r0.5.csv"
CSV_R10 = BASE / "output" / "crossmatch" / "gaia_euclid" / "euclid_ero_xmatch_gaia_dr3_r1.0.csv"

OUTDIR = BASE / "plots" / "experiments" / "stamp_inspection_outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SOURCES_TO_SHOW = 8  # 8 rows x 4 sizes = 32 panels (readable)
STAMP_SIZES_ARCSEC = [0.5, 1.0, 2.0, 4.0]  # try these, then decide

# If CSVs are huge, keep only what is needed for this step:
USECOLS = [
    # Euclid catalogue coords + image coords
    "v_alpha_j2000", "v_delta_j2000", "v_x_image", "v_y_image",
    # Gaia coords
    "ra", "dec",
    # optional, useful for quick filtering/debug
    "dist_match"
]


def open_fits_first_image_hdu(path):
    """Open FITS and return (data2d, wcs). Works even if image is not in HDU 0."""
    hdul = fits.open(path, memmap=True)
    hdu_idx = None
    for i, hdu in enumerate(hdul):
        if hdu.data is None:
            continue
        if hdu.data.ndim == 2:
            hdu_idx = i
            break
    if hdu_idx is None:
        raise RuntimeError("No 2D image HDU found in FITS file.")
    data = hdul[hdu_idx].data
    wcs = WCS(hdul[hdu_idx].header)
    return hdul, data, wcs


def pixel_scale_arcsec_per_pix(wcs):
    """Approximate arcsec/pixel using WCS (mean of axes)."""
    # returns deg/pix for each axis
    scales_deg = proj_plane_pixel_scales(wcs)
    scale_arcsec = np.mean(scales_deg) * 3600.0
    return float(scale_arcsec)


def safe_read_csv(path, usecols):
    df = pd.read_csv(path, usecols=[c for c in usecols if c in pd.read_csv(path, nrows=0).columns])
    return df


def verify_coords(df, label):
    """Print quick sanity checks: Gaia vs Euclid sky coords; WCS pixels vs v_x/y."""
    print("\n==============================")
    print(f"VERIFYING: {label}")
    print("==============================")

    # Check 1: Gaia sky vs Euclid sky
    if {"ra", "dec", "v_alpha_j2000", "v_delta_j2000"}.issubset(df.columns):
        dsub = df[["ra", "dec", "v_alpha_j2000", "v_delta_j2000"]].dropna().head(5000)
        if len(dsub) > 0:
            c_gaia = SkyCoord(dsub["ra"].values * u.deg, dsub["dec"].values * u.deg, frame="icrs")
            c_eucl = SkyCoord(dsub["v_alpha_j2000"].values * u.deg, dsub["v_delta_j2000"].values * u.deg, frame="icrs")
            sep = c_gaia.separation(c_eucl).arcsec
            print(f"[Sky] Gaia (ra,dec) vs Euclid (v_alpha_j2000,v_delta_j2000): "
                  f"median sep = {np.median(sep):.3f}\" ; 95% = {np.percentile(sep,95):.3f}\" ; N={len(sep)}")
        else:
            print("[Sky] Not enough non-NaN rows to compare Gaia vs Euclid sky coords.")
    else:
        print("[Sky] Missing columns for Gaia↔Euclid sky coord comparison.")



def choose_center_columns(df):
    """
    For stamping, prefer Euclid J2000 coords if present; otherwise fallback to Gaia.
    """
    if {"v_alpha_j2000", "v_delta_j2000"}.issubset(df.columns):
        return "v_alpha_j2000", "v_delta_j2000"
    if {"ra", "dec"}.issubset(df.columns):
        return "ra", "dec"
    raise RuntimeError("No usable RA/Dec columns found (expected v_alpha_j2000/v_delta_j2000 or ra/dec).")


def filter_sources_inside_image(df, wcs, data_shape, stamp_sizes_arcsec):
    """Drop sources too close to edges for the *largest* stamp size."""
    ra_col, dec_col = choose_center_columns(df)
    d = df[[ra_col, dec_col]].dropna().copy()
    if len(d) == 0:
        raise RuntimeError("No rows with valid RA/Dec for stamping.")

    # Convert world -> pixel (0-based)
    sky = SkyCoord(d[ra_col].values * u.deg, d[dec_col].values * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(sky)

    d["x_pix"] = x
    d["y_pix"] = y

    # Compute margin in pixels for the largest stamp
    scale = pixel_scale_arcsec_per_pix(wcs)
    max_size = max(stamp_sizes_arcsec)
    half_size_pix = (max_size / scale) / 2.0

    ny, nx = data_shape
    ok = (
        (d["x_pix"] > half_size_pix) &
        (d["x_pix"] < (nx - 1 - half_size_pix)) &
        (d["y_pix"] > half_size_pix) &
        (d["y_pix"] < (ny - 1 - half_size_pix))
    )
    d_ok = d.loc[ok].copy()
    return d_ok, ra_col, dec_col, scale


def plot_stamp_grid(data, wcs, df_ok, ra_col, dec_col, scale_arcsec_per_pix, csv_label):
    np.random.seed(RANDOM_SEED)

    if len(df_ok) < N_SOURCES_TO_SHOW:
        raise RuntimeError(f"Not enough sources inside image after edge filtering: {len(df_ok)}")

    sample = df_ok.sample(N_SOURCES_TO_SHOW, random_state=RANDOM_SEED).reset_index(drop=True)

    nrows = N_SOURCES_TO_SHOW
    ncols = len(STAMP_SIZES_ARCSEC)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.2 * ncols, 3.2 * nrows))

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(nrows):
        ra0 = sample.loc[i, ra_col]
        dec0 = sample.loc[i, dec_col]
        sky0 = SkyCoord(ra0 * u.deg, dec0 * u.deg, frame="icrs")
        x0, y0 = wcs.world_to_pixel(sky0)

        for j, size_arcsec in enumerate(STAMP_SIZES_ARCSEC):
            ax = axes[i, j]

            size_pix = size_arcsec / scale_arcsec_per_pix
            # Cutout2D expects (x, y) in pixel coords
            cut = Cutout2D(data, position=(x0, y0),
                           size=(size_pix, size_pix),
                           wcs=wcs, mode="strict")  # strict because we pre-filtered edges

            # Display with robust scaling so faint stuff is visible
            img = cut.data
            vmin, vmax = np.percentile(img[np.isfinite(img)], [5, 99.5])
            ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

            if i == 0:
                ax.set_title(f'{size_arcsec}"', fontsize=12)

           # --- markers ---
            # WCS / requested centre (sub-pixel)
            px, py = cut.position_cutout
            ax.plot([px], [py], marker="+", markersize=10, color="blue")

            # brightest pixel (diagnostic)
            yy, xx = np.unravel_index(np.nanargmax(img), img.shape)
            ax.plot([xx], [yy], marker="x", markersize=8, color="orange")

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Stamp size comparison - {csv_label}\n",
        fontsize=14
    )
    # One legend for the whole figure
    handles = [
        Line2D([0], [0], marker="+", color="blue", linestyle="None", markersize=10,
            label="WCS centre (catalogue RA/Dec)"),
        Line2D([0], [0], marker="x", color="orange", linestyle="None", markersize=8,
            label="Brightest pixel"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=True)

    plt.tight_layout()

    outpath = os.path.join(OUTDIR, f"stamp_sizes_{csv_label}.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    hdul, data, wcs = open_fits_first_image_hdu(FITS_IMAGE)
    print("Opened FITS:", FITS_IMAGE)
    print("Image shape (ny,nx):", data.shape)
    print("Approx pixel scale [arcsec/pix]:", pixel_scale_arcsec_per_pix(wcs))

    # Load both CSVs
    for csv_path, label in [(CSV_R05, "r0p5"), (CSV_R10, "r1p0")]:
        print("\nLoading:", csv_path)
        # safer: read header first, then use USECOLS intersection
        header_cols = pd.read_csv(csv_path, nrows=0).columns
        use = [c for c in USECOLS if c in header_cols]
        df = pd.read_csv(csv_path, usecols=use)

        print(f"Rows loaded: {len(df)} | Columns loaded: {list(df.columns)}")
        verify_coords(df, label)

        df_ok, ra_col, dec_col, scale = filter_sources_inside_image(df, wcs, data.shape, STAMP_SIZES_ARCSEC)
        print(f"Sources usable (not near edges for max stamp): {len(df_ok)}")

        plot_stamp_grid(data, wcs, df_ok, ra_col, dec_col, scale, label)

    hdul.close()


if __name__ == "__main__":
    main()
