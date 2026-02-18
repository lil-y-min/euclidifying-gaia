import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales

from feature_schema import (
    GAIA_COLS_OPTIONAL_EXTENDED,
    META_FEATURE_COLS_16D,
    Y_FEATURE_COLS_10D,
    Y_FEATURE_COLS_8D,
    Y_FEATURE_COLS_16D,
)


# =========================
# CONFIG
# =========================
BASE = Path(__file__).resolve().parents[1]  # .../Codes

# NEW: CSV naming convention is now per field:
# output/crossmatch/gaia_euclid/euclid_xmatch_gaia_{tag}.csv
CSV_DIR = BASE / "output" / "crossmatch" / "gaia_euclid"

# FITS images to process (one field per FITS, each with its own CSV)
    # r"D:/NY_IoA/images/ERO-Abell2390/Euclid-VIS-ERO-Abell2390-Flattened.v4-005.fits",
    # r"D:/NY_IoA/images/ERO-Abell2764/Euclid-VIS-ERO-Abell2764-Flattened.v3-003.fits",
    # r"D:/NY_IoA/images/ERO-Barnard30/Euclid-VIS-ERO-Barnard30-Flattened.v3.fits",
    # r"D:/NY_IoA/images/ERO-Dorado/Euclid-VIS-ERO-Dorado-Flattened.v3.fits",
    # r"D:/NY_IoA/images/ERO-Fornax/Euclid-VIS-ERO-Fornax-Flattened.DR3.fits",
    # r"D:/NY_IoA/images/ERO-HolmbergII/Euclid-VIS-ERO-HolmbergII-Flattened.v3.fits",
    # r"D:/NY_IoA/images/ERO-Horsehead/Euclid-VIS-ERO-Horsehead-Flattened.v4.fits",
    # r"D:/NY_IoA/images/ERO-IC10/Euclid-VIS-ERO-IC10-Flattened.v5.fits",
    # r"D:/NY_IoA/images/ERO-IC342/Euclid-VIS-ERO-IC342-Flattened.v4.fits",
    # r"D:/NY_IoA/images/ERO-Messier78/Euclid-VIS-ERO-Messier78-Flattened.v3.fits",
    # r"D:/NY_IoA/images/ERO-NGC2403/Euclid-VIS-ERO-NGC2403-Flattened.v3.fits",
    # r"D:/NY_IoA/images/ERO-NGC6254/Euclid-VIS-ERO-NGC6254-Flattened.v5.fits",
    # r"D:/NY_IoA/images/ERO-NGC6397/Euclid-VIS-ERO-NGC6397-Flattened.v4.fits",
FITS_IMAGES = [
    r"D:/NY_IoA/images/ERO-NGC6744/Euclid-VIS-ERO-NGC6744-Flattened.v4.fits",
    r"D:/NY_IoA/images/ERO-NGC6822/Euclid-VIS-ERO-NGC6822-Flattened.v3-024.fits",
    r"D:/NY_IoA/images/ERO-Perseus/Euclid-VIS-ERO-Perseus-Flattened.v9-004.fits",
    r"D:/NY_IoA/images/ERO-Taurus/Euclid-VIS-ERO-Taurus-Flattened.v4.fits",
]

# Root output folder; each field writes into OUTDIR/tag/
OUTDIR = BASE / "output" / "dataset_npz"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_N = None
NPZ_CHUNK_SIZE = 5_000

STAMP_ARCSEC = 2.0
BLOCK_PIX = 2000
RANDOM_SEED = 42

FOOTPRINT_MARGIN_ARCSEC = 10.0

# NEW (fix leakage)
DEDUP_BY_SOURCE_ID = True  # keep only one row per Gaia source_id (closest dist_match)

COL_RA = "v_alpha_j2000"
COL_DEC = "v_delta_j2000"

GAIA_RA = "ra"
GAIA_DEC = "dec"

GAIA_COLS_REQUIRED = [
    "source_id",
    "phot_g_mean_mag",
    "bp_rp",
    "phot_g_mean_flux_over_error",
    "ruwe",
    "astrometric_excess_noise",
    "parallax_over_error",
    "visibility_periods_used",
    "ipd_frac_multi_peak",
    "phot_bp_rp_excess_factor",
    "pmra",
    "pmdec",
    "pmra_error",
    "pmdec_error",
    "dist_match",
]
GAIA_COLS_OPTIONAL = GAIA_COLS_OPTIONAL_EXTENDED[:]
GAIA_COLS = GAIA_COLS_REQUIRED + GAIA_COLS_OPTIONAL
META_FEATURE_COLS = META_FEATURE_COLS_16D


def field_tag_from_fits(path: str) -> str:
    """
    Human-readable field tag derived from the FITS path.
    We use the parent folder name (e.g. 'ERO-Barnard30').
    """
    p = Path(path)
    tag = p.parent.name.strip()
    return tag if tag else p.stem


# -------------------------
# FITS + WCS helpers
# -------------------------
def open_fits_first_2d(path):
    hdul = fits.open(path, memmap=True)
    hdu_idx = None
    for i, h in enumerate(hdul):
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            hdu_idx = i
            break
    if hdu_idx is None:
        hdul.close()
        raise RuntimeError(f"No 2D image HDU found in FITS: {path}")
    data = hdul[hdu_idx].data
    wcs = WCS(hdul[hdu_idx].header)
    ny, nx = data.shape
    return hdul, data, wcs, nx, ny


def pixel_scale_arcsec_per_pix(wcs):
    scales_deg = proj_plane_pixel_scales(wcs)
    return float(np.mean(scales_deg) * 3600.0)


def extract_stamp_direct(data, cx, cy, stamp_pix):
    half = stamp_pix // 2
    cxi = int(round(cx))
    cyi = int(round(cy))
    x0, x1 = cxi - half, cxi - half + stamp_pix
    y0, y1 = cyi - half, cyi - half + stamp_pix

    if x0 < 0 or y0 < 0 or x1 > data.shape[1] or y1 > data.shape[0]:
        return None, None, None, None, None

    stamp = data[y0:y1, x0:x1]
    if stamp.shape != (stamp_pix, stamp_pix):
        return None, None, None, None, None

    dx = float(cx - cxi)
    dy = float(cy - cyi)
    return stamp, cxi, cyi, dx, dy


def assign_split_from_pixels(x_pix, y_pix, block_pix=2000):
    bx = (x_pix // block_pix).astype(np.int64)
    by = (y_pix // block_pix).astype(np.int64)
    h = (bx * 73856093) ^ (by * 19349663)
    r = np.mod(np.abs(h), 10)
    split = np.zeros_like(r, dtype=np.uint8)
    split[r == 8] = 1  # val
    split[r == 9] = 2  # test
    return split


# -------------------------
# Gaia feature helpers
# -------------------------
def corrected_flux_excess_c_star(bp_rp, phot_bp_rp_excess_factor):
    x = np.asarray(bp_rp, dtype=float)
    c = np.asarray(phot_bp_rp_excess_factor, dtype=float)

    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x) & np.isfinite(c)
    if not np.any(ok):
        return out

    xo = x[ok]
    co = c[ok]
    f = np.empty_like(xo, dtype=float)

    m1 = xo < 0.5
    m2 = (xo >= 0.5) & (xo < 4.0)
    m3 = xo >= 4.0

    if np.any(m1):
        xx = xo[m1]
        f[m1] = 1.154360 + 0.033772 * xx + 0.032277 * (xx ** 2)

    if np.any(m2):
        xx = xo[m2]
        f[m2] = 1.162004 + 0.011464 * xx + 0.049255 * (xx ** 2) - 0.005879 * (xx ** 3)

    if np.any(m3):
        xx = xo[m3]
        f[m3] = 1.057572 + 0.140537 * xx

    out[ok] = co - f
    return out


def pm_significance(pmra, pmdec, pmra_error, pmdec_error):
    pmra = np.asarray(pmra, dtype=float)
    pmdec = np.asarray(pmdec, dtype=float)
    pmra_e = np.asarray(pmra_error, dtype=float)
    pmdec_e = np.asarray(pmdec_error, dtype=float)

    pm_tot = np.hypot(pmra, pmdec)
    pm_tot_err = np.hypot(pmra_e, pmdec_e)

    sig = np.full_like(pm_tot, np.nan, dtype=float)
    ok = np.isfinite(pm_tot) & np.isfinite(pm_tot_err) & (pm_tot_err > 0)
    sig[ok] = pm_tot[ok] / pm_tot_err[ok]
    return pm_tot, pm_tot_err, sig


# -------------------------
# Footprint helpers 
# -------------------------
def _ra_ranges_from_ras(ras_deg, margin_deg):
    ras_deg = np.asarray(ras_deg, dtype=float) % 360.0
    ras_wrap = ((ras_deg + 180.0) % 360.0) - 180.0
    rmin_w = float(np.min(ras_wrap) - margin_deg)
    rmax_w = float(np.max(ras_wrap) + margin_deg)

    r1 = (rmin_w + 360.0) % 360.0
    r2 = (rmax_w + 360.0) % 360.0

    if r1 <= r2:
        return [(r1, r2)]
    return [(0.0, r2), (r1, 360.0)]


def footprint_box_from_tile(wcs, nx, ny, margin_arcsec=10.0):
    margin_deg = float(margin_arcsec) / 3600.0
    xs = np.array([0, nx - 1, nx - 1, 0], dtype=float)
    ys = np.array([0, 0, ny - 1, ny - 1], dtype=float)

    sky = wcs.pixel_to_world(xs, ys)
    if not isinstance(sky, SkyCoord):
        sky = SkyCoord(sky)

    ras = sky.ra.deg % 360.0
    decs = sky.dec.deg

    dec_min = max(-90.0, float(np.min(decs) - margin_deg))
    dec_max = min(90.0, float(np.max(decs) + margin_deg))
    ra_ranges = _ra_ranges_from_ras(ras, margin_deg)
    return dict(dec_min=dec_min, dec_max=dec_max, ra_ranges=ra_ranges)


def mask_inside_box(ra_deg, dec_deg, box):
    ra = (np.asarray(ra_deg, dtype=float) % 360.0)
    dec = np.asarray(dec_deg, dtype=float)

    dec_ok = (dec >= box["dec_min"]) & (dec <= box["dec_max"])
    ra_ok = np.zeros_like(dec_ok, dtype=bool)
    for rmin, rmax in box["ra_ranges"]:
        ra_ok |= (ra >= rmin) & (ra <= rmax)
    return dec_ok & ra_ok


# -------------------------
# Main
# -------------------------
def main():
    np.random.seed(RANDOM_SEED)

    if len(FITS_IMAGES) == 0:
        raise RuntimeError("FITS_IMAGES is empty.")

    # Process each FITS as its own field (tag), with its own CSV and its own output folder.
    for fits_path in FITS_IMAGES:
        tag = field_tag_from_fits(fits_path)

        csv_path = CSV_DIR / f"euclid_xmatch_gaia_{tag}.csv"
        if not Path(csv_path).exists():
            print(f"\n[SKIP] Missing CSV for tag={tag}")
            print("  Expected:", csv_path)
            print("  FITS:", fits_path)
            continue
        if not Path(fits_path).exists():
            print(f"\n[SKIP] Missing FITS for tag={tag}")
            print("  FITS:", fits_path)
            continue

        # Per-field output folder (keeps everything organized and avoids overwriting)
        FIELD_OUTDIR = OUTDIR / tag
        FIELD_OUTDIR.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"FIELD: {tag}")
        print("CSV :", csv_path)
        print("FITS:", fits_path)
        print("OUT :", FIELD_OUTDIR)
        print("=" * 70)

        # sanity: required columns exist in CSV
        head = pd.read_csv(csv_path, nrows=1)
        need_cols = [COL_RA, COL_DEC, GAIA_RA, GAIA_DEC] + GAIA_COLS_REQUIRED
        missing = [c for c in need_cols if c not in head.columns]
        if missing:
            raise RuntimeError(f"CSV is missing columns: {missing}\nCSV: {csv_path}")

        # open tiles
        tiles = []
        for p in [fits_path]:
            hdul, data, wcs, nx, ny = open_fits_first_2d(p)
            box = footprint_box_from_tile(wcs, nx, ny, margin_arcsec=FOOTPRINT_MARGIN_ARCSEC)
            tiles.append(dict(path=p, hdul=hdul, data=data, wcs=wcs, nx=nx, ny=ny, box=box))
        if len(tiles) == 0:
            raise RuntimeError("No tiles opened (unexpected).")

        # stamp size from first tile
        scale0 = pixel_scale_arcsec_per_pix(tiles[0]["wcs"])
        stamp_pix = int(round(STAMP_ARCSEC / scale0))
        if stamp_pix % 2 == 1:
            stamp_pix += 1
        half = stamp_pix // 2

        for t in tiles[1:]:
            sc = pixel_scale_arcsec_per_pix(t["wcs"])
            sp = int(round(STAMP_ARCSEC / sc))
            if sp % 2 == 1:
                sp += 1
            if sp != stamp_pix:
                raise RuntimeError("Stamp pixel size mismatch across tiles.")

        print(f"Tiles: {len(tiles)}")
        for i, t in enumerate(tiles):
            print(f"  [{i}] {t['path']}")
            print(f"      dec [{t['box']['dec_min']:.4f}, {t['box']['dec_max']:.4f}] | ra ranges {t['box']['ra_ranges']}")
        print(f"Pixel scale: {scale0:.5f}\"/pix")
        print(f"Stamp: {STAMP_ARCSEC}\" -> {stamp_pix} pix (half={half})")
        print("TARGET_N:", TARGET_N)
        print("Y features (10D order):", Y_FEATURE_COLS_10D)
        print("Y features (16D order):", Y_FEATURE_COLS_16D)

        # clean old outputs FOR THIS FIELD ONLY
        meta_path = FIELD_OUTDIR / "metadata.csv"
        if meta_path.exists():
            meta_path.unlink()
        for p in sorted(FIELD_OUTDIR.glob("stamps_*.npz")):
            try:
                p.unlink()
            except Exception:
                pass

        # read full CSV + numeric + dropna
        head_cols = set(head.columns.tolist())
        usecols = [COL_RA, COL_DEC, GAIA_RA, GAIA_DEC] + GAIA_COLS_REQUIRED + [c for c in GAIA_COLS_OPTIONAL if c in head_cols]
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
        for c in usecols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in GAIA_COLS_OPTIONAL:
            if c not in df.columns:
                df[c] = np.nan

        before = len(df)
        df = df.dropna(subset=[COL_RA, COL_DEC, GAIA_RA, GAIA_DEC] + GAIA_COLS_REQUIRED)
        drop_missing = before - len(df)

        # dedup by source_id to eliminate leakage + ambiguous targets
        drop_dedup = 0
        if DEDUP_BY_SOURCE_ID:
            before = len(df)
            df = df.sort_values("dist_match", ascending=True).drop_duplicates("source_id", keep="first").reset_index(drop=True)
            drop_dedup = before - len(df)

        print(f"CSV rows after dropna: {len(df)} (dropped missing={drop_missing})")
        if DEDUP_BY_SOURCE_ID:
            print(f"After DEDUP_BY_SOURCE_ID: {len(df)} (dropped duplicates={drop_dedup})")

        # metadata columns
        meta_cols = [
            "npz_file", "index_in_file",
            "fits_index", "fits_path",
            "stamp_arcsec", "stamp_pix", "pixscale_arcsec_per_pix",
            "source_id", "dist_match",
            "x_pix_round", "y_pix_round", "dx_subpix", "dy_subpix",
            "split_code",
            COL_RA, COL_DEC,
            GAIA_RA, GAIA_DEC,
            "phot_g_mean_mag", "bp_rp", "phot_g_mean_flux_over_error",
            "ruwe", "astrometric_excess_noise", "parallax_over_error",
            "visibility_periods_used", "ipd_frac_multi_peak",
            "phot_bp_rp_excess_factor",
            "pmra", "pmdec", "pmra_error", "pmdec_error",
            *GAIA_COLS_OPTIONAL,
            "log10_snr",
            "pm_total", "pm_total_error", "pm_significance",
            "c_star",
            *META_FEATURE_COLS,
        ]
        meta_f = open(meta_path, "w", newline="")
        writer = csv.DictWriter(meta_f, fieldnames=meta_cols)
        writer.writeheader()

        buf_X, buf_y, buf_ids = [], [], []
        buf_meta = []
        npz_index = 0
        total_kept = 0

        drop_no_tile = drop_wcs_nan = drop_edge = drop_stamp_fail = drop_bad_feat = 0
        assigned = np.zeros(len(df), dtype=bool)

        def reached_target():
            return (TARGET_N is not None) and (total_kept >= TARGET_N)

        for ti, t in enumerate(tiles):
            if reached_target():
                break

            m = (~assigned) & mask_inside_box(df[COL_RA].values, df[COL_DEC].values, t["box"])
            if not np.any(m):
                continue

            assigned[m] = True
            sub = df.loc[m].copy().reset_index(drop=True)

            sky = SkyCoord(sub[COL_RA].values * u.deg, sub[COL_DEC].values * u.deg, frame="icrs")
            x, y = t["wcs"].world_to_pixel(sky)

            finite = np.isfinite(x) & np.isfinite(y)
            drop_wcs_nan += int(np.sum(~finite))
            sub = sub.loc[finite].copy().reset_index(drop=True)
            x = x[finite]
            y = y[finite]
            if len(sub) == 0:
                continue

            inside = (x > half) & (x < (t["nx"] - 1 - half)) & (y > half) & (y < (t["ny"] - 1 - half))
            drop_edge += int(np.sum(~inside))
            sub = sub.loc[inside].copy().reset_index(drop=True)
            x = x[inside]
            y = y[inside]
            if len(sub) == 0:
                continue

            snr = sub["phot_g_mean_flux_over_error"].astype(float).values
            good_snr = snr > 0
            drop_bad_feat += int(np.sum(~good_snr))
            sub = sub.loc[good_snr].copy().reset_index(drop=True)
            x = x[good_snr]
            y = y[good_snr]
            snr = snr[good_snr]
            if len(sub) == 0:
                continue
            log10_snr = np.log10(snr)

            pm_tot, pm_tot_err, pm_sig = pm_significance(
                sub["pmra"].astype(float).values,
                sub["pmdec"].astype(float).values,
                sub["pmra_error"].astype(float).values,
                sub["pmdec_error"].astype(float).values,
            )

            c_star = corrected_flux_excess_c_star(
                sub["bp_rp"].astype(float).values,
                sub["phot_bp_rp_excess_factor"].astype(float).values,
            )

            feat10 = np.vstack([
                log10_snr,
                sub["ruwe"].astype(float).values,
                sub["astrometric_excess_noise"].astype(float).values,
                sub["parallax_over_error"].astype(float).values,
                sub["visibility_periods_used"].astype(float).values,
                sub["ipd_frac_multi_peak"].astype(float).values,
                c_star,
                pm_sig,
                sub["phot_g_mean_mag"].astype(float).values,
                sub["bp_rp"].astype(float).values,
            ]).T
            feat16 = np.vstack([
                log10_snr,
                sub["ruwe"].astype(float).values,
                sub["astrometric_excess_noise"].astype(float).values,
                sub["parallax_over_error"].astype(float).values,
                sub["visibility_periods_used"].astype(float).values,
                sub["ipd_frac_multi_peak"].astype(float).values,
                c_star,
                pm_sig,
                sub["astrometric_excess_noise_sig"].astype(float).values,
                sub["ipd_gof_harmonic_amplitude"].astype(float).values,
                sub["ipd_gof_harmonic_phase"].astype(float).values,
                sub["ipd_frac_odd_win"].astype(float).values,
                sub["phot_bp_n_contaminated_transits"].astype(float).values,
                sub["phot_bp_n_blended_transits"].astype(float).values,
                sub["phot_rp_n_contaminated_transits"].astype(float).values,
                sub["phot_rp_n_blended_transits"].astype(float).values,
            ]).T

            good_feat = np.all(np.isfinite(feat10), axis=1)
            drop_bad_feat += int(np.sum(~good_feat))
            sub = sub.loc[good_feat].copy().reset_index(drop=True)
            x = x[good_feat]
            y = y[good_feat]
            feat10 = feat10[good_feat]
            feat16 = feat16[good_feat]
            log10_snr = log10_snr[good_feat]
            pm_tot = pm_tot[good_feat]
            pm_tot_err = pm_tot_err[good_feat]
            pm_sig = pm_sig[good_feat]
            c_star = c_star[good_feat]
            if len(sub) == 0:
                continue

            x_round = np.round(x).astype(np.int64)
            y_round = np.round(y).astype(np.int64)
            split_code = assign_split_from_pixels(x_round, y_round, block_pix=BLOCK_PIX)

            for k in range(len(sub)):
                if reached_target():
                    break

                stamp, cxi, cyi, dx, dy = extract_stamp_direct(t["data"], float(x[k]), float(y[k]), stamp_pix)
                if stamp is None:
                    drop_stamp_fail += 1
                    continue

                yvec = feat10[k].astype(np.float32)
                sid = int(sub["source_id"].iloc[k])

                buf_X.append(stamp.astype(np.float32))
                buf_y.append(yvec)
                buf_ids.append(sid)

                row = {
                    "npz_file": "",
                    "index_in_file": "",
                    "fits_index": int(ti),
                    "fits_path": str(t["path"]),
                    "stamp_arcsec": float(STAMP_ARCSEC),
                    "stamp_pix": int(stamp_pix),
                    "pixscale_arcsec_per_pix": float(scale0),
                    "source_id": sid,
                    "dist_match": float(sub["dist_match"].iloc[k]),
                    "x_pix_round": int(cxi),
                    "y_pix_round": int(cyi),
                    "dx_subpix": float(dx),
                    "dy_subpix": float(dy),
                    "split_code": int(split_code[k]),
                    COL_RA: float(sub[COL_RA].iloc[k]),
                    COL_DEC: float(sub[COL_DEC].iloc[k]),
                    GAIA_RA: float(sub[GAIA_RA].iloc[k]),
                    GAIA_DEC: float(sub[GAIA_DEC].iloc[k]),
                    "phot_g_mean_mag": float(sub["phot_g_mean_mag"].iloc[k]),
                    "bp_rp": float(sub["bp_rp"].iloc[k]),
                    "phot_g_mean_flux_over_error": float(sub["phot_g_mean_flux_over_error"].iloc[k]),
                    "ruwe": float(sub["ruwe"].iloc[k]),
                    "astrometric_excess_noise": float(sub["astrometric_excess_noise"].iloc[k]),
                    "parallax_over_error": float(sub["parallax_over_error"].iloc[k]),
                    "visibility_periods_used": float(sub["visibility_periods_used"].iloc[k]),
                    "ipd_frac_multi_peak": float(sub["ipd_frac_multi_peak"].iloc[k]),
                    "phot_bp_rp_excess_factor": float(sub["phot_bp_rp_excess_factor"].iloc[k]),
                    "pmra": float(sub["pmra"].iloc[k]),
                    "pmdec": float(sub["pmdec"].iloc[k]),
                    "pmra_error": float(sub["pmra_error"].iloc[k]),
                    "pmdec_error": float(sub["pmdec_error"].iloc[k]),
                    "astrometric_excess_noise_sig": float(sub["astrometric_excess_noise_sig"].iloc[k]),
                    "ipd_gof_harmonic_amplitude": float(sub["ipd_gof_harmonic_amplitude"].iloc[k]),
                    "ipd_gof_harmonic_phase": float(sub["ipd_gof_harmonic_phase"].iloc[k]),
                    "ipd_frac_odd_win": float(sub["ipd_frac_odd_win"].iloc[k]),
                    "phot_bp_n_contaminated_transits": float(sub["phot_bp_n_contaminated_transits"].iloc[k]),
                    "phot_bp_n_blended_transits": float(sub["phot_bp_n_blended_transits"].iloc[k]),
                    "phot_rp_n_contaminated_transits": float(sub["phot_rp_n_contaminated_transits"].iloc[k]),
                    "phot_rp_n_blended_transits": float(sub["phot_rp_n_blended_transits"].iloc[k]),
                    "log10_snr": float(log10_snr[k]),
                    "pm_total": float(pm_tot[k]),
                    "pm_total_error": float(pm_tot_err[k]),
                    "pm_significance": float(pm_sig[k]),
                    "c_star": float(c_star[k]),
                }
                for j, name in enumerate(Y_FEATURE_COLS_10D):
                    row[name] = float(yvec[j])
                for j, name in enumerate(Y_FEATURE_COLS_16D):
                    row[name] = float(feat16[k, j])
                buf_meta.append(row)

                if len(buf_X) >= NPZ_CHUNK_SIZE:
                    out_name = f"stamps_{npz_index:05d}.npz"
                    out_path = FIELD_OUTDIR / out_name
                    X = np.stack(buf_X, axis=0)
                    Y = np.stack(buf_y, axis=0)
                    ids = np.array(buf_ids, dtype=np.int64)
                    np.savez_compressed(out_path, X=X, y=Y, source_id=ids)

                    for i_in_file, rrow in enumerate(buf_meta):
                        rrow["npz_file"] = out_name
                        rrow["index_in_file"] = i_in_file
                        writer.writerow(rrow)

                    total_kept += len(ids)
                    print(
                        f"[SAVE] {out_name} | N={len(ids)} | total_kept={total_kept}"
                        f" | drops: missing={drop_missing}, dedup={drop_dedup}, no_tile={drop_no_tile}, wcs_nan={drop_wcs_nan}, "
                        f"edge={drop_edge}, stamp_fail={drop_stamp_fail}, bad_feat={drop_bad_feat}"
                    )

                    buf_X.clear(); buf_y.clear(); buf_ids.clear(); buf_meta.clear()
                    npz_index += 1

        drop_no_tile = int(np.sum(~assigned))

        # flush remainder
        if len(buf_X) > 0 and (not reached_target()):
            out_name = f"stamps_{npz_index:05d}.npz"
            out_path = FIELD_OUTDIR / out_name
            X = np.stack(buf_X, axis=0)
            Y = np.stack(buf_y, axis=0)
            ids = np.array(buf_ids, dtype=np.int64)
            np.savez_compressed(out_path, X=X, y=Y, source_id=ids)

            for i_in_file, rrow in enumerate(buf_meta):
                rrow["npz_file"] = out_name
                rrow["index_in_file"] = i_in_file
                writer.writerow(rrow)

            total_kept += len(ids)
            print(f"[FINAL SAVE] {out_name} | N={len(ids)} | total_kept={total_kept}")

        meta_f.close()

        for t in tiles:
            try:
                t["hdul"].close()
            except Exception:
                pass

        print("\nDONE field:", tag)
        print("OUTDIR:", FIELD_OUTDIR)

    print("\nALL FIELDS DONE.")
    print("ROOT OUTDIR:", OUTDIR)


if __name__ == "__main__":
    main()
