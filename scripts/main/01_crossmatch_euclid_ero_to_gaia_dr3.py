"""
01_crossmatch_euclid_ero_to_gaia_dr3.py

Euclid ERO merge_cat -> Gaia DR3 gaia_source crossmatch.

Outputs a compact CSV designed for the stamp exporter:
- nearest Gaia match only (per Euclid row)
- includes dist_match in ARCSEC (column name: dist_match)
- optional FITS footprint prefiltering to speed up query and reduce output size.

Notes:
- If USE_FITS_FOOTPRINT_FILTER=True, the query only considers Euclid sources
  inside the RA/Dec bounding box of the FITS footprint (+ margin).
- If False, may get a huge CSV where most rows fall outside the FITS tile.
"""

import pandas as pd
import numpy as np
import sqlutilpy
from pathlib import Path
import os
from feature_schema import GAIA_COLS_OPTIONAL_EXTENDED

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


# ----------------------------
# PATHS
# ----------------------------
BASE = Path(__file__).resolve().parents[1]   # .../Codes
OUTDIR = BASE / "output" / "crossmatch" / "gaia_euclid"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Output CSV name depends on radius
R_ARCSEC = 0.5

# If True: run once per FITS and write one CSV per field.
# If False: run once using ALL FITS footprints together and write ONE combined CSV.
OUTPUT_PER_FITS = True

FITS_LIST = [
    r"D:/NY_IoA/images/ERO-Abell2390/Euclid-VIS-ERO-Abell2390-Flattened.v4-005.fits",
    r"D:/NY_IoA/images/ERO-Abell2764/Euclid-VIS-ERO-Abell2764-Flattened.v3-003.fits",
    r"D:/NY_IoA/images/ERO-Barnard30/Euclid-VIS-ERO-Barnard30-Flattened.v3.fits",
    r"D:/NY_IoA/images/ERO-Dorado/Euclid-VIS-ERO-Dorado-Flattened.v3.fits",
    r"D:/NY_IoA/images/ERO-Fornax/Euclid-VIS-ERO-Fornax-Flattened.DR3.fits",
    r"D:/NY_IoA/images/ERO-HolmbergII/Euclid-VIS-ERO-HolmbergII-Flattened.v3.fits",
    r"D:/NY_IoA/images/ERO-Horsehead/Euclid-VIS-ERO-Horsehead-Flattened.v4.fits",
    r"D:/NY_IoA/images/ERO-IC10/Euclid-VIS-ERO-IC10-Flattened.v5.fits",
    r"D:/NY_IoA/images/ERO-IC342/Euclid-VIS-ERO-IC342-Flattened.v4.fits",
    r"D:/NY_IoA/images/ERO-Messier78/Euclid-VIS-ERO-Messier78-Flattened.v3.fits",
    r"D:/NY_IoA/images/ERO-NGC2403/Euclid-VIS-ERO-NGC2403-Flattened.v3.fits",
    r"D:/NY_IoA/images/ERO-NGC6254/Euclid-VIS-ERO-NGC6254-Flattened.v5.fits",
    r"D:/NY_IoA/images/ERO-NGC6397/Euclid-VIS-ERO-NGC6397-Flattened.v4.fits",
    r"D:/NY_IoA/images/ERO-NGC6744/Euclid-VIS-ERO-NGC6744-Flattened.v4.fits",
    r"D:/NY_IoA/images/ERO-NGC6822/Euclid-VIS-ERO-NGC6822-Flattened.v3-024.fits",
    r"D:/NY_IoA/images/ERO-Perseus/Euclid-VIS-ERO-Perseus-Flattened.v9-004.fits",
    r"D:/NY_IoA/images/ERO-Taurus/Euclid-VIS-ERO-Taurus-Flattened.v4.fits",
]

USE_FITS_FOOTPRINT_FILTER = True   # strongly recommended
FOOTPRINT_MARGIN_ARCSEC = 10.0     # extra margin around image footprint


def field_tag_from_fits(path: str) -> str:
    """
    Human-readable tag used in output filenames.
    Uses the parent folder name (e.g. 'ERO-Barnard30').
    """
    p = Path(path)
    tag = p.parent.name.strip()
    return tag if tag else p.stem


# ----------------------------
# CONNECTION
# ----------------------------
CONN = dict(
    host="wsdb.ast.cam.ac.uk",
    db="wsdb",
    user="yasmine_nourlil_2026",
    password=(os.getenv("WSDB_PASSWORD") or os.getenv("PGPASSWORD")),
)


# ----------------------------
# TABLES / COLUMNS
# ----------------------------
EUCLID_SCHEMA = "euclid_ero"
EUCLID_NAME = "merge_cat"
EUCLID_TABLE = f"{EUCLID_SCHEMA}.{EUCLID_NAME}"

GAIA_SCHEMA = "gaia_dr3"
GAIA_NAME = "gaia_source"
GAIA_TABLE = f"{GAIA_SCHEMA}.{GAIA_NAME}"

EUCLID_RA = "v_alpha_j2000"
EUCLID_DEC = "v_delta_j2000"

LIMIT_EUCLID = None  # set to an int for debug (e.g. 200000). Keep None for full.


# ----------------------------
# Columns to keep in output CSV
# ----------------------------
EUCLID_KEEP = [
    EUCLID_RA,
    EUCLID_DEC,
    "v_mag_isocor",  # optional; will be skipped if missing
]

GAIA_KEEP = [
    "source_id",
    "ra",
    "dec",
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
]
GAIA_KEEP += GAIA_COLS_OPTIONAL_EXTENDED


# ----------------------------
# FITS footprint helpers
# ----------------------------
def _open_fits_first_2d_header(path: str):
    """Return (header, nx, ny) for the first 2D image HDU."""
    hdul = fits.open(path, memmap=True)
    try:
        for h in hdul:
            naxis = int(h.header.get("NAXIS", 0))
            if naxis >= 2 and h.header.get("NAXIS1") and h.header.get("NAXIS2"):
                nx = int(h.header["NAXIS1"])
                ny = int(h.header["NAXIS2"])
                header = h.header
                return header, nx, ny
    finally:
        hdul.close()
    raise RuntimeError(f"No 2D image HDU found in FITS: {path}")


def _ra_ranges_from_ras(ras_deg, margin_deg):
    """
    Convert a set of RA values (deg) into 1 or 2 non-wrapping ranges in [0,360),
    expanded by margin_deg.
    """
    ras_deg = np.asarray(ras_deg, dtype=float) % 360.0

    # Wrap to [-180, +180) to find a tight interval
    ras_wrap = ((ras_deg + 180.0) % 360.0) - 180.0
    rmin_w = float(np.min(ras_wrap) - margin_deg)
    rmax_w = float(np.max(ras_wrap) + margin_deg)

    # Convert ends back to [0,360)
    r1 = (rmin_w + 360.0) % 360.0
    r2 = (rmax_w + 360.0) % 360.0

    # If no wrap across 0
    if r1 <= r2:
        return [(r1, r2)]

    # Wrap across 0 => [0,r2] U [r1,360]
    return [(0.0, r2), (r1, 360.0)]


def footprint_box_from_fits(path: str, margin_arcsec=10.0):
    header, nx, ny = _open_fits_first_2d_header(path)
    wcs = WCS(header)

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

    return dict(fits_path=path, dec_min=dec_min, dec_max=dec_max, ra_ranges=ra_ranges)


def footprint_boxes_from_fits(fits_list, margin_arcsec=10.0):
    return [footprint_box_from_fits(p, margin_arcsec=margin_arcsec) for p in fits_list]


def sql_where_from_boxes(boxes, ra_col, dec_col, alias="a"):
    """
    Build a SQL WHERE clause that selects rows inside ANY of the boxes.
    Handles RA wrap by using up to two ranges per box.
    """
    parts = []
    for b in boxes:
        dec_part = f"({alias}.{dec_col} BETWEEN {b['dec_min']} AND {b['dec_max']})"
        ra_parts = []
        for rmin, rmax in b["ra_ranges"]:
            ra_parts.append(f"({alias}.{ra_col} BETWEEN {rmin} AND {rmax})")
        ra_part = "(" + " OR ".join(ra_parts) + ")"
        parts.append("(" + dec_part + " AND " + ra_part + ")")
    return "(" + " OR ".join(parts) + ")" if parts else ""


# ----------------------------
# Main
# ----------------------------
def main():
    # ----------------------------
    # 1) Build output columns
    # ----------------------------
    keep_e = EUCLID_KEEP[:]
    keep_g = GAIA_KEEP[:]
    out_cols = keep_e + keep_g + ["dist_match"]

    # ----------------------------
    # 2) Build runs (per FITS or combined)
    # ----------------------------
    runs = []
    if OUTPUT_PER_FITS and len(FITS_LIST) > 0:
        for p in FITS_LIST:
            tag = field_tag_from_fits(p)
            outcsv = OUTDIR / f"euclid_xmatch_gaia_{tag}.csv"
            runs.append(dict(tag=tag, fits_list=[p], outcsv=outcsv))
    else:
        # one combined run
        outcsv = OUTDIR / f"euclid_ero_xmatch_gaia_dr3_r{R_ARCSEC}.csv"
        runs.append(dict(tag="ALL", fits_list=FITS_LIST[:], outcsv=outcsv))

    # ----------------------------
    # 3) Execute each run
    # ----------------------------
    radius_deg = R_ARCSEC / 3600.0

    for run in runs:
        tag = run["tag"]
        fits_list = run["fits_list"]
        OUTCSV = run["outcsv"]

        # ----------------------------
        # 3a) Optional footprint filter
        # ----------------------------
        footprint_where = ""
        if USE_FITS_FOOTPRINT_FILTER:
            if len(fits_list) == 0:
                print("\nWARNING: USE_FITS_FOOTPRINT_FILTER=True but FITS_LIST is empty for this run.")
                print("Set FITS_LIST = [r'...x.fits...'] or set USE_FITS_FOOTPRINT_FILTER=False.\n")
            else:
                boxes = footprint_boxes_from_fits(fits_list, margin_arcsec=FOOTPRINT_MARGIN_ARCSEC)
                footprint_where = sql_where_from_boxes(boxes, EUCLID_RA, EUCLID_DEC, alias="a")

                print(f"\n=== Run tag: {tag} ===")
                print(f"Footprint filter ENABLED using {len(fits_list)} FITS file(s).")
                for b in boxes:
                    print(" -", b["fits_path"])
                    print(f"   dec: [{b['dec_min']:.4f}, {b['dec_max']:.4f}]")
                    print("   ra ranges:", ", ".join([f"[{r[0]:.4f},{r[1]:.4f}]" for r in b["ra_ranges"]]))
        else:
            print(f"\n=== Run tag: {tag} ===")
            print("Footprint filter DISABLED.")

        # ----------------------------
        # 3b) Crossmatch SQL (nearest-only)
        # ----------------------------
        euclid_where_parts = []
        if footprint_where:
            euclid_where_parts.append(footprint_where)

        euclid_where_sql = ("WHERE " + " AND ".join(euclid_where_parts)) if euclid_where_parts else ""
        limit_sql = f"LIMIT {int(LIMIT_EUCLID)}" if LIMIT_EUCLID is not None else ""

        sel_e = ", ".join([f"a.{c}" for c in keep_e])
        sel_g = ", ".join([f"b.{c}" for c in keep_g])

        sql = f"""
        WITH a AS MATERIALIZED (
            SELECT {", ".join(keep_e)}
            FROM {EUCLID_TABLE} a
            {euclid_where_sql}
            {limit_sql}
        )
        SELECT
            {sel_e},
            {sel_g},
            (q3c_dist(a.{EUCLID_RA}, a.{EUCLID_DEC}, b.ra, b.dec) * 3600.0) AS dist_match
        FROM a
        JOIN LATERAL (
            SELECT {", ".join([f"b.{c}" for c in keep_g])}
            FROM {GAIA_TABLE} b
            WHERE q3c_join(a.{EUCLID_RA}, a.{EUCLID_DEC}, b.ra, b.dec, {radius_deg})
            ORDER BY q3c_dist(a.{EUCLID_RA}, a.{EUCLID_DEC}, b.ra, b.dec) ASC
            LIMIT 1
        ) b ON true;
        """

        print("\nRunning SQL (nearest-only crossmatch)...")
        res = sqlutilpy.get(sql, **CONN)

        if len(res) != len(out_cols):
            raise RuntimeError(
                f"SQL returned {len(res)} columns, expected {len(out_cols)}.\n"
                f"Expected columns: {out_cols}"
            )

        df = pd.DataFrame({col: np.array(arr) for col, arr in zip(out_cols, res)})

        # Add provenance columns (constant per output file, keeps things organized later)
        df.insert(0, "field_tag", tag)
        if OUTPUT_PER_FITS and len(fits_list) == 1:
            df.insert(1, "fits_path", fits_list[0])
        else:
            df.insert(1, "fits_path", "MULTI_OR_NONE")

        # ----------------------------
        # 3c) Save + quick sanity
        # ----------------------------
        if len(df) > 0:
            dm = pd.to_numeric(df["dist_match"], errors="coerce")
            print("\nSanity check dist_match (arcsec):")
            print("  N:", len(dm))
            print("  median:", float(np.nanmedian(dm)))
            print("  95%:", float(np.nanpercentile(dm, 95)))
            print("  max:", float(np.nanmax(dm)))
            print("  should be <= radius:", R_ARCSEC)

        df.to_csv(OUTCSV, index=False)
        print("\nSaved:", OUTCSV)
        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print(df.head(3))


if __name__ == "__main__":
    main()
