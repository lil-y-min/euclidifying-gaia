import numpy as np
import pandas as pd
import sqlutilpy
from pathlib import Path

# ----------------------------
# PATHS (relative to /Codes)
# ----------------------------
BASE = Path(__file__).resolve().parents[1]  # .../Codes
OUTDIR = BASE / "output"
OUTDIR.mkdir(exist_ok=True)

OUTCSV_ALL = OUTDIR / "euclid_ero_to_gaia_dr3_all.csv"
OUTCSV_MATCHED = OUTDIR / "euclid_ero_to_gaia_dr3_matched.csv"

# ----------------------------
# CONNECTION
# ----------------------------
CONN = dict(host="wsdb.ast.cam.ac.uk", db="wsdb", user="yasmine_nourlil_2026")

# ----------------------------
# SETTINGS
# ----------------------------
EUCLID_SCHEMA = "euclid_ero"
EUCLID_NAME = "merge_cat"
EUCLID_TABLE = f"{EUCLID_SCHEMA}.{EUCLID_NAME}"

GAIA_TABLE = "gaia_dr3.gaia_source"

N_EUCLID = 500
R_ARCSEC = 2.0

# ----------------------------
# HELPERS: auto-detect columns
# ----------------------------
def find_ra_dec(cols_lower):
    candidates = [
        ("ra", "dec"),
        ("v_ra", "v_dec"),
        ("v_ra_deg", "v_dec_deg"),
        ("ra_deg", "dec_deg"),
        ("alpha_j2000", "delta_j2000"),
        ("v_alpha_j2000", "v_delta_j2000"),
        ("raj2000", "dej2000"),
        ("v_raj2000", "v_dej2000"),
        ("coord_ra", "coord_dec"),
    ]
    for ra_c, dec_c in candidates:
        if ra_c in cols_lower and dec_c in cols_lower:
            return ra_c, dec_c
    return None, None


def pick_existing(cols_lower, want_list):
    return [c for c in want_list if c.lower() in cols_lower]


# ----------------------------
# 1) QUICK CONNECTION TEST
# ----------------------------
(x,) = sqlutilpy.get("select 1", **CONN)
print("Connected:", x[0])

# ----------------------------
# 2) INSPECT EUCLID COLUMNS TO FIND RA/DEC
# ----------------------------
cols = sqlutilpy.get(
    f"""
    select column_name
    from information_schema.columns
    where table_schema = '{EUCLID_SCHEMA}' and table_name = '{EUCLID_NAME}'
    order by ordinal_position
    """,
    **CONN
)[0]

lower_to_actual = {c.lower(): c for c in cols}
cols_lower = list(lower_to_actual.keys())

ra_l, dec_l = find_ra_dec(cols_lower)
if ra_l is None:
    raise ValueError(
        "Could not auto-detect RA/Dec columns in euclid_ero.merge_cat.\n"
        "Here are the first ~60 columns:\n"
        + ", ".join(cols[:60])
    )

ra_col = lower_to_actual[ra_l]
dec_col = lower_to_actual[dec_l]
print(f"Using Euclid RA/Dec columns: {ra_col}, {dec_col}")

# Euclid photometry we keep (only if present)
euclid_phot_wanted = [
    "v_mag_iso", "v_magerr_iso", "v_flux_iso", "v_fluxerr_iso",
    "v_mag_isocor", "v_magerr_isocor",
]
euclid_phot_cols_l = pick_existing(cols_lower, euclid_phot_wanted)
euclid_phot_cols = [lower_to_actual[c.lower()] for c in euclid_phot_cols_l]

# ----------------------------
# 3) PULL SMALL EUCLID SAMPLE (with a UNIQUE ID)
# ----------------------------
select_sql = f"""
select
  row_number() over (order by {ra_col}, {dec_col}) as euclid_rowid,
  {ra_col} as ra,
  {dec_col} as dec
  {"," if euclid_phot_cols else ""}{", ".join(euclid_phot_cols)}
from {EUCLID_TABLE}
order by {ra_col}, {dec_col}
limit {N_EUCLID}
"""

euclid_data = sqlutilpy.get(select_sql, **CONN)

euclid_rowid = euclid_data[0]
ra = euclid_data[1]
dec = euclid_data[2]
extra_arrays = euclid_data[3:]  # matches euclid_phot_cols order

print("Pulled Euclid rows:", len(ra))
print("Extra Euclid phot cols kept:", euclid_phot_cols)

# ----------------------------
# 4) CROSSMATCH TO GAIA DR3 (nearest within radius)
# ----------------------------
gaia_source_id, gmag, bp_rp, ruwe, parallax, pmra, pmdec, dist_deg = sqlutilpy.local_join(
    f"""
    select tt.source_id::bigint as source_id,
           tt.phot_g_mean_mag,
           tt.bp_rp,
           tt.ruwe,
           tt.parallax,
           tt.pmra,
           tt.pmdec,
           q3c_dist(m.ra, m.dec, tt.ra, tt.dec) as dist_deg
    from mytable m
    left join lateral (
        select source_id, ra, dec,
               phot_g_mean_mag, bp_rp, ruwe, parallax, pmra, pmdec
        from {GAIA_TABLE} g
        where q3c_join(m.ra, m.dec, g.ra, g.dec, {R_ARCSEC}/3600.0)
        order by q3c_dist(m.ra, m.dec, g.ra, g.dec)
        limit 1
    ) tt on true
    order by xid
    """,
    "mytable",
    (euclid_rowid, ra, dec, np.arange(len(dec))),
    ("euclid_rowid", "ra", "dec", "xid"),
    **CONN
)

# ----------------------------
# 5) SAVE CSV (correct missing handling + correct types)
# ----------------------------
df = pd.DataFrame({
    "euclid_rowid": euclid_rowid,
    "ra": ra,
    "dec": dec,
    "gaia_source_id": gaia_source_id,
    "gaia_gmag": gmag,
    "gaia_bp_rp": bp_rp,
    "gaia_ruwe": ruwe,
    "gaia_parallax": parallax,
    "gaia_pmra": pmra,
    "gaia_pmdec": pmdec,
    "match_dist_arcsec": pd.to_numeric(dist_deg, errors="coerce") * 3600.0,
})

for colname, arr in zip(euclid_phot_cols, extra_arrays):
    df[colname] = arr

# -9999 → missing, keep source_id as Int64 (avoid scientific notation in pandas)
df["gaia_source_id"] = pd.to_numeric(df["gaia_source_id"], errors="coerce").astype("Int64")
df.loc[df["gaia_source_id"] == -9999, "gaia_source_id"] = pd.NA
df["has_gaia_match"] = df["gaia_source_id"].notna()

df.to_csv(OUTCSV_ALL, index=False)
df[df["has_gaia_match"]].to_csv(OUTCSV_MATCHED, index=False)

print("Saved ALL:", OUTCSV_ALL)
print("Saved MATCHED:", OUTCSV_MATCHED)
print("Rows:", len(df), "| Gaia matches:", int(df["has_gaia_match"].sum()))
print(df.head(10))
