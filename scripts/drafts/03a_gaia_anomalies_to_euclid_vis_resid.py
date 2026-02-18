import pandas as pd
import sqlutilpy

# ----------------------------
# CONNECTION
# ----------------------------
CONN = dict(host="wsdb.ast.cam.ac.uk", db="wsdb", user="yasmine_nourlil_2026")

EUCLID_TABLE = "euclid_ero.merge_cat"
GAIA_TABLE   = "gaia_dr3.gaia_source"

# Euclid VIS sky coords (we already confirmed these from your earlier run)
EUCLID_RA  = "v_alpha_j2000"
EUCLID_DEC = "v_delta_j2000"

# ----------------------------
# SETTINGS (start permissive; tighten later)
# ----------------------------
N_KEEP = 2000          # how many Gaia anomalies (with Euclid match) to save
R_ARCSEC = 1.0         # start at 1.0" to avoid "too strict"; later you can go 0.5"

# magnitude range (optional; helps avoid extreme weirdness)
G_MIN = 10.0
G_MAX = 21.0

# "anomaly" thresholds (start not-too-strict)
RUWE_MIN = 1.2
IPD_MULTI_PEAK_MIN = 0.01
EXCESS_RESID_MIN = 0.10   # phot_bp_rp_excess_factor - (1.3 + 0.06*bp_rp^2)

OUTCSV = r"c:\Users\HP\Documents\Internship\CAMBRIDGE 2026\Codes\gaia_anomalies_to_euclid_vis.csv"

# ----------------------------
# 1) Connection test
# ----------------------------
(x,) = sqlutilpy.get("select 1", **CONN)
print("Connected:", x[0])

print("Using Euclid RA/Dec:", EUCLID_RA, EUCLID_DEC)

# ----------------------------
# 2) Gaia anomaly definition (simple, good enough to start)
# ----------------------------
# Note: This "excess residual" is a commonly used approximation:
# resid = phot_bp_rp_excess_factor - (1.3 + 0.06 * bp_rp^2)
# Large positive resid -> BP/RP picking up more flux than expected for a clean point source.
anomaly_where = f"""
(
    (ruwe > {RUWE_MIN})
 OR (ipd_frac_multi_peak > {IPD_MULTI_PEAK_MIN})
 OR ((phot_bp_rp_excess_factor - (1.3 + 0.06*power(bp_rp, 2))) > {EXCESS_RESID_MIN})
)
"""

# ----------------------------
# 3) Gaia-first selection BUT restricted to Euclid footprint (key fix)
# ----------------------------
sql = f"""
with gaia_in_euclid as (
    select
        g.source_id,
        g.ra,
        g.dec,
        g.phot_g_mean_mag as gaia_gmag,
        g.bp_rp as gaia_bp_rp,
        g.phot_bp_rp_excess_factor,
        g.ruwe as gaia_ruwe,
        g.ipd_frac_multi_peak,

        (g.phot_bp_rp_excess_factor - (1.3 + 0.06*power(g.bp_rp, 2))) as bp_rp_excess_resid
    from {GAIA_TABLE} g
    where
        g.phot_g_mean_mag between {G_MIN} and {G_MAX}
        and {anomaly_where}
        and exists (
            select 1
            from {EUCLID_TABLE} e
            where q3c_join(g.ra, g.dec, e.{EUCLID_RA}, e.{EUCLID_DEC}, {R_ARCSEC}/3600.0)
        )
    order by g.source_id
    limit {N_KEEP}
)
select
    gc.source_id as gaia_source_id,
    gc.ra as gaia_ra,
    gc.dec as gaia_dec,
    gc.gaia_gmag,
    gc.gaia_bp_rp,
    gc.phot_bp_rp_excess_factor,
    gc.gaia_ruwe,
    gc.ipd_frac_multi_peak,
    gc.bp_rp_excess_resid,

    ee.{EUCLID_RA}  as euclid_ra,
    ee.{EUCLID_DEC} as euclid_dec,
    ee.v_mag_iso,
    ee.v_mag_isocor,
    ee.v_magerr_iso,
    ee.v_flux_iso,
    ee.v_fluxerr_iso,

    q3c_dist(gc.ra, gc.dec, ee.{EUCLID_RA}, ee.{EUCLID_DEC}) * 3600.0 as match_dist_arcsec
from gaia_in_euclid gc
join lateral (
    select *
    from {EUCLID_TABLE} ee
    where q3c_join(gc.ra, gc.dec, ee.{EUCLID_RA}, ee.{EUCLID_DEC}, {R_ARCSEC}/3600.0)
    order by q3c_dist(gc.ra, gc.dec, ee.{EUCLID_RA}, ee.{EUCLID_DEC})
    limit 1
) ee on true
;
"""

res = sqlutilpy.get(sql, **CONN)

cols = [
    "gaia_source_id","gaia_ra","gaia_dec","gaia_gmag","gaia_bp_rp",
    "phot_bp_rp_excess_factor","gaia_ruwe","ipd_frac_multi_peak","bp_rp_excess_resid",
    "euclid_ra","euclid_dec","v_mag_iso","v_mag_isocor","v_magerr_iso","v_flux_iso","v_fluxerr_iso",
    "match_dist_arcsec"
]

df = pd.DataFrame({c: a for c, a in zip(cols, res)})

df["gaia_source_id"] = pd.to_numeric(df["gaia_source_id"], errors="coerce").astype("Int64")
df["match_dist_arcsec"] = pd.to_numeric(df["match_dist_arcsec"], errors="coerce")

df.to_csv(OUTCSV, index=False)

print("Saved:", OUTCSV)
print("Rows (Gaia anomalies WITH Euclid VIS match):", len(df))
print(df.head(10))
