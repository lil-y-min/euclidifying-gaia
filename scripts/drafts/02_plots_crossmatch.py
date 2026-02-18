import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# PATHS (relative to /Codes)
# ----------------------------
BASE = Path(__file__).resolve().parents[1]  # .../Codes
OUTDIR = BASE / "output"
PLOTDIR = BASE / "plots"
PLOTDIR.mkdir(exist_ok=True)

CSV_MATCHED = OUTDIR / "euclid_ero_to_gaia_dr3_matched.csv"

# ----------------------------
# SETTINGS (plot hygiene)
# ----------------------------
DIST_CLEAN_ARCSEC = 0.5
DEDUPLICATE_GAIA = True

# ----------------------------
# LOAD
# ----------------------------
df = pd.read_csv(CSV_MATCHED, dtype={"gaia_source_id": "Int64"})
print("Loaded matched rows:", len(df))

# ----------------------------
# CLEAN SUBSET FOR SANITY PLOTS
# ----------------------------
d = df.copy()
d = d[d["match_dist_arcsec"].notna()].copy()
d = d.sort_values("match_dist_arcsec")

d_clean = d[d["match_dist_arcsec"] < DIST_CLEAN_ARCSEC].copy()

if DEDUPLICATE_GAIA:
    d_clean = d_clean.drop_duplicates("gaia_source_id", keep="first")

print("Clean rows for plots:", len(d_clean))
print("median dist (arcsec):", d_clean["match_dist_arcsec"].median())
print("95% dist (arcsec):", d_clean["match_dist_arcsec"].quantile(0.95))

# ----------------------------
# 1) Match distance histogram
# ----------------------------
plt.figure()
plt.hist(d_clean["match_dist_arcsec"], bins=30)
plt.xlabel("Match distance (arcsec)")
plt.ylabel("Count")
plt.title("Euclid ERO ↔ Gaia DR3 match distance (clean)")
p1 = PLOTDIR / "plot_match_distance.png"
plt.savefig(p1, dpi=200, bbox_inches="tight")
print("Saved:", p1)
plt.show()

# ----------------------------
# 2) Euclid VIS mag vs Gaia G
# ----------------------------
if "v_mag_iso" in d_clean.columns and "gaia_gmag" in d_clean.columns:
    x = d_clean["v_mag_iso"]
    y = d_clean["gaia_gmag"]
    ok = x.notna() & y.notna()

    plt.figure()
    plt.scatter(x[ok], y[ok], s=8)
    plt.xlabel("Euclid v_mag_iso")
    plt.ylabel("Gaia G")
    plt.title("Euclid VIS mag vs Gaia G (clean)")
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    p2 = PLOTDIR / "plot_euclid_mag_vs_gaia_g.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    print("Saved:", p2)
    plt.show()

# ----------------------------
# 3) RUWE histogram
# ----------------------------
if "gaia_ruwe" in d_clean.columns:
    plt.figure()
    plt.hist(d_clean["gaia_ruwe"].dropna(), bins=30)
    plt.xlabel("Gaia RUWE")
    plt.ylabel("Count")
    plt.title("Gaia RUWE distribution (clean)")
    p3 = PLOTDIR / "plot_gaia_ruwe.png"
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    print("Saved:", p3)
    plt.show()

# ----------------------------
# Fractions
# ----------------------------
print("frac < 0.2 arcsec:", (d_clean["match_dist_arcsec"] < 0.2).mean())
print("frac < 0.5 arcsec:", (d_clean["match_dist_arcsec"] < 0.5).mean())
print("frac < 1.0 arcsec:", (d_clean["match_dist_arcsec"] < 1.0).mean())
