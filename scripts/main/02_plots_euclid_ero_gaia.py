import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as colors


# ----------------------------
# PATHS
# ----------------------------
BASE = Path(__file__).resolve().parents[1]
OUTDIR = BASE / "output" / "crossmatch" / "gaia_euclid"

R_ARCSEC = 1.0
CSV = OUTDIR / f"euclid_ero_xmatch_gaia_dr3_r{R_ARCSEC}.csv"

 
PLOTDIR = BASE / "plots" / "experiments" / "preliminary_tests"
PLOTDIR.mkdir(parents=True, exist_ok=True)


print("Reading:", CSV)

# ----------------------------
# Read only needed columns
# ----------------------------


cols = [
    "v_alpha_j2000", "v_delta_j2000",   # Euclid coords
    "ra", "dec",                       # Gaia coords
    "v_mag_isocor",                    # Euclid mag
    "phot_g_mean_mag",                 # Gaia G
    "ruwe",                            # Gaia RUWE
    "bp_rp"                            # Gaia colour
]
df = pd.read_csv(CSV, usecols=lambda c: c in set(cols), low_memory=False)
print("Loaded rows:", len(df))

# numeric
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# drop missing coords
d = df.dropna(subset=["v_alpha_j2000", "v_delta_j2000", "ra", "dec"]).copy()
print("Rows with coords:", len(d))

# ----------------------------
# Compute match distance (arcsec) (simple approximation)
# ----------------------------
dra = (d["ra"].to_numpy() - d["v_alpha_j2000"].to_numpy())
dra = (dra + 180.0) % 360.0 - 180.0  # RA wrap safety

dec_rad = np.deg2rad(d["v_delta_j2000"].to_numpy())
ddec = (d["dec"].to_numpy() - d["v_delta_j2000"].to_numpy())

sep_deg = np.sqrt((dra * np.cos(dec_rad))**2 + ddec**2)
d["match_dist_arcsec"] = sep_deg * 3600.0

print("Median match dist (arcsec):", float(d["match_dist_arcsec"].median()))


# ----------------------------
# Scatter: Euclid mag vs Gaia G
# ----------------------------
m = d.dropna(subset=["v_mag_isocor", "phot_g_mean_mag"]).copy()
plt.figure()
plt.scatter(m["phot_g_mean_mag"], m["v_mag_isocor"], s=4)
plt.xlabel("Gaia phot_g_mean_mag (G)")
plt.ylabel("Euclid v_mag_isocor")
plt.title("Euclid VIS mag vs Gaia G mag")
plt.tight_layout()
plt.savefig(PLOTDIR / f"02_euclid_vs_gaia_mag_scatter_r{R_ARCSEC}.png", dpi=200)
plt.close()

# ----------------------------
# 2D histogram (grayscale density) for the same plot
# ----------------------------
plt.figure()
plt.hist2d(
    m["phot_g_mean_mag"],
    m["v_mag_isocor"],
    bins=150,
    cmap="gray",
    norm=colors.LogNorm(vmin=1),  
)
plt.colorbar(label="Count per bin (log scale)")
plt.xlabel("Gaia phot_g_mean_mag (G)")
plt.ylabel("Euclid v_mag_isocor")
plt.title("Euclid VIS vs Gaia G (2D histogram, log scale)")
plt.tight_layout()
plt.savefig(PLOTDIR / f"04_euclid_vs_gaia_mag_hist2d_r{R_ARCSEC}.png", dpi=200)
plt.close()

print("Saved plots to:", PLOTDIR)
