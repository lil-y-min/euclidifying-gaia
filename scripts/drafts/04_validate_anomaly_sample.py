import pandas as pd
import numpy as np
import sys
from pathlib import Path

# ----------------------------
# PATHS
# ----------------------------
BASE = Path(__file__).resolve().parents[1]   # .../Codes
OUTDIR = BASE / "output"
OUTDIR.mkdir(exist_ok=True)

# default input
DEFAULT_CSV = OUTDIR / "gaia_anomalies_to_euclid_vis_cstar.csv"

# allow: python 04_validate...py path/to/file.csv
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV

# outputs (named based on input)
stem = CSV.stem
OUT_STRONG = OUTDIR / f"{stem}_STRONG.csv"
OUT_EXCESS = OUTDIR / f"{stem}_excess_only.csv"
OUT_IPD    = OUTDIR / f"{stem}_ipd_only.csv"
OUT_RUWE   = OUTDIR / f"{stem}_ruwe_only.csv"

df = pd.read_csv(CSV)
print("Loaded rows:", len(df))
print("Input:", CSV)

# --- ensure numeric ---
num_cols = [
    "gaia_gmag", "gaia_bp_rp", "phot_bp_rp_excess_factor", "gaia_ruwe",
    "ipd_frac_multi_peak", "ipd_frac_multi_peak_scaled",
    "bp_rp_excess_resid", "c_star", "match_dist_arcsec", "v_mag_iso"
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def show_range(col):
    if col not in df.columns:
        print(f"{col}: (missing column)")
        return
    s = df[col].dropna()
    if len(s) == 0:
        print(f"{col}: all NaN")
        return
    print(f"{col}: n={len(s)}  min={s.min():.4g}  p1={s.quantile(0.01):.4g}  "
          f"med={s.median():.4g}  p99={s.quantile(0.99):.4g}  max={s.max():.4g}")

print("\n--- RANGES / SANITY ---")
for c in ["gaia_ruwe", "ipd_frac_multi_peak", "ipd_frac_multi_peak_scaled",
          "phot_bp_rp_excess_factor", "bp_rp_excess_resid", "c_star",
          "match_dist_arcsec"]:
    show_range(c)

# ----------------------------
# Strong thresholds (tune later)
# ----------------------------
RUWE_STRONG = 1.4
IPD_STRONG = 0.2  # scaled fraction
CSTAR_STRONG = 0.3

# flags (handle missing columns gracefully)
df["flag_ruwe"] = df["gaia_ruwe"] > RUWE_STRONG if "gaia_ruwe" in df.columns else False

if "ipd_frac_multi_peak_scaled" in df.columns:
    df["flag_ipd"] = df["ipd_frac_multi_peak_scaled"] > IPD_STRONG
else:
    # fallback: if only raw is present and looks percent-like, scale
    if "ipd_frac_multi_peak" in df.columns and df["ipd_frac_multi_peak"].dropna().quantile(0.99) > 1.5:
        df["flag_ipd"] = (df["ipd_frac_multi_peak"] * 0.01) > IPD_STRONG
    else:
        df["flag_ipd"] = False

# choose photometric anomaly source: prefer c_star if present, else bp_rp_excess_resid
if "c_star" in df.columns and df["c_star"].notna().any():
    df["flag_excess"] = df["c_star"] > CSTAR_STRONG
    excess_name = "c_star"
else:
    df["flag_excess"] = df["bp_rp_excess_resid"] > 0.3 if "bp_rp_excess_resid" in df.columns else False
    excess_name = "bp_rp_excess_resid"

print(f"\nUsing photometric-excess metric: {excess_name}")

print("\n--- FLAG COUNTS (strong thresholds) ---")
print("ruwe:", int(df["flag_ruwe"].sum()))
print("excess:", int(df["flag_excess"].sum()))
print("ipd:", int(df["flag_ipd"].sum()))

strong = df["flag_ruwe"] | df["flag_excess"] | df["flag_ipd"]
print("any strong:", int(strong.sum()))

print("\n--- OVERLAPS ---")
print("ruwe & excess:", int((df["flag_ruwe"] & df["flag_excess"]).sum()))
print("ruwe & ipd:", int((df["flag_ruwe"] & df["flag_ipd"]).sum()))
print("excess & ipd:", int((df["flag_excess"] & df["flag_ipd"]).sum()))
print("all three:", int((df["flag_ruwe"] & df["flag_excess"] & df["flag_ipd"]).sum()))

# ----------------------------
# Save subsets
# ----------------------------
df_strong = df[strong].copy()
df_strong.to_csv(OUT_STRONG, index=False)

df[(df["flag_excess"]) & (~df["flag_ruwe"]) & (~df["flag_ipd"])].to_csv(OUT_EXCESS, index=False)
df[(df["flag_ipd"]) & (~df["flag_ruwe"]) & (~df["flag_excess"])].to_csv(OUT_IPD, index=False)
df[(df["flag_ruwe"]) & (~df["flag_excess"]) & (~df["flag_ipd"])].to_csv(OUT_RUWE, index=False)

print("\nSaved strong sample:", OUT_STRONG, "rows:", len(df_strong))
print("Saved excess-only:", OUT_EXCESS)
print("Saved ipd-only:", OUT_IPD)
print("Saved ruwe-only:", OUT_RUWE)
