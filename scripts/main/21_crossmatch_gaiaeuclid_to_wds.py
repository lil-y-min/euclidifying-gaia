from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u


BASE = Path(__file__).resolve().parents[1]  # .../Codes

WDS_COLS = [
    "COORD", "DISC", "COMP", "FSTDATE", "LSTDATE", "NOBS", "FSTPA", "LSTPA",
    "FSTSEP", "LSTSEP", "FSTMAG", "SECMAG", "STYPE", "PM1RA", "PM1DC", "PM2RA",
    "PM2DC", "DNUM", "NOTES", "ACOORD",
]

# ACOORD format example: '000006.64+752859.8'
# RA = HHMMSS.ss ; Dec = sDDMMSS.s
ACOORD_RE = re.compile(r"^(\d{2})(\d{2})(\d{2}(?:\.\d+)?)((?:\+|\-))(\d{2})(\d{2})(\d{2}(?:\.\d+)?)$")


def parse_numeric_or_none(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    if t == "" or t.upper() == "NULL":
        return None
    try:
        return float(t)
    except Exception:
        return None


def parse_acoord_to_deg(acoord: str) -> Tuple[Optional[float], Optional[float]]:
    if acoord is None:
        return None, None
    t = acoord.strip()
    m = ACOORD_RE.match(t)
    if not m:
        return None, None

    hh, mm, ss, sign, dd, dm, ds = m.groups()

    ra_h = float(hh)
    ra_m = float(mm)
    ra_s = float(ss)
    ra_deg = (ra_h + ra_m / 60.0 + ra_s / 3600.0) * 15.0

    dec_d = float(dd)
    dec_m = float(dm)
    dec_s = float(ds)
    dec_abs = dec_d + dec_m / 60.0 + dec_s / 3600.0
    dec_deg = dec_abs if sign == "+" else -dec_abs

    return ra_deg, dec_deg


def _clean_sql_value(v: str) -> Optional[str]:
    t = v.strip()
    if t.upper() == "NULL":
        return None
    return t


def parse_wds_sql(sql_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    with open(sql_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("INSERT INTO WDS VALUES"):
                continue

            lpar = s.find("(")
            rpar = s.rfind(")")
            if lpar < 0 or rpar < 0 or rpar <= lpar:
                continue

            payload = s[lpar + 1 : rpar]
            try:
                vals = next(csv.reader([payload], delimiter=",", quotechar="'", skipinitialspace=True))
            except Exception:
                continue

            if len(vals) != len(WDS_COLS):
                continue

            rec = {k: _clean_sql_value(v) for k, v in zip(WDS_COLS, vals)}

            acoord = rec.get("ACOORD")
            ra_deg, dec_deg = parse_acoord_to_deg(acoord if isinstance(acoord, str) else "")
            if ra_deg is None or dec_deg is None:
                continue

            rec["wds_ra"] = ra_deg
            rec["wds_dec"] = dec_deg
            rec["LSTSEP"] = parse_numeric_or_none(rec.get("LSTSEP"))
            rec["FSTMAG"] = parse_numeric_or_none(rec.get("FSTMAG"))
            rec["SECMAG"] = parse_numeric_or_none(rec.get("SECMAG"))
            rows.append(rec)

    if not rows:
        raise RuntimeError(f"No parseable WDS rows found in {sql_path}")

    wds = pd.DataFrame(rows)
    wds = wds.reset_index(drop=True)
    return wds


def crossmatch_one_file(
    in_csv: Path,
    out_dir: Path,
    wds_df: pd.DataFrame,
    radius_arcsec: float,
    ra_col: str,
    dec_col: str,
) -> Tuple[Path, Path]:
    df = pd.read_csv(in_csv, low_memory=False)

    if ra_col not in df.columns or dec_col not in df.columns:
        raise RuntimeError(f"{in_csv} missing required columns: {ra_col}, {dec_col}")

    ra = pd.to_numeric(df[ra_col], errors="coerce")
    dec = pd.to_numeric(df[dec_col], errors="coerce")
    m = np.isfinite(ra.to_numpy()) & np.isfinite(dec.to_numpy())

    out_best = df.copy()
    out_best["wds_dist_arcsec"] = np.nan
    out_best["wds_coord"] = pd.Series([None] * len(out_best), dtype="object")
    out_best["wds_disc"] = pd.Series([None] * len(out_best), dtype="object")
    out_best["wds_comp"] = pd.Series([None] * len(out_best), dtype="object")
    out_best["wds_acoord"] = pd.Series([None] * len(out_best), dtype="object")
    out_best["wds_stype"] = pd.Series([None] * len(out_best), dtype="object")
    out_best["wds_notes"] = pd.Series([None] * len(out_best), dtype="object")
    out_best["wds_lstsep"] = np.nan
    out_best["wds_fstmag"] = np.nan
    out_best["wds_secmag"] = np.nan
    out_best["wds_ra"] = np.nan
    out_best["wds_dec"] = np.nan

    if not np.any(m):
        best_path = out_dir / f"{in_csv.stem}_wds_best.csv"
        out_best.to_csv(best_path, index=False)
        all_path = out_dir / f"{in_csv.stem}_wds_all.csv"
        pd.DataFrame(columns=["input_index", "wds_dist_arcsec"]).to_csv(all_path, index=False)
        return best_path, all_path

    src_idx = np.flatnonzero(m)
    src = SkyCoord(ra=ra.iloc[src_idx].to_numpy() * u.deg, dec=dec.iloc[src_idx].to_numpy() * u.deg)
    wds = SkyCoord(ra=wds_df["wds_ra"].to_numpy() * u.deg, dec=wds_df["wds_dec"].to_numpy() * u.deg)

    # One best WDS match per source (if within radius)
    best_j, sep2d, _ = src.match_to_catalog_sky(wds)
    sep_arcsec = sep2d.arcsec
    keep = sep_arcsec <= float(radius_arcsec)

    if np.any(keep):
        use_src = src_idx[keep]
        use_wds = best_j[keep]
        out_best.loc[use_src, "wds_dist_arcsec"] = sep_arcsec[keep]
        out_best.loc[use_src, "wds_coord"] = wds_df.iloc[use_wds]["COORD"].to_numpy()
        out_best.loc[use_src, "wds_disc"] = wds_df.iloc[use_wds]["DISC"].to_numpy()
        out_best.loc[use_src, "wds_comp"] = wds_df.iloc[use_wds]["COMP"].to_numpy()
        out_best.loc[use_src, "wds_acoord"] = wds_df.iloc[use_wds]["ACOORD"].to_numpy()
        out_best.loc[use_src, "wds_stype"] = wds_df.iloc[use_wds]["STYPE"].to_numpy()
        out_best.loc[use_src, "wds_notes"] = wds_df.iloc[use_wds]["NOTES"].to_numpy()
        out_best.loc[use_src, "wds_lstsep"] = wds_df.iloc[use_wds]["LSTSEP"].to_numpy()
        out_best.loc[use_src, "wds_fstmag"] = wds_df.iloc[use_wds]["FSTMAG"].to_numpy()
        out_best.loc[use_src, "wds_secmag"] = wds_df.iloc[use_wds]["SECMAG"].to_numpy()
        out_best.loc[use_src, "wds_ra"] = wds_df.iloc[use_wds]["wds_ra"].to_numpy()
        out_best.loc[use_src, "wds_dec"] = wds_df.iloc[use_wds]["wds_dec"].to_numpy()

    best_path = out_dir / f"{in_csv.stem}_wds_best.csv"
    out_best.to_csv(best_path, index=False)

    # Optional all matches within radius (one-to-many table)
    src2, wds2, sep2, _ = search_around_sky(src, wds, seplimit=float(radius_arcsec) * u.arcsec)
    all_rows = []
    for i_src, i_wds, sep in zip(src2, wds2, sep2.arcsec):
        input_i = int(src_idx[int(i_src)])
        w = wds_df.iloc[int(i_wds)]
        all_rows.append(
            {
                "input_index": input_i,
                "source_id": df.iloc[input_i]["source_id"] if "source_id" in df.columns else np.nan,
                ra_col: float(df.iloc[input_i][ra_col]),
                dec_col: float(df.iloc[input_i][dec_col]),
                "wds_dist_arcsec": float(sep),
                "wds_coord": w.get("COORD"),
                "wds_disc": w.get("DISC"),
                "wds_comp": w.get("COMP"),
                "wds_acoord": w.get("ACOORD"),
                "wds_stype": w.get("STYPE"),
                "wds_notes": w.get("NOTES"),
                "wds_lstsep": w.get("LSTSEP"),
                "wds_fstmag": w.get("FSTMAG"),
                "wds_secmag": w.get("SECMAG"),
                "wds_ra": w.get("wds_ra"),
                "wds_dec": w.get("wds_dec"),
            }
        )

    all_df = pd.DataFrame(all_rows)
    if not all_df.empty:
        all_df = all_df.sort_values(["input_index", "wds_dist_arcsec"], ascending=[True, True]).reset_index(drop=True)

    all_path = out_dir / f"{in_csv.stem}_wds_all.csv"
    all_df.to_csv(all_path, index=False)

    return best_path, all_path


def main():
    ap = argparse.ArgumentParser(description="Crossmatch GaiaxEuclid CSV(s) to WDS using WDS SQL dump.")
    ap.add_argument("--wds_sql", default=str(BASE / "data" / "wds.sql"), help="Path to wds.sql")
    ap.add_argument(
        "--input_glob",
        default=str(BASE / "output" / "crossmatch" / "gaia_euclid" / "euclid_xmatch_gaia_*.csv"),
        help="Input CSV glob",
    )
    ap.add_argument(
        "--out_dir",
        default=str(BASE / "output" / "crossmatch" / "wds" / "wds_xmatch"),
        help="Output folder",
    )
    ap.add_argument("--radius_arcsec", type=float, default=2.0, help="Matching radius in arcsec")
    ap.add_argument("--ra_col", default="ra", help="RA column in input CSV")
    ap.add_argument("--dec_col", default="dec", help="Dec column in input CSV")
    args = ap.parse_args()

    wds_sql = Path(args.wds_sql)
    in_paths = [Path(p) for p in sorted(glob.glob(args.input_glob))] if "*" in args.input_glob else [Path(args.input_glob)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wds_sql.exists():
        raise FileNotFoundError(f"WDS SQL not found: {wds_sql}")
    if not in_paths:
        raise RuntimeError(f"No input CSV found for pattern: {args.input_glob}")

    print(f"Loading WDS from: {wds_sql}")
    wds_df = parse_wds_sql(wds_sql)
    print(f"WDS parsed rows with valid coordinates: {len(wds_df):,}")

    summary = []

    for in_csv in in_paths:
        print("\n" + "=" * 80)
        print(f"INPUT: {in_csv}")
        best_path, all_path = crossmatch_one_file(
            in_csv=in_csv,
            out_dir=out_dir,
            wds_df=wds_df,
            radius_arcsec=float(args.radius_arcsec),
            ra_col=args.ra_col,
            dec_col=args.dec_col,
        )

        bdf = pd.read_csv(best_path)
        n = len(bdf)
        n_hit = int(np.sum(pd.to_numeric(bdf["wds_dist_arcsec"], errors="coerce").notna()))
        print(f"Saved BEST: {best_path} | matched={n_hit}/{n} ({(100.0*n_hit/max(n,1)):.3f}%)")
        print(f"Saved ALL : {all_path}")

        summary.append({
            "input_csv": str(in_csv),
            "out_best_csv": str(best_path),
            "out_all_csv": str(all_path),
            "n_rows": int(n),
            "n_matched_best": int(n_hit),
            "frac_matched_best": float(n_hit / max(n, 1)),
            "radius_arcsec": float(args.radius_arcsec),
        })

    summary_df = pd.DataFrame(summary)
    summary_path = out_dir / "wds_crossmatch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\nSaved summary:", summary_path)
    print("Done.")


if __name__ == "__main__":
    main()
