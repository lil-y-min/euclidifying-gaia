"""
Build metadata_16d.csv per field by enriching existing metadata.csv with
extended Gaia columns from crossmatch CSVs, keyed by source_id.

This is a non-overwriting bridge when FITS inputs are unavailable for a full
dataset re-export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from feature_schema import GAIA_COLS_OPTIONAL_EXTENDED, Y_FEATURE_COLS_16D


BASE = Path(__file__).resolve().parents[1]
DATASET_ROOT = BASE / "output" / "dataset_npz"
XMATCH_DIR = BASE / "output" / "crossmatch" / "gaia_euclid"
OUT_NAME = "metadata_16d.csv"


def list_field_dirs(dataset_root: Path) -> list[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / "metadata.csv").exists()])


def dedup_xmatch_by_source(xm: pd.DataFrame) -> pd.DataFrame:
    if "source_id" not in xm.columns:
        raise RuntimeError("Crossmatch table missing source_id")
    if "dist_match" in xm.columns:
        xm["dist_match"] = pd.to_numeric(xm["dist_match"], errors="coerce")
        xm = xm.sort_values("dist_match", ascending=True, na_position="last")
    return xm.drop_duplicates("source_id", keep="first")


def as_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def main() -> None:
    fields = list_field_dirs(DATASET_ROOT)
    if not fields:
        raise RuntimeError(f"No field dirs with metadata.csv under {DATASET_ROOT}")

    print(f"Found {len(fields)} field directories.")
    for fdir in fields:
        tag = fdir.name
        meta_in = fdir / "metadata.csv"
        meta_out = fdir / OUT_NAME
        xm_path = XMATCH_DIR / f"euclid_xmatch_gaia_{tag}.csv"
        if not xm_path.exists():
            print(f"[SKIP] {tag}: missing crossmatch file {xm_path.name}")
            continue

        meta = pd.read_csv(meta_in, low_memory=False)
        xm_usecols = ["source_id", *GAIA_COLS_OPTIONAL_EXTENDED]
        xm = pd.read_csv(xm_path, usecols=lambda c: c in set(xm_usecols + ["dist_match"]), low_memory=False)

        if "source_id" not in meta.columns:
            print(f"[SKIP] {tag}: metadata.csv missing source_id")
            continue

        as_numeric(meta, ["source_id"])
        as_numeric(xm, ["source_id", "dist_match", *GAIA_COLS_OPTIONAL_EXTENDED])
        meta = meta.dropna(subset=["source_id"]).copy()
        xm = xm.dropna(subset=["source_id"]).copy()
        meta["source_id"] = meta["source_id"].astype(np.int64)
        xm["source_id"] = xm["source_id"].astype(np.int64)

        xm = dedup_xmatch_by_source(xm)
        xm = xm[["source_id", *[c for c in GAIA_COLS_OPTIONAL_EXTENDED if c in xm.columns]]].copy()
        merged = meta.merge(xm, on="source_id", how="left", suffixes=("", "_xmatch"))

        # Build 16D feature columns in canonical order.
        feat_map = {
            "feat_astrometric_excess_noise_sig": "astrometric_excess_noise_sig",
            "feat_ipd_gof_harmonic_amplitude": "ipd_gof_harmonic_amplitude",
            "feat_ipd_gof_harmonic_phase": "ipd_gof_harmonic_phase",
            "feat_ipd_frac_odd_win": "ipd_frac_odd_win",
            "feat_phot_bp_n_contaminated_transits": "phot_bp_n_contaminated_transits",
            "feat_phot_bp_n_blended_transits": "phot_bp_n_blended_transits",
            "feat_phot_rp_n_contaminated_transits": "phot_rp_n_contaminated_transits",
            "feat_phot_rp_n_blended_transits": "phot_rp_n_blended_transits",
        }
        for feat_col, raw_col in feat_map.items():
            merged[feat_col] = pd.to_numeric(merged.get(raw_col), errors="coerce")

        missing_feat_cols = [c for c in Y_FEATURE_COLS_16D if c not in merged.columns]
        if missing_feat_cols:
            raise RuntimeError(f"{tag}: missing required feature columns after merge: {missing_feat_cols}")

        merged.to_csv(meta_out, index=False)
        frac = {}
        for c in feat_map:
            v = pd.to_numeric(merged[c], errors="coerce")
            frac[c] = float(np.isfinite(v).mean()) if len(v) else float("nan")
        print(f"[WRITE] {tag}: {meta_out.name} rows={len(merged):,} | finite(frac)={frac}")

    print("\nDone.")


if __name__ == "__main__":
    main()
