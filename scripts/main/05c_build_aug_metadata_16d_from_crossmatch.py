"""
Build aug_rot/metadata_aug_16d.csv per field by enriching metadata_aug.csv with
extended Gaia columns from crossmatch CSVs, keyed by source_id.

Non-overwriting: original metadata_aug.csv is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from feature_schema import GAIA_COLS_OPTIONAL_EXTENDED, Y_FEATURE_COLS_16D


BASE = Path(__file__).resolve().parents[1]
DATASET_ROOT = BASE / "output" / "dataset_npz"
XMATCH_DIR = BASE / "output" / "crossmatch" / "gaia_euclid"
AUG_SUBDIR = "aug_rot"
OUT_NAME = "metadata_aug_16d.csv"


def list_field_dirs(dataset_root: Path) -> list[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def as_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def main() -> None:
    fields = list_field_dirs(DATASET_ROOT)
    if not fields:
        raise RuntimeError(f"No field dirs under {DATASET_ROOT}")

    print(f"Found {len(fields)} field directories.")
    for fdir in fields:
        tag = fdir.name
        aug_dir = fdir / AUG_SUBDIR
        meta_in = aug_dir / "metadata_aug.csv"
        meta_out = aug_dir / OUT_NAME
        xm_path = XMATCH_DIR / f"euclid_xmatch_gaia_{tag}.csv"
        if not meta_in.exists():
            print(f"[SKIP] {tag}: no {meta_in}")
            continue
        if not xm_path.exists():
            print(f"[SKIP] {tag}: missing crossmatch file {xm_path.name}")
            continue

        aug = pd.read_csv(meta_in, low_memory=False)
        xm_usecols = ["source_id", *GAIA_COLS_OPTIONAL_EXTENDED]
        xm = pd.read_csv(xm_path, usecols=lambda c: c in set(xm_usecols + ["dist_match"]), low_memory=False)

        if "source_id" not in aug.columns:
            print(f"[SKIP] {tag}: metadata_aug.csv missing source_id")
            continue

        as_numeric(aug, ["source_id"])
        as_numeric(xm, ["source_id", "dist_match", *GAIA_COLS_OPTIONAL_EXTENDED])
        aug = aug.dropna(subset=["source_id"]).copy()
        xm = xm.dropna(subset=["source_id"]).copy()
        aug["source_id"] = aug["source_id"].astype(np.int64)
        xm["source_id"] = xm["source_id"].astype(np.int64)
        if "dist_match" in xm.columns:
            xm = xm.sort_values("dist_match", ascending=True, na_position="last")
        xm = xm.drop_duplicates("source_id", keep="first")

        xm = xm[["source_id", *[c for c in GAIA_COLS_OPTIONAL_EXTENDED if c in xm.columns]]].copy()
        merged = aug.merge(xm, on="source_id", how="left", suffixes=("", "_xmatch"))

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
