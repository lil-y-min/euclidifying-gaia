"""
12_build_valid_row_manifests.py

Build "valid row" manifests for each field dataset (raw + aug_rot).

Goal:
- Decide which samples are usable based on image integral:
    integral = sum(clip(X, 0, +inf))
  A sample is valid if integral is finite and > EPS_INTEGRAL.

We DO NOT rewrite NPZ files (no massive regeneration).
We only write small manifest files listing valid metadata row indices.

Outputs (per field):
- output/dataset_npz/<TAG>/valid_rows.npy              (for metadata.csv)
- output/dataset_npz/<TAG>/aug_rot/valid_rows.npy      (for metadata_aug.csv)

Run:
  python -u scripts/12_build_valid_row_manifests.py
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
BASE = Path(__file__).resolve().parents[1]  # .../Codes
DATASET_ROOT = BASE / "output" / "dataset_npz"

# process both raw and augmentation
PROCESS_RAW = True
PROCESS_AUG = True

AUG_SUBDIR = "aug_rot"
RAW_META_NAME = "metadata.csv"
AUG_META_NAME = "metadata_aug.csv"

# Validity rule (your choice "A"):
# integral = sum(clip(X, 0, +inf))
USE_POSITIVE_ONLY = True
EPS_INTEGRAL = 1e-12

# Optional: process only specific fields (set to None to process all)
ONLY_TAGS = None
# Example:
# ONLY_TAGS = {"ERO-IC342", "ERO-Messier78"}

# Safety: if metadata references a missing npz file, skip those rows (invalid)
ALLOW_MISSING_NPZ = True


# =========================
# Helpers
# =========================
def list_field_dirs(dataset_root: Path):
    """
    New layout:
      output/dataset_npz/<tag>/metadata.csv
    (also supports legacy single-folder layout: output/dataset_npz/metadata.csv)
    """
    if (dataset_root / RAW_META_NAME).exists():
        return [dataset_root]

    out = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and (p / RAW_META_NAME).exists():
            out.append(p)
    return out


def load_npz_X(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    if "X" not in d:
        raise RuntimeError(f"NPZ missing 'X': {npz_path}")
    X = d["X"]
    if X.ndim != 3:
        raise RuntimeError(f"Unexpected X shape in {npz_path}: {X.shape}")
    return X


def compute_integrals(X: np.ndarray) -> np.ndarray:
    """
    integral per stamp:
      sum(clip(X,0,inf)) if USE_POSITIVE_ONLY else sum(X)
    """
    Xf = X.astype(np.float64, copy=False)
    if USE_POSITIVE_ONLY:
        Xf = np.clip(Xf, 0.0, np.inf)
    integ = np.sum(Xf, axis=(1, 2))
    return integ


def build_valid_rows_for_dataset(in_dir: Path, meta_path: Path, out_valid_rows: Path):
    """
    Reads metadata (npz_file, index_in_file), loads NPZ stamps in groups,
    and marks which metadata rows point to stamps whose integral is valid.
    Saves valid row indices into out_valid_rows (.npy).
    """
    if not meta_path.exists():
        print(f"[SKIP] metadata not found: {meta_path}")
        return

    # Read only what we need (keep it fast + explicit)
    usecols = ["npz_file", "index_in_file"]
    head = pd.read_csv(meta_path, nrows=1)
    missing = [c for c in usecols if c not in head.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {meta_path}: {missing}")

    meta = pd.read_csv(meta_path, usecols=usecols, low_memory=False)

    # Clean types
    meta["npz_file"] = meta["npz_file"].astype(str)
    meta["index_in_file"] = pd.to_numeric(meta["index_in_file"], errors="coerce")

    n_meta = len(meta)
    valid_mask = np.zeros(n_meta, dtype=bool)

    # Counters for clean reporting
    n_bad_index = 0
    n_missing_npz = 0
    n_bad_integral = 0
    n_loaded_total = 0

    # Group by NPZ to avoid reloading same file repeatedly
    for npz_name, sub in meta.groupby("npz_file", sort=False):
        npz_path = in_dir / npz_name

        if not npz_path.exists():
            n_missing_npz += len(sub)
            if not ALLOW_MISSING_NPZ:
                raise RuntimeError(f"NPZ referenced by metadata not found: {npz_path}")
            continue

        X = load_npz_X(npz_path)
        n_loaded_total += X.shape[0]

        integrals = compute_integrals(X)
        ok_stamp = np.isfinite(integrals) & (integrals > EPS_INTEGRAL)

        # Metadata rows that refer to this NPZ:
        meta_rows = sub.index.to_numpy(dtype=np.int64)
        idxs = sub["index_in_file"].to_numpy(dtype=float)

        # Validate indices are finite integers in bounds
        idx_ok = np.isfinite(idxs)
        idx_int = np.zeros_like(idxs, dtype=np.int64)
        idx_int[idx_ok] = idxs[idx_ok].astype(np.int64)

        in_bounds = idx_ok & (idx_int >= 0) & (idx_int < X.shape[0])

        # For in-bounds rows, validity = ok_stamp[index_in_file]
        good_rows = meta_rows[in_bounds]
        good_idxs = idx_int[in_bounds]

        # Mark valid
        valid_mask[good_rows] = ok_stamp[good_idxs]

        # Count problems
        n_bad_index += int(np.sum(~in_bounds))
        n_bad_integral += int(np.sum(in_bounds & (~ok_stamp[good_idxs])))

    valid_rows = np.flatnonzero(valid_mask).astype(np.int64)

    # Save manifest
    out_valid_rows.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_valid_rows, valid_rows)

    # Print summary (explicit)
    print("\n--- Validity manifest summary ---")
    print("IN_DIR   :", in_dir)
    print("META     :", meta_path)
    print("OUT      :", out_valid_rows)
    print(f"Rows(meta): {n_meta}")
    print(f"Valid     : {len(valid_rows)}")
    print(f"Dropped   : {n_meta - len(valid_rows)}")
    print("Breakdown of drops:")
    print(f"  - missing npz references : {n_missing_npz}")
    print(f"  - bad index_in_file      : {n_bad_index}")
    print(f"  - bad integral (<=eps or nonfinite): {n_bad_integral}")
    print(f"Loaded total stamps (sum of X.shape[0] over loaded files): {n_loaded_total}")

    return


# =========================
# Main
# =========================
def main():
    if not DATASET_ROOT.exists():
        raise RuntimeError(f"DATASET_ROOT not found: {DATASET_ROOT}")

    field_dirs = list_field_dirs(DATASET_ROOT)
    if ONLY_TAGS is not None:
        field_dirs = [d for d in field_dirs if d.name in ONLY_TAGS]

    if len(field_dirs) == 0:
        raise RuntimeError(f"No field datasets found under: {DATASET_ROOT}")

    print("Found field datasets:", len(field_dirs))
    for d in field_dirs:
        print(" -", d)

    for field_dir in field_dirs:
        tag = field_dir.name if field_dir != DATASET_ROOT else "LEGACY_OR_SINGLE"

        print("\n" + "=" * 90)
        print("FIELD:", tag)
        print("=" * 90)

        # RAW
        if PROCESS_RAW:
            raw_dir = field_dir
            raw_meta = raw_dir / RAW_META_NAME
            raw_out = raw_dir / "valid_rows.npy"
            build_valid_rows_for_dataset(raw_dir, raw_meta, raw_out)

        # AUG
        if PROCESS_AUG:
            aug_dir = field_dir / AUG_SUBDIR
            aug_meta = aug_dir / AUG_META_NAME
            aug_out = aug_dir / "valid_rows.npy"
            build_valid_rows_for_dataset(aug_dir, aug_meta, aug_out)

    print("\nDONE.")


if __name__ == "__main__":
    main()
