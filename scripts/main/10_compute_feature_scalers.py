from pathlib import Path
import os
import numpy as np
import pandas as pd

from feature_schema import get_feature_cols, normalize_feature_set, scaler_stem

BASE = Path(__file__).resolve().parents[1]

DATASET_ROOT = BASE / "output" / "dataset_npz"
OUT_ROOT = BASE / "output" / "scalers"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PER_FIELD = False  # If True, compute scalers per field dataset; else, compute global scalers

USE_AUG_META = False  # If True, use augmentation metadata files to get feature stats
AUG_SUBDIR = "aug_root"
TRAIN_SPLIT_CODE = 0  # Split code for training set
FEATURE_SET = os.getenv("FEATURE_SET", "10D")  # "8D", "10D", or "16D" (legacy alias: "17D")

Q_LOW = 0.25
Q_HIGH = 0.75
EPS = 1e-12 # Small value to avoid division by zero

def list_field_dirs(dataset_root: Path):
    if (dataset_root / "metadata.csv").exists():
        return [dataset_root]
    return sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / "metadata.csv").exists()])

def meta_path_for_field(field_dir: Path) -> Path:
    if USE_AUG_META:
        return field_dir / AUG_SUBDIR / "augmentation_metadata.csv"
    return field_dir / "metadata.csv"

def compute_scaler_from_frames(frames: list, feature_cols: list):
    df = pd.concat([f[feature_cols] for f in frames], axis=0, ignore_index=True)

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out_min = np.zeros(len(feature_cols), dtype=float)
    out_q25 = np.zeros(len(feature_cols), dtype=float)
    out_q75 = np.zeros(len(feature_cols), dtype=float)
    out_iqr = np.zeros(len(feature_cols), dtype=float)
    out_n = np.zeros(len(feature_cols), dtype=np.int64)

    for i, c in enumerate(feature_cols):
        v = df[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        out_n[i] = int(v.size)
        if v.size == 0:
            out_min[i] = np.nan
            out_q25[i] = np.nan
            out_q75[i] = np.nan
            out_iqr[i] = np.nan
            continue
        out_min[i] = float(np.min(v))
        out_q25[i] = float(np.quantile(v, Q_LOW))
        out_q75[i] = float(np.quantile(v, Q_HIGH))
        iqr = out_q75[i] - out_q25[i]
        if not np.isfinite(iqr) or iqr < EPS:
            iqr = 1.0
        out_iqr[i] = float(iqr)
    return dict(
        y_min=out_min,
        q25=out_q25,
        q75=out_q75,
        y_iqr=out_iqr,
        n_finite=out_n,
    )

def save_scaler(out_npz: Path, out_csv: Path, feature_cols: list, scaler: dict):
    np.savez_compressed(
        out_npz,
        feature_names=np.array(feature_cols, dtype=object),
        y_min=scaler["y_min"].astype(np.float32),
        y_iqr=scaler["y_iqr"].astype(np.float32),
        q25=scaler["q25"].astype(np.float32),
        q75=scaler["q75"].astype(np.float32),
        n_finite=scaler["n_finite"].astype(np.int64),
    )
    tab = pd.DataFrame({
        "feature": feature_cols,
        "min": scaler["y_min"],
        "q25": scaler["q25"],
        "q75": scaler["q75"],
        "iqr": scaler["y_iqr"],
        "n_finite": scaler["n_finite"],
    })
    tab.to_csv(out_csv, index=False)

def main():
    if not DATASET_ROOT.exists():
        raise RuntimeError(f"Dataset root does not exist: {DATASET_ROOT}")
    field_dirs = list_field_dirs(DATASET_ROOT)
    if len(field_dirs) == 0:
        raise RuntimeError(f"No field datasets found under: {DATASET_ROOT}")
    print("Found field datasets:", len(field_dirs))
    for d in field_dirs:
        print(" -", d)
    feature_cols = get_feature_cols(FEATURE_SET)
    out_stem = scaler_stem(FEATURE_SET)
    feature_set_norm = normalize_feature_set(FEATURE_SET).lower()
    print("FEATURE_SET:", normalize_feature_set(FEATURE_SET), "| n_features:", len(feature_cols))

    frames = []
    total_rows = 0
    total_train_rows = 0
    for field_dir in field_dirs:
        tag = field_dir.name if field_dir != DATASET_ROOT else "LEGACY_OR_SINGLE"
        mp= meta_path_for_field(field_dir)
        if not mp.exists():
            print(f"Warning: Metadata file not found for field {tag}, skipping.")
            continue
        head = pd.read_csv(mp, nrows=1)
        missing = [c for c in (feature_cols + ["split_code"]) if c not in head.columns]
        if missing:
            raise RuntimeError(f"Missing feature columns in field {tag}: {missing}")
        usecols = ["split_code"] + feature_cols
        df = pd.read_csv(mp, usecols=usecols, low_memory=False)
        total_rows += len(df)
        df["split_code"] = pd.to_numeric(df["split_code"], errors= "coerce")
        df_train = df.loc[df["split_code"] == TRAIN_SPLIT_CODE, feature_cols].copy()
        total_train_rows += len(df_train)
        print(f"Field {tag}: total rows={len(df):,}, training rows={len(df_train):,}")
        if len(df_train) == 0:
            print(f"[WARN] {tag}: 0 train rows found (split_code=={TRAIN_SPLIT_CODE}), skipping.")
            continue
        frames.append(df_train)
    if len(frames) == 0:
        raise RuntimeError("No valid metadata frames found to compute scalers.")
    print("\nRows loaded (all):", total_rows)
    print("Rows loaded (train):", total_train_rows)
    scaler = compute_scaler_from_frames(frames, feature_cols)
    out_npz = OUT_ROOT / f"{out_stem}.npz"
    out_csv = OUT_ROOT / f"{out_stem}.csv"
    save_scaler(out_npz, out_csv, feature_cols, scaler)

    print("\n === Saved global scaler ===")
    print(f"NPZ: {out_npz}")
    print(f"CSV: {out_csv}")
    print("\nPreview (feature, min, iqr):")
    for f, mn, mx, iqr in zip(feature_cols, scaler["y_min"], scaler["q75"], scaler["y_iqr"]):
        print(f"  {f:>28}: min={mn: .6g} | max={mx: .6g} | iqr={iqr: .6g}")

    if PER_FIELD:
        print("\n=== Computing per-field scalers (TRAIN) ===")
        for field_dir in field_dirs:
            tag = field_dir.name if field_dir != DATASET_ROOT else "LEGACY_OR_SINGLE"
            mp= meta_path_for_field(field_dir)
            if not mp.exists():
                print(f"Warning: Metadata file not found for field {tag}, skipping.")
                continue
            usecols = ["split_code"] + feature_cols
            df = pd.read_csv(mp, usecols=usecols, low_memory=False)
            df["split_code"] = pd.to_numeric(df["split_code"], errors= "coerce")
            df_train = df.loc[df["split_code"] == TRAIN_SPLIT_CODE, feature_cols].copy()
            if len(df_train) == 0:
                print(f"[WARN] {tag}: 0 train rows found (split_code=={TRAIN_SPLIT_CODE}), skipping.")
                continue
            sc = compute_scaler_from_frames([df_train], feature_cols)

            out_npz_f = field_dir / f"y_scaler_iqr_{feature_set_norm}.npz"
            out_csv_f = field_dir / f"y_scaler_iqr_{feature_set_norm}.csv"
            save_scaler(out_npz_f, out_csv_f, feature_cols, sc)
            print(f"\n --- Saved scaler for field {tag} ---")
    print("\n DONE.")

if __name__ == "__main__":
    main()  
