"""
13_fit_pca.py

Fit a GLOBAL PCA model on Euclid postage stamps, using TRAIN split only.

- Uses BOTH raw and augmented datasets (raw + aug_rot) by default.
- Uses valid_rows.npy to drop stamps with bad integral / bad indexing.
  IMPORTANT: valid_rows.npy can be either:
    (A) boolean mask of length len(metadata)
    (B) integer array of valid row indices (length = number of valid rows)
  This script supports BOTH formats.

- Normalizes each stamp by its POSITIVE integral:
    Xp = clip(X, 0, inf)
    integral = sum(Xp)
    Xnorm = X / integral

- Fits PCA with IncrementalPCA (streaming batches) to keep memory low.

Outputs:
- output/pca/pca_stampnorm_trainonly_k{K}.npz
- plots/pca/explained_variance_k{K}.png

Run:
  python -u scripts/13_fit_pca.py
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
BASE = Path(__file__).resolve().parents[1]
DATASET_ROOT = BASE / "output" / "dataset_npz"

USE_AUG = True
AUG_SUBDIR = "aug_rot"
RAW_META_NAME = "metadata.csv"
AUG_META_NAME = "metadata_aug.csv"

TRAIN_SPLIT_CODE = 0  # 0=train, 1=val, 2=test

# PCA settings
N_COMPONENTS = 32
BATCH_STAMPS = 4096
RANDOM_SEED = 123
MAX_TRAIN_SAMPLES_PER_DATASET = None
# If too slow with aug, set e.g. 200000

# Normalization
USE_POSITIVE_ONLY = True
EPS_INTEGRAL = 1e-12

# Optional restrict tags
ONLY_TAGS = None  # or set like {"ERO-NGC6397", "ERO-Barnard30"}

# Outputs
OUT_DIR = BASE / "output" / "pca"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = BASE / "plots" / "pca"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MAKE_PLOT = True

OUT_NPZ = OUT_DIR / f"pca_stampnorm_trainonly_k{N_COMPONENTS:03d}.npz"
OUT_PNG = PLOT_DIR / f"explained_variance_k{N_COMPONENTS:03d}.png"

# sklearn
try:
    from sklearn.decomposition import IncrementalPCA
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required.\nInstall with:\n  pip install -U scikit-learn\n"
        f"Import error: {e}"
    )

# =========================
# Helpers
# =========================
def list_field_dirs(dataset_root: Path):
    if (dataset_root / RAW_META_NAME).exists():
        return [dataset_root]
    dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / RAW_META_NAME).exists()])
    if ONLY_TAGS is not None:
        dirs = [d for d in dirs if d.name in ONLY_TAGS]
    return dirs


def infer_stamp_pix_any(npz_dir: Path):
    files = sorted(glob.glob(str(npz_dir / "stamps_*.npz")))
    if not files:
        return None
    d = np.load(files[0])
    X = d["X"]
    if X.ndim != 3 or X.shape[1] != X.shape[2]:
        return None
    return int(X.shape[1])


def load_valid_mask(valid_path: Path, nrows: int) -> np.ndarray:
    """
    Support two formats:

    (A) bool mask of length nrows
    (B) int array of valid row indices (length = n_valid)

    Returns bool mask of length nrows.
    """
    if not valid_path.exists():
        raise RuntimeError(f"valid_rows.npy not found: {valid_path}")

    arr = np.load(valid_path)

    # Case A: boolean mask
    if arr.dtype == np.bool_:
        if arr.shape[0] == nrows:
            return arr
        # If bool but wrong length, that's ambiguous => error with a helpful message
        raise RuntimeError(
            f"valid_rows.npy is bool but has length {arr.shape[0]} != metadata rows {nrows}.\n"
            f"File: {valid_path}\n"
            "If this file is intended to be indices, it must be saved as integer array.\n"
            "Best fix: regenerate valid_rows.npy as either full-length bool mask or int indices."
        )

    # Case B: indices
    if np.issubdtype(arr.dtype, np.integer):
        idx = arr.astype(np.int64, copy=False)
        if idx.size == 0:
            return np.zeros(nrows, dtype=bool)

        if np.min(idx) < 0 or np.max(idx) >= nrows:
            raise RuntimeError(
                f"valid index array has out-of-range values for nrows={nrows}.\n"
                f"min={int(np.min(idx))}, max={int(np.max(idx))}\n"
                f"File: {valid_path}"
            )

        mask = np.zeros(nrows, dtype=bool)
        mask[idx] = True
        return mask

    raise RuntimeError(
        f"valid_rows.npy has unsupported dtype {arr.dtype}. Expected bool mask or int indices.\n"
        f"File: {valid_path}"
    )


def positive_integral_per_stamp(X: np.ndarray):
    if USE_POSITIVE_ONLY:
        Xp = np.clip(X, 0.0, np.inf)
        return np.sum(Xp, axis=(1, 2))
    return np.sum(X, axis=(1, 2))


def normalize_stamps_by_integral(X: np.ndarray):
    X = X.astype(np.float32, copy=False)
    integ = positive_integral_per_stamp(X).astype(np.float32)
    ok = np.isfinite(integ) & (integ > EPS_INTEGRAL)

    Xn = X.copy()
    if np.any(ok):
        Xn[ok] /= integ[ok, None, None]
    return Xn, ok


def iter_train_rows(meta_path: Path, valid_path: Path):
    """
    Returns DataFrame with npz_file/index_in_file for rows that are:
    - split_code == TRAIN_SPLIT_CODE
    - valid_rows == True
    """
    head = pd.read_csv(meta_path, nrows=1)
    need = ["npz_file", "index_in_file", "split_code"]
    missing = [c for c in need if c not in head.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {meta_path}: {missing}")

    df = pd.read_csv(meta_path, usecols=need, low_memory=False)
    nrows = len(df)
    valid = load_valid_mask(valid_path, nrows=nrows)

    df["split_code"] = pd.to_numeric(df["split_code"], errors="coerce")
    df["index_in_file"] = pd.to_numeric(df["index_in_file"], errors="coerce")

    m_train = (df["split_code"].to_numpy(dtype=float) == float(TRAIN_SPLIT_CODE))
    m_ok = valid & m_train

    sub = df.loc[m_ok, ["npz_file", "index_in_file"]].copy()
    sub = sub.dropna(subset=["npz_file", "index_in_file"])
    sub["index_in_file"] = sub["index_in_file"].astype(int)

    return sub


def maybe_subsample_rows(rows_df: pd.DataFrame, max_n: int, seed: int):
    if (max_n is None) or (len(rows_df) <= max_n):
        return rows_df
    return rows_df.sample(n=max_n, random_state=seed).reset_index(drop=True)


def stream_batches_from_rows(in_dir: Path, rows_df: pd.DataFrame, batch_stamps: int):
    groups = rows_df.groupby("npz_file", sort=False)
    buf = []
    stamp_pix_seen = None

    for npz_name, sub in groups:
        npz_path = in_dir / str(npz_name)
        if not npz_path.exists():
            continue

        d = np.load(npz_path)
        X = d["X"]
        if X.ndim != 3 or X.shape[1] != X.shape[2]:
            continue

        if stamp_pix_seen is None:
            stamp_pix_seen = int(X.shape[1])

        idxs = sub["index_in_file"].to_numpy(dtype=int)
        idxs = idxs[(idxs >= 0) & (idxs < X.shape[0])]
        if idxs.size == 0:
            continue

        stamps = X[idxs].astype(np.float32, copy=False)
        stamps_norm, ok = normalize_stamps_by_integral(stamps)
        stamps_norm = stamps_norm[ok]
        if stamps_norm.shape[0] == 0:
            continue

        flat = stamps_norm.reshape(stamps_norm.shape[0], -1)

        for i in range(flat.shape[0]):
            buf.append(flat[i])
            if len(buf) >= batch_stamps:
                out = np.stack(buf, axis=0).astype(np.float32, copy=False)
                yield out, stamp_pix_seen
                buf.clear()

    if len(buf) > 0:
        out = np.stack(buf, axis=0).astype(np.float32, copy=False)
        yield out, stamp_pix_seen


def plot_explained_variance(pca, out_png: Path):
    evr = np.array(pca.explained_variance_ratio_, dtype=float)
    cum = np.cumsum(evr)

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(evr) + 1), cum, marker="o")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("PCA explained variance (train-only, stamp-normalized)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================
# Main
# =========================
def main():
    np.random.seed(RANDOM_SEED)

    field_dirs = list_field_dirs(DATASET_ROOT)
    if len(field_dirs) == 0:
        raise RuntimeError(f"No field datasets found under: {DATASET_ROOT}")

    print("Found field datasets:", len(field_dirs))
    for d in field_dirs:
        print(" -", d)

    # infer stamp size
    stamp_pix_guess = None
    for fd in field_dirs:
        sp = infer_stamp_pix_any(fd)
        if sp is not None:
            stamp_pix_guess = sp
            break
        if USE_AUG and (fd / AUG_SUBDIR).exists():
            sp = infer_stamp_pix_any(fd / AUG_SUBDIR)
            if sp is not None:
                stamp_pix_guess = sp
                break

    if stamp_pix_guess is None:
        raise RuntimeError("Could not infer stamp_pix from any stamps_*.npz files.")
    n_features = stamp_pix_guess * stamp_pix_guess

    print("\n=== PCA configuration ===")
    print("N_COMPONENTS:", N_COMPONENTS)
    print("BATCH_STAMPS:", BATCH_STAMPS)
    print("USE_AUG:", USE_AUG, "| AUG_SUBDIR:", AUG_SUBDIR)
    print("TRAIN_SPLIT_CODE:", TRAIN_SPLIT_CODE)
    print("stamp_pix:", stamp_pix_guess, "| flattened dim:", n_features)
    print("MAX_TRAIN_SAMPLES_PER_DATASET:", MAX_TRAIN_SAMPLES_PER_DATASET)

    pca = IncrementalPCA(n_components=N_COMPONENTS)

    total_seen = 0
    stamp_pix_seen_global = None

    def process_dataset(in_dir: Path, meta_path: Path, valid_path: Path, label: str):
        nonlocal total_seen, stamp_pix_seen_global

        if not meta_path.exists():
            print(f"[SKIP] {label}: missing metadata: {meta_path}")
            return
        if not valid_path.exists():
            print(f"[SKIP] {label}: missing valid_rows.npy: {valid_path}")
            return

        print("\n" + "-" * 90)
        print("DATASET:", label)
        print("IN_DIR :", in_dir)
        print("META   :", meta_path)
        print("VALID  :", valid_path)

        rows = iter_train_rows(meta_path, valid_path)
        n_train_valid = len(rows)
        print("Train+valid rows:", n_train_valid)

        if n_train_valid == 0:
            print("[WARN] No train+valid rows. Skipping.")
            return

        rows = maybe_subsample_rows(rows, MAX_TRAIN_SAMPLES_PER_DATASET, seed=RANDOM_SEED)
        if len(rows) != n_train_valid:
            print(f"Subsampled to {len(rows)} rows (from {n_train_valid}).")

        batch_count = 0
        for batch, spix in stream_batches_from_rows(in_dir, rows, batch_stamps=BATCH_STAMPS):
            if spix is not None:
                stamp_pix_seen_global = spix
            if batch.shape[1] != n_features:
                raise RuntimeError(
                    f"Flattened dimension mismatch: got {batch.shape[1]}, expected {n_features}."
                )

            pca.partial_fit(batch)
            total_seen += batch.shape[0]
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"  partial_fit batches={batch_count:4d} | total_seen={total_seen}")

        print(f"[DONE] {label}: batches={batch_count} | total_seen={total_seen}")

    # run across fields
    for field_dir in field_dirs:
        tag = field_dir.name if field_dir != DATASET_ROOT else "LEGACY_OR_SINGLE"

        # RAW
        process_dataset(
            in_dir=field_dir,
            meta_path=field_dir / RAW_META_NAME,
            valid_path=field_dir / "valid_rows.npy",
            label=f"{tag} (RAW)"
        )

        # AUG
        if USE_AUG:
            aug_in = field_dir / AUG_SUBDIR
            if aug_in.exists():
                process_dataset(
                    in_dir=aug_in,
                    meta_path=aug_in / AUG_META_NAME,
                    valid_path=aug_in / "valid_rows.npy",
                    label=f"{tag} (AUG)"
                )
            else:
                print(f"[SKIP] {tag} (AUG): folder not found: {aug_in}")

    if total_seen == 0:
        raise RuntimeError("No training stamps were seen by PCA. Check valid_rows and split_code filtering.")

    # save PCA params
    print("\n=== Saving PCA ===")
    np.savez_compressed(
        OUT_NPZ,
        stamp_pix=np.int64(stamp_pix_seen_global if stamp_pix_seen_global is not None else stamp_pix_guess),
        n_features=np.int64(n_features),
        n_components=np.int64(N_COMPONENTS),
        total_seen=np.int64(total_seen),
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        singular_values=pca.singular_values_.astype(np.float32),
    )

    print("Saved:", OUT_NPZ)
    print("mean shape:", pca.mean_.shape)
    print("components shape:", pca.components_.shape)
    print("explained variance ratio (first 5):", pca.explained_variance_ratio_[:5])

    if MAKE_PLOT:
        plot_explained_variance(pca, OUT_PNG)
        print("Saved:", OUT_PNG)

    print("\nDONE.")


if __name__ == "__main__":
    main()
