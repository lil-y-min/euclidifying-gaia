"""
14_train_xgb_unified.py
======================

Robust trainer: Gaia features -> XGBoost -> PCA coeffs (K components)

Key upgrade (Step A: traceability, TEST-only)
--------------------------------------------
While writing X_test.mmap / Y_test.mmap, we also write a bulletproof mapping:

  output/ml_runs/<RUN_NAME>/trace/trace_test.csv

Each row records where the test sample came from:
- field_tag
- dataset_kind (RAW/AUG)
- npz_relpath, npz_file, index_in_file
- meta_row (row index in metadata.csv / metadata_aug.csv)
- optional stable columns if present: source_id, v_alpha_j2000, v_delta_j2000, ra, dec, dist_match

This means: no more “rebuild test ordering later” failures.

Other improvements
------------------
- load_valid_mask(): if bool mask length != metadata length, we pad/truncate with False and warn
  (this avoids crashes when metadata got appended without rebuilding valid_rows.npy).
- Plots go to: Codes/plots/ml_runs/<RUN_NAME>/  (as requested)

Outputs
-------
Codes/output/ml_runs/<RUN_NAME>/
  config.json
  manifest_arrays.npz   (now includes trace_test_csv + dataset_root)
  arrays/               (memmaps)
  models/               (one Booster JSON per PCA component)
  metrics/              (component RMSE CSV)
  trace/trace_test.csv  (NEW: bulletproof mapping)

Plots
-----
Codes/plots/ml_runs/<RUN_NAME>/
"""

from __future__ import annotations

import os
import json
import math
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from feature_schema import get_feature_cols, normalize_feature_set


# ======================================================================
# CONFIG — EDIT HERE
# ======================================================================

@dataclass
class RunConfig:
    base_dir: Path = Path(__file__).resolve().parents[1]  # .../Codes
    dataset_root: Path = None
    pca_npz: Path = None
    gaia_scaler_npz: Path = None

    feature_set: str = "8D"         # "8D", "10D", or "16D" (legacy alias: "17D")
    use_aug: bool = True
    aug_subdir: str = "aug_rot"
    aug_train_only: bool = True      # recommended

    n_components_to_train: Optional[int] = None  # None = all K
    random_seed: int = 123

    # Optional caps for debugging
    max_train: Optional[int] = None
    max_val: Optional[int] = None
    max_test: Optional[int] = None

    # XGBoost training (native API)
    num_boost_round: int = 6000
    learning_rate: float = 0.03
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    tree_method: str = "hist"

    early_stopping_rounds: int = 200
    eval_metric: str = "rmse"

    # Stamp normalization
    use_positive_only: bool = True
    eps_integral: float = 1e-12
    gaia_g_mag_min: Optional[float] = 17.0

    # Output naming
    run_name: str = ""  # leave empty to auto-generate

    # NEW: plots go to .../Codes/plots/ml_runs/<RUN_NAME>/
    plots_root: Path = None

    def __post_init__(self):
        if self.dataset_root is None:
            self.dataset_root = self.base_dir / "output" / "dataset_npz"
        if self.pca_npz is None:
            self.pca_npz = self.base_dir / "output" / "pca" / "pca_stampnorm_trainonly_k032.npz"
        if self.gaia_scaler_npz is None:
            self.gaia_scaler_npz = self.base_dir / "output" / "scalers" / "y_feature_iqr.npz"
        if self.plots_root is None:
            self.plots_root = self.base_dir / "plots" / "ml_runs"


CFG = RunConfig(
    feature_set="8D",     # "8D", "10D", or "16D"
    use_aug=True,          # True/False
    aug_train_only=True,   # keep True for your “AUG train only” choice
    run_name="ml_xgb_8d",  # set your run name
)


# ======================================================================
# File layout constants
# ======================================================================

RAW_META = "metadata.csv"
AUG_META = "metadata_aug.csv"
VALID_NAME = "valid_rows.npy"


# ======================================================================
# Helpers
# ======================================================================

def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def list_field_dirs(dataset_root: Path) -> List[Path]:
    """dataset_root contains 17 folders: ERO-Abell2390, ... Taurus."""
    if (dataset_root / RAW_META).exists():
        return [dataset_root]
    return sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / RAW_META).exists()])


def load_valid_mask(valid_path: Path, nrows: int) -> np.ndarray:
    """
    Supports:
      (A) bool mask of length nrows
      (B) integer array of valid indices

    Robustness upgrade:
    - if bool mask length mismatches metadata length, we pad/truncate with False and WARN.
      This avoids crashes when metadata.csv was appended but valid_rows.npy wasn't regenerated.
    """
    arr = np.load(valid_path, allow_pickle=True)

    if arr.dtype == np.bool_:
        if arr.shape[0] == nrows:
            return arr
        # pad/truncate with False
        m = np.zeros(nrows, dtype=bool)
        n_copy = min(nrows, arr.shape[0])
        m[:n_copy] = arr[:n_copy]
        print(f"[WARN] valid_rows length mismatch in {valid_path}")
        print(f"       valid_rows has {arr.shape[0]} but metadata has {nrows}")
        if arr.shape[0] < nrows:
            print("       -> padded missing rows with False")
        else:
            print("       -> truncated extra rows")
        return m

    if np.issubdtype(arr.dtype, np.integer):
        idx = arr.astype(np.int64, copy=False)
        m = np.zeros(nrows, dtype=bool)
        if idx.size:
            if idx.min() < 0 or idx.max() >= nrows:
                raise RuntimeError(f"valid index out of range in {valid_path}")
            m[idx] = True
        return m

    raise RuntimeError(f"Unsupported valid_rows dtype: {arr.dtype} in {valid_path}")


def load_pca(pca_npz: Path) -> Tuple[np.ndarray, np.ndarray, int, int, int, Optional[np.ndarray]]:
    d = np.load(pca_npz, allow_pickle=True)
    mean = d["mean"].astype(np.float32)              # (D,)
    components = d["components"].astype(np.float32)  # (K,D)
    K = int(d["n_components"]) if "n_components" in d else int(components.shape[0])
    D = int(d["n_features"]) if "n_features" in d else int(mean.shape[0])
    stamp_pix = int(d["stamp_pix"]) if "stamp_pix" in d else int(round(math.sqrt(D)))
    evr = d["explained_variance_ratio"].astype(np.float32) if "explained_variance_ratio" in d else None
    return mean, components, K, D, stamp_pix, evr


def load_gaia_scaler(npz_path: Path, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    y_min = d["y_min"].astype(np.float32)
    y_iqr = d["y_iqr"].astype(np.float32)

    name_to_idx = {n: i for i, n in enumerate(names)}
    missing = [c for c in feature_cols if c not in name_to_idx]
    if missing:
        raise RuntimeError(f"Scaler missing columns: {missing}")

    idxs = np.array([name_to_idx[c] for c in feature_cols], dtype=int)
    return y_min[idxs], y_iqr[idxs]


def positive_integral(stamps: np.ndarray, use_positive_only: bool) -> np.ndarray:
    if use_positive_only:
        s = np.clip(stamps, 0.0, np.inf)
        return np.sum(s, axis=(1, 2))
    return np.sum(stamps, axis=(1, 2))


def normalize_by_integral(stamps: np.ndarray, use_positive_only: bool, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    stamps = stamps.astype(np.float32, copy=False)
    integ = positive_integral(stamps, use_positive_only=use_positive_only).astype(np.float32)
    ok = np.isfinite(integ) & (integ > eps)
    out = stamps.copy()
    if np.any(ok):
        out[ok] /= integ[ok, None, None]
    return out, ok


def pca_transform(flat_X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (flat_X - mean[None, :]) @ components.T


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

def count_csv_rows_fast(path: Path) -> int:
    """Counts data rows in a CSV (excluding header) without pandas."""
    n = 0
    with open(path, "rb") as f:
        # skip header
        f.readline()
        for _ in f:
            n += 1
    return n

def pick_gaia_g_col(columns: List[str]) -> Optional[str]:
    for c in ("phot_g_mean_mag", "feat_phot_g_mean_mag"):
        if c in columns:
            return c
    return None


def count_rows(
    meta_path: Path,
    valid_path: Path,
    feature_cols: List[str],
    allow_splits: Set[int],
    gaia_g_mag_min: Optional[float],
) -> Dict[int, int]:
    need = ["npz_file", "index_in_file", "split_code"] + feature_cols
    head = pd.read_csv(meta_path, nrows=0, engine="python")
    g_col = pick_gaia_g_col(head.columns.tolist())
    if g_col is not None and g_col not in need:
        need.append(g_col)

    # Get nrows without pandas (avoids OOM on gigantic CSV)
    nrows = count_csv_rows_fast(meta_path)
    valid = load_valid_mask(valid_path, nrows=nrows)

    counts = {s: 0 for s in allow_splits}

    # Stream CSV in chunks
    chunksize = 200_000  # tune if needed
    offset = 0

    for chunk in pd.read_csv(
         meta_path,
         usecols=need,
         chunksize=chunksize,
         engine="python",      # robust
         on_bad_lines="skip",  # survive malformed lines
     ):
        # Slice valid mask for this chunk
        v = valid[offset: offset + len(chunk)]
        offset += len(chunk)

        chunk["split_code"] = pd.to_numeric(chunk["split_code"], errors="coerce")
        chunk["index_in_file"] = pd.to_numeric(chunk["index_in_file"], errors="coerce")
        for c in feature_cols:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        if g_col is not None:
            chunk[g_col] = pd.to_numeric(chunk[g_col], errors="coerce")

        ok = (
            v
            & chunk["npz_file"].notna().to_numpy()
            & chunk["index_in_file"].notna().to_numpy()
            & np.all(np.isfinite(chunk[feature_cols].to_numpy(dtype=float)), axis=1)
        )
        if (gaia_g_mag_min is not None) and (g_col is not None):
            g = chunk[g_col].to_numpy(dtype=float)
            ok &= np.isfinite(g) & (g >= float(gaia_g_mag_min))

        split = chunk["split_code"].to_numpy()
        for s in allow_splits:
            counts[s] += int(np.sum(ok & (split == s)))

        # Early exit if we consumed all valid mask rows
        if offset >= nrows:
            break

    return counts




def open_memmaps(array_dir: Path, n_train: int, n_val: int, n_test: int, n_feat: int, K: int) -> Dict[str, np.memmap]:
    array_dir.mkdir(parents=True, exist_ok=True)
    return dict(
        X_train=np.memmap(array_dir / "X_train.mmap", dtype="float32", mode="w+", shape=(n_train, n_feat)),
        Y_train=np.memmap(array_dir / "Y_train.mmap", dtype="float32", mode="w+", shape=(n_train, K)),
        X_val=np.memmap(array_dir / "X_val.mmap", dtype="float32", mode="w+", shape=(n_val, n_feat)),
        Y_val=np.memmap(array_dir / "Y_val.mmap", dtype="float32", mode="w+", shape=(n_val, K)),
        X_test=np.memmap(array_dir / "X_test.mmap", dtype="float32", mode="w+", shape=(n_test, n_feat)),
        Y_test=np.memmap(array_dir / "Y_test.mmap", dtype="float32", mode="w+", shape=(n_test, K)),
    )


def _safe_col(df: pd.DataFrame, name: str) -> Optional[np.ndarray]:
    """Return df[name] as numpy array if it exists, else None."""
    if name not in df.columns:
        return None
    return df[name].to_numpy()


def fill_arrays_for_meta(
    *,
    dataset_root: Path,
    field_tag: str,
    dataset_kind: str,          # "RAW" or "AUG"
    in_dir: Path,
    meta_path: Path,
    valid_path: Path,
    feature_cols: List[str],
    gaia_min: np.ndarray,
    gaia_iqr: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    D: int,
    mmaps: Dict[str, np.memmap],
    ptrs: Dict[str, int],
    limits: Dict[str, int],
    allow_splits: Set[int],
    cfg: RunConfig,
    label: str,
    trace_test_writer: Optional[csv.DictWriter],
):
    print("\n" + "-" * 90)
    print("DATASET:", label)
    print("IN_DIR :", in_dir)
    print("META   :", meta_path)
    print("VALID  :", valid_path)

    base_need = ["npz_file", "index_in_file", "split_code"] + feature_cols
    extra_maybe = ["source_id", "v_alpha_j2000", "v_delta_j2000", "ra", "dec", "dist_match"]

    # Detect which optional columns exist without loading the whole file
    head = pd.read_csv(meta_path, nrows=0, engine="python")
    g_col = pick_gaia_g_col(head.columns.tolist())
    extra_have = [c for c in extra_maybe if c in head.columns]
    need = base_need + extra_have
    if g_col is not None and g_col not in need:
        need.append(g_col)

    # Get nrows without pandas full read + load valid
    nrows = count_csv_rows_fast(meta_path)
    valid = load_valid_mask(valid_path, nrows=nrows)

    chunksize = 200_000
    offset = 0

    # Small cache to avoid reloading the same NPZ repeatedly inside a chunk loop
    last_npz_path = None
    last_npz_X = None

    for chunk in pd.read_csv(
        meta_path,
        usecols=need,
        chunksize=chunksize,
        engine="python",
        on_bad_lines="skip",
    ):
        if (ptrs["train"] >= limits["train"]) and (ptrs["val"] >= limits["val"]) and (ptrs["test"] >= limits["test"]):
            return

        # meta_row = absolute row index in metadata file
        chunk = chunk.copy()
        chunk["meta_row"] = np.arange(offset, offset + len(chunk), dtype=np.int64)

        v = valid[offset: offset + len(chunk)]
        offset += len(chunk)

        chunk["split_code"] = pd.to_numeric(chunk["split_code"], errors="coerce")
        chunk["index_in_file"] = pd.to_numeric(chunk["index_in_file"], errors="coerce")
        for c in feature_cols:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        if g_col is not None:
            chunk[g_col] = pd.to_numeric(chunk[g_col], errors="coerce")

        m = (
            v
            & chunk["npz_file"].notna().to_numpy()
            & chunk["index_in_file"].notna().to_numpy()
            & chunk["split_code"].isin(list(allow_splits)).to_numpy()
            & np.all(np.isfinite(chunk[feature_cols].to_numpy(dtype=float)), axis=1)
        )
        if (cfg.gaia_g_mag_min is not None) and (g_col is not None):
            g = chunk[g_col].to_numpy(dtype=float)
            m &= np.isfinite(g) & (g >= float(cfg.gaia_g_mag_min))
        chunk = chunk.loc[m]
        if len(chunk) == 0:
            if offset >= nrows:
                break
            continue

        # Group within this chunk by npz_file
        for npz_name, sub in chunk.groupby("npz_file", sort=False):
            if (ptrs["train"] >= limits["train"]) and (ptrs["val"] >= limits["val"]) and (ptrs["test"] >= limits["test"]):
                return

            npz_path = in_dir / str(npz_name)
            if not npz_path.exists():
                continue

            # Load or reuse cached NPZ
            if last_npz_path != npz_path:
                dnpz = np.load(npz_path)
                last_npz_X = dnpz["X"]
                last_npz_path = npz_path

            Xall = last_npz_X

            idxs = sub["index_in_file"].to_numpy(dtype=int)
            okb = (idxs >= 0) & (idxs < Xall.shape[0])
            sub = sub.iloc[okb]
            idxs = idxs[okb]
            if idxs.size == 0:
                continue

            stamps = Xall[idxs].astype(np.float32, copy=False)
            stamps_n, ok = normalize_by_integral(stamps, cfg.use_positive_only, cfg.eps_integral)
            sub = sub.iloc[ok]
            stamps_n = stamps_n[ok]
            if stamps_n.shape[0] == 0:
                continue

            flat = stamps_n.reshape(stamps_n.shape[0], -1)
            if flat.shape[1] != D:
                raise RuntimeError(f"Flattened dim mismatch: got {flat.shape[1]} expected {D}")

            coeffs = pca_transform(flat, mean, components).astype(np.float32)

            feats = sub[feature_cols].to_numpy(dtype=np.float32)
            feats = (feats - gaia_min[None, :]) / gaia_iqr[None, :]
            split_codes = sub["split_code"].to_numpy(dtype=int)

            meta_row = sub["meta_row"].to_numpy(dtype=np.int64)
            idx_in_file = sub["index_in_file"].to_numpy(dtype=np.int64)

            # Optional trace columns
            sid = sub["source_id"].to_numpy(dtype=object) if "source_id" in sub.columns else None
            eu_ra = sub["v_alpha_j2000"].to_numpy(dtype=object) if "v_alpha_j2000" in sub.columns else None
            eu_de = sub["v_delta_j2000"].to_numpy(dtype=object) if "v_delta_j2000" in sub.columns else None
            ga_ra = sub["ra"].to_numpy(dtype=object) if "ra" in sub.columns else None
            ga_de = sub["dec"].to_numpy(dtype=object) if "dec" in sub.columns else None
            dmat = sub["dist_match"].to_numpy(dtype=object) if "dist_match" in sub.columns else None

            for split_code, split_name in [(0, "train"), (1, "val"), (2, "test")]:
                if split_code not in allow_splits:
                    continue
                ms = (split_codes == split_code)
                if not np.any(ms):
                    continue

                Xs = feats[ms]
                Ys = coeffs[ms]

                room = limits[split_name] - ptrs[split_name]
                if room <= 0:
                    continue

                n_take = min(Xs.shape[0], room)
                Xs = Xs[:n_take]
                Ys = Ys[:n_take]

                p0 = ptrs[split_name]
                mmaps[f"X_{split_name}"][p0:p0 + n_take] = Xs
                mmaps[f"Y_{split_name}"][p0:p0 + n_take] = Ys
                ptrs[split_name] += n_take

                # TEST-only trace writing
                if (split_name == "test") and (trace_test_writer is not None) and (n_take > 0):
                    rel_dir = os.path.relpath(in_dir, dataset_root).replace("\\", "/")
                    npz_relpath = f"{rel_dir}/{str(npz_name)}"

                    mr = meta_row[ms][:n_take]
                    iif = idx_in_file[ms][:n_take]

                    sid_t = sid[ms][:n_take] if sid is not None else None
                    eu_ra_t = eu_ra[ms][:n_take] if eu_ra is not None else None
                    eu_de_t = eu_de[ms][:n_take] if eu_de is not None else None
                    ga_ra_t = ga_ra[ms][:n_take] if ga_ra is not None else None
                    ga_de_t = ga_de[ms][:n_take] if ga_de is not None else None
                    dmat_t = dmat[ms][:n_take] if dmat is not None else None

                    for i in range(n_take):
                        row = {
                            "test_index": int(p0 + i),
                            "field_tag": field_tag,
                            "dataset_kind": dataset_kind,
                            "npz_relpath": npz_relpath,
                            "npz_file": str(npz_name),
                            "index_in_file": int(iif[i]),
                            "meta_row": int(mr[i]),
                        }
                        if sid_t is not None: row["source_id"] = sid_t[i]
                        if eu_ra_t is not None: row["v_alpha_j2000"] = eu_ra_t[i]
                        if eu_de_t is not None: row["v_delta_j2000"] = eu_de_t[i]
                        if ga_ra_t is not None: row["ra"] = ga_ra_t[i]
                        if ga_de_t is not None: row["dec"] = ga_de_t[i]
                        if dmat_t is not None: row["dist_match"] = dmat_t[i]
                        trace_test_writer.writerow(row)

        if offset >= nrows:
            break



def plot_explained_vs_rmse(outpath: Path, evr: np.ndarray, rmse_val: np.ndarray, title: str):
    plt.figure(figsize=(7.5, 5.2))
    plt.scatter(evr, rmse_val, s=45)
    plt.xlabel("PCA explained variance ratio")
    plt.ylabel("Component RMSE (VAL)")
    plt.title(title)
    savefig(outpath)




# ======================================================================
# MAIN
# ======================================================================

def main():
    np.random.seed(CFG.random_seed)

    feature_cols = get_feature_cols(CFG.feature_set)
    feature_set_norm = normalize_feature_set(CFG.feature_set)

    default_scaler_10d = CFG.base_dir / "output" / "scalers" / "y_feature_iqr.npz"
    if feature_set_norm == "16D" and CFG.gaia_scaler_npz == default_scaler_10d:
        CFG.gaia_scaler_npz = CFG.base_dir / "output" / "scalers" / "y_feature_iqr_16d.npz"

    # Auto run naming
    if not CFG.run_name.strip():
        aug_tag = "AUGtrainonly" if (CFG.use_aug and CFG.aug_train_only) else ("AUG" if CFG.use_aug else "NOAUG")
        CFG.run_name = f"xgb_{CFG.pca_npz.stem}_{feature_set_norm}_{aug_tag}"

    out_root = CFG.base_dir / "output" / "ml_runs" / CFG.run_name
    array_dir = out_root / "arrays"
    model_dir = out_root / "models"
    metrics_dir = out_root / "metrics"
    trace_dir = out_root / "trace"

    # Plots go to .../Codes/plots/...
    plot_dir = CFG.plots_root / CFG.run_name

    for d in [out_root, array_dir, model_dir, metrics_dir, trace_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save config.json
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(CFG).items()}, f, indent=2)

    # Load PCA + scaler
    mean, components, K, D, stamp_pix, evr = load_pca(CFG.pca_npz)
    gaia_min, gaia_iqr = load_gaia_scaler(CFG.gaia_scaler_npz, feature_cols)

    print("\n=== RUN ===")
    print("RUN_NAME:", CFG.run_name)
    print("OUT_ROOT:", out_root)
    print("PLOTS  :", plot_dir)

    print("\n=== PCA ===")
    print("PCA:", CFG.pca_npz)
    print(f"stamp_pix={stamp_pix}  D={D}  K={K}")

    print("\n=== FEATURES ===")
    print("feature_set:", feature_set_norm)
    print("n_features:", len(feature_cols))
    print(feature_cols)
    print("gaia_g_mag_min:", CFG.gaia_g_mag_min)

    print("\n=== AUG ===")
    print("use_aug:", CFG.use_aug, "| aug_train_only:", CFG.aug_train_only)

    # Field dirs
    field_dirs = list_field_dirs(CFG.dataset_root)
    if len(field_dirs) == 0:
        raise RuntimeError(f"No datasets found under {CFG.dataset_root}")

    print("\nFound field datasets:", len(field_dirs))
    for p in field_dirs:
        print(" -", p)

    # Count rows
    allow_raw = {0, 1, 2}
    allow_aug = {0} if CFG.aug_train_only else {0, 1, 2}

    total_train = total_val = total_test = 0
    for field_dir in field_dirs:
        c = count_rows(field_dir / RAW_META, field_dir / VALID_NAME, feature_cols, allow_raw, CFG.gaia_g_mag_min)
        total_train += c.get(0, 0)
        total_val += c.get(1, 0)
        total_test += c.get(2, 0)

        if CFG.use_aug:
            aug_dir = field_dir / CFG.aug_subdir
            if aug_dir.exists():
                c2 = count_rows(aug_dir / AUG_META, aug_dir / VALID_NAME, feature_cols, allow_aug, CFG.gaia_g_mag_min)
                total_train += c2.get(0, 0)
                if not CFG.aug_train_only:
                    total_val += c2.get(1, 0)
                    total_test += c2.get(2, 0)

    n_train = total_train if CFG.max_train is None else min(total_train, CFG.max_train)
    n_val = total_val if CFG.max_val is None else min(total_val, CFG.max_val)
    n_test = total_test if CFG.max_test is None else min(total_test, CFG.max_test)

    print("\n=== ARRAY SIZES ===")
    print("train:", n_train, f"(raw+aug={total_train})")
    print("val  :", n_val, f"(raw+aug={total_val})")
    print("test :", n_test, f"(raw+aug={total_test})")

    # Allocate memmaps
    mmaps = open_memmaps(array_dir, n_train, n_val, n_test, n_feat=len(feature_cols), K=K)
    ptrs = {"train": 0, "val": 0, "test": 0}
    limits = {"train": n_train, "val": n_val, "test": n_test}

    # Prepare TEST trace writer (Step A)
    trace_test_csv = trace_dir / "trace_test.csv"
    trace_fields = [
        "test_index",
        "field_tag",
        "dataset_kind",
        "npz_relpath",
        "npz_file",
        "index_in_file",
        "meta_row",
        # optional (written only if present)
        "source_id",
        "v_alpha_j2000",
        "v_delta_j2000",
        "ra",
        "dec",
        "dist_match",
    ]
    trace_fh = open(trace_test_csv, "w", newline="", encoding="utf-8")
    trace_writer = csv.DictWriter(trace_fh, fieldnames=trace_fields, extrasaction="ignore")
    trace_writer.writeheader()

    # Fill memmaps + trace
    try:
        for field_dir in field_dirs:
            tag = field_dir.name

            # RAW
            fill_arrays_for_meta(
                dataset_root=CFG.dataset_root,
                field_tag=tag,
                dataset_kind="RAW",
                in_dir=field_dir,
                meta_path=field_dir / RAW_META,
                valid_path=field_dir / VALID_NAME,
                feature_cols=feature_cols,
                gaia_min=gaia_min,
                gaia_iqr=gaia_iqr,
                mean=mean,
                components=components,
                D=D,
                mmaps=mmaps,
                ptrs=ptrs,
                limits=limits,
                allow_splits={0, 1, 2},
                cfg=CFG,
                label=f"{tag} (RAW)",
                trace_test_writer=trace_writer,
            )

            # AUG (usually train-only; still supported)
            if CFG.use_aug:
                aug_dir = field_dir / CFG.aug_subdir
                if aug_dir.exists():
                    fill_arrays_for_meta(
                        dataset_root=CFG.dataset_root,
                        field_tag=tag,
                        dataset_kind="AUG",
                        in_dir=aug_dir,
                        meta_path=aug_dir / AUG_META,
                        valid_path=aug_dir / VALID_NAME,
                        feature_cols=feature_cols,
                        gaia_min=gaia_min,
                        gaia_iqr=gaia_iqr,
                        mean=mean,
                        components=components,
                        D=D,
                        mmaps=mmaps,
                        ptrs=ptrs,
                        limits=limits,
                        allow_splits={0} if CFG.aug_train_only else {0, 1, 2},
                        cfg=CFG,
                        label=f"{tag} (AUG)",
                        trace_test_writer=trace_writer,
                    )

            if (ptrs["train"] >= n_train) and (ptrs["val"] >= n_val) and (ptrs["test"] >= n_test):
                break
    finally:
        # Close trace file even if something errors
        trace_fh.close()

    # Flush memmaps
    for arr in mmaps.values():
        arr.flush()

    # Save manifest (now includes trace + dataset root)
    manifest_path = out_root / "manifest_arrays.npz"
    np.savez_compressed(
        manifest_path,
        n_train=np.int64(ptrs["train"]),
        n_val=np.int64(ptrs["val"]),
        n_test=np.int64(ptrs["test"]),
        n_features=np.int64(len(feature_cols)),
        feature_cols=np.array(feature_cols, dtype=object),
        K=np.int64(K),
        D=np.int64(D),
        stamp_pix=np.int64(stamp_pix),
        X_train_path=str(array_dir / "X_train.mmap"),
        Y_train_path=str(array_dir / "Y_train.mmap"),
        X_val_path=str(array_dir / "X_val.mmap"),
        Y_val_path=str(array_dir / "Y_val.mmap"),
        X_test_path=str(array_dir / "X_test.mmap"),
        Y_test_path=str(array_dir / "Y_test.mmap"),
        pca_npz=str(CFG.pca_npz),
        scaler_npz=str(CFG.gaia_scaler_npz),
        dataset_root=str(CFG.dataset_root),
        trace_test_csv=str(trace_test_csv),
        plots_dir=str(plot_dir),
    )

    print("\nSaved manifest:", manifest_path)
    print("Saved TEST trace:", trace_test_csv)
    print("Arrays written:")
    print(f" train {ptrs['train']}/{n_train}")
    print(f" val   {ptrs['val']}/{n_val}")
    print(f" test  {ptrs['test']}/{n_test}")

    # Open memmaps read-only for training
    n_train_eff = int(ptrs["train"])
    n_val_eff = int(ptrs["val"])
    n_feat = len(feature_cols)

    X_train = np.memmap(array_dir / "X_train.mmap", dtype="float32", mode="r", shape=(n_train_eff, n_feat))
    Y_train = np.memmap(array_dir / "Y_train.mmap", dtype="float32", mode="r", shape=(n_train_eff, K))
    X_val = np.memmap(array_dir / "X_val.mmap", dtype="float32", mode="r", shape=(n_val_eff, n_feat))
    Y_val = np.memmap(array_dir / "Y_val.mmap", dtype="float32", mode="r", shape=(n_val_eff, K))

    # Train boosters (native API)
    K_train = K if (CFG.n_components_to_train is None) else int(min(CFG.n_components_to_train, K))
    print(f"\n=== TRAINING (native xgb.train) components: {K_train}/{K} ===")

    params = {
        "objective": "reg:squarederror",
        "eta": CFG.learning_rate,
        "max_depth": CFG.max_depth,
        "subsample": CFG.subsample,
        "colsample_bytree": CFG.colsample_bytree,
        "lambda": CFG.reg_lambda,
        "tree_method": CFG.tree_method,
        "eval_metric": CFG.eval_metric,
        "seed": CFG.random_seed,
    }

    metrics_rows = []
    rmse_val = np.full(K_train, np.nan, dtype=float)

    for j in range(K_train):
        ytr = Y_train[:, j]
        yva = Y_val[:, j]

        dtrain = xgb.DMatrix(X_train, label=ytr)
        dval = xgb.DMatrix(X_val, label=yva)

        evals_result = {}
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=CFG.num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=CFG.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=False,
        )

        pred_val = booster.predict(dval)
        r = rmse(yva, pred_val)
        rmse_val[j] = r

        model_path = model_dir / f"booster_comp_{j:03d}.json"
        booster.save_model(str(model_path))

        best_iter = int(booster.best_iteration) if hasattr(booster, "best_iteration") else -1
        metrics_rows.append({"component": j, "rmse_val": r, "best_iteration": best_iter})

        if (j == 0) or ((j + 1) % 4 == 0) or (j == K_train - 1):
            print(f"  comp {j:03d} | RMSE(val)={r:.6f} | best_iter={best_iter}")

    # Save metrics
    metrics_csv = metrics_dir / "metrics_components.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)
    print("\nSaved metrics:", metrics_csv)

    # Plot RMSE per component
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(K_train), rmse_val, marker="o", linewidth=1)
    plt.xlabel("PCA component index")
    plt.ylabel("RMSE (VAL)")
    plt.title(f"Validation RMSE per PCA component\n{CFG.run_name}")
    savefig(plot_dir / "rmse_per_component_val.png")

    # Explained variance vs RMSE if available
    if evr is not None:
        plot_explained_vs_rmse(
            plot_dir / "explained_variance_vs_rmse_val.png",
            evr[:K_train],
            rmse_val,
            title=f"Explained variance vs component RMSE (VAL)\n{CFG.run_name}",
        )
    else:
        print("[INFO] PCA file has no explained_variance_ratio; skipping that plot.")

    print("\nDONE.")
    print("Run folder :", out_root)
    print("Plots folder:", plot_dir)
    print("Trace file  :", trace_test_csv)


if __name__ == "__main__":
    main()
