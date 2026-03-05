"""
14_bis_train_xgb_pixels_flux.py
===============================

Train XGBoost directly on pixels (no PCA), with Path 2:
- Model A: predict normalized stamp pixels (shape) -> D outputs (one booster per pixel)
- Model B: predict log10(positive integral) (flux scale) -> 1 booster

Absolute flux reconstruction later:
  I_pred = S_pred * (10 ** logF_pred)

Outputs:
Codes/output/ml_runs/<RUN_NAME>/
  arrays/
    X_train.mmap
    Yshape_train.mmap
    Yflux_train.mmap
    X_val.mmap
    Yshape_val.mmap
    Yflux_val.mmap
    X_test.mmap
    Yshape_test.mmap
    Yflux_test.mmap
  models/
    booster_pix_0000.json ... booster_pix_{D-1}.json
    booster_flux.json
  trace/trace_test.csv
  manifest_arrays.npz

Plots:
Codes/plots/ml_runs/<RUN_NAME>/
"""

from __future__ import annotations

import argparse
import os
import json
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from feature_schema import get_feature_cols, normalize_feature_set


# ======================================================================
# CONFIG
# ======================================================================

@dataclass
class RunConfig:
    base_dir: Path = Path(__file__).resolve().parents[2]  # .../Codes
    dataset_root: Path = None
    gaia_scaler_npz: Path = None

    feature_set: str = "8D"        # "8D", "10D", or "16D" (legacy alias: "17D")
    use_aug: bool = True
    aug_subdir: str = "aug_rot"
    aug_train_only: bool = True

    random_seed: int = 123

    # Optional caps for debugging
    max_train: Optional[int] = None
    max_val: Optional[int] = None
    max_test: Optional[int] = None

    # XGBoost training (native API)
    num_boost_round: int = 2000
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    tree_method: str = "hist"

    early_stopping_rounds: int = 100
    eval_metric: str = "rmse"

    # Stamp normalization
    use_positive_only: bool = True
    eps_integral: float = 1e-12
    gaia_g_mag_min: Optional[float] = 17.0
    # Optional positional/data-quality filters (applied before split assignment)
    border_quantile: Optional[float] = None
    border_x_col: str = "x_pix_round"
    border_y_col: str = "y_pix_round"
    max_dist_match: Optional[float] = None
    # Optional Euclid quality-bit exclusion (from prebuilt quality_flag CSV)
    quality_flags_csv: str = ""
    quality_id_col: str = "source_id"
    quality_flag_col: str = "quality_flag"
    quality_bits_mask: int = 0
    quality_drop_unknown_source_id: bool = False
    # Optional stamp artifact filter for cut/seam-like stamps
    drop_seam_cuts: bool = False
    seam_low_threshold: float = 0.0
    seam_line_frac_min: float = 0.90
    seam_diag_len_min: int = 8

    run_name: str = "ml_xgb_pixelsflux_8d"
    plots_root: Path = None

    def __post_init__(self):
        if self.dataset_root is None:
            self.dataset_root = self.base_dir / "output" / "dataset_npz"
        if self.gaia_scaler_npz is None:
            self.gaia_scaler_npz = self.base_dir / "output" / "scalers" / "y_feature_iqr.npz"
        if self.plots_root is None:
            self.plots_root = self.base_dir / "plots" / "ml_runs"


CFG = RunConfig(
    feature_set="8D",
    use_aug=True,
    aug_train_only=True,
    run_name="ml_xgb_pixelsflux_8d_augtrain_g17",
)


# ======================================================================
# Features
# ======================================================================

# ======================================================================
# File layout constants
# ======================================================================

RAW_META = "metadata.csv"
AUG_META = "metadata_aug.csv"
RAW_META_16D = "metadata_16d.csv"
AUG_META_16D = "metadata_aug_16d.csv"
VALID_NAME = "valid_rows.npy"


# ======================================================================
# Helpers
# ======================================================================

def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def list_field_dirs(dataset_root: Path, raw_meta_name: str = RAW_META) -> List[Path]:
    if (dataset_root / raw_meta_name).exists():
        return [dataset_root]
    return sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / raw_meta_name).exists()])


def count_csv_rows_fast(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        f.readline()
        for _ in f:
            n += 1
    return n


def load_valid_mask(valid_path: Path, nrows: int) -> np.ndarray:
    arr = np.load(valid_path, allow_pickle=True)

    if arr.dtype == np.bool_:
        if arr.shape[0] == nrows:
            return arr
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
            ok = (idx >= 0) & (idx < nrows)
            if not np.all(ok):
                bad = int((~ok).sum())
                print(f"[WARN] {valid_path}: dropped {bad} out-of-range valid indices")
            idx = idx[ok]
            if idx.size:
                m[idx] = True
        return m

    raise RuntimeError(f"Unsupported valid_rows dtype: {arr.dtype} in {valid_path}")


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


def normalize_by_integral(stamps: np.ndarray, use_positive_only: bool, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    stamps = stamps.astype(np.float32, copy=False)
    integ = positive_integral(stamps, use_positive_only=use_positive_only).astype(np.float32)
    ok = np.isfinite(integ) & (integ > eps)
    out = stamps.copy()
    if np.any(ok):
        out[ok] /= integ[ok, None, None]
    return out, ok, integ


def seam_cut_mask(
    stamps: np.ndarray,
    low_threshold: float,
    line_frac_min: float,
    diag_len_min: int,
) -> np.ndarray:
    """
    Detect seam/cut-like artifacts directly from raw stamps.
    A stamp is flagged if a near-zero line spans most of any row/col/diagonal.
    """
    if stamps.ndim != 3:
        raise RuntimeError(f"Expected stamps (N,H,W), got {stamps.shape}")
    N, H, W = stamps.shape
    if H != W:
        # Current pipeline expects square stamps; keep safe fallback.
        return np.zeros(N, dtype=bool)
    low = stamps <= float(low_threshold)
    # Horizontal / vertical line cuts.
    row_line = np.max(np.mean(low, axis=2), axis=1) >= float(line_frac_min)
    col_line = np.max(np.mean(low, axis=1), axis=1) >= float(line_frac_min)
    out = row_line | col_line
    # Diagonal cuts (both directions), requiring a minimum segment length.
    n = H
    diag_flag = np.zeros(N, dtype=bool)
    min_len = max(2, int(diag_len_min))
    for i in range(N):
        if out[i]:
            continue
        im = low[i]
        m = 0.0
        for k in range(-(n - 1), n):
            d1 = np.diag(im, k=k)
            if d1.size >= min_len:
                m = max(m, float(np.mean(d1)))
            d2 = np.diag(np.fliplr(im), k=k)
            if d2.size >= min_len:
                m = max(m, float(np.mean(d2)))
            if m >= float(line_frac_min):
                diag_flag[i] = True
                break
    return out | diag_flag


def infer_stamp_pix(dataset_root: Path) -> int:
    # Find first npz in first field to infer stamp size
    fields = list_field_dirs(dataset_root, raw_meta_name=RAW_META)
    for fdir in fields:
        for npz in sorted(fdir.glob("*.npz")):
            with np.load(npz) as d:
                X = d["X"]
                if X.ndim != 3:
                    continue
                return int(X.shape[1])
    raise RuntimeError("Could not infer stamp_pix (no suitable NPZ found).")


def pick_gaia_g_col(columns: List[str]) -> Optional[str]:
    for c in ("phot_g_mean_mag", "feat_phot_g_mean_mag"):
        if c in columns:
            return c
    return None


def get_border_bounds(
    meta_path: Path,
    x_col: str,
    y_col: str,
    q: float,
) -> Optional[Tuple[float, float, float, float]]:
    if (not np.isfinite(q)) or (q <= 0.0) or (q >= 0.5):
        return None
    try:
        xy = pd.read_csv(meta_path, usecols=[x_col, y_col], engine="python")
    except Exception:
        return None
    if (x_col not in xy.columns) or (y_col not in xy.columns):
        return None
    x = pd.to_numeric(xy[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(xy[y_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return None
    x = x[ok]
    y = y[ok]
    x_lo = float(np.quantile(x, q))
    x_hi = float(np.quantile(x, 1.0 - q))
    y_lo = float(np.quantile(y, q))
    y_hi = float(np.quantile(y, 1.0 - q))
    return (x_lo, x_hi, y_lo, y_hi)


def load_psf_source_ids(labels_csv: Path, label_keep: int) -> np.ndarray:
    df = pd.read_csv(labels_csv, usecols=["source_id", "label"])
    df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["source_id", "label"]).copy()
    df["source_id"] = df["source_id"].astype(np.int64)
    df["label"] = df["label"].astype(np.int64)
    keep = df.loc[df["label"] == int(label_keep), "source_id"].drop_duplicates().to_numpy(dtype=np.int64)
    if keep.size == 0:
        raise RuntimeError(f"No source_id found in {labels_csv} for label={label_keep}")
    return keep


def load_weight_source_ids(
    labels_csv: Path,
    label_target: int,
    min_confidence: float,
) -> np.ndarray:
    cols = ["source_id", "label"]
    use_conf = np.isfinite(min_confidence)
    if use_conf:
        cols.append("confidence")
    df = pd.read_csv(labels_csv, usecols=cols)
    df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    if use_conf:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["source_id", "label"]).copy()
    df["source_id"] = df["source_id"].astype(np.int64)
    df["label"] = df["label"].astype(np.int64)
    m = (df["label"] == int(label_target))
    if use_conf:
        m &= np.isfinite(df["confidence"]) & (df["confidence"] >= float(min_confidence))
    keep = df.loc[m, "source_id"].drop_duplicates().to_numpy(dtype=np.int64)
    if keep.size == 0:
        conf_txt = "none" if (not use_conf) else f">={float(min_confidence):.3f}"
        raise RuntimeError(
            f"No source_id found in {labels_csv} for label={label_target} with confidence {conf_txt}"
        )
    return keep


def load_bad_quality_source_ids(
    quality_csv: Path,
    id_col: str,
    flag_col: str,
    bits_mask: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if int(bits_mask) <= 0:
        raise RuntimeError("bits_mask must be > 0 for quality filtering.")

    df = pd.read_csv(quality_csv, usecols=[id_col, flag_col])
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df[flag_col] = pd.to_numeric(df[flag_col], errors="coerce")
    df = df.dropna(subset=[id_col, flag_col]).copy()
    if len(df) == 0:
        return np.array([], dtype=np.int64), {"rows_in_csv": 0, "rows_valid": 0, "rows_flagged": 0, "unique_source_ids_flagged": 0}

    sid = df[id_col].to_numpy(dtype=np.int64)
    qf = df[flag_col].to_numpy(dtype=np.int64)
    bad = (qf & int(bits_mask)) != 0
    bad_ids = np.unique(sid[bad]).astype(np.int64)
    stats = {
        "rows_in_csv": int(len(df)),
        "rows_valid": int(len(df)),
        "rows_flagged": int(np.sum(bad)),
        "unique_source_ids_flagged": int(bad_ids.size),
    }
    return bad_ids, stats


def count_rows(
    meta_path: Path,
    valid_path: Path,
    feature_cols: List[str],
    allow_splits: Set[int],
    gaia_g_mag_min: Optional[float],
    border_bounds: Optional[Tuple[float, float, float, float]] = None,
    border_x_col: Optional[str] = None,
    border_y_col: Optional[str] = None,
    max_dist_match: Optional[float] = None,
    source_ids_keep: Optional[np.ndarray] = None,
    source_ids_bad_quality: Optional[np.ndarray] = None,
    quality_drop_unknown_source_id: bool = False,
) -> Dict[int, int]:
    head = pd.read_csv(meta_path, nrows=0, engine="python")
    g_col = pick_gaia_g_col(head.columns.tolist())
    need_source_id = (source_ids_keep is not None) or (source_ids_bad_quality is not None)
    if need_source_id and ("source_id" not in head.columns):
        raise RuntimeError(f"source_id-based filter requested but source_id missing in {meta_path}")

    need = ["npz_file", "index_in_file", "split_code"] + feature_cols
    if need_source_id:
        need.append("source_id")
    if g_col is not None and g_col not in need:
        need.append(g_col)
    has_border = (
        (border_bounds is not None)
        and (border_x_col is not None)
        and (border_y_col is not None)
        and (border_x_col in head.columns)
        and (border_y_col in head.columns)
    )
    if has_border:
        if border_x_col not in need:
            need.append(border_x_col)
        if border_y_col not in need:
            need.append(border_y_col)
    has_dist_cut = (max_dist_match is not None) and np.isfinite(float(max_dist_match)) and ("dist_match" in head.columns)
    if has_dist_cut and ("dist_match" not in need):
        need.append("dist_match")
    nrows = count_csv_rows_fast(meta_path)
    valid = load_valid_mask(valid_path, nrows=nrows)

    counts = {s: 0 for s in allow_splits}
    chunksize = 200_000
    offset = 0

    for chunk in pd.read_csv(meta_path, usecols=need, chunksize=chunksize, engine="python", on_bad_lines="skip"):
        v = valid[offset: offset + len(chunk)]
        offset += len(chunk)

        chunk["split_code"] = pd.to_numeric(chunk["split_code"], errors="coerce")
        chunk["index_in_file"] = pd.to_numeric(chunk["index_in_file"], errors="coerce")
        for c in feature_cols:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        if g_col is not None:
            chunk[g_col] = pd.to_numeric(chunk[g_col], errors="coerce")
        if need_source_id:
            chunk["source_id"] = pd.to_numeric(chunk["source_id"], errors="coerce")

        ok = (
            v
            & chunk["npz_file"].notna().to_numpy()
            & chunk["index_in_file"].notna().to_numpy()
            & np.all(np.isfinite(chunk[feature_cols].to_numpy(dtype=float)), axis=1)
        )
        if source_ids_keep is not None:
            sid = chunk["source_id"].to_numpy(dtype=float)
            sid_ok = np.isfinite(sid)
            sid_keep = np.zeros(len(chunk), dtype=bool)
            if np.any(sid_ok):
                sid_keep[sid_ok] = np.isin(sid[sid_ok].astype(np.int64), source_ids_keep, assume_unique=False)
            ok &= sid_keep
        if source_ids_bad_quality is not None:
            sid = chunk["source_id"].to_numpy(dtype=float)
            sid_ok = np.isfinite(sid)
            sid_bad = np.zeros(len(chunk), dtype=bool)
            if np.any(sid_ok):
                sid_bad[sid_ok] = np.isin(sid[sid_ok].astype(np.int64), source_ids_bad_quality, assume_unique=False)
            if bool(quality_drop_unknown_source_id):
                ok &= sid_ok & (~sid_bad)
            else:
                ok &= (~sid_bad)
        if (gaia_g_mag_min is not None) and (g_col is not None):
            g = chunk[g_col].to_numpy(dtype=float)
            ok &= np.isfinite(g) & (g >= float(gaia_g_mag_min))
        if has_border:
            x_lo, x_hi, y_lo, y_hi = border_bounds
            x = pd.to_numeric(chunk[border_x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(chunk[border_y_col], errors="coerce").to_numpy(dtype=float)
            ok &= np.isfinite(x) & np.isfinite(y) & (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
        if has_dist_cut:
            d = pd.to_numeric(chunk["dist_match"], errors="coerce").to_numpy(dtype=float)
            ok &= np.isfinite(d) & (d <= float(max_dist_match))

        split = chunk["split_code"].to_numpy()
        for s in allow_splits:
            counts[s] += int(np.sum(ok & (split == s)))

        if offset >= nrows:
            break

    return counts


def open_memmaps(array_dir: Path, n_train: int, n_val: int, n_test: int, n_feat: int, D: int) -> Dict[str, np.memmap]:
    array_dir.mkdir(parents=True, exist_ok=True)
    return dict(
        X_train=np.memmap(array_dir / "X_train.mmap", dtype="float32", mode="w+", shape=(n_train, n_feat)),
        Yshape_train=np.memmap(array_dir / "Yshape_train.mmap", dtype="float32", mode="w+", shape=(n_train, D)),
        Yflux_train=np.memmap(array_dir / "Yflux_train.mmap", dtype="float32", mode="w+", shape=(n_train,)),
        X_val=np.memmap(array_dir / "X_val.mmap", dtype="float32", mode="w+", shape=(n_val, n_feat)),
        Yshape_val=np.memmap(array_dir / "Yshape_val.mmap", dtype="float32", mode="w+", shape=(n_val, D)),
        Yflux_val=np.memmap(array_dir / "Yflux_val.mmap", dtype="float32", mode="w+", shape=(n_val,)),
        W_train=np.memmap(array_dir / "W_train.mmap", dtype="float32", mode="w+", shape=(n_train,)),
        W_val=np.memmap(array_dir / "W_val.mmap", dtype="float32", mode="w+", shape=(n_val,)),
        X_test=np.memmap(array_dir / "X_test.mmap", dtype="float32", mode="w+", shape=(n_test, n_feat)),
        Yshape_test=np.memmap(array_dir / "Yshape_test.mmap", dtype="float32", mode="w+", shape=(n_test, D)),
        Yflux_test=np.memmap(array_dir / "Yflux_test.mmap", dtype="float32", mode="w+", shape=(n_test,)),
    )


def fill_arrays_for_meta_pixelsflux(
    *,
    dataset_root: Path,
    field_tag: str,
    dataset_kind: str,      # "RAW" or "AUG"
    in_dir: Path,
    meta_path: Path,
    valid_path: Path,
    feature_cols: List[str],
    gaia_min: np.ndarray,
    gaia_iqr: np.ndarray,
    stamp_pix: int,
    mmaps: Dict[str, np.memmap],
    ptrs: Dict[str, int],
    limits: Dict[str, int],
    allow_splits: Set[int],
    cfg: RunConfig,
    label: str,
    trace_test_writer: Optional[csv.DictWriter],
    source_ids_keep: Optional[np.ndarray] = None,
    source_ids_weighted: Optional[np.ndarray] = None,
    source_ids_bad_quality: Optional[np.ndarray] = None,
    nonpsf_weight: float = 1.0,
):
    print("\n" + "-" * 90)
    print("DATASET:", label)
    print("IN_DIR :", in_dir)
    print("META   :", meta_path)
    print("VALID  :", valid_path)

    base_need = ["npz_file", "index_in_file", "split_code"] + feature_cols
    extra_maybe = ["source_id", "v_alpha_j2000", "v_delta_j2000", "ra", "dec", "dist_match"]

    head = pd.read_csv(meta_path, nrows=0, engine="python")
    g_col = pick_gaia_g_col(head.columns.tolist())
    need_source_id = (source_ids_keep is not None) or (source_ids_weighted is not None) or (source_ids_bad_quality is not None)
    if need_source_id and ("source_id" not in head.columns):
        raise RuntimeError(f"PSF filter requested but source_id missing in {meta_path}")
    extra_have = [c for c in extra_maybe if c in head.columns]
    need = base_need + extra_have
    if g_col is not None and g_col not in need:
        need.append(g_col)
    border_bounds = get_border_bounds(
        meta_path,
        x_col=cfg.border_x_col,
        y_col=cfg.border_y_col,
        q=float(cfg.border_quantile) if (cfg.border_quantile is not None) else np.nan,
    )
    has_border = (
        (border_bounds is not None)
        and (cfg.border_x_col in head.columns)
        and (cfg.border_y_col in head.columns)
    )
    if has_border:
        if cfg.border_x_col not in need:
            need.append(cfg.border_x_col)
        if cfg.border_y_col not in need:
            need.append(cfg.border_y_col)
    has_dist_cut = (cfg.max_dist_match is not None) and np.isfinite(float(cfg.max_dist_match)) and ("dist_match" in head.columns)

    nrows = count_csv_rows_fast(meta_path)
    valid = load_valid_mask(valid_path, nrows=nrows)

    chunksize = 200_000
    offset = 0
    chunk_i = 0
    t0 = time.time()
    seam_dropped_total = 0

    last_npz_path = None
    last_npz_X = None

    for chunk in pd.read_csv(meta_path, usecols=need, chunksize=chunksize, engine="python", on_bad_lines="skip"):
        chunk_i += 1
        if (ptrs["train"] >= limits["train"]) and (ptrs["val"] >= limits["val"]) and (ptrs["test"] >= limits["test"]):
            return

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
        if need_source_id:
            chunk["source_id"] = pd.to_numeric(chunk["source_id"], errors="coerce")

        m = (
            v
            & chunk["npz_file"].notna().to_numpy()
            & chunk["index_in_file"].notna().to_numpy()
            & chunk["split_code"].isin(list(allow_splits)).to_numpy()
            & np.all(np.isfinite(chunk[feature_cols].to_numpy(dtype=float)), axis=1)
        )
        if source_ids_keep is not None:
            sid = chunk["source_id"].to_numpy(dtype=float)
            sid_ok = np.isfinite(sid)
            sid_keep = np.zeros(len(chunk), dtype=bool)
            if np.any(sid_ok):
                sid_keep[sid_ok] = np.isin(sid[sid_ok].astype(np.int64), source_ids_keep, assume_unique=False)
            m &= sid_keep
        if source_ids_bad_quality is not None:
            sid = chunk["source_id"].to_numpy(dtype=float)
            sid_ok = np.isfinite(sid)
            sid_bad = np.zeros(len(chunk), dtype=bool)
            if np.any(sid_ok):
                sid_bad[sid_ok] = np.isin(sid[sid_ok].astype(np.int64), source_ids_bad_quality, assume_unique=False)
            if bool(cfg.quality_drop_unknown_source_id):
                m &= sid_ok & (~sid_bad)
            else:
                m &= (~sid_bad)
        if (cfg.gaia_g_mag_min is not None) and (g_col is not None):
            g = chunk[g_col].to_numpy(dtype=float)
            m &= np.isfinite(g) & (g >= float(cfg.gaia_g_mag_min))
        if has_border:
            x_lo, x_hi, y_lo, y_hi = border_bounds
            x = pd.to_numeric(chunk[cfg.border_x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(chunk[cfg.border_y_col], errors="coerce").to_numpy(dtype=float)
            m &= np.isfinite(x) & np.isfinite(y) & (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
        if has_dist_cut:
            d = pd.to_numeric(chunk["dist_match"], errors="coerce").to_numpy(dtype=float)
            m &= np.isfinite(d) & (d <= float(cfg.max_dist_match))
        chunk = chunk.loc[m]
        if len(chunk) == 0:
            if (chunk_i == 1) or (chunk_i % 10 == 0) or (offset >= nrows):
                frac = (offset / nrows) if nrows > 0 else 1.0
                elapsed = time.time() - t0
                eta = ((elapsed / frac) - elapsed) if frac > 0 else np.nan
                print(
                    f"[LOAD] {label}: rows {offset}/{nrows} ({100.0 * frac:5.1f}%)"
                    f" | elapsed {fmt_duration(elapsed)} | eta {fmt_duration(eta)}"
                    f" | written train/val/test={ptrs['train']}/{ptrs['val']}/{ptrs['test']}"
                )
            if offset >= nrows:
                break
            continue

        for npz_name, sub in chunk.groupby("npz_file", sort=False):
            if (ptrs["train"] >= limits["train"]) and (ptrs["val"] >= limits["val"]) and (ptrs["test"] >= limits["test"]):
                return

            npz_path = in_dir / str(npz_name)
            if not npz_path.exists():
                continue

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
            if cfg.drop_seam_cuts:
                bad_seam = seam_cut_mask(
                    stamps,
                    low_threshold=float(cfg.seam_low_threshold),
                    line_frac_min=float(cfg.seam_line_frac_min),
                    diag_len_min=int(cfg.seam_diag_len_min),
                )
                if np.any(bad_seam):
                    keep = ~bad_seam
                    seam_dropped_total += int(np.sum(bad_seam))
                    sub = sub.iloc[keep]
                    stamps = stamps[keep]
                    if stamps.shape[0] == 0:
                        continue
            stamps_n, ok, integ = normalize_by_integral(stamps, cfg.use_positive_only, cfg.eps_integral)
            sub = sub.iloc[ok]
            stamps_n = stamps_n[ok]
            integ = integ[ok]
            if stamps_n.shape[0] == 0:
                continue

            if stamps_n.shape[1] != stamp_pix or stamps_n.shape[2] != stamp_pix:
                raise RuntimeError(f"stamp_pix mismatch: got {stamps_n.shape[1:]} expected {(stamp_pix, stamp_pix)}")

            # Targets
            yshape = stamps_n.reshape(stamps_n.shape[0], -1).astype(np.float32)  # (n, D)
            yflux = np.log10(integ.astype(np.float32))                           # (n,)

            feats = sub[feature_cols].to_numpy(dtype=np.float32)
            feats = (feats - gaia_min[None, :]) / gaia_iqr[None, :]
            split_codes = sub["split_code"].to_numpy(dtype=int)

            meta_row = sub["meta_row"].to_numpy(dtype=np.int64)
            idx_in_file = sub["index_in_file"].to_numpy(dtype=np.int64)

            sid = sub["source_id"].to_numpy(dtype=np.float64) if "source_id" in sub.columns else None
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
                Ys = yshape[ms]
                Fs = yflux[ms]

                room = limits[split_name] - ptrs[split_name]
                if room <= 0:
                    continue

                n_take = min(Xs.shape[0], room)
                Xs = Xs[:n_take]
                Ys = Ys[:n_take]
                Fs = Fs[:n_take]

                p0 = ptrs[split_name]
                mmaps[f"X_{split_name}"][p0:p0 + n_take] = Xs
                mmaps[f"Yshape_{split_name}"][p0:p0 + n_take] = Ys
                mmaps[f"Yflux_{split_name}"][p0:p0 + n_take] = Fs
                if split_name in ("train", "val"):
                    wv = np.ones(n_take, dtype=np.float32)
                    if (source_ids_weighted is not None) and (sid is not None):
                        sid_s = sid[ms][:n_take]
                        sid_ok = np.isfinite(sid_s)
                        if np.any(sid_ok):
                            w_mask = np.zeros(n_take, dtype=bool)
                            w_mask[sid_ok] = np.isin(
                                sid_s[sid_ok].astype(np.int64),
                                source_ids_weighted,
                                assume_unique=False,
                            )
                            wv[w_mask] = float(nonpsf_weight)
                    mmaps[f"W_{split_name}"][p0:p0 + n_take] = wv
                ptrs[split_name] += n_take

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
                        if sid_t is not None: row["source_id"] = int(sid_t[i]) if np.isfinite(sid_t[i]) else ""
                        if eu_ra_t is not None: row["v_alpha_j2000"] = eu_ra_t[i]
                        if eu_de_t is not None: row["v_delta_j2000"] = eu_de_t[i]
                        if ga_ra_t is not None: row["ra"] = ga_ra_t[i]
                        if ga_de_t is not None: row["dec"] = ga_de_t[i]
                        if dmat_t is not None: row["dist_match"] = dmat_t[i]
                        trace_test_writer.writerow(row)

        if (chunk_i == 1) or (chunk_i % 10 == 0) or (offset >= nrows):
            frac = (offset / nrows) if nrows > 0 else 1.0
            elapsed = time.time() - t0
            eta = ((elapsed / frac) - elapsed) if frac > 0 else np.nan
            print(
                f"[LOAD] {label}: rows {offset}/{nrows} ({100.0 * frac:5.1f}%)"
                f" | elapsed {fmt_duration(elapsed)} | eta {fmt_duration(eta)}"
                f" | written train/val/test={ptrs['train']}/{ptrs['val']}/{ptrs['test']}"
            )

        if offset >= nrows:
            break

    if cfg.drop_seam_cuts and (seam_dropped_total > 0):
        print(f"[SEAM FILTER] {label}: dropped {seam_dropped_total} seam/cut-like stamps")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def fmt_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "n/a"
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ======================================================================
# MAIN
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", default=CFG.feature_set, choices=["8D", "10D", "16D", "17D"])
    ap.add_argument("--run_name", default=CFG.run_name)
    ap.add_argument("--gaia_scaler_npz", default="", help="Optional path to scaler npz (overrides default).")
    ap.add_argument("--max_train", type=int, default=-1)
    ap.add_argument("--max_val", type=int, default=-1)
    ap.add_argument("--max_test", type=int, default=-1)
    ap.add_argument("--num_boost_round", type=int, default=CFG.num_boost_round)
    ap.add_argument("--early_stopping_rounds", type=int, default=CFG.early_stopping_rounds)
    ap.add_argument("--learning_rate", type=float, default=CFG.learning_rate)
    ap.add_argument("--gaia_g_mag_min", type=float, default=CFG.gaia_g_mag_min if CFG.gaia_g_mag_min is not None else float("nan"))
    ap.add_argument("--border_quantile", type=float, default=float("nan"), help="Exclude border rows using x/y quantile bands per metadata file (e.g. 0.01 drops outer 1%% on each side).")
    ap.add_argument("--border_x_col", type=str, default=CFG.border_x_col)
    ap.add_argument("--border_y_col", type=str, default=CFG.border_y_col)
    ap.add_argument("--max_dist_match", type=float, default=float("nan"), help="Optional dist_match upper bound; rows above are excluded.")
    ap.add_argument("--drop_seam_cuts", action="store_true", help="Drop raw stamps with cut/seam-like lines (row/col/diag near-zero spans).")
    ap.add_argument("--seam_low_threshold", type=float, default=CFG.seam_low_threshold, help="Pixel threshold used to define seam-line pixels.")
    ap.add_argument("--seam_line_frac_min", type=float, default=CFG.seam_line_frac_min, help="Minimum fraction of low pixels along a row/col/diag to flag seam.")
    ap.add_argument("--seam_diag_len_min", type=int, default=CFG.seam_diag_len_min, help="Minimum diagonal length considered for seam detection.")
    ap.add_argument("--count_only", action="store_true", help="Only compute and print train/val/test counts after filtering, then exit.")
    ap.add_argument("--psf_labels_csv", default="", help="Optional labels CSV with source_id,label")
    ap.add_argument("--psf_label_keep", type=int, default=-1, help="0=psf_like, 1=non_psf_like")
    ap.add_argument("--weight_labels_csv", default="", help="Optional labels CSV for sample weighting by source_id.")
    ap.add_argument("--weight_label_target", type=int, default=1, help="Label to upweight in --weight_labels_csv.")
    ap.add_argument("--weight_multiplier", type=float, default=1.0, help="Sample weight multiplier for target label rows.")
    ap.add_argument("--weight_min_confidence", type=float, default=float("nan"), help="Optional confidence threshold for weighted target rows.")
    ap.add_argument("--quality_flags_csv", default="", help="Optional CSV containing source_id-level quality_flag bitmask.")
    ap.add_argument("--quality_id_col", type=str, default="source_id", help="ID column in --quality_flags_csv.")
    ap.add_argument("--quality_flag_col", type=str, default="quality_flag", help="Bitmask column in --quality_flags_csv.")
    ap.add_argument("--quality_bits_mask", type=int, default=0, help="Drop rows whose quality_flag has any bits in this mask set (e.g. 39 for 1|2|4|32).")
    ap.add_argument("--quality_drop_unknown_source_id", action="store_true", help="If set, also drop rows with missing source_id when quality filtering is enabled.")
    args = ap.parse_args()

    CFG.feature_set = str(args.feature_set)
    CFG.run_name = str(args.run_name)
    if str(args.gaia_scaler_npz).strip():
        CFG.gaia_scaler_npz = Path(str(args.gaia_scaler_npz))
    CFG.max_train = None if int(args.max_train) < 0 else int(args.max_train)
    CFG.max_val = None if int(args.max_val) < 0 else int(args.max_val)
    CFG.max_test = None if int(args.max_test) < 0 else int(args.max_test)
    CFG.num_boost_round = int(args.num_boost_round)
    CFG.early_stopping_rounds = int(args.early_stopping_rounds)
    CFG.learning_rate = float(args.learning_rate)
    CFG.gaia_g_mag_min = None if (not np.isfinite(args.gaia_g_mag_min)) else float(args.gaia_g_mag_min)
    CFG.border_quantile = None if (not np.isfinite(args.border_quantile)) else float(args.border_quantile)
    CFG.border_x_col = str(args.border_x_col)
    CFG.border_y_col = str(args.border_y_col)
    CFG.max_dist_match = None if (not np.isfinite(args.max_dist_match)) else float(args.max_dist_match)
    CFG.drop_seam_cuts = bool(args.drop_seam_cuts)
    CFG.seam_low_threshold = float(args.seam_low_threshold)
    CFG.seam_line_frac_min = float(args.seam_line_frac_min)
    CFG.seam_diag_len_min = int(args.seam_diag_len_min)
    CFG.quality_flags_csv = str(args.quality_flags_csv)
    CFG.quality_id_col = str(args.quality_id_col)
    CFG.quality_flag_col = str(args.quality_flag_col)
    CFG.quality_bits_mask = int(args.quality_bits_mask)
    CFG.quality_drop_unknown_source_id = bool(args.quality_drop_unknown_source_id)

    source_ids_keep = None
    if str(args.psf_labels_csv).strip():
        if int(args.psf_label_keep) not in (0, 1):
            raise RuntimeError("--psf_label_keep must be 0 or 1 when --psf_labels_csv is provided.")
        source_ids_keep = load_psf_source_ids(Path(args.psf_labels_csv), int(args.psf_label_keep))

    source_ids_weighted = None
    if str(args.weight_labels_csv).strip():
        if float(args.weight_multiplier) < 1.0:
            raise RuntimeError("--weight_multiplier must be >= 1.0")
        source_ids_weighted = load_weight_source_ids(
            labels_csv=Path(args.weight_labels_csv),
            label_target=int(args.weight_label_target),
            min_confidence=float(args.weight_min_confidence),
        )

    source_ids_bad_quality = None
    quality_stats = None
    if str(args.quality_flags_csv).strip():
        if int(args.quality_bits_mask) <= 0:
            raise RuntimeError("--quality_bits_mask must be >0 when --quality_flags_csv is provided.")
        source_ids_bad_quality, quality_stats = load_bad_quality_source_ids(
            quality_csv=Path(args.quality_flags_csv),
            id_col=str(args.quality_id_col),
            flag_col=str(args.quality_flag_col),
            bits_mask=int(args.quality_bits_mask),
        )

    np.random.seed(CFG.random_seed)
    feature_cols = get_feature_cols(CFG.feature_set)
    feature_set_norm = normalize_feature_set(CFG.feature_set)

    default_scaler_10d = CFG.base_dir / "output" / "scalers" / "y_feature_iqr.npz"
    if feature_set_norm == "16D" and CFG.gaia_scaler_npz == default_scaler_10d:
        CFG.gaia_scaler_npz = CFG.base_dir / "output" / "scalers" / "y_feature_iqr_16d.npz"

    out_root = CFG.base_dir / "output" / "ml_runs" / CFG.run_name
    array_dir = out_root / "arrays"
    model_dir = out_root / "models"
    metrics_dir = out_root / "metrics"
    trace_dir = out_root / "trace"
    plot_dir = CFG.plots_root / CFG.run_name

    for d in [out_root, array_dir, model_dir, metrics_dir, trace_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(CFG).items()}, f, indent=2)

    stamp_pix = infer_stamp_pix(CFG.dataset_root)
    D = stamp_pix * stamp_pix
    gaia_min, gaia_iqr = load_gaia_scaler(CFG.gaia_scaler_npz, feature_cols)

    print("\n=== RUN (PIXELS+FLUX) ===")
    print("RUN_NAME:", CFG.run_name)
    print("stamp_pix:", stamp_pix, "D:", D)
    print("features:", feature_cols)
    print("gaia_g_mag_min:", CFG.gaia_g_mag_min)
    print("border_quantile:", CFG.border_quantile, "border_cols:", (CFG.border_x_col, CFG.border_y_col))
    print("max_dist_match:", CFG.max_dist_match)
    print(
        "drop_seam_cuts:", CFG.drop_seam_cuts,
        f"(low<= {CFG.seam_low_threshold}, frac>= {CFG.seam_line_frac_min}, diag_len>={CFG.seam_diag_len_min})"
    )
    print("use_aug:", CFG.use_aug, "aug_train_only:", CFG.aug_train_only)
    if source_ids_keep is not None:
        print("PSF label filter enabled:", int(args.psf_label_keep), f"(source_ids={len(source_ids_keep)})")
    if source_ids_weighted is not None:
        cval = float(args.weight_min_confidence)
        ctext = "none" if (not np.isfinite(cval)) else f">={cval:.3f}"
        print(
            "Sample weighting enabled:",
            f"target_label={int(args.weight_label_target)}",
            f"multiplier={float(args.weight_multiplier):.3f}",
            f"confidence={ctext}",
            f"(source_ids={len(source_ids_weighted)})",
        )
    if source_ids_bad_quality is not None:
        print(
            "Quality-bit exclusion enabled:",
            f"mask={int(args.quality_bits_mask)}",
            f"bad_source_ids={len(source_ids_bad_quality)}",
            f"unknown_source_id_policy={'drop' if bool(args.quality_drop_unknown_source_id) else 'keep'}",
        )
        if quality_stats is not None:
            print(
                "Quality CSV stats:",
                f"rows_valid={quality_stats['rows_valid']}",
                f"rows_flagged={quality_stats['rows_flagged']}",
                f"unique_source_ids_flagged={quality_stats['unique_source_ids_flagged']}",
            )
    print("OUT_ROOT:", out_root)
    print("PLOTS   :", plot_dir)

    raw_meta_name = RAW_META_16D if feature_set_norm == "16D" else RAW_META
    aug_meta_name = AUG_META_16D if feature_set_norm == "16D" else AUG_META

    field_dirs = list_field_dirs(CFG.dataset_root, raw_meta_name=raw_meta_name)
    if len(field_dirs) == 0:
        raise RuntimeError(f"No datasets found under {CFG.dataset_root}")

    allow_raw = {0, 1, 2}
    allow_aug = {0} if CFG.aug_train_only else {0, 1, 2}

    total_train = total_val = total_test = 0
    for field_dir in field_dirs:
        raw_meta_path = field_dir / raw_meta_name
        raw_border_bounds = get_border_bounds(
            raw_meta_path,
            x_col=CFG.border_x_col,
            y_col=CFG.border_y_col,
            q=float(CFG.border_quantile) if (CFG.border_quantile is not None) else np.nan,
        )
        c = count_rows(
            raw_meta_path,
            field_dir / VALID_NAME,
            feature_cols,
            allow_raw,
            CFG.gaia_g_mag_min,
            border_bounds=raw_border_bounds,
            border_x_col=CFG.border_x_col,
            border_y_col=CFG.border_y_col,
            max_dist_match=CFG.max_dist_match,
            source_ids_keep=source_ids_keep,
            source_ids_bad_quality=source_ids_bad_quality,
            quality_drop_unknown_source_id=bool(CFG.quality_drop_unknown_source_id),
        )
        total_train += c.get(0, 0)
        total_val += c.get(1, 0)
        total_test += c.get(2, 0)

        if CFG.use_aug:
            aug_dir = field_dir / CFG.aug_subdir
            if aug_dir.exists():
                aug_meta_path = aug_dir / aug_meta_name
                aug_border_bounds = get_border_bounds(
                    aug_meta_path,
                    x_col=CFG.border_x_col,
                    y_col=CFG.border_y_col,
                    q=float(CFG.border_quantile) if (CFG.border_quantile is not None) else np.nan,
                )
                c2 = count_rows(
                    aug_meta_path,
                    aug_dir / VALID_NAME,
                    feature_cols,
                    allow_aug,
                    CFG.gaia_g_mag_min,
                    border_bounds=aug_border_bounds,
                    border_x_col=CFG.border_x_col,
                    border_y_col=CFG.border_y_col,
                    max_dist_match=CFG.max_dist_match,
                    source_ids_keep=source_ids_keep,
                    source_ids_bad_quality=source_ids_bad_quality,
                    quality_drop_unknown_source_id=bool(CFG.quality_drop_unknown_source_id),
                )
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
    if CFG.drop_seam_cuts:
        print("[NOTE] Seam-cut filtering is applied during NPZ load, so final written counts can be lower than pre-count estimates.")
    if args.count_only:
        print("\nCount-only mode enabled; exiting before array write/training.")
        return

    mmaps = open_memmaps(array_dir, n_train, n_val, n_test, n_feat=len(feature_cols), D=D)
    ptrs = {"train": 0, "val": 0, "test": 0}
    limits = {"train": n_train, "val": n_val, "test": n_test}

    trace_test_csv = trace_dir / "trace_test.csv"
    trace_fields = [
        "test_index","field_tag","dataset_kind","npz_relpath","npz_file","index_in_file","meta_row",
        "source_id","v_alpha_j2000","v_delta_j2000","ra","dec","dist_match"
    ]
    trace_fh = open(trace_test_csv, "w", newline="", encoding="utf-8")
    trace_writer = csv.DictWriter(trace_fh, fieldnames=trace_fields, extrasaction="ignore")
    trace_writer.writeheader()

    load_tasks = []
    for field_dir in field_dirs:
        tag = field_dir.name
        load_tasks.append({
            "field_tag": tag,
            "dataset_kind": "RAW",
            "in_dir": field_dir,
            "meta_path": field_dir / raw_meta_name,
            "valid_path": field_dir / VALID_NAME,
            "allow_splits": {0, 1, 2},
            "label": f"{tag} (RAW)",
        })
        if CFG.use_aug:
            aug_dir = field_dir / CFG.aug_subdir
            if aug_dir.exists():
                load_tasks.append({
                    "field_tag": tag,
                    "dataset_kind": "AUG",
                    "in_dir": aug_dir,
                    "meta_path": aug_dir / aug_meta_name,
                    "valid_path": aug_dir / VALID_NAME,
                    "allow_splits": {0} if CFG.aug_train_only else {0, 1, 2},
                    "label": f"{tag} (AUG)",
                })

    n_tasks = len(load_tasks)
    print(f"\n=== LOAD TASKS ===\ncount: {n_tasks}")
    load_phase_t0 = time.time()

    try:
        for task_i, task in enumerate(load_tasks, start=1):
            task_t0 = time.time()
            print(f"\n=== LOAD TASK {task_i}/{n_tasks}: {task['label']} ===")
            fill_arrays_for_meta_pixelsflux(
                dataset_root=CFG.dataset_root,
                field_tag=task["field_tag"],
                dataset_kind=task["dataset_kind"],
                in_dir=task["in_dir"],
                meta_path=task["meta_path"],
                valid_path=task["valid_path"],
                feature_cols=feature_cols,
                gaia_min=gaia_min,
                gaia_iqr=gaia_iqr,
                stamp_pix=stamp_pix,
                mmaps=mmaps,
                ptrs=ptrs,
                limits=limits,
                allow_splits=task["allow_splits"],
                cfg=CFG,
                label=task["label"],
                trace_test_writer=trace_writer,
                source_ids_keep=source_ids_keep,
                source_ids_weighted=source_ids_weighted,
                source_ids_bad_quality=source_ids_bad_quality,
                nonpsf_weight=float(args.weight_multiplier),
            )

            elapsed_phase = time.time() - load_phase_t0
            done = task_i
            avg = elapsed_phase / done
            remaining = n_tasks - done
            eta = avg * remaining
            task_elapsed = time.time() - task_t0
            print(
                f"[TASK DONE] {task['label']} in {fmt_duration(task_elapsed)}"
                f" | tasks_left={remaining} | phase_eta={fmt_duration(eta)}"
                f" | ptr train/val/test={ptrs['train']}/{ptrs['val']}/{ptrs['test']}"
            )

            if (ptrs["train"] >= n_train) and (ptrs["val"] >= n_val) and (ptrs["test"] >= n_test):
                print("Reached requested array limits; stopping load phase early.")
                break
    finally:
        trace_fh.close()

    for arr in mmaps.values():
        arr.flush()

    # Save manifest
    manifest_path = out_root / "manifest_arrays.npz"
    np.savez_compressed(
        manifest_path,
        n_train=np.int64(ptrs["train"]),
        n_val=np.int64(ptrs["val"]),
        n_test=np.int64(ptrs["test"]),
        n_features=np.int64(len(feature_cols)),
        feature_cols=np.array(feature_cols, dtype=object),
        stamp_pix=np.int64(stamp_pix),
        D=np.int64(D),
        X_train_path=str(array_dir / "X_train.mmap"),
        Yshape_train_path=str(array_dir / "Yshape_train.mmap"),
        Yflux_train_path=str(array_dir / "Yflux_train.mmap"),
        X_val_path=str(array_dir / "X_val.mmap"),
        Yshape_val_path=str(array_dir / "Yshape_val.mmap"),
        Yflux_val_path=str(array_dir / "Yflux_val.mmap"),
        W_train_path=str(array_dir / "W_train.mmap"),
        W_val_path=str(array_dir / "W_val.mmap"),
        X_test_path=str(array_dir / "X_test.mmap"),
        Yshape_test_path=str(array_dir / "Yshape_test.mmap"),
        Yflux_test_path=str(array_dir / "Yflux_test.mmap"),
        scaler_npz=str(CFG.gaia_scaler_npz),
        dataset_root=str(CFG.dataset_root),
        trace_test_csv=str(trace_test_csv),
        plots_dir=str(plot_dir),
    )

    print("\nSaved manifest:", manifest_path)
    print("Saved TEST trace:", trace_test_csv)
    print(f"Arrays written: train {ptrs['train']}/{n_train} | val {ptrs['val']}/{n_val} | test {ptrs['test']}/{n_test}")

    # Open memmaps read-only for training
    n_train_eff = int(ptrs["train"])
    n_val_eff = int(ptrs["val"])
    n_feat = len(feature_cols)

    X_train = np.memmap(array_dir / "X_train.mmap", dtype="float32", mode="r", shape=(n_train_eff, n_feat))
    Yshape_train = np.memmap(array_dir / "Yshape_train.mmap", dtype="float32", mode="r", shape=(n_train_eff, D))
    Yflux_train = np.memmap(array_dir / "Yflux_train.mmap", dtype="float32", mode="r", shape=(n_train_eff,))
    X_val = np.memmap(array_dir / "X_val.mmap", dtype="float32", mode="r", shape=(n_val_eff, n_feat))
    Yshape_val = np.memmap(array_dir / "Yshape_val.mmap", dtype="float32", mode="r", shape=(n_val_eff, D))
    Yflux_val = np.memmap(array_dir / "Yflux_val.mmap", dtype="float32", mode="r", shape=(n_val_eff,))
    W_train = np.memmap(array_dir / "W_train.mmap", dtype="float32", mode="r", shape=(n_train_eff,))
    W_val = np.memmap(array_dir / "W_val.mmap", dtype="float32", mode="r", shape=(n_val_eff,))

    weight_summary = pd.DataFrame(
        [
            {"split": "train", "n": n_train_eff, "weight_mean": float(np.mean(W_train)), "weight_max": float(np.max(W_train)), "weighted_frac": float(np.mean(np.asarray(W_train) > 1.0))},
            {"split": "val", "n": n_val_eff, "weight_mean": float(np.mean(W_val)), "weight_max": float(np.max(W_val)), "weighted_frac": float(np.mean(np.asarray(W_val) > 1.0))},
        ]
    )
    weight_summary.to_csv(metrics_dir / "sample_weight_summary.csv", index=False)

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

    # ---- Train flux model ----
    print("\n=== TRAINING FLUX MODEL: log10(F) ===")
    dtrainF = xgb.DMatrix(X_train, label=Yflux_train, weight=W_train)
    dvalF = xgb.DMatrix(X_val, label=Yflux_val, weight=W_val)
    evals_result_F = {}
    booster_flux = xgb.train(
        params=params,
        dtrain=dtrainF,
        num_boost_round=CFG.num_boost_round,
        evals=[(dtrainF, "train"), (dvalF, "val")],
        early_stopping_rounds=CFG.early_stopping_rounds,
        evals_result=evals_result_F,
        verbose_eval=False,
    )
    predF_val = booster_flux.predict(dvalF)
    rF = rmse(Yflux_val, predF_val)
    print(f"Flux RMSE(val) on log10(F): {rF:.6f} | best_iter={int(booster_flux.best_iteration)}")
    booster_flux.save_model(str(model_dir / "booster_flux.json"))

    # ---- Train shape pixel models ----
    print(f"\n=== TRAINING SHAPE MODELS (one booster per pixel): D={D} ===")
    dtrain = xgb.DMatrix(X_train, weight=W_train)
    dval = xgb.DMatrix(X_val, weight=W_val)

    rmse_pix_val = np.full(D, np.nan, dtype=float)
    shape_t0 = time.time()

    for j in range(D):
        ytr = Yshape_train[:, j]
        yva = Yshape_val[:, j]

        dtrain.set_label(ytr)
        dval.set_label(yva)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=CFG.num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=CFG.early_stopping_rounds,
            verbose_eval=False,
        )

        pred_val = booster.predict(dval)
        r = rmse(yva, pred_val)
        rmse_pix_val[j] = r

        booster.save_model(str(model_dir / f"booster_pix_{j:04d}.json"))

        if (j == 0) or ((j + 1) % 50 == 0) or (j == D - 1):
            done = j + 1
            elapsed = time.time() - shape_t0
            eta = (elapsed / done) * (D - done)
            print(
                f"  pix {j:04d}/{D-1:04d} | RMSE(val)={r:.6g} | best_iter={int(booster.best_iteration)}"
                f" | elapsed={fmt_duration(elapsed)} | eta={fmt_duration(eta)}"
            )

    # Basic plot: per-pixel RMSE distribution
    plt.figure(figsize=(7.2, 4.6))
    plt.hist(rmse_pix_val[np.isfinite(rmse_pix_val)], bins=60)
    plt.xlabel("RMSE(val) per pixel in normalized space")
    plt.ylabel("Count")
    plt.title(f"Per-pixel RMSE distribution (shape targets)\n{CFG.run_name}")
    savefig(plot_dir / "rmse_val_per_pixel_hist.png")

    pd.DataFrame({"pixel": np.arange(D), "rmse_val": rmse_pix_val}).to_csv(metrics_dir / "metrics_shape_pixels.csv", index=False)
    pd.DataFrame({"metric": ["rmse_val_log10F"], "value": [rF]}).to_csv(metrics_dir / "metrics_flux.csv", index=False)

    print("\nDONE.")
    print("Run folder :", out_root)
    print("Plots folder:", plot_dir)


if __name__ == "__main__":
    main()
