"""
69_morph_metrics_xgb.py
=======================

Train 11 XGBoost regressors mapping Gaia 16D features to morphology metrics
computed on normalized 20x20 Euclid VIS stamps.

Metrics (11):
  concentration, ellipticity, roundness, asymmetry_180, mirror_asymmetry,
  gini, m20, kurtosis, smoothness, multipeak_ratio, hf_artifacts

Pipeline:
  1. Load raw 20x20 stamps from all ERO field NPZ files.
  2. Normalize by positive integral (same protocol as existing pipeline).
  3. Compute 11 morphology metrics on the normalized stamps.
     (gini, m20, kurtosis additionally subtract border-estimated background)
  4. Load 16D Gaia features and apply existing IQR scaler.
  5. Filter: phot_g_mean_mag >= 17, valid_rows mask, drop NaN.
  6. Reuse split_code (0=train, 1=val, 2=test) from metadata_16d.csv.
  7. Train 11 independent XGBoost regressors.
  8. Evaluate: R2, RMSE, MAD, Pearson r per metric on test set.
  9. Diagnostics: stamp SNR distribution (border-MAD estimate).

Outputs:
  output/ml_runs/<RUN_NAME>/
    models/booster_<metric>.json  (x11)
    metrics_summary.csv
    config.json
  plots/ml_runs/<RUN_NAME>/
    01_scatter_predicted_vs_true.png
    02_snr_diagnostic.png
    03_metric_distributions.png
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy import stats as scipy_stats
from scipy.ndimage import gaussian_filter, laplace, maximum_filter
from sklearn.metrics import r2_score

from feature_schema import get_feature_cols, scaler_stem


# ======================================================================
# CONFIG
# ======================================================================

@dataclass
class RunConfig:
    base_dir: Path = Path(__file__).resolve().parents[2]
    dataset_root: Path = None
    gaia_scaler_npz: Path = None

    feature_set: str = "16D"
    run_name: str = "morph_metrics_16d"

    # Quality filters
    gaia_g_mag_min: float = 17.0

    # XGBoost hyperparameters (same as existing baseline)
    num_boost_round: int = 2000
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    early_stopping_rounds: int = 100

    # Normalization
    use_positive_only: bool = True
    eps_integral: float = 1e-12

    # Metric parameters
    smoothness_sigma: float = 1.0       # Gaussian kernel sigma for smoothness
    multipeak_min_distance: int = 2     # Min pixel distance between peaks
    multipeak_threshold_rel: float = 0.1  # Peak threshold relative to stamp max

    random_seed: int = 123
    plots_root: Path = None
    output_root: Path = None

    def __post_init__(self):
        if self.dataset_root is None:
            self.dataset_root = self.base_dir / "output" / "dataset_npz"
        if self.gaia_scaler_npz is None:
            stem = scaler_stem(self.feature_set)
            self.gaia_scaler_npz = self.base_dir / "output" / "scalers" / f"{stem}.npz"
        if self.plots_root is None:
            self.plots_root = self.base_dir / "plots" / "ml_runs" / self.run_name
        if self.output_root is None:
            self.output_root = self.base_dir / "output" / "ml_runs" / self.run_name


CFG = RunConfig()

METRIC_NAMES: List[str] = [
    "concentration",
    "ellipticity",
    "roundness",
    "asymmetry_180",
    "mirror_asymmetry",
    "gini",
    "m20",
    "kurtosis",
    "smoothness",
    "multipeak_ratio",
    "hf_artifacts",
]

RAW_META_16D = "metadata_16d.csv"
VALID_NAME = "valid_rows.npy"


# ======================================================================
# I/O HELPERS
# ======================================================================

def list_field_dirs(dataset_root: Path) -> List[Path]:
    if (dataset_root / RAW_META_16D).exists():
        return [dataset_root]
    return sorted([
        p for p in dataset_root.iterdir()
        if p.is_dir() and (p / RAW_META_16D).exists()
    ])


def load_valid_mask(valid_path: Path, nrows: int) -> np.ndarray:
    arr = np.load(valid_path, allow_pickle=True)
    if arr.dtype == np.bool_:
        if arr.shape[0] == nrows:
            return arr.copy()
        m = np.zeros(nrows, dtype=bool)
        m[:min(nrows, arr.shape[0])] = arr[:min(nrows, arr.shape[0])]
        return m
    if np.issubdtype(arr.dtype, np.integer):
        idx = arr.astype(np.int64)
        m = np.zeros(nrows, dtype=bool)
        ok = (idx >= 0) & (idx < nrows)
        m[idx[ok]] = True
        return m
    raise RuntimeError(f"Unsupported valid_rows dtype: {arr.dtype}")


def load_gaia_scaler(
    npz_path: Path, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
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


# ======================================================================
# STAMP NORMALIZATION
# ======================================================================

def positive_integral(stamps: np.ndarray, use_positive_only: bool) -> np.ndarray:
    s = np.clip(stamps, 0, None) if use_positive_only else stamps
    return np.sum(s, axis=(1, 2))


def normalize_stamps(
    stamps: np.ndarray,
    use_positive_only: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (normalized_stamps, ok_mask, integrals)."""
    stamps = stamps.astype(np.float32, copy=False)
    integ = positive_integral(stamps, use_positive_only).astype(np.float32)
    ok = np.isfinite(integ) & (integ > eps)
    out = stamps.copy()
    out[ok] /= integ[ok, None, None]
    return out, ok, integ


# ======================================================================
# BORDER STATISTICS (for background estimation and SNR diagnostic)
# ======================================================================

def border_pixels(stamp: np.ndarray) -> np.ndarray:
    """Outermost ring of a 2D stamp."""
    return np.concatenate([
        stamp[0, :], stamp[-1, :],
        stamp[1:-1, 0], stamp[1:-1, -1],
    ])


def border_stats(stamp_raw: np.ndarray) -> Tuple[float, float]:
    """Returns (bg_estimate, sigma_estimate) from border pixels of raw stamp."""
    bp = border_pixels(stamp_raw)
    bg = float(np.median(bp))
    mad = float(np.median(np.abs(bp - bg)))
    return bg, 1.4826 * mad


# ======================================================================
# MORPHOLOGY METRICS
# ======================================================================

def flux_centroid(stamp: np.ndarray) -> Tuple[float, float]:
    s = np.clip(stamp, 0, None)
    total = s.sum()
    H, W = stamp.shape
    if total <= 0:
        return H / 2.0, W / 2.0
    yy, xx = np.mgrid[:H, :W]
    cy = float(np.sum(s * yy) / total)
    cx = float(np.sum(s * xx) / total)
    return cy, cx


def compute_concentration(stamp: np.ndarray, cy: float, cx: float) -> float:
    """C = 5 * log10(r80 / r20) via cumulative flux in circular apertures."""
    H, W = stamp.shape
    yy, xx = np.mgrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).flatten()
    s = np.clip(stamp, 0, None).flatten()
    total = s.sum()
    if total <= 0:
        return np.nan
    order = np.argsort(r)
    r_sorted = r[order]
    cum = np.cumsum(s[order])
    i20 = min(np.searchsorted(cum, 0.20 * total), len(r_sorted) - 1)
    i80 = min(np.searchsorted(cum, 0.80 * total), len(r_sorted) - 1)
    r20 = r_sorted[i20]
    r80 = r_sorted[i80]
    if r20 <= 0:
        return np.nan
    return float(5.0 * np.log10(r80 / r20))


def compute_ellipticity_roundness(
    stamp: np.ndarray, cy: float, cx: float
) -> Tuple[float, float]:
    """Returns (ellipticity, roundness) from flux-weighted 2nd moments."""
    s = np.clip(stamp, 0, None)
    total = s.sum()
    if total <= 0:
        return np.nan, np.nan
    H, W = stamp.shape
    yy, xx = np.mgrid[:H, :W]
    dy, dx = yy - cy, xx - cx
    Qxx = float(np.sum(s * dx ** 2) / total)
    Qyy = float(np.sum(s * dy ** 2) / total)
    Qxy = float(np.sum(s * dx * dy) / total)
    trace = Qxx + Qyy
    det = Qxx * Qyy - Qxy ** 2
    disc = np.sqrt(max(0.0, (trace / 2) ** 2 - det))
    lam1 = trace / 2 + disc
    lam2 = trace / 2 - disc
    if lam1 <= 0:
        return np.nan, np.nan
    a = np.sqrt(max(0.0, lam1))
    b = np.sqrt(max(0.0, lam2))
    if a <= 0:
        return np.nan, np.nan
    return float(1.0 - b / a), float(b / a)


def compute_asymmetry_180(stamp: np.ndarray) -> float:
    """CAS asymmetry: rotate 180 degrees around center."""
    denom = 2.0 * np.sum(np.abs(stamp))
    if denom <= 0:
        return np.nan
    return float(np.sum(np.abs(stamp - stamp[::-1, ::-1])) / denom)


def compute_mirror_asymmetry(stamp: np.ndarray) -> float:
    """Mean of left-right and top-bottom flip residuals."""
    denom = 2.0 * np.sum(np.abs(stamp))
    if denom <= 0:
        return np.nan
    a_lr = float(np.sum(np.abs(stamp - stamp[:, ::-1])) / denom)
    a_tb = float(np.sum(np.abs(stamp - stamp[::-1, :])) / denom)
    return 0.5 * (a_lr + a_tb)


def compute_gini(stamp: np.ndarray, bg_norm: float) -> float:
    """Gini coefficient after background subtraction and zero-clipping."""
    s = np.sort(np.clip((stamp - bg_norm).flatten(), 0, None))
    total = s.sum()
    if total <= 0:
        return np.nan
    n = len(s)
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(idx, s) / (n * total)) - (n + 1.0) / n)


def compute_m20(
    stamp: np.ndarray, cy: float, cx: float, bg_norm: float
) -> float:
    """log10(M_brightest20% / M_total) after background subtraction."""
    s = np.clip(stamp - bg_norm, 0, None)
    total = s.sum()
    if total <= 0:
        return np.nan
    H, W = s.shape
    yy, xx = np.mgrid[:H, :W]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    M_total = float(np.sum(s * r2))
    if M_total <= 0:
        return np.nan
    flat_s = s.flatten()
    flat_r2 = r2.flatten()
    order = np.argsort(flat_s)[::-1]
    cum = np.cumsum(flat_s[order])
    thresh_idx = min(np.searchsorted(cum, 0.20 * total), len(flat_s) - 1)
    M_20 = float(np.sum(flat_s[order[:thresh_idx + 1]] * flat_r2[order[:thresh_idx + 1]]))
    if M_20 <= 0:
        return np.nan
    return float(np.log10(M_20 / M_total))


def compute_kurtosis(stamp: np.ndarray, bg_norm: float) -> float:
    """Excess kurtosis of positive pixel values after background subtraction."""
    s = (stamp - bg_norm).flatten()
    s = s[s > 0]
    if len(s) < 4:
        return np.nan
    return float(scipy_stats.kurtosis(s, fisher=True))


def compute_smoothness(stamp: np.ndarray, sigma: float = 1.0) -> float:
    """CAS smoothness: residual after Gaussian smoothing."""
    denom = 2.0 * np.sum(np.abs(stamp))
    if denom <= 0:
        return np.nan
    smooth = gaussian_filter(stamp, sigma=sigma)
    return float(np.sum(np.abs(stamp - smooth)) / denom)


def compute_multipeak_ratio(
    stamp: np.ndarray,
    min_distance: int = 2,
    threshold_rel: float = 0.1,
) -> float:
    """Flux ratio of 2nd brightest local maximum to brightest (0 if single peak)."""
    smax = float(stamp.max())
    if smax <= 0:
        return np.nan
    threshold = smax * threshold_rel
    size = 2 * min_distance + 1
    local_max = maximum_filter(stamp, size=size)
    peaks_mask = (stamp == local_max) & (stamp >= threshold)
    peak_vals = sorted(stamp[peaks_mask].tolist(), reverse=True)
    if len(peak_vals) < 2:
        return 0.0
    return float(peak_vals[1] / peak_vals[0])


def compute_hf_artifacts(stamp: np.ndarray) -> float:
    """Laplacian power ratio: high-frequency content relative to total power."""
    denom = float(np.sum(stamp ** 2))
    if denom <= 0:
        return np.nan
    lap = laplace(stamp)
    return float(np.sum(lap ** 2) / denom)


def compute_all_metrics(
    stamp_norm: np.ndarray,
    bg_norm: float,
    cfg: RunConfig,
) -> Dict[str, float]:
    """Compute all 11 morphology metrics for one normalized stamp."""
    cy, cx = flux_centroid(stamp_norm)
    ellipticity, roundness = compute_ellipticity_roundness(stamp_norm, cy, cx)
    return {
        "concentration":   compute_concentration(stamp_norm, cy, cx),
        "ellipticity":     ellipticity,
        "roundness":       roundness,
        "asymmetry_180":   compute_asymmetry_180(stamp_norm),
        "mirror_asymmetry": compute_mirror_asymmetry(stamp_norm),
        "gini":            compute_gini(stamp_norm, bg_norm),
        "m20":             compute_m20(stamp_norm, cy, cx, bg_norm),
        "kurtosis":        compute_kurtosis(stamp_norm, bg_norm),
        "smoothness":      compute_smoothness(stamp_norm, sigma=cfg.smoothness_sigma),
        "multipeak_ratio": compute_multipeak_ratio(
            stamp_norm,
            min_distance=cfg.multipeak_min_distance,
            threshold_rel=cfg.multipeak_threshold_rel,
        ),
        "hf_artifacts":    compute_hf_artifacts(stamp_norm),
    }


# ======================================================================
# DATA LOADING
# ======================================================================

def load_all_data(
    cfg: RunConfig,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load stamps + Gaia features from all ERO fields, compute metrics.

    Returns:
        X:      (N, n_feat) IQR-scaled Gaia features
        Y:      (N, 11) morphology metrics
        splits: (N,) split codes (0=train, 1=val, 2=test)
        snr:    (N,) per-stamp SNR diagnostic (positive_integral / border_sigma)
    """
    gaia_min, gaia_iqr = load_gaia_scaler(cfg.gaia_scaler_npz, feature_cols)
    field_dirs = list_field_dirs(cfg.dataset_root)
    print(f"Found {len(field_dirs)} field directories.")

    all_X, all_Y, all_splits, all_snr = [], [], [], []

    for fdir in field_dirs:
        field_tag = fdir.name
        meta_path = fdir / RAW_META_16D
        valid_path = fdir / VALID_NAME

        print(f"\n[{field_tag}]")
        meta = pd.read_csv(meta_path, engine="python", on_bad_lines="skip")
        nrows = len(meta)

        # --- Build valid mask ---
        valid = (
            load_valid_mask(valid_path, nrows)
            if valid_path.exists()
            else np.ones(nrows, dtype=bool)
        )

        # Magnitude filter
        if "phot_g_mean_mag" in meta.columns and cfg.gaia_g_mag_min is not None:
            g = pd.to_numeric(meta["phot_g_mean_mag"], errors="coerce").to_numpy()
            valid &= np.isfinite(g) & (g >= cfg.gaia_g_mag_min)

        # Feature validity
        feat_raw = meta[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid &= np.all(np.isfinite(feat_raw), axis=1)

        # Split code validity
        splits_raw = pd.to_numeric(meta["split_code"], errors="coerce").to_numpy()
        valid &= np.isfinite(splits_raw) & np.isin(splits_raw, [0, 1, 2])

        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            print("  No valid rows after filters — skipping.")
            continue

        # --- Load raw stamps grouped by NPZ file ---
        all_stamps_raw = np.full((nrows, 20, 20), np.nan, dtype=np.float32)
        for npz_name, group in meta.groupby("npz_file"):
            npz_path = fdir / str(npz_name)
            if not npz_path.exists():
                print(f"  [WARN] {npz_name} not found, skipping.")
                continue
            with np.load(npz_path) as d:
                data = d["X"].astype(np.float32)
            file_indices = pd.to_numeric(
                group["index_in_file"], errors="coerce"
            ).to_numpy(dtype=float)
            meta_idx = group.index.to_numpy()
            file_ok = (
                np.isfinite(file_indices)
                & (file_indices >= 0)
                & (file_indices < len(data))
            )
            ok_meta = meta_idx[file_ok]
            ok_file = file_indices[file_ok].astype(int)
            all_stamps_raw[ok_meta] = data[ok_file]

        # --- Normalize ---
        stamps_norm, norm_ok, integ = normalize_stamps(
            all_stamps_raw,
            use_positive_only=cfg.use_positive_only,
            eps=cfg.eps_integral,
        )
        valid &= norm_ok
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            print("  No valid rows after normalization — skipping.")
            continue

        # --- Compute metrics and SNR ---
        print(f"  Computing metrics for {len(valid_idx)} stamps ...", flush=True)
        t0 = time.time()
        metrics_list = []
        snr_list = []

        for i in valid_idx:
            s_raw = all_stamps_raw[i]
            s_norm = stamps_norm[i]
            integ_val = float(integ[i])

            bg_raw, sigma_raw = border_stats(s_raw)
            bg_norm = bg_raw / integ_val if integ_val > cfg.eps_integral else 0.0

            m = compute_all_metrics(s_norm, bg_norm, cfg)
            metrics_list.append([m[k] for k in METRIC_NAMES])

            snr = integ_val / sigma_raw if sigma_raw > 0 else np.nan
            snr_list.append(snr)

        print(f"  Done in {time.time() - t0:.1f}s")

        Y_field = np.array(metrics_list, dtype=np.float32)
        snr_field = np.array(snr_list, dtype=np.float32)
        X_field = feat_raw[valid_idx].astype(np.float32)
        X_field = (X_field - gaia_min) / np.where(gaia_iqr > 0, gaia_iqr, 1.0)
        splits_field = splits_raw[valid_idx].astype(int)

        # Drop rows with NaN in any metric
        metric_ok = np.all(np.isfinite(Y_field), axis=1)
        n_nan = int((~metric_ok).sum())
        if n_nan > 0:
            print(f"  Dropped {n_nan} rows with NaN in at least one metric.")
        Y_field = Y_field[metric_ok]
        X_field = X_field[metric_ok]
        splits_field = splits_field[metric_ok]
        snr_field = snr_field[metric_ok]

        n_tr = int(np.sum(splits_field == 0))
        n_vl = int(np.sum(splits_field == 1))
        n_te = int(np.sum(splits_field == 2))
        print(f"  Kept {len(Y_field)} rows  (train={n_tr}, val={n_vl}, test={n_te})")

        all_X.append(X_field)
        all_Y.append(Y_field)
        all_splits.append(splits_field)
        all_snr.append(snr_field)

    if not all_X:
        raise RuntimeError("No data loaded — check dataset_root and metadata files.")

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    splits = np.concatenate(all_splits, axis=0)
    snr = np.concatenate(all_snr, axis=0)

    print(
        f"\nTotal loaded: {len(X)} rows  "
        f"(train={np.sum(splits==0)}, val={np.sum(splits==1)}, test={np.sum(splits==2)})"
    )
    return X, Y, splits, snr


# ======================================================================
# TRAINING
# ======================================================================

def train_regressors(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    cfg: RunConfig,
    models_dir: Path,
) -> List[xgb.Booster]:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_lambda": cfg.reg_lambda,
        "tree_method": cfg.tree_method,
        "seed": cfg.random_seed,
    }
    boosters = []
    for i, name in enumerate(METRIC_NAMES):
        y_tr = Y_train[:, i]
        y_vl = Y_val[:, i]
        ok_tr = np.isfinite(y_tr)
        ok_vl = np.isfinite(y_vl)
        dtrain = xgb.DMatrix(X_train[ok_tr], label=y_tr[ok_tr])
        dval = xgb.DMatrix(X_val[ok_vl], label=y_vl[ok_vl])

        print(
            f"\n[{i+1}/{len(METRIC_NAMES)}] {name}  "
            f"(n_train={ok_tr.sum()}, n_val={ok_vl.sum()})"
        )
        t0 = time.time()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=cfg.num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=cfg.early_stopping_rounds,
            verbose_eval=200,
        )
        elapsed = time.time() - t0
        print(f"  best_round={booster.best_iteration}  elapsed={elapsed:.1f}s")
        out_path = models_dir / f"booster_{name}.json"
        booster.save_model(str(out_path))
        boosters.append(booster)

    return boosters


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate(
    boosters: List[xgb.Booster],
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for booster, name in zip(boosters, METRIC_NAMES):
        y_true = Y_test[:, METRIC_NAMES.index(name)]
        ok = np.isfinite(y_true)
        if not np.any(ok):
            rows.append(dict(metric=name, r2=np.nan, rmse=np.nan, mad=np.nan,
                             pearson_r=np.nan, n_test=0))
            continue
        y_pred = booster.predict(xgb.DMatrix(X_test[ok]))
        y_t = y_true[ok]
        r2 = float(r2_score(y_t, y_pred))
        rmse = float(np.sqrt(np.mean((y_t - y_pred) ** 2)))
        mad = float(np.median(np.abs(y_t - y_pred)))
        pearson_r = float(np.corrcoef(y_t, y_pred)[0, 1])
        rows.append(dict(metric=name, r2=r2, rmse=rmse, mad=mad,
                         pearson_r=pearson_r, n_test=int(ok.sum())))
    return pd.DataFrame(rows)


# ======================================================================
# PLOTS
# ======================================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scatter(
    boosters: List[xgb.Booster],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    ncols = 4
    nrows = (len(METRIC_NAMES) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, (booster, name) in enumerate(zip(boosters, METRIC_NAMES)):
        ax = axes[i]
        y_true = Y_test[:, i]
        ok = np.isfinite(y_true)
        if not np.any(ok):
            ax.set_visible(False)
            continue
        y_pred = booster.predict(xgb.DMatrix(X_test[ok]))
        y_t = y_true[ok]
        row = summary[summary["metric"] == name].iloc[0]

        ax.scatter(y_t, y_pred, s=2, alpha=0.15, rasterized=True, color="steelblue")
        mn = min(y_t.min(), y_pred.min())
        mx = max(y_t.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="1:1")
        ax.set_xlabel("True", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.set_title(
            f"{name}\nR²={row['r2']:.3f}  r={row['pearson_r']:.3f}",
            fontsize=9,
        )

    for j in range(len(METRIC_NAMES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Morphology regressors: Predicted vs True (test set)", fontsize=12, y=1.01
    )
    fig.tight_layout()
    savefig(fig, out_path)


def plot_snr_diagnostic(
    snr: np.ndarray, splits: np.ndarray, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for code, label, color in [(0, "train", "steelblue"), (1, "val", "orange"), (2, "test", "green")]:
        mask = splits == code
        vals = snr[mask]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        ax.hist(
            np.log10(vals), bins=60, histtype="step",
            label=f"{label} (n={mask.sum()})", color=color,
        )
    ax.set_xlabel("log₁₀(SNR)  [positive_integral / border_sigma]")
    ax.set_ylabel("Count")
    ax.set_title("Stamp SNR diagnostic (border-MAD noise estimate)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_path)


def plot_metric_distributions(
    Y: np.ndarray, splits: np.ndarray, out_path: Path
) -> None:
    ncols = 4
    nrows = (len(METRIC_NAMES) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for i, name in enumerate(METRIC_NAMES):
        ax = axes[i]
        vals = Y[:, i]
        ok = np.isfinite(vals)
        ax.hist(
            vals[ok & (splits == 0)], bins=60, histtype="step",
            label="train", color="steelblue", density=True,
        )
        ax.hist(
            vals[ok & (splits == 2)], bins=60, histtype="step",
            label="test", color="green", density=True,
        )
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Value", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(METRIC_NAMES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Morphology metric distributions: train vs test", fontsize=12, y=1.01)
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    cfg = CFG
    feature_cols = get_feature_cols(cfg.feature_set)

    print("=" * 70)
    print(f"RUN            : {cfg.run_name}")
    print(f"Feature set    : {cfg.feature_set}  ({len(feature_cols)} features)")
    print(f"Mag filter     : phot_g_mean_mag >= {cfg.gaia_g_mag_min}")
    print(f"Metrics        : {METRIC_NAMES}")
    print(f"Output         : {cfg.output_root}")
    print("=" * 70)

    models_dir = cfg.output_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    cfg_dict = {k: str(v) for k, v in asdict(cfg).items()}
    with open(cfg.output_root / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # ---- Load data ----
    print("\n--- Loading data and computing metrics ---")
    X, Y, splits, snr = load_all_data(cfg, feature_cols)

    X_train, Y_train = X[splits == 0], Y[splits == 0]
    X_val,   Y_val   = X[splits == 1], Y[splits == 1]
    X_test,  Y_test  = X[splits == 2], Y[splits == 2]
    print(f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ---- Train ----
    print("\n--- Training 11 regressors ---")
    boosters = train_regressors(X_train, Y_train, X_val, Y_val, cfg, models_dir)

    # ---- Evaluate ----
    print("\n--- Evaluating on test set ---")
    summary = evaluate(boosters, X_test, Y_test)
    print("\nTest set summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(cfg.output_root / "metrics_summary.csv", index=False)
    print(f"\nSaved: {cfg.output_root / 'metrics_summary.csv'}")

    # ---- Plots ----
    print("\n--- Generating plots ---")
    plot_scatter(
        boosters, X_test, Y_test, summary,
        cfg.plots_root / "01_scatter_predicted_vs_true.png",
    )
    plot_snr_diagnostic(snr, splits, cfg.plots_root / "02_snr_diagnostic.png")
    plot_metric_distributions(Y, splits, cfg.plots_root / "03_metric_distributions.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
