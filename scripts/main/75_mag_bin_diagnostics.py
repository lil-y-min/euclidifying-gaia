"""
75_mag_bin_diagnostics.py
=========================

Split the test set into three equal-population magnitude bins (tertiles)
and evaluate morphology regressor performance within each bin.

phot_g_mean_mag is not stored in the cached 16D feature arrays, so this
script re-runs data loading (same logic as script 69) while also collecting
the raw magnitude alongside X / Y / splits.

Plots:
  13_mag_bin_r2_heatmap.png  — R² heatmap: 3 mag bins × 11 metrics
  14_mag_bin_density_top3.png — density scatter (gini, kurtosis, ellipticity)
                                 across 3 mag bins  (3 rows × 3 cols)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score

import importlib
_mod = importlib.import_module("69_morph_metrics_xgb")
CFG          = _mod.CFG
METRIC_NAMES = _mod.METRIC_NAMES

from feature_schema import get_feature_cols


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR   = CFG.base_dir
MODELS_DIR = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "models"
PLOTS_DIR  = BASE_DIR / "plots"  / "ml_runs" / "morph_metrics_16d"

# Cache for this script (to avoid reloading on reruns)
MAG_CACHE_DIR = BASE_DIR / "output" / "ml_runs" / "morph_metrics_16d" / "mag_cache"


# ======================================================================
# DATA LOADING (extends script 69's load_all_data to also return mag)
# ======================================================================

def load_all_data_with_mag(
    cfg, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Like 69's load_all_data but also returns phot_g_mean_mag (unscaled).
    Returns: X, Y, splits, mag  (all with same length N)
    """
    # Check cache
    MAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_files = {
        k: MAG_CACHE_DIR / f"{k}.npy"
        for k in ["X", "Y", "splits", "mag"]
    }
    if all(f.exists() for f in cache_files.values()):
        print("Loading cached mag data ...")
        return tuple(np.load(cache_files[k]) for k in ["X", "Y", "splits", "mag"])

    print("Cache not found — re-running data loading (this takes ~3 min) ...")
    gaia_min, gaia_iqr = _mod.load_gaia_scaler(cfg.gaia_scaler_npz, feature_cols)
    field_dirs = _mod.list_field_dirs(cfg.dataset_root)
    print(f"Found {len(field_dirs)} field directories.")

    all_X, all_Y, all_splits, all_mag = [], [], [], []

    for fdir in field_dirs:
        field_tag = fdir.name
        meta_path = fdir / _mod.RAW_META_16D
        valid_path = fdir / _mod.VALID_NAME

        print(f"\n[{field_tag}]")
        meta  = pd.read_csv(meta_path, engine="python", on_bad_lines="skip")
        nrows = len(meta)

        valid = (
            _mod.load_valid_mask(valid_path, nrows)
            if valid_path.exists()
            else np.ones(nrows, dtype=bool)
        )

        g = pd.to_numeric(meta["phot_g_mean_mag"], errors="coerce").to_numpy()
        valid &= np.isfinite(g) & (g >= cfg.gaia_g_mag_min)

        feat_raw = meta[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid &= np.all(np.isfinite(feat_raw), axis=1)

        splits_raw = pd.to_numeric(meta["split_code"], errors="coerce").to_numpy()
        valid &= np.isfinite(splits_raw) & np.isin(splits_raw, [0, 1, 2])

        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            print("  No valid rows — skipping.")
            continue

        # Load stamps
        all_stamps_raw = np.full((nrows, 20, 20), np.nan, dtype=np.float32)
        for npz_name, group in meta.groupby("npz_file"):
            npz_path = fdir / str(npz_name)
            if not npz_path.exists():
                continue
            with np.load(npz_path) as d:
                data = d["X"].astype(np.float32)
            file_idx = pd.to_numeric(
                group["index_in_file"], errors="coerce"
            ).to_numpy(dtype=float)
            meta_idx = group.index.to_numpy()
            ok_f = (
                np.isfinite(file_idx)
                & (file_idx >= 0)
                & (file_idx < len(data))
            )
            all_stamps_raw[meta_idx[ok_f]] = data[file_idx[ok_f].astype(int)]

        # Filter border-clipped stamps
        border_bad = np.array([
            _mod.is_border_stamp(all_stamps_raw[i]) if valid[i] else False
            for i in range(nrows)
        ])
        valid &= ~border_bad
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            continue

        stamps_norm, norm_ok, integ = _mod.normalize_stamps(
            all_stamps_raw,
            use_positive_only=cfg.use_positive_only,
            eps=cfg.eps_integral,
        )
        valid &= norm_ok
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            continue

        # Compute metrics
        print(f"  Computing metrics for {len(valid_idx)} stamps ...", flush=True)
        t0 = time.time()
        metrics_list = []
        for i in valid_idx:
            s_raw  = all_stamps_raw[i]
            s_norm = stamps_norm[i]
            iv     = float(integ[i])
            bg_raw, _ = _mod.border_stats(s_raw)
            bg_norm = bg_raw / iv if iv > cfg.eps_integral else 0.0
            m = _mod.compute_all_metrics(s_norm, bg_norm, cfg)
            metrics_list.append([m[k] for k in METRIC_NAMES])
        print(f"  Done in {time.time() - t0:.1f}s")

        Y_field = np.array(metrics_list, dtype=np.float32)
        X_field = (
            (feat_raw[valid_idx].astype(np.float32) - gaia_min)
            / np.where(gaia_iqr > 0, gaia_iqr, 1.0)
        )
        splits_field = splits_raw[valid_idx].astype(int)
        mag_field    = g[valid_idx].astype(np.float32)

        ok_m = np.all(np.isfinite(Y_field), axis=1)
        if not ok_m.all():
            print(f"  Dropped {(~ok_m).sum()} rows with NaN metrics.")
        Y_field      = Y_field[ok_m]
        X_field      = X_field[ok_m]
        splits_field = splits_field[ok_m]
        mag_field    = mag_field[ok_m]

        all_X.append(X_field)
        all_Y.append(Y_field)
        all_splits.append(splits_field)
        all_mag.append(mag_field)

    X      = np.concatenate(all_X)
    Y      = np.concatenate(all_Y)
    splits = np.concatenate(all_splits)
    mag    = np.concatenate(all_mag)

    for k, arr in zip(["X", "Y", "splits", "mag"], [X, Y, splits, mag]):
        np.save(cache_files[k], arr)
    print(f"\nCached to {MAG_CACHE_DIR}")
    print(f"Total: {len(X)}  (train={np.sum(splits==0)}, val={np.sum(splits==1)}, test={np.sum(splits==2)})")
    return X, Y, splits, mag


# ======================================================================
# BOOSTER LOADING & EVALUATION
# ======================================================================

def load_boosters(feature_cols: List[str]) -> List[xgb.Booster]:
    fnames = [f"f{i}" for i in range(len(feature_cols))]
    boosters = []
    for name in METRIC_NAMES:
        b = xgb.Booster()
        b.load_model(str(MODELS_DIR / f"booster_{name}.json"))
        b.feature_names = fnames
        boosters.append(b)
    return boosters


def predict_all(
    boosters: List[xgb.Booster],
    X: np.ndarray,
    feature_cols: List[str],
) -> np.ndarray:
    fnames = [f"f{i}" for i in range(len(feature_cols))]
    dm = xgb.DMatrix(X, feature_names=fnames)
    return np.column_stack([b.predict(dm) for b in boosters])


def r2_per_metric(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    r2s = []
    for i in range(len(METRIC_NAMES)):
        ok = np.isfinite(Y_true[:, i])
        if ok.sum() < 5:
            r2s.append(np.nan)
        else:
            r2s.append(float(r2_score(Y_true[ok, i], Y_pred[ok, i])))
    return np.array(r2s)


# ======================================================================
# HELPERS
# ======================================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ======================================================================
# PLOT A — R² heatmap  (3 mag bins × 11 metrics)
# ======================================================================

def plot_r2_heatmap(
    r2_matrix: np.ndarray,   # shape (3, 11)
    bin_labels: List[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.5))

    vmin = max(0, np.nanmin(r2_matrix))
    vmax = min(1, np.nanmax(r2_matrix))
    im = ax.imshow(r2_matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=max(vmax, 0.6))
    plt.colorbar(im, ax=ax, label="R²", shrink=0.85)

    ax.set_xticks(range(len(METRIC_NAMES)))
    ax.set_xticklabels(METRIC_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels(bin_labels, fontsize=9)

    for row in range(3):
        for col in range(len(METRIC_NAMES)):
            val = r2_matrix[row, col]
            color = "white" if val < 0.25 or val > 0.75 else "black"
            ax.text(col, row, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title(
        "R² per morphology metric split by magnitude tertile  "
        "(brighter → fainter)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# PLOT B — density scatter for top 3 regressors × 3 mag bins
# ======================================================================

def _density_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2: float,
    n_bins: int = 50,
    vmax_pct: int = 95,
) -> None:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    lo = min(np.percentile(yt, 1), np.percentile(yp, 1))
    hi = max(np.percentile(yt, 99), np.percentile(yp, 99))

    H, _, _ = np.histogram2d(yt, yp, bins=n_bins, range=[[lo, hi], [lo, hi]])
    H = np.log1p(H)
    vmax = np.percentile(H[H > 0], vmax_pct) if np.any(H > 0) else H.max()

    im = ax.imshow(H.T, origin="lower", aspect="auto",
                   extent=[lo, hi, lo, hi], cmap="viridis",
                   interpolation="nearest", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046, shrink=0.8,
                 label="log(1+N)")
    ax.plot([lo, hi], [lo, hi], "r--", lw=0.8, alpha=0.7)
    ax.set_xlabel("True", fontsize=7)
    ax.set_ylabel("Predicted", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.text(0.05, 0.92, f"R²={r2:.3f}  n={ok.sum()}",
            transform=ax.transAxes, fontsize=7, color="white",
            bbox=dict(fc="black", alpha=0.4, pad=1))


def plot_density_top3(
    Y_test_list: List[np.ndarray],
    Y_pred_list: List[np.ndarray],
    r2_matrix: np.ndarray,    # (3, 11)
    bin_labels: List[str],
    out_path: Path,
) -> None:
    top3_metrics = ["gini", "kurtosis", "ellipticity"]
    top3_idx     = [METRIC_NAMES.index(m) for m in top3_metrics]

    nrows, ncols = 3, 3   # mag bins × metrics
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))

    for col, (metric, midx) in enumerate(zip(top3_metrics, top3_idx)):
        for row, (Y_t, Y_p, blabel) in enumerate(
            zip(Y_test_list, Y_pred_list, bin_labels)
        ):
            ax = axes[row, col]
            r2 = r2_matrix[row, midx]
            _density_panel(ax, Y_t[:, midx], Y_p[:, midx], r2)

            if row == 0:
                ax.set_title(metric, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{blabel}\nPredicted", fontsize=7)

    fig.suptitle(
        "Density scatter: top 3 regressors × magnitude bin  (p95 vmax)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    feature_cols = get_feature_cols(CFG.feature_set)

    # ---- Data ----
    X, Y, splits, mag = load_all_data_with_mag(CFG, feature_cols)
    X_test  = X[splits == 2]
    Y_test  = Y[splits == 2]
    mag_test = mag[splits == 2]
    print(f"\nTest set: {len(X_test)} objects")

    # ---- Magnitude tertiles ----
    p33 = float(np.percentile(mag_test, 33.33))
    p67 = float(np.percentile(mag_test, 66.67))
    bin_masks = [
        mag_test <= p33,
        (mag_test > p33) & (mag_test <= p67),
        mag_test > p67,
    ]
    mag_lo = float(mag_test.min())
    mag_hi = float(mag_test.max())
    bin_labels = [
        f"Bright  {mag_lo:.1f}–{p33:.1f}",
        f"Mid     {p33:.1f}–{p67:.1f}",
        f"Faint   {p67:.1f}–{mag_hi:.1f}",
    ]
    for label, mask in zip(bin_labels, bin_masks):
        print(f"  {label}  n={mask.sum()}")

    # ---- Boosters ----
    print("\nLoading boosters ...")
    boosters = load_boosters(feature_cols)

    # ---- Evaluate per bin ----
    r2_matrix   = np.zeros((3, len(METRIC_NAMES)))
    Y_test_list, Y_pred_list = [], []

    for b_idx, mask in enumerate(bin_masks):
        X_b = X_test[mask]
        Y_b = Y_test[mask]
        Y_p = predict_all(boosters, X_b, feature_cols)
        r2_matrix[b_idx] = r2_per_metric(Y_b, Y_p)
        Y_test_list.append(Y_b)
        Y_pred_list.append(Y_p)

    # ---- Summary ----
    df = pd.DataFrame(r2_matrix, columns=METRIC_NAMES, index=bin_labels)
    df.index.name = "mag_bin"
    print("\n=== R² PER MAG BIN ===")
    print(df.round(3).to_string())

    # ---- Plots ----
    print("\nGenerating plots ...")
    plot_r2_heatmap(
        r2_matrix, bin_labels,
        PLOTS_DIR / "13_mag_bin_r2_heatmap.png",
    )
    plot_density_top3(
        Y_test_list, Y_pred_list, r2_matrix, bin_labels,
        PLOTS_DIR / "14_mag_bin_density_top3.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
