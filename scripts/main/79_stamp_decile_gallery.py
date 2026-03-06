"""
79_stamp_decile_gallery.py
==========================

For each of gini, kurtosis, ellipticity, asymmetry_180:
  Sort the test set by true metric value, sample one stamp per decile
  (10 stamps from very low to very high), display in a row.

Result: 4 rows × 10 stamps showing the visual meaning of each metric.
Makes the morphology metrics intuitively interpretable.

Outputs:
  plots/ml_runs/morph_metrics_16d/
    23_stamp_decile_gallery.png
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

import importlib
_mod = importlib.import_module("69_morph_metrics_xgb")
CFG          = _mod.CFG
METRIC_NAMES = _mod.METRIC_NAMES

from feature_schema import get_feature_cols


# ======================================================================
# CONFIG
# ======================================================================

PLOTS_DIR = CFG.base_dir / "plots" / "ml_runs" / "morph_metrics_16d"
N_DECILES = 10
METRICS_TO_SHOW = ["gini", "kurtosis", "ellipticity", "asymmetry_180"]


# ======================================================================
# DATA LOADING WITH PROVENANCE  (mirrors script 76)
# ======================================================================

def load_test_with_provenance(cfg, feature_cols):
    """Load test set: returns Y_test, stamps_norm, prov info."""
    field_dirs = _mod.list_field_dirs(cfg.dataset_root)
    gaia_min, gaia_iqr = _mod.load_gaia_scaler(cfg.gaia_scaler_npz, feature_cols)

    Y_all, stamps_all, splits_all = [], [], []

    for fdir in field_dirs:
        field_tag = fdir.name
        meta_path = fdir / _mod.RAW_META_16D
        valid_path = fdir / _mod.VALID_NAME

        import pandas as pd
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
            ok_f = np.isfinite(file_idx) & (file_idx >= 0) & (file_idx < len(data))
            all_stamps_raw[meta_idx[ok_f]] = data[file_idx[ok_f].astype(int)]

        # Border filter
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

        print(f"  [{field_tag}] computing metrics for {len(valid_idx)} stamps ...", flush=True)
        t0 = time.time()
        metrics_list = []
        for i in valid_idx:
            iv = float(integ[i])
            bg_raw, _ = _mod.border_stats(all_stamps_raw[i])
            bg_norm   = bg_raw / iv if iv > cfg.eps_integral else 0.0
            m = _mod.compute_all_metrics(stamps_norm[i], bg_norm, cfg)
            metrics_list.append([m[k] for k in METRIC_NAMES])
        print(f"    done in {time.time()-t0:.1f}s")

        Y_field      = np.array(metrics_list, dtype=np.float32)
        splits_field = splits_raw[valid_idx].astype(int)
        stamps_field = stamps_norm[valid_idx]  # normalized stamps

        ok_m = np.all(np.isfinite(Y_field), axis=1)
        Y_field      = Y_field[ok_m]
        splits_field = splits_field[ok_m]
        stamps_field = stamps_field[ok_m]

        Y_all.append(Y_field)
        stamps_all.append(stamps_field)
        splits_all.append(splits_field)

    Y      = np.concatenate(Y_all)
    stamps = np.concatenate(stamps_all)
    splits = np.concatenate(splits_all)

    mask = splits == 2
    return Y[mask], stamps[mask]


# ======================================================================
# PLOT
# ======================================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_decile_gallery(
    Y_test: np.ndarray,
    stamps: np.ndarray,
    out_path: Path,
) -> None:
    nrows = len(METRICS_TO_SHOW)
    ncols = N_DECILES
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.8 * nrows))

    rng = np.random.default_rng(42)

    for row, metric_name in enumerate(METRICS_TO_SHOW):
        midx = METRIC_NAMES.index(metric_name)
        vals = Y_test[:, midx]
        ok   = np.isfinite(vals)
        vals_ok = vals[ok]
        idx_ok  = np.where(ok)[0]

        # Sort and split into N_DECILES equal groups
        order   = np.argsort(vals_ok)
        n       = len(order)
        groups  = np.array_split(order, N_DECILES)

        for col, group in enumerate(groups):
            ax = axes[row, col]
            # Pick one stamp randomly from this decile group
            pick  = int(rng.choice(group))
            stamp = stamps[idx_ok[pick]]
            val   = vals_ok[pick]

            vmax = np.percentile(np.abs(stamp), 99.5)
            ax.imshow(stamp, origin="lower", cmap="viridis",
                      vmin=-0.05 * vmax, vmax=vmax, interpolation="nearest")
            ax.axis("off")

            # Top: decile label; bottom: metric value
            if row == 0:
                ax.set_title(f"D{col+1}", fontsize=8, pad=2)
            ax.text(0.5, -0.06, f"{val:.3f}", transform=ax.transAxes,
                    fontsize=7, ha="center", va="top", color="dimgrey")

        # Row label on left
        axes[row, 0].set_ylabel(metric_name, fontsize=9, labelpad=4)
        axes[row, 0].yaxis.label.set_visible(True)
        fig.text(0.01, 1 - (row + 0.5) / nrows,
                 metric_name, va="center", ha="left",
                 fontsize=10, fontweight="bold", rotation=0)

    fig.suptitle(
        "Euclid stamps sorted by morphology metric value — decile D1 (low) → D10 (high)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, out_path)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    feature_cols = get_feature_cols(CFG.feature_set)
    print("Loading test set with provenance ...")
    Y_test, stamps = load_test_with_provenance(CFG, feature_cols)
    print(f"\nTest set: {len(Y_test)} objects, stamps shape: {stamps.shape}")

    print("\nGenerating decile gallery ...")
    plot_decile_gallery(Y_test, stamps, PLOTS_DIR / "23_stamp_decile_gallery.png")
    print("Done.")


if __name__ == "__main__":
    main()
