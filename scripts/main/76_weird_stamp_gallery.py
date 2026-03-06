"""
76_weird_stamp_gallery.py
=========================

Visual gallery of Euclid stamps at the extremes of morphology metric space,
cross-checked with Gaia quality flags to distinguish physically-weird sources
from image artefacts.

For each of gini and kurtosis (the two best regressors):
  - Top 2% true value   (high-gini / high-kurtosis)
  - Bottom 2% true value (low-gini  / low-kurtosis)

Stamps are annotated with:
  true metric value | predicted value | RUWE | ipd_frac_multi_peak | aen_sig

Also generates a "largest errors" gallery for gini: objects where the
prediction is most wrong (largest |true - pred|), to see what trips the model.

Outputs (in plots/ml_runs/morph_metrics_16d/):
  15_gallery_gini_extremes.png
  16_gallery_kurtosis_extremes.png
  17_gallery_gini_worst_errors.png
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb

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

# Raw Gaia columns we want to annotate (not in 16D scaled features)
# RUWE is in 16D (feat_ruwe index=1) but we want the raw value for display
RAW_ANNOT_COLS = [
    "ruwe",
    "ipd_frac_multi_peak",
    "astrometric_excess_noise",
]

N_PER_PANEL = 15    # stamps to show per extreme (top/bottom/error)
PCT_EXTREME  = 2.0  # percentile cutoff for "extreme"


# ======================================================================
# DATA LOADING WITH PROVENANCE
# ======================================================================

def load_data_with_provenance(
    cfg, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Like load_all_data but also records provenance (field, npz_file, index_in_file)
    and raw annotation columns (ruwe, ipd_frac_multi_peak, astrometric_excess_noise).

    Returns: X, Y, splits, prov_df
      prov_df columns: field_name, npz_file, index_in_file, ruwe,
                       ipd_frac_multi_peak, astrometric_excess_noise
    """
    gaia_min, gaia_iqr = _mod.load_gaia_scaler(cfg.gaia_scaler_npz, feature_cols)
    field_dirs = _mod.list_field_dirs(cfg.dataset_root)
    print(f"Found {len(field_dirs)} field directories.")

    all_X, all_Y, all_splits = [], [], []
    prov_rows = []

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
            continue

        # Load raw annotation columns (best-effort)
        annot = {}
        for col in RAW_ANNOT_COLS:
            annot[col] = (
                pd.to_numeric(meta[col], errors="coerce").to_numpy()
                if col in meta.columns
                else np.full(nrows, np.nan)
            )

        # Load stamps
        all_stamps_raw = np.full((nrows, 20, 20), np.nan, dtype=np.float32)
        npz_file_col  = np.full(nrows, "", dtype=object)
        idx_in_file   = np.full(nrows, -1, dtype=np.int32)

        for npz_name, group in meta.groupby("npz_file"):
            npz_path = fdir / str(npz_name)
            if not npz_path.exists():
                continue
            with np.load(npz_path) as d:
                data = d["X"].astype(np.float32)
            file_idx  = pd.to_numeric(
                group["index_in_file"], errors="coerce"
            ).to_numpy(dtype=float)
            meta_idx  = group.index.to_numpy()
            ok_f = (
                np.isfinite(file_idx)
                & (file_idx >= 0)
                & (file_idx < len(data))
            )
            all_stamps_raw[meta_idx[ok_f]] = data[file_idx[ok_f].astype(int)]
            npz_file_col[meta_idx[ok_f]]   = str(npz_name)
            idx_in_file[meta_idx[ok_f]]    = file_idx[ok_f].astype(int)

        # Filter border-clipped stamps
        border_bad = np.array([
            _mod.is_border_stamp(all_stamps_raw[i]) if valid[i] else False
            for i in range(nrows)
        ])
        valid &= ~border_bad

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
            iv = float(integ[i])
            bg_raw, _ = _mod.border_stats(all_stamps_raw[i])
            bg_norm   = bg_raw / iv if iv > cfg.eps_integral else 0.0
            m = _mod.compute_all_metrics(stamps_norm[i], bg_norm, cfg)
            metrics_list.append([m[k] for k in METRIC_NAMES])
        print(f"  Done in {time.time() - t0:.1f}s")

        Y_field      = np.array(metrics_list, dtype=np.float32)
        X_field      = (
            (feat_raw[valid_idx].astype(np.float32) - gaia_min)
            / np.where(gaia_iqr > 0, gaia_iqr, 1.0)
        )
        splits_field = splits_raw[valid_idx].astype(int)

        ok_m = np.all(np.isfinite(Y_field), axis=1)
        if not ok_m.all():
            print(f"  Dropped {(~ok_m).sum()} metric-NaN rows.")
        Y_field      = Y_field[ok_m]
        X_field      = X_field[ok_m]
        splits_field = splits_field[ok_m]
        kept_idx     = valid_idx[ok_m]

        # Provenance rows
        for k in kept_idx:
            row = {
                "field_name":   field_tag,
                "npz_file":     npz_file_col[k],
                "index_in_file": int(idx_in_file[k]),
            }
            for col in RAW_ANNOT_COLS:
                row[col] = float(annot[col][k])
            prov_rows.append(row)

        all_X.append(X_field)
        all_Y.append(Y_field)
        all_splits.append(splits_field)

    X      = np.concatenate(all_X)
    Y      = np.concatenate(all_Y)
    splits = np.concatenate(all_splits)
    prov   = pd.DataFrame(prov_rows)
    print(f"\nTotal: {len(X)}  (train={np.sum(splits==0)}, val={np.sum(splits==1)}, test={np.sum(splits==2)})")
    return X, Y, splits, prov


# ======================================================================
# STAMP LOADING
# ======================================================================

def load_stamp(row: pd.Series) -> np.ndarray:
    """Load one raw 20×20 stamp given a provenance row."""
    fdir = CFG.dataset_root
    if not (fdir / _mod.RAW_META_16D).exists():
        fdir = fdir / row["field_name"]
    npz_path = fdir / row["npz_file"]
    with np.load(npz_path) as d:
        return d["X"][int(row["index_in_file"])].astype(np.float32)


def normalize_stamp(raw: np.ndarray) -> np.ndarray:
    """Positive-integral normalization for display."""
    pos = np.clip(raw, 0, None)
    total = pos.sum()
    return raw / total if total > 0 else raw


# ======================================================================
# GALLERY PLOT
# ======================================================================

def plot_gallery(
    stamps: List[np.ndarray],         # normalized stamps
    titles: List[str],                # one-line title per stamp
    subtitles: List[str],             # two-line annotation per stamp
    suptitle: str,
    out_path: Path,
    ncols: int = 5,
) -> None:
    n = len(stamps)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 3.6 * nrows))
    axes = np.array(axes).flatten()

    for k, (stamp, title, subtitle) in enumerate(zip(stamps, titles, subtitles)):
        ax = axes[k]
        vmax = np.percentile(np.abs(stamp), 99)
        ax.imshow(stamp, origin="lower", cmap="viridis",
                  vmin=-vmax * 0.1, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=8, fontweight="bold", pad=2)
        ax.text(0.5, -0.08, subtitle, transform=ax.transAxes,
                fontsize=6.5, ha="center", va="top",
                color="dimgrey")
        ax.axis("off")

    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()
    savefig(fig, out_path)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ======================================================================
# BUILD GALLERY FOR ONE METRIC
# ======================================================================

def build_extreme_gallery(
    metric_name: str,
    metric_idx: int,
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    prov_test: pd.DataFrame,
    out_path: Path,
) -> None:
    y_true = Y_test[:, metric_idx]
    ok     = np.isfinite(y_true)
    idxs   = np.where(ok)[0]

    vals_sorted = np.argsort(y_true[idxs])
    n_pick = min(N_PER_PANEL, len(idxs) // 2)

    bottom_idxs = idxs[vals_sorted[:n_pick]]
    top_idxs    = idxs[vals_sorted[-n_pick:][::-1]]

    all_idxs  = list(top_idxs) + list(bottom_idxs)
    kind_tags = ["HIGH"] * n_pick + ["LOW"] * n_pick

    stamps, titles, subtitles = [], [], []
    for i, kind in zip(all_idxs, kind_tags):
        row = prov_test.iloc[i]
        try:
            raw  = load_stamp(row)
            norm = normalize_stamp(raw)
        except Exception as e:
            print(f"  [WARN] Could not load stamp {i}: {e}")
            norm = np.zeros((20, 20))
        stamps.append(norm)

        pred = float(Y_pred[i, metric_idx]) if np.isfinite(Y_pred[i, metric_idx]) else float("nan")
        ruwe = row.get("ruwe", np.nan)
        ipd  = row.get("ipd_frac_multi_peak", np.nan)
        aen  = row.get("astrometric_excess_noise", np.nan)

        titles.append(f"{kind}  true={y_true[i]:.3f}\npred={pred:.3f}")
        subtitles.append(
            f"RUWE={ruwe:.2f}  ipd_frac={ipd:.0f}%  AEN={aen:.2f}"
        )

    n_high = n_pick
    suptitle = (
        f"{metric_name} extremes  —  "
        f"top {PCT_EXTREME:.0f}%  (rows 1–{n_high})  |  "
        f"bottom {PCT_EXTREME:.0f}%  (rows {n_high+1}–{2*n_high})"
    )
    plot_gallery(stamps, titles, subtitles, suptitle, out_path, ncols=5)


def build_error_gallery(
    metric_name: str,
    metric_idx: int,
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    prov_test: pd.DataFrame,
    out_path: Path,
) -> None:
    y_true = Y_test[:, metric_idx]
    y_pred = Y_pred[:, metric_idx]
    ok     = np.isfinite(y_true) & np.isfinite(y_pred)
    idxs   = np.where(ok)[0]

    errors = np.abs(y_true[idxs] - y_pred[idxs])
    order  = np.argsort(errors)[::-1]
    n_pick = min(N_PER_PANEL, len(idxs))
    worst  = idxs[order[:n_pick]]

    stamps, titles, subtitles = [], [], []
    for i in worst:
        row = prov_test.iloc[i]
        try:
            raw  = load_stamp(row)
            norm = normalize_stamp(raw)
        except Exception as e:
            print(f"  [WARN] Could not load stamp {i}: {e}")
            norm = np.zeros((20, 20))
        stamps.append(norm)

        err  = float(y_true[i] - y_pred[i])
        ruwe = row.get("ruwe", np.nan)
        ipd  = row.get("ipd_frac_multi_peak", np.nan)
        aen  = row.get("astrometric_excess_noise", np.nan)

        titles.append(f"true={y_true[i]:.3f}  pred={y_pred[i]:.3f}\nerr={err:+.3f}")
        subtitles.append(
            f"RUWE={ruwe:.2f}  ipd_frac={ipd:.0f}%  AEN={aen:.2f}"
        )

    plot_gallery(
        stamps, titles, subtitles,
        f"{metric_name} — largest prediction errors  (|true − pred|)",
        out_path,
        ncols=5,
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    feature_cols = get_feature_cols(CFG.feature_set)

    # ---- Load data with provenance ----
    print("Loading data with provenance ...")
    X, Y, splits, prov = load_data_with_provenance(CFG, feature_cols)

    X_test    = X[splits == 2]
    Y_test    = Y[splits == 2]
    prov_test = prov[splits == 2].reset_index(drop=True)
    print(f"\nTest set: {len(X_test)} objects")

    # ---- Load boosters ----
    print("\nLoading boosters ...")
    fnames = [f"f{i}" for i in range(len(feature_cols))]
    boosters = []
    for name in METRIC_NAMES:
        b = xgb.Booster()
        b.load_model(str(MODELS_DIR / f"booster_{name}.json"))
        b.feature_names = fnames
        boosters.append(b)

    Y_pred = np.column_stack([
        b.predict(xgb.DMatrix(X_test, feature_names=fnames))
        for b in boosters
    ])

    # ---- Galleries ----
    print("\nBuilding gini extremes gallery ...")
    gini_idx = METRIC_NAMES.index("gini")
    build_extreme_gallery(
        "gini", gini_idx, Y_test, Y_pred, prov_test,
        PLOTS_DIR / "15_gallery_gini_extremes.png",
    )

    print("Building kurtosis extremes gallery ...")
    kurt_idx = METRIC_NAMES.index("kurtosis")
    build_extreme_gallery(
        "kurtosis", kurt_idx, Y_test, Y_pred, prov_test,
        PLOTS_DIR / "16_gallery_kurtosis_extremes.png",
    )

    print("Building gini worst-errors gallery ...")
    build_error_gallery(
        "gini", gini_idx, Y_test, Y_pred, prov_test,
        PLOTS_DIR / "17_gallery_gini_worst_errors.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
