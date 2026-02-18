"""
15_eval_xgb_reconstruct.py
=========================

Evaluate a trained XGBoost->PCA-coeff model (trained by Code 14),
reconstruct Euclid stamps from predicted PCA coefficients, compute per-sample errors,
export outlier indices + outlier trace CSVs, and generate a rich diagnostic plot set.

Outputs:
.../Codes/plots/ml_runs/<RUN_NAME>/

Core artifacts (Step A -> Step B):
- outliers_indices_test.npy
- outliers_indices_val.npy
- outliers_trace_test.csv
- outliers_trace_val.csv

Diagnostics (similar to your previous screenshot):
- 01_rmse_per_component_test.png
- 01b_rmse_normalized_by_scatter_test.png
- 02_explainedvar_vs_rmse_test.png  (if PCA has EVR)
- 03_perstamp_pixel_rmse_hist.png   (test)
- 04_mean_maps_true_pred_absresid.png  (test, reconstructed space)
- 05_examples_true_pred_resid.png      (test, small montage)
- 06_error_vs_features_abs_spearman.png
- 06_error_vs_features_spearman_signed.png
- 07_radial_profile_mean_true_pred.png
- 08_feature_vs_error_hexbin_XX_<feat>.png  (test)

Notes:
- TRUE/PRED/RESID are in normalized PCA reconstruction space.
- RAW cutouts are handled by Code 16 using trace mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


# ======================================================================
# CONFIG
# ======================================================================

@dataclass
class EvalConfig:
    base_dir: Path = Path(__file__).resolve().parents[1]  # .../Codes
    run_name: str = "ml_xgb_8d"  # <-- CHANGE THIS

    top_n_test: int = 250
    top_n_val: int = 250

    batch_size: int = 200_000  # for prediction + reconstruction RMSE

    # For plots that need recon images, we subsample to avoid heavy compute
    mean_maps_max: int = 120_000        # how many test samples to include in mean maps
    examples_n: int = 16                # 4x4 montage
    radial_profile_max: int = 80_000    # samples for mean radial profile

    plots_root: Path = None  # default: .../Codes/plots/ml_runs

    def __post_init__(self):
        if self.plots_root is None:
            self.plots_root = self.base_dir / "plots" / "ml_runs"


CFG = EvalConfig()


# ======================================================================
# Helpers
# ======================================================================

def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def rmse_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    d = a - b
    return np.sqrt(np.mean(d * d, axis=1))


def load_manifest(run_root: Path) -> Dict:
    mpath = run_root / "manifest_arrays.npz"
    if not mpath.exists():
        raise FileNotFoundError(f"Missing manifest: {mpath}")
    d = np.load(mpath, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if "feature_cols" in out:
        out["feature_cols"] = [str(x) for x in out["feature_cols"].tolist()]
    for k in ["n_train", "n_val", "n_test", "n_features", "K", "D", "stamp_pix"]:
        if k in out:
            out[k] = int(out[k])
    for k in [
        "X_train_path", "Y_train_path", "X_val_path", "Y_val_path", "X_test_path", "Y_test_path",
        "pca_npz", "scaler_npz", "dataset_root", "trace_test_csv", "plots_dir"
    ]:
        if k in out:
            out[k] = str(out[k])
    return out


def load_pca_from_npz(pca_npz: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    d = np.load(pca_npz, allow_pickle=True)
    mean = d["mean"].astype(np.float32)
    components = d["components"].astype(np.float32)
    evr = d["explained_variance_ratio"].astype(np.float32) if "explained_variance_ratio" in d else None
    return mean, components, evr


def load_models(model_dir: Path, K: int) -> List[xgb.Booster]:
    boosters = []
    for j in range(K):
        p = model_dir / f"booster_comp_{j:03d}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")
        b = xgb.Booster()
        b.load_model(str(p))
        boosters.append(b)
    return boosters


def predict_coeffs_in_batches(boosters: List[xgb.Booster], X: np.memmap, batch_size: int) -> np.ndarray:
    N = X.shape[0]
    K = len(boosters)
    out = np.empty((N, K), dtype=np.float32)

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        Xb = np.asarray(X[s:e], dtype=np.float32, order="C")
        dmat = xgb.DMatrix(Xb)
        for j, booster in enumerate(boosters):
            out[s:e, j] = booster.predict(dmat).astype(np.float32, copy=False)

    return out


def reconstruct_flat(coeffs: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return coeffs @ components + mean[None, :]


def top_outliers(rmse_pix: np.ndarray, top_n: int) -> np.ndarray:
    N = rmse_pix.shape[0]
    top_n = int(min(max(top_n, 1), N))
    idx = np.argpartition(rmse_pix, -top_n)[-top_n:]
    idx = idx[np.argsort(rmse_pix[idx])[::-1]]
    return idx.astype(np.int64)


def write_outliers_trace(
    *,
    split_name: str,
    out_indices: np.ndarray,
    rmse_pix: np.ndarray,
    rmse_coeff: np.ndarray,
    trace_csv: Optional[Path],
    out_csv: Path,
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metrics = pd.DataFrame({
        f"{split_name}_index": out_indices,
        "rmse_pix": rmse_pix[out_indices],
        "rmse_coeff": rmse_coeff[out_indices],
        "rank": np.arange(len(out_indices), dtype=int),
    })

    if trace_csv is not None and trace_csv.exists():
        df_trace = pd.read_csv(trace_csv)
        key = "test_index" if split_name == "test" else "val_index"
        if key not in df_trace.columns:
            df_metrics.to_csv(out_csv, index=False)
            return
        df_join = df_trace.merge(df_metrics, left_on=key, right_on=f"{split_name}_index", how="inner")
        df_join = df_join.sort_values("rank", ascending=True)
        df_join.to_csv(out_csv, index=False)
    else:
        df_metrics.to_csv(out_csv, index=False)


# ---------- Plot helpers ----------

def plot_rmse_per_component(plot_dir: Path, rmse_components: np.ndarray, tag: str):
    plt.figure(figsize=(8.5, 4.6))
    plt.plot(np.arange(len(rmse_components)), rmse_components, marker="o", linewidth=1)
    plt.xlabel("PCA component index")
    plt.ylabel("RMSE (component)")
    plt.title(f"RMSE per PCA component ({tag})")
    savefig(plot_dir / f"01_rmse_per_component_{tag}.png")


def plot_rmse_normalized(plot_dir: Path, rmse_components: np.ndarray, true_coeff: np.ndarray, tag: str):
    # Normalize by coefficient scatter (std)
    std = np.std(true_coeff, axis=0)
    std = np.where(std > 0, std, np.nan)
    norm = rmse_components / std
    plt.figure(figsize=(8.5, 4.6))
    plt.plot(np.arange(len(norm)), norm, marker="o", linewidth=1)
    plt.xlabel("PCA component index")
    plt.ylabel("RMSE / std(component)")
    plt.title(f"RMSE normalized by coeff scatter ({tag})")
    savefig(plot_dir / f"01b_rmse_normalized_by_scatter_{tag}.png")


def plot_explainedvar_vs_rmse(plot_dir: Path, evr: np.ndarray, rmse_components: np.ndarray, tag: str):
    k = min(len(evr), len(rmse_components))
    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(evr[:k], rmse_components[:k], s=35)
    plt.xlabel("Explained variance ratio")
    plt.ylabel("RMSE (component)")
    plt.title(f"Explained variance vs RMSE ({tag})")
    savefig(plot_dir / f"02_explainedvar_vs_rmse_{tag}.png")


def mean_maps(true_coeff: np.ndarray, pred_coeff: np.ndarray, mean: np.ndarray, components: np.ndarray, stamp_pix: int,
              max_n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = true_coeff.shape[0]
    if N > max_n:
        idx = np.random.default_rng(123).choice(N, size=max_n, replace=False)
        tc = true_coeff[idx]
        pc = pred_coeff[idx]
    else:
        tc, pc = true_coeff, pred_coeff

    true_flat = reconstruct_flat(tc, mean, components)
    pred_flat = reconstruct_flat(pc, mean, components)
    resid_flat = np.abs(pred_flat - true_flat)

    tmean = true_flat.mean(axis=0).reshape(stamp_pix, stamp_pix)
    pmean = pred_flat.mean(axis=0).reshape(stamp_pix, stamp_pix)
    rmean = resid_flat.mean(axis=0).reshape(stamp_pix, stamp_pix)
    return tmean, pmean, rmean


def plot_mean_maps(plot_dir: Path, tmean: np.ndarray, pmean: np.ndarray, rmean: np.ndarray):
    # shared scaling for t/p
    vmin = float(np.percentile(np.concatenate([tmean.ravel(), pmean.ravel()]), 1))
    vmax = float(np.percentile(np.concatenate([tmean.ravel(), pmean.ravel()]), 99))

    fig = plt.figure(figsize=(10.5, 3.6))
    gs = fig.add_gridspec(1, 3, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    for ax in (ax1, ax2, ax3):
        ax.set_xticks([]); ax.set_yticks([])

    ax1.set_title("Mean TRUE (recon)", fontsize=10)
    ax2.set_title("Mean PRED (recon)", fontsize=10)
    ax3.set_title("Mean |RESID|", fontsize=10)

    ax1.imshow(tmean, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax2.imshow(pmean, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax3.imshow(rmean, origin="lower", cmap="magma")

    savefig(plot_dir / "04_mean_maps_true_pred_absresid.png")


def plot_examples(plot_dir: Path, true_coeff: np.ndarray, pred_coeff: np.ndarray, mean: np.ndarray, components: np.ndarray,
                  stamp_pix: int, rmse_pix: np.ndarray, n: int = 16):
    # pick top-n errors for visual examples
    idx = top_outliers(rmse_pix, n)
    tc = true_coeff[idx]
    pc = pred_coeff[idx]

    true_flat = reconstruct_flat(tc, mean, components)
    pred_flat = reconstruct_flat(pc, mean, components)
    resid_flat = pred_flat - true_flat

    # grid 4x4; for each example show TRUE/PRED/RESID in mini triplets would be huge
    # Instead: show RESID only (fast) + label rank
    # (RAW is handled by Code 16)
    side = int(math.sqrt(n))
    side = side if side * side == n else 4

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(side, side, wspace=0.15, hspace=0.25)
    for i in range(n):
        ax = fig.add_subplot(gs[i // side, i % side])
        ax.set_xticks([]); ax.set_yticks([])
        rimg = resid_flat[i].reshape(stamp_pix, stamp_pix)
        # symmetric scaling
        v = np.percentile(np.abs(rimg.ravel()), 99)
        v = float(v if v > 0 else 1e-6)
        ax.imshow(rimg, origin="lower", cmap="RdBu_r", vmin=-v, vmax=+v)
        ax.set_title(f"rank {i}", fontsize=9)
    fig.suptitle("Example residuals (top errors)", fontsize=12)
    savefig(plot_dir / "05_examples_true_pred_resid.png")


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    # Spearman via rank correlation (no scipy dependency)
    # returns nan if constant
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != y.size or x.size < 3:
        return float("nan")
    # rank
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    cx = rx - rx.mean()
    cy = ry - ry.mean()
    denom = np.sqrt(np.sum(cx * cx) * np.sum(cy * cy))
    if denom <= 0:
        return float("nan")
    return float(np.sum(cx * cy) / denom)


def plot_error_vs_features(plot_dir: Path, X: np.memmap, rmse_pix: np.ndarray, feature_cols: List[str], tag: str):
    # Subsample for speed
    N = X.shape[0]
    max_n = 400_000
    if N > max_n:
        idx = np.random.default_rng(123).choice(N, size=max_n, replace=False)
        Xs = np.asarray(X[idx], dtype=np.float32)
        es = rmse_pix[idx]
    else:
        Xs = np.asarray(X, dtype=np.float32)
        es = rmse_pix

    # signed + abs Spearman
    signed = []
    absd = []
    for j, name in enumerate(feature_cols):
        r = spearman_corr(Xs[:, j], es)
        signed.append(r)
        absd.append(abs(r) if np.isfinite(r) else np.nan)

    # signed plot
    plt.figure(figsize=(9.5, 4.8))
    plt.bar(np.arange(len(feature_cols)), signed)
    plt.xticks(np.arange(len(feature_cols)), feature_cols, rotation=45, ha="right")
    plt.ylabel("Spearman r (signed)")
    plt.title(f"Error vs features Spearman (signed) [{tag}]")
    savefig(plot_dir / f"06_error_vs_features_spearman_signed.png")

    # abs plot
    plt.figure(figsize=(9.5, 4.8))
    plt.bar(np.arange(len(feature_cols)), absd)
    plt.xticks(np.arange(len(feature_cols)), feature_cols, rotation=45, ha="right")
    plt.ylabel("|Spearman r|")
    plt.title(f"Error vs features Spearman (absolute) [{tag}]")
    savefig(plot_dir / f"06_error_vs_features_abs_spearman.png")


def plot_feature_vs_error_hexbin(plot_dir: Path, X: np.memmap, rmse_pix: np.ndarray, feature_cols: List[str], tag: str):
    # Use a smaller sample
    N = X.shape[0]
    max_n = 400_000
    if N > max_n:
        idx = np.random.default_rng(123).choice(N, size=max_n, replace=False)
        Xs = np.asarray(X[idx], dtype=np.float32)
        es = rmse_pix[idx]
    else:
        Xs = np.asarray(X, dtype=np.float32)
        es = rmse_pix

    for j, name in enumerate(feature_cols):
        plt.figure(figsize=(7.2, 5.6))
        plt.hexbin(Xs[:, j], es, gridsize=60, bins="log")
        plt.xlabel(name)
        plt.ylabel("rmse_pix")
        plt.title(f"Feature vs error (hexbin, log counts) [{tag}]")
        savefig(plot_dir / f"08_feature_vs_error_hexbin_{j+1:02d}_{name}.png")


def radial_profile(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_int = np.floor(r).astype(int)
    rmax = r_int.max()
    prof = np.zeros(rmax + 1, dtype=np.float64)
    cnt = np.zeros(rmax + 1, dtype=np.float64)
    for k in range(rmax + 1):
        m = (r_int == k)
        if np.any(m):
            prof[k] = float(np.mean(img[m]))
            cnt[k] = float(np.sum(m))
    return prof


def plot_radial_profile(plot_dir: Path, true_coeff: np.ndarray, pred_coeff: np.ndarray, mean: np.ndarray, components: np.ndarray,
                        stamp_pix: int, max_n: int):
    N = true_coeff.shape[0]
    if N > max_n:
        idx = np.random.default_rng(123).choice(N, size=max_n, replace=False)
        tc = true_coeff[idx]
        pc = pred_coeff[idx]
    else:
        tc, pc = true_coeff, pred_coeff

    tmean, pmean, _ = mean_maps(tc, pc, mean, components, stamp_pix, max_n=max_n)
    pr_t = radial_profile(tmean)
    pr_p = radial_profile(pmean)

    n = min(len(pr_t), len(pr_p))
    plt.figure(figsize=(7.6, 4.8))
    plt.plot(np.arange(n), pr_t[:n], marker="o", linewidth=1, label="TRUE mean")
    plt.plot(np.arange(n), pr_p[:n], marker="o", linewidth=1, label="PRED mean")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Mean intensity (recon space)")
    plt.title("Radial profile of mean recon stamps")
    plt.legend()
    savefig(plot_dir / "07_radial_profile_mean_true_pred.png")


# ======================================================================
# MAIN
# ======================================================================

def main():
    base = CFG.base_dir
    run_root = base / "output" / "ml_runs" / CFG.run_name
    model_dir = run_root / "models"
    trace_dir = run_root / "trace"

    manifest = load_manifest(run_root)
    plot_dir = Path(manifest.get("plots_dir", str(CFG.plots_root / CFG.run_name)))
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== EVAL RUN ===")
    print("RUN_NAME:", CFG.run_name)
    print("RUN_ROOT:", run_root)
    print("PLOTS   :", plot_dir)

    n_feat = int(manifest["n_features"])
    K = int(manifest["K"])
    stamp_pix = int(manifest["stamp_pix"])
    feature_cols = manifest.get("feature_cols", [f"feat_{i}" for i in range(n_feat)])

    X_val_path = Path(manifest["X_val_path"])
    Y_val_path = Path(manifest["Y_val_path"])
    X_test_path = Path(manifest["X_test_path"])
    Y_test_path = Path(manifest["Y_test_path"])

    n_val = int(manifest["n_val"])
    n_test = int(manifest["n_test"])

    mean, components, evr = load_pca_from_npz(Path(manifest["pca_npz"]))
    boosters = load_models(model_dir, K=K)

    trace_test_csv = Path(manifest.get("trace_test_csv", str(trace_dir / "trace_test.csv")))
    trace_val_csv = Path(manifest.get("trace_val_csv", str(trace_dir / "trace_val.csv")))  # may not exist

    # memmaps for feature/error plots
    X_test_m = np.memmap(X_test_path, dtype="float32", mode="r", shape=(n_test, n_feat))

    # --- evaluate VAL ---
    Xv = np.memmap(X_val_path, dtype="float32", mode="r", shape=(n_val, n_feat))
    Yv = np.memmap(Y_val_path, dtype="float32", mode="r", shape=(n_val, K))
    pred_val = predict_coeffs_in_batches(boosters, Xv, batch_size=CFG.batch_size)
    true_val = np.asarray(Yv, dtype=np.float32)
    rmse_coeff_val = rmse_per_row(true_val, pred_val)

    rmse_pix_val = np.empty(n_val, dtype=np.float32)
    for s in range(0, n_val, CFG.batch_size):
        e = min(n_val, s + CFG.batch_size)
        pf = reconstruct_flat(pred_val[s:e], mean, components)
        tf = reconstruct_flat(true_val[s:e], mean, components)
        rmse_pix_val[s:e] = rmse_per_row(tf, pf)

    # --- evaluate TEST ---
    Xt = np.memmap(X_test_path, dtype="float32", mode="r", shape=(n_test, n_feat))
    Yt = np.memmap(Y_test_path, dtype="float32", mode="r", shape=(n_test, K))
    pred_test = predict_coeffs_in_batches(boosters, Xt, batch_size=CFG.batch_size)
    true_test = np.asarray(Yt, dtype=np.float32)
    rmse_coeff_test = rmse_per_row(true_test, pred_test)

    rmse_pix_test = np.empty(n_test, dtype=np.float32)
    for s in range(0, n_test, CFG.batch_size):
        e = min(n_test, s + CFG.batch_size)
        pf = reconstruct_flat(pred_test[s:e], mean, components)
        tf = reconstruct_flat(true_test[s:e], mean, components)
        rmse_pix_test[s:e] = rmse_per_row(tf, pf)

    # ------------------------------------------------------------
    # Outliers exports (unchanged behavior)
    # ------------------------------------------------------------
    idx_val = top_outliers(rmse_pix_val, CFG.top_n_val)
    idx_test = top_outliers(rmse_pix_test, CFG.top_n_test)

    np.save(plot_dir / "outliers_indices_val.npy", idx_val)
    np.save(plot_dir / "outliers_indices_test.npy", idx_test)

    write_outliers_trace(
        split_name="test",
        out_indices=idx_test,
        rmse_pix=rmse_pix_test,
        rmse_coeff=rmse_coeff_test,
        trace_csv=trace_test_csv if trace_test_csv.exists() else None,
        out_csv=plot_dir / "outliers_trace_test.csv",
    )

    if trace_val_csv.exists():
        write_outliers_trace(
            split_name="val",
            out_indices=idx_val,
            rmse_pix=rmse_pix_val,
            rmse_coeff=rmse_coeff_val,
            trace_csv=trace_val_csv,
            out_csv=plot_dir / "outliers_trace_val.csv",
        )
    else:
        write_outliers_trace(
            split_name="val",
            out_indices=idx_val,
            rmse_pix=rmse_pix_val,
            rmse_coeff=rmse_coeff_val,
            trace_csv=None,
            out_csv=plot_dir / "outliers_trace_val.csv",
        )

    # ------------------------------------------------------------
    # Plots (the “nice set”)
    # ------------------------------------------------------------

    # 03 per-stamp pixel rmse hist (test)
    plt.figure(figsize=(7.2, 4.6))
    plt.hist(rmse_pix_test, bins=90)
    plt.xlabel("Per-sample RMSE (pixel space, reconstructed)")
    plt.ylabel("Count")
    plt.title("TEST per-stamp pixel RMSE histogram")
    savefig(plot_dir / "03_perstamp_pixel_rmse_hist.png")

    # component-wise rmse (test): compute per-component RMSE across all test samples
    comp_rmse = np.sqrt(np.mean((true_test - pred_test) ** 2, axis=0))
    plot_rmse_per_component(plot_dir, comp_rmse, tag="test")
    plot_rmse_normalized(plot_dir, comp_rmse, true_test, tag="test")

    # explained var vs rmse
    if evr is not None:
        plot_explainedvar_vs_rmse(plot_dir, evr, comp_rmse, tag="test")

    # mean maps
    tmean, pmean, rmean = mean_maps(true_test, pred_test, mean, components, stamp_pix, max_n=CFG.mean_maps_max)
    plot_mean_maps(plot_dir, tmean, pmean, rmean)

    # examples montage (resid only)
    plot_examples(plot_dir, true_test, pred_test, mean, components, stamp_pix, rmse_pix_test, n=CFG.examples_n)

    # error vs features (Spearman)
    plot_error_vs_features(plot_dir, X_test_m, rmse_pix_test, feature_cols, tag="test")

    # radial profile
    plot_radial_profile(plot_dir, true_test, pred_test, mean, components, stamp_pix, max_n=CFG.radial_profile_max)

    # hexbin feature vs error
    plot_feature_vs_error_hexbin(plot_dir, X_test_m, rmse_pix_test, feature_cols, tag="test")

    print("\nDONE. Outputs in:", plot_dir)


if __name__ == "__main__":
    main()
