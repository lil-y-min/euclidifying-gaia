"""
15_bis_eval_xgb_pixels_flux_chi2.py (pandas-free, cluster-safe)
==============================================================

Loads a "pixels+flux" run produced by 14_bis:
- shape targets: normalized pixels (D)
- flux targets: log10(F)

Predicts:
- shape: D boosters
- flux: 1 booster (still computed for diagnostics)

Reconstructs raw stamps:
  I_true = S_true * F_true
  I_pred = S_pred * F_true   (flux_mode=true; DEFAULT)
      OR I_pred = S_pred * F_pred (flux_mode=pred)

Computes chi^2 and reduced chi^2 per stamp using sigma_mode:

A) sigma_mode = sqrt_Itrue (DEFAULT, recommended)
   Pixelwise noise model:
     sigma_p^2 = sigma_bg^2 + alpha * max(I_true_p, 0) + sigma0^2
   where:
     - sigma_bg is estimated PER STAMP from border pixels using MAD
     - alpha controls the Poisson-like scaling (default 1.0)
     - sigma0 is a small sigma floor (std; we add sigma0^2 to variance)

B) sigma_mode = border_mad (legacy)
   Constant sigma per stamp applied to all pixels:
     sigma_p^2 = sigma_bg^2

Reduced chi2 per stamp:
  chi2nu = chi2 / nu, with nu = D (p = 0)

Outputs to:
.../Codes/plots/ml_runs/<RUN_NAME>/
  outliers_indices_test.npy
  outliers_indices_val.npy
  outliers_trace_test.csv
  outliers_trace_val.csv
  01_chi2nu_hist_test.png
  01b_chi2nu_hist_test_log10.png
  02_mean_maps_true_pred_absresid_raw.png
  03_examples_resid_raw.png
  test_chi2_metrics.csv

Run examples:
  python .../15_bis_eval_xgb_pixels_flux_chi2.py --run ml_xgb_pixelsflux_8d_augtrain
  python .../15_bis_eval_xgb_pixels_flux_chi2.py --run ml_xgb_pixelsflux_8d_augtrain --flux_mode pred
  python .../15_bis_eval_xgb_pixels_flux_chi2.py --run ml_xgb_pixelsflux_8d_augtrain --sigma_mode border_mad
  python .../15_bis_eval_xgb_pixels_flux_chi2.py --run ml_xgb_pixelsflux_8d_augtrain --alpha 0.5 --sigma0 1e-3
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import xgboost as xgb


# ======================================================================
# CONFIG
# ======================================================================

@dataclass
class EvalConfig:
    base_dir: Path = Path(__file__).resolve().parents[2]  # .../Codes

    # ranking / exports
    top_n_test: int = 250
    top_n_val: int = 250

    batch_size: int = 100_000

    # sigma_mode = sqrt_Itrue
    sigma0_default: float = 1e-4  # sigma floor (std). We add sigma0^2 to variance.
    alpha_default: float = 1.0    # Poisson-like scaling in sigma^2 = ... + alpha*I_true_pos

    # sigma_bg estimation from border MAD (used in BOTH modes)
    border_w: int = 2
    sigma_bg_floor: float = 1e-6  # floor for sigma_bg (std)

    mean_maps_max: int = 60_000
    examples_n: int = 16

    plots_root: Path = None

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


def load_manifest(run_root: Path) -> Dict:
    mpath = run_root / "manifest_arrays.npz"
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest file not found: {mpath}")

    d = np.load(mpath, allow_pickle=True)
    out = {k: d[k] for k in d.files}

    # normalize types
    for k in ["n_train", "n_val", "n_test", "n_features", "stamp_pix", "D"]:
        if k in out:
            out[k] = int(out[k])
    for k in [
        "X_val_path", "Yshape_val_path", "Yflux_val_path",
        "X_test_path", "Yshape_test_path", "Yflux_test_path",
        "trace_test_csv", "plots_dir"
    ]:
        if k in out:
            out[k] = str(out[k])

    # feature cols
    if "feature_cols" in out:
        out["feature_cols"] = [str(x) for x in out["feature_cols"].tolist()]

    return out


def load_shape_models(model_dir: Path, D: int) -> List[xgb.Booster]:
    boosters = []
    for j in range(D):
        p = model_dir / f"booster_pix_{j:04d}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing pixel model: {p}")
        b = xgb.Booster()
        b.load_model(str(p))
        boosters.append(b)
    return boosters


def load_flux_model(model_dir: Path) -> xgb.Booster:
    p = model_dir / "booster_flux.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing flux model: {p}")
    b = xgb.Booster()
    b.load_model(str(p))
    return b


def predict_shape_in_batches(boosters: List[xgb.Booster], X: np.memmap, batch_size: int) -> np.ndarray:
    N = X.shape[0]
    D = len(boosters)
    out = np.empty((N, D), dtype=np.float32)

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        Xb = np.asarray(X[s:e], dtype=np.float32, order="C")
        dmat = xgb.DMatrix(Xb)
        for j, booster in enumerate(boosters):
            out[s:e, j] = booster.predict(dmat).astype(np.float32, copy=False)

    return out


def predict_flux_in_batches(booster_flux: xgb.Booster, X: np.memmap, batch_size: int) -> np.ndarray:
    N = X.shape[0]
    out = np.empty(N, dtype=np.float32)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        Xb = np.asarray(X[s:e], dtype=np.float32, order="C")
        dmat = xgb.DMatrix(Xb)
        out[s:e] = booster_flux.predict(dmat).astype(np.float32, copy=False)
    return out


def top_outliers(score: np.ndarray, top_n: int) -> np.ndarray:
    N = score.shape[0]
    top_n = int(min(max(top_n, 1), N))
    idx = np.argpartition(score, -top_n)[-top_n:]
    idx = idx[np.argsort(score[idx])[::-1]]
    return idx.astype(np.int64)


def sigma_bg_from_border_mad(I_true: np.ndarray, stamp_pix: int, w: int, floor: float) -> np.ndarray:
    """
    Per-stamp sigma_bg (std) estimated from border pixels via MAD.
    Returns shape (N,).
    """
    N = I_true.shape[0]
    imgs = I_true.reshape(N, stamp_pix, stamp_pix)

    mask = np.zeros((stamp_pix, stamp_pix), dtype=bool)
    mask[:w, :] = True
    mask[-w:, :] = True
    mask[:, :w] = True
    mask[:, -w:] = True

    b = imgs[:, mask]  # (N, nb)
    med = np.median(b, axis=1)
    mad = np.median(np.abs(b - med[:, None]), axis=1)
    sigma = 1.4826 * mad  # robust std
    sigma = np.where(np.isfinite(sigma), sigma, floor)
    sigma = np.maximum(sigma, floor)
    return sigma.astype(np.float32)


def chi2nu_sqrt_Itrue(
    I_true: np.ndarray,
    I_pred: np.ndarray,
    sigma_bg: np.ndarray,
    sigma0: float,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pixelwise chi^2 per stamp with:
      sigma_p^2 = sigma_bg^2 + alpha * max(I_true_p, 0) + sigma0^2

    Returns (chi2, chi2nu) per stamp.
    """
    I_true = np.asarray(I_true, dtype=np.float32)
    I_pred = np.asarray(I_pred, dtype=np.float32)
    resid = I_pred - I_true

    # sigma_bg is per stamp -> broadcast to pixels
    sig2 = (sigma_bg.astype(np.float32) ** 2)[:, None]

    # Poisson-like term
    a = float(alpha)
    if a != 0.0:
        sig2 = sig2 + a * np.maximum(I_true, 0.0)

    # sigma floor term (sigma0 is std)
    sig2 = sig2 + (float(sigma0) ** 2)

    # guard
    sig2 = np.maximum(sig2, 1e-30).astype(np.float32)

    chi2 = np.sum((resid * resid) / sig2, axis=1)
    nu = float(I_true.shape[1])  # D pixels, p=0
    chi2nu = chi2 / nu
    return chi2.astype(np.float32), chi2nu.astype(np.float32)


def chi2nu_border_mad(I_true: np.ndarray, I_pred: np.ndarray, sigma_bg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constant sigma per stamp, applied to all pixels in that stamp:
      sigma_p^2 = sigma_bg^2
    """
    resid = (I_pred - I_true).astype(np.float32)
    sig2 = (sigma_bg.astype(np.float32) ** 2)[:, None]
    sig2 = np.maximum(sig2, 1e-30)
    chi2 = np.sum((resid * resid) / sig2, axis=1)
    chi2nu = chi2 / float(I_true.shape[1])
    return chi2.astype(np.float32), chi2nu.astype(np.float32)


def mean_maps_raw(I_true: np.ndarray, I_pred: np.ndarray, stamp_pix: int, max_n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = I_true.shape[0]
    if N > max_n:
        idx = np.random.default_rng(123).choice(N, size=max_n, replace=False)
        t = I_true[idx]
        p = I_pred[idx]
    else:
        t, p = I_true, I_pred

    absr = np.abs(p - t)
    tmean = t.mean(axis=0).reshape(stamp_pix, stamp_pix)
    pmean = p.mean(axis=0).reshape(stamp_pix, stamp_pix)
    rmean = absr.mean(axis=0).reshape(stamp_pix, stamp_pix)
    return tmean, pmean, rmean


def plot_mean_maps(plot_dir: Path, tmean: np.ndarray, pmean: np.ndarray, rmean: np.ndarray):
    vmin = float(np.percentile(np.concatenate([tmean.ravel(), pmean.ravel()]), 1))
    vmax = float(np.percentile(np.concatenate([tmean.ravel(), pmean.ravel()]), 99))

    fig = plt.figure(figsize=(10.5, 3.6))
    gs = fig.add_gridspec(1, 3, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    for ax in (ax1, ax2, ax3):
        ax.set_xticks([]); ax.set_yticks([])

    ax1.set_title("Mean TRUE (raw)", fontsize=10)
    ax2.set_title("Mean PRED (raw)", fontsize=10)
    ax3.set_title("Mean |RESID|", fontsize=10)

    ax1.imshow(tmean, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax2.imshow(pmean, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax3.imshow(rmean, origin="lower", cmap="magma")
    savefig(plot_dir / "02_mean_maps_true_pred_absresid_raw.png")


def plot_examples_resid(plot_dir: Path, I_true: np.ndarray, I_pred: np.ndarray,
                        stamp_pix: int, score: np.ndarray, n: int):
    idx = top_outliers(score, n)
    resid = (I_pred[idx] - I_true[idx]).reshape(n, stamp_pix, stamp_pix)

    side = int(math.sqrt(n))
    side = side if side * side == n else int(math.ceil(math.sqrt(n)))

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(side, side, wspace=0.15, hspace=0.25)

    for i in range(n):
        ax = fig.add_subplot(gs[i // side, i % side])
        ax.set_xticks([]); ax.set_yticks([])
        rimg = resid[i]
        v = np.percentile(np.abs(rimg.ravel()), 99)
        v = float(v if v > 0 else 1e-6)
        ax.imshow(rimg, origin="lower", cmap="RdBu_r", vmin=-v, vmax=+v)
        ax.set_title(f"rank {i}", fontsize=9)

    fig.suptitle("Example residuals (raw space, top chi2nu)", fontsize=12)
    savefig(plot_dir / "03_examples_resid_raw.png")


def read_trace_csv(path: Path) -> List[Dict[str, str]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def write_outliers_trace_test(plot_dir: Path, trace_rows: List[Dict[str, str]],
                             idx_test: np.ndarray, chi2nu_test: np.ndarray):
    out_csv = plot_dir / "outliers_trace_test.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # map test_index -> trace row
    m = {}
    for row in trace_rows:
        try:
            k = int(row.get("test_index", "-1"))
            m[k] = row
        except Exception:
            continue

    fieldnames = list(trace_rows[0].keys()) if trace_rows else ["test_index"]
    for x in ["chi2nu", "rank"]:
        if x not in fieldnames:
            fieldnames.append(x)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rank, ti in enumerate(idx_test.tolist()):
            row = dict(m.get(int(ti), {"test_index": str(int(ti))}))
            row["chi2nu"] = f"{float(chi2nu_test[int(ti)]):.6g}"
            row["rank"] = str(rank)
            w.writerow(row)


def write_outliers_metrics_only(plot_dir: Path, split: str, idx: np.ndarray, score: np.ndarray):
    out_csv = plot_dir / f"outliers_trace_{split}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[f"{split}_index", "chi2nu", "rank"])
        w.writeheader()
        for rank, i in enumerate(idx.tolist()):
            w.writerow({f"{split}_index": int(i), "chi2nu": float(score[int(i)]), "rank": rank})


def quick_debug_prints(I_true: np.ndarray, sigma_bg: np.ndarray, name: str):
    """
    Prints two things that often explain 'weird chi2':
      - sigma_bg stats
      - fraction of pixels <= 0
    """
    frac_le0 = float(np.mean(I_true <= 0.0))
    print(f"[DEBUG {name}] sigma_bg: median={float(np.median(sigma_bg)):.4g} "
          f"(p16={float(np.percentile(sigma_bg,16)):.4g}, p84={float(np.percentile(sigma_bg,84)):.4g})")
    print(f"[DEBUG {name}] fraction of pixels with I_true<=0: {frac_le0:.4f}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run name under output/ml_runs/")
    ap.add_argument("--flux_mode", choices=["true", "pred"], default="true",
                    help="Reconstruction uses F_true (default) or F_pred")
    ap.add_argument("--sigma_mode", choices=["sqrt_Itrue", "border_mad"], default="sqrt_Itrue",
                    help="Noise model for chi^2 (default: sqrt_Itrue)")
    ap.add_argument("--sigma0", type=float, default=CFG.sigma0_default,
                    help="sigma floor (std) for sqrt_Itrue: sigma^2 includes + sigma0^2")
    ap.add_argument("--alpha", type=float, default=CFG.alpha_default,
                    help="Poisson-like scaling in sqrt_Itrue: sigma^2 includes + alpha*max(I_true,0)")
    ap.add_argument("--no_log10_hist", action="store_true",
                    help="If set, do not write the log10 histogram plot")
    args = ap.parse_args()

    run_root = CFG.base_dir / "output" / "ml_runs" / args.run
    model_dir = run_root / "models"
    trace_dir = run_root / "trace"

    manifest = load_manifest(run_root)
    plot_dir = Path(manifest.get("plots_dir", str(CFG.plots_root / args.run)))
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== EVAL RUN (PIXELS+FLUX, CHI2) ===")
    print("RUN_NAME   :", args.run)
    print("RUN_ROOT   :", run_root)
    print("PLOTS      :", plot_dir)
    print("flux_mode  :", args.flux_mode)
    print("sigma_mode :", args.sigma_mode)
    if args.sigma_mode == "sqrt_Itrue":
        print("alpha      :", args.alpha)
        print("sigma0(std):", args.sigma0)

    n_feat = int(manifest["n_features"])
    stamp_pix = int(manifest["stamp_pix"])
    D = int(manifest["D"])
    n_val = int(manifest["n_val"])
    n_test = int(manifest["n_test"])

    X_val_path = Path(manifest["X_val_path"])
    Yshape_val_path = Path(manifest["Yshape_val_path"])
    Yflux_val_path = Path(manifest["Yflux_val_path"])
    X_test_path = Path(manifest["X_test_path"])
    Yshape_test_path = Path(manifest["Yshape_test_path"])
    Yflux_test_path = Path(manifest["Yflux_test_path"])

    trace_test_csv = Path(manifest.get("trace_test_csv", str(trace_dir / "trace_test.csv")))

    boosters_shape = load_shape_models(model_dir, D=D)
    booster_flux = load_flux_model(model_dir)  # still used for diagnostics + optional flux_mode=pred

    # -------------------
    # VAL
    # -------------------
    Xv = np.memmap(X_val_path, dtype="float32", mode="r", shape=(n_val, n_feat))
    Yshape_v = np.memmap(Yshape_val_path, dtype="float32", mode="r", shape=(n_val, D))
    Yflux_v = np.memmap(Yflux_val_path, dtype="float32", mode="r", shape=(n_val,))

    pred_shape_v = predict_shape_in_batches(boosters_shape, Xv, batch_size=CFG.batch_size)
    pred_logF_v = predict_flux_in_batches(booster_flux, Xv, batch_size=CFG.batch_size)

    logF_true_v = np.asarray(Yflux_v, dtype=np.float32)
    F_true_v = (10.0 ** logF_true_v).astype(np.float32)
    F_pred_v = (10.0 ** np.asarray(pred_logF_v, dtype=np.float32)).astype(np.float32)

    I_true_v = np.asarray(Yshape_v, dtype=np.float32) * F_true_v[:, None]
    if args.flux_mode == "true":
        I_pred_v = np.asarray(pred_shape_v, dtype=np.float32) * F_true_v[:, None]
    else:
        I_pred_v = np.asarray(pred_shape_v, dtype=np.float32) * F_pred_v[:, None]

    # sigma_bg is estimated from TRUE image border (always)
    sigma_bg_v = sigma_bg_from_border_mad(I_true_v, stamp_pix, CFG.border_w, CFG.sigma_bg_floor)

    if args.sigma_mode == "sqrt_Itrue":
        _, chi2nu_v = chi2nu_sqrt_Itrue(I_true_v, I_pred_v, sigma_bg_v, sigma0=args.sigma0, alpha=args.alpha)
    else:
        _, chi2nu_v = chi2nu_border_mad(I_true_v, I_pred_v, sigma_bg_v)

    # -------------------
    # TEST
    # -------------------
    Xt = np.memmap(X_test_path, dtype="float32", mode="r", shape=(n_test, n_feat))
    Yshape_t = np.memmap(Yshape_test_path, dtype="float32", mode="r", shape=(n_test, D))
    Yflux_t = np.memmap(Yflux_test_path, dtype="float32", mode="r", shape=(n_test,))

    pred_shape_t = predict_shape_in_batches(boosters_shape, Xt, batch_size=CFG.batch_size)
    pred_logF_t = predict_flux_in_batches(booster_flux, Xt, batch_size=CFG.batch_size)

    logF_true_t = np.asarray(Yflux_t, dtype=np.float32)
    F_true_t = (10.0 ** logF_true_t).astype(np.float32)
    F_pred_t = (10.0 ** np.asarray(pred_logF_t, dtype=np.float32)).astype(np.float32)

    I_true_t = np.asarray(Yshape_t, dtype=np.float32) * F_true_t[:, None]
    if args.flux_mode == "true":
        I_pred_t = np.asarray(pred_shape_t, dtype=np.float32) * F_true_t[:, None]
    else:
        I_pred_t = np.asarray(pred_shape_t, dtype=np.float32) * F_pred_t[:, None]

    sigma_bg_t = sigma_bg_from_border_mad(I_true_t, stamp_pix, CFG.border_w, CFG.sigma_bg_floor)

    # Debug prints that explain most "wtf chi2" moments
    quick_debug_prints(I_true_t, sigma_bg_t, name="TEST")

    if args.sigma_mode == "sqrt_Itrue":
        _, chi2nu_t = chi2nu_sqrt_Itrue(I_true_t, I_pred_t, sigma_bg_t, sigma0=args.sigma0, alpha=args.alpha)
    else:
        _, chi2nu_t = chi2nu_border_mad(I_true_t, I_pred_t, sigma_bg_t)

    # -------------------
    # Outliers exports
    # -------------------
    idx_val = top_outliers(chi2nu_v, CFG.top_n_val)
    idx_test = top_outliers(chi2nu_t, CFG.top_n_test)

    np.save(plot_dir / "outliers_indices_val.npy", idx_val)
    np.save(plot_dir / "outliers_indices_test.npy", idx_test)

    trace_rows = read_trace_csv(trace_test_csv)
    if trace_rows:
        write_outliers_trace_test(plot_dir, trace_rows, idx_test, chi2nu_t)
    else:
        write_outliers_metrics_only(plot_dir, "test", idx_test, chi2nu_t)

    write_outliers_metrics_only(plot_dir, "val", idx_val, chi2nu_v)

    # -------------------
    # Plots
    # -------------------
    v = chi2nu_t[np.isfinite(chi2nu_t)]
    v = v[v > 0]

    plt.figure(figsize=(7.2, 4.6))
    plt.hist(v, bins=90)
    plt.xlabel("Reduced chi^2 (chi2 / D)")
    plt.ylabel("Count")
    plt.title(f"TEST reduced chi^2 | flux_mode={args.flux_mode} | sigma_mode={args.sigma_mode}")
    savefig(plot_dir / "01_chi2nu_hist_test.png")

    if (not args.no_log10_hist) and (v.size > 0):
        plt.figure(figsize=(7.2, 4.6))
        plt.hist(np.log10(v), bins=120)
        plt.xlabel("log10(chi2nu)")
        plt.ylabel("Count")
        plt.title(f"TEST reduced chi^2 histogram (log10)\nflux_mode={args.flux_mode}, sigma_mode={args.sigma_mode}")
        savefig(plot_dir / "01b_chi2nu_hist_test_log10.png")

    tmean, pmean, rmean = mean_maps_raw(I_true_t, I_pred_t, stamp_pix, CFG.mean_maps_max)
    plot_mean_maps(plot_dir, tmean, pmean, rmean)
    plot_examples_resid(plot_dir, I_true_t, I_pred_t, stamp_pix, chi2nu_t, CFG.examples_n)

    # -------------------
    # Save compact metrics CSV
    # -------------------
    with open(plot_dir / "test_chi2_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "chi2nu",
                "sigma_bg",
                "logF_true",
                "logF_pred",
                "dlogF",
                "flux_mode",
                "sigma_mode",
                "alpha",
                "sigma0",
            ],
        )
        w.writeheader()
        for i in range(n_test):
            lt = float(logF_true_t[i])
            lp = float(pred_logF_t[i])
            w.writerow({
                "chi2nu": float(chi2nu_t[i]),
                "sigma_bg": float(sigma_bg_t[i]),
                "logF_true": lt,
                "logF_pred": lp,
                "dlogF": float(lp - lt),
                "flux_mode": args.flux_mode,
                "sigma_mode": args.sigma_mode,
                "alpha": float(args.alpha),
                "sigma0": float(args.sigma0),
            })

    print("\nDONE. Outputs in:", plot_dir)


if __name__ == "__main__":
    main()
