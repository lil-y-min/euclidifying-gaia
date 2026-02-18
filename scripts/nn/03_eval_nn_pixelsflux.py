#!/usr/bin/env python3
"""
Evaluate NN stamp reconstruction run and generate publication-quality QA plots:
- chi2 histograms
- mean TRUE/PRED/|RESID| maps with colorbars
- top residual examples with diverging colormap
- permutation feature importance CSV + plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn


@dataclass
class EvalCfg:
    base_dir: Path = Path(__file__).resolve().parents[2]
    batch_size: int = 1024
    sigma0: float = 1e-4
    alpha: float = 1.0
    border_w: int = 2
    sigma_bg_floor: float = 1e-6
    examples_n: int = 16
    mean_maps_max: int = 60000
    importance_max_samples: int = 8000
    triplet_rows_n: int = 12


CFG = EvalCfg()

sns.set_theme(style="ticks", context="paper", font_scale=0.95)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
})


class ShapeFluxMLP(nn.Module):
    def __init__(self, n_feat: int, D: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        blocks: List[nn.Module] = []
        in_dim = n_feat
        for _ in range(depth):
            blocks.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.head_shape = nn.Linear(hidden_dim, D)
        self.head_flux = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        shape_logits = self.head_shape(z)
        shape = torch.softmax(shape_logits, dim=1)
        flux = self.head_flux(z).squeeze(1)
        return shape, flux


def savefig(path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    if fig.get_layout_engine() is None:
        fig.tight_layout(pad=0.8)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _to_int(v) -> int:
    return int(np.asarray(v).reshape(()))


def _to_path(v) -> Path:
    return Path(str(np.asarray(v).reshape(())))


def load_manifest(path: Path) -> Dict[str, object]:
    d = np.load(path, allow_pickle=True)
    out = {
        "n_train": _to_int(d["n_train"]),
        "n_val": _to_int(d["n_val"]),
        "n_test": _to_int(d["n_test"]),
        "n_features": _to_int(d["n_features"]),
        "D": _to_int(d["D"]),
        "stamp_pix": _to_int(d["stamp_pix"]),
        "X_val_path": _to_path(d["X_val_path"]),
        "Yshape_val_path": _to_path(d["Yshape_val_path"]),
        "Yflux_val_path": _to_path(d["Yflux_val_path"]),
        "X_test_path": _to_path(d["X_test_path"]),
        "Yshape_test_path": _to_path(d["Yshape_test_path"]),
        "Yflux_test_path": _to_path(d["Yflux_test_path"]),
    }
    if "feature_cols" in d.files:
        out["feature_cols"] = [str(x) for x in d["feature_cols"].tolist()]
    else:
        out["feature_cols"] = [f"feat_{i}" for i in range(out["n_features"])]
    return out


def sigma_bg_from_border_mad(I_true: np.ndarray, stamp_pix: int, w: int, floor: float) -> np.ndarray:
    N = I_true.shape[0]
    imgs = I_true.reshape(N, stamp_pix, stamp_pix)
    mask = np.zeros((stamp_pix, stamp_pix), dtype=bool)
    mask[:w, :] = True
    mask[-w:, :] = True
    mask[:, :w] = True
    mask[:, -w:] = True
    b = imgs[:, mask]
    med = np.median(b, axis=1)
    mad = np.median(np.abs(b - med[:, None]), axis=1)
    sig = 1.4826 * mad
    sig = np.where(np.isfinite(sig), sig, floor)
    sig = np.maximum(sig, floor)
    return sig.astype(np.float32)


def chi2nu_sqrt_Itrue(I_true: np.ndarray, I_pred: np.ndarray, sigma_bg: np.ndarray, sigma0: float, alpha: float) -> np.ndarray:
    resid = (I_pred - I_true).astype(np.float32)
    sig2 = (sigma_bg.astype(np.float32) ** 2)[:, None]
    if float(alpha) != 0.0:
        sig2 = sig2 + float(alpha) * np.maximum(I_true, 0.0)
    sig2 = sig2 + float(sigma0) ** 2
    sig2 = np.maximum(sig2, 1e-30)
    chi2 = np.sum((resid * resid) / sig2, axis=1)
    return (chi2 / float(I_true.shape[1])).astype(np.float32)


def top_outliers(score: np.ndarray, top_n: int) -> np.ndarray:
    N = score.shape[0]
    k = int(min(max(1, top_n), N))
    idx = np.argpartition(score, -k)[-k:]
    idx = idx[np.argsort(score[idx])[::-1]]
    return idx.astype(np.int64)


def raw_from_shape_flux(shape: np.ndarray, flux: np.ndarray, clip_min: float = -6.0, clip_max: float = 8.0) -> np.ndarray:
    f = np.clip(np.asarray(flux, dtype=np.float32), clip_min, clip_max)
    scale = (10.0 ** f).astype(np.float32)
    return np.asarray(shape, dtype=np.float32) * scale[:, None]


def raw_from_shape_and_true_flux(shape_pred: np.ndarray, flux_true: np.ndarray, clip_min: float = -6.0, clip_max: float = 8.0) -> np.ndarray:
    f_true = np.clip(np.asarray(flux_true, dtype=np.float32), clip_min, clip_max)
    scale_true = (10.0 ** f_true).astype(np.float32)
    return np.asarray(shape_pred, dtype=np.float32) * scale_true[:, None]


def predict_all(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    N = X.shape[0]
    D = int(model.head_shape.out_features)
    out_shape = np.empty((N, D), dtype=np.float32)
    out_flux = np.empty(N, dtype=np.float32)

    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            xb = torch.from_numpy(np.asarray(X[s:e], dtype=np.float32).copy()).to(device)
            ys, yf = model(xb)
            out_shape[s:e] = ys.detach().cpu().numpy().astype(np.float32)
            out_flux[s:e] = yf.detach().cpu().numpy().astype(np.float32)

    return out_shape, out_flux


def compute_metrics(yshape_true: np.ndarray, yshape_pred: np.ndarray, yflux_true: np.ndarray, yflux_pred: np.ndarray) -> Dict[str, float]:
    rmse_shape = float(np.sqrt(np.mean((yshape_pred - yshape_true) ** 2)))
    rmse_flux = float(np.sqrt(np.mean((yflux_pred - yflux_true) ** 2)))
    I_true = raw_from_shape_flux(yshape_true, yflux_true)
    I_pred = raw_from_shape_flux(yshape_pred, yflux_pred)
    rmse_raw = float(np.sqrt(np.mean((I_pred - I_true) ** 2)))
    return {"rmse_shape": rmse_shape, "rmse_flux": rmse_flux, "rmse_raw": rmse_raw}


def permutation_importance(
    model: nn.Module,
    X_val: np.ndarray,
    Yshape_val: np.ndarray,
    Yflux_val: np.ndarray,
    feature_names: List[str],
    max_samples: int,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    n = int(min(max_samples, X_val.shape[0]))
    X = np.asarray(X_val[:n], dtype=np.float32)
    ys_true = np.asarray(Yshape_val[:n], dtype=np.float32)
    yf_true = np.asarray(Yflux_val[:n], dtype=np.float32)

    ys_pred, yf_pred = predict_all(model, X, batch_size=batch_size, device=device)
    base = compute_metrics(ys_true, ys_pred, yf_true, yf_pred)

    rng = np.random.default_rng(123)
    out = []
    for j, name in enumerate(feature_names):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        yp_s, yp_f = predict_all(model, Xp, batch_size=batch_size, device=device)
        m = compute_metrics(ys_true, yp_s, yf_true, yp_f)
        out.append({
            "feature": name,
            "delta_rmse_raw": float(m["rmse_raw"] - base["rmse_raw"]),
            "delta_rmse_shape": float(m["rmse_shape"] - base["rmse_shape"]),
            "delta_rmse_flux": float(m["rmse_flux"] - base["rmse_flux"]),
        })
    out.sort(key=lambda r: r["delta_rmse_raw"], reverse=True)
    return out


def plot_chi2_hist(v: np.ndarray, out_dir: Path) -> None:
    vv = v[np.isfinite(v)]
    vv = vv[vv > 0]
    if vv.size == 0:
        return

    plt.figure(figsize=(8.5, 5.0))
    sns.histplot(vv, bins=90, color="#2B6CB0")
    plt.xlabel(r"Reduced $\chi^2$ ($\chi^2/D$)")
    plt.ylabel("Count")
    plt.title("NN TEST reduced chi-square")
    savefig(out_dir / "01_chi2nu_hist_test.png")

    plt.figure(figsize=(8.5, 5.0))
    sns.histplot(np.log10(vv), bins=120, color="#1F4E79")
    plt.xlabel(r"$\log_{10}(\chi^2_\nu)$")
    plt.ylabel("Count")
    plt.title("NN TEST reduced chi-square (log10)")
    savefig(out_dir / "01b_chi2nu_hist_test_log10.png")


def plot_mean_maps(I_true_t: np.ndarray, I_pred_t: np.ndarray, stamp_pix: int, out_dir: Path, max_n: int) -> None:
    N = I_true_t.shape[0]
    if N == 0:
        return
    if N > max_n:
        idx = np.random.default_rng(123).choice(N, size=max_n, replace=False)
        t = I_true_t[idx]
        p = I_pred_t[idx]
    else:
        t, p = I_true_t, I_pred_t

    tmean = t.mean(axis=0).reshape(stamp_pix, stamp_pix)
    pmean = p.mean(axis=0).reshape(stamp_pix, stamp_pix)
    rmean = np.abs(p - t).mean(axis=0).reshape(stamp_pix, stamp_pix)

    vmin = float(np.percentile(np.concatenate([tmean.ravel(), pmean.ravel()]), 1))
    vmax = float(np.percentile(np.concatenate([tmean.ravel(), pmean.ravel()]), 99))

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.2), constrained_layout=True)

    im0 = axes[0].imshow(tmean, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Mean TRUE (raw)")
    axes[0].set_xlabel("x [pix]")
    axes[0].set_ylabel("y [pix]")
    c0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    c0.set_label("intensity")

    im1 = axes[1].imshow(pmean, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Mean PRED (raw)")
    axes[1].set_xlabel("x [pix]")
    axes[1].set_ylabel("y [pix]")
    c1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    c1.set_label("intensity")

    im2 = axes[2].imshow(rmean, origin="lower", cmap="magma")
    axes[2].set_title("Mean |RESID| (raw)")
    axes[2].set_xlabel("x [pix]")
    axes[2].set_ylabel("y [pix]")
    c2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)
    c2.set_label("|residual|")

    savefig(out_dir / "02_mean_maps_true_pred_absresid_raw.png")


def plot_resid_examples(I_true_t: np.ndarray, I_pred_t: np.ndarray, chi2nu_t: np.ndarray, stamp_pix: int, out_dir: Path, n: int) -> None:
    idx = top_outliers(chi2nu_t, n)
    n = int(idx.size)
    if n == 0:
        return
    side = int(math.ceil(math.sqrt(n)))

    fig, axes = plt.subplots(side, side, figsize=(2.9 * side, 2.7 * side), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    im = None

    for i in range(side * side):
        ax = axes[i]
        if i >= n:
            ax.axis("off")
            continue
        rimg = (I_pred_t[idx[i]] - I_true_t[idx[i]]).reshape(stamp_pix, stamp_pix)
        vmax = float(np.percentile(np.abs(rimg), 99))
        vmax = vmax if vmax > 0 else 1e-6
        im = ax.imshow(rimg, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"#{i}  chi2={chi2nu_t[idx[i]]:.2g}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    if im is not None:
        used_axes = [ax for i, ax in enumerate(axes) if i < n]
        cbar = fig.colorbar(im, ax=used_axes, fraction=0.02, pad=0.01)
        cbar.set_label("Residual")
    fig.suptitle("Top residual examples (raw space)", y=1.01)
    savefig(out_dir / "03_examples_resid_raw.png")


def plot_importance(rows: List[Dict[str, float]], out_dir: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["delta_rmse_raw"], reverse=True)
    names = [r["feature"] for r in rows]
    vals = np.array([r["delta_rmse_raw"] for r in rows], dtype=float)

    plt.figure(figsize=(10, max(4.5, 0.42 * len(names))))
    sns.barplot(x=vals, y=names, orient="h", color="#2C7FB8")
    plt.xlabel("Permutation importance: delta RMSE(raw)")
    plt.ylabel("Feature")
    plt.title("NN feature importance (validation subset)")
    savefig(out_dir / "04_feature_importance_permutation_rmse_raw.png")


def _as_1d(a: np.ndarray) -> np.ndarray:
    return np.atleast_1d(np.asarray(a))


def plot_learning_curves(run_root: Path, out_dir: Path) -> None:
    hist_path = run_root / "history.csv"
    if not hist_path.exists():
        return

    hist = np.genfromtxt(hist_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if hist.size == 0:
        return

    epoch = _as_1d(hist["epoch"]).astype(float)
    train_loss = _as_1d(hist["train_loss"]).astype(float)
    val_loss = _as_1d(hist["val_loss"]).astype(float)
    train_rmse_flux = _as_1d(hist["train_rmse_flux"]).astype(float)
    val_rmse_flux = _as_1d(hist["val_rmse_flux"]).astype(float)
    train_rmse_shape = _as_1d(hist["train_rmse_shape"]).astype(float)
    val_rmse_shape = _as_1d(hist["val_rmse_shape"]).astype(float)

    plt.figure(figsize=(8.0, 4.6))
    plt.plot(epoch, train_loss, label="train_loss", lw=2.0)
    plt.plot(epoch, val_loss, label="val_loss", lw=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs validation loss")
    plt.legend(loc="best")
    savefig(out_dir / "00_training_loss_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    axes[0].plot(epoch, train_rmse_shape, label="train", lw=2.0)
    axes[0].plot(epoch, val_rmse_shape, label="val", lw=2.0)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE(shape)")
    axes[0].set_title("Shape RMSE")
    axes[0].legend(loc="best")

    axes[1].plot(epoch, train_rmse_flux, label="train", lw=2.0)
    axes[1].plot(epoch, val_rmse_flux, label="val", lw=2.0)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE(log10 flux)")
    axes[1].set_title("Flux RMSE")
    axes[1].legend(loc="best")

    savefig(out_dir / "00b_training_rmse_curves.png")


def plot_flux_conventional(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[m]
    yp = y_pred[m]
    if yt.size == 0:
        return

    q1 = float(np.percentile(np.concatenate([yt, yp]), 1))
    q99 = float(np.percentile(np.concatenate([yt, yp]), 99))

    plt.figure(figsize=(6.3, 5.8))
    hb = plt.hexbin(yt, yp, gridsize=70, mincnt=1, bins="log", cmap="viridis")
    plt.plot([q1, q99], [q1, q99], color="#D62728", lw=1.8, ls="--", label="y=x")
    plt.xlabel("True log10 flux")
    plt.ylabel("Pred log10 flux")
    plt.title("Flux parity (test)")
    plt.xlim(q1, q99)
    plt.ylim(q1, q99)
    plt.legend(loc="upper left")
    c = plt.colorbar(hb)
    c.set_label("log10(count)")
    savefig(out_dir / "05_flux_parity_hexbin.png")

    resid = yp - yt
    plt.figure(figsize=(8.0, 4.6))
    sns.histplot(resid, bins=100, color="#2B6CB0")
    plt.axvline(0.0, color="k", lw=1.2, ls="--")
    plt.xlabel("Residual in log10 flux (pred - true)")
    plt.ylabel("Count")
    plt.title("Flux residual distribution (test)")
    savefig(out_dir / "06_flux_residual_hist.png")


def plot_per_stamp_rmse(I_true_t: np.ndarray, I_pred_t: np.ndarray, out_dir: Path) -> None:
    rmse = np.sqrt(np.mean((I_pred_t - I_true_t) ** 2, axis=1))
    rmse = rmse[np.isfinite(rmse)]
    if rmse.size == 0:
        return

    plt.figure(figsize=(8.0, 4.6))
    sns.histplot(rmse, bins=100, color="#1F4E79")
    plt.xlabel("Per-stamp RMSE (raw intensity)")
    plt.ylabel("Count")
    plt.title("Per-stamp reconstruction RMSE (test)")
    savefig(out_dir / "07_per_stamp_rmse_hist_raw.png")


def plot_stamp_triplets(I_true_t: np.ndarray, I_pred_t: np.ndarray, chi2nu_t: np.ndarray, stamp_pix: int, out_dir: Path, n_rows: int) -> None:
    idx = top_outliers(chi2nu_t, n_rows)
    n = int(idx.size)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(10.5, 2.7 * n), constrained_layout=True)
    if n == 1:
        axes = np.asarray(axes).reshape(1, 3)

    for r, k in enumerate(idx):
        timg = I_true_t[k].reshape(stamp_pix, stamp_pix)
        pimg = I_pred_t[k].reshape(stamp_pix, stamp_pix)
        dimg = pimg - timg

        vmin = float(np.percentile(np.concatenate([timg.ravel(), pimg.ravel()]), 1))
        vmax = float(np.percentile(np.concatenate([timg.ravel(), pimg.ravel()]), 99))
        dv = float(np.percentile(np.abs(dimg), 99))
        dv = dv if dv > 0 else 1e-6

        im0 = axes[r, 0].imshow(timg, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        im1 = axes[r, 1].imshow(pimg, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        im2 = axes[r, 2].imshow(dimg, origin="lower", cmap="RdBu_r", vmin=-dv, vmax=dv)

        axes[r, 0].set_title("TRUE" if r == 0 else "", fontsize=10)
        axes[r, 1].set_title("PRED" if r == 0 else "", fontsize=10)
        axes[r, 2].set_title("RESID" if r == 0 else "", fontsize=10)
        axes[r, 0].set_ylabel(f"#{r} chi2={chi2nu_t[k]:.2g}", fontsize=9)
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

        if r == 0:
            c0 = fig.colorbar(im0, ax=axes[:, 0], fraction=0.02, pad=0.01)
            c0.set_label("intensity")
            c1 = fig.colorbar(im2, ax=axes[:, 2], fraction=0.02, pad=0.01)
            c1.set_label("residual")

    fig.suptitle("Rows of TRUE / PRED / RESID (top chi2 outliers)", y=1.005)
    savefig(out_dir / "08_rows_true_pred_resid_top_outliers.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run id under output/ml_runs/")
    ap.add_argument("--out_dir", default="", help="Optional custom output plot dir")
    ap.add_argument("--importance_max_samples", type=int, default=CFG.importance_max_samples)
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument(
        "--recon_flux_mode",
        type=str,
        default="pred",
        choices=["pred", "true"],
        help="How to scale predicted shape for raw reconstruction: pred flux or true flux.",
    )
    ap.add_argument("--triplet_rows_n", type=int, default=CFG.triplet_rows_n)
    args = ap.parse_args()

    run_root = CFG.base_dir / "output" / "ml_runs" / args.run
    if not run_root.exists():
        raise FileNotFoundError(f"Run not found: {run_root}")

    with open(run_root / "config.json", "r", encoding="utf-8") as f:
        train_cfg = json.load(f)
    with open(run_root / "metrics.json", "r", encoding="utf-8") as f:
        train_metrics = json.load(f)

    manifest = load_manifest(Path(train_cfg["manifest_npz"]))

    date_tag = time.strftime("%Y%m%d")
    if str(args.out_dir).strip():
        plot_dir = Path(args.out_dir)
    else:
        plot_dir = CFG.base_dir / "plots" / "qa" / "model_decision" / f"nn_true_pred_resid_review_{date_tag}" / args.run
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_feat = int(manifest["n_features"])
    D = int(manifest["D"])
    stamp_pix = int(manifest["stamp_pix"])
    feature_names = list(manifest.get("feature_cols", [f"feat_{i}" for i in range(n_feat)]))

    n_val_all = int(manifest["n_val"])
    n_test_all = int(manifest["n_test"])
    n_val = int(min(n_val_all, int(train_metrics.get("n_val", n_val_all))))
    n_test = int(min(n_test_all, int(train_metrics.get("n_test", n_test_all))))

    X_val = np.memmap(manifest["X_val_path"], dtype="float32", mode="r", shape=(n_val_all, n_feat))
    Yshape_val = np.memmap(manifest["Yshape_val_path"], dtype="float32", mode="r", shape=(n_val_all, D))
    Yflux_val = np.memmap(manifest["Yflux_val_path"], dtype="float32", mode="r", shape=(n_val_all,))

    X_test = np.memmap(manifest["X_test_path"], dtype="float32", mode="r", shape=(n_test_all, n_feat))
    Yshape_test = np.memmap(manifest["Yshape_test_path"], dtype="float32", mode="r", shape=(n_test_all, D))
    Yflux_test = np.memmap(manifest["Yflux_test_path"], dtype="float32", mode="r", shape=(n_test_all,))

    Xv = np.asarray(X_val[:n_val], dtype=np.float32)
    Ysv = np.asarray(Yshape_val[:n_val], dtype=np.float32)
    Yfv = np.asarray(Yflux_val[:n_val], dtype=np.float32)
    Xt = np.asarray(X_test[:n_test], dtype=np.float32)
    Yst = np.asarray(Yshape_test[:n_test], dtype=np.float32)
    Yft = np.asarray(Yflux_test[:n_test], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShapeFluxMLP(
        n_feat=n_feat,
        D=D,
        hidden_dim=int(train_cfg["hidden_dim"]),
        depth=int(train_cfg["depth"]),
        dropout=float(train_cfg["dropout"]),
    ).to(device)

    best_model = run_root / "best_model.pt"
    state = torch.load(best_model if best_model.exists() else (run_root / "last_model.pt"), map_location="cpu")
    model.load_state_dict(state)

    pred_shape_t, pred_flux_t = predict_all(model, Xt, batch_size=int(args.batch_size), device=device)
    pred_shape_v, pred_flux_v = predict_all(model, Xv, batch_size=int(args.batch_size), device=device)

    I_true_t = raw_from_shape_flux(Yst, Yft)
    I_true_v = raw_from_shape_flux(Ysv, Yfv)
    if args.recon_flux_mode == "true":
        I_pred_v = raw_from_shape_and_true_flux(pred_shape_v, Yfv)
        I_pred_t = raw_from_shape_and_true_flux(pred_shape_t, Yft)
    else:
        I_pred_v = raw_from_shape_flux(pred_shape_v, pred_flux_v)
        I_pred_t = raw_from_shape_flux(pred_shape_t, pred_flux_t)

    sigma_bg_t = sigma_bg_from_border_mad(I_true_t, stamp_pix, CFG.border_w, CFG.sigma_bg_floor)
    chi2nu_t = chi2nu_sqrt_Itrue(I_true_t, I_pred_t, sigma_bg_t, sigma0=CFG.sigma0, alpha=CFG.alpha)

    plot_learning_curves(run_root, plot_dir)
    plot_chi2_hist(chi2nu_t, plot_dir)
    plot_mean_maps(I_true_t, I_pred_t, stamp_pix, plot_dir, max_n=CFG.mean_maps_max)
    plot_resid_examples(I_true_t, I_pred_t, chi2nu_t, stamp_pix, plot_dir, n=CFG.examples_n)
    plot_stamp_triplets(I_true_t, I_pred_t, chi2nu_t, stamp_pix, plot_dir, n_rows=max(1, int(args.triplet_rows_n)))
    plot_flux_conventional(Yft, pred_flux_t, plot_dir)
    plot_per_stamp_rmse(I_true_t, I_pred_t, plot_dir)

    imp = permutation_importance(
        model=model,
        X_val=Xv,
        Yshape_val=Ysv,
        Yflux_val=Yfv,
        feature_names=feature_names,
        max_samples=int(args.importance_max_samples),
        batch_size=int(args.batch_size),
        device=device,
    )

    with open(plot_dir / "04_feature_importance_permutation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "delta_rmse_raw", "delta_rmse_shape", "delta_rmse_flux"])
        w.writeheader()
        for r in imp:
            w.writerow(r)

    plot_importance(imp, plot_dir)

    v = chi2nu_t[np.isfinite(chi2nu_t)]
    v = v[v > 0]

    metrics_val = compute_metrics(Ysv, pred_shape_v, Yfv, pred_flux_v)
    metrics_test = compute_metrics(Yst, pred_shape_t, Yft, pred_flux_t)
    metrics_val_recon = {"rmse_raw_recon": float(np.sqrt(np.mean((I_pred_v - I_true_v) ** 2)))}
    metrics_test_recon = {"rmse_raw_recon": float(np.sqrt(np.mean((I_pred_t - I_true_t) ** 2)))}

    summary = {
        "run": args.run,
        "n_val": n_val,
        "n_test": n_test,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "metrics_val_recon_mode": metrics_val_recon,
        "metrics_test_recon_mode": metrics_test_recon,
        "recon_flux_mode": args.recon_flux_mode,
        "chi2nu_test": {
            "median": float(np.median(v)) if v.size else float("nan"),
            "p90": float(np.percentile(v, 90)) if v.size else float("nan"),
            "p99": float(np.percentile(v, 99)) if v.size else float("nan"),
        },
    }

    with open(plot_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("DONE. Outputs in:", plot_dir)


if __name__ == "__main__":
    main()
