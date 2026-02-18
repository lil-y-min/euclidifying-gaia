#!/usr/bin/env python3
"""
Mixture-of-experts evaluation for 16D pixels+flux XGB runs.

Given:
- full baseline run (provides full test split arrays + trace_test source_id)
- PSF expert run
- non-PSF expert run
- weak labels CSV (source_id,label)

This script:
1) trains a gating model p = P(non_psf_like | 16D Gaia features)
2) predicts with both experts on the same full test set
3) compares:
   - PSF expert only
   - non-PSF expert only
   - hard route (thresholds)
   - soft mix (recommended)

Outputs:
- metrics table CSV
- gate diagnostics CSV
- summary JSON
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class Cfg:
    base_dir: Path = Path(__file__).resolve().parents[2]
    batch_size: int = 50000
    sigma0: float = 1e-4
    alpha: float = 1.0
    border_w: int = 2
    sigma_bg_floor: float = 1e-6


CFG = Cfg()


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
        "X_test_path": _to_path(d["X_test_path"]),
        "Yshape_test_path": _to_path(d["Yshape_test_path"]),
        "Yflux_test_path": _to_path(d["Yflux_test_path"]),
        "dataset_root": _to_path(d["dataset_root"]),
        "trace_test_csv": _to_path(d["trace_test_csv"]),
        "scaler_npz": _to_path(d["scaler_npz"]) if "scaler_npz" in d.files else None,
    }
    if "feature_cols" in d.files:
        out["feature_cols"] = [str(x) for x in d["feature_cols"].tolist()]
    else:
        out["feature_cols"] = [f"feat_{i}" for i in range(out["n_features"])]
    return out


def load_scaler(scaler_npz: Path, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(scaler_npz, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    y_min = d["y_min"].astype(np.float32)
    y_iqr = d["y_iqr"].astype(np.float32)
    idx = {n: i for i, n in enumerate(names)}
    miss = [c for c in feature_cols if c not in idx]
    if miss:
        raise RuntimeError(f"Scaler missing columns: {miss}")
    ii = np.array([idx[c] for c in feature_cols], dtype=int)
    return y_min[ii], y_iqr[ii]


def raw_from_shape_flux(shape: np.ndarray, flux: np.ndarray, clip_min: float = -6.0, clip_max: float = 8.0) -> np.ndarray:
    f = np.clip(np.asarray(flux, dtype=np.float32), clip_min, clip_max)
    scale = (10.0 ** f).astype(np.float32)
    return np.asarray(shape, dtype=np.float32) * scale[:, None]


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


def predict_shape_all(model_dir: Path, X: np.ndarray, D: int, batch_size: int) -> np.ndarray:
    boosters: List[xgb.Booster] = []
    for j in range(D):
        p = model_dir / f"booster_pix_{j:04d}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing model: {p}")
        b = xgb.Booster()
        b.load_model(str(p))
        boosters.append(b)

    N = X.shape[0]
    out = np.empty((N, D), dtype=np.float32)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        dmat = xgb.DMatrix(np.asarray(X[s:e], dtype=np.float32, order="C"))
        for j, b in enumerate(boosters):
            out[s:e, j] = b.predict(dmat).astype(np.float32, copy=False)
    return out


def normalize_shape_nonneg(shape: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.maximum(np.asarray(shape, dtype=np.float32), 0.0)
    den = np.sum(s, axis=1, keepdims=True)
    den = np.where(den > eps, den, 1.0)
    return s / den


def qstats(v: np.ndarray) -> Dict[str, float]:
    vv = np.asarray(v, dtype=float)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return {"mean": np.nan, "median": np.nan, "p90": np.nan, "p99": np.nan}
    return {
        "mean": float(np.mean(vv)),
        "median": float(np.median(vv)),
        "p90": float(np.quantile(vv, 0.90)),
        "p99": float(np.quantile(vv, 0.99)),
    }


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_triplet_gallery(
    *,
    I_true: np.ndarray,
    I_pred: np.ndarray,
    Z_pred: np.ndarray,
    score: np.ndarray,
    stamp_pix: int,
    out_png: Path,
    mode: str,
    n: int = 12,
    seed: int = 123,
) -> np.ndarray:
    N = int(I_true.shape[0])
    n = int(min(max(1, n), N))
    rng = np.random.default_rng(seed)

    if mode == "best":
        idx = np.argsort(score)[:n]
    elif mode == "worst":
        idx = np.argsort(score)[-n:][::-1]
    elif mode == "random":
        idx = rng.choice(N, size=n, replace=False)
    else:
        raise ValueError(f"Unsupported mode={mode}")

    fig, axes = plt.subplots(n, 3, figsize=(10.5, 2.55 * n), constrained_layout=True)
    if n == 1:
        axes = np.asarray(axes).reshape(1, 3)

    for r, k in enumerate(idx):
        t = I_true[k].reshape(stamp_pix, stamp_pix)
        p = I_pred[k].reshape(stamp_pix, stamp_pix)
        z = Z_pred[k].reshape(stamp_pix, stamp_pix)

        vmin = float(np.percentile(np.concatenate([t.ravel(), p.ravel()]), 1))
        vmax = float(np.percentile(np.concatenate([t.ravel(), p.ravel()]), 99))
        zv = 10.0

        im0 = axes[r, 0].imshow(t, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        im1 = axes[r, 1].imshow(p, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        im2 = axes[r, 2].imshow(z, origin="lower", cmap="RdBu_r", vmin=-zv, vmax=zv)

        if r == 0:
            axes[r, 0].set_title("TRUE")
            axes[r, 1].set_title("PRED")
            axes[r, 2].set_title("Z")
            c0 = fig.colorbar(im0, ax=axes[:, 0], fraction=0.02, pad=0.01)
            c0.set_label("intensity")
            c1 = fig.colorbar(im2, ax=axes[:, 2], fraction=0.02, pad=0.01)
            c1.set_label("z-score [sigma]")

        axes[r, 0].set_ylabel(f"idx={int(k)}\nchi2={float(score[k]):.2g}", fontsize=8)
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle(f"Soft-mix TRUE / PRED / Z ({mode})", y=1.003)
    savefig(out_png)
    return idx.astype(np.int64)


def build_gate_dataset(dataset_root: Path, feature_cols: List[str], labels_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv, usecols=["source_id", "label"]).copy()
    labels["source_id"] = pd.to_numeric(labels["source_id"], errors="coerce")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    labels = labels.dropna(subset=["source_id", "label"]).copy()
    labels["source_id"] = labels["source_id"].astype(np.int64)
    labels["label"] = labels["label"].astype(np.int64)
    labels = labels.drop_duplicates(subset=["source_id"], keep="first")

    fields = sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / "metadata_16d.csv").exists()])
    rows = []
    for f in fields:
        p = f / "metadata_16d.csv"
        usecols = ["source_id", "split_code"] + feature_cols
        df = pd.read_csv(p, usecols=usecols, low_memory=False)
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
        df["split_code"] = pd.to_numeric(df["split_code"], errors="coerce")
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["source_id", "split_code"] + feature_cols)
        df["source_id"] = df["source_id"].astype(np.int64)
        df["split_code"] = df["split_code"].astype(int)
        # keep first occurrence per source in this field to limit duplicates
        df = df.drop_duplicates(subset=["source_id"], keep="first")
        rows.append(df)

    full = pd.concat(rows, ignore_index=True)
    full = full.drop_duplicates(subset=["source_id"], keep="first")
    out = full.merge(labels, on="source_id", how="inner")
    return out


def train_gate_xgb(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[xgb.Booster, Dict[str, float], pd.DataFrame]:
    tr = df[df["split_code"] == 0]
    va = df[df["split_code"] == 1]
    te = df[df["split_code"] == 2]
    if tr.empty or te.empty:
        raise RuntimeError("Gate dataset too small: need at least train(split=0) and test(split=2) labeled rows.")

    Xtr = tr[feature_cols].to_numpy(dtype=np.float32)
    ytr = tr["label"].to_numpy(dtype=np.float32)
    Xva = va[feature_cols].to_numpy(dtype=np.float32) if not va.empty else tr[feature_cols].to_numpy(dtype=np.float32)
    yva = va["label"].to_numpy(dtype=np.float32) if not va.empty else tr["label"].to_numpy(dtype=np.float32)
    Xte = te[feature_cols].to_numpy(dtype=np.float32)
    yte = te["label"].to_numpy(dtype=np.float32)

    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    dte = xgb.DMatrix(Xte, label=yte)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "tree_method": "hist",
        "seed": 11,
    }
    gate = xgb.train(params, dtr, num_boost_round=500, evals=[(dtr, "train"), (dva, "val")], early_stopping_rounds=40, verbose_eval=False)

    pte = gate.predict(dte).astype(np.float32)
    pred = (pte >= 0.5).astype(np.int32)
    acc = float(np.mean(pred == yte.astype(np.int32)))
    logloss = float(-np.mean(yte * np.log(np.clip(pte, 1e-7, 1 - 1e-7)) + (1 - yte) * np.log(np.clip(1 - pte, 1e-7, 1 - 1e-7))))
    diag = {
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "test_accuracy@0.5": acc,
        "test_logloss": logloss,
    }
    diag_df = pd.DataFrame({"y_true": yte, "p_nonpsf": pte})
    return gate, diag, diag_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_run", required=True, help="Full-set 16D run (used for test set + trace).")
    ap.add_argument("--psf_run", required=True, help="PSF expert run.")
    ap.add_argument("--nonpsf_run", required=True, help="non-PSF expert run.")
    ap.add_argument("--labels_csv", required=True, help="Weak labels CSV (source_id,label).")
    ap.add_argument("--hard_low", type=float, default=0.10)
    ap.add_argument("--hard_high", type=float, default=0.90)
    ap.add_argument("--gallery_n", type=int, default=12)
    ap.add_argument("--out_dir", default="", help="Optional custom output dir.")
    args = ap.parse_args()

    full_root = CFG.base_dir / "output" / "ml_runs" / args.full_run
    psf_root = CFG.base_dir / "output" / "ml_runs" / args.psf_run
    non_root = CFG.base_dir / "output" / "ml_runs" / args.nonpsf_run
    labels_csv = Path(args.labels_csv)
    for p in [full_root, psf_root, non_root]:
        if not p.exists():
            raise FileNotFoundError(f"Run not found: {p}")

    mf = load_manifest(full_root / "manifest_arrays.npz")
    mp = load_manifest(psf_root / "manifest_arrays.npz")
    mn = load_manifest(non_root / "manifest_arrays.npz")
    if int(mf["n_features"]) != int(mp["n_features"]) or int(mf["n_features"]) != int(mn["n_features"]):
        raise RuntimeError("Feature dimension mismatch between runs.")
    if int(mf["D"]) != int(mp["D"]) or int(mf["D"]) != int(mn["D"]):
        raise RuntimeError("Output D mismatch between runs.")

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (CFG.base_dir / "report" / "model_decision" / "20260217_psf_split_experiment")
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(mf["feature_cols"])
    gate_df = build_gate_dataset(dataset_root=Path(mf["dataset_root"]), feature_cols=feature_cols, labels_csv=labels_csv)
    if mf.get("scaler_npz") is not None:
        smin, siqr = load_scaler(Path(mf["scaler_npz"]), feature_cols=feature_cols)
        Xg = gate_df[feature_cols].to_numpy(dtype=np.float32)
        Xg = (Xg - smin[None, :]) / siqr[None, :]
        gate_df.loc[:, feature_cols] = Xg
    gate, gate_diag, gate_diag_df = train_gate_xgb(gate_df, feature_cols=feature_cols)
    gate.save_model(str(out_dir / "gate_xgb_16d.json"))
    gate_diag_df.to_csv(out_dir / "gate_test_probabilities.csv", index=False)

    n_test = int(mf["n_test"])
    n_feat = int(mf["n_features"])
    D = int(mf["D"])
    stamp_pix = int(mf["stamp_pix"])

    X_test = np.memmap(mf["X_test_path"], dtype="float32", mode="r", shape=(n_test, n_feat))
    Yshape = np.memmap(mf["Yshape_test_path"], dtype="float32", mode="r", shape=(n_test, D))
    Yflux = np.memmap(mf["Yflux_test_path"], dtype="float32", mode="r", shape=(n_test,))
    X = np.asarray(X_test, dtype=np.float32)
    ys_true = np.asarray(Yshape, dtype=np.float32)
    yf_true = np.asarray(Yflux, dtype=np.float32)

    psf_shape = predict_shape_all(psf_root / "models", X, D=D, batch_size=CFG.batch_size)
    non_shape = predict_shape_all(non_root / "models", X, D=D, batch_size=CFG.batch_size)

    psf_shape = normalize_shape_nonneg(psf_shape)
    non_shape = normalize_shape_nonneg(non_shape)

    p = gate.predict(xgb.DMatrix(X)).astype(np.float32)
    p = np.clip(p, 0.0, 1.0)
    p_col = p[:, None]

    soft_shape = (1.0 - p_col) * psf_shape + p_col * non_shape
    hard_shape = np.where(p_col >= float(args.hard_high), non_shape, np.where(p_col <= float(args.hard_low), psf_shape, soft_shape))

    I_true = raw_from_shape_flux(ys_true, yf_true)
    I_psf = raw_from_shape_flux(psf_shape, yf_true)
    I_non = raw_from_shape_flux(non_shape, yf_true)
    I_soft = raw_from_shape_flux(soft_shape, yf_true)
    I_hard = raw_from_shape_flux(hard_shape, yf_true)

    sigma_bg = sigma_bg_from_border_mad(I_true, stamp_pix=stamp_pix, w=CFG.border_w, floor=CFG.sigma_bg_floor)
    c_psf = chi2nu_sqrt_Itrue(I_true, I_psf, sigma_bg, sigma0=CFG.sigma0, alpha=CFG.alpha)
    c_non = chi2nu_sqrt_Itrue(I_true, I_non, sigma_bg, sigma0=CFG.sigma0, alpha=CFG.alpha)
    c_soft = chi2nu_sqrt_Itrue(I_true, I_soft, sigma_bg, sigma0=CFG.sigma0, alpha=CFG.alpha)
    c_hard = chi2nu_sqrt_Itrue(I_true, I_hard, sigma_bg, sigma0=CFG.sigma0, alpha=CFG.alpha)
    sig2 = (sigma_bg.astype(np.float32) ** 2)[:, None] + float(CFG.alpha) * np.maximum(I_true, 0.0) + float(CFG.sigma0) ** 2
    sig = np.sqrt(np.maximum(sig2, 1e-30)).astype(np.float32)
    Z_soft = ((I_soft - I_true) / sig).astype(np.float32)

    def shape_rmse_row(pred: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean((pred - ys_true) ** 2, axis=1))

    rows = []
    for name, chi, shp in [
        ("psf_only", c_psf, shape_rmse_row(psf_shape)),
        ("nonpsf_only", c_non, shape_rmse_row(non_shape)),
        ("soft_mix", c_soft, shape_rmse_row(soft_shape)),
        ("hard_route", c_hard, shape_rmse_row(hard_shape)),
    ]:
        r = {"mode": name, "n_test": int(n_test)}
        for k, v in qstats(chi).items():
            r[f"chi2nu_{k}"] = v
        for k, v in qstats(shp).items():
            r[f"shape_rmse_{k}"] = v
        rows.append(r)

    pd.DataFrame(rows).to_csv(out_dir / "moe_metrics_test_trueflux.csv", index=False)

    # Galleries for visual QA on soft-mix predictions
    idx_best = make_triplet_gallery(
        I_true=I_true,
        I_pred=I_soft,
        Z_pred=Z_soft,
        score=c_soft,
        stamp_pix=stamp_pix,
        out_png=out_dir / "moe_softmix_stamps_best_true_pred_z.png",
        mode="best",
        n=int(args.gallery_n),
    )
    idx_worst = make_triplet_gallery(
        I_true=I_true,
        I_pred=I_soft,
        Z_pred=Z_soft,
        score=c_soft,
        stamp_pix=stamp_pix,
        out_png=out_dir / "moe_softmix_stamps_worst_true_pred_z.png",
        mode="worst",
        n=int(args.gallery_n),
    )
    idx_random = make_triplet_gallery(
        I_true=I_true,
        I_pred=I_soft,
        Z_pred=Z_soft,
        score=c_soft,
        stamp_pix=stamp_pix,
        out_png=out_dir / "moe_softmix_stamps_random_true_pred_z.png",
        mode="random",
        n=int(args.gallery_n),
    )
    pd.DataFrame({
        "best_idx": idx_best,
        "worst_idx": idx_worst,
        "random_idx": idx_random,
    }).to_csv(out_dir / "moe_softmix_gallery_indices.csv", index=False)

    summary = {
        "full_run": args.full_run,
        "psf_run": args.psf_run,
        "nonpsf_run": args.nonpsf_run,
        "hard_low": float(args.hard_low),
        "hard_high": float(args.hard_high),
        "gate_diag": gate_diag,
        "p_nonpsf_stats": qstats(p),
        "gallery_n": int(args.gallery_n),
    }
    with open(out_dir / "moe_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
