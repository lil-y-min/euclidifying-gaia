#!/usr/bin/env python3
"""
Calibrate PSF weak-score weights from existing labels and generate diagnostic plots.

Inputs:
- labels CSV from scripts/nn/00_psf_labeling.py (must include morphology columns, label, split_code)

Outputs:
- JSON summaries
- CSV tables
- plot pack (coefficients, score distributions, ROC, PR, calibration, confusion)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MORPH_COLS = [
    "m_concentration_r2_r6",
    "m_asymmetry_180",
    "m_ellipticity",
    "m_peak_sep_pix",
    "m_edge_flux_frac",
    "m_peak_ratio_2over1",
]

HAND_WEIGHTS = {
    "m_concentration_r2_r6": +1.3,
    "m_asymmetry_180": -1.0,
    "m_ellipticity": -0.8,
    "m_peak_sep_pix": -1.1,
    "m_edge_flux_frac": -0.6,
    "m_peak_ratio_2over1": -0.5,
}


@dataclass
class RobustScaler1D:
    median: float
    scale: float


def robust_fit(x: np.ndarray) -> RobustScaler1D:
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    scale = float(1.4826 * mad + 1e-9)
    return RobustScaler1D(median=med, scale=scale)


def robust_transform(x: np.ndarray, s: RobustScaler1D, clip_abs: float = 8.0) -> np.ndarray:
    z = (x - s.median) / s.scale
    return np.clip(z, -clip_abs, clip_abs)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logreg_l2(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-3,
    lr: float = 0.05,
    max_iter: int = 5000,
    tol: float = 1e-8,
    freeze_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, float, List[float]]:
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    hist: List[float] = []

    if freeze_mask is None:
        freeze_mask = np.zeros(d, dtype=bool)
    else:
        freeze_mask = np.asarray(freeze_mask, dtype=bool)
        if freeze_mask.shape != (d,):
            raise ValueError("freeze_mask has wrong shape")

    for _ in range(max_iter):
        z = X @ w + b
        p = sigmoid(z)

        # Binary cross-entropy + L2 penalty.
        loss = -np.mean(y * np.log(np.clip(p, 1e-9, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-9, 1.0)))
        loss += 0.5 * l2 * float(np.sum(w * w))
        hist.append(float(loss))

        g_w = (X.T @ (p - y)) / n + l2 * w
        g_w = np.where(freeze_mask, 0.0, g_w)
        g_b = float(np.mean(p - y))

        step_norm = float(np.sqrt(np.sum(g_w * g_w) + g_b * g_b))
        w -= lr * g_w
        b -= lr * g_b

        if step_norm < tol:
            break

    return w, b, hist


def binary_metrics(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y_true, dtype=int)
    yhat = (p >= thr).astype(int)

    tp = int(np.sum((y == 1) & (yhat == 1)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))

    acc = float((tp + tn) / max(1, len(y)))
    prec = float(tp / max(1, tp + fp))
    rec = float(tp / max(1, tp + fn))
    f1 = float(2 * prec * rec / max(1e-12, prec + rec))
    bce = float(-np.mean(y * np.log(np.clip(p, 1e-9, 1.0)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1.0))))

    return {
        "n": int(len(y)),
        "threshold": float(thr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "logloss": bce,
    }


def roc_curve_manual(y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    order = np.argsort(-p)
    y = y[order]
    p = p[order]

    P = max(1, int(np.sum(y == 1)))
    N = max(1, int(np.sum(y == 0)))

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)

    tpr = tps / P
    fpr = fps / N

    # add origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thr = np.concatenate([[1.0], p])

    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, thr, auc


def pr_curve_manual(y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    order = np.argsort(-p)
    y = y[order]
    p = p[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    P = max(1, int(np.sum(y == 1)))

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / P

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    thr = np.concatenate([[1.0], p])

    ap = float(np.trapezoid(precision, recall))
    return recall, precision, thr, ap


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i < n_bins - 1:
            m = (p >= lo) & (p < hi)
        else:
            m = (p >= lo) & (p <= hi)
        n = int(np.sum(m))
        if n == 0:
            rows.append({"bin": i, "p_lo": lo, "p_hi": hi, "n": 0, "p_mean": np.nan, "y_mean": np.nan})
            continue
        rows.append(
            {
                "bin": i,
                "p_lo": lo,
                "p_hi": hi,
                "n": n,
                "p_mean": float(np.mean(p[m])),
                "y_mean": float(np.mean(y[m])),
            }
        )
    return pd.DataFrame(rows)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max_iter", type=int, default=5000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--z_clip", type=float, default=8.0)
    ap.add_argument("--min_scale", type=float, default=1e-4)
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base_dir / "report" / "model_decision" / "20260220_psf_score_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    need = ["label", "split_code"] + MORPH_COLS
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns in labels csv: {miss}")

    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need).copy()
    df["label"] = df["label"].astype(int)
    df["split_code"] = df["split_code"].astype(int)

    tr = df[df["split_code"] == 0].copy()
    va = df[df["split_code"] == 1].copy()
    te = df[df["split_code"] == 2].copy()
    if tr.empty or te.empty:
        raise RuntimeError("Need non-empty train(split=0) and test(split=2)")

    # Fit robust scalers only on train.
    scalers: Dict[str, RobustScaler1D] = {}
    for c in MORPH_COLS:
        scalers[c] = robust_fit(tr[c].to_numpy(dtype=float))

    def make_z(dsub: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in MORPH_COLS:
            cols.append(robust_transform(dsub[c].to_numpy(dtype=float), scalers[c], clip_abs=float(args.z_clip)))
        return np.column_stack(cols)

    Xtr = make_z(tr)
    Xva = make_z(va) if not va.empty else make_z(tr)
    Xte = make_z(te)
    ytr = tr["label"].to_numpy(dtype=int)
    yva = va["label"].to_numpy(dtype=int) if not va.empty else tr["label"].to_numpy(dtype=int)
    yte = te["label"].to_numpy(dtype=int)

    # Handcrafted score converted to non-PSF probability.
    hand_w = np.array([HAND_WEIGHTS[c] for c in MORPH_COLS], dtype=float)
    hand_p_tr = sigmoid(-(Xtr @ hand_w))
    hand_p_va = sigmoid(-(Xva @ hand_w))
    hand_p_te = sigmoid(-(Xte @ hand_w))

    # Freeze near-constant metrics to avoid unstable coefficients.
    feature_scales = np.array([scalers[c].scale for c in MORPH_COLS], dtype=float)
    freeze_mask = feature_scales < float(args.min_scale)

    # Fit calibrated logistic model.
    w, b, loss_hist = fit_logreg_l2(
        Xtr,
        ytr,
        l2=float(args.l2),
        lr=float(args.lr),
        max_iter=int(args.max_iter),
        freeze_mask=freeze_mask,
    )

    p_tr = sigmoid(Xtr @ w + b)
    p_va = sigmoid(Xva @ w + b)
    p_te = sigmoid(Xte @ w + b)

    thr = float(args.threshold)
    metrics = {
        "handcrafted_train": binary_metrics(ytr, hand_p_tr, thr=thr),
        "handcrafted_val": binary_metrics(yva, hand_p_va, thr=thr),
        "handcrafted_test": binary_metrics(yte, hand_p_te, thr=thr),
        "calibrated_train": binary_metrics(ytr, p_tr, thr=thr),
        "calibrated_val": binary_metrics(yva, p_va, thr=thr),
        "calibrated_test": binary_metrics(yte, p_te, thr=thr),
    }

    fpr_h, tpr_h, thr_h, auc_h = roc_curve_manual(yte, hand_p_te)
    fpr_c, tpr_c, thr_c, auc_c = roc_curve_manual(yte, p_te)
    rec_h, pre_h, _, ap_h = pr_curve_manual(yte, hand_p_te)
    rec_c, pre_c, _, ap_c = pr_curve_manual(yte, p_te)

    # Save coeff table.
    coef_df = pd.DataFrame(
        {
            "metric": MORPH_COLS,
            "robust_scale_train": feature_scales,
            "is_frozen": freeze_mask.astype(int),
            "handcrafted_weight": hand_w,
            "calibrated_weight": w,
            "weight_delta": w - hand_w,
        }
    ).sort_values("calibrated_weight", ascending=False)
    coef_df.to_csv(out_dir / "calibrated_weights_comparison.csv", index=False)

    # Save probabilities for audit.
    pd.DataFrame(
        {
            "split_code": te["split_code"].to_numpy(dtype=int),
            "y_true": yte,
            "p_nonpsf_handcrafted": hand_p_te,
            "p_nonpsf_calibrated": p_te,
        }
    ).to_csv(out_dir / "test_probabilities_handcrafted_vs_calibrated.csv", index=False)

    # Calibration bins.
    cal_h = calibration_bins(yte, hand_p_te, n_bins=15)
    cal_c = calibration_bins(yte, p_te, n_bins=15)
    cal_h.to_csv(out_dir / "calibration_bins_handcrafted_test.csv", index=False)
    cal_c.to_csv(out_dir / "calibration_bins_calibrated_test.csv", index=False)

    # Summary json.
    summary = {
        "labels_csv": str(args.labels_csv),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "l2": float(args.l2),
        "lr": float(args.lr),
        "max_iter": int(args.max_iter),
        "threshold": float(thr),
        "z_clip": float(args.z_clip),
        "min_scale": float(args.min_scale),
        "frozen_features": [c for c, m in zip(MORPH_COLS, freeze_mask.tolist()) if m],
        "intercept_calibrated": float(b),
        "auc_test_handcrafted": float(auc_h),
        "auc_test_calibrated": float(auc_c),
        "ap_test_handcrafted": float(ap_h),
        "ap_test_calibrated": float(ap_c),
        "metrics": metrics,
    }
    (out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots.
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(loss_hist, lw=1.5)
    plt.xlabel("iteration")
    plt.ylabel("train loss")
    plt.title("Calibrated logistic training loss")
    savefig(plot_dir / "loss_curve.png")

    cdf = coef_df.sort_values("metric")
    x = np.arange(len(cdf))
    wbar = 0.38
    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x - wbar / 2, cdf["handcrafted_weight"], width=wbar, label="handcrafted")
    plt.bar(x + wbar / 2, cdf["calibrated_weight"], width=wbar, label="calibrated")
    plt.xticks(x, cdf["metric"], rotation=25, ha="right")
    plt.ylabel("weight")
    plt.title("Morphology score weights: handcrafted vs calibrated")
    plt.legend()
    savefig(plot_dir / "weights_comparison_bar.png")

    plt.figure(figsize=(7.0, 5.4))
    plt.plot(fpr_h, tpr_h, label=f"handcrafted AUC={auc_h:.4f}")
    plt.plot(fpr_c, tpr_c, label=f"calibrated AUC={auc_c:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (test)")
    plt.legend(loc="lower right")
    savefig(plot_dir / "roc_test.png")

    plt.figure(figsize=(7.0, 5.4))
    plt.plot(rec_h, pre_h, label=f"handcrafted AP={ap_h:.4f}")
    plt.plot(rec_c, pre_c, label=f"calibrated AP={ap_c:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (test)")
    plt.legend(loc="lower left")
    savefig(plot_dir / "pr_test.png")

    # Probability calibration plot.
    plt.figure(figsize=(7.0, 5.4))
    m = cal_h["n"] > 0
    plt.plot(cal_h.loc[m, "p_mean"], cal_h.loc[m, "y_mean"], marker="o", label="handcrafted")
    m = cal_c["n"] > 0
    plt.plot(cal_c.loc[m, "p_mean"], cal_c.loc[m, "y_mean"], marker="o", label="calibrated")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed non-PSF frequency")
    plt.title("Calibration curve (test)")
    plt.legend(loc="upper left")
    savefig(plot_dir / "calibration_curve_test.png")

    # Probability distribution by class.
    plt.figure(figsize=(8.4, 5.4))
    bins = np.linspace(0, 1, 41)
    plt.hist(hand_p_te[yte == 0], bins=bins, alpha=0.45, density=True, label="handcrafted | y=0")
    plt.hist(hand_p_te[yte == 1], bins=bins, alpha=0.45, density=True, label="handcrafted | y=1")
    plt.hist(p_te[yte == 0], bins=bins, histtype="step", lw=1.8, density=True, label="calibrated | y=0")
    plt.hist(p_te[yte == 1], bins=bins, histtype="step", lw=1.8, density=True, label="calibrated | y=1")
    plt.xlabel("P(non-PSF)")
    plt.ylabel("density")
    plt.title("Test probability distributions by class")
    plt.legend(loc="upper center", ncol=2, fontsize=8)
    savefig(plot_dir / "probability_distributions_test.png")

    # Score distributions (signed score for interpretability).
    hand_score_te = Xte @ hand_w
    cal_score_te = Xte @ w + b
    plt.figure(figsize=(8.4, 5.4))
    bins = np.linspace(np.quantile(np.concatenate([hand_score_te, cal_score_te]), 0.01), np.quantile(np.concatenate([hand_score_te, cal_score_te]), 0.99), 60)
    plt.hist(hand_score_te[yte == 0], bins=bins, alpha=0.35, density=True, label="hand score | y=0")
    plt.hist(hand_score_te[yte == 1], bins=bins, alpha=0.35, density=True, label="hand score | y=1")
    plt.hist(cal_score_te[yte == 0], bins=bins, histtype="step", lw=1.8, density=True, label="cal score | y=0")
    plt.hist(cal_score_te[yte == 1], bins=bins, histtype="step", lw=1.8, density=True, label="cal score | y=1")
    plt.xlabel("linear score")
    plt.ylabel("density")
    plt.title("Score distributions on test")
    plt.legend(loc="upper center", ncol=2, fontsize=8)
    savefig(plot_dir / "score_distributions_test.png")

    # Confusion matrices.
    def conf(y_true: np.ndarray, p: np.ndarray, thrv: float) -> np.ndarray:
        yhat = (p >= thrv).astype(int)
        tp = int(np.sum((y_true == 1) & (yhat == 1)))
        tn = int(np.sum((y_true == 0) & (yhat == 0)))
        fp = int(np.sum((y_true == 0) & (yhat == 1)))
        fn = int(np.sum((y_true == 1) & (yhat == 0)))
        return np.array([[tn, fp], [fn, tp]], dtype=int)

    cm_h = conf(yte, hand_p_te, thr)
    cm_c = conf(yte, p_te, thr)

    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.9), constrained_layout=True)
    for ax, cm, title in [(axs[0], cm_h, "handcrafted"), (axs[1], cm_c, "calibrated")]:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.03, pad=0.02)
    fig.suptitle(f"Confusion matrices at threshold={thr:.2f}")
    savefig(plot_dir / "confusion_test_threshold.png")

    print("DONE")
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
