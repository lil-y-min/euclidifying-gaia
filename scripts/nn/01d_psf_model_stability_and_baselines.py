#!/usr/bin/env python3
"""
Model stability pass for PSF/non-PSF morphology classification.

Includes:
- confidently wrong test-sample audit (high logloss contributors)
- baseline comparison: logistic vs random forest vs xgboost
- feature collinearity diagnostics via VIF
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

import xgboost as xgb


MORPH_COLS = [
    "m_concentration_r2_r6",
    "m_asymmetry_180",
    "m_ellipticity",
    "m_peak_sep_pix",
    "m_edge_flux_frac",
    "m_peak_ratio_2over1",
]


@dataclass
class RobustScaler1D:
    median: float
    scale: float


def robust_fit(x: np.ndarray) -> RobustScaler1D:
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    return RobustScaler1D(median=med, scale=float(1.4826 * mad + 1e-9))


def robust_transform(x: np.ndarray, s: RobustScaler1D, clip_abs: float = 8.0) -> np.ndarray:
    z = (x - s.median) / s.scale
    return np.clip(z, -clip_abs, clip_abs)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(y: np.ndarray, p: np.ndarray, name: str) -> Dict[str, float | str]:
    return {
        "model": name,
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9))),
    }


def compute_vif(X: np.ndarray, cols: List[str]) -> pd.DataFrame:
    n, d = X.shape
    rows = []
    for j in range(d):
        yj = X[:, j]
        others = [k for k in range(d) if k != j]
        Xo = X[:, others]
        # Add intercept.
        A = np.column_stack([np.ones(n, dtype=float), Xo])
        beta, *_ = np.linalg.lstsq(A, yj, rcond=None)
        yhat = A @ beta
        ss_res = float(np.sum((yj - yhat) ** 2))
        ss_tot = float(np.sum((yj - np.mean(yj)) ** 2))
        if ss_tot <= 0:
            r2 = 1.0
        else:
            r2 = max(0.0, min(0.999999, 1.0 - ss_res / ss_tot))
        vif = float(1.0 / max(1e-6, 1.0 - r2))
        rows.append({"metric": cols[j], "r2_against_others": r2, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--z_clip", type=float, default=8.0)
    ap.add_argument("--hard_case_top", type=int, default=250)
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base_dir / "report" / "model_decision" / "20260220_psf_stability_pass")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    need = ["label", "split_code", "source_id"] + MORPH_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    extra_num = [c for c in ["confidence", "ruwe", "ipd_frac_multi_peak", "score_psf_like"] if c in df.columns]
    for c in extra_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=need).copy()
    df["label"] = df["label"].astype(int)
    df["split_code"] = df["split_code"].astype(int)

    tr = df[df["split_code"] == 0].copy()
    va = df[df["split_code"] == 1].copy()
    te = df[df["split_code"] == 2].copy()
    if tr.empty or te.empty:
        raise RuntimeError("Need non-empty train(split=0) and test(split=2)")

    scalers: Dict[str, RobustScaler1D] = {c: robust_fit(tr[c].to_numpy(dtype=float)) for c in MORPH_COLS}

    def mk(dsub: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            robust_transform(dsub[c].to_numpy(dtype=float), scalers[c], clip_abs=float(args.z_clip)) for c in MORPH_COLS
        ])

    Xtr = mk(tr)
    Xva = mk(va) if not va.empty else mk(tr)
    Xte = mk(te)
    ytr = tr["label"].to_numpy(dtype=int)
    yva = va["label"].to_numpy(dtype=int) if not va.empty else tr["label"].to_numpy(dtype=int)
    yte = te["label"].to_numpy(dtype=int)

    # Model 1: logistic (linear baseline)
    logreg = LogisticRegression(
        penalty="l2",
        C=1000.0,  # very light regularization
        solver="lbfgs",
        max_iter=2000,
    )
    logreg.fit(Xtr, ytr)
    p_log_tr = logreg.predict_proba(Xtr)[:, 1]
    p_log_va = logreg.predict_proba(Xva)[:, 1]
    p_log_te = logreg.predict_proba(Xte)[:, 1]

    # Model 2: random forest
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(Xtr, ytr)
    p_rf_tr = rf.predict_proba(Xtr)[:, 1]
    p_rf_va = rf.predict_proba(Xva)[:, 1]
    p_rf_te = rf.predict_proba(Xte)[:, 1]

    # Model 3: xgboost
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    dte = xgb.DMatrix(Xte, label=yte)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 1.0,
        "tree_method": "hist",
        "seed": 42,
    }
    xgb_model = xgb.train(
        params,
        dtr,
        num_boost_round=600,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    p_xgb_tr = xgb_model.predict(dtr)
    p_xgb_va = xgb_model.predict(dva)
    p_xgb_te = xgb_model.predict(dte)

    rows = []
    for split, y, plist in [
        ("train", ytr, [("logistic", p_log_tr), ("rf", p_rf_tr), ("xgb", p_xgb_tr)]),
        ("val", yva, [("logistic", p_log_va), ("rf", p_rf_va), ("xgb", p_xgb_va)]),
        ("test", yte, [("logistic", p_log_te), ("rf", p_rf_te), ("xgb", p_xgb_te)]),
    ]:
        for name, pp in plist:
            m = compute_metrics(y, pp, name)
            m["split"] = split
            rows.append(m)
    metrics_df = pd.DataFrame(rows).sort_values(["split", "auc"], ascending=[True, False])
    metrics_df.to_csv(out_dir / "baseline_metrics_compare.csv", index=False)

    # Confidently wrong audit on test using calibrated logistic baseline.
    p = np.clip(p_log_te, 1e-9, 1 - 1e-9)
    ll = -(yte * np.log(p) + (1 - yte) * np.log(1 - p))
    yhat = (p >= 0.5).astype(int)
    wrong = (yhat != yte)
    conf_wrong = wrong & (((yte == 1) & (p >= 0.8)) | ((yte == 0) & (p <= 0.2)))

    hard = te.copy()
    hard["p_nonpsf_logistic"] = p
    hard["pred_label"] = yhat
    hard["is_wrong"] = wrong.astype(int)
    hard["is_confidently_wrong"] = conf_wrong.astype(int)
    hard["sample_logloss"] = ll
    hard = hard.sort_values("sample_logloss", ascending=False)

    topn = int(max(1, args.hard_case_top))
    top_hard = hard.head(topn).copy()
    cols_out = [
        "source_id", "label", "pred_label", "p_nonpsf_logistic", "sample_logloss", "is_wrong", "is_confidently_wrong"
    ] + [c for c in ["confidence", "ruwe", "ipd_frac_multi_peak", "score_psf_like"] if c in hard.columns] + MORPH_COLS
    top_hard[cols_out].to_csv(out_dir / "confidently_wrong_top_samples_test.csv", index=False)

    hard_summary = {
        "n_test": int(len(te)),
        "n_wrong": int(np.sum(wrong)),
        "n_confidently_wrong": int(np.sum(conf_wrong)),
        "wrong_rate": float(np.mean(wrong)),
        "confidently_wrong_rate": float(np.mean(conf_wrong)),
        "mean_logloss_all": float(np.mean(ll)),
        "mean_logloss_wrong": float(np.mean(ll[wrong])) if np.any(wrong) else float("nan"),
    }
    (out_dir / "confidently_wrong_summary.json").write_text(json.dumps(hard_summary, indent=2), encoding="utf-8")

    # VIF diagnostics on train robust-z features.
    vif_df = compute_vif(Xtr, MORPH_COLS)
    vif_df.to_csv(out_dir / "vif_train_morphology.csv", index=False)

    # Plot 1: metrics bar on test
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    fig, axs = plt.subplots(1, 3, figsize=(10.8, 3.8), constrained_layout=True)
    for ax, metric in zip(axs, ["auc", "ap", "logloss"]):
        d = test_df.sort_values(metric, ascending=(metric == "logloss"))
        ax.bar(d["model"], d[metric])
        ax.set_title(f"test {metric}")
    savefig(plot_dir / "test_metrics_models.png")

    # Plot 2: sample-logloss histogram split correct vs wrong
    plt.figure(figsize=(7.5, 4.8))
    bins = np.linspace(0, np.quantile(ll, 0.995), 60)
    plt.hist(ll[~wrong], bins=bins, alpha=0.6, density=True, label="correct")
    if np.any(wrong):
        plt.hist(ll[wrong], bins=bins, alpha=0.6, density=True, label="wrong")
    plt.xlabel("per-sample logloss (test, logistic)")
    plt.ylabel("density")
    plt.title("Where mistakes concentrate")
    plt.legend()
    savefig(plot_dir / "sample_logloss_correct_vs_wrong.png")

    # Plot 3: VIF bar
    plt.figure(figsize=(8.2, 4.8))
    d = vif_df.sort_values("vif", ascending=False)
    plt.bar(d["metric"], d["vif"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("VIF")
    plt.title("Feature collinearity (train robust-z)")
    savefig(plot_dir / "vif_bar.png")

    # Plot 4: probability scatter logistic vs xgb (test)
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(p_log_te, p_xgb_te, s=6, alpha=0.25)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("p_nonpsf logistic")
    plt.ylabel("p_nonpsf xgb")
    plt.title("Test probability agreement")
    savefig(plot_dir / "prob_scatter_logistic_vs_xgb_test.png")

    summary = {
        "labels_csv": str(args.labels_csv),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "z_clip": float(args.z_clip),
        "hard_case_top": int(topn),
        "best_test_model_by_auc": str(test_df.sort_values("auc", ascending=False).iloc[0]["model"]),
        "best_test_model_by_logloss": str(test_df.sort_values("logloss", ascending=True).iloc[0]["model"]),
        "hard_case_summary": hard_summary,
    }
    (out_dir / "stability_pass_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
