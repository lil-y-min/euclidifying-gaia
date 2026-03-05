#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_manifest(run_root: Path) -> dict:
    d = np.load(run_root / "manifest_arrays.npz", allow_pickle=True)
    out = {}
    for k in d.files:
        v = d[k]
        if np.ndim(v) == 0:
            v = v.item()
        out[k] = v
    return out


def open_split_arrays(manifest: dict, split: str):
    n = int(manifest[f"n_{split}"])
    n_feat = int(manifest["n_features"])
    d = int(manifest["D"])
    x = np.memmap(Path(str(manifest[f"X_{split}_path"])), dtype="float32", mode="r", shape=(n, n_feat))
    ys = np.memmap(Path(str(manifest[f"Yshape_{split}_path"])), dtype="float32", mode="r", shape=(n, d))
    yf = np.memmap(Path(str(manifest[f"Yflux_{split}_path"])), dtype="float32", mode="r", shape=(n,))
    return x, ys, yf


def bin_metrics(y_true: np.ndarray, y_hat01: np.ndarray) -> Dict[str, float]:
    t = y_true.astype(bool)
    p = y_hat01.astype(bool)
    tp = float(np.sum(t & p))
    fp = float(np.sum((~t) & p))
    fn = float(np.sum(t & (~p)))
    tn = float(np.sum((~t) & (~p)))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_rate": float(np.mean(p)),
        "base_rate": float(np.mean(t)),
    }


def threshold_for_target_rate(p_val: np.ndarray, target_rate: float) -> float:
    thr_grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 999), np.quantile(p_val, np.linspace(0, 1, 200))]))
    rates = np.array([np.mean(p_val >= t) for t in thr_grid], dtype=float)
    i = int(np.argmin(np.abs(rates - float(target_rate))))
    return float(thr_grid[i])


def pr_curve_df(y_true: np.ndarray, p_score: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), p_score.astype(float))
    thr = np.full_like(precision, np.nan, dtype=np.float64)
    if thresholds.size > 0:
        thr[1:] = thresholds
    return pd.DataFrame({"threshold": thr, "precision": precision, "recall": recall})


def reliability_df(y_true: np.ndarray, p_score: np.ndarray, n_bins: int = 12) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true.astype(int), p_score.astype(float), n_bins=n_bins, strategy="quantile")
    # Add counts per bin for context.
    q = np.quantile(p_score, np.linspace(0, 1, n_bins + 1))
    q[0] = min(q[0], np.min(p_score) - 1e-12)
    bins = np.digitize(p_score, q[1:-1], right=True)
    counts = np.array([(bins == i).sum() for i in range(n_bins)], dtype=int)
    n = min(len(frac_pos), len(counts))
    return pd.DataFrame(
        {
            "prob_bin_center": mean_pred[:n],
            "frac_positive": frac_pos[:n],
            "count": counts[:n],
        }
    )


def model_scores(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if model_name == "logreg":
        clf = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        solver="saga",
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        max_iter=400,
                        n_jobs=-1,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
        clf.fit(x_train, y_train)
        return clf.predict_proba(x_val)[:, 1], clf.predict_proba(x_test)[:, 1]

    if model_name == "hgb":
        # Balanced sample weights for fair class treatment.
        yb = y_train.astype(int)
        n_pos = max(1, int(np.sum(yb == 1)))
        n_neg = max(1, int(np.sum(yb == 0)))
        w_pos = n_neg / n_pos
        sw = np.where(yb == 1, w_pos, 1.0).astype(np.float32)
        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=6,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            max_iter=1200,
            random_state=int(seed),
        )
        clf.fit(x_train, y_train, sample_weight=sw)
        return clf.predict_proba(x_val)[:, 1], clf.predict_proba(x_test)[:, 1]

    raise ValueError(f"Unsupported model_name={model_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="base_v16_clean_final_manualv8")
    ap.add_argument("--task", choices=["multipeak", "nonpsf"], default="multipeak")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    base = Path("/data/yn316/Codes")
    run_root = base / "output" / "ml_runs" / str(args.run_name)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base / "report" / "model_decision" / f"20260303_benchmark_{args.task}_{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "multipeak":
        xgb_dir = base / "report" / "model_decision" / f"20260302_gaia_first_multipeak_xgb_{args.run_name}"
        labels_csv = xgb_dir / "labels_multipeak.csv"
        y_col = "true_multipeak_flag"
    else:
        xgb_dir = base / "report" / "model_decision" / f"20260302_gaia_first_nonpsf_xgb_{args.run_name}"
        labels_csv = xgb_dir / "labels_nonpsf.csv"
        y_col = "true_nonpsf_flag"

    manifest = load_manifest(run_root)
    x_tr, _, _ = open_split_arrays(manifest, "train")
    x_va, _, _ = open_split_arrays(manifest, "val")
    x_te, _, _ = open_split_arrays(manifest, "test")
    labels = pd.read_csv(labels_csv, usecols=["split", y_col], low_memory=False)
    y_train = labels.loc[labels["split"] == "train", y_col].to_numpy(dtype=np.int32)
    y_val = labels.loc[labels["split"] == "val", y_col].to_numpy(dtype=np.int32)
    y_test = labels.loc[labels["split"] == "test", y_col].to_numpy(dtype=np.int32)
    x_train = np.asarray(x_tr[: len(y_train)], dtype=np.float32)
    x_val = np.asarray(x_va[: len(y_val)], dtype=np.float32)
    x_test = np.asarray(x_te[: len(y_test)], dtype=np.float32)

    # Frozen XGB baseline predictions from existing run.
    p_val_xgb = pd.read_csv(xgb_dir / "val_predictions.csv", usecols=["p_xgb"], low_memory=False)["p_xgb"].to_numpy(dtype=np.float32)
    p_test_xgb = pd.read_csv(xgb_dir / "test_predictions.csv", usecols=["p_xgb"], low_memory=False)["p_xgb"].to_numpy(dtype=np.float32)
    if len(p_val_xgb) != len(y_val) or len(p_test_xgb) != len(y_test):
        raise RuntimeError("XGB prediction lengths do not match labels for val/test.")

    scores = {
        "xgb_frozen": (p_val_xgb, p_test_xgb),
    }
    for model_name in ["logreg", "hgb"]:
        scores[model_name] = model_scores(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            x_test=x_test,
            seed=int(args.seed),
        )

    target_rate = float(np.mean(y_val))
    summary_rows = []
    op_rows = []

    plt.figure(figsize=(6.8, 5.2))
    for mname, (p_val, p_test) in scores.items():
        thr = threshold_for_target_rate(p_val, target_rate=target_rate)
        mt = bin_metrics(y_test, (p_test >= thr))
        pr = pr_curve_df(y_test, p_test)
        pr.to_csv(out_dir / f"pr_curve_{args.task}_{mname}.csv", index=False)
        rel = reliability_df(y_test, p_test, n_bins=12)
        rel.to_csv(out_dir / f"reliability_{args.task}_{mname}.csv", index=False)

        summary_rows.append(
            {
                "task": args.task,
                "model": mname,
                "base_rate_test": float(np.mean(y_test)),
                "auprc_test": float(average_precision_score(y_test, p_test)),
                "auroc_test": float(roc_auc_score(y_test, p_test)),
                "brier_test": float(brier_score_loss(y_test.astype(int), p_test.astype(float))),
                "op_rule": "rate_match_val",
                "threshold_val": float(thr),
                "pred_rate_test": float(mt["pred_rate"]),
                "precision_test": float(mt["precision"]),
                "recall_test": float(mt["recall"]),
                "f1_test": float(mt["f1"]),
                "tp_test": float(mt["tp"]),
                "fp_test": float(mt["fp"]),
                "fn_test": float(mt["fn"]),
            }
        )
        op_rows.append({"model": mname, "precision": float(mt["precision"]), "recall": float(mt["recall"])})
        plt.plot(pr["recall"], pr["precision"], lw=2.0, label=f"{mname} AUPRC={average_precision_score(y_test, p_test):.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Overlay ({args.task})")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"pr_overlay_{args.task}.png", dpi=170)
    plt.close()

    # Reliability overlay.
    plt.figure(figsize=(6.8, 5.2))
    for mname in scores.keys():
        rel = pd.read_csv(out_dir / f"reliability_{args.task}_{mname}.csv")
        plt.plot(rel["prob_bin_center"], rel["frac_positive"], marker="o", lw=1.8, label=mname)
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, alpha=0.8)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.title(f"Reliability Overlay ({args.task})")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"reliability_overlay_{args.task}.png", dpi=170)
    plt.close()

    # Op bars.
    op_df = pd.DataFrame(op_rows)
    x = np.arange(len(op_df), dtype=float)
    w = 0.36
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(x - w / 2, op_df["precision"], width=w, label="precision")
    plt.bar(x + w / 2, op_df["recall"], width=w, label="recall")
    plt.xticks(x, op_df["model"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"Operating Point @ Matched Candidate Rate ({args.task})")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"op_bars_{args.task}.png", dpi=170)
    plt.close()

    summary = pd.DataFrame(summary_rows).sort_values("auprc_test", ascending=False).reset_index(drop=True)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "task": args.task,
                "target_rate_val": target_rate,
                "models": list(scores.keys()),
                "note": "LightGBM/CatBoost unavailable in env; used HistGradientBoosting as nonlinear tree baseline.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()

