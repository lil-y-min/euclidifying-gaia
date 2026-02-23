#!/usr/bin/env python3
"""
Compare logistic performance for:
1) morphology only
2) CVAE latent only
3) morphology + latent

Uses identical train/val/test splits from labels split_code.
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
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


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


def make_robust_mats(df_tr: pd.DataFrame, df_va: pd.DataFrame, df_te: pd.DataFrame, cols: List[str], clip_abs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, RobustScaler1D]]:
    scalers: Dict[str, RobustScaler1D] = {c: robust_fit(df_tr[c].to_numpy(dtype=float)) for c in cols}

    def trf(dsub: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            robust_transform(dsub[c].to_numpy(dtype=float), scalers[c], clip_abs=clip_abs) for c in cols
        ]).astype(np.float32)

    return trf(df_tr), trf(df_va), trf(df_te), scalers


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p)),
    }


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
    ap.add_argument("--latent_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--z_clip", type=float, default=8.0)
    ap.add_argument("--C", type=float, default=1000.0, help="Inverse regularization strength")
    ap.add_argument("--max_iter", type=int, default=3000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels_csv)
    need_l = ["source_id", "split_code", "label"] + MORPH_COLS
    miss_l = [c for c in need_l if c not in labels.columns]
    if miss_l:
        raise RuntimeError(f"labels csv missing columns: {miss_l}")
    for c in need_l:
        labels[c] = pd.to_numeric(labels[c], errors="coerce")
    labels = labels.dropna(subset=need_l).copy()
    labels["source_id"] = labels["source_id"].astype(np.int64)
    labels["split_code"] = labels["split_code"].astype(int)
    labels["label"] = labels["label"].astype(int)
    labels = labels.sort_values(["source_id"]).drop_duplicates("source_id", keep="first")

    lat = pd.read_csv(args.latent_csv)
    z_cols = [c for c in lat.columns if c.startswith("z_")]
    if not z_cols:
        raise RuntimeError("latent csv has no z_* columns")
    need_z = ["source_id", "split_code"] + z_cols
    miss_z = [c for c in need_z if c not in lat.columns]
    if miss_z:
        raise RuntimeError(f"latent csv missing columns: {miss_z}")
    for c in need_z:
        lat[c] = pd.to_numeric(lat[c], errors="coerce")
    lat = lat.dropna(subset=need_z).copy()
    lat["source_id"] = lat["source_id"].astype(np.int64)
    lat["split_code"] = lat["split_code"].astype(int)
    lat = lat.sort_values(["source_id"]).drop_duplicates("source_id", keep="first")

    df = labels.merge(lat[["source_id", "split_code"] + z_cols], on="source_id", how="inner", suffixes=("", "_lat"))
    if df.empty:
        raise RuntimeError("No overlap rows after merge labels/latents on source_id")

    # Guard split mismatch if any
    split_mismatch = (df["split_code"] != df["split_code_lat"]).sum() if "split_code_lat" in df.columns else 0
    if "split_code_lat" in df.columns:
        df = df[df["split_code"] == df["split_code_lat"]].copy()
        df = df.drop(columns=["split_code_lat"])

    tr = df[df["split_code"] == 0].copy()
    va = df[df["split_code"] == 1].copy()
    te = df[df["split_code"] == 2].copy()
    if tr.empty or te.empty:
        raise RuntimeError("Need non-empty train(split=0) and test(split=2)")
    if va.empty:
        va = tr.copy()

    ytr = tr["label"].to_numpy(dtype=int)
    yva = va["label"].to_numpy(dtype=int)
    yte = te["label"].to_numpy(dtype=int)

    feature_sets = {
        "morph_only": MORPH_COLS,
        "latent_only": z_cols,
        "morph_plus_latent": MORPH_COLS + z_cols,
    }

    rows = []
    roc_curves = {}
    pr_curves = {}
    coefs_combined = None

    for name, cols in feature_sets.items():
        Xtr, Xva, Xte, _ = make_robust_mats(tr, va, te, cols=cols, clip_abs=float(args.z_clip))
        clf = LogisticRegression(C=float(args.C), solver="lbfgs", max_iter=int(args.max_iter))
        clf.fit(Xtr, ytr)

        p_tr = clf.predict_proba(Xtr)[:, 1]
        p_va = clf.predict_proba(Xva)[:, 1]
        p_te = clf.predict_proba(Xte)[:, 1]

        for split, y, p in [("train", ytr, p_tr), ("val", yva, p_va), ("test", yte, p_te)]:
            m = metrics(y, p)
            rows.append({"model": name, "split": split, "n": int(len(y)), **m})

        # ROC points (manual for plots)
        def roc(y, p):
            o = np.argsort(-p)
            yy = y[o]
            P = max(1, int(np.sum(yy == 1)))
            N = max(1, int(np.sum(yy == 0)))
            tps = np.cumsum(yy == 1)
            fps = np.cumsum(yy == 0)
            tpr = np.concatenate([[0.0], tps / P])
            fpr = np.concatenate([[0.0], fps / N])
            return fpr, tpr

        def pr(y, p):
            o = np.argsort(-p)
            yy = y[o]
            tp = np.cumsum(yy == 1)
            fp = np.cumsum(yy == 0)
            P = max(1, int(np.sum(yy == 1)))
            prec = np.concatenate([[1.0], tp / np.maximum(tp + fp, 1)])
            rec = np.concatenate([[0.0], tp / P])
            return rec, prec

        roc_curves[name] = roc(yte, p_te)
        pr_curves[name] = pr(yte, p_te)

        if name == "morph_plus_latent":
            coefs_combined = pd.DataFrame(
                {
                    "feature": cols,
                    "coef": clf.coef_.reshape(-1),
                    "abs_coef": np.abs(clf.coef_.reshape(-1)),
                }
            ).sort_values("abs_coef", ascending=False)

    met = pd.DataFrame(rows).sort_values(["split", "auc"], ascending=[True, False])
    met.to_csv(out_dir / "morph_latent_combined_metrics.csv", index=False)

    if coefs_combined is not None:
        coefs_combined.to_csv(out_dir / "combined_model_coefficients.csv", index=False)

    # Plot: test metrics by model
    test_df = met[met["split"] == "test"].copy()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    for ax, metric in zip(axs, ["auc", "ap", "logloss"]):
        d = test_df.sort_values(metric, ascending=(metric == "logloss"))
        ax.bar(d["model"], d[metric])
        ax.set_title(f"test {metric}")
        ax.tick_params(axis="x", rotation=15)
    savefig(plot_dir / "test_metrics_three_models.png")

    # ROC
    plt.figure(figsize=(6.3, 5.2))
    for name in ["morph_only", "latent_only", "morph_plus_latent"]:
        fpr, tpr = roc_curves[name]
        auc = float(test_df.loc[test_df["model"] == name, "auc"].iloc[0])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC on test")
    plt.legend(loc="lower right")
    savefig(plot_dir / "roc_test_three_models.png")

    # PR
    plt.figure(figsize=(6.3, 5.2))
    for name in ["morph_only", "latent_only", "morph_plus_latent"]:
        rec, pre = pr_curves[name]
        apv = float(test_df.loc[test_df["model"] == name, "ap"].iloc[0])
        plt.plot(rec, pre, label=f"{name} (AP={apv:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR on test")
    plt.legend(loc="lower left")
    savefig(plot_dir / "pr_test_three_models.png")

    if coefs_combined is not None:
        top = coefs_combined.head(25).iloc[::-1]
        plt.figure(figsize=(8.4, 7.0))
        plt.barh(top["feature"], top["coef"])
        plt.xlabel("coefficient")
        plt.title("Top combined-model coefficients (|coef|)")
        savefig(plot_dir / "combined_top_coefficients.png")

    summary = {
        "labels_csv": str(args.labels_csv),
        "latent_csv": str(args.latent_csv),
        "n_merged": int(len(df)),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "split_mismatch_dropped": int(split_mismatch),
        "z_dim": int(len(z_cols)),
        "best_test_auc_model": str(test_df.sort_values("auc", ascending=False).iloc[0]["model"]),
        "best_test_logloss_model": str(test_df.sort_values("logloss", ascending=True).iloc[0]["model"]),
    }
    (out_dir / "morph_latent_combined_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
