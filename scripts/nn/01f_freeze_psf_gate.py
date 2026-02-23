#!/usr/bin/env python3
"""
Freeze morph-only logistic PSF gate into a deployable package.

Outputs:
- gate_package.json (scalers + coefficients + threshold)
- gate_metrics.csv
- gate_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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


def robust_fit(x: np.ndarray, scale_floor: float = 0.0) -> Tuple[float, float, float]:
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    raw_scale = float(1.4826 * mad + 1e-9)
    scale = max(raw_scale, float(scale_floor))
    return med, raw_scale, scale


def robust_transform(x: np.ndarray, med: float, scale: float, clip_abs: float) -> np.ndarray:
    z = (np.asarray(x, dtype=float) - med) / scale
    return np.clip(z, -clip_abs, clip_abs)


def metric_row(split: str, y: np.ndarray, p: np.ndarray) -> Dict[str, float | str | int]:
    pp = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    yy = np.asarray(y, dtype=int)
    return {
        "split": split,
        "n": int(len(yy)),
        "auc": float(roc_auc_score(yy, pp)),
        "ap": float(average_precision_score(yy, pp)),
        "logloss": float(log_loss(yy, pp)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--z_clip", type=float, default=8.0)
    ap.add_argument("--C", type=float, default=1000.0)
    ap.add_argument("--max_iter", type=int, default=3000)
    ap.add_argument("--drop_features", default="", help="Comma-separated feature names to exclude.")
    ap.add_argument("--scale_floor", type=float, default=0.0, help="Minimum robust scale floor to avoid near-zero normalization.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    need = ["source_id", "split_code", "label"] + MORPH_COLS
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"labels csv missing required columns: {miss}")

    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    extra = [c for c in ["confidence", "score_psf_like"] if c in df.columns]
    for c in extra:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=need).copy()
    df["source_id"] = df["source_id"].astype(np.int64)
    df["split_code"] = df["split_code"].astype(int)
    df["label"] = df["label"].astype(int)
    df = df.sort_values(["source_id"]).drop_duplicates("source_id", keep="first")

    tr = df[df["split_code"] == 0].copy()
    va = df[df["split_code"] == 1].copy()
    te = df[df["split_code"] == 2].copy()
    if tr.empty or te.empty:
        raise RuntimeError("Need non-empty train(split=0) and test(split=2)")
    if va.empty:
        va = tr.copy()

    drop = {x.strip() for x in str(args.drop_features).split(",") if x.strip()}
    feat_cols = [c for c in MORPH_COLS if c not in drop]
    if not feat_cols:
        raise RuntimeError("No features left after applying --drop_features")

    scaler: Dict[str, Dict[str, float]] = {}
    for c in feat_cols:
        med, raw_scale, scale = robust_fit(tr[c].to_numpy(dtype=float), scale_floor=float(args.scale_floor))
        scaler[c] = {"median": med, "raw_scale": raw_scale, "scale": scale}

    def make_x(dsub: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in feat_cols:
            s = scaler[c]
            cols.append(robust_transform(dsub[c].to_numpy(dtype=float), s["median"], s["scale"], clip_abs=float(args.z_clip)))
        return np.column_stack(cols).astype(np.float32)

    Xtr = make_x(tr)
    Xva = make_x(va)
    Xte = make_x(te)
    ytr = tr["label"].to_numpy(dtype=int)
    yva = va["label"].to_numpy(dtype=int)
    yte = te["label"].to_numpy(dtype=int)

    clf = LogisticRegression(C=float(args.C), solver="lbfgs", max_iter=int(args.max_iter))
    clf.fit(Xtr, ytr)

    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]

    met = pd.DataFrame([
        metric_row("train", ytr, p_tr),
        metric_row("val", yva, p_va),
        metric_row("test", yte, p_te),
    ])
    met.to_csv(out_dir / "gate_metrics.csv", index=False)

    gate_package = {
        "model_type": "logistic_regression_binary",
        "target_positive_class": "non_psf_like_label1",
        "feature_order": feat_cols,
        "scaler": scaler,
        "z_clip": float(args.z_clip),
        "threshold_p_nonpsf": float(args.threshold),
        "intercept": float(clf.intercept_.reshape(-1)[0]),
        "coef": [float(x) for x in clf.coef_.reshape(-1).tolist()],
        "training": {
            "C": float(args.C),
            "max_iter": int(args.max_iter),
            "scale_floor": float(args.scale_floor),
        },
    }
    (out_dir / "gate_package.json").write_text(json.dumps(gate_package, indent=2), encoding="utf-8")

    summary = {
        "labels_csv": str(args.labels_csv),
        "drop_features": sorted(list(drop)),
        "feature_order": feat_cols,
        "scale_floor": float(args.scale_floor),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "threshold_p_nonpsf": float(args.threshold),
        "test_auc": float(met.loc[met["split"] == "test", "auc"].iloc[0]),
        "test_ap": float(met.loc[met["split"] == "test", "ap"].iloc[0]),
        "test_logloss": float(met.loc[met["split"] == "test", "logloss"].iloc[0]),
    }
    (out_dir / "gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
