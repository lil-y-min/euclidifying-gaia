#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


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
    D = int(manifest["D"])
    x = np.memmap(Path(str(manifest[f"X_{split}_path"])), dtype="float32", mode="r", shape=(n, n_feat))
    ys = np.memmap(Path(str(manifest[f"Yshape_{split}_path"])), dtype="float32", mode="r", shape=(n, D))
    yf = np.memmap(Path(str(manifest[f"Yflux_{split}_path"])), dtype="float32", mode="r", shape=(n,))
    return x, ys, yf


def extract_morphology_chunk(stamps: np.ndarray) -> Dict[str, np.ndarray]:
    img = np.nan_to_num(stamps.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    bg = np.percentile(img, 10.0, axis=(1, 2))
    ip = img - bg[:, None, None]
    ip[ip < 0] = 0.0
    n, h, w = ip.shape
    yy, xx = np.indices((h, w))

    flux = np.sum(ip, axis=(1, 2))
    flux_safe = flux + 1e-9
    cx = np.sum(ip * xx[None, :, :], axis=(1, 2)) / flux_safe
    cy = np.sum(ip * yy[None, :, :], axis=(1, 2)) / flux_safe

    dx = xx[None, :, :] - cx[:, None, None]
    dy = yy[None, :, :] - cy[:, None, None]
    qxx = np.sum(ip * dx * dx, axis=(1, 2)) / flux_safe
    qyy = np.sum(ip * dy * dy, axis=(1, 2)) / flux_safe
    qxy = np.sum(ip * dx * dy, axis=(1, 2)) / flux_safe
    tr = qxx + qyy
    det = qxx * qyy - qxy * qxy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    lam1 = np.maximum((tr + np.sqrt(disc)) * 0.5, 0.0)
    lam2 = np.maximum((tr - np.sqrt(disc)) * 0.5, 0.0)
    a = np.sqrt(lam1)
    b = np.sqrt(lam2)
    ell = (a - b) / (a + b + 1e-9)
    ell[~np.isfinite(ell)] = 0.0
    ell[flux <= 0] = 0.0

    rr = np.sqrt((xx[None, :, :] - cx[:, None, None]) ** 2 + (yy[None, :, :] - cy[:, None, None]) ** 2)
    f_r2 = np.sum(ip * (rr <= 2.0), axis=(1, 2))
    f_r6 = np.sum(ip * (rr <= 6.0), axis=(1, 2))
    concentration = f_r2 / (f_r6 + 1e-9)

    border = np.zeros((h, w), dtype=bool)
    border[:2, :] = True
    border[-2:, :] = True
    border[:, :2] = True
    border[:, -2:] = True
    edge_flux_frac = np.sum(ip[:, border], axis=1) / (flux + 1e-9)

    rot180 = np.flip(ip, axis=(1, 2))
    asym = np.sum(np.abs(ip - rot180), axis=(1, 2)) / (np.sum(np.abs(ip), axis=(1, 2)) + 1e-9)

    return {
        "m_concentration_r2_r6": concentration.astype(np.float32),
        "m_asymmetry_180": asym.astype(np.float32),
        "m_ellipticity": ell.astype(np.float32),
        "m_edge_flux_frac": edge_flux_frac.astype(np.float32),
    }


def build_or_load_labels_nonpsf(
    manifest: dict,
    run_root: Path,
    out_dir: Path,
    chunk_size: int,
    max_train: int,
    max_val: int,
    max_test: int,
) -> pd.DataFrame:
    cache = out_dir / "labels_nonpsf.csv"
    thr_json = out_dir / "labels_nonpsf_thresholds.json"
    if cache.exists() and thr_json.exists():
        return pd.read_csv(cache)

    split_max = {"train": int(max_train), "val": int(max_val), "test": int(max_test)}
    stamp_pix = int(manifest["stamp_pix"])

    trace_test = pd.read_csv(run_root / "trace" / "trace_test.csv", usecols=["test_index", "source_id"], low_memory=False)
    trace_test = trace_test.sort_values("test_index").reset_index(drop=True)
    source_test = pd.to_numeric(trace_test["source_id"], errors="coerce")

    rows: List[pd.DataFrame] = []
    for split in ["train", "val", "test"]:
        _, ys, yf = open_split_arrays(manifest, split=split)
        n0 = int(manifest[f"n_{split}"])
        n = n0 if split_max[split] <= 0 else min(n0, split_max[split])
        recs: List[pd.DataFrame] = []
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            yshape = np.asarray(ys[s:e], dtype=np.float32)
            yflux = np.asarray(yf[s:e], dtype=np.float32)
            I = (yshape * (10.0 ** yflux[:, None])).reshape((-1, stamp_pix, stamp_pix))
            m = extract_morphology_chunk(I)
            d = pd.DataFrame(
                {
                    "split": split,
                    "row_index": np.arange(s, e, dtype=np.int64),
                    "source_id": np.nan,
                    "m_concentration_r2_r6": m["m_concentration_r2_r6"],
                    "m_asymmetry_180": m["m_asymmetry_180"],
                    "m_ellipticity": m["m_ellipticity"],
                    "m_edge_flux_frac": m["m_edge_flux_frac"],
                }
            )
            if split == "test":
                d["source_id"] = source_test.iloc[s:e].to_numpy(dtype=float)
            recs.append(d)
        rows.append(pd.concat(recs, ignore_index=True))
    d = pd.concat(rows, ignore_index=True)

    tr = d[d["split"] == "train"].copy()
    thr = {
        "q90_edge_flux_frac": float(np.nanquantile(pd.to_numeric(tr["m_edge_flux_frac"], errors="coerce"), 0.90)),
        "q90_asymmetry_180": float(np.nanquantile(pd.to_numeric(tr["m_asymmetry_180"], errors="coerce"), 0.90)),
        "q90_ellipticity": float(np.nanquantile(pd.to_numeric(tr["m_ellipticity"], errors="coerce"), 0.90)),
        "q10_concentration_r2_r6": float(np.nanquantile(pd.to_numeric(tr["m_concentration_r2_r6"], errors="coerce"), 0.10)),
    }

    d["true_nonpsf_flag"] = (
        (pd.to_numeric(d["m_edge_flux_frac"], errors="coerce") >= thr["q90_edge_flux_frac"])
        | (pd.to_numeric(d["m_asymmetry_180"], errors="coerce") >= thr["q90_asymmetry_180"])
        | (pd.to_numeric(d["m_ellipticity"], errors="coerce") >= thr["q90_ellipticity"])
        | (pd.to_numeric(d["m_concentration_r2_r6"], errors="coerce") <= thr["q10_concentration_r2_r6"])
    ).astype(np.int8)

    d.to_csv(cache, index=False)
    thr_json.write_text(json.dumps(thr, indent=2), encoding="utf-8")
    return d


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
        "pred_pos_rate": float(np.mean(p)),
        "base_rate": float(np.mean(t)),
    }


def operating_points_from_val(y_val: np.ndarray, p_val: np.ndarray, target_rate: float, precision_floor: float) -> pd.DataFrame:
    thr_grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 999), np.quantile(p_val, np.linspace(0, 1, 200))]))
    rows = []
    for thr in thr_grid:
        m = bin_metrics(y_val, p_val >= thr)
        rows.append({"threshold": float(thr), **m})
    ops = pd.DataFrame(rows).sort_values("threshold", ascending=False).reset_index(drop=True)
    ops["is_target_rate"] = False
    ops["is_precision_floor"] = False
    if len(ops) > 0:
        i_rate = int(np.argmin(np.abs(ops["pred_pos_rate"].to_numpy(dtype=float) - target_rate)))
        ops.loc[i_rate, "is_target_rate"] = True
        cand = ops[ops["precision"] >= float(precision_floor)].copy()
        if not cand.empty:
            j = cand.sort_values(["recall", "threshold"], ascending=[False, False]).index[0]
            ops.loc[j, "is_precision_floor"] = True
    return ops


def make_pr_curve(y_true: np.ndarray, p_score: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), p_score.astype(float))
    thr = np.full_like(precision, np.nan, dtype=np.float64)
    if thresholds.size > 0:
        thr[1:] = thresholds
    return pd.DataFrame({"threshold": thr, "precision": precision, "recall": recall})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--quality_flags_csv", default="")
    ap.add_argument("--quality_bits_mask", type=int, default=2087)
    ap.add_argument("--chunk_size", type=int, default=4096)
    ap.add_argument("--max_train", type=int, default=-1)
    ap.add_argument("--max_val", type=int, default=-1)
    ap.add_argument("--max_test", type=int, default=-1)
    ap.add_argument("--target_rate", type=float, default=-1.0)
    ap.add_argument("--precision_floor", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path("/data/yn316/Codes")
    run_root = base / "output" / "ml_runs" / str(args.run_name)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base / "report" / "model_decision" / f"20260302_gaia_first_nonpsf_xgb_{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(run_root)
    feature_cols = list(manifest["feature_cols"])
    labels = build_or_load_labels_nonpsf(
        manifest=manifest,
        run_root=run_root,
        out_dir=out_dir,
        chunk_size=int(args.chunk_size),
        max_train=int(args.max_train),
        max_val=int(args.max_val),
        max_test=int(args.max_test),
    )

    X_tr, _, _ = open_split_arrays(manifest, "train")
    X_va, _, _ = open_split_arrays(manifest, "val")
    X_te, _, _ = open_split_arrays(manifest, "test")
    y_train = labels.loc[labels["split"] == "train", "true_nonpsf_flag"].to_numpy(dtype=np.int32)
    y_val = labels.loc[labels["split"] == "val", "true_nonpsf_flag"].to_numpy(dtype=np.int32)
    y_test = labels.loc[labels["split"] == "test", "true_nonpsf_flag"].to_numpy(dtype=np.int32)
    X_train = np.asarray(X_tr[: len(y_train)], dtype=np.float32)
    X_val = np.asarray(X_va[: len(y_val)], dtype=np.float32)
    X_test = np.asarray(X_te[: len(y_test)], dtype=np.float32)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = float(n_neg / max(1, n_pos))

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["aucpr", "auc"],
        "tree_method": "hist",
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 5.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.0,
        "scale_pos_weight": scale_pos_weight,
        "seed": int(args.seed),
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    booster.save_model(str(out_dir / "model_xgb.json"))
    (out_dir / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    p_val = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    p_test = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    pd.DataFrame({"val_index": np.arange(len(y_val), dtype=np.int64), "y_true": y_val.astype(np.int8), "p_xgb": p_val.astype(np.float32)}).to_csv(out_dir / "val_predictions.csv", index=False)

    trace_test = pd.read_csv(run_root / "trace" / "trace_test.csv", usecols=["test_index", "source_id"], low_memory=False).sort_values("test_index")
    source_id_test = pd.to_numeric(trace_test["source_id"], errors="coerce").to_numpy(dtype=float)
    manual_removed_test = np.zeros_like(y_test, dtype=np.int8)
    if str(args.quality_flags_csv).strip():
        qf = pd.read_csv(args.quality_flags_csv, usecols=["source_id", "quality_flag"], low_memory=False)
        qf["source_id"] = pd.to_numeric(qf["source_id"], errors="coerce")
        qf["quality_flag"] = pd.to_numeric(qf["quality_flag"], errors="coerce").fillna(0).astype(np.int64)
        qf = qf.dropna(subset=["source_id"]).copy()
        qf["source_id"] = qf["source_id"].astype(np.int64)
        qf["manual_removed"] = (qf["quality_flag"] & int(args.quality_bits_mask)) != 0
        t = trace_test.copy()
        t["source_id"] = pd.to_numeric(t["source_id"], errors="coerce")
        t = t.merge(qf[["source_id", "manual_removed"]], on="source_id", how="left")
        manual_removed_test = t["manual_removed"].fillna(False).astype(np.int8).to_numpy(dtype=np.int8)
    pd.DataFrame(
        {
            "test_index": np.arange(len(y_test), dtype=np.int64),
            "source_id": source_id_test,
            "y_true": y_test.astype(np.int8),
            "p_xgb": p_test.astype(np.float32),
            "manual_removed": manual_removed_test,
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)

    target_rate = float(args.target_rate) if float(args.target_rate) > 0 else float(np.mean(y_val))
    ops_val = operating_points_from_val(y_val.astype(bool), p_val, target_rate=target_rate, precision_floor=float(args.precision_floor))
    rows = []
    if np.any(ops_val["is_target_rate"]):
        r = ops_val[ops_val["is_target_rate"]].iloc[0].copy()
        r["op_tag"] = "target_rate"
        rows.append(r)
    if np.any(ops_val["is_precision_floor"]):
        r = ops_val[ops_val["is_precision_floor"]].iloc[0].copy()
        r["op_tag"] = "precision_floor"
        rows.append(r)
    op_rows = pd.DataFrame(rows)[["threshold", "op_tag"]].reset_index(drop=True)
    out_ops = []
    for i in range(len(op_rows)):
        thr = float(op_rows.iloc[i]["threshold"])
        m = bin_metrics(y_test.astype(bool), p_test >= thr)
        out_ops.append({"threshold": thr, "op_tag": str(op_rows.iloc[i]["op_tag"]), **m})
    pd.DataFrame(out_ops).to_csv(out_dir / "operating_points.csv", index=False)

    pr_df = make_pr_curve(y_test.astype(bool), p_test)
    pr_df.to_csv(out_dir / "pr_curve.csv", index=False)
    plt.figure(figsize=(6.8, 5.2))
    plt.plot(pr_df["recall"], pr_df["precision"], lw=2.0, label="XGB")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve: Gaia-first non-PSF classifier (test)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=160)
    plt.close()

    feat_to_idx = {c: i for i, c in enumerate(feature_cols)}
    heu = None
    if "feat_ruwe" in feat_to_idx and "feat_ipd_frac_multi_peak" in feat_to_idx:
        ruwe = X_test[:, feat_to_idx["feat_ruwe"]]
        ipd = X_test[:, feat_to_idx["feat_ipd_frac_multi_peak"]]
        heu = (ruwe > 1.4) | (ipd > 0.1)

    eval_rows: List[dict] = []
    if heu is not None:
        eval_rows.append({"subset": "all", "method": "heuristic_ruwe_or_ipd", "threshold": np.nan, **bin_metrics(y_test.astype(bool), heu.astype(bool))})
    for r in out_ops:
        m = bin_metrics(y_test.astype(bool), p_test >= float(r["threshold"]))
        eval_rows.append({"subset": "all", "method": f"xgb_{r['op_tag']}", "threshold": float(r["threshold"]), **m})
    eval_rows.append({"subset": "all", "method": "xgb_thr0p5", "threshold": 0.5, **bin_metrics(y_test.astype(bool), p_test >= 0.5)})

    if str(args.quality_flags_csv).strip():
        m_removed = manual_removed_test.astype(bool)
        for subset_name, mask in [("manual_removed", m_removed), ("manual_kept", ~m_removed)]:
            yt = y_test[mask].astype(bool)
            pt = p_test[mask]
            if heu is not None:
                eval_rows.append({"subset": subset_name, "method": "heuristic_ruwe_or_ipd", "threshold": np.nan, **bin_metrics(yt, heu[mask].astype(bool))})
            for r in out_ops:
                eval_rows.append({"subset": subset_name, "method": f"xgb_{r['op_tag']}", "threshold": float(r["threshold"]), **bin_metrics(yt, pt >= float(r["threshold"]))})
            eval_rows.append({"subset": subset_name, "method": "xgb_thr0p5", "threshold": 0.5, **bin_metrics(yt, pt >= 0.5)})
    pd.DataFrame(eval_rows).to_csv(out_dir / "baseline_vs_xgb.csv", index=False)

    imp = booster.get_score(importance_type="gain")
    if len(imp) > 0:
        fi = pd.DataFrame({"feature": list(imp.keys()), "gain": list(imp.values())}).sort_values("gain", ascending=False)
        fi.to_csv(out_dir / "feature_importance_gain.csv", index=False)
        top = fi.head(20).iloc[::-1]
        plt.figure(figsize=(8.2, 6.0))
        plt.barh(top["feature"], top["gain"])
        plt.title("XGB Feature Importance (gain, top 20)")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance_gain_top20.png", dpi=160)
        plt.close()

    metrics_summary = {
        "run_name": str(args.run_name),
        "label_definition": "Option A Euclid morphology: non-PSF if any of edge/asym/ellip high-tail or concentration low-tail (train quantiles).",
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "feature_cols": feature_cols,
        "class_balance": {
            "train_base_rate": float(np.mean(y_train)),
            "val_base_rate": float(np.mean(y_val)),
            "test_base_rate": float(np.mean(y_test)),
            "scale_pos_weight": scale_pos_weight,
        },
        "xgb": {
            "best_iteration": int(booster.best_iteration),
            "val_auprc": float(average_precision_score(y_val, p_val)),
            "val_auroc": float(roc_auc_score(y_val, p_val)),
            "test_auprc": float(average_precision_score(y_test, p_test)),
            "test_auroc": float(roc_auc_score(y_test, p_test)),
        },
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
