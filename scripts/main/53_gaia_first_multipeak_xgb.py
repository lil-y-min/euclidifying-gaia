#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.ndimage import gaussian_filter, maximum_filter
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


def robust_multipeak_labels_chunk(
    I_true: np.ndarray,
    sigma_smooth: float,
    min_peak_sep: float,
    k_mad: float,
    f_flux: float,
    flux_tiny: float = 1e-6,
    mad_eps: float = 1e-9,
) -> Dict[str, np.ndarray]:
    img = np.nan_to_num(I_true.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    bg = np.percentile(img, 10.0, axis=(1, 2))
    ip = img - bg[:, None, None]
    ip[ip < 0] = 0.0

    total_flux = np.sum(ip, axis=(1, 2))
    maxv = np.max(ip, axis=(1, 2))
    flat = ip.reshape(ip.shape[0], -1)
    top2 = np.partition(flat, -2, axis=1)[:, -2:]
    top2_sum = np.sum(top2, axis=1)

    sm = gaussian_filter(ip, sigma=(0.0, sigma_smooth, sigma_smooth), mode="nearest") if sigma_smooth > 0 else ip
    med = np.median(sm, axis=(1, 2))
    mad = np.median(np.abs(sm - med[:, None, None]), axis=(1, 2))
    mad = np.maximum(mad, mad_eps)
    thr = med + k_mad * mad

    local_max = sm >= maximum_filter(sm, size=(1, 3, 3), mode="nearest")
    frac = sm / (total_flux[:, None, None] + 1e-12)
    cand_mask = local_max & (sm > thr[:, None, None]) & (frac > f_flux)

    n = ip.shape[0]
    out_flag = np.zeros(n, dtype=np.int8)
    out_sep = np.full(n, np.nan, dtype=np.float64)
    out_npeaks = np.zeros(n, dtype=np.int16)

    for i in range(n):
        if (not np.isfinite(total_flux[i])) or (total_flux[i] <= flux_tiny):
            continue
        coords = np.argwhere(cand_mask[i])
        if coords.size == 0:
            continue
        vals = sm[i][cand_mask[i]]
        order = np.argsort(vals)[::-1]
        keep: List[Tuple[int, int, float]] = []
        for idx in order:
            y, x = int(coords[idx, 0]), int(coords[idx, 1])
            v = float(vals[idx])
            ok = True
            for yy, xx, _ in keep:
                if np.hypot(x - xx, y - yy) < min_peak_sep:
                    ok = False
                    break
            if ok:
                keep.append((y, x, v))
        out_npeaks[i] = int(len(keep))
        if len(keep) >= 2:
            (y1, x1, _), (y2, x2, _) = keep[0], keep[1]
            out_flag[i] = 1
            out_sep[i] = float(np.hypot(x2 - x1, y2 - y1))

    return {
        "true_multipeak_flag": out_flag.astype(np.int8),
        "true_peak_sep_pix": out_sep.astype(np.float32),
        "true_n_peaks": out_npeaks.astype(np.int16),
        "true_top2_flux_frac": (top2_sum / (total_flux + 1e-12)).astype(np.float32),
        "true_max_over_sum": (maxv / (total_flux + 1e-12)).astype(np.float32),
    }


def build_or_load_labels(
    manifest: dict,
    run_root: Path,
    out_dir: Path,
    sigma_smooth: float,
    min_peak_sep: float,
    k_mad: float,
    f_flux: float,
    chunk_size: int,
    max_train: int,
    max_val: int,
    max_test: int,
) -> pd.DataFrame:
    cache_path = out_dir / "labels_multipeak.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    stamp_pix = int(manifest["stamp_pix"])
    rows: List[pd.DataFrame] = []

    trace_test = pd.read_csv(run_root / "trace" / "trace_test.csv", usecols=["test_index", "source_id"], low_memory=False)
    trace_test = trace_test.sort_values("test_index").reset_index(drop=True)
    source_test = pd.to_numeric(trace_test["source_id"], errors="coerce")

    split_max = {"train": int(max_train), "val": int(max_val), "test": int(max_test)}
    for split in ["train", "val", "test"]:
        _, ys, yf = open_split_arrays(manifest, split=split)
        n0 = int(manifest[f"n_{split}"])
        n = n0 if split_max[split] <= 0 else min(n0, split_max[split])
        split_chunks: List[pd.DataFrame] = []
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            yshape = np.asarray(ys[s:e], dtype=np.float32)
            yflux = np.asarray(yf[s:e], dtype=np.float32)
            I = (yshape * (10.0 ** yflux[:, None])).reshape((-1, stamp_pix, stamp_pix))
            lab = robust_multipeak_labels_chunk(
                I_true=I,
                sigma_smooth=sigma_smooth,
                min_peak_sep=min_peak_sep,
                k_mad=k_mad,
                f_flux=f_flux,
            )
            dfc = pd.DataFrame(
                {
                    "split": split,
                    "row_index": np.arange(s, e, dtype=np.int64),
                    "source_id": np.nan,
                    "true_multipeak_flag": lab["true_multipeak_flag"],
                    "true_peak_sep_pix": lab["true_peak_sep_pix"],
                    "true_n_peaks": lab["true_n_peaks"],
                    "true_top2_flux_frac": lab["true_top2_flux_frac"],
                    "true_max_over_sum": lab["true_max_over_sum"],
                }
            )
            if split == "test":
                dfc["source_id"] = source_test.iloc[s:e].to_numpy(dtype=float)
            split_chunks.append(dfc)
        rows.append(pd.concat(split_chunks, ignore_index=True))

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(cache_path, index=False)
    return out


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


def operating_points_from_val(
    y_val: np.ndarray,
    p_val: np.ndarray,
    target_rate: float,
    precision_floor: float,
) -> pd.DataFrame:
    thr_grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 999), np.quantile(p_val, np.linspace(0, 1, 200))]))
    rows = []
    for thr in thr_grid:
        yh = p_val >= thr
        m = bin_metrics(y_val, yh)
        rows.append({"threshold": float(thr), **m})
    ops = pd.DataFrame(rows).sort_values("threshold", ascending=False).reset_index(drop=True)

    ops["is_target_rate"] = False
    ops["is_precision_floor"] = False
    ops["is_thr_0p5"] = np.isclose(ops["threshold"], 0.5, atol=5e-4)

    if len(ops) > 0:
        i_rate = int(np.argmin(np.abs(ops["pred_pos_rate"].to_numpy(dtype=float) - float(target_rate))))
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


def subset_metrics_rows(
    subset_name: str,
    y_true: np.ndarray,
    p_score: np.ndarray,
    heuristic_pred: np.ndarray | None,
    op_rows_test: pd.DataFrame,
) -> List[Dict[str, float | str]]:
    out: List[Dict[str, float | str]] = []
    if heuristic_pred is not None:
        m = bin_metrics(y_true, heuristic_pred.astype(bool))
        out.append({"subset": subset_name, "method": "heuristic_ruwe_or_ipd", "threshold": np.nan, **m})
    for _, r in op_rows_test.iterrows():
        yhat = p_score >= float(r["threshold"])
        m = bin_metrics(y_true, yhat.astype(bool))
        tag = str(r["op_tag"])
        out.append({"subset": subset_name, "method": f"xgb_{tag}", "threshold": float(r["threshold"]), **m})
    yhat05 = p_score >= 0.5
    m05 = bin_metrics(y_true, yhat05.astype(bool))
    out.append({"subset": subset_name, "method": "xgb_thr0p5", "threshold": 0.5, **m05})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True, help="Data run to evaluate, e.g., base_v16_clean_final_manualv8.")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--quality_flags_csv", default="", help="Optional file with source_id,quality_flag for manual_removed subsets.")
    ap.add_argument("--quality_bits_mask", type=int, default=2087)
    ap.add_argument("--sigma_smooth", type=float, default=0.8)
    ap.add_argument("--min_peak_sep", type=float, default=2.0)
    ap.add_argument("--k_mad", type=float, default=3.0)
    ap.add_argument("--f_flux", type=float, default=0.005)
    ap.add_argument("--chunk_size", type=int, default=4096)
    ap.add_argument("--max_train", type=int, default=-1)
    ap.add_argument("--max_val", type=int, default=-1)
    ap.add_argument("--max_test", type=int, default=-1)
    ap.add_argument("--target_rate", type=float, default=-1.0, help="If <=0, uses validation base rate.")
    ap.add_argument("--precision_floor", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage2_min_recall", type=float, default=0.1)
    args = ap.parse_args()

    base = Path("/data/yn316/Codes")
    run_root = base / "output" / "ml_runs" / str(args.run_name)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base / "report" / "model_decision" / f"20260302_gaia_first_multipeak_xgb_{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(run_root)
    feature_cols = list(manifest["feature_cols"])

    labels = build_or_load_labels(
        manifest=manifest,
        run_root=run_root,
        out_dir=out_dir,
        sigma_smooth=float(args.sigma_smooth),
        min_peak_sep=float(args.min_peak_sep),
        k_mad=float(args.k_mad),
        f_flux=float(args.f_flux),
        chunk_size=int(args.chunk_size),
        max_train=int(args.max_train),
        max_val=int(args.max_val),
        max_test=int(args.max_test),
    )

    X_tr, _, _ = open_split_arrays(manifest, "train")
    X_va, _, _ = open_split_arrays(manifest, "val")
    X_te, _, _ = open_split_arrays(manifest, "test")

    y_train = labels.loc[labels["split"] == "train", "true_multipeak_flag"].to_numpy(dtype=np.int32)
    y_val = labels.loc[labels["split"] == "val", "true_multipeak_flag"].to_numpy(dtype=np.int32)
    y_test = labels.loc[labels["split"] == "test", "true_multipeak_flag"].to_numpy(dtype=np.int32)
    sep_test = labels.loc[labels["split"] == "test", "true_peak_sep_pix"].to_numpy(dtype=np.float32)
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

    target_rate = float(args.target_rate) if float(args.target_rate) > 0 else float(np.mean(y_val))
    ops_val = operating_points_from_val(
        y_val=y_val.astype(bool),
        p_val=p_val,
        target_rate=target_rate,
        precision_floor=float(args.precision_floor),
    )
    ops_rows = []
    if np.any(ops_val["is_target_rate"]):
        r = ops_val[ops_val["is_target_rate"]].iloc[0].copy()
        r["op_tag"] = "target_rate"
        ops_rows.append(r)
    if np.any(ops_val["is_precision_floor"]):
        r = ops_val[ops_val["is_precision_floor"]].iloc[0].copy()
        r["op_tag"] = "precision_floor"
        ops_rows.append(r)
    if len(ops_rows) == 0:
        r = ops_val.iloc[(np.abs(ops_val["threshold"] - 0.5)).argmin()].copy()
        r["op_tag"] = "fallback_0p5"
        ops_rows.append(r)
    op_rows_val = pd.DataFrame(ops_rows)
    op_rows_test = op_rows_val[["threshold", "op_tag"]].copy().reset_index(drop=True)
    op_rows_test["precision"] = np.nan
    op_rows_test["recall"] = np.nan
    op_rows_test["f1"] = np.nan
    op_rows_test["pred_pos_rate"] = np.nan
    op_rows_test["tp"] = np.nan
    op_rows_test["fp"] = np.nan
    op_rows_test["fn"] = np.nan
    for i in range(len(op_rows_test)):
        thr = float(op_rows_test.iloc[i]["threshold"])
        m = bin_metrics(y_test.astype(bool), (p_test >= thr))
        for c in ["precision", "recall", "f1", "pred_pos_rate", "tp", "fp", "fn"]:
            op_rows_test.iloc[i, op_rows_test.columns.get_loc(c)] = m[c]
    op_rows_test.to_csv(out_dir / "operating_points.csv", index=False)

    pr_df = make_pr_curve(y_true=y_test.astype(bool), p_score=p_test)
    pr_df.to_csv(out_dir / "pr_curve.csv", index=False)

    plt.figure(figsize=(6.8, 5.2))
    plt.plot(pr_df["recall"], pr_df["precision"], lw=2.0, label="XGB")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve: Gaia-first Multi-peak Classifier (test)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=160)
    plt.close()

    # Baseline heuristic (if feature columns exist).
    feat_to_idx = {c: i for i, c in enumerate(feature_cols)}
    heu = None
    if "feat_ruwe" in feat_to_idx and "feat_ipd_frac_multi_peak" in feat_to_idx:
        ruwe = X_test[:, feat_to_idx["feat_ruwe"]]
        ipd = X_test[:, feat_to_idx["feat_ipd_frac_multi_peak"]]
        heu = (ruwe > 1.4) | (ipd > 0.1)

    trace_test = pd.read_csv(run_root / "trace" / "trace_test.csv", usecols=["test_index", "source_id"], low_memory=False)
    trace_test = trace_test.sort_values("test_index").reset_index(drop=True)
    source_id_test = pd.to_numeric(trace_test["source_id"], errors="coerce").to_numpy(dtype=float)

    manual_removed_test = np.zeros_like(y_test, dtype=bool)
    rows = []
    rows.extend(
        subset_metrics_rows(
            subset_name="all",
            y_true=y_test.astype(bool),
            p_score=p_test.astype(float),
            heuristic_pred=heu,
            op_rows_test=op_rows_test[["threshold", "op_tag"]].copy(),
        )
    )

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
        t["manual_removed"] = t["manual_removed"].fillna(False).astype(bool)
        m_removed = t["manual_removed"].to_numpy(dtype=bool)
        manual_removed_test = m_removed.copy()
        m_kept = ~m_removed
        for subset_name, mask in [("manual_removed", m_removed), ("manual_kept", m_kept)]:
            if int(np.sum(mask)) == 0:
                continue
            heu_sub = heu[mask] if heu is not None else None
            rows.extend(
                subset_metrics_rows(
                    subset_name=subset_name,
                    y_true=y_test[mask].astype(bool),
                    p_score=p_test[mask].astype(float),
                    heuristic_pred=heu_sub,
                    op_rows_test=op_rows_test[["threshold", "op_tag"]].copy(),
                )
            )

    baseline_vs_xgb = pd.DataFrame(rows)
    baseline_vs_xgb.to_csv(out_dir / "baseline_vs_xgb.csv", index=False)

    pred_rows = pd.DataFrame(
        {
            "test_index": np.arange(len(y_test), dtype=np.int64),
            "source_id": source_id_test,
            "y_true": y_test.astype(np.int8),
            "p_xgb": p_test.astype(np.float32),
            "manual_removed": manual_removed_test.astype(np.int8),
        }
    )
    pred_rows.to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame(
        {
            "val_index": np.arange(len(y_val), dtype=np.int64),
            "y_true": y_val.astype(np.int8),
            "p_xgb": p_val.astype(np.float32),
        }
    ).to_csv(out_dir / "val_predictions.csv", index=False)

    # Optional Stage 2: sep regression if stage-1 recall is meaningful.
    stage2 = {"ran": False}
    best_recall = float(np.nanmax(op_rows_test["recall"].to_numpy(dtype=float))) if len(op_rows_test) > 0 else 0.0
    if best_recall >= float(args.stage2_min_recall):
        y_sep_tr = labels.loc[(labels["split"] == "train") & (labels["true_multipeak_flag"] > 0), "true_peak_sep_pix"].to_numpy(dtype=np.float32)
        y_sep_va = labels.loc[(labels["split"] == "val") & (labels["true_multipeak_flag"] > 0), "true_peak_sep_pix"].to_numpy(dtype=np.float32)
        mtr = labels.loc[(labels["split"] == "train"), "true_multipeak_flag"].to_numpy(dtype=np.int8) > 0
        mva = labels.loc[(labels["split"] == "val"), "true_multipeak_flag"].to_numpy(dtype=np.int8) > 0
        mte = y_test > 0
        if np.sum(mtr) >= 100 and np.sum(mva) >= 50 and np.sum(mte) >= 50:
            dtr_r = xgb.DMatrix(X_train[mtr], label=y_sep_tr)
            dva_r = xgb.DMatrix(X_val[mva], label=y_sep_va)
            dte_r = xgb.DMatrix(X_test[mte], label=sep_test[mte])
            reg = xgb.train(
                {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "tree_method": "hist",
                    "eta": 0.05,
                    "max_depth": 6,
                    "min_child_weight": 5.0,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "seed": int(args.seed),
                },
                dtr_r,
                num_boost_round=2000,
                evals=[(dtr_r, "train"), (dva_r, "val")],
                early_stopping_rounds=100,
                verbose_eval=False,
            )
            pred_sep = reg.predict(dte_r, iteration_range=(0, reg.best_iteration + 1))
            true_sep = sep_test[mte].astype(float)
            rmse_sep = float(np.sqrt(np.mean((pred_sep - true_sep) ** 2)))
            spearman = float(pd.Series(pred_sep).rank().corr(pd.Series(true_sep).rank(), method="pearson"))
            stage2 = {
                "ran": True,
                "n_train_pos": int(np.sum(mtr)),
                "n_val_pos": int(np.sum(mva)),
                "n_test_pos": int(np.sum(mte)),
                "rmse": rmse_sep,
                "spearman": spearman,
            }

    if stage2.get("ran", False):
        (out_dir / "stage2_peaksep_regression.json").write_text(json.dumps(stage2, indent=2), encoding="utf-8")

    # Feature importance plot (optional).
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
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "feature_cols": feature_cols,
        "detector_params": {
            "sigma_smooth": float(args.sigma_smooth),
            "min_peak_sep": float(args.min_peak_sep),
            "k_mad": float(args.k_mad),
            "f_flux": float(args.f_flux),
        },
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
        "operating_point_target_rate": float(target_rate),
        "precision_floor": float(args.precision_floor),
        "heuristic_available": bool(heu is not None),
        "stage2": stage2,
        "notes": [
            "Train/val/test arrays come from precomputed run splits (source_id-safe upstream).",
            "source_id is only directly available for test via trace_test.csv in this run artifact.",
        ],
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
