#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:
    gaussian_filter = None


def load_manifest(run_root: Path) -> dict:
    d = np.load(run_root / "manifest_arrays.npz", allow_pickle=True)
    out = {}
    for k in d.files:
        v = d[k]
        if np.ndim(v) == 0:
            v = v.item()
        out[k] = v
    return out


def open_test_arrays(manifest: dict):
    n_test = int(manifest["n_test"])
    n_feat = int(manifest["n_features"])
    D = int(manifest["D"])
    stamp_pix = int(manifest["stamp_pix"])
    X_test = np.memmap(Path(str(manifest["X_test_path"])), dtype="float32", mode="r", shape=(n_test, n_feat))
    Yshape_test = np.memmap(Path(str(manifest["Yshape_test_path"])), dtype="float32", mode="r", shape=(n_test, D))
    Yflux_test = np.memmap(Path(str(manifest["Yflux_test_path"])), dtype="float32", mode="r", shape=(n_test,))
    return X_test, Yshape_test, Yflux_test, stamp_pix, D


def predict_shape(models_dir: Path, X_sub: np.ndarray, D: int) -> np.ndarray:
    dmat = xgb.DMatrix(X_sub)
    out = np.empty((X_sub.shape[0], D), dtype=np.float32)
    for j in range(D):
        b = xgb.Booster()
        b.load_model(str(models_dir / f"booster_pix_{j:04d}.json"))
        out[:, j] = b.predict(dmat).astype(np.float32, copy=False)
    return out


def _safe_percentile(a: np.ndarray, q: float) -> float:
    v = a[np.isfinite(a)]
    if v.size == 0:
        return 0.0
    return float(np.percentile(v, q))


def _prep_ip(stamp: np.ndarray) -> np.ndarray:
    img = np.array(stamp, dtype=np.float64)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    bg = _safe_percentile(img, 10.0)
    ip = img - bg
    ip[ip < 0] = 0.0
    return ip


def old_peak_sep(stamp: np.ndarray) -> float:
    ip = _prep_ip(stamp)
    flat = ip.ravel()
    if flat.size < 2:
        return np.nan
    idx2 = np.argpartition(flat, -2)[-2:]
    vals2 = flat[idx2]
    ord2 = np.argsort(vals2)[::-1]
    i1 = int(idx2[ord2[0]])
    i2 = int(idx2[ord2[1]])
    h, w = ip.shape
    y1, x1 = divmod(i1, w)
    y2, x2 = divmod(i2, w)
    return float(np.hypot(x2 - x1, y2 - y1))


def _local_maxima_candidates(img: np.ndarray) -> list[Tuple[int, int, float]]:
    h, w = img.shape
    out = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            v = img[y, x]
            if not np.isfinite(v):
                continue
            patch = img[y - 1 : y + 2, x - 1 : x + 2]
            if v >= np.max(patch):
                out.append((y, x, float(v)))
    return out


def robust_peaksep_and_flag(
    stamp: np.ndarray,
    sigma_smooth: float = 0.8,
    min_peak_sep: float = 2.0,
    k_mad: float = 3.0,
    f_flux: float = 0.02,
    flux_tiny: float = 1e-6,
    mad_eps: float = 1e-9,
) -> Dict[str, float]:
    ip = _prep_ip(stamp)
    total_flux = float(np.sum(ip))
    if not np.isfinite(total_flux) or total_flux <= flux_tiny:
        return {
            "peak_sep_pix": np.nan,
            "multipeak_flag": 0.0,
            "n_valid_peaks": 0.0,
            "peak_score": 0.0,
        }

    if gaussian_filter is not None and sigma_smooth > 0:
        sm = gaussian_filter(ip, sigma=sigma_smooth, mode="nearest")
    else:
        sm = ip

    med = float(np.median(sm))
    mad = float(np.median(np.abs(sm - med)))
    mad = max(mad, mad_eps)
    thr = med + k_mad * mad

    cands = _local_maxima_candidates(sm)
    # threshold and flux-fraction filtering
    filt = []
    for y, x, v in cands:
        if v <= thr:
            continue
        if (v / (total_flux + 1e-12)) <= f_flux:
            continue
        filt.append((y, x, v))
    if not filt:
        return {
            "peak_sep_pix": np.nan,
            "multipeak_flag": 0.0,
            "n_valid_peaks": 0.0,
            "peak_score": 0.0,
        }

    # non-maximum suppression by minimum separation
    filt = sorted(filt, key=lambda t: t[2], reverse=True)
    keep = []
    for y, x, v in filt:
        ok = True
        for yy, xx, _ in keep:
            if np.hypot(x - xx, y - yy) < min_peak_sep:
                ok = False
                break
        if ok:
            keep.append((y, x, v))

    n_peaks = len(keep)
    if n_peaks < 2:
        # score tracks how close we are to threshold (for AUROC-like ranking)
        s = float(keep[0][2] / (thr + 1e-12)) if n_peaks == 1 else 0.0
        return {
            "peak_sep_pix": np.nan,
            "multipeak_flag": 0.0,
            "n_valid_peaks": float(n_peaks),
            "peak_score": s,
        }

    (y1, x1, v1), (y2, x2, v2) = keep[0], keep[1]
    sep = float(np.hypot(x2 - x1, y2 - y1))
    score = float(v2 / (thr + 1e-12))
    return {
        "peak_sep_pix": sep,
        "multipeak_flag": 1.0,
        "n_valid_peaks": float(n_peaks),
        "peak_score": score,
    }




def robust_best_effort_sep(
    stamp: np.ndarray,
    sigma_smooth: float = 0.8,
    min_peak_sep: float = 2.0,
    flux_tiny: float = 1e-6,
) -> float:
    """Always attempt a 2-peak separation from local maxima for diagnostics.

    This ignores threshold/prominence gating so it can be evaluated on all rows.
    """
    ip = _prep_ip(stamp)
    total_flux = float(np.sum(ip))
    if not np.isfinite(total_flux) or total_flux <= flux_tiny:
        return np.nan

    if gaussian_filter is not None and sigma_smooth > 0:
        sm = gaussian_filter(ip, sigma=sigma_smooth, mode="nearest")
    else:
        sm = ip

    cands = _local_maxima_candidates(sm)
    if len(cands) < 2:
        flat = sm.ravel()
        if flat.size < 2:
            return np.nan
        idx2 = np.argpartition(flat, -2)[-2:]
        vals2 = flat[idx2]
        ord2 = np.argsort(vals2)[::-1]
        i1 = int(idx2[ord2[0]])
        i2 = int(idx2[ord2[1]])
        h, w = sm.shape
        y1, x1 = divmod(i1, w)
        y2, x2 = divmod(i2, w)
        return float(np.hypot(x2 - x1, y2 - y1))

    cands = sorted(cands, key=lambda t: t[2], reverse=True)
    keep = []
    for y, x, v in cands:
        ok = True
        for yy, xx, _ in keep:
            if np.hypot(x - xx, y - yy) < min_peak_sep:
                ok = False
                break
        if ok:
            keep.append((y, x, v))

    if len(keep) < 2:
        (y1, x1, _), (y2, x2, _) = cands[0], cands[1]
    else:
        (y1, x1, _), (y2, x2, _) = keep[0], keep[1]
    return float(np.hypot(x2 - x1, y2 - y1))

def corr_pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return np.nan
    x = a[m]
    y = b[m]
    sx = np.std(x)
    sy = np.std(y)
    if sx <= 0 or sy <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def corr_spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return np.nan
    x = pd.Series(a[m]).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(b[m]).rank(method="average").to_numpy(dtype=float)
    return corr_pearson(x, y)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def bin_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    t = (y_true > 0.5)
    p = (y_pred > 0.5)
    tp = float(np.sum(t & p))
    fp = float(np.sum((~t) & p))
    fn = float(np.sum(t & (~p)))
    tn = float(np.sum((~t) & (~p)))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "p_true_flag1": float(np.mean(t)),
        "p_pred_flag1": float(np.mean(p)),
    }


def approx_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_score)
    y = y_true[m].astype(int)
    s = y_score[m].astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    # Mann-Whitney U formulation
    r = pd.Series(s).rank(method="average").to_numpy(dtype=float)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    sum_r_pos = float(np.sum(r[y == 1]))
    u = sum_r_pos - n_pos * (n_pos + 1.0) / 2.0
    auc = u / (n_pos * n_neg + 1e-12)
    return float(auc)


def evaluate_subset(df: pd.DataFrame, subset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_peak = []
    rows_flag = []

    def _push_peak_rows(
        *,
        method: str,
        tf: np.ndarray,
        pf: np.ndarray,
        t_sep: np.ndarray,
        p_sep: np.ndarray,
        p_sep_best_effort: np.ndarray | None,
    ) -> None:
        tmask = tf > 0.5
        pmask = pf > 0.5
        tp_mask = tmask & pmask

        # A) Detection-conditional (TP-only)
        t = t_sep[tp_mask]
        p = p_sep[tp_mask]
        rows_peak.append(
            {
                "subset": subset_name,
                "method": method,
                "condition": "true_flag==1_and_pred_flag==1",
                "eval_mode": "tp_only",
                "n_true_flag1": int(np.sum(tmask)),
                "n_eval": int(np.sum(np.isfinite(t) & np.isfinite(p))),
                "pearson": corr_pearson(t, p),
                "spearman": corr_spearman(t, p),
                "rmse": rmse(t, p),
                "true_std": float(np.nanstd(t)),
                "pred_std": float(np.nanstd(p)),
                "collapse_ratio_predstd_over_truestd": float(np.nanstd(p) / (np.nanstd(t) + 1e-12)),
            }
        )

        # B) True-conditional with fallback (penalizes FN end-to-end)
        t = t_sep[tmask]
        p = p_sep[tmask]
        p_fb = np.where(np.isfinite(p), p, 1.0)
        rows_peak.append(
            {
                "subset": subset_name,
                "method": method,
                "condition": "true_flag==1",
                "eval_mode": "true_cond_fallback1",
                "n_true_flag1": int(np.sum(tmask)),
                "n_eval": int(np.sum(np.isfinite(t) & np.isfinite(p_fb))),
                "pearson": corr_pearson(t, p_fb),
                "spearman": corr_spearman(t, p_fb),
                "rmse": rmse(t, p_fb),
                "true_std": float(np.nanstd(t)),
                "pred_std": float(np.nanstd(p_fb)),
                "collapse_ratio_predstd_over_truestd": float(np.nanstd(p_fb) / (np.nanstd(t) + 1e-12)),
            }
        )

        # C) True-conditional best-effort (diagnostic upper bound)
        if p_sep_best_effort is not None:
            p_be = p_sep_best_effort[tmask]
            p_be = np.where(np.isfinite(p_be), p_be, 1.0)
            rows_peak.append(
                {
                    "subset": subset_name,
                    "method": method,
                    "condition": "true_flag==1",
                    "eval_mode": "true_cond_best_effort",
                    "n_true_flag1": int(np.sum(tmask)),
                    "n_eval": int(np.sum(np.isfinite(t) & np.isfinite(p_be))),
                    "pearson": corr_pearson(t, p_be),
                    "spearman": corr_spearman(t, p_be),
                    "rmse": rmse(t, p_be),
                    "true_std": float(np.nanstd(t)),
                    "pred_std": float(np.nanstd(p_be)),
                    "collapse_ratio_predstd_over_truestd": float(np.nanstd(p_be) / (np.nanstd(t) + 1e-12)),
                }
            )

    # OLD
    tf = df["old_true_flag"].to_numpy(dtype=float)
    pf = df["old_pred_flag"].to_numpy(dtype=float)
    bs = bin_metrics(tf, pf)
    bs["subset"] = subset_name
    bs["method"] = "old_top2pixel"
    bs["auroc"] = approx_auc(tf, df["old_pred_score"].to_numpy(dtype=float))
    rows_flag.append(bs)
    _push_peak_rows(
        method="old_top2pixel",
        tf=tf,
        pf=pf,
        t_sep=df["old_true_sep"].to_numpy(dtype=float),
        p_sep=df["old_pred_sep"].to_numpy(dtype=float),
        p_sep_best_effort=df["old_pred_sep"].to_numpy(dtype=float),
    )

    # NEW
    tf = df["new_true_flag"].to_numpy(dtype=float)
    pf = df["new_pred_flag"].to_numpy(dtype=float)
    bs = bin_metrics(tf, pf)
    bs["subset"] = subset_name
    bs["method"] = "new_localmax"
    bs["auroc"] = approx_auc(tf, df["new_pred_score"].to_numpy(dtype=float))
    rows_flag.append(bs)
    _push_peak_rows(
        method="new_localmax",
        tf=tf,
        pf=pf,
        t_sep=df["new_true_sep"].to_numpy(dtype=float),
        p_sep=df["new_pred_sep"].to_numpy(dtype=float),
        p_sep_best_effort=df["new_pred_sep_best_effort"].to_numpy(dtype=float),
    )

    return pd.DataFrame(rows_flag), pd.DataFrame(rows_peak)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--labels_csv", default="/data/yn316/Codes/output/ml_runs/nn_psf_labels/labels_psf_weak.csv")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--n_compute", type=int, default=-1)
    # detector defaults
    ap.add_argument("--sigma_smooth", type=float, default=0.8)
    ap.add_argument("--min_peak_sep", type=float, default=2.0)
    ap.add_argument("--k_mad", type=float, default=3.0)
    ap.add_argument("--f_flux", type=float, default=0.02)
    args = ap.parse_args()

    base = Path("/data/yn316/Codes")
    run_root = base / "output" / "ml_runs" / str(args.run_name)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base / "report" / "model_decision" / f"20260227_robust_peaksep_{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(run_root)
    X_test, Yshape_test, Yflux_test, stamp_pix, D = open_test_arrays(manifest)
    n_test = int(manifest["n_test"])
    n_compute = n_test if int(args.n_compute) <= 0 else min(int(args.n_compute), n_test)
    idx = np.arange(n_compute, dtype=int)

    X_sub = np.asarray(X_test[idx], dtype=np.float32)
    yshape_true = np.asarray(Yshape_test[idx], dtype=np.float32)
    yflux_true = np.asarray(Yflux_test[idx], dtype=np.float32)
    yshape_pred = predict_shape(run_root / "models", X_sub, D=D)
    I_true = (yshape_true * (10.0 ** yflux_true[:, None])).reshape((-1, stamp_pix, stamp_pix))
    I_pred = (yshape_pred * (10.0 ** yflux_true[:, None])).reshape((-1, stamp_pix, stamp_pix))

    rows = []
    for i in range(n_compute):
        ot = old_peak_sep(I_true[i])
        op = old_peak_sep(I_pred[i])
        nt = robust_peaksep_and_flag(
            I_true[i],
            sigma_smooth=float(args.sigma_smooth),
            min_peak_sep=float(args.min_peak_sep),
            k_mad=float(args.k_mad),
            f_flux=float(args.f_flux),
        )
        npd = robust_peaksep_and_flag(
            I_pred[i],
            sigma_smooth=float(args.sigma_smooth),
            min_peak_sep=float(args.min_peak_sep),
            k_mad=float(args.k_mad),
            f_flux=float(args.f_flux),
        )
        rows.append(
            {
                "test_index": int(i),
                "old_true_sep": ot,
                "old_pred_sep": op,
                "old_true_flag": float(np.isfinite(ot) and (ot >= 2.0)),
                "old_pred_flag": float(np.isfinite(op) and (op >= 2.0)),
                "old_pred_score": float(op) if np.isfinite(op) else 0.0,
                "new_true_sep": nt["peak_sep_pix"],
                "new_pred_sep": npd["peak_sep_pix"],
                "new_true_flag": nt["multipeak_flag"],
                "new_pred_flag": npd["multipeak_flag"],
                "new_pred_score": npd["peak_score"],
                "new_true_n_peaks": nt["n_valid_peaks"],
                "new_pred_n_peaks": npd["n_valid_peaks"],
                "new_pred_sep_best_effort": robust_best_effort_sep(
                    I_pred[i],
                    sigma_smooth=float(args.sigma_smooth),
                    min_peak_sep=float(args.min_peak_sep),
                ),
            }
        )
    df = pd.DataFrame(rows)

    tr = pd.read_csv(run_root / "trace" / "trace_test.csv")
    tr = tr[tr["test_index"] < n_compute].copy()
    tr["source_id"] = pd.to_numeric(tr["source_id"], errors="coerce")
    df = df.merge(tr[["test_index", "source_id", "field_tag", "ra", "dec"]], on="test_index", how="left")

    labels = pd.read_csv(args.labels_csv, usecols=["source_id", "label"], low_memory=False)
    labels["source_id"] = pd.to_numeric(labels["source_id"], errors="coerce")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    labels = labels.dropna(subset=["source_id", "label"]).copy()
    labels["source_id"] = labels["source_id"].astype(np.int64)
    labels["label"] = labels["label"].astype(np.int64)
    df = df.merge(labels, on="source_id", how="left")
    df.to_csv(out_dir / "peaksep_rows.csv", index=False)

    flag_all, peak_all = evaluate_subset(df.copy(), "all")
    flag_non, peak_non = evaluate_subset(df[df["label"] == 1].copy(), "nonpsf_label1")
    flag_tbl = pd.concat([flag_all, flag_non], ignore_index=True)
    peak_tbl = pd.concat([peak_all, peak_non], ignore_index=True)
    flag_tbl.to_csv(out_dir / "multipeak_flag_metrics.csv", index=False)
    peak_tbl.to_csv(out_dir / "peaksep_conditional_metrics.csv", index=False)

    summary = {
        "run_name": args.run_name,
        "n_test_used": int(n_compute),
        "detector_params": {
            "sigma_smooth": float(args.sigma_smooth),
            "min_peak_sep": float(args.min_peak_sep),
            "k_mad": float(args.k_mad),
            "f_flux": float(args.f_flux),
        },
        "outputs": {
            "rows_csv": str(out_dir / "peaksep_rows.csv"),
            "flag_metrics_csv": str(out_dir / "multipeak_flag_metrics.csv"),
            "peaksep_metrics_csv": str(out_dir / "peaksep_conditional_metrics.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()

