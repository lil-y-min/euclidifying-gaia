#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:
    gaussian_filter = None


def parse_floats_csv(s: str) -> List[float]:
    out: List[float] = []
    for tok in str(s).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    return out


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


def _local_maxima_candidates(img: np.ndarray) -> List[Tuple[int, int, float]]:
    h, w = img.shape
    out: List[Tuple[int, int, float]] = []
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
    sigma_smooth: float,
    min_peak_sep: float,
    k_mad: float,
    f_flux: float,
    flux_tiny: float = 1e-6,
    mad_eps: float = 1e-9,
) -> Dict[str, float]:
    ip = _prep_ip(stamp)
    total_flux = float(np.sum(ip))
    if (not np.isfinite(total_flux)) or (total_flux <= flux_tiny):
        return {
            "peak_sep_pix": np.nan,
            "multipeak_flag": 0.0,
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
    filt: List[Tuple[int, int, float]] = []
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
        }

    filt = sorted(filt, key=lambda t: t[2], reverse=True)
    keep: List[Tuple[int, int, float]] = []
    for y, x, v in filt:
        ok = True
        for yy, xx, _ in keep:
            if np.hypot(x - xx, y - yy) < min_peak_sep:
                ok = False
                break
        if ok:
            keep.append((y, x, v))

    if len(keep) < 2:
        return {
            "peak_sep_pix": np.nan,
            "multipeak_flag": 0.0,
        }
    (y1, x1, _), (y2, x2, _) = keep[0], keep[1]
    sep = float(np.hypot(x2 - x1, y2 - y1))
    return {
        "peak_sep_pix": sep,
        "multipeak_flag": 1.0,
    }


def build_peak_cache(
    stamps: np.ndarray,
    sigma_smooth: float,
    flux_tiny: float = 1e-6,
    mad_eps: float = 1e-9,
) -> List[Dict[str, object]]:
    cache: List[Dict[str, object]] = []
    for i in range(stamps.shape[0]):
        ip = _prep_ip(stamps[i])
        total_flux = float(np.sum(ip))
        if (not np.isfinite(total_flux)) or (total_flux <= flux_tiny):
            cache.append({"valid": False})
            continue
        if gaussian_filter is not None and sigma_smooth > 0:
            sm = gaussian_filter(ip, sigma=sigma_smooth, mode="nearest")
        else:
            sm = ip
        med = float(np.median(sm))
        mad = float(np.median(np.abs(sm - med)))
        mad = max(mad, mad_eps)
        cands = _local_maxima_candidates(sm)
        cands = sorted(cands, key=lambda t: t[2], reverse=True)
        cache.append(
            {
                "valid": True,
                "total_flux": total_flux,
                "med": med,
                "mad": mad,
                "cands": cands,
            }
        )
    return cache


def detect_from_cache(
    cache: List[Dict[str, object]],
    k_mad: float,
    f_flux: float,
    min_peak_sep: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(cache)
    flag = np.zeros(n, dtype=np.float64)
    sep = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        c = cache[i]
        if not bool(c.get("valid", False)):
            continue
        total_flux = float(c["total_flux"])  # type: ignore[index]
        med = float(c["med"])  # type: ignore[index]
        mad = float(c["mad"])  # type: ignore[index]
        cands = c["cands"]  # type: ignore[index]
        thr = med + float(k_mad) * mad

        keep: List[Tuple[int, int, float]] = []
        for y, x, v in cands:  # type: ignore[assignment]
            if v <= thr:
                continue
            if (v / (total_flux + 1e-12)) <= f_flux:
                continue
            ok = True
            for yy, xx, _ in keep:
                if np.hypot(x - xx, y - yy) < min_peak_sep:
                    ok = False
                    break
            if ok:
                keep.append((y, x, v))
                if len(keep) >= 2:
                    break
        if len(keep) >= 2:
            (y1, x1, _), (y2, x2, _) = keep[0], keep[1]
            sep[i] = float(np.hypot(x2 - x1, y2 - y1))
            flag[i] = 1.0
    return flag, sep


def corr_pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return np.nan
    x = a[m]
    y = b[m]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0 or sy <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def summarize_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    t = y_true > 0.5
    p = y_pred > 0.5
    tp = float(np.sum(t & p))
    fp = float(np.sum((~t) & p))
    fn = float(np.sum(t & (~p)))
    tn = float(np.sum((~t) & (~p)))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "p_true_flag1": float(np.mean(t)),
        "p_pred_flag1": float(np.mean(p)),
        "n_pred_flag1": float(np.sum(p)),
    }


def peaky_scalar_rows(I_true: np.ndarray, I_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    lap = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    for i in range(I_true.shape[0]):
        t = _prep_ip(I_true[i])
        p = _prep_ip(I_pred[i])
        t_sum = float(np.sum(t)) + 1e-12
        p_sum = float(np.sum(p)) + 1e-12

        t_flat = t.ravel()
        p_flat = p.ravel()
        t_top2 = np.partition(t_flat, -2)[-2:] if t_flat.size >= 2 else np.array([np.nan, np.nan], dtype=np.float64)
        p_top2 = np.partition(p_flat, -2)[-2:] if p_flat.size >= 2 else np.array([np.nan, np.nan], dtype=np.float64)

        # Small, explicit 3x3 convolution for Laplacian energy on 20x20 stamps.
        lt = 0.0
        lp = 0.0
        for y in range(1, t.shape[0] - 1):
            for x in range(1, t.shape[1] - 1):
                lt += float((t[y - 1 : y + 2, x - 1 : x + 2] * lap).sum() ** 2)
                lp += float((p[y - 1 : y + 2, x - 1 : x + 2] * lap).sum() ** 2)

        rows.append(
            {
                "max_over_sum_true": float(np.nanmax(t_flat) / t_sum),
                "max_over_sum_pred": float(np.nanmax(p_flat) / p_sum),
                "top2_over_sum_true": float(np.nansum(t_top2) / t_sum),
                "top2_over_sum_pred": float(np.nansum(p_top2) / p_sum),
                "lap_energy_true": lt,
                "lap_energy_pred": lp,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_run_name", required=True, help="Run providing boosters.")
    ap.add_argument("--data_run_name", default="", help="Run providing X_test/Y_test/trace. Defaults to model run.")
    ap.add_argument("--labels_csv", default="/data/yn316/Codes/output/ml_runs/nn_psf_labels/labels_psf_weak.csv")
    ap.add_argument("--quality_flags_csv", default="", help="Optional CSV with source_id, quality_flag.")
    ap.add_argument("--quality_bits_mask", type=int, default=2087, help="Rows with (quality_flag & mask)!=0 are manual-removed.")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--n_compute", type=int, default=-1)
    ap.add_argument("--sigma_smooth", type=float, default=0.8)
    ap.add_argument("--min_peak_sep", type=float, default=2.0)
    ap.add_argument("--base_k_mad", type=float, default=3.0)
    ap.add_argument("--base_f_flux", type=float, default=0.005)
    ap.add_argument("--k_mad_values", default="3.0,2.5,2.0,1.5,1.0,0.5")
    ap.add_argument("--f_flux_values", default="0.02,0.01,0.0075,0.005,0.003,0.002,0.001")
    args = ap.parse_args()

    base = Path("/data/yn316/Codes")
    model_run = str(args.model_run_name)
    data_run = str(args.data_run_name).strip() if str(args.data_run_name).strip() else model_run

    model_root = base / "output" / "ml_runs" / model_run
    data_root = base / "output" / "ml_runs" / data_run
    if not model_root.exists():
        raise FileNotFoundError(f"Missing model run: {model_root}")
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data run: {data_root}")

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base / "report" / "model_decision" / f"20260302_peaksep_sweep_model_{model_run}_data_{data_run}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(data_root)
    X_test, Yshape_test, Yflux_test, stamp_pix, D = open_test_arrays(manifest)
    if int(D) != 400 or int(stamp_pix) != 20:
        raise RuntimeError(f"Unexpected shape config D={D}, stamp_pix={stamp_pix}.")

    n_test = int(manifest["n_test"])
    n_compute = n_test if int(args.n_compute) <= 0 else min(int(args.n_compute), n_test)
    idx = np.arange(n_compute, dtype=int)

    X_sub = np.asarray(X_test[idx], dtype=np.float32)
    yshape_true = np.asarray(Yshape_test[idx], dtype=np.float32)
    yflux_true = np.asarray(Yflux_test[idx], dtype=np.float32)
    yshape_pred = predict_shape(model_root / "models", X_sub, D=D)

    I_true = (yshape_true * (10.0 ** yflux_true[:, None])).reshape((-1, stamp_pix, stamp_pix))
    I_pred = (yshape_pred * (10.0 ** yflux_true[:, None])).reshape((-1, stamp_pix, stamp_pix))

    tr = pd.read_csv(data_root / "trace" / "trace_test.csv")
    tr = tr[tr["test_index"] < n_compute].copy()
    tr["source_id"] = pd.to_numeric(tr["source_id"], errors="coerce")
    meta = tr[["test_index", "source_id", "field_tag"]].copy()

    labels = pd.read_csv(args.labels_csv, usecols=["source_id", "label"], low_memory=False)
    labels["source_id"] = pd.to_numeric(labels["source_id"], errors="coerce")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    labels = labels.dropna(subset=["source_id", "label"]).copy()
    labels["source_id"] = labels["source_id"].astype(np.int64)
    labels["label"] = labels["label"].astype(np.int64)
    meta = meta.merge(labels, on="source_id", how="left")

    if str(args.quality_flags_csv).strip():
        qf = pd.read_csv(args.quality_flags_csv, usecols=["source_id", "quality_flag"], low_memory=False)
        qf["source_id"] = pd.to_numeric(qf["source_id"], errors="coerce")
        qf["quality_flag"] = pd.to_numeric(qf["quality_flag"], errors="coerce").fillna(0).astype(np.int64)
        qf = qf.dropna(subset=["source_id"]).copy()
        qf["source_id"] = qf["source_id"].astype(np.int64)
        qf["manual_removed"] = (qf["quality_flag"] & int(args.quality_bits_mask)) != 0
        meta = meta.merge(qf[["source_id", "manual_removed"]], on="source_id", how="left")
        meta["manual_removed"] = meta["manual_removed"].fillna(False).astype(bool)
    else:
        meta["manual_removed"] = False

    peaky = peaky_scalar_rows(I_true, I_pred)
    row_meta = pd.DataFrame(
        {
            "test_index": idx,
            "source_id": pd.to_numeric(meta["source_id"], errors="coerce"),
            "label": pd.to_numeric(meta["label"], errors="coerce"),
            "manual_removed": meta["manual_removed"].astype(bool),
        }
    )
    peaky_rows = pd.concat([row_meta.reset_index(drop=True), peaky.reset_index(drop=True)], axis=1)
    peaky_rows.to_csv(out_dir / "peaky_rows.csv", index=False)

    def peaky_summary(subset_name: str, mask: np.ndarray) -> pd.DataFrame:
        d = peaky_rows.loc[mask].copy()
        if d.empty:
            return pd.DataFrame([{"subset": subset_name, "n_rows": 0}])
        out = {
            "subset": subset_name,
            "n_rows": int(len(d)),
        }
        for c in [
            "max_over_sum_true",
            "max_over_sum_pred",
            "top2_over_sum_true",
            "top2_over_sum_pred",
            "lap_energy_true",
            "lap_energy_pred",
        ]:
            arr = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
            out[f"{c}_mean"] = float(np.nanmean(arr))
            out[f"{c}_median"] = float(np.nanmedian(arr))
        return pd.DataFrame([out])

    subset_masks: List[Tuple[str, np.ndarray]] = [("all", np.ones(n_compute, dtype=bool))]
    if np.any(meta["label"].to_numpy(dtype=float) == 1):
        subset_masks.append(("nonpsf_label1", meta["label"].to_numpy(dtype=float) == 1))
    if str(args.quality_flags_csv).strip():
        mrem = meta["manual_removed"].to_numpy(dtype=bool)
        subset_masks.append(("manual_removed", mrem))
        subset_masks.append(("manual_kept", ~mrem))

    peaky_sum_rows: List[pd.DataFrame] = []

    k_values = parse_floats_csv(args.k_mad_values)
    f_values = parse_floats_csv(args.f_flux_values)

    # Compute detector outputs per setting, then derive metrics per subset.
    fixed_rows: List[pd.DataFrame] = []
    true_cache = build_peak_cache(I_true, sigma_smooth=float(args.sigma_smooth))
    pred_cache = build_peak_cache(I_pred, sigma_smooth=float(args.sigma_smooth))

    for axis, val_list in [("k_mad", k_values), ("f_flux", f_values)]:
        for val in val_list:
            k = float(val) if axis == "k_mad" else float(args.base_k_mad)
            f = float(args.base_f_flux) if axis == "k_mad" else float(val)
            t_flag, t_sep = detect_from_cache(true_cache, k_mad=k, f_flux=f, min_peak_sep=float(args.min_peak_sep))
            p_flag, p_sep = detect_from_cache(pred_cache, k_mad=k, f_flux=f, min_peak_sep=float(args.min_peak_sep))
            for subset_name, mask in subset_masks:
                tt = t_flag[mask]
                pp = p_flag[mask]
                ts = t_sep[mask]
                ps = p_sep[mask]
                bm = summarize_binary(tt, pp)
                pm = pp > 0.5
                tpm = (tt > 0.5) & pm
                pred_cond = pm & np.isfinite(ts) & np.isfinite(ps)
                tp_cond = tpm & np.isfinite(ts) & np.isfinite(ps)
                fixed_rows.append(
                    pd.DataFrame(
                        [
                            {
                                "subset": subset_name,
                                "sweep_axis": axis,
                                "k_mad": k,
                                "f_flux": f,
                                "n_rows": int(np.sum(mask)),
                                **bm,
                                "n_tp_only_eval": int(np.sum(tp_cond)),
                                "tp_only_pearson_sep": corr_pearson(ts[tp_cond], ps[tp_cond]),
                                "tp_only_rmse_sep": rmse(ts[tp_cond], ps[tp_cond]),
                                "n_predflag_eval": int(np.sum(pred_cond)),
                                "predflag_pearson_sep": corr_pearson(ts[pred_cond], ps[pred_cond]),
                                "predflag_rmse_sep": rmse(ts[pred_cond], ps[pred_cond]),
                            }
                        ]
                    )
                )

    sweep_tbl = pd.concat(fixed_rows, ignore_index=True)
    sweep_tbl.to_csv(out_dir / "threshold_sweep_metrics.csv", index=False)

    for name, mask in subset_masks:
        peaky_sum_rows.append(peaky_summary(name, mask))
    peaky_summary_tbl = pd.concat(peaky_sum_rows, ignore_index=True)
    peaky_summary_tbl.to_csv(out_dir / "peaky_summary.csv", index=False)

    summary = {
        "model_run_name": model_run,
        "data_run_name": data_run,
        "n_test_data_run": int(n_test),
        "n_used": int(n_compute),
        "sigma_smooth": float(args.sigma_smooth),
        "min_peak_sep": float(args.min_peak_sep),
        "base_k_mad": float(args.base_k_mad),
        "base_f_flux": float(args.base_f_flux),
        "k_mad_values": k_values,
        "f_flux_values": f_values,
        "quality_flags_csv": str(args.quality_flags_csv),
        "quality_bits_mask": int(args.quality_bits_mask),
        "outputs": {
            "threshold_sweep_metrics_csv": str(out_dir / "threshold_sweep_metrics.csv"),
            "peaky_rows_csv": str(out_dir / "peaky_rows.csv"),
            "peaky_summary_csv": str(out_dir / "peaky_summary.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
