#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


METRICS = [
    "m_concentration_r2_r6",
    "m_asymmetry_180",
    "m_ellipticity",
    "m_peak_sep_pix",
    "m_edge_flux_frac",
]


def _savefig(path: Path, facecolor: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    if facecolor is None:
        plt.savefig(path, dpi=150)
    else:
        plt.savefig(path, dpi=150, facecolor=facecolor)
    plt.close()


def load_manifest(run_root: Path) -> dict:
    mpath = run_root / "manifest_arrays.npz"
    if not mpath.exists():
        raise FileNotFoundError(f"Missing manifest: {mpath}")
    d = np.load(mpath, allow_pickle=True)
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


def extract_morphology(stamp: np.ndarray) -> Dict[str, float]:
    img = np.array(stamp, dtype=np.float64)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    bg = _safe_percentile(img, 10.0)
    ip = img - bg
    ip[ip < 0] = 0.0

    h, w = ip.shape
    yy, xx = np.indices((h, w))
    flux_sum = float(np.sum(ip))
    if flux_sum <= 0:
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        ell = 0.0
    else:
        cx = float(np.sum(ip * xx) / flux_sum)
        cy = float(np.sum(ip * yy) / flux_sum)
        dx = xx - cx
        dy = yy - cy
        qxx = float(np.sum(ip * dx * dx) / flux_sum)
        qyy = float(np.sum(ip * dy * dy) / flux_sum)
        qxy = float(np.sum(ip * dx * dy) / flux_sum)
        tr = qxx + qyy
        det = qxx * qyy - qxy * qxy
        disc = max(tr * tr - 4.0 * det, 0.0)
        lam1 = max((tr + np.sqrt(disc)) * 0.5, 0.0)
        lam2 = max((tr - np.sqrt(disc)) * 0.5, 0.0)
        a = float(np.sqrt(lam1))
        b = float(np.sqrt(lam2))
        ell = float((a - b) / (a + b + 1e-9))

    flat = ip.ravel()
    if flat.size >= 2:
        idx2 = np.argpartition(flat, -2)[-2:]
        vals2 = flat[idx2]
        ord2 = np.argsort(vals2)[::-1]
        i1 = int(idx2[ord2[0]])
        i2 = int(idx2[ord2[1]])
        y1, x1 = divmod(i1, w)
        y2, x2 = divmod(i2, w)
        peak_sep = float(np.hypot(x2 - x1, y2 - y1))
    else:
        peak_sep = 0.0

    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    f_r2 = float(np.sum(ip[rr <= 2.0]))
    f_r6 = float(np.sum(ip[rr <= 6.0]))
    concentration = float(f_r2 / (f_r6 + 1e-9))

    border = np.zeros_like(ip, dtype=bool)
    border[:2, :] = True
    border[-2:, :] = True
    border[:, :2] = True
    border[:, -2:] = True
    edge_flux_frac = float(np.sum(ip[border]) / (flux_sum + 1e-9))

    rot180 = np.flipud(np.fliplr(ip))
    asym = float(np.sum(np.abs(ip - rot180)) / (np.sum(np.abs(ip)) + 1e-9))

    return {
        "m_concentration_r2_r6": concentration,
        "m_asymmetry_180": asym,
        "m_ellipticity": ell,
        "m_peak_sep_pix": peak_sep,
        "m_edge_flux_frac": edge_flux_frac,
    }


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


def plot_metric_diagnostics(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    t = pd.to_numeric(df[f"{metric}_true"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(df[f"{metric}_pred"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(t) & np.isfinite(p)
    if np.sum(m) < 10:
        return
    t = t[m]
    p = p[m]
    r = p - t

    lo = np.nanpercentile(np.r_[t, p], 1)
    hi = np.nanpercentile(np.r_[t, p], 99)

    plt.figure(figsize=(5, 5))
    plt.scatter(t, p, s=5, alpha=0.25)
    plt.plot([lo, hi], [lo, hi], "r--", lw=1)
    plt.xlabel(f"{metric} true")
    plt.ylabel(f"{metric} pred")
    plt.title(f"{metric}: true vs pred")
    _savefig(out_dir / f"{metric}_01_scatter_true_vs_pred.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(t, r, s=5, alpha=0.25)
    plt.axhline(0, color="r", ls="--", lw=1)
    plt.xlabel(f"{metric} true")
    plt.ylabel("pred - true")
    plt.title(f"{metric}: residual vs true")
    _savefig(out_dir / f"{metric}_02_residual_vs_true.png")

    plt.figure(figsize=(6, 4))
    plt.hist(t, bins=60, alpha=0.6, label="true")
    plt.hist(p, bins=60, alpha=0.6, label="pred")
    plt.xlabel(metric)
    plt.ylabel("count")
    plt.title(f"{metric}: distribution")
    plt.legend()
    _savefig(out_dir / f"{metric}_03_distribution_true_vs_pred.png")


def make_range_panel(
    df: pd.DataFrame,
    metric: str,
    group: str,
    idx_col: str,
    I_true: np.ndarray,
    I_pred: np.ndarray,
    out_png: Path,
    n_show: int = 25,
) -> None:
    sub = df[df["range_group"] == group].copy()
    if len(sub) == 0:
        return
    sub = sub.head(n_show).copy().reset_index(drop=True)
    n = len(sub)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols * 3, figsize=(ncols * 3.0, nrows * 2.4), facecolor="black")
    axes = np.array(axes).reshape(nrows, ncols * 3)
    for i in range(nrows * ncols):
        c0 = i * 3
        if i >= n:
            for j in range(3):
                axes.flat[c0 + j].axis("off")
            continue
        row = sub.iloc[i]
        ti = int(row[idx_col])
        t = I_true[ti]
        p = I_pred[ti]
        z = t - p
        lo = np.nanpercentile(np.r_[t.ravel(), p.ravel()], 5)
        hi = np.nanpercentile(np.r_[t.ravel(), p.ravel()], 99.5)
        zr = np.nanpercentile(np.abs(z), 99)
        for j, (img, cmap, vmin, vmax, ttl) in enumerate(
            [(t, "gray", lo, hi, "T"), (p, "gray", lo, hi, "P"), (z, "RdBu_r", -zr, zr, "R")]
        ):
            ax = axes.flat[c0 + j]
            ax.set_facecolor("black")
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_title(f"{int(row['source_id'])}\n{metric}={float(row[f'{metric}_true']):.3g}", color="white", fontsize=6)
            else:
                ax.set_title(ttl, color="white", fontsize=6)
    fig.suptitle(f"{metric} range test: {group}", color="white")
    _savefig(out_png, facecolor="black")


def run_subset(df: pd.DataFrame, name: str, out_dir: Path, I_true: np.ndarray, I_pred: np.ndarray) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in METRICS:
        t = pd.to_numeric(df[f"{m}_true"], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(df[f"{m}_pred"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "subset": name,
                "metric": m,
                "n": int(np.sum(np.isfinite(t) & np.isfinite(p))),
                "pearson": corr_pearson(t, p),
                "spearman": corr_spearman(t, p),
                "rmse": rmse(t, p),
                "true_std": float(np.nanstd(t)),
                "pred_std": float(np.nanstd(p)),
                "collapse_ratio_predstd_over_truestd": float(np.nanstd(p) / (np.nanstd(t) + 1e-12)),
            }
        )
        plot_metric_diagnostics(df, m, out_dir / "metric_diagnostics")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "morph_metric_summary.csv", index=False)

    # range tests on two most informative metrics
    for m in ["m_ellipticity", "m_peak_sep_pix"]:
        v = pd.to_numeric(df[f"{m}_true"], errors="coerce")
        q01 = float(np.nanquantile(v, 0.01))
        q99 = float(np.nanquantile(v, 0.99))
        d = df.copy()
        d["range_group"] = "mid"
        d.loc[v <= q01, "range_group"] = "low_1pct"
        d.loc[v >= q99, "range_group"] = "high_1pct"
        d = d[d["range_group"].isin(["low_1pct", "high_1pct"])].copy()
        d = d.sort_values(f"{m}_true").reset_index(drop=True)
        d.to_csv(out_dir / f"range_test_{m}_rows.csv", index=False)
        make_range_panel(d, m, "low_1pct", "test_index", I_true, I_pred, out_dir / f"range_test_{m}_low_panel.png")
        make_range_panel(d.sort_values(f"{m}_true", ascending=False), m, "high_1pct", "test_index", I_true, I_pred, out_dir / f"range_test_{m}_high_panel.png")

    return {
        "subset": name,
        "n_rows": int(len(df)),
        "summary_csv": str(out_dir / "morph_metric_summary.csv"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--labels_csv", default="/data/yn316/Codes/output/ml_runs/nn_psf_labels/labels_psf_weak.csv")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--n_compute", type=int, default=-1, help="Optional cap on test rows to compute morphology.")
    args = ap.parse_args()

    base = Path("/data/yn316/Codes")
    run_root = base / "output" / "ml_runs" / str(args.run_name)
    plots_root = base / "plots" / "ml_runs" / str(args.run_name)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base / "report" / "model_decision" / f"20260227_morph_sanity_{args.run_name}")
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

    recs = []
    for i in range(n_compute):
        t = extract_morphology(I_true[i])
        p = extract_morphology(I_pred[i])
        row = {"test_index": int(i)}
        for m in METRICS:
            row[f"{m}_true"] = t[m]
            row[f"{m}_pred"] = p[m]
        recs.append(row)
    dm = pd.DataFrame(recs)

    tr = pd.read_csv(run_root / "trace" / "trace_test.csv")
    tr = tr[tr["test_index"] < n_compute].copy()
    tr["source_id"] = pd.to_numeric(tr["source_id"], errors="coerce")
    dm = dm.merge(tr[["test_index", "source_id", "field_tag", "ra", "dec"]], on="test_index", how="left")

    labels = pd.read_csv(args.labels_csv, usecols=["source_id", "label"], low_memory=False)
    labels["source_id"] = pd.to_numeric(labels["source_id"], errors="coerce")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    labels = labels.dropna(subset=["source_id", "label"]).copy()
    labels["source_id"] = labels["source_id"].astype(np.int64)
    labels["label"] = labels["label"].astype(np.int64)
    dm = dm.merge(labels, on="source_id", how="left")

    dm.to_csv(out_dir / "morph_pred_vs_true_rows.csv", index=False)

    all_summary = run_subset(dm.copy(), "all", out_dir / "all", I_true, I_pred)
    non = dm[dm["label"] == 1].copy()
    non_summary = run_subset(non, "nonpsf_label1", out_dir / "nonpsf", I_true, I_pred)

    summary = {
        "run_name": args.run_name,
        "n_test_used": int(n_compute),
        "all": all_summary,
        "nonpsf": non_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()

