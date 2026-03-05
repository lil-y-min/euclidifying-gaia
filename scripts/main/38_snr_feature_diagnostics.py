"""
38_snr_feature_diagnostics.py
============================

Run diagnostics centered on feat_log10_snr:
1) Plot SNR vs every Gaia feature (hexbin + Spearman).
2) Plot model error metric vs SNR bins.
3) Quantify reliance on SNR via permutation test.

Supports two run types produced in this repo:
- Reconstruct run (14a + 16): uses booster_comp_*.json + PCA to compute rmse_pix.
- Pixels+flux run (14b + 15): uses booster_flux.json to compute abs flux error on log10(F).

Outputs:
  plots/ml_runs/<RUN>/snr_diagnostics/
    00_snr_vs_feature_spearman.csv
    00_snr_vs_feature_spearman_bar.png
    01_snr_vs_feature_XX_<feature>.png
    10_metric_vs_snr_bins.csv
    10_metric_vs_snr_bins.png
    11_snr_permutation_importance.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

MORPH_COLS: List[str] = [
    "m_concentration_r2_r6",
    "m_asymmetry_180",
    "m_ellipticity",
    "m_peak_sep_pix",
    "m_edge_flux_frac",
    "m_peak_ratio_2over1",
]


@dataclass
class Config:
    base_dir: Path = Path(__file__).resolve().parents[2]  # .../Codes
    run_name: str = ""
    out_subdir: str = "snr_diagnostics"

    max_points_plot: int = 150_000
    snr_bins: int = 12

    # Metric computation
    batch_size: int = 20_000

    # Permutation reliance test
    perm_repeats: int = 3
    perm_sample_size: int = 20_000
    seed: int = 123


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def load_manifest(run_root: Path) -> Dict:
    p = run_root / "manifest_arrays.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing manifest: {p}")
    d = np.load(p, allow_pickle=True)
    out = {k: d[k] for k in d.files}

    if "feature_cols" in out:
        out["feature_cols"] = [str(x) for x in out["feature_cols"].tolist()]

    int_keys = [
        "n_train",
        "n_val",
        "n_test",
        "n_features",
        "K",
        "D",
        "stamp_pix",
    ]
    for k in int_keys:
        if k in out:
            out[k] = int(out[k])

    str_keys = [
        "X_test_path",
        "Y_test_path",
        "Yflux_test_path",
        "pca_npz",
        "scaler_npz",
        "plots_dir",
        "trace_test_csv",
    ]
    for k in str_keys:
        if k in out:
            out[k] = str(out[k])
    return out


def resolve_manifest_path(
    p_raw: str,
    *,
    base_dir: Path,
    run_root: Path,
    run_name: str,
    must_exist: bool = True,
) -> Path:
    p = Path(str(p_raw))
    if p.exists():
        return p

    s = str(p_raw).replace("\\", "/")

    anchor_run = f"/output/ml_runs/{run_name}/"
    if anchor_run in s:
        rel = s.split(anchor_run, 1)[1].lstrip("/")
        cand = base_dir / "output" / "ml_runs" / run_name / rel
        if cand.exists() or not must_exist:
            return cand

    if "/output/" in s:
        rel = s.split("/output/", 1)[1].lstrip("/")
        cand = base_dir / "output" / rel
        if cand.exists() or not must_exist:
            return cand

    anchor_plot = f"/plots/ml_runs/{run_name}/"
    if anchor_plot in s:
        rel = s.split(anchor_plot, 1)[1].lstrip("/")
        cand = base_dir / "plots" / "ml_runs" / run_name / rel
        if cand.exists() or not must_exist:
            return cand

    if "/plots/" in s:
        rel = s.split("/plots/", 1)[1].lstrip("/")
        cand = base_dir / "plots" / rel
        if cand.exists() or not must_exist:
            return cand

    # Last resort: same filename under run root.
    cand = run_root / p.name
    if cand.exists() or not must_exist:
        return cand

    if must_exist:
        raise FileNotFoundError(f"Could not resolve path: {p_raw}")
    return p


def load_scaler(npz_path: Path, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    idx = {n: i for i, n in enumerate(names)}
    miss = [c for c in feature_cols if c not in idx]
    if miss:
        raise RuntimeError(f"Scaler missing features: {miss}")
    ii = np.array([idx[c] for c in feature_cols], dtype=int)
    y_min = d["y_min"].astype(np.float32)[ii]
    y_iqr = d["y_iqr"].astype(np.float32)[ii]
    y_iqr = np.where(np.abs(y_iqr) > 0, y_iqr, 1.0).astype(np.float32)
    return y_min, y_iqr


def maybe_to_raw(X_scaled: np.ndarray, manifest: Dict, feature_cols: List[str]) -> np.ndarray:
    scaler_npz = manifest.get("scaler_npz")
    if not scaler_npz:
        return X_scaled
    sp = Path(scaler_npz)
    if not sp.exists():
        return X_scaled
    try:
        y_min, y_iqr = load_scaler(sp, feature_cols)
        return (X_scaled * y_iqr[None, :]) + y_min[None, :]
    except Exception:
        return X_scaled


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def detect_metric_mode(run_root: Path, manifest: Dict) -> str:
    model_dir = run_root / "models"
    # reconstruct mode: component boosters + pca + Y_test_path
    if (
        "K" in manifest
        and "pca_npz" in manifest
        and "Y_test_path" in manifest
        and (model_dir / "booster_comp_000.json").exists()
    ):
        return "reconstruct_rmse"

    # pixels+flux mode: flux booster + Yflux
    if "Yflux_test_path" in manifest and (model_dir / "booster_flux.json").exists():
        return "flux_abs_err"

    return "none"


def _predict_coeffs(boosters: List[xgb.Booster], X: np.ndarray, batch_size: int) -> np.ndarray:
    n = X.shape[0]
    k = len(boosters)
    out = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        dmat = xgb.DMatrix(np.asarray(X[s:e], dtype=np.float32, order="C"))
        for j, b in enumerate(boosters):
            out[s:e, j] = b.predict(dmat).astype(np.float32, copy=False)
    return out


def compute_reconstruct_rmse(
    run_root: Path,
    manifest: Dict,
    X_test: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model_dir = run_root / "models"
    k = int(manifest["K"])
    n = X_test.shape[0]

    boosters: List[xgb.Booster] = []
    for j in range(k):
        p = model_dir / f"booster_comp_{j:03d}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")
        b = xgb.Booster()
        b.load_model(str(p))
        boosters.append(b)

    base_dir = Path(__file__).resolve().parents[2]
    run_name = run_root.name
    pca_path = resolve_manifest_path(
        manifest["pca_npz"],
        base_dir=base_dir,
        run_root=run_root,
        run_name=run_name,
        must_exist=True,
    )
    pca = np.load(pca_path, allow_pickle=True)
    mean = pca["mean"].astype(np.float32)
    components = pca["components"].astype(np.float32)

    y_path = resolve_manifest_path(
        manifest["Y_test_path"],
        base_dir=base_dir,
        run_root=run_root,
        run_name=run_name,
        must_exist=True,
    )
    Y_test = np.memmap(y_path, dtype="float32", mode="r", shape=(n, k))

    rmse = np.empty(n, dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        pred = _predict_coeffs(boosters, X_test[s:e], batch_size=max(1024, min(batch_size, e - s)))
        true = np.asarray(Y_test[s:e], dtype=np.float32)
        tf = true @ components + mean[None, :]
        pf = pred @ components + mean[None, :]
        d = tf - pf
        rmse[s:e] = np.sqrt(np.mean(d * d, axis=1)).astype(np.float32)
    return rmse


def load_reconstruct_artifacts(
    run_root: Path,
    manifest: Dict,
    base_dir: Path,
    run_name: str,
) -> Tuple[List[xgb.Booster], np.ndarray, np.ndarray]:
    model_dir = run_root / "models"
    k = int(manifest["K"])
    boosters: List[xgb.Booster] = []
    for j in range(k):
        p = model_dir / f"booster_comp_{j:03d}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")
        b = xgb.Booster()
        b.load_model(str(p))
        boosters.append(b)
    pca_path = resolve_manifest_path(
        manifest["pca_npz"],
        base_dir=base_dir,
        run_root=run_root,
        run_name=run_name,
        must_exist=True,
    )
    pca = np.load(pca_path, allow_pickle=True)
    mean = pca["mean"].astype(np.float32)
    components = pca["components"].astype(np.float32)
    return boosters, mean, components


def reconstruct_rmse_from_arrays(
    boosters: List[xgb.Booster],
    mean: np.ndarray,
    components: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    n = X.shape[0]
    rmse = np.empty(n, dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        pred = _predict_coeffs(boosters, X[s:e], batch_size=max(512, e - s))
        true = np.asarray(Y[s:e], dtype=np.float32)
        tf = true @ components + mean[None, :]
        pf = pred @ components + mean[None, :]
        d = tf - pf
        rmse[s:e] = np.sqrt(np.mean(d * d, axis=1)).astype(np.float32)
    return rmse


def compute_flux_abs_err(run_root: Path, manifest: Dict, X_test: np.ndarray, batch_size: int) -> np.ndarray:
    model_dir = run_root / "models"
    b = xgb.Booster()
    b.load_model(str(model_dir / "booster_flux.json"))

    y_path = resolve_manifest_path(
        manifest["Yflux_test_path"],
        base_dir=Path(__file__).resolve().parents[2],
        run_root=run_root,
        run_name=run_root.name,
        must_exist=True,
    )
    n = X_test.shape[0]
    y_true = np.memmap(y_path, dtype="float32", mode="r", shape=(n,))

    y_pred = np.empty(n, dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        dmat = xgb.DMatrix(np.asarray(X_test[s:e], dtype=np.float32, order="C"))
        y_pred[s:e] = b.predict(dmat).astype(np.float32, copy=False)
    return np.abs(y_pred - np.asarray(y_true, dtype=np.float32))


def compute_metric(run_root: Path, manifest: Dict, X_test: np.ndarray, batch_size: int) -> Tuple[str, Optional[np.ndarray]]:
    mode = detect_metric_mode(run_root, manifest)
    if mode == "reconstruct_rmse":
        return mode, compute_reconstruct_rmse(run_root, manifest, X_test, batch_size)
    if mode == "flux_abs_err":
        return mode, compute_flux_abs_err(run_root, manifest, X_test, batch_size)
    return mode, None


def choose_sample_idx(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False).astype(np.int64))


def plot_snr_vs_features(
    out_dir: Path,
    X_raw: np.ndarray,
    feature_cols: List[str],
    snr_idx: int,
    max_points_plot: int,
    seed: int,
) -> None:
    n = X_raw.shape[0]
    idx = choose_sample_idx(n, max_points_plot, seed)
    Xp = X_raw[idx]

    snr = Xp[:, snr_idx]
    rows = []
    feat_order = [f for f in feature_cols if f != "feat_log10_snr"]
    for j, feat in enumerate(feature_cols):
        if j == snr_idx:
            continue
        y = Xp[:, j]
        good = np.isfinite(snr) & np.isfinite(y)
        if int(np.sum(good)) < 50:
            rows.append({"feature": feat, "spearman_snr": np.nan, "n": int(np.sum(good))})
            continue

        c = spearman_corr(snr[good], y[good])
        rows.append({"feature": feat, "spearman_snr": c, "n": int(np.sum(good))})

        plt.figure(figsize=(6.5, 5.2))
        plt.hexbin(snr[good], y[good], gridsize=75, bins="log", mincnt=1, cmap="viridis")
        cb = plt.colorbar()
        cb.set_label("log10(count)")
        plt.xlabel("feat_log10_snr")
        plt.ylabel(feat)
        plt.title(f"SNR vs {feat} | Spearman={c:.3f}")
        savefig(out_dir / f"01_snr_vs_feature_{j + 1:02d}_{feat}.png")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "00_snr_vs_feature_spearman.csv", index=False)

    d = df.dropna(subset=["spearman_snr"]).copy()
    if not d.empty:
        # Keep canonical feature order so bars align across different runs.
        order_map = {name: i for i, name in enumerate(feat_order)}
        d["__ord"] = d["feature"].map(order_map)
        d = d.sort_values("__ord", ascending=True).drop(columns=["__ord"])
        plt.figure(figsize=(8.8, max(4.8, 0.35 * len(d))))
        plt.barh(d["feature"], np.abs(d["spearman_snr"].to_numpy(dtype=float)))
        plt.xlabel("|Spearman(feat_log10_snr, feature)|")
        plt.title("SNR correlation strength with Gaia features")
        savefig(out_dir / "00_snr_vs_feature_spearman_bar.png")


def plot_anchor_vs_features(
    out_dir: Path,
    anchor: np.ndarray,
    anchor_name: str,
    feat_mat: np.ndarray,
    feat_names: List[str],
    csv_name: str,
    bar_png_name: str,
    hex_prefix: str,
    max_points_plot: int,
    seed: int,
) -> None:
    n = int(anchor.shape[0])
    idx = choose_sample_idx(n, max_points_plot, seed)
    a = np.asarray(anchor[idx], dtype=np.float32)
    f = np.asarray(feat_mat[idx], dtype=np.float32)

    rows = []
    for j, feat in enumerate(feat_names):
        y = f[:, j]
        good = np.isfinite(a) & np.isfinite(y)
        ng = int(np.sum(good))
        if ng < 50:
            rows.append({"feature": feat, "spearman": np.nan, "n": ng})
            continue

        c = spearman_corr(a[good], y[good])
        rows.append({"feature": feat, "spearman": c, "n": ng})

        plt.figure(figsize=(6.5, 5.2))
        plt.hexbin(a[good], y[good], gridsize=75, bins="log", mincnt=1, cmap="viridis")
        cb = plt.colorbar()
        cb.set_label("log10(count)")
        plt.xlabel(anchor_name)
        plt.ylabel(feat)
        plt.title(f"{anchor_name} vs {feat} | Spearman={c:.3f}")
        savefig(out_dir / f"{hex_prefix}_{j + 1:02d}_{feat}.png")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / csv_name, index=False)

    d = df.dropna(subset=["spearman"]).copy()
    if d.empty:
        return
    order_map = {name: i for i, name in enumerate(feat_names)}
    d["__ord"] = d["feature"].map(order_map)
    d = d.sort_values("__ord", ascending=True).drop(columns=["__ord"])
    plt.figure(figsize=(8.8, max(4.8, 0.45 * len(d))))
    plt.barh(d["feature"], np.abs(d["spearman"].to_numpy(dtype=float)))
    plt.xlabel(f"|Spearman({anchor_name}, feature)|")
    plt.title(f"Correlation strength vs {anchor_name}")
    savefig(out_dir / bar_png_name)


def plot_metric_vs_snr_bins(
    out_dir: Path,
    snr: np.ndarray,
    metric: np.ndarray,
    metric_label: str,
    n_bins: int,
) -> pd.DataFrame:
    good = np.isfinite(snr) & np.isfinite(metric)
    s = snr[good]
    m = metric[good]

    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(s, q)
    edges = np.unique(edges)
    if edges.size < 3:
        raise RuntimeError("SNR has too few unique values to form bins")

    b = np.digitize(s, edges[1:-1], right=False)
    rows = []
    for i in range(edges.size - 1):
        mask = b == i
        if np.sum(mask) == 0:
            continue
        sm = s[mask]
        mm = m[mask]
        rows.append(
            {
                "bin": i,
                "snr_left": float(edges[i]),
                "snr_right": float(edges[i + 1]),
                "snr_center": float(np.median(sm)),
                "n": int(mask.sum()),
                "metric_mean": float(np.mean(mm)),
                "metric_median": float(np.median(mm)),
                "metric_p90": float(np.percentile(mm, 90)),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "10_metric_vs_snr_bins.csv", index=False)

    plt.figure(figsize=(7.4, 4.8))
    plt.plot(df["snr_center"], df["metric_mean"], marker="o", label="mean")
    plt.plot(df["snr_center"], df["metric_median"], marker="s", label="median")
    plt.xlabel("feat_log10_snr (bin median)")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} vs feat_log10_snr")
    plt.legend()
    savefig(out_dir / "10_metric_vs_snr_bins.png")

    return df


def metric_mean_for_subset(
    mode: str,
    run_root: Path,
    manifest: Dict,
    X_sub: np.ndarray,
    batch_size: int,
) -> float:
    if mode == "reconstruct_rmse":
        raise RuntimeError("Use reconstruct_rmse_from_arrays for reconstruct subset metric")
    if mode == "flux_abs_err":
        err = compute_flux_abs_err(run_root, manifest, X_sub, batch_size)
        return float(np.mean(err))
    return float("nan")


def snr_permutation_test(
    out_dir: Path,
    mode: str,
    run_root: Path,
    manifest: Dict,
    X_test: np.ndarray,
    feature_cols: List[str],
    snr_idx: int,
    baseline_metric: np.ndarray,
    perm_sample_size: int,
    perm_repeats: int,
    seed: int,
    batch_size: int,
) -> None:
    if baseline_metric is None:
        return

    rng = np.random.default_rng(seed)
    n = X_test.shape[0]
    idx = choose_sample_idx(n, min(perm_sample_size, n), seed)

    Xs = np.asarray(X_test[idx], dtype=np.float32)
    base = float(np.mean(np.asarray(baseline_metric[idx], dtype=np.float32)))

    rows = []

    if mode == "reconstruct_rmse":
        boosters, mean, components = load_reconstruct_artifacts(
            run_root,
            manifest,
            base_dir=Path(__file__).resolve().parents[2],
            run_name=run_root.name,
        )
        k = int(manifest["K"])
        y_path = resolve_manifest_path(
            manifest["Y_test_path"],
            base_dir=Path(__file__).resolve().parents[2],
            run_root=run_root,
            run_name=run_root.name,
            must_exist=True,
        )
        y_all = np.memmap(y_path, dtype="float32", mode="r", shape=(X_test.shape[0], k))
        Ys = np.asarray(y_all[idx], dtype=np.float32)
        snr = Xs[:, snr_idx]

        for r in range(perm_repeats):
            p = rng.permutation(snr)
            Xp = Xs.copy()
            Xp[:, snr_idx] = p
            rm = reconstruct_rmse_from_arrays(boosters, mean, components, Xp, Ys, batch_size=batch_size)
            perm_mean = float(np.mean(rm))
            rows.append(
                {
                    "repeat": r,
                    "baseline_metric_mean": base,
                    "perm_metric_mean": perm_mean,
                    "delta_metric_mean": perm_mean - base,
                    "baseline_spearman_snr_metric": np.nan,
                    "perm_spearman_snr_metric": np.nan,
                    "delta_abs_spearman": np.nan,
                    "note": "reconstruct mode: prediction degradation after SNR permutation",
                }
            )
    elif mode == "flux_abs_err":
        for r in range(perm_repeats):
            Xp = Xs.copy()
            Xp[:, snr_idx] = rng.permutation(Xp[:, snr_idx])
            perm_mean = metric_mean_for_subset(mode, run_root, manifest, Xp, batch_size)
            rows.append(
                {
                    "repeat": r,
                    "baseline_metric_mean": base,
                    "perm_metric_mean": perm_mean,
                    "delta_metric_mean": perm_mean - base,
                    "baseline_spearman_snr_metric": np.nan,
                    "perm_spearman_snr_metric": np.nan,
                    "delta_abs_spearman": np.nan,
                    "note": "flux mode: prediction degradation after SNR permutation",
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / "11_snr_permutation_importance.csv", index=False)


def load_morph_joined_test_table(
    *,
    base_dir: Path,
    run_root: Path,
    run_name: str,
    manifest: Dict,
    n_test: int,
    morph_csv_raw: str,
) -> pd.DataFrame:
    trace_raw = str(manifest.get("trace_test_csv", str(run_root / "trace" / "trace_test.csv")))
    trace_path = resolve_manifest_path(
        trace_raw,
        base_dir=base_dir,
        run_root=run_root,
        run_name=run_name,
        must_exist=True,
    )
    morph_path = resolve_manifest_path(
        morph_csv_raw,
        base_dir=base_dir,
        run_root=run_root,
        run_name=run_name,
        must_exist=True,
    )

    tr = pd.read_csv(trace_path, usecols=["test_index", "source_id"])
    tr["test_index"] = pd.to_numeric(tr["test_index"], errors="coerce")
    tr["source_id"] = pd.to_numeric(tr["source_id"], errors="coerce")
    tr = tr.dropna(subset=["test_index", "source_id"]).copy()
    tr["test_index"] = tr["test_index"].astype(np.int64)
    tr["source_id"] = tr["source_id"].astype(np.int64)
    tr = tr[(tr["test_index"] >= 0) & (tr["test_index"] < int(n_test))].copy()
    tr = tr.drop_duplicates(subset=["test_index"], keep="first")

    m = pd.read_csv(morph_path, low_memory=False)
    keep_cols = ["source_id"] + [c for c in MORPH_COLS if c in m.columns]
    if len(keep_cols) <= 1:
        raise RuntimeError(f"No morphology columns found in morph CSV: {morph_path}")
    m = m[keep_cols].copy()
    m["source_id"] = pd.to_numeric(m["source_id"], errors="coerce")
    for c in keep_cols[1:]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["source_id"]).copy()
    m["source_id"] = m["source_id"].astype(np.int64)
    m = m.drop_duplicates(subset=["source_id"], keep="first")

    j = tr.merge(m, on="source_id", how="inner")
    if j.empty:
        raise RuntimeError("No overlapping source_id between trace_test.csv and morph CSV")
    return j


def main() -> None:
    ap = argparse.ArgumentParser(description="SNR-centric Gaia feature diagnostics")
    ap.add_argument("--run", required=True, help="Run name under output/ml_runs/")
    ap.add_argument("--max_points_plot", type=int, default=150000)
    ap.add_argument("--snr_bins", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=20000)
    ap.add_argument("--perm_repeats", type=int, default=3)
    ap.add_argument("--perm_sample_size", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--morph_csv",
        default="",
        help="Optional morphology CSV with source_id + m_* columns. "
        "Default: output/ml_runs/nn_psf_labels/labels_psf_weak.csv (if present).",
    )
    args = ap.parse_args()

    cfg = Config(
        run_name=args.run,
        max_points_plot=args.max_points_plot,
        snr_bins=args.snr_bins,
        batch_size=args.batch_size,
        perm_repeats=args.perm_repeats,
        perm_sample_size=args.perm_sample_size,
        seed=args.seed,
    )

    run_root = cfg.base_dir / "output" / "ml_runs" / cfg.run_name
    manifest = load_manifest(run_root)

    feature_cols = list(manifest.get("feature_cols", []))
    if not feature_cols:
        raise RuntimeError("Manifest does not include feature_cols")
    if "feat_log10_snr" not in feature_cols:
        raise RuntimeError("feat_log10_snr not found in feature_cols")
    snr_idx = feature_cols.index("feat_log10_snr")

    n_test = int(manifest["n_test"])
    n_feat = int(manifest["n_features"])
    X_test_path = Path(manifest["X_test_path"])
    X_test_path = resolve_manifest_path(
        str(X_test_path),
        base_dir=cfg.base_dir,
        run_root=run_root,
        run_name=cfg.run_name,
        must_exist=True,
    )
    X_test = np.memmap(X_test_path, dtype="float32", mode="r", shape=(n_test, n_feat))

    plot_root_raw = str(manifest.get("plots_dir", str(cfg.base_dir / "plots" / "ml_runs" / cfg.run_name)))
    plot_root = resolve_manifest_path(
        plot_root_raw,
        base_dir=cfg.base_dir,
        run_root=run_root,
        run_name=cfg.run_name,
        must_exist=False,
    )
    out_dir = plot_root / cfg.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if "scaler_npz" in manifest:
        try:
            manifest["scaler_npz"] = str(
                resolve_manifest_path(
                    manifest["scaler_npz"],
                    base_dir=cfg.base_dir,
                    run_root=run_root,
                    run_name=cfg.run_name,
                    must_exist=True,
                )
            )
        except Exception:
            pass

    # For SNR-vs-feature charts, convert back to raw feature scale when scaler is available.
    idx_plot = choose_sample_idx(n_test, cfg.max_points_plot, cfg.seed)
    X_plot_scaled = np.asarray(X_test[idx_plot], dtype=np.float32)
    X_plot_raw = maybe_to_raw(X_plot_scaled, manifest, feature_cols)
    plot_snr_vs_features(
        out_dir=out_dir,
        X_raw=X_plot_raw,
        feature_cols=feature_cols,
        snr_idx=snr_idx,
        max_points_plot=cfg.max_points_plot,
        seed=cfg.seed,
    )

    mode, metric = compute_metric(run_root, manifest, X_test, cfg.batch_size)
    print("metric_mode:", mode)

    if metric is not None:
        # Use raw SNR for interpretability on metric plots.
        snr_scaled = np.asarray(X_test[:, snr_idx], dtype=np.float32)
        snr_full_raw = snr_scaled.copy()
        scaler_npz = manifest.get("scaler_npz")
        if scaler_npz and Path(scaler_npz).exists():
            try:
                y_min, y_iqr = load_scaler(Path(scaler_npz), feature_cols)
                snr_full_raw = snr_scaled * float(y_iqr[snr_idx]) + float(y_min[snr_idx])
            except Exception:
                pass

        label = "rmse_pix" if mode == "reconstruct_rmse" else "abs_error_log10_flux"
        plot_metric_vs_snr_bins(
            out_dir=out_dir,
            snr=snr_full_raw,
            metric=metric,
            metric_label=label,
            n_bins=cfg.snr_bins,
        )

        snr_permutation_test(
            out_dir=out_dir,
            mode=mode,
            run_root=run_root,
            manifest=manifest,
            X_test=X_test,
            feature_cols=feature_cols,
            snr_idx=snr_idx,
            baseline_metric=metric,
            perm_sample_size=cfg.perm_sample_size,
            perm_repeats=cfg.perm_repeats,
            seed=cfg.seed,
            batch_size=cfg.batch_size,
        )
    else:
        print("[WARN] Could not detect supported metric mode for this run; generated only SNR-vs-feature plots.")

    morph_csv = str(args.morph_csv).strip()
    if not morph_csv:
        default_morph = cfg.base_dir / "output" / "ml_runs" / "nn_psf_labels" / "labels_psf_weak.csv"
        if default_morph.exists():
            morph_csv = str(default_morph)

    if morph_csv:
        try:
            mj = load_morph_joined_test_table(
                base_dir=cfg.base_dir,
                run_root=run_root,
                run_name=cfg.run_name,
                manifest=manifest,
                n_test=n_test,
                morph_csv_raw=morph_csv,
            )
            morph_cols_present = [c for c in MORPH_COLS if c in mj.columns]
            ti = mj["test_index"].to_numpy(dtype=np.int64)
            if metric is not None:
                snr_src = snr_full_raw
            else:
                snr_src = np.asarray(X_test[:, snr_idx], dtype=np.float32)

            snr_m = np.asarray(snr_src[ti], dtype=np.float32)
            fm = mj[morph_cols_present].to_numpy(dtype=np.float32)

            plot_anchor_vs_features(
                out_dir=out_dir,
                anchor=snr_m,
                anchor_name="feat_log10_snr",
                feat_mat=fm,
                feat_names=morph_cols_present,
                csv_name="20_snr_vs_morph_spearman.csv",
                bar_png_name="20_snr_vs_morph_spearman_bar.png",
                hex_prefix="21_snr_vs_morph",
                max_points_plot=cfg.max_points_plot,
                seed=cfg.seed,
            )

            if metric is not None:
                met_m = np.asarray(metric[ti], dtype=np.float32)
                met_name = "rmse_pix" if mode == "reconstruct_rmse" else "abs_error_log10_flux"
                plot_anchor_vs_features(
                    out_dir=out_dir,
                    anchor=met_m,
                    anchor_name=met_name,
                    feat_mat=fm,
                    feat_names=morph_cols_present,
                    csv_name="22_metric_vs_morph_spearman.csv",
                    bar_png_name="22_metric_vs_morph_spearman_bar.png",
                    hex_prefix="23_metric_vs_morph",
                    max_points_plot=cfg.max_points_plot,
                    seed=cfg.seed + 11,
                )
        except Exception as exc:
            print(f"[WARN] Morphology diagnostics skipped: {exc}")

    print("DONE")
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
