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
    return X_test, Yshape_test, Yflux_test, stamp_pix


def _safe_percentile(a: np.ndarray, q: float) -> float:
    v = a[np.isfinite(a)]
    if v.size == 0:
        return 0.0
    return float(np.percentile(v, q))


def extract_morph(stamp: np.ndarray) -> Dict[str, float]:
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
        "m_edge_flux_frac": edge_flux_frac,
    }


def save_montage(I: np.ndarray, idx: np.ndarray, title: str, out_png: Path, n_show: int) -> None:
    n = min(len(idx), int(n_show))
    if n == 0:
        return
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n:
            continue
        img = I[idx[i]]
        lo = np.nanpercentile(img, 1.0)
        hi = np.nanpercentile(img, 99.0)
        ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--science_products_dir", required=True, help="Output of script 55 containing umap_embedding.csv")
    ap.add_argument("--pred_run_dir", required=True, help="Output of script 53 containing test predictions")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--grid_n", type=int, default=10)
    ap.add_argument("--n_random_per_cell", type=int, default=10)
    ap.add_argument("--n_highprob_per_cell", type=int, default=10)
    ap.add_argument("--highprob_threshold", type=float, default=0.7)
    ap.add_argument("--top_cells_k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path("/data/yn316/Codes")
    run_root = base / "output" / "ml_runs" / str(args.run_name)
    sci_dir = Path(args.science_products_dir)
    pred_dir = Path(args.pred_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cell_montages").mkdir(parents=True, exist_ok=True)

    emb = pd.read_csv(sci_dir / "umap_embedding.csv")
    pred = pd.read_csv(pred_dir / "test_predictions.csv")
    trace = pd.read_csv(run_root / "trace" / "trace_test.csv", usecols=["test_index", "source_id"], low_memory=False).sort_values("test_index")
    df = emb.merge(pred[["test_index", "y_true", "p_xgb"]], on="test_index", how="inner")
    df = df.merge(trace, on="test_index", how="left")

    manifest = load_manifest(run_root)
    feature_cols = list(manifest["feature_cols"])
    idx_ruwe = feature_cols.index("feat_ruwe") if "feat_ruwe" in feature_cols else None
    idx_ipd = feature_cols.index("feat_ipd_frac_multi_peak") if "feat_ipd_frac_multi_peak" in feature_cols else None
    X_test, Yshape_test, Yflux_test, stamp_pix = open_test_arrays(manifest)
    X = np.asarray(X_test, dtype=np.float32)
    if idx_ruwe is not None:
        df["feat_ruwe"] = X[df["test_index"].to_numpy(dtype=int), idx_ruwe]
    else:
        df["feat_ruwe"] = np.nan
    if idx_ipd is not None:
        df["feat_ipd_frac_multi_peak"] = X[df["test_index"].to_numpy(dtype=int), idx_ipd]
    else:
        df["feat_ipd_frac_multi_peak"] = np.nan

    # magnitude lookup cache
    mag_cache = out_dir / "source_mag_cache.csv"
    if mag_cache.exists():
        mags = pd.read_csv(mag_cache)
    else:
        rows = []
        root = Path(str(manifest["dataset_root"]))
        for p in sorted(root.glob("*/metadata_16d.csv")):
            try:
                d = pd.read_csv(p, usecols=["source_id", "phot_g_mean_mag"], low_memory=False)
                rows.append(d)
            except Exception:
                continue
        mags = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["source_id", "phot_g_mean_mag"])
        mags["source_id"] = pd.to_numeric(mags["source_id"], errors="coerce")
        mags = mags.dropna(subset=["source_id"]).copy()
        mags["source_id"] = mags["source_id"].astype(np.int64)
        mags = mags.drop_duplicates(subset=["source_id"], keep="first")
        mags.to_csv(mag_cache, index=False)
    df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
    df = df.merge(mags, on="source_id", how="left")

    # true morphology metrics from stamps
    I_true = (np.asarray(Yshape_test, dtype=np.float32) * (10.0 ** np.asarray(Yflux_test, dtype=np.float32)[:, None])).reshape((-1, stamp_pix, stamp_pix))
    morph_rows = []
    for ti in df["test_index"].to_numpy(dtype=int):
        m = extract_morph(I_true[ti])
        morph_rows.append(m)
    dm = pd.DataFrame(morph_rows)
    df = pd.concat([df.reset_index(drop=True), dm.reset_index(drop=True)], axis=1)

    # grid assignment
    gx = np.clip(((df["umap1"] - df["umap1"].min()) / (df["umap1"].max() - df["umap1"].min() + 1e-12) * int(args.grid_n)).astype(int), 0, int(args.grid_n) - 1)
    gy = np.clip(((df["umap2"] - df["umap2"].min()) / (df["umap2"].max() - df["umap2"].min() + 1e-12) * int(args.grid_n)).astype(int), 0, int(args.grid_n) - 1)
    df["grid_x"] = gx
    df["grid_y"] = gy
    df["cell_id"] = df["grid_y"] * int(args.grid_n) + df["grid_x"]

    stats = (
        df.groupby(["cell_id", "grid_x", "grid_y"], as_index=False)
        .agg(
            n=("test_index", "size"),
            mean_p_multipeak=("p_xgb", "mean"),
            mean_ruwe=("feat_ruwe", "mean"),
            mean_ipd_frac_multi_peak=("feat_ipd_frac_multi_peak", "mean"),
            mean_phot_g_mean_mag=("phot_g_mean_mag", "mean"),
            mean_m_ellipticity=("m_ellipticity", "mean"),
            mean_m_concentration_r2_r6=("m_concentration_r2_r6", "mean"),
            mean_m_edge_flux_frac=("m_edge_flux_frac", "mean"),
            mean_m_asymmetry_180=("m_asymmetry_180", "mean"),
        )
    )
    stats.to_csv(out_dir / "umap_grid_cell_stats.csv", index=False)

    # heatmaps
    grid_n = int(args.grid_n)
    def _heat(col: str, title: str, out: Path) -> None:
        arr = np.full((grid_n, grid_n), np.nan, dtype=float)
        for _, r in stats.iterrows():
            arr[int(r["grid_y"]), int(r["grid_x"])] = float(r[col])
        plt.figure(figsize=(6.4, 5.5))
        plt.imshow(arr, origin="lower", cmap="viridis")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("grid_x")
        plt.ylabel("grid_y")
        plt.tight_layout()
        plt.savefig(out, dpi=170)
        plt.close()

    _heat("n", "UMAP Grid: density", out_dir / "heatmap_density.png")
    _heat("mean_p_multipeak", "UMAP Grid: mean p(multipeak)", out_dir / "heatmap_mean_prob.png")
    _heat("mean_phot_g_mean_mag", "UMAP Grid: mean phot_g_mean_mag", out_dir / "heatmap_mean_mag.png")

    # montages for interesting cells
    rng = np.random.default_rng(int(args.seed))
    interesting = stats[stats["n"] >= 20].sort_values("mean_p_multipeak", ascending=False).head(int(args.top_cells_k))
    meta_rows = []
    for _, r in interesting.iterrows():
        gx, gy = int(r["grid_x"]), int(r["grid_y"])
        sub = df[(df["grid_x"] == gx) & (df["grid_y"] == gy)].copy()
        idx = sub["test_index"].to_numpy(dtype=int)
        rng.shuffle(idx)
        ridx = idx[: int(args.n_random_per_cell)]
        hsub = sub[sub["p_xgb"] >= float(args.highprob_threshold)].copy()
        hidx = hsub.sort_values("p_xgb", ascending=False)["test_index"].to_numpy(dtype=int)[: int(args.n_highprob_per_cell)]
        save_montage(I_true, ridx, f"Cell ({gx},{gy}) random", out_dir / "cell_montages" / f"cell_{gy}_{gx}_random.png", int(args.n_random_per_cell))
        if len(hidx) > 0:
            save_montage(I_true, hidx, f"Cell ({gx},{gy}) high-prob", out_dir / "cell_montages" / f"cell_{gy}_{gx}_highprob.png", int(args.n_highprob_per_cell))
        meta_rows.append({"grid_x": gx, "grid_y": gy, "n": int(r["n"]), "mean_p_multipeak": float(r["mean_p_multipeak"]), "n_highprob": int(len(hidx))})
    pd.DataFrame(meta_rows).to_csv(out_dir / "interesting_cells_selected.csv", index=False)

    summary = {
        "run_name": str(args.run_name),
        "grid_n": int(args.grid_n),
        "highprob_threshold": float(args.highprob_threshold),
        "outputs": {
            "cell_stats_csv": str(out_dir / "umap_grid_cell_stats.csv"),
            "heatmap_density_png": str(out_dir / "heatmap_density.png"),
            "heatmap_mean_prob_png": str(out_dir / "heatmap_mean_prob.png"),
            "heatmap_mean_mag_png": str(out_dir / "heatmap_mean_mag.png"),
            "cell_montages_dir": str(out_dir / "cell_montages"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
