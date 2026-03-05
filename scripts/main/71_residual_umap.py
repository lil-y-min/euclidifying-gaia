#!/usr/bin/env python3
"""
Residual UMAP: embed (true_stamp - predicted_stamp) vectors for the test set.

Pipeline:
  1. Load X_test (16D Gaia features) and Yshape_test (true normalized stamps).
  2. Load all 400 pixel XGBoost boosters and predict → Ypred.
  3. Compute per-source residuals R = Ytrue - Ypred  (shape N x 400).
  4. Compute per-source reconstruction RMSE.
  5. Standardize R, run UMAP.
  6. Merge morphology + gmag from pixel embedding CSV (by source_id).
  7. Produce colored plots.

Outputs: report/model_decision/2026-03-05_umap_improvements/01_residual_umap/
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import umap

BASE = Path(__file__).resolve().parents[2]
ML_RUN = BASE / "output" / "ml_runs" / "base_v11_16d"
PIXEL_EMB_CSV = BASE / "output" / "experiments" / "embeddings" / "double_stars_pixels" / "email_pixels_all_filtered" / "umap_standard" / "embedding_umap.csv"
OUT_DIR = BASE / "report" / "model_decision" / "2026-03-05_umap_improvements" / "01_residual_umap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MORPH_PLOT_COLS = [
    "morph_asym_180",
    "morph_concentration_r80r20",
    "morph_ellipticity_e2",
    "morph_peakedness_kurtosis",
    "morph_roundness",
    "morph_gini",
    "morph_m20",
    "morph_smoothness",
    "morph_edge_asym_180",
    "morph_peak_to_total",
    "morph_texture_laplacian",
]

MORPH_LABELS = {
    "morph_asym_180": "Asymmetry 180",
    "morph_concentration_r80r20": "Concentration r80/r20",
    "morph_ellipticity_e2": "Ellipticity e2",
    "morph_peakedness_kurtosis": "Kurtosis",
    "morph_roundness": "Roundness",
    "morph_gini": "Gini",
    "morph_m20": "M20",
    "morph_smoothness": "Smoothness",
    "morph_edge_asym_180": "Edge Asymmetry",
    "morph_peak_to_total": "Peak/Total",
    "morph_texture_laplacian": "Laplacian Texture",
}

FEAT_LABELS = {
    "feat_log10_snr": "log10 SNR",
    "feat_ruwe": "RUWE",
    "feat_astrometric_excess_noise": "Astrometric Excess Noise",
    "feat_ipd_frac_multi_peak": "IPD Frac Multi-Peak",
    "feat_c_star": "C*",
    "feat_ipd_gof_harmonic_amplitude": "IPD GoF Harmonic Amp",
}


def savefig(path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def scatter_colored(x, y, c, title, cbar_label, out_png, cmap="viridis", s=4, alpha=0.7, clip_pct=(1, 99)):
    vals = np.asarray(c, dtype=float)
    finite = np.isfinite(vals)
    if finite.sum() > 10:
        lo, hi = np.nanpercentile(vals[finite], list(clip_pct))
        vals = np.clip(vals, lo, hi)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(x, y, c=vals, s=s, cmap=cmap, alpha=alpha, linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=10)
    ax.set_xlabel("Residual UMAP-1", fontsize=11)
    ax.set_ylabel("Residual UMAP-2", fontsize=11)
    ax.set_title(title, fontsize=12)
    savefig(out_png)


def plot_field_colors(x, y, field_tags, out_png):
    tags = sorted(pd.Series(field_tags).dropna().astype(str).unique().tolist())
    cmap = plt.cm.get_cmap("tab20", max(len(tags), 1))
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, tag in enumerate(tags):
        m = np.array(field_tags) == tag
        ax.scatter(x[m], y[m], s=4, color=cmap(i), alpha=0.6, linewidths=0, label=tag, rasterized=True)
    ax.set_xlabel("Residual UMAP-1", fontsize=11)
    ax.set_ylabel("Residual UMAP-2", fontsize=11)
    ax.set_title("Residual UMAP — colored by field", fontsize=12)
    if len(tags) <= 20:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=7)
    savefig(out_png)


def plot_morph_panels(df, x_col, y_col, morph_cols, out_png, suptitle):
    cols_present = [c for c in morph_cols if c in df.columns]
    if not cols_present:
        return
    n = len(cols_present)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).ravel()
    for i, col in enumerate(cols_present):
        ax = axes[i]
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() > 10:
            lo, hi = np.nanpercentile(vals[finite], [1, 99])
            vals = np.clip(vals, lo, hi)
        sc = ax.scatter(df[x_col], df[y_col], c=vals, s=3, cmap="viridis", alpha=0.65, linewidths=0, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(MORPH_LABELS.get(col, col), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(out_png)


def main():
    # ── 1. Load manifest and arrays ──────────────────────────────────────────
    print("[1/7] Loading arrays...")
    manifest = np.load(ML_RUN / "manifest_arrays.npz", allow_pickle=True)
    n_test   = int(manifest["n_test"])
    n_feat   = int(manifest["n_features"])
    D        = int(manifest["D"])

    X_test = np.memmap(ML_RUN / "arrays" / "X_test.mmap",  dtype="float32", mode="r", shape=(n_test, n_feat))
    Y_true = np.memmap(ML_RUN / "arrays" / "Yshape_test.mmap", dtype="float32", mode="r", shape=(n_test, D))
    X_test = np.array(X_test, dtype=np.float32)  # copy into RAM
    Y_true = np.array(Y_true, dtype=np.float32)

    trace = pd.read_csv(ML_RUN / "trace" / "trace_test.csv", low_memory=False)
    trace["source_id"] = pd.to_numeric(trace["source_id"], errors="coerce").astype(np.int64)
    print(f"  X_test: {X_test.shape}, Y_true: {Y_true.shape}, trace: {len(trace)}")

    # ── 2. Predict with all 400 pixel boosters ───────────────────────────────
    print("[2/7] Running inference on 400 pixel boosters...")
    t0 = time.time()
    dmat = xgb.DMatrix(X_test)
    Y_pred = np.zeros((n_test, D), dtype=np.float32)
    model_dir = ML_RUN / "models"
    for pix in range(D):
        b = xgb.Booster()
        b.load_model(model_dir / f"booster_pix_{pix:04d}.json")
        Y_pred[:, pix] = b.predict(dmat).astype(np.float32)
        if pix % 50 == 0:
            print(f"  pixel {pix}/{D}  elapsed={time.time()-t0:.0f}s")
    print(f"  Inference done in {time.time()-t0:.0f}s")

    # ── 3. Residuals and per-source RMSE ─────────────────────────────────────
    print("[3/7] Computing residuals...")
    R = Y_true - Y_pred                                      # (N, 400)
    rmse = np.sqrt(np.mean(R ** 2, axis=1))                  # per-source
    mae  = np.mean(np.abs(R), axis=1)
    # chi2-like: sum of squared residuals (no noise model, relative to true scale)
    chi2 = np.sum(R ** 2, axis=1)

    np.save(OUT_DIR / "residuals.npy", R)
    np.save(OUT_DIR / "Y_pred.npy", Y_pred)
    np.save(OUT_DIR / "rmse_per_source.npy", rmse)

    # ── 4. Standardize and run UMAP ──────────────────────────────────────────
    print("[4/7] Standardizing residuals...")
    Rs = StandardScaler().fit_transform(R.astype(np.float64)).astype(np.float32)

    print("[5/7] Fitting UMAP on residuals...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=123,
        init="spectral",
    )
    emb = reducer.fit_transform(Rs)
    print(f"  UMAP done in {time.time()-t0:.0f}s")

    # ── 5. Assemble result dataframe ─────────────────────────────────────────
    print("[6/7] Assembling result dataframe...")
    df = trace[["source_id", "field_tag"]].copy()
    df["x"] = emb[:, 0]
    df["y"] = emb[:, 1]
    df["rmse"]  = rmse
    df["mae"]   = mae
    df["chi2"]  = chi2

    # Merge morphology + gmag from pixel embedding CSV
    if PIXEL_EMB_CSV.exists():
        pix_emb = pd.read_csv(PIXEL_EMB_CSV, low_memory=False,
                               usecols=lambda c: c in {"source_id", "phot_g_mean_mag"} | set(MORPH_PLOT_COLS))
        pix_emb["source_id"] = pd.to_numeric(pix_emb["source_id"], errors="coerce")
        pix_emb = pix_emb.dropna(subset=["source_id"]).copy()
        pix_emb["source_id"] = pix_emb["source_id"].astype(np.int64)
        pix_emb = pix_emb.drop_duplicates(subset=["source_id"], keep="first")
        df = df.merge(pix_emb, on="source_id", how="left")
        print(f"  Merged morphology: {df[MORPH_PLOT_COLS[0]].notna().sum()}/{len(df)} rows have morph data")

    df.to_csv(OUT_DIR / "residual_umap_embedding.csv", index=False)

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    print("[7/7] Plotting...")
    x, y = df["x"].to_numpy(), df["y"].to_numpy()

    # Reconstruction error
    scatter_colored(x, y, rmse, "Residual UMAP — colored by per-source RMSE",
                    "RMSE (true − pred)", OUT_DIR / "01_rmse.png", cmap="hot_r")

    scatter_colored(x, y, chi2, "Residual UMAP — colored by Σ(residual²)",
                    "Σ(true − pred)²", OUT_DIR / "02_chi2.png", cmap="hot_r")

    # Field
    plot_field_colors(x, y, df["field_tag"].to_numpy(dtype=str), OUT_DIR / "03_field_colored.png")

    # G magnitude
    if "phot_g_mean_mag" in df.columns:
        scatter_colored(x, y, df["phot_g_mean_mag"], "Residual UMAP — colored by Gaia G mag",
                        "G mag", OUT_DIR / "04_gmag.png", cmap="magma_r")

    # Morphology panels
    plot_morph_panels(df, "x", "y", MORPH_PLOT_COLS, OUT_DIR / "05_morphology_panels.png",
                      "Residual UMAP — colored by morphology metrics")

    # RMSE distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rmse, bins=120, color="#2d6a4f", edgecolor="none")
    ax.set_xlabel("Per-source RMSE (true − predicted stamp)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of reconstruction RMSE on test set", fontsize=12)
    ax.axvline(np.median(rmse), color="#d62828", lw=1.5, label=f"median={np.median(rmse):.4f}")
    ax.axvline(np.percentile(rmse, 90), color="#f77f00", lw=1.5, ls="--", label=f"p90={np.percentile(rmse, 90):.4f}")
    ax.legend()
    savefig(OUT_DIR / "06_rmse_distribution.png")

    # Save summary
    summary = {
        "n_test": int(n_test),
        "D": int(D),
        "rmse_median": float(np.median(rmse)),
        "rmse_p90": float(np.percentile(rmse, 90)),
        "rmse_p99": float(np.percentile(rmse, 99)),
        "chi2_median": float(np.median(chi2)),
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "scaling": "StandardScaler on residuals",
        "ml_run": str(ML_RUN),
        "pixel_emb_csv": str(PIXEL_EMB_CSV),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== DONE ===")
    print("Out dir:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("*.png")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
