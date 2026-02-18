"""
17_bis_patch_field_sigma_chi2nu.py
==================================

Goal (now aligned with Code 15 chi^2 definition):
-----------------------------------------------
Compute chi2 and chi2nu exactly like your Code 15 function:

  sigma_p^2 = sigma_bg^2 + alpha * max(I_true_p, 0) + sigma0^2

where:
- sigma_bg is PER-STAMP (one value per stamp), broadcast to pixels
- alpha is scalar (Poisson-like weight)
- sigma0 is scalar STD floor (added as sigma0^2)

We then compute:
  chi2   = sum_p [ (I_pred_p - I_true_p)^2 / sigma_p^2 ]
  chi2nu = chi2 / D

Also produce z-residuals in "sigma units":
  z_p = (I_pred_p - I_true_p) / sqrt(sigma_p^2)

Outputs (organized):
--------------------
plots/ml_runs/<RUN>/patch_17bis_field_sigma/
  tables/
    - chi2nu_field_values.csv
    - per_stamp_z_chi2_sigma.csv
    - per_pixel_z_hist_stats.csv
  histograms/
    - chi2nu_hist_log10.png
    - chi2nu_hist_linear_clipped.png
    - chi2nu_cdf_log10.png
  trends/
    - chi2nu_vs_logFtrue_binned_median.png
  stamps/
    - stamps_best6_raw_pred_z.png
    - stamps_worst6_raw_pred_z.png
    - stamps_random6_raw_pred_z.png
    - best_stamp_zonly.png
    - worst_stamp_zonly.png
    - random_stamp_zonly.png
  per_pixel/
    - per_pixel_z_hist_grid.png
  logs/
    - 17_run_settings.txt

Run examples:
-------------
python scripts/17_bis_patch_field_sigma_chi2nu.py --run ml_xgb_pixelsflux_8d_augtrain
python scripts/17_bis_patch_field_sigma_chi2nu.py --run ml_xgb_pixelsflux_8d_augtrain --flux_mode pred
python scripts/17_bis_patch_field_sigma_chi2nu.py --run ml_xgb_pixelsflux_8d_augtrain --sigma_bg_mode zero
python scripts/17_bis_patch_field_sigma_chi2nu.py --run ml_xgb_pixelsflux_8d_augtrain --alpha 1.0 --sigma0 1e-4

Notes about "clipped" plots:
----------------------------
Clipping is ONLY for readability in a *secondary* plot. The main log10 histogram and CDF are unclipped.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb


# =========================
# CONFIG
# =========================

@dataclass
class Cfg:
    base_dir: Path = Path(__file__).resolve().parents[1]  # .../Codes

    batch_size: int = 50_000

    # Default chi^2 params (match your earlier 15_bis style)
    sigma0_default: float = 1e-4   # sigma0 is STD floor (added as sigma0^2)
    alpha_default: float = 1.0

    # border MAD (for sigma_bg_mode=border_mad)
    border_w: int = 2
    sigma_floor: float = 1e-6

    # montages
    examples_n: int = 6
    random_seed: int = 123

    # chi2nu plotting
    clip_percentile: float = 99.5
    n_bins_binned_plot: int = 20

    # Z plotting
    z_clip: float = 25.0

    # per-pixel z hist grid
    z_hist_bins: int = 81          # odd number is nice (center bin at 0 if symmetric)
    z_hist_range: float = 10.0     # hist over [-z_hist_range, +z_hist_range] for the grid figure


CFG = Cfg()


# =========================
# Helpers
# =========================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def savefig(path: Path):
    ensure_dir(path.parent)
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()

def load_manifest(run_root: Path) -> Dict:
    mpath = run_root / "manifest_arrays.npz"
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    d = np.load(mpath, allow_pickle=True)
    out = {k: d[k] for k in d.files}

    for k in ["n_test", "n_features", "stamp_pix", "D"]:
        out[k] = int(out[k])

    for k in ["X_test_path", "Yshape_test_path", "Yflux_test_path", "plots_dir", "trace_test_csv"]:
        out[k] = str(out[k])

    if "feature_cols" in out:
        out["feature_cols"] = [str(x) for x in out["feature_cols"].tolist()]

    return out

def read_trace_fieldtags(trace_csv: Path, n_test: int) -> List[str]:
    field_tag = ["UNKNOWN"] * n_test
    if not trace_csv.exists():
        print("[WARN] trace_test.csv missing; all field tags will be UNKNOWN.")
        return field_tag

    with open(trace_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                i = int(row["test_index"])
            except Exception:
                continue
            if 0 <= i < n_test:
                field_tag[i] = row.get("field_tag", "UNKNOWN") or "UNKNOWN"
    return field_tag

def load_shape_models(model_dir: Path, D: int) -> List[xgb.Booster]:
    boosters = []
    for j in range(D):
        p = model_dir / f"booster_pix_{j:04d}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing pixel model: {p}")
        b = xgb.Booster()
        b.load_model(str(p))
        boosters.append(b)
    return boosters

def load_flux_model(model_dir: Path) -> xgb.Booster:
    p = model_dir / "booster_flux.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing flux model: {p}")
    b = xgb.Booster()
    b.load_model(str(p))
    return b

def sigma_from_border_mad(I_true_flat: np.ndarray, stamp_pix: int, w: int, floor: float) -> np.ndarray:
    """Per-stamp constant sigma_bg estimated from border pixels via MAD."""
    N = I_true_flat.shape[0]
    imgs = I_true_flat.reshape(N, stamp_pix, stamp_pix)

    m = np.zeros((stamp_pix, stamp_pix), dtype=bool)
    m[:w, :] = True
    m[-w:, :] = True
    m[:, :w] = True
    m[:, -w:] = True

    b = imgs[:, m]
    med = np.median(b, axis=1)
    mad = np.median(np.abs(b - med[:, None]), axis=1)
    sigma = 1.4826 * mad
    sigma = np.where(np.isfinite(sigma), sigma, floor)
    sigma = np.maximum(sigma, floor)
    return sigma.astype(np.float32)

def predict_flux_batch(booster_flux: xgb.Booster, Xb: np.ndarray) -> np.ndarray:
    return booster_flux.predict(
        xgb.DMatrix(np.asarray(Xb, dtype=np.float32, order="C"))
    ).astype(np.float32, copy=False)

def predict_shape_batch(boosters_shape: List[xgb.Booster], Xb: np.ndarray) -> np.ndarray:
    dmat = xgb.DMatrix(np.asarray(Xb, dtype=np.float32, order="C"))
    N = Xb.shape[0]
    D = len(boosters_shape)
    out = np.empty((N, D), dtype=np.float32)
    for j, booster in enumerate(boosters_shape):
        out[:, j] = booster.predict(dmat).astype(np.float32, copy=False)
    return out


# =========================
# chi2 + z (MATCH CODE 15)
# =========================

def chi2nu_sqrt_Itrue_like15(
    I_true: np.ndarray,
    I_pred: np.ndarray,
    sigma_bg: np.ndarray,
    sigma0: float,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    EXACTLY like your Code 15 definition:

      sigma_p^2 = sigma_bg^2 + alpha * max(I_true_p, 0) + sigma0^2

    Returns:
      (chi2, chi2nu, sig2) where sig2 is the pixelwise variance array (N,D).
    """
    I_true = np.asarray(I_true, dtype=np.float32)
    I_pred = np.asarray(I_pred, dtype=np.float32)
    resid = I_pred - I_true

    # sigma_bg is per-stamp -> broadcast
    sig2 = (sigma_bg.astype(np.float32) ** 2)[:, None]

    a = float(alpha)
    if a != 0.0:
        sig2 = sig2 + a * np.maximum(I_true, 0.0)

    sig2 = sig2 + (float(sigma0) ** 2)
    sig2 = np.maximum(sig2, 1e-30).astype(np.float32)

    chi2 = np.sum((resid * resid) / sig2, axis=1)
    nu = float(I_true.shape[1])  # D pixels
    chi2nu = chi2 / nu
    return chi2.astype(np.float32), chi2nu.astype(np.float32), sig2

def z_from_sig2(I_true: np.ndarray, I_pred: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """z = (pred-true)/sqrt(sig2)"""
    I_true = np.asarray(I_true, dtype=np.float32)
    I_pred = np.asarray(I_pred, dtype=np.float32)
    return (I_pred - I_true) / np.sqrt(sig2.astype(np.float32))


# =========================
# Plot helpers
# =========================

def stretch_limits_for_raw(r2d: np.ndarray, p2d: np.ndarray) -> Tuple[float, float]:
    v = np.concatenate([r2d.ravel(), p2d.ravel()])
    v = v[np.isfinite(v)]
    if v.size < 10:
        return float(np.nanmin(v)), float(np.nanmax(v))
    vmin, vmax = np.percentile(v, [5, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    return float(vmin), float(vmax)

def triplet_montage_raw_pred_z(out_png: Path, raw: np.ndarray, pred: np.ndarray, z: np.ndarray,
                              stamp_pix: int, title: str, z_clip: float):
    N = raw.shape[0]
    fig = plt.figure(figsize=(10.5, 3.0 * N))
    gs = fig.add_gridspec(N, 3, wspace=0.15, hspace=0.25)

    for i in range(N):
        r = raw[i].reshape(stamp_pix, stamp_pix)
        p = pred[i].reshape(stamp_pix, stamp_pix)
        zz = z[i].reshape(stamp_pix, stamp_pix)

        vmin, vmax = stretch_limits_for_raw(r, p)

        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        ax3 = fig.add_subplot(gs[i, 2])
        for ax in (ax1, ax2, ax3):
            ax.set_xticks([]); ax.set_yticks([])

        ax1.imshow(r, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax2.imshow(p, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax3.imshow(np.clip(zz, -z_clip, +z_clip), origin="lower",
                   cmap="RdBu_r", vmin=-z_clip, vmax=+z_clip)

        if i == 0:
            ax1.set_title("RAW true", fontsize=11)
            ax2.set_title("RAW pred", fontsize=11)
            ax3.set_title("Z = (pred-true)/sigma", fontsize=11)

    fig.suptitle(title, fontsize=14)
    savefig(out_png)

def plot_single_stamp_zonly(out_png: Path, z2d: np.ndarray, title: str, z_clip: float):
    plt.figure(figsize=(4.6, 4.2))
    im = plt.imshow(np.clip(z2d, -z_clip, +z_clip), origin="lower",
                    cmap="RdBu_r", vmin=-z_clip, vmax=+z_clip)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([]); plt.yticks([])
    plt.title(title, fontsize=11)
    savefig(out_png)


# =========================
# Per-pixel z histogram grid
# =========================

def update_per_pixel_hist(per_pix_hist: np.ndarray, z_vals: np.ndarray, vmin: float, vmax: float) -> None:
    D, nb = per_pix_hist.shape
    Z = np.asarray(z_vals, dtype=np.float32)
    ok = np.isfinite(Z)
    Z = np.where(ok, np.clip(Z, vmin, vmax), np.nan)

    t = (Z - vmin) / (vmax - vmin)
    bi = (t * nb).astype(np.int64)
    bi = np.clip(bi, 0, nb - 1)

    for j in range(D):
        m = np.isfinite(Z[:, j])
        if np.any(m):
            np.add.at(per_pix_hist[j], bi[m, j], 1)

def hist_percentile_from_counts(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    c = np.asarray(counts, dtype=np.int64)
    tot = int(np.sum(c))
    if tot <= 0:
        return float("nan")
    target = (q / 100.0) * tot
    cs = np.cumsum(c)
    k = int(np.searchsorted(cs, target, side="left"))
    k = min(max(k, 0), c.shape[0] - 1)
    return float(0.5 * (edges[k] + edges[k + 1]))

def plot_per_pixel_hist_grid(out_png: Path, per_pix_hist: np.ndarray, edges: np.ndarray, stamp_pix: int):
    D, nb = per_pix_hist.shape
    assert D == stamp_pix * stamp_pix
    xcent = 0.5 * (edges[:-1] + edges[1:])

    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(stamp_pix, stamp_pix, wspace=0.05, hspace=0.05)

    vmax = np.percentile(per_pix_hist.astype(np.float64).ravel(), 99.9)
    vmax = float(vmax if vmax > 1 else 1.0)

    for i in range(stamp_pix):
        for j in range(stamp_pix):
            pix = i * stamp_pix + j
            ax = fig.add_subplot(gs[i, j])
            ax.plot(xcent, per_pix_hist[pix], linewidth=0.6)
            ax.set_xlim(float(edges[0]), float(edges[-1]))
            ax.set_ylim(0, vmax)
            ax.axis("off")

    fig.suptitle("Per-pixel z histogram grid", fontsize=14)
    savefig(out_png)


# =========================
# MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run name under output/ml_runs/")
    ap.add_argument("--flux_mode", choices=["true", "pred"], default="true",
                    help="Reconstruction uses F_true (default) or F_pred")
    ap.add_argument("--sigma_bg_mode", choices=["border_mad", "zero"], default="border_mad",
                    help="How to define sigma_bg (per-stamp). 'border_mad' or 'zero'.")
    ap.add_argument("--sigma0", type=float, default=CFG.sigma0_default,
                    help="sigma0 STD floor (added as sigma0^2) [matches Code 15]")
    ap.add_argument("--alpha", type=float, default=CFG.alpha_default,
                    help="alpha multiplier of max(I_true,0) term [matches Code 15]")
    ap.add_argument("--no_pixel_hist_grid", action="store_true",
                    help="Disable the 20x20 per-pixel histogram grid (can be heavy).")
    args = ap.parse_args()

    rng = np.random.default_rng(CFG.random_seed)

    run_root = CFG.base_dir / "output" / "ml_runs" / args.run
    model_dir = run_root / "models"
    trace_dir = run_root / "trace"

    manifest = load_manifest(run_root)
    plot_dir_main = Path(manifest.get("plots_dir", str(CFG.base_dir / "plots" / "ml_runs" / args.run)))

    # Base output + subfolders
    base_out = ensure_dir(plot_dir_main / "patch_17bis_field_sigma")
    out_tables = ensure_dir(base_out / "tables")
    out_hist   = ensure_dir(base_out / "histograms")
    out_trends = ensure_dir(base_out / "trends")
    out_stamps = ensure_dir(base_out / "stamps")
    out_perpix = ensure_dir(base_out / "per_pixel")
    out_logs   = ensure_dir(base_out / "logs")

    # Save run settings
    with open(out_logs / "17_run_settings.txt", "w", encoding="utf-8") as f:
        f.write(f"run={args.run}\n")
        f.write(f"flux_mode={args.flux_mode}\n")
        f.write(f"sigma_bg_mode={args.sigma_bg_mode}\n")
        f.write(f"alpha={args.alpha}\n")
        f.write(f"sigma0={args.sigma0}\n")
        f.write(f"no_pixel_hist_grid={args.no_pixel_hist_grid}\n")

    n_test = manifest["n_test"]
    n_feat = manifest["n_features"]
    stamp_pix = manifest["stamp_pix"]
    D = manifest["D"]

    X_test = np.memmap(Path(manifest["X_test_path"]), dtype="float32", mode="r", shape=(n_test, n_feat))
    Yshape_test = np.memmap(Path(manifest["Yshape_test_path"]), dtype="float32", mode="r", shape=(n_test, D))
    Yflux_test = np.memmap(Path(manifest["Yflux_test_path"]), dtype="float32", mode="r", shape=(n_test,))

    trace_csv = Path(manifest.get("trace_test_csv", str(trace_dir / "trace_test.csv")))
    field_tags = read_trace_fieldtags(trace_csv, n_test=n_test)

    print("\n=== 17_bis DIAGNOSTICS (chi2 like Code 15) ===")
    print("RUN          :", args.run)
    print("OUT_BASE     :", base_out)
    print("n_test       :", n_test, "| stamp_pix:", stamp_pix, "| D:", D)
    print("flux_mode    :", args.flux_mode)
    print("sigma_bg_mode:", args.sigma_bg_mode)
    print("alpha        :", args.alpha)
    print("sigma0       :", args.sigma0)
    print("pixel_hist_grid:", (not args.no_pixel_hist_grid))

    print("\nLoading models...")
    boosters_shape = load_shape_models(model_dir, D=D)
    booster_flux = load_flux_model(model_dir)

    chi2 = np.empty(n_test, dtype=np.float32)
    chi2nu = np.empty(n_test, dtype=np.float32)
    sigma_bg_all = np.empty(n_test, dtype=np.float32)
    logF_pred_all = np.empty(n_test, dtype=np.float32)

    z_abs_mean = np.empty(n_test, dtype=np.float32)
    z_abs_p99 = np.empty(n_test, dtype=np.float32)
    z_abs_max = np.empty(n_test, dtype=np.float32)

    nb = int(CFG.z_hist_bins)
    zR = float(CFG.z_hist_range)
    z_edges = np.linspace(-zR, +zR, nb + 1, dtype=np.float32)
    per_pix_hist = np.zeros((D, nb), dtype=np.int64)

    for s in range(0, n_test, CFG.batch_size):
        e = min(n_test, s + CFG.batch_size)

        Xb = np.asarray(X_test[s:e], dtype=np.float32)
        S_true = np.asarray(Yshape_test[s:e], dtype=np.float32)
        logF_true = np.asarray(Yflux_test[s:e], dtype=np.float32)

        logF_pred = predict_flux_batch(booster_flux, Xb)
        S_pred = predict_shape_batch(boosters_shape, Xb)
        logF_pred_all[s:e] = logF_pred

        F_true = (10.0 ** logF_true).astype(np.float32)
        F_pred = (10.0 ** logF_pred).astype(np.float32)

        I_true = S_true * F_true[:, None]
        I_pred = (S_pred * F_true[:, None]) if args.flux_mode == "true" else (S_pred * F_pred[:, None])

        sigma_bg = sigma_from_border_mad(I_true, stamp_pix, CFG.border_w, CFG.sigma_floor) \
            if args.sigma_bg_mode == "border_mad" else np.zeros(e - s, dtype=np.float32)
        sigma_bg_all[s:e] = sigma_bg

        chi2_b, chi2nu_b, sig2 = chi2nu_sqrt_Itrue_like15(
            I_true=I_true, I_pred=I_pred,
            sigma_bg=sigma_bg, sigma0=args.sigma0, alpha=args.alpha
        )
        chi2[s:e] = chi2_b
        chi2nu[s:e] = chi2nu_b

        Z = z_from_sig2(I_true, I_pred, sig2)
        absZ = np.abs(Z)
        z_abs_mean[s:e] = np.mean(absZ, axis=1).astype(np.float32)
        z_abs_p99[s:e] = np.percentile(absZ, 99, axis=1).astype(np.float32)
        z_abs_max[s:e] = np.max(absZ, axis=1).astype(np.float32)

        if not args.no_pixel_hist_grid:
            update_per_pixel_hist(per_pix_hist, Z, vmin=-zR, vmax=+zR)

        if (e == n_test) or ((s // CFG.batch_size) % 5 == 0):
            print(f"Processed {e}/{n_test}")

    # -----------------------------
    # CSVs
    # -----------------------------
    out_csv = out_tables / "chi2nu_field_values.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "test_index", "field_tag",
                "chi2", "chi2nu_field",
                "sigma_bg", "alpha", "sigma0",
                "logF_true", "logF_pred", "dlogF",
                "flux_mode", "sigma_bg_mode",
                "z_abs_mean", "z_abs_p99", "z_abs_max",
            ]
        )
        w.writeheader()
        for i in range(n_test):
            lt = float(Yflux_test[i])
            lp = float(logF_pred_all[i])
            w.writerow({
                "test_index": int(i),
                "field_tag": field_tags[i],
                "chi2": float(chi2[i]),
                "chi2nu_field": float(chi2nu[i]),
                "sigma_bg": float(sigma_bg_all[i]),
                "alpha": float(args.alpha),
                "sigma0": float(args.sigma0),
                "logF_true": lt,
                "logF_pred": lp,
                "dlogF": float(lp - lt),
                "flux_mode": args.flux_mode,
                "sigma_bg_mode": args.sigma_bg_mode,
                "z_abs_mean": float(z_abs_mean[i]),
                "z_abs_p99": float(z_abs_p99[i]),
                "z_abs_max": float(z_abs_max[i]),
            })
    print("Saved:", out_csv)

    out_csv2 = out_tables / "per_stamp_z_chi2_sigma.csv"
    with open(out_csv2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "test_index",
                "chi2", "chi2nu",
                "sigma_bg", "alpha", "sigma0",
                "z_abs_mean", "z_abs_p99", "z_abs_max",
            ]
        )
        w.writeheader()
        for i in range(n_test):
            w.writerow({
                "test_index": int(i),
                "chi2": float(chi2[i]),
                "chi2nu": float(chi2nu[i]),
                "sigma_bg": float(sigma_bg_all[i]),
                "alpha": float(args.alpha),
                "sigma0": float(args.sigma0),
                "z_abs_mean": float(z_abs_mean[i]),
                "z_abs_p99": float(z_abs_p99[i]),
                "z_abs_max": float(z_abs_max[i]),
            })
    print("Saved:", out_csv2)

    # -----------------------------
    # Plots: chi2nu distributions
    # -----------------------------
    x = chi2nu[np.isfinite(chi2nu) & (chi2nu > 0)]
    if x.size == 0:
        raise RuntimeError("chi2nu has no finite positive values.")

    lx = np.log10(x)
    plt.figure(figsize=(7.8, 4.8))
    plt.hist(lx, bins=120)
    plt.xlabel("log10(chi2nu)")
    plt.ylabel("count")
    plt.title(
        "TEST reduced chi^2 histogram (log10)\n"
        f"flux_mode={args.flux_mode}, sigma_bg_mode={args.sigma_bg_mode}, alpha={args.alpha}, sigma0={args.sigma0:g}"
    )
    savefig(out_hist / "chi2nu_hist_log10.png")

    clip = np.percentile(x, CFG.clip_percentile)
    xc = np.clip(chi2nu, 0, clip)
    plt.figure(figsize=(7.8, 4.8))
    plt.hist(xc[np.isfinite(xc)], bins=120)
    plt.xlabel(f"chi2nu (clipped at p{CFG.clip_percentile}={clip:.3g})")
    plt.ylabel("count")
    plt.title("TEST reduced chi^2 histogram (linear, clipped)")
    savefig(out_hist / "chi2nu_hist_linear_clipped.png")

    lx_sorted = np.sort(lx)
    yy = np.linspace(0, 1, lx_sorted.size, endpoint=True)
    plt.figure(figsize=(7.8, 4.8))
    plt.plot(lx_sorted, yy, linewidth=1)
    plt.xlabel("log10(chi2nu)")
    plt.ylabel("CDF")
    plt.title("TEST reduced chi^2 CDF (log10)")
    savefig(out_hist / "chi2nu_cdf_log10.png")

    logF_true_all = np.asarray(Yflux_test, dtype=np.float32)
    m = np.isfinite(logF_true_all) & np.isfinite(chi2nu) & (chi2nu > 0)
    xf = logF_true_all[m]
    yf = chi2nu[m]

    bins = np.linspace(np.min(xf), np.max(xf), CFG.n_bins_binned_plot + 1)
    xmid = 0.5 * (bins[:-1] + bins[1:])
    med = np.full_like(xmid, np.nan, dtype=np.float32)
    p16 = np.full_like(xmid, np.nan, dtype=np.float32)
    p84 = np.full_like(xmid, np.nan, dtype=np.float32)

    for i in range(len(xmid)):
        sel = (xf >= bins[i]) & (xf < bins[i + 1])
        if np.sum(sel) < 30:
            continue
        v = yf[sel]
        v = v[np.isfinite(v)]
        if v.size < 30:
            continue
        med[i] = np.median(v)
        p16[i] = np.percentile(v, 16)
        p84[i] = np.percentile(v, 84)

    plt.figure(figsize=(7.4, 5.2))
    ok = np.isfinite(med)
    plt.plot(xmid[ok], med[ok], marker="o", linewidth=1)
    plt.fill_between(xmid[ok], p16[ok], p84[ok], alpha=0.25)
    plt.yscale("log")
    plt.xlabel("logF_true")
    plt.ylabel("median chi2nu (log scale)")
    plt.title("Reduced chi^2 vs logF_true (binned median ±[16,84])")
    savefig(out_trends / "chi2nu_vs_logFtrue_binned_median.png")

    # -----------------------------
    # Best / Worst / Random montages
    # -----------------------------
    order = np.argsort(chi2nu)
    n = int(CFG.examples_n)
    idx_best = order[:n]
    idx_worst = order[-n:][::-1]
    idx_rand = rng.choice(n_test, size=n, replace=False)

    def triplet_for_indices(idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xb = np.asarray(X_test[idxs], dtype=np.float32)
        S_true = np.asarray(Yshape_test[idxs], dtype=np.float32)
        logF_true = np.asarray(Yflux_test[idxs], dtype=np.float32)

        logF_pred = predict_flux_batch(booster_flux, Xb)
        S_pred = predict_shape_batch(boosters_shape, Xb)

        F_true = (10.0 ** logF_true).astype(np.float32)
        F_pred = (10.0 ** logF_pred).astype(np.float32)

        I_true = S_true * F_true[:, None]
        I_pred = (S_pred * F_true[:, None]) if args.flux_mode == "true" else (S_pred * F_pred[:, None])

        sigma_bg = sigma_from_border_mad(I_true, stamp_pix, CFG.border_w, CFG.sigma_floor) \
            if args.sigma_bg_mode == "border_mad" else np.zeros(I_true.shape[0], dtype=np.float32)

        _, _, sig2 = chi2nu_sqrt_Itrue_like15(I_true, I_pred, sigma_bg=sigma_bg, sigma0=args.sigma0, alpha=args.alpha)
        Z = z_from_sig2(I_true, I_pred, sig2)
        return I_true, I_pred, Z

    Itrue, Ipred, Z = triplet_for_indices(idx_best)
    triplet_montage_raw_pred_z(
        out_stamps / "stamps_best6_raw_pred_z.png",
        Itrue, Ipred, Z, stamp_pix,
        f"Best {n} (lowest chi2nu) — RAW/PRED/Z",
        z_clip=CFG.z_clip
    )
    plot_single_stamp_zonly(out_stamps / "best_stamp_zonly.png", Z[0].reshape(stamp_pix, stamp_pix),
                            title="BEST stamp: Z only", z_clip=CFG.z_clip)

    Itrue, Ipred, Z = triplet_for_indices(idx_worst)
    triplet_montage_raw_pred_z(
        out_stamps / "stamps_worst6_raw_pred_z.png",
        Itrue, Ipred, Z, stamp_pix,
        f"Worst {n} (highest chi2nu) — RAW/PRED/Z",
        z_clip=CFG.z_clip
    )
    plot_single_stamp_zonly(out_stamps / "worst_stamp_zonly.png", Z[0].reshape(stamp_pix, stamp_pix),
                            title="WORST stamp: Z only", z_clip=CFG.z_clip)

    Itrue, Ipred, Z = triplet_for_indices(idx_rand)
    triplet_montage_raw_pred_z(
        out_stamps / "stamps_random6_raw_pred_z.png",
        Itrue, Ipred, Z, stamp_pix,
        f"Random {n} — RAW/PRED/Z",
        z_clip=CFG.z_clip
    )
    plot_single_stamp_zonly(out_stamps / "random_stamp_zonly.png", Z[0].reshape(stamp_pix, stamp_pix),
                            title="RANDOM stamp: Z only", z_clip=CFG.z_clip)

    # -----------------------------
    # Per-pixel z hist grid + stats CSV
    # -----------------------------
    if not args.no_pixel_hist_grid:
        grid_png = out_perpix / "per_pixel_z_hist_grid.png"
        plot_per_pixel_hist_grid(grid_png, per_pix_hist, z_edges, stamp_pix=stamp_pix)
        print("Saved:", grid_png)

        stats_csv = out_tables / "per_pixel_z_hist_stats.csv"
        with open(stats_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["pix_index","i","j","count","p1","p5","p50","p95","p99","frac_|z|>5","frac_|z|>10"]
            )
            w.writeheader()
            centers = 0.5 * (z_edges[:-1] + z_edges[1:])
            for pix in range(D):
                c = int(np.sum(per_pix_hist[pix]))
                ii, jj = divmod(pix, stamp_pix)
                if c <= 0:
                    w.writerow({
                        "pix_index": pix, "i": ii, "j": jj, "count": 0,
                        "p1": float("nan"), "p5": float("nan"), "p50": float("nan"),
                        "p95": float("nan"), "p99": float("nan"),
                        "frac_|z|>5": float("nan"), "frac_|z|>10": float("nan"),
                    })
                    continue

                p1  = hist_percentile_from_counts(per_pix_hist[pix], z_edges, 1.0)
                p5  = hist_percentile_from_counts(per_pix_hist[pix], z_edges, 5.0)
                p50 = hist_percentile_from_counts(per_pix_hist[pix], z_edges, 50.0)
                p95 = hist_percentile_from_counts(per_pix_hist[pix], z_edges, 95.0)
                p99 = hist_percentile_from_counts(per_pix_hist[pix], z_edges, 99.0)

                gt5  = int(np.sum(per_pix_hist[pix][np.abs(centers) > 5.0]))
                gt10 = int(np.sum(per_pix_hist[pix][np.abs(centers) > 10.0]))

                w.writerow({
                    "pix_index": pix, "i": ii, "j": jj, "count": c,
                    "p1": p1, "p5": p5, "p50": p50, "p95": p95, "p99": p99,
                    "frac_|z|>5": float(gt5 / c),
                    "frac_|z|>10": float(gt10 / c),
                })
        print("Saved:", stats_csv)

    print("\nDONE.")
    print("Outputs in:", base_out)


if __name__ == "__main__":
    main()
