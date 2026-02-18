#!/usr/bin/env python
"""
20_chi2_2d_hist.py
==================

Produces (organized output):

(1) Column-normalized 2D histogram of chi2nu vs Gaia G:
    x = Gaia G magnitude
    y = log10(chi2nu)
    Each x-bin normalized so sum_y P(y | x-bin) = 1.

(2) Column-normalized 2D histogram of mean(z^2) vs Gaia G:
    Uses z_resid_per_stamp_summary.csv from 17_bis:
      mean(z^2) = z_std^2 + z_mean^2
    x = Gaia G magnitude
    y = log10(mean(z^2))

(3) Optional: histogram of per-pixel chi2 contributions for one stamp:
      chi2_pix = (I_pred - I_true)^2 / sig2
    for a chosen test_index.

Reads:
  - plots/ml_runs/<RUN>/patch_17bis_field_sigma/tables/chi2nu_field_values.csv  (fallback: patch root)
  - plots/ml_runs/<RUN>/patch_17bis_field_sigma/tables/z_resid_per_stamp_summary.csv (fallback: patch root)
  - output/ml_runs/<RUN>/trace/trace_test.csv
  - output/dataset_npz/<field_tag>/metadata.csv
  - output/ml_runs/<RUN>/manifest_arrays.npz
  - output/ml_runs/<RUN>/models/booster_pix_*.json + booster_flux.json
  - memmaps listed in manifest_arrays.npz

Outputs:
  plots/ml_runs/<RUN>/patch_17bis_field_sigma/analytics_20/
    2dhist/
      - chi2nu_vs_G_colnorm_2dhist_log10.(png|pdf)
      - z2_vs_G_colnorm_2dhist_log10.(png|pdf)
    stamp_chi2pix/
      - stamp_<idx>_chi2pix_hist.(png|pdf)   (if --stamp_index provided)
    logs/
      - 20_run_settings.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Robust Codes/ discovery
# -------------------------

def find_codes_dir(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "plots").exists() and (cur / "output").exists():
            return cur
        if cur.name.lower() == "codes" and (cur / "output").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(
        f"Could not auto-detect Codes/ starting from {start}. "
        f"Pass --codes_dir /path/to/Codes."
    )


# -------------------------
# Small helpers
# -------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def pick_first_existing(cols: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None

def finite_percentile_edges(x: np.ndarray, lo: float, hi: float, nbins: int) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise RuntimeError("No finite values for bin edges.")
    a = float(np.percentile(x, lo))
    b = float(np.percentile(x, hi))
    if (not np.isfinite(a)) or (not np.isfinite(b)) or (b <= a):
        a = float(np.nanmin(x))
        b = float(np.nanmax(x))
    if b <= a:
        b = a + 1.0
    return np.linspace(a, b, nbins + 1)

def column_normalize(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    raw_colsum = H.sum(axis=1)  # (nx,)
    Hn = np.zeros_like(H, dtype=float)
    nz = raw_colsum > 0
    Hn[nz] = H[nz] / raw_colsum[nz, None]
    return Hn, raw_colsum

def save_both(fig: plt.Figure, png: Path, pdf: Path, dpi: int):
    ensure_dir(png.parent)
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    print("Saved:", png)
    print("Saved:", pdf)

def resolve_csv(patch_dir: Path, name: str) -> Path:
    """
    Prefer patch_dir/tables/<name>, fallback to patch_dir/<name>
    """
    p1 = patch_dir / "tables" / name
    if p1.exists():
        return p1
    p2 = patch_dir / name
    return p2


# -------------------------
# CSV column detection
# -------------------------

def detect_columns_in_chi_csv(chi_csv: Path) -> Tuple[str, str]:
    head = pd.read_csv(chi_csv, nrows=0)
    cols = list(head.columns)

    field_col = pick_first_existing(cols, ("field_tag", "field", "fieldname"))
    if field_col is None:
        raise KeyError(f"Could not find field column in {chi_csv}. Columns are: {cols}")

    chi_col = pick_first_existing(cols, ("chi2nu", "chi2nu_field", "chi2nu_test", "chi2nu_val"))
    if chi_col is None:
        raise KeyError(f"Could not find chi2nu column in {chi_csv}. Columns are: {cols}")

    if "test_index" not in cols:
        raise KeyError(f"Missing required 'test_index' column in {chi_csv}. Columns are: {cols}")

    return field_col, chi_col


# -------------------------
# Trace map
# -------------------------

def read_trace_map(trace_csv: Path) -> pd.DataFrame:
    if not trace_csv.exists():
        raise FileNotFoundError(f"trace_test.csv not found: {trace_csv}")
    df = pd.read_csv(trace_csv, usecols=["test_index", "field_tag", "meta_row"])
    df["test_index"] = pd.to_numeric(df["test_index"], errors="coerce").astype("Int64")
    df["meta_row"] = pd.to_numeric(df["meta_row"], errors="coerce").astype("Int64")
    df["field_tag"] = df["field_tag"].astype(str)
    df = df.dropna(subset=["test_index", "meta_row"])
    df["test_index"] = df["test_index"].astype(int)
    df["meta_row"] = df["meta_row"].astype(int)
    return df


# -------------------------
# metadata.csv -> G mapping
# -------------------------

def build_meta_row_to_G(
    metadata_csv: Path,
    wanted_rows: Set[int],
    g_col_candidates: Tuple[str, ...] = ("phot_g_mean_mag", "feat_phot_g_mean_mag"),
    chunksize: int = 200_000,
) -> Dict[int, float]:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")

    head = pd.read_csv(metadata_csv, nrows=0, engine="python")
    g_col = pick_first_existing(list(head.columns), g_col_candidates)
    if g_col is None:
        raise KeyError(f"No G column found in {metadata_csv}. Tried {g_col_candidates}.")

    out: Dict[int, float] = {}
    offset = 0

    wanted_arr = np.fromiter(wanted_rows, dtype=np.int64)
    wanted_arr.sort()

    for chunk in pd.read_csv(
        metadata_csv,
        usecols=[g_col],
        chunksize=chunksize,
        engine="python",
        on_bad_lines="skip",
    ):
        n = len(chunk)
        if n == 0:
            continue

        meta_rows = np.arange(offset, offset + n, dtype=np.int64)
        offset += n

        # SAFE selection (avoid index tricks that caused your earlier IndexError)
        hit = np.isin(meta_rows, wanted_arr, assume_unique=False)
        if not np.any(hit):
            continue

        gvals = pd.to_numeric(chunk[g_col], errors="coerce").to_numpy(dtype=float)
        sel_rows = meta_rows[hit]
        sel_g = gvals[hit]

        for r, gv in zip(sel_rows.tolist(), sel_g.tolist()):
            if np.isfinite(gv):
                out[int(r)] = float(gv)

        if len(out) >= len(wanted_rows):
            break

    return out


def attach_G_for_test_indices(
    *,
    dataset_root: Path,
    merged: pd.DataFrame,
) -> np.ndarray:
    """
    merged must contain columns: field_tag, meta_row
    Returns array G aligned to merged rows (NaN if missing).
    """
    g_all = np.full(len(merged), np.nan, dtype=float)

    for field, sub_idx in merged.groupby("field_tag").indices.items():
        meta_rows = set(int(x) for x in merged.loc[sub_idx, "meta_row"].to_numpy())
        meta_csv = dataset_root / field / "metadata.csv"
        print(f"\nField {field}: need {len(meta_rows)} meta_rows | metadata: {meta_csv}")
        row_to_g = build_meta_row_to_G(meta_csv, meta_rows)

        mr = merged.loc[sub_idx, "meta_row"].to_numpy(dtype=int)
        gvals = np.array([row_to_g.get(int(r), np.nan) for r in mr], dtype=float)
        g_all[sub_idx] = gvals

    return g_all


# -------------------------
# Plot: column-normalized 2D hist + median + optional count curve
# -------------------------

def plot_colnorm_2dhist(
    *,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
    nbin_x: int,
    nbin_y: int,
    x_prc_lo: float,
    x_prc_hi: float,
    y_prc_lo: float,
    y_prc_hi: float,
    show_counts: bool,
    debug_norm: bool,
    norm_tol: float,
    min_per_bin_median: int = 50,
):
    x_edges = finite_percentile_edges(x, x_prc_lo, x_prc_hi, nbin_x)
    y_edges = finite_percentile_edges(y, y_prc_lo, y_prc_hi, nbin_y)

    H, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges])
    Hn, raw_colsum = column_normalize(H)

    if debug_norm:
        sums = Hn.sum(axis=1)
        nonempty = raw_colsum > 0
        dif = np.abs(sums[nonempty] - 1.0)

        print("\n=== DEBUG: column-normalization ===")
        print(f"Non-empty columns: {int(np.sum(nonempty))}/{len(raw_colsum)}")
        print(f"Column-sum stats: min={float(np.min(sums[nonempty])):.6f} "
              f"median={float(np.median(sums[nonempty])):.6f} max={float(np.max(sums[nonempty])):.6f}")
        print(f"|sum-1| max: {float(np.max(dif)):.6g}  (tol={norm_tol})")

        worst_i = np.where(nonempty)[0][int(np.argmax(dif))]
        print(f"Worst column: i={worst_i}  {x_label}=[{xedges[worst_i]:.2f},{xedges[worst_i+1]:.2f}] "
              f"sum={sums[worst_i]:.6f} raw={raw_colsum[worst_i]:.0f}")

        if float(np.max(dif)) <= norm_tol:
            print("[OK] Columns sum to 1 within tolerance.")
        else:
            print("[WARN] Some columns deviate from 1 beyond tolerance.")

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)

    im = ax.imshow(
        Hn.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Density (column-normalized)")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # median curve (left axis)
    xmid = 0.5 * (xedges[:-1] + xedges[1:])
    med = np.full_like(xmid, np.nan, dtype=float)
    for i in range(xmid.size):
        sel = (x >= xedges[i]) & (x < xedges[i + 1])
        if np.sum(sel) < min_per_bin_median:
            continue
        med[i] = np.median(y[sel])

    ok = np.isfinite(med)
    handles = []
    labels = []
    if np.any(ok):
        h_med, = ax.plot(xmid[ok], med[ok], linewidth=2)
        handles.append(h_med); labels.append("median")

    # counts per x-bin (right axis)
    if show_counts:
        counts = H.sum(axis=1)
        ax2 = ax.twinx()
        h_cnt, = ax2.plot(xmid, counts, linewidth=1)
        ax2.set_ylabel("Count (sources per G-bin)")
        ax2.tick_params(axis="y", pad=3)
        handles.append(h_cnt); labels.append("count per G bin")

    if handles:
        ax.legend(handles, labels, loc="upper right", frameon=True)

    save_both(fig, out_png, out_pdf, dpi=dpi)


# -------------------------
# One-stamp per-pixel chi2 histogram
# -------------------------

def load_manifest(run_root: Path) -> Dict:
    mpath = run_root / "manifest_arrays.npz"
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")
    d = np.load(mpath, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    for k in ["n_test", "n_features", "stamp_pix", "D"]:
        out[k] = int(out[k])
    for k in ["X_test_path", "Yshape_test_path", "Yflux_test_path"]:
        out[k] = str(out[k])
    return out

def sigma_bg_from_border_mad(I_true_flat: np.ndarray, stamp_pix: int, w: int, floor: float) -> float:
    img = I_true_flat.reshape(stamp_pix, stamp_pix)
    m = np.zeros((stamp_pix, stamp_pix), dtype=bool)
    m[:w, :] = True
    m[-w:, :] = True
    m[:, :w] = True
    m[:, -w:] = True
    b = img[m]
    med = np.median(b)
    mad = np.median(np.abs(b - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma):
        sigma = floor
    return float(max(sigma, floor))

def per_pixel_sig2(I_true: np.ndarray, sigma_bg: float, sigma0: float, alpha: float) -> np.ndarray:
    sig2 = (sigma_bg ** 2) + (float(sigma0) ** 2) + float(alpha) * np.maximum(I_true, 0.0)
    return np.maximum(sig2.astype(np.float32), 1e-30)

def plot_one_stamp_chi2pix_hist(
    *,
    run_root: Path,
    out_dir: Path,
    stamp_index: int,
    flux_mode: str,
    alpha: float,
    sigma0: float,
    sigma_bg_mode: str,
    border_w: int,
    sigma_floor: float,
    dpi: int,
):
    import xgboost as xgb

    manifest = load_manifest(run_root)
    n_test = manifest["n_test"]
    n_feat = manifest["n_features"]
    stamp_pix = manifest["stamp_pix"]
    D = manifest["D"]

    if not (0 <= stamp_index < n_test):
        raise ValueError(f"--stamp_index {stamp_index} out of range [0, {n_test-1}]")

    model_dir = run_root / "models"
    booster_flux_path = model_dir / "booster_flux.json"
    if not booster_flux_path.exists():
        raise FileNotFoundError(f"Missing flux model: {booster_flux_path}")

    boosters_shape = []
    for j in range(D):
        p = model_dir / f"booster_pix_{j:04d}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing pixel model: {p}")
        b = xgb.Booster()
        b.load_model(str(p))
        boosters_shape.append(b)

    booster_flux = xgb.Booster()
    booster_flux.load_model(str(booster_flux_path))

    X_test = np.memmap(Path(manifest["X_test_path"]), dtype="float32", mode="r", shape=(n_test, n_feat))
    Yshape_test = np.memmap(Path(manifest["Yshape_test_path"]), dtype="float32", mode="r", shape=(n_test, D))
    Yflux_test = np.memmap(Path(manifest["Yflux_test_path"]), dtype="float32", mode="r", shape=(n_test,))

    Xb = np.asarray(X_test[stamp_index:stamp_index+1], dtype=np.float32, order="C")
    S_true = np.asarray(Yshape_test[stamp_index], dtype=np.float32)
    logF_true = float(np.asarray(Yflux_test[stamp_index], dtype=np.float32))

    dmat = xgb.DMatrix(Xb)

    logF_pred = float(booster_flux.predict(dmat)[0])
    S_pred = np.empty((D,), dtype=np.float32)
    for j, booster in enumerate(boosters_shape):
        S_pred[j] = float(booster.predict(dmat)[0])

    F_true = float(10.0 ** logF_true)
    F_pred = float(10.0 ** logF_pred)

    I_true = S_true * F_true
    I_pred = (S_pred * F_true) if flux_mode == "true" else (S_pred * F_pred)

    sigma_bg = sigma_bg_from_border_mad(I_true, stamp_pix=stamp_pix, w=border_w, floor=sigma_floor) \
        if sigma_bg_mode == "border_mad" else 0.0

    sig2 = per_pixel_sig2(I_true, sigma_bg=sigma_bg, sigma0=sigma0, alpha=alpha)
    resid = I_pred - I_true
    chi2_pix = (resid * resid) / sig2

    chi2_stamp = float(np.sum(chi2_pix))
    chi2nu_stamp = chi2_stamp / float(D)

    v = chi2_pix[np.isfinite(chi2_pix)]
    v = v[v >= 0]
    if v.size == 0:
        raise RuntimeError("No finite per-pixel chi2 values for this stamp.")

    lv = np.log10(np.maximum(v, 1e-30))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.hist(lv, bins=80)
    ax.set_xlabel("log10(chi2 contribution per pixel)")
    ax.set_ylabel("Count (pixels)")
    ax.set_title("Per-pixel chi2 contributions (one stamp)")

    txt = (
        f"test_index={stamp_index}\n"
        f"chi2={chi2_stamp:.3g}  chi2nu={chi2nu_stamp:.3g}\n"
        f"flux_mode={flux_mode}  alpha={alpha:g}  sigma0={sigma0:g}\n"
        f"sigma_bg={sigma_bg:.3g}  logF_true={logF_true:.3f}  logF_pred={logF_pred:.3f}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9)

    out_png = out_dir / f"stamp_{stamp_index:06d}_chi2pix_hist.png"
    out_pdf = out_dir / f"stamp_{stamp_index:06d}_chi2pix_hist.pdf"
    save_both(fig, out_png, out_pdf, dpi=dpi)


# -------------------------
# MAIN
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--run", required=True)
    ap.add_argument("--codes_dir", default=None)
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--patch_dirname", default="patch_17bis_field_sigma")

    # 2D hist settings
    ap.add_argument("--nbin_g", type=int, default=40)
    ap.add_argument("--nbin_y", type=int, default=90)
    ap.add_argument("--g_prc_lo", type=float, default=0.5)
    ap.add_argument("--g_prc_hi", type=float, default=99.5)
    ap.add_argument("--y_prc_lo", type=float, default=0.5)
    ap.add_argument("--y_prc_hi", type=float, default=99.5)

    ap.add_argument("--show_counts", action="store_true")
    ap.add_argument("--debug_norm", action="store_true")
    ap.add_argument("--norm_tol", type=float, default=5e-3)

    ap.add_argument("--dpi", type=int, default=180)

    # z2 plot control
    ap.add_argument("--no_z2_plot", action="store_true")
    ap.add_argument("--z_summary_csvname", default="z_resid_per_stamp_summary.csv")

    # One-stamp plot controls
    ap.add_argument("--stamp_index", type=int, default=None)
    ap.add_argument("--flux_mode", choices=["true", "pred"], default="true")

    # Noise model parameters for one-stamp reconstruction (match 17/15 defaults)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--sigma0", type=float, default=1e-4)
    ap.add_argument("--sigma_bg_mode", choices=["border_mad", "const0"], default="border_mad")
    ap.add_argument("--border_w", type=int, default=2)
    ap.add_argument("--sigma_floor", type=float, default=1e-8)

    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    codes_dir = Path(args.codes_dir).resolve() if args.codes_dir else find_codes_dir(script_dir)
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else (codes_dir / "output" / "dataset_npz")

    plots_run_dir = codes_dir / "plots" / "ml_runs" / args.run
    patch_dir = plots_run_dir / args.patch_dirname

    # read CSVs (prefer tables/)
    chi_csv = resolve_csv(patch_dir, "chi2nu_field_values.csv")
    zsum_csv = resolve_csv(patch_dir, args.z_summary_csvname)

    trace_csv = codes_dir / "output" / "ml_runs" / args.run / "trace" / "trace_test.csv"
    run_root = codes_dir / "output" / "ml_runs" / args.run

    # organized output for Code 20
    out_base = ensure_dir(patch_dir / "analytics_20")
    out_2dhist = ensure_dir(out_base / "2dhist")
    out_stamp = ensure_dir(out_base / "stamp_chi2pix")
    out_logs = ensure_dir(out_base / "logs")

    with open(out_logs / "20_run_settings.txt", "w", encoding="utf-8") as f:
        f.write(f"run={args.run}\n")
        f.write(f"codes_dir={codes_dir}\n")
        f.write(f"dataset_root={dataset_root}\n")
        f.write(f"patch_dir={patch_dir}\n")
        f.write(f"chi_csv={chi_csv}\n")
        f.write(f"zsum_csv={zsum_csv}\n")
        f.write(f"trace_csv={trace_csv}\n")
        f.write(f"nbin_g={args.nbin_g} nbin_y={args.nbin_y}\n")
        f.write(f"show_counts={args.show_counts} debug_norm={args.debug_norm}\n")

    print("\n=== BUILD chi2nu/z2 vs G (column-normalized 2D hists) ===")
    print("RUN        :", args.run)
    print("CODES_DIR  :", codes_dir)
    print("DATASETROOT:", dataset_root)
    print("PATCH_DIR  :", patch_dir)
    print("CHI CSV    :", chi_csv)
    print("ZSUM CSV   :", zsum_csv)
    print("TRACE CSV  :", trace_csv)
    print("OUT_BASE   :", out_base)

    if not chi_csv.exists():
        raise FileNotFoundError(f"Missing: {chi_csv} (run 17_bis first)")

    # -----------------------
    # 1) chi2nu vs G
    # -----------------------
    field_col, chi_col = detect_columns_in_chi_csv(chi_csv)
    print("Detected in chi CSV:", f"field_col={field_col}", f"chi_col={chi_col}")

    chi_df = pd.read_csv(chi_csv, usecols=["test_index", field_col, chi_col])
    chi_df = chi_df.rename(columns={field_col: "field_tag", chi_col: "chi2nu"})
    chi_df["test_index"] = pd.to_numeric(chi_df["test_index"], errors="coerce")
    chi_df["chi2nu"] = pd.to_numeric(chi_df["chi2nu"], errors="coerce")
    chi_df["field_tag"] = chi_df["field_tag"].astype(str)
    chi_df = chi_df.dropna(subset=["test_index", "chi2nu"])
    chi_df["test_index"] = chi_df["test_index"].astype(int)

    tr = read_trace_map(trace_csv)
    merged = chi_df.merge(tr, on="test_index", how="inner", suffixes=("", "_trace"))
    if merged.empty:
        raise RuntimeError("Join chi2nu_field_values.csv × trace_test.csv gave 0 rows.")

    if "field_tag_trace" in merged.columns:
        merged["field_tag"] = merged["field_tag_trace"]
        merged = merged.drop(columns=["field_tag_trace"], errors="ignore")

    merged["G"] = attach_G_for_test_indices(dataset_root=dataset_root, merged=merged)

    g = merged["G"].to_numpy(dtype=float)
    chi2nu = merged["chi2nu"].to_numpy(dtype=float)
    m = np.isfinite(g) & np.isfinite(chi2nu) & (chi2nu > 0)
    g1 = g[m]
    y1 = np.log10(chi2nu[m])

    out_png = out_2dhist / "chi2nu_vs_G_colnorm_2dhist_log10.png"
    out_pdf = out_2dhist / "chi2nu_vs_G_colnorm_2dhist_log10.pdf"

    plot_colnorm_2dhist(
        x=g1,
        y=y1,
        x_label="Gaia G",
        y_label="log10(chi2nu)",
        title="chi2nu vs Gaia G (column-normalized)",
        out_png=out_png,
        out_pdf=out_pdf,
        dpi=args.dpi,
        nbin_x=args.nbin_g,
        nbin_y=args.nbin_y,
        x_prc_lo=args.g_prc_lo,
        x_prc_hi=args.g_prc_hi,
        y_prc_lo=args.y_prc_lo,
        y_prc_hi=args.y_prc_hi,
        show_counts=args.show_counts,
        debug_norm=args.debug_norm,
        norm_tol=args.norm_tol,
        min_per_bin_median=50,
    )
    print("\nUsed rows (chi2nu plot):", int(g1.size))

    # -----------------------
    # 2) mean(z^2) vs G
    # -----------------------
    if not args.no_z2_plot:
        if not zsum_csv.exists():
            print(f"[WARN] Missing {zsum_csv}; skipping z2 plot.")
        else:
            zdf = pd.read_csv(zsum_csv, usecols=["test_index", "z_mean", "z_std"])
            zdf["test_index"] = pd.to_numeric(zdf["test_index"], errors="coerce")
            zdf["z_mean"] = pd.to_numeric(zdf["z_mean"], errors="coerce")
            zdf["z_std"] = pd.to_numeric(zdf["z_std"], errors="coerce")
            zdf = zdf.dropna(subset=["test_index", "z_mean", "z_std"])
            zdf["test_index"] = zdf["test_index"].astype(int)

            zmerged = zdf.merge(tr, on="test_index", how="inner")
            if zmerged.empty:
                raise RuntimeError("Join z_resid_per_stamp_summary.csv × trace_test.csv gave 0 rows.")

            zmerged["G"] = attach_G_for_test_indices(dataset_root=dataset_root, merged=zmerged)

            z_mean = zmerged["z_mean"].to_numpy(dtype=float)
            z_std = zmerged["z_std"].to_numpy(dtype=float)
            z2 = (z_std * z_std) + (z_mean * z_mean)

            g2 = zmerged["G"].to_numpy(dtype=float)
            m2 = np.isfinite(g2) & np.isfinite(z2) & (z2 > 0)
            g2 = g2[m2]
            y2 = np.log10(z2[m2])

            out_png2 = out_2dhist / "z2_vs_G_colnorm_2dhist_log10.png"
            out_pdf2 = out_2dhist / "z2_vs_G_colnorm_2dhist_log10.pdf"

            plot_colnorm_2dhist(
                x=g2,
                y=y2,
                x_label="Gaia G",
                y_label="log10(mean(z^2))",
                title="mean(z^2) vs Gaia G (column-normalized)",
                out_png=out_png2,
                out_pdf=out_pdf2,
                dpi=args.dpi,
                nbin_x=args.nbin_g,
                nbin_y=args.nbin_y,
                x_prc_lo=args.g_prc_lo,
                x_prc_hi=args.g_prc_hi,
                y_prc_lo=args.y_prc_lo,
                y_prc_hi=args.y_prc_hi,
                show_counts=args.show_counts,
                debug_norm=args.debug_norm,
                norm_tol=args.norm_tol,
                min_per_bin_median=50,
            )
            print("Used rows (z2 plot):", int(g2.size))

    # -----------------------
    # 3) One-stamp: chi2_pixel vs count (hist)
    # -----------------------
    if args.stamp_index is not None:
        print("\n=== ONE STAMP chi2_pixel histogram ===")
        print("stamp_index :", args.stamp_index)
        plot_one_stamp_chi2pix_hist(
            run_root=run_root,
            out_dir=out_stamp,
            stamp_index=int(args.stamp_index),
            flux_mode=args.flux_mode,
            alpha=float(args.alpha),
            sigma0=float(args.sigma0),
            sigma_bg_mode=args.sigma_bg_mode,
            border_w=int(args.border_w),
            sigma_floor=float(args.sigma_floor),
            dpi=int(args.dpi),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
