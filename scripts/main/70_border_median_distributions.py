#!/usr/bin/env python3
"""
70_border_median_distributions.py

For each field, extract the per-stamp border-pixel median (configurable border
width, default 3 px) and plot the distributions overlaid across fields.

Outputs (in --outdir):
  border_median_kde_all.png          — all fields overlaid (normalised KDE-style)
  border_median_kde_topk.png         — top-k most-shifted fields highlighted
  border_median_hist_raw.png         — raw pixel values from border region
  border_stats_summary.csv           — per-field median / IQR / delta from global
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def list_fields(dataset_root: Path) -> List[Path]:
    return sorted(
        p for p in dataset_root.iterdir()
        if p.is_dir() and (p / "metadata.csv").exists()
    )


def iter_npz(field_dir: Path) -> List[Path]:
    return sorted(field_dir.glob("stamps_*.npz"))


def load_field(
    field_dir: Path,
    border_width: int,
    max_stamps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    border_medians : (N,) float64  — per-stamp median of border pixels
    border_pixels  : (M,) float64  — all finite border pixel values (random subsample)
    """
    bw = max(1, int(border_width))
    med_chunks: List[np.ndarray] = []
    pix_chunks: List[np.ndarray] = []

    for npz_path in iter_npz(field_dir):
        try:
            with np.load(npz_path, allow_pickle=False) as z:
                if "X" not in z:
                    continue
                x = np.asarray(z["X"], dtype=np.float64)
        except Exception:
            continue

        if x.ndim != 3:
            continue

        n, h, w = x.shape
        # build border mask once (same for all stamps)
        mask = np.zeros((h, w), dtype=bool)
        mask[:bw, :] = True
        mask[-bw:, :] = True
        mask[:, :bw] = True
        mask[:, -bw:] = True

        # per-stamp border median — shape (n,)
        border_pixels_all = x[:, mask]  # (n, n_border_pixels)
        border_pixels_all[~np.isfinite(border_pixels_all)] = np.nan
        stamp_meds = np.nanmedian(border_pixels_all, axis=1)
        med_chunks.append(stamp_meds)

        # collect raw border pixel values (subsample for memory)
        raw = border_pixels_all.reshape(-1)
        raw = raw[np.isfinite(raw)]
        if raw.size > 50_000:
            raw = rng.choice(raw, size=50_000, replace=False)
        pix_chunks.append(raw)

    if not med_chunks:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    meds = np.concatenate(med_chunks)
    meds = meds[np.isfinite(meds)]
    pixs = np.concatenate(pix_chunks) if pix_chunks else np.array([], dtype=np.float64)

    if max_stamps > 0 and meds.size > max_stamps:
        idx = rng.choice(meds.size, size=max_stamps, replace=False)
        meds = meds[idx]

    return meds, pixs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def _kde_curve(values: np.ndarray, x_grid: np.ndarray, bw: str | float = "scott") -> np.ndarray:
    if values.size < 3:
        return np.zeros_like(x_grid)
    try:
        kde = gaussian_kde(values, bw_method=bw)
        return kde(x_grid)
    except Exception:
        return np.zeros_like(x_grid)


def plot_counts_log(
    field_data: Dict[str, np.ndarray],
    out_png: Path,
    title: str,
    nbins: int = 200,
    xlim: Tuple[float, float] | None = None,
    highlight: List[str] | None = None,
) -> None:
    """Histogram of per-stamp border medians, log count y-axis."""
    all_vals = np.concatenate([v for v in field_data.values() if v.size])
    lo = float(np.percentile(all_vals, 0.1)) if xlim is None else xlim[0]
    hi = float(np.percentile(all_vals, 99.9)) if xlim is None else xlim[1]
    bin_edges = np.linspace(lo, hi, nbins + 1)
    x = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=(11, 6))
    rest = [f for f in field_data if highlight is None or f not in highlight]
    for fname in rest:
        vals = field_data[fname]
        if vals.size == 0:
            continue
        counts, _ = np.histogram(vals, bins=bin_edges)
        color = "#aaaaaa" if highlight else None
        alpha = 0.25 if highlight else 0.55
        ax.plot(x, counts, lw=0.9, alpha=alpha, color=color)
    if highlight:
        cmap = plt.get_cmap("tab10", max(1, len(highlight)))
        for i, fname in enumerate(highlight):
            vals = field_data[fname]
            if vals.size == 0:
                continue
            counts, _ = np.histogram(vals, bins=bin_edges)
            ax.plot(x, counts, lw=2.2, alpha=0.92, color=cmap(i), label=fname)
        ax.legend(fontsize=8, loc="best", frameon=True)
    ax.set_yscale("log")
    ax.set_xlabel("Border-pixel median per stamp  [ADU]")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_kde_all(
    field_data: Dict[str, np.ndarray],
    x_grid: np.ndarray,
    out_png: Path,
    title: str,
    highlight: List[str] | None = None,
    xlim: Tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    if highlight is None:
        for fname, vals in field_data.items():
            y = _kde_curve(vals, x_grid)
            ax.plot(x_grid, y, lw=1.0, alpha=0.55)
        ax.set_title(title)
    else:
        rest = [f for f in field_data if f not in highlight]
        for fname in rest:
            y = _kde_curve(field_data[fname], x_grid)
            ax.plot(x_grid, y, lw=0.9, alpha=0.20, color="#aaaaaa")
        cmap = plt.get_cmap("tab10", max(1, len(highlight)))
        for i, fname in enumerate(highlight):
            y = _kde_curve(field_data[fname], x_grid)
            ax.plot(x_grid, y, lw=2.2, alpha=0.92, color=cmap(i), label=fname)
        ax.legend(fontsize=8, loc="best", frameon=True)
        ax.set_title(title)

    ax.set_xlabel("Border-pixel median per stamp  [ADU]")
    ax.set_ylabel("Density (KDE)")
    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hist_overlay(
    field_pixs: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    out_png: Path,
    title: str,
    xlim: Tuple[float, float] | None = None,
) -> None:
    """Overlaid KDE curves of raw border pixel values, one per field."""
    lo = float(bin_edges[0])
    hi = float(bin_edges[-1])
    x_grid = np.linspace(lo, hi, 800)

    fig, ax = plt.subplots(figsize=(11, 6))
    for fname, vals in field_pixs.items():
        if vals.size < 3:
            continue
        y = _kde_curve(vals, x_grid, bw="scott")
        ax.plot(x_grid, y, lw=1.1, alpha=0.55, label=fname)
    ax.set_xlabel("Border pixel intensity  [ADU]")
    ax.set_ylabel("Density (KDE)")
    ax.set_title(title)
    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Border-pixel median distributions across fields.")
    ap.add_argument("--dataset_root", default="output/dataset_npz")
    ap.add_argument("--outdir",       default="plots/border_median_distributions")
    ap.add_argument("--border_width", type=int,   default=3,         help="Border width in pixels (default 3)")
    ap.add_argument("--max_stamps",   type=int,   default=-1,        help="Max stamps per field for KDE (-1 = all)")
    ap.add_argument("--nbins",        type=int,   default=200,       help="Bins for raw-pixel histogram")
    ap.add_argument("--kde_points",   type=int,   default=500,       help="KDE evaluation grid points")
    ap.add_argument("--top_k",        type=int,   default=6,         help="Fields to highlight in top-k plot")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--qlo",          type=float, default=0.5,       help="Lower percentile for axis clipping")
    ap.add_argument("--qhi",          type=float, default=99.5,      help="Upper percentile for axis clipping")
    ap.add_argument("--zoom_lo",      type=float, default=-10.0,     help="Low-intensity zoom x-axis lower bound [ADU]")
    ap.add_argument("--zoom_hi",      type=float, default=50.0,      help="Low-intensity zoom x-axis upper bound [ADU]")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fields = list_fields(dataset_root)
    fields = [f for f in fields if iter_npz(f)]
    if not fields:
        raise RuntimeError(f"No fields with stamp files found under {dataset_root}")

    rng = np.random.default_rng(args.seed)
    field_meds: Dict[str, np.ndarray] = {}
    field_pixs: Dict[str, np.ndarray] = {}
    stats_rows: List[dict] = []

    for i, fdir in enumerate(fields, 1):
        meds, pixs = load_field(fdir, args.border_width, args.max_stamps, rng)
        field_meds[fdir.name] = meds
        field_pixs[fdir.name] = pixs
        if meds.size:
            row = {
                "field":   fdir.name,
                "n_stamps": int(meds.size),
                "median":  float(np.median(meds)),
                "mean":    float(np.mean(meds)),
                "p25":     float(np.percentile(meds, 25)),
                "p75":     float(np.percentile(meds, 75)),
                "iqr":     float(np.percentile(meds, 75) - np.percentile(meds, 25)),
            }
        else:
            row = {"field": fdir.name, "n_stamps": 0,
                   "median": np.nan, "mean": np.nan,
                   "p25": np.nan, "p75": np.nan, "iqr": np.nan}
        stats_rows.append(row)
        print(f"  {i:02d}/{len(fields):02d}  {fdir.name:<30}  n={meds.size:,}  "
              f"med={row['median']:.1f}  IQR={row['iqr']:.1f}")

    # --- global stats for delta
    all_meds = np.concatenate([v for v in field_meds.values() if v.size])
    global_median = float(np.median(all_meds)) if all_meds.size else 0.0
    stats_df = pd.DataFrame(stats_rows)
    stats_df["delta_median"] = stats_df["median"] - global_median
    stats_df["abs_delta"] = stats_df["delta_median"].abs()
    stats_df = stats_df.sort_values("abs_delta", ascending=False).reset_index(drop=True)
    stats_df.to_csv(outdir / "border_stats_summary.csv", index=False)
    print(f"\nGlobal border median: {global_median:.1f}")
    print(stats_df[["field", "n_stamps", "median", "delta_median", "iqr"]].to_string(index=False))

    # --- build x-grid from pooled medians
    lo = float(np.percentile(all_meds, args.qlo))
    hi = float(np.percentile(all_meds, args.qhi))
    x_grid = np.linspace(lo, hi, args.kde_points)

    # --- top-k fields by |delta_median|
    top_fields = stats_df["field"].head(args.top_k).tolist()

    zoom_xlim = (args.zoom_lo, args.zoom_hi)
    # build a fine x_grid covering at least the zoom range
    x_grid_zoom = np.linspace(args.zoom_lo, args.zoom_hi, args.kde_points)

    # Count histograms with log y-scale — zoomed and full range
    plot_counts_log(
        field_data=field_meds,
        out_png=outdir / "border_median_counts_log_zoom.png",
        title=f"Border-pixel median — log counts, all fields, zoom [{args.zoom_lo:.0f}, {args.zoom_hi:.0f}] ADU  (bw={args.border_width} px)",
        xlim=zoom_xlim,
    )
    plot_counts_log(
        field_data=field_meds,
        out_png=outdir / "border_median_counts_log_full.png",
        title=f"Border-pixel median — log counts, all fields, full range  (bw={args.border_width} px)",
    )

    # KDE — all fields, full and zoomed
    plot_kde_all(
        field_data=field_meds,
        x_grid=x_grid,
        out_png=outdir / "border_median_kde_all.png",
        title=f"Border-pixel median distributions — all fields  (border_width={args.border_width} px)",
    )
    plot_kde_all(
        field_data=field_meds,
        x_grid=x_grid_zoom,
        out_png=outdir / "border_median_kde_all_zoom.png",
        title=f"Border-pixel median distributions — all fields, zoom [{args.zoom_lo:.0f}, {args.zoom_hi:.0f}] ADU  (bw={args.border_width} px)",
        xlim=zoom_xlim,
    )

    # Plot 3: raw border pixel histogram overlaid
    all_pixs = np.concatenate([v for v in field_pixs.values() if v.size])
    if all_pixs.size:
        plo = float(np.percentile(all_pixs, args.qlo))
        phi = float(np.percentile(all_pixs, args.qhi))
        bin_edges = np.linspace(plo, phi, args.nbins + 1)
        plot_hist_overlay(
            field_pixs=field_pixs,
            bin_edges=bin_edges,
            out_png=outdir / "border_median_hist_raw_zoom.png",
            title=f"Raw border pixel distributions by field  (border_width={args.border_width} px)  [zoom]",
            xlim=(-1.0, 100.0),
        )
        plot_hist_overlay(
            field_pixs=field_pixs,
            bin_edges=bin_edges,
            out_png=outdir / "border_median_hist_raw_full.png",
            title=f"Raw border pixel distributions by field  (border_width={args.border_width} px)  [full range]",
        )

    print("\nDONE — outputs in:", outdir)


if __name__ == "__main__":
    main()
