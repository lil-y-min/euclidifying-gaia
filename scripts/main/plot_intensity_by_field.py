#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def list_fields(dataset_root: Path) -> List[Path]:
    fields: List[Path] = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and (p / "metadata.csv").exists():
            fields.append(p)
    return fields


def iter_npz_files(field_dir: Path, mode: str = "raw") -> List[Path]:
    if mode == "raw":
        return sorted(field_dir.glob("stamps_*.npz"))
    if mode == "aug":
        return sorted((field_dir / "aug_rot").glob("stamps_*.npz"))
    raise ValueError(f"Unsupported mode={mode}")


def update_histogram(counts: np.ndarray, values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return counts
    h, _ = np.histogram(values, bins=bin_edges)
    counts += h.astype(np.int64, copy=False)
    return counts


def _parse_limits(text: str) -> Tuple[float, float] | None:
    s = str(text).strip()
    if not s:
        return None
    a, b = s.split(",")
    return float(a), float(b)


def _global_sample_for_bins(
    fields: List[Path],
    mode: str,
    seed: int,
    max_values: int,
    per_file_values: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    arrs: List[np.ndarray] = []
    n_vals = 0
    for f in fields:
        for npz_path in iter_npz_files(f, mode):
            try:
                with np.load(npz_path, allow_pickle=False) as z:
                    if "X" not in z:
                        continue
                    x = np.asarray(z["X"])
            except Exception:
                continue
            v = x.reshape(-1)
            m = np.isfinite(v)
            v = v[m]
            if v.size == 0:
                continue
            take = min(int(per_file_values), int(v.size))
            idx = rng.choice(v.size, size=take, replace=False) if take < v.size else None
            s = v[idx] if idx is not None else v
            arrs.append(s.astype(np.float64, copy=False))
            n_vals += int(s.size)
            if n_vals >= int(max_values):
                break
        if n_vals >= int(max_values):
            break
    if not arrs:
        return np.array([], dtype=np.float64)
    out = np.concatenate(arrs)
    if out.size > int(max_values):
        idx = rng.choice(out.size, size=int(max_values), replace=False)
        out = out[idx]
    return out


def _build_bin_edges_linear(
    fields: List[Path],
    mode: str,
    nbins: int,
    qlo: float,
    qhi: float,
    seed: int,
    sample_max_values: int,
    sample_per_file_values: int,
) -> np.ndarray:
    samp = _global_sample_for_bins(
        fields=fields,
        mode=mode,
        seed=int(seed),
        max_values=int(sample_max_values),
        per_file_values=int(sample_per_file_values),
    )
    if samp.size == 0:
        raise RuntimeError("No finite sample values found for global bin-edge estimation.")
    lo = float(np.nanpercentile(samp, float(qlo)))
    hi = float(np.nanpercentile(samp, float(qhi)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(samp))
        hi = float(np.nanmax(samp))
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, int(nbins) + 1)


def _file_stamp_counts(npz_files: List[Path]) -> Tuple[List[int], int]:
    counts: List[int] = []
    total = 0
    for p in npz_files:
        try:
            with np.load(p, allow_pickle=False) as z:
                n = int(np.asarray(z["X"]).shape[0]) if "X" in z else 0
        except Exception:
            n = 0
        counts.append(n)
        total += n
    return counts, total


def _select_indices_global(total_stamps: int, max_stamps: int, seed: int) -> np.ndarray | None:
    if int(max_stamps) <= 0 or total_stamps <= int(max_stamps):
        return None
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total_stamps, size=int(max_stamps), replace=False).astype(np.int64))


def _selected_local(global_sel: np.ndarray | None, start: int, n_local: int) -> np.ndarray | None:
    if global_sel is None:
        return None
    lo = int(np.searchsorted(global_sel, start, side="left"))
    hi = int(np.searchsorted(global_sel, start + n_local, side="left"))
    if hi <= lo:
        return np.array([], dtype=np.int64)
    return (global_sel[lo:hi] - int(start)).astype(np.int64, copy=False)


def compute_field_hist(
    field_dir: Path,
    bin_edges: np.ndarray,
    mode: str = "raw",
    max_stamps_per_field: int = -1,
    seed: int = 42,
    collect_bg_stats: bool = True,
    border_width: int = 2,
) -> Dict[str, object]:
    npz_files = iter_npz_files(field_dir, mode)
    per_file_n, total_stamps = _file_stamp_counts(npz_files)
    selected = _select_indices_global(total_stamps, int(max_stamps_per_field), int(seed))

    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    stamps_processed = 0
    total_pixels = 0
    finite_pixels = 0
    vmin = np.inf
    vmax = -np.inf
    bg_chunks: List[pd.DataFrame] = []

    start = 0
    stamp_idx_field = 0
    for npz_path, n_local in zip(npz_files, per_file_n):
        if n_local <= 0:
            continue
        keep_local = _selected_local(selected, start=start, n_local=n_local)
        start += n_local
        if keep_local is not None and keep_local.size == 0:
            continue

        try:
            with np.load(npz_path, allow_pickle=False) as z:
                if "X" not in z:
                    continue
                x = np.asarray(z["X"])
        except Exception:
            continue
        if keep_local is not None:
            x = x[keep_local]
        if x.size == 0:
            continue

        stamps_processed += int(x.shape[0])
        total_pixels += int(np.prod(x.shape))

        vf = x.reshape(-1)
        m = np.isfinite(vf)
        vf = vf[m]
        if vf.size > 0:
            finite_pixels += int(vf.size)
            vmin = min(vmin, float(np.min(vf)))
            vmax = max(vmax, float(np.max(vf)))
            update_histogram(counts, vf.astype(np.float64, copy=False), bin_edges)

        if collect_bg_stats:
            h, w = int(x.shape[1]), int(x.shape[2])
            bw = max(1, int(border_width))
            border = np.zeros((h, w), dtype=bool)
            border[:bw, :] = True
            border[-bw:, :] = True
            border[:, :bw] = True
            border[:, -bw:] = True
            center = ~border
            xx = np.asarray(x, dtype=np.float64)
            xx[~np.isfinite(xx)] = np.nan
            bmed = np.nanmedian(xx[:, border], axis=1)
            cmed = np.nanmedian(xx[:, center], axis=1)
            grad = bmed - cmed
            n_here = int(x.shape[0])
            idxs = np.arange(stamp_idx_field, stamp_idx_field + n_here, dtype=np.int64)
            stamp_idx_field += n_here
            bg_chunks.append(
                pd.DataFrame(
                    {
                        "field": field_dir.name,
                        "stamp_index_field": idxs,
                        "border_median": bmed.astype(np.float64),
                        "center_median": cmed.astype(np.float64),
                        "bg_gradient": grad.astype(np.float64),
                    }
                )
            )

    count_norm = counts.astype(np.float64) / max(1, finite_pixels)
    bg_df = pd.concat(bg_chunks, ignore_index=True) if bg_chunks else pd.DataFrame(
        columns=["field", "stamp_index_field", "border_median", "center_median", "bg_gradient"]
    )
    return {
        "field": field_dir.name,
        "counts": counts,
        "count_norm": count_norm,
        "stamps_processed": int(stamps_processed),
        "total_pixels": int(total_pixels),
        "finite_pixels": int(finite_pixels),
        "min": float(vmin) if np.isfinite(vmin) else float("nan"),
        "median": float(np.nanmedian(bg_df["center_median"])) if len(bg_df) else float("nan"),
        "p95": float(np.nanpercentile(bg_df["center_median"], 95)) if len(bg_df) else float("nan"),
        "max": float(vmax) if np.isfinite(vmax) else float("nan"),
        "bg_df": bg_df,
    }


def _apply_common_axes(ax: plt.Axes, xlim: Tuple[float, float] | None, ylim: Tuple[float, float] | None) -> None:
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _plot_bg_boxplot(df: pd.DataFrame, value_col: str, out_png: Path, title: str, ylabel: str) -> None:
    fields = sorted(df["field"].dropna().astype(str).unique().tolist())
    data = [pd.to_numeric(df.loc[df["field"] == f, value_col], errors="coerce").dropna().to_numpy(dtype=float) for f in fields]
    fig, ax = plt.subplots(figsize=(max(10.0, 0.62 * len(fields)), 5.4))
    ax.boxplot(data, tick_labels=fields, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Field")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    results_dict: Dict[str, Dict[str, object]],
    bin_centers: np.ndarray,
    outdir: Path,
    mode: str,
    min_count: int = 5000,
    global_norm_min: float = 1e-7,
    ylog: bool = True,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    top_k: int = 5,
    mid_lo: float = 2000.0,
    mid_hi: float = 15000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    fields = sorted(results_dict.keys())
    if not fields:
        raise RuntimeError("No field results to plot.")

    mat_counts = np.vstack([np.asarray(results_dict[f]["counts"], dtype=np.int64) for f in fields])
    mat_norm = np.vstack([np.asarray(results_dict[f]["count_norm"], dtype=np.float64) for f in fields])
    global_counts = np.sum(mat_counts, axis=0)
    global_norm = global_counts.astype(np.float64) / max(1, int(np.sum(global_counts)))
    reliable = (global_counts >= int(min_count)) & (global_norm >= float(global_norm_min))

    # Deviation scores.
    eps = 1e-15
    rows = []
    mid_mask = (bin_centers >= float(mid_lo)) & (bin_centers <= float(mid_hi))
    for f in fields:
        y = np.asarray(results_dict[f]["count_norm"], dtype=np.float64)
        ratio = y / (global_norm + eps)
        ratio[~reliable] = np.nan
        lr = np.log(np.clip(ratio, 1e-15, 1e15))
        score = float(np.nanmean(np.abs(lr))) if np.any(np.isfinite(lr)) else float("nan")
        mm = reliable & mid_mask
        lrm = np.log(np.clip((ratio[mm]), 1e-15, 1e15)) if np.any(mm) else np.array([], dtype=float)
        mid_score = float(np.nanmean(np.abs(lrm))) if lrm.size else float("nan")
        rows.append(
            {
                "field": f,
                "score": score,
                "midrange_score": mid_score,
                "n_reliable_bins": int(np.sum(reliable)),
                "n_midrange_reliable_bins": int(np.sum(mm)),
                "stamps_processed": int(results_dict[f]["stamps_processed"]),
                "finite_pixels": int(results_dict[f]["finite_pixels"]),
            }
        )
    score_df = pd.DataFrame(rows).sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    score_df.to_csv(outdir / "field_deviation_scores.csv", index=False)

    # Plot 1: overlay all fields (no huge legend).
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    for f in fields:
        y = np.asarray(results_dict[f]["count_norm"], dtype=np.float64)
        ax.plot(bin_centers, y, lw=1.0, alpha=0.55, color="#355C7D")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Normalized pixel count")
    ax.set_title("RAW normalized intensity distributions by field (all fields)")
    if ylog:
        ax.set_yscale("log")
    _apply_common_axes(ax, xlim=xlim, ylim=ylim)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "overlay_all_fields.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: top-k + bundle.
    top_fields = score_df["field"].head(int(top_k)).tolist()
    bundle_fields = [f for f in fields if f not in top_fields]
    cmap = plt.get_cmap("tab10", max(1, len(top_fields)))
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    for f in bundle_fields:
        y = np.asarray(results_dict[f]["count_norm"], dtype=np.float64)
        ax.plot(bin_centers, y, lw=0.9, alpha=0.28, color="#9e9e9e")
    for i, f in enumerate(top_fields):
        y = np.asarray(results_dict[f]["count_norm"], dtype=np.float64)
        ax.plot(bin_centers, y, lw=2.4, alpha=0.95, color=cmap(i), label=f)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Normalized pixel count")
    ax.set_title(f"RAW normalized intensity distributions (top {len(top_fields)} deviation + bundle)")
    if ylog:
        ax.set_yscale("log")
    _apply_common_axes(ax, xlim=xlim, ylim=ylim)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "overlay_top5_plus_bundle.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: ratio top-k + bundle with masking gaps.
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    for f in bundle_fields:
        y = np.asarray(results_dict[f]["count_norm"], dtype=np.float64)
        ratio = y / (global_norm + eps)
        ratio[~reliable] = np.nan
        ax.plot(bin_centers, ratio, lw=0.9, alpha=0.28, color="#9e9e9e")
    for i, f in enumerate(top_fields):
        y = np.asarray(results_dict[f]["count_norm"], dtype=np.float64)
        ratio = y / (global_norm + eps)
        ratio[~reliable] = np.nan
        ax.plot(bin_centers, ratio, lw=2.4, alpha=0.95, color=cmap(i), label=f)
    ax.axhline(1.0, color="black", lw=1.0, alpha=0.9)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Field / global normalized count")
    ax.set_title(
        f"RAW ratio-to-global (top {len(top_fields)} + bundle), masked bins: "
        f"global_count<{int(min_count)} or global_norm<{global_norm_min:.0e}"
    )
    _apply_common_axes(ax, xlim=xlim, ylim=None)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "ratio_top5_plus_bundle.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Raw histogram export (normalized + raw counts).
    out_rows = []
    for f in fields:
        c = np.asarray(results_dict[f]["counts"], dtype=np.int64)
        cn = np.asarray(results_dict[f]["count_norm"], dtype=np.float64)
        for b, cc, cnn in zip(bin_centers, c, cn):
            out_rows.append({"field": f, "bin_center": float(b), "count": int(cc), "count_norm": float(cnn)})
    hist_df = pd.DataFrame(out_rows)
    hist_df.to_csv(outdir / f"intensity_hists_{mode}.csv", index=False)

    return score_df, hist_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Improved per-field intensity distribution analysis (RAW default).")
    ap.add_argument("--dataset_root", default="output/dataset_npz")
    ap.add_argument("--mode", choices=["raw", "aug"], default="raw")
    ap.add_argument("--outdir", default="plots/intensity_by_field_raw")
    ap.add_argument("--nbins", type=int, default=250)
    ap.add_argument("--qlo", type=float, default=0.1)
    ap.add_argument("--qhi", type=float, default=99.9)
    ap.add_argument("--sample_max_values", type=int, default=2_000_000)
    ap.add_argument("--sample_per_file_values", type=int, default=20_000)
    ap.add_argument("--max_stamps_per_field", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ylog", action="store_true", default=True)
    ap.add_argument("--xlim", default="")
    ap.add_argument("--ylim", default="")
    ap.add_argument("--min_count", type=int, default=5000)
    ap.add_argument("--global_norm_min", type=float, default=1e-7)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--mid_lo", type=float, default=2000.0)
    ap.add_argument("--mid_hi", type=float, default=15000.0)
    ap.add_argument("--border_width", type=int, default=2)
    ap.add_argument("--skip_bg_diagnostics", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    xlim = _parse_limits(args.xlim)
    ylim = _parse_limits(args.ylim)

    fields = list_fields(dataset_root)
    fields = [f for f in fields if len(iter_npz_files(f, args.mode)) > 0]
    if not fields:
        raise RuntimeError(f"No fields found in {dataset_root} for mode={args.mode}.")

    bin_edges = _build_bin_edges_linear(
        fields=fields,
        mode=str(args.mode),
        nbins=int(args.nbins),
        qlo=float(args.qlo),
        qhi=float(args.qhi),
        seed=int(args.seed),
        sample_max_values=int(args.sample_max_values),
        sample_per_file_values=int(args.sample_per_file_values),
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    results: Dict[str, Dict[str, object]] = {}
    bg_all: List[pd.DataFrame] = []
    for i, field in enumerate(fields, start=1):
        res = compute_field_hist(
            field_dir=field,
            bin_edges=bin_edges,
            mode=str(args.mode),
            max_stamps_per_field=int(args.max_stamps_per_field),
            seed=int(args.seed) + i,
            collect_bg_stats=(not bool(args.skip_bg_diagnostics)),
            border_width=int(args.border_width),
        )
        results[field.name] = res
        print(
            f"[{args.mode}] {i:02d}/{len(fields):02d} {field.name}: "
            f"stamps={res['stamps_processed']:,}, finite_pixels={res['finite_pixels']:,}, "
            f"min={res['min']:.5g}, med={res['median']:.5g}, p95={res['p95']:.5g}, max={res['max']:.5g}"
        )
        if not bool(args.skip_bg_diagnostics):
            bg_all.append(res["bg_df"])

    make_plots(
        results_dict=results,
        bin_centers=bin_centers,
        outdir=outdir,
        mode=str(args.mode),
        min_count=int(args.min_count),
        global_norm_min=float(args.global_norm_min),
        ylog=bool(args.ylog),
        xlim=xlim,
        ylim=ylim,
        top_k=int(args.top_k),
        mid_lo=float(args.mid_lo),
        mid_hi=float(args.mid_hi),
    )

    if not bool(args.skip_bg_diagnostics):
        bg_df = pd.concat(bg_all, ignore_index=True) if bg_all else pd.DataFrame(
            columns=["field", "stamp_index_field", "border_median", "center_median", "bg_gradient"]
        )
        bg_df.to_csv(outdir / "per_stamp_bg_stats.csv", index=False)
        if len(bg_df) > 0:
            _plot_bg_boxplot(
                df=bg_df,
                value_col="border_median",
                out_png=outdir / "border_median_by_field.png",
                title="RAW border median by field",
                ylabel="Border median intensity",
            )
            _plot_bg_boxplot(
                df=bg_df,
                value_col="bg_gradient",
                out_png=outdir / "bg_gradient_by_field.png",
                title="RAW background gradient by field (border_median - center_median)",
                ylabel="Background gradient",
            )

    print("DONE")
    print("outdir:", outdir)


if __name__ == "__main__":
    main()
