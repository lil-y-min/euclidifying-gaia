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
    out: List[Path] = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and (p / "metadata.csv").exists():
            out.append(p)
    return out


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


def _global_sample_for_bins(
    fields: List[Path],
    mode: str,
    seed: int,
    max_values: int,
    sample_per_file: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    chunks: List[np.ndarray] = []
    n = 0
    for field in fields:
        for npz_path in iter_npz_files(field, mode):
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
            take = min(int(sample_per_file), int(v.size))
            idx = rng.choice(v.size, size=take, replace=False) if take < v.size else None
            s = v[idx] if idx is not None else v
            chunks.append(s.astype(np.float64, copy=False))
            n += int(s.size)
            if n >= int(max_values):
                break
        if n >= int(max_values):
            break
    if not chunks:
        return np.array([], dtype=np.float64)
    vals = np.concatenate(chunks)
    if vals.size > int(max_values):
        idx = rng.choice(vals.size, size=int(max_values), replace=False)
        vals = vals[idx]
    return vals


def build_shared_bin_edges(
    fields: List[Path],
    mode: str,
    nbins: int,
    qlo: float,
    qhi: float,
    seed: int,
    max_values: int,
    sample_per_file: int,
) -> np.ndarray:
    samp = _global_sample_for_bins(
        fields=fields,
        mode=mode,
        seed=seed,
        max_values=max_values,
        sample_per_file=sample_per_file,
    )
    if samp.size == 0:
        raise RuntimeError("Could not estimate global bin edges: no finite sampled values.")
    lo = float(np.nanpercentile(samp, float(qlo)))
    hi = float(np.nanpercentile(samp, float(qhi)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(samp))
        hi = float(np.nanmax(samp))
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, int(nbins) + 1)


def _sample_pixels_for_quantiles(
    values: np.ndarray,
    rng: np.random.Generator,
    sample_per_file: int,
) -> np.ndarray:
    if values.size == 0:
        return values
    take = min(int(sample_per_file), int(values.size))
    if take >= values.size:
        return values
    idx = rng.choice(values.size, size=take, replace=False)
    return values[idx]


def compute_field_data(
    field_dir: Path,
    bin_edges: np.ndarray,
    mode: str,
    seed: int,
    quantile_sample_max: int,
    quantile_sample_per_file: int,
    border_width: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(int(seed))
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    finite_pixels = 0
    stamps_processed = 0
    samples: List[np.ndarray] = []
    sample_n = 0
    bg_rows: List[pd.DataFrame] = []

    for npz_path in iter_npz_files(field_dir, mode):
        try:
            with np.load(npz_path, allow_pickle=False) as z:
                if "X" not in z:
                    continue
                x = np.asarray(z["X"])
        except Exception:
            continue
        if x.size == 0:
            continue
        stamps_processed += int(x.shape[0])

        v = x.reshape(-1)
        m = np.isfinite(v)
        vf = v[m].astype(np.float64, copy=False)
        if vf.size:
            finite_pixels += int(vf.size)
            update_histogram(counts, vf, bin_edges)
            if sample_n < int(quantile_sample_max):
                s = _sample_pixels_for_quantiles(vf, rng, sample_per_file=int(quantile_sample_per_file))
                samples.append(s)
                sample_n += int(s.size)

        # Per-stamp background diagnostics.
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
        bg_rows.append(
            pd.DataFrame(
                {
                    "field": field_dir.name,
                    "border_median": bmed.astype(np.float64),
                    "center_median": cmed.astype(np.float64),
                    "bg_offset": grad.astype(np.float64),
                }
            )
        )

    if samples:
        vals = np.concatenate(samples)
        if vals.size > int(quantile_sample_max):
            idx = rng.choice(vals.size, size=int(quantile_sample_max), replace=False)
            vals = vals[idx]
    else:
        vals = np.array([], dtype=np.float64)
    norm = counts.astype(np.float64) / max(1, int(finite_pixels))
    cdf = np.cumsum(counts.astype(np.float64))
    cdf = cdf / max(1.0, float(cdf[-1]) if cdf.size else 1.0)
    bg_df = pd.concat(bg_rows, ignore_index=True) if bg_rows else pd.DataFrame(
        columns=["field", "border_median", "center_median", "bg_offset"]
    )
    return {
        "field": field_dir.name,
        "counts": counts,
        "norm": norm,
        "cdf": cdf,
        "finite_pixels": int(finite_pixels),
        "stamps_processed": int(stamps_processed),
        "quant_sample": vals,
        "bg_df": bg_df,
    }


def _score_zoom(
    field_norm: np.ndarray,
    global_norm: np.ndarray,
    x: np.ndarray,
    reliable: np.ndarray,
    x_lo: float,
    x_hi: float,
) -> float:
    m = reliable & (x >= float(x_lo)) & (x <= float(x_hi))
    if not np.any(m):
        return float("nan")
    eps = 1e-15
    ratio = (field_norm[m] + eps) / (global_norm[m] + eps)
    return float(np.mean(np.abs(np.log(ratio))))


def _plot_bundle(
    x: np.ndarray,
    curves: Dict[str, np.ndarray],
    top_fields: List[str],
    out_png: Path,
    title: str,
    ylabel: str,
    xlim: Tuple[float, float],
    ylog: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    for f, y in curves.items():
        if f in top_fields:
            continue
        ax.plot(x, y, lw=0.9, alpha=0.22, color="#9e9e9e")
    cmap = plt.get_cmap("tab10", max(1, len(top_fields)))
    for i, f in enumerate(top_fields):
        y = curves[f]
        ax.plot(x, y, lw=2.4, alpha=0.95, color=cmap(i), label=f)
    ax.set_xlim(*xlim)
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Background-shift analysis from per-field RAW stamp intensity distributions.")
    ap.add_argument("--dataset_root", default="output/dataset_npz")
    ap.add_argument("--mode", choices=["raw"], default="raw")
    ap.add_argument("--outdir", default="plots/intensity_bg_zoom_raw")
    ap.add_argument("--nbins", type=int, default=500)
    ap.add_argument("--qlo", type=float, default=0.1)
    ap.add_argument("--qhi", type=float, default=99.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_max_values", type=int, default=2_000_000, help="Global pooled sample size for bin-edge estimation.")
    ap.add_argument("--sample_per_file_values", type=int, default=20_000)
    ap.add_argument("--quantile_sample_max_per_field", type=int, default=5_000_000)
    ap.add_argument("--quantile_sample_per_file", type=int, default=250_000)
    ap.add_argument("--min_count", type=int, default=5000)
    ap.add_argument("--global_norm_min", type=float, default=1e-7)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--zoom1_hi", type=float, default=2000.0)
    ap.add_argument("--zoom2_hi", type=float, default=5000.0)
    ap.add_argument("--mid_lo", type=float, default=2000.0)
    ap.add_argument("--mid_hi", type=float, default=15000.0)
    ap.add_argument("--border_width", type=int, default=2)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fields = list_fields(dataset_root)
    fields = [f for f in fields if len(iter_npz_files(f, args.mode)) > 0]
    if not fields:
        raise RuntimeError(f"No fields found for mode={args.mode} under {dataset_root}")

    bin_edges = build_shared_bin_edges(
        fields=fields,
        mode=args.mode,
        nbins=int(args.nbins),
        qlo=float(args.qlo),
        qhi=float(args.qhi),
        seed=int(args.seed),
        max_values=int(args.sample_max_values),
        sample_per_file=int(args.sample_per_file_values),
    )
    x = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    results: Dict[str, Dict[str, object]] = {}
    bg_chunks: List[pd.DataFrame] = []
    for i, field in enumerate(fields, start=1):
        r = compute_field_data(
            field_dir=field,
            bin_edges=bin_edges,
            mode=args.mode,
            seed=int(args.seed) + i,
            quantile_sample_max=int(args.quantile_sample_max_per_field),
            quantile_sample_per_file=int(args.quantile_sample_per_file),
            border_width=int(args.border_width),
        )
        results[field.name] = r
        bg_chunks.append(r["bg_df"])
        print(
            f"[{args.mode}] {i:02d}/{len(fields):02d} {field.name}: "
            f"stamps={r['stamps_processed']:,}, finite_pixels={r['finite_pixels']:,}, quant_sample={len(r['quant_sample']):,}"
        )

    fields_sorted = sorted(results.keys())
    mat_counts = np.vstack([np.asarray(results[f]["counts"], dtype=np.int64) for f in fields_sorted])
    mat_norm = np.vstack([np.asarray(results[f]["norm"], dtype=np.float64) for f in fields_sorted])
    global_counts = np.sum(mat_counts, axis=0)
    global_norm = global_counts.astype(np.float64) / max(1.0, float(np.sum(global_counts)))
    reliable = (global_counts >= int(args.min_count)) & (global_norm >= float(args.global_norm_min))

    # Export histogram table.
    hist_rows = []
    for f in fields_sorted:
        c = np.asarray(results[f]["counts"], dtype=np.int64)
        cn = np.asarray(results[f]["norm"], dtype=np.float64)
        for xv, cc, cnn in zip(x, c, cn):
            hist_rows.append({"field": f, "bin_center": float(xv), "count": int(cc), "count_norm": float(cnn)})
    pd.DataFrame(hist_rows).to_csv(outdir / "intensity_hists_raw.csv", index=False)

    # Quantile-based horizontal shift metrics.
    quant_rows = []
    field_quant = {}
    for f in fields_sorted:
        v = np.asarray(results[f]["quant_sample"], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            q = {k: np.nan for k in ["p05", "p10", "p25", "median", "p75"]}
        else:
            q = {
                "p05": float(np.nanpercentile(v, 5)),
                "p10": float(np.nanpercentile(v, 10)),
                "p25": float(np.nanpercentile(v, 25)),
                "median": float(np.nanpercentile(v, 50)),
                "p75": float(np.nanpercentile(v, 75)),
            }
        field_quant[f] = q

    global_sample_chunks = [np.asarray(results[f]["quant_sample"], dtype=np.float64) for f in fields_sorted if len(results[f]["quant_sample"]) > 0]
    if global_sample_chunks:
        gv = np.concatenate(global_sample_chunks)
        gv = gv[np.isfinite(gv)]
    else:
        gv = np.array([], dtype=np.float64)
    if gv.size == 0:
        raise RuntimeError("Global quantile sample is empty; cannot compute background shift metrics.")
    gq = {
        "p05": float(np.nanpercentile(gv, 5)),
        "p10": float(np.nanpercentile(gv, 10)),
        "p25": float(np.nanpercentile(gv, 25)),
        "median": float(np.nanpercentile(gv, 50)),
        "p75": float(np.nanpercentile(gv, 75)),
    }
    giqr = gq["p75"] - gq["p25"]

    for f in fields_sorted:
        q = field_quant[f]
        iqr = (q["p75"] - q["p25"]) if np.isfinite(q["p75"]) and np.isfinite(q["p25"]) else np.nan
        iqr_ratio = (iqr / giqr) if np.isfinite(iqr) and np.isfinite(giqr) and giqr != 0 else np.nan
        s2k = _score_zoom(np.asarray(results[f]["norm"], dtype=np.float64), global_norm, x, reliable, 0.0, float(args.zoom1_hi))
        s5k = _score_zoom(np.asarray(results[f]["norm"], dtype=np.float64), global_norm, x, reliable, 0.0, float(args.zoom2_hi))
        smid = _score_zoom(np.asarray(results[f]["norm"], dtype=np.float64), global_norm, x, reliable, float(args.mid_lo), float(args.mid_hi))
        quant_rows.append(
            {
                "field": f,
                "p05": q["p05"],
                "p10": q["p10"],
                "p25": q["p25"],
                "median": q["median"],
                "p75": q["p75"],
                "delta_median": q["median"] - gq["median"],
                "delta_p10": q["p10"] - gq["p10"],
                "delta_p25": q["p25"] - gq["p25"],
                "iqr": iqr,
                "iqr_ratio": iqr_ratio,
                "zoom_score_0_2000": s2k,
                "zoom_score_0_5000": s5k,
                "midrange_score": smid,
                "n_reliable_bins": int(np.sum(reliable)),
                "stamps_processed": int(results[f]["stamps_processed"]),
                "finite_pixels": int(results[f]["finite_pixels"]),
            }
        )

    mdf = pd.DataFrame(quant_rows)
    mdf["abs_delta_median"] = np.abs(pd.to_numeric(mdf["delta_median"], errors="coerce"))
    mdf = mdf.sort_values("abs_delta_median", ascending=False).reset_index(drop=True)
    mdf.to_csv(outdir / "background_shift_metrics.csv", index=False)

    # Top-k by zoom score for zoom plots.
    top_2k = (
        mdf.sort_values("zoom_score_0_2000", ascending=False, na_position="last")["field"]
        .head(int(args.top_k))
        .tolist()
    )
    top_5k = (
        mdf.sort_values("zoom_score_0_5000", ascending=False, na_position="last")["field"]
        .head(int(args.top_k))
        .tolist()
    )

    norm_curves = {f: np.asarray(results[f]["norm"], dtype=np.float64) for f in fields_sorted}
    cdf_curves = {f: np.asarray(results[f]["cdf"], dtype=np.float64) for f in fields_sorted}

    _plot_bundle(
        x=x,
        curves=norm_curves,
        top_fields=top_2k,
        out_png=outdir / "overlay_bg_zoom_0_2000.png",
        title=f"RAW normalized histograms, low-intensity zoom [0, {int(args.zoom1_hi)}]",
        ylabel="Normalized pixel count",
        xlim=(0.0, float(args.zoom1_hi)),
        ylog=True,
    )
    _plot_bundle(
        x=x,
        curves=norm_curves,
        top_fields=top_5k,
        out_png=outdir / "overlay_bg_zoom_0_5000.png",
        title=f"RAW normalized histograms, low-intensity zoom [0, {int(args.zoom2_hi)}]",
        ylabel="Normalized pixel count",
        xlim=(0.0, float(args.zoom2_hi)),
        ylog=True,
    )
    _plot_bundle(
        x=x,
        curves=cdf_curves,
        top_fields=top_2k,
        out_png=outdir / "cdf_bg_zoom_0_2000.png",
        title=f"RAW CDF overlay in background range [0, {int(args.zoom1_hi)}]",
        ylabel="CDF",
        xlim=(0.0, float(args.zoom1_hi)),
        ylog=False,
    )
    _plot_bundle(
        x=x,
        curves=cdf_curves,
        top_fields=top_5k,
        out_png=outdir / "cdf_bg_zoom_0_5000.png",
        title=f"RAW CDF overlay in background range [0, {int(args.zoom2_hi)}]",
        ylabel="CDF",
        xlim=(0.0, float(args.zoom2_hi)),
        ylog=False,
    )

    # Ratio curves with reliability masking.
    ratio_curves = {}
    eps = 1e-15
    for f in fields_sorted:
        r = np.asarray(results[f]["norm"], dtype=np.float64) / (global_norm + eps)
        r[~reliable] = np.nan
        ratio_curves[f] = r
    _plot_bundle(
        x=x,
        curves=ratio_curves,
        top_fields=top_2k,
        out_png=outdir / "ratio_bg_zoom_0_2000.png",
        title=(
            f"RAW ratio to global, masked bins (count<{int(args.min_count)} or global_norm<{args.global_norm_min:.0e}), "
            f"zoom [0, {int(args.zoom1_hi)}]"
        ),
        ylabel="Field / global normalized count",
        xlim=(0.0, float(args.zoom1_hi)),
        ylog=False,
    )
    _plot_bundle(
        x=x,
        curves=ratio_curves,
        top_fields=top_5k,
        out_png=outdir / "ratio_bg_zoom_0_5000.png",
        title=(
            f"RAW ratio to global, masked bins (count<{int(args.min_count)} or global_norm<{args.global_norm_min:.0e}), "
            f"zoom [0, {int(args.zoom2_hi)}]"
        ),
        ylabel="Field / global normalized count",
        xlim=(0.0, float(args.zoom2_hi)),
        ylog=False,
    )

    # Background per-stamp stats.
    bg_df = pd.concat(bg_chunks, ignore_index=True) if bg_chunks else pd.DataFrame(
        columns=["field", "border_median", "center_median", "bg_offset"]
    )
    bg_df.to_csv(outdir / "per_stamp_bg_stats.csv", index=False)
    if len(bg_df):
        _plot_bg_boxplot(
            df=bg_df,
            value_col="border_median",
            out_png=outdir / "border_median_by_field.png",
            title="RAW border_median by field",
            ylabel="border_median",
        )
        _plot_bg_boxplot(
            df=bg_df,
            value_col="bg_offset",
            out_png=outdir / "bg_offset_by_field.png",
            title="RAW bg_offset (border_median - center_median) by field",
            ylabel="bg_offset",
        )
        bmed = (
            bg_df.groupby("field", as_index=False)["border_median"]
            .median()
            .rename(columns={"border_median": "field_border_median"})
        )
        global_border_median = float(np.nanmedian(pd.to_numeric(bg_df["border_median"], errors="coerce")))
        bmed["global_border_median"] = global_border_median
        bmed["delta_border_median"] = bmed["field_border_median"] - global_border_median
        bmed = bmed.sort_values("delta_border_median", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
        bmed.to_csv(outdir / "field_border_median_summary.csv", index=False)

    print("DONE")
    print("outdir:", outdir)


if __name__ == "__main__":
    main()

