#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_source_lookup(dataset_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for meta in sorted(dataset_root.glob("*/metadata_16d.csv")):
        field = meta.parent.name
        use = ["source_id", "phot_g_mean_mag", "npz_file", "index_in_file"]
        df = pd.read_csv(meta, usecols=use, low_memory=False)
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
        df["phot_g_mean_mag"] = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
        df["index_in_file"] = pd.to_numeric(df["index_in_file"], errors="coerce")
        df = df.dropna(subset=["source_id", "npz_file", "index_in_file"]).copy()
        df["source_id"] = df["source_id"].astype(np.int64)
        df["index_in_file"] = df["index_in_file"].astype(np.int64)
        df["field_tag_meta"] = field
        df["npz_relpath"] = df.apply(lambda r: f"{field}/{r['npz_file']}", axis=1)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return out


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor="black")
    plt.close()


def make_montage(df: pd.DataFrame, dataset_root: Path, out_png: Path, title: str) -> None:
    n = len(df)
    if n == 0:
        return
    ncols = 10
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.45, nrows * 1.5), facecolor="black")
    axes = np.array(axes).reshape(nrows, ncols)
    cache: Dict[str, np.ndarray] = {}

    for i in range(nrows * ncols):
        ax = axes.flat[i]
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        r = df.iloc[i]
        rel = str(r["npz_relpath"])
        idx = int(r["index_in_file"])
        npz_path = dataset_root / rel
        if rel not in cache:
            try:
                cache[rel] = np.load(npz_path)["X"]
            except Exception:
                ax.axis("off")
                continue
        X = cache[rel]
        if idx < 0 or idx >= X.shape[0]:
            ax.axis("off")
            continue
        im = X[idx].astype(float)
        vmin = np.percentile(im, 5)
        vmax = np.percentile(im, 99)
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(
            f"{int(r['short_id']):03d} KEEP\nm={r['phot_g_mean_mag']:.1f}",
            fontsize=6,
            color="orange",
        )

    fig.suptitle(title, color="white")
    _savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags_csv", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_bit", type=int, default=60)
    ap.add_argument("--exclude_bits_mask", type=int, default=2048, help="Exclude rows whose quality_flag has any of these bits set.")
    ap.add_argument("--isolate_review_bits", action="store_true", help="When selecting near-threshold keeps for one bit, require other review bits (1,2,4,32) to be off.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    df = pd.read_csv(args.flags_csv, low_memory=False)
    qf = pd.to_numeric(df["quality_flag"], errors="coerce").fillna(0).astype(np.int64)
    if int(args.exclude_bits_mask) > 0:
        keep = (qf & int(args.exclude_bits_mask)) == 0
        df = df.loc[keep].copy()
        qf = pd.to_numeric(df["quality_flag"], errors="coerce").fillna(0).astype(np.int64)
    for b in [1, 2, 4, 32]:
        df[f"b{b}"] = ((qf & b) != 0).astype(np.uint8)

    # Derived metrics
    df["v_niter_model"] = pd.to_numeric(df.get("v_niter_model"), errors="coerce")
    df["v_noisearea_model"] = pd.to_numeric(df.get("v_noisearea_model"), errors="coerce")
    df["v_spread_model"] = pd.to_numeric(df.get("v_spread_model"), errors="coerce")
    df["v_spreaderr_model"] = pd.to_numeric(df.get("v_spreaderr_model"), errors="coerce")
    df["v_flux_auto"] = pd.to_numeric(df.get("v_flux_auto"), errors="coerce")
    df["v_flux_model"] = pd.to_numeric(df.get("v_flux_model"), errors="coerce")
    df["spread_sig"] = np.abs(df["v_spread_model"] / df["v_spreaderr_model"])
    df["deltaF_frac"] = np.abs(df["v_flux_auto"] - df["v_flux_model"]) / np.maximum(np.abs(df["v_flux_auto"]), 1e-12)

    # Field-wise q95 for bit1
    niter_q95 = df.groupby("field_tag")["v_niter_model"].quantile(0.95).to_dict()
    df["bit1_threshold"] = df["field_tag"].map(niter_q95).astype(float)

    # Keep-side margins (smaller positive => closer to threshold while still kept)
    df["m1"] = df["bit1_threshold"] - df["v_niter_model"]                   # keep if >0 and b1==0
    df["m2"] = np.minimum(df["v_noisearea_model"] - 5.0, 500.0 - df["v_noisearea_model"])  # keep if >0 and b2==0
    df["m4"] = df["spread_sig"] - 2.0                                       # keep if >0 and b4==0
    df["m32"] = 0.3 - df["deltaF_frac"]                                     # keep if >0 and b32==0

    lookup = build_source_lookup(dataset_root)
    df = df.merge(lookup, on="source_id", how="left")

    plan = [
        (1, "m1", "v_niter_model", "bit1_threshold"),
        (2, "m2", "v_noisearea_model", None),
        (4, "m4", "spread_sig", None),
        (32, "m32", "deltaF_frac", None),
    ]

    for bit, margin_col, metric_col, thr_col in plan:
        keep = df[(df[f"b{bit}"] == 0) & np.isfinite(df[margin_col]) & (df[margin_col] > 0)].copy()
        if bool(args.isolate_review_bits):
            other_bits = [x for x in [1, 2, 4, 32] if x != bit]
            for ob in other_bits:
                keep = keep[keep[f"b{ob}"] == 0]
        keep = keep[np.isfinite(pd.to_numeric(keep["index_in_file"], errors="coerce")) & keep["npz_relpath"].notna()].copy()
        keep = keep.sort_values(margin_col, ascending=True).head(int(args.n_per_bit)).reset_index(drop=True)
        keep["short_id"] = np.arange(1, len(keep) + 1, dtype=int)
        keep["bit"] = bit
        keep["metric_name"] = metric_col
        keep["metric_value"] = pd.to_numeric(keep[metric_col], errors="coerce")
        if thr_col is not None:
            keep["threshold"] = pd.to_numeric(keep[thr_col], errors="coerce")
        elif bit == 2:
            keep["threshold"] = 5.0
        elif bit == 4:
            keep["threshold"] = 2.0
        elif bit == 32:
            keep["threshold"] = 0.3
        else:
            keep["threshold"] = np.nan
        keep["margin_to_threshold"] = pd.to_numeric(keep[margin_col], errors="coerce")

        cols = [
            "short_id", "bit", "source_id", "field_tag", "field_tag_meta", "phot_g_mean_mag",
            "metric_name", "metric_value", "threshold", "margin_to_threshold",
            "npz_relpath", "index_in_file",
        ]
        keep[cols].to_csv(out_dir / f"bit{bit}_near_threshold_keeps_id_map.csv", index=False)
        make_montage(
            keep,
            dataset_root=dataset_root,
            out_png=out_dir / f"bit{bit}_near_threshold_keeps.png",
            title=f"Bit {bit}: Near-threshold KEPT samples (short IDs)",
        )

    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
