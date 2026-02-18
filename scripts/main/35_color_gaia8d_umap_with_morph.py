#!/usr/bin/env python3
"""
Color an existing UMAP embedding with morphology metrics computed from cutout pixels.

Reads:
- output/experiments/embeddings/*/embedding_umap.csv
- output/dataset_npz/*/metadata.csv + NPZ files

Writes:
- configurable output roots, defaulting to:
  - plots/qa/embeddings/double_stars_8d_morph/<tag>/
  - output/experiments/embeddings/double_stars_8d_morph/<tag>/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


MORPH_PLOT_COLS = [
    "morph_asym_180",
    "morph_mirror_asym_lr",
    "morph_concentration_r80r20",
    "morph_ellipticity_e2",
    "morph_peakedness_kurtosis",
    "morph_roundness",
    "morph_texture_laplacian",
    "morph_gini",
    "morph_m20",
    "morph_smoothness",
    "morph_edge_asym_180",
]

MORPH_LABELS = {
    "morph_asym_180": "Asymmetry 180",
    "morph_mirror_asym_lr": "Mirror Asymmetry",
    "morph_concentration_r80r20": "Flux Ratio / Concentration",
    "morph_ellipticity_e2": "Ellipticity",
    "morph_peakedness_kurtosis": "Kurtosis",
    "morph_roundness": "Roundness",
    "morph_texture_laplacian": "High-Frequency Artifacts",
    "morph_gini": "Gini Coefficient",
    "morph_m20": "M20",
    "morph_smoothness": "Smoothness",
    "morph_edge_asym_180": "Edge Asymmetry",
}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def load_morph_module(base: Path):
    p = base / "scripts" / "34_embed_double_stars_pixels.py"
    spec = importlib.util.spec_from_file_location("pix_morph", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_embedding(csv_path: Path) -> pd.DataFrame:
    d = pd.read_csv(csv_path, low_memory=False)
    for c in ["source_id", "x", "y"]:
        if c not in d.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    d["source_id"] = pd.to_numeric(d["source_id"], errors="coerce")
    d = d.dropna(subset=["source_id", "x", "y"]).copy()
    d["source_id"] = d["source_id"].astype(np.int64)
    d = d.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return d


def load_source_index(dataset_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for meta in sorted(dataset_root.glob("*/metadata.csv")):
        tag = meta.parent.name
        use = ["source_id", "npz_file", "index_in_file", "phot_g_mean_mag", "field_tag", "fits_path"]
        df = pd.read_csv(meta, usecols=lambda c: c in set(use), low_memory=False)
        for c in ["source_id", "index_in_file", "phot_g_mean_mag"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        m = df["source_id"].notna() & df["npz_file"].notna() & df["index_in_file"].notna()
        df = df.loc[m].copy()
        df["source_id"] = df["source_id"].astype(np.int64)
        df["dataset_tag"] = tag
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No metadata.csv under {dataset_root}")
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    keep = [c for c in ["source_id", "dataset_tag", "npz_file", "index_in_file", "phot_g_mean_mag", "field_tag"] if c in out.columns]
    return out[keep]


def compute_morph_for_embedding(df: pd.DataFrame, dataset_root: Path, mod, eps: float) -> pd.DataFrame:
    need = ["source_id", "dataset_tag", "npz_file", "index_in_file"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' after merge.")

    out = {c: np.full(len(df), np.nan, dtype=np.float32) for c in MORPH_PLOT_COLS}

    grouped_field = list(df.groupby("dataset_tag", sort=True))
    for fi, (tag, dft) in enumerate(grouped_field, start=1):
        print(f"[MORPH] field {fi}/{len(grouped_field)}: {tag}")
        field_dir = dataset_root / str(tag)
        for npz_file, sub in dft.groupby("npz_file", sort=False):
            npz_path = field_dir / str(npz_file)
            if not npz_path.exists():
                continue
            with np.load(npz_path) as dnpz:
                if "X" not in dnpz:
                    continue
                Xall = dnpz["X"]

            idx = sub["index_in_file"].to_numpy(dtype=int)
            ok = (idx >= 0) & (idx < Xall.shape[0])
            if not np.any(ok):
                continue

            pos = sub.index.to_numpy()[ok]
            idx2 = idx[ok]
            stamps = Xall[idx2].astype(np.float32, copy=False)
            # Normalize with same default as pixel UMAP script.
            stamps_n, okn = mod.normalize_stamps(stamps, mode="integral_pos", eps=eps)
            if not np.any(okn):
                continue

            pos2 = pos[okn]
            feats: Dict[str, np.ndarray] = mod.compute_morphology_features(stamps_n[okn], eps=eps)
            for c in MORPH_PLOT_COLS:
                if c in feats:
                    out[c][pos2] = feats[c]

    for c in MORPH_PLOT_COLS:
        df[c] = out[c]
    return df


def plot_morph_panels(df: pd.DataFrame, out_png: Path) -> None:
    cols = [c for c in MORPH_PLOT_COLS if c in df.columns]
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, c in zip(axes, cols):
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(vals)
        if np.any(m):
            lo, hi = np.nanpercentile(vals[m], [1, 99])
            vals = np.clip(vals, lo, hi)
        sc = ax.scatter(df["x"], df["y"], c=vals, s=5, cmap="viridis", alpha=0.75, linewidths=0)
        if "is_double" in df.columns:
            d = df[df["is_double"] == True]  # noqa: E712
            ax.scatter(d["x"], d["y"], s=12, facecolors="none", edgecolors="#f94144", linewidths=0.35)
        ax.set_title(MORPH_LABELS.get(c, c), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Gaia-8D UMAP color-coded by morphology metrics", fontsize=13)
    savefig(out_png)


def load_one_stamp(dataset_root: Path, dataset_tag: str, npz_file: str, index_in_file: int) -> np.ndarray | None:
    npz_path = dataset_root / str(dataset_tag) / str(npz_file)
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path) as dnpz:
            if "X" not in dnpz:
                return None
            Xall = dnpz["X"]
            idx = int(index_in_file)
            if idx < 0 or idx >= Xall.shape[0]:
                return None
            return Xall[idx].astype(np.float32)
    except Exception:
        return None


def _center_crop(img: np.ndarray, frac: float = 0.7) -> np.ndarray:
    ny, nx = img.shape
    cy = ny // 2
    cx = nx // 2
    hy = max(3, int(round(0.5 * frac * ny)))
    hx = max(3, int(round(0.5 * frac * nx)))
    y0 = max(0, cy - hy)
    y1 = min(ny, cy + hy)
    x0 = max(0, cx - hx)
    x1 = min(nx, cx + hx)
    return img[y0:y1, x0:x1]


def _stamp_to_display(stamp: np.ndarray) -> np.ndarray:
    """
    Build a human-readable stamp view.
    Use robust background subtraction + asinh stretch to avoid binary-looking panels.
    """
    img = np.asarray(stamp, dtype=np.float32)
    ny, nx = img.shape
    y, x = np.mgrid[0:ny, 0:nx]
    x0 = 0.5 * (nx - 1.0)
    y0 = 0.5 * (ny - 1.0)
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    Rmax = max(2.0, 0.5 * min(nx, ny) - 0.5)
    ann = (r2 >= (0.7 * Rmax) ** 2) & (r2 <= (Rmax) ** 2)
    vals = img[ann] if np.any(ann) else img.ravel()
    bkg = float(np.median(vals))
    mad = float(np.median(np.abs(vals - bkg)))
    sigma = 1.4826 * mad
    img_b = img - bkg
    scale = max(3.0 * sigma, float(np.std(img_b)), 1e-6)
    disp = np.arcsinh(img_b / scale).astype(np.float32)
    # Show core morphology; this avoids edge bars dominating the tiny thumbnails.
    return _center_crop(disp, frac=0.7)


def plot_metric_extremes_stamps(
    df: pd.DataFrame,
    dataset_root: Path,
    mod,
    metric: str,
    out_png: Path,
    n_each: int = 5,
    candidate_mask: np.ndarray | None = None,
) -> None:
    vals = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(vals)
    if candidate_mask is not None:
        finite &= np.asarray(candidate_mask, dtype=bool)
    finite_idx = np.where(finite)[0]
    if finite_idx.size < (2 * n_each):
        return

    ord_idx = finite_idx[np.argsort(vals[finite_idx])]
    low_idx = ord_idx[:n_each]
    high_idx = ord_idx[-n_each:][::-1]

    fig, axes = plt.subplots(2, n_each, figsize=(2.5 * n_each, 5.2))
    axes = np.atleast_2d(axes)

    groups = [("Low", low_idx), ("High", high_idx)]
    all_disp = []
    for _, idxs in groups:
        for ii in idxs:
            row = df.iloc[int(ii)]
            stamp = load_one_stamp(
                dataset_root=dataset_root,
                dataset_tag=str(row["dataset_tag"]),
                npz_file=str(row["npz_file"]),
                index_in_file=int(row["index_in_file"]),
            )
            if stamp is None:
                continue
            all_disp.append(_stamp_to_display(stamp))
    if all_disp:
        vv = np.concatenate([a.ravel() for a in all_disp])
        vv = vv[np.isfinite(vv)]
        if vv.size:
            gvmin, gvmax = np.percentile(vv, [5, 99.5])
            if gvmax <= gvmin:
                gvmax = gvmin + 1e-6
        else:
            gvmin, gvmax = -1.0, 2.0
    else:
        gvmin, gvmax = -1.0, 2.0

    for r, (gname, idxs) in enumerate(groups):
        for c, ii in enumerate(idxs):
            ax = axes[r, c]
            row = df.iloc[int(ii)]
            stamp = load_one_stamp(
                dataset_root=dataset_root,
                dataset_tag=str(row["dataset_tag"]),
                npz_file=str(row["npz_file"]),
                index_in_file=int(row["index_in_file"]),
            )
            if stamp is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.axis("off")
                continue
            sd = _stamp_to_display(stamp)
            ax.imshow(
                sd,
                cmap="magma",
                origin="lower",
                interpolation="nearest",
                norm=Normalize(vmin=gvmin, vmax=gvmax),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            sid = int(row["source_id"]) if np.isfinite(row["source_id"]) else -1
            val = float(row[metric]) if np.isfinite(row[metric]) else np.nan
            ax.set_title(f"{gname} #{c+1}\nID {sid}\n{val:.3g}", fontsize=8)

    pretty = MORPH_LABELS.get(metric, metric)
    fig.suptitle(f"{pretty}: lowest (top) vs highest (bottom) stamps", fontsize=12)
    savefig(out_png, dpi=180)


def build_star_like_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Conservative quality mask to reduce obviously weird/non-stellar cutouts
    when selecting extreme examples.
    """
    n = len(df)
    m = np.ones(n, dtype=bool)

    def col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.full(n, np.nan, dtype=float)
        raw = pd.to_numeric(df[name], errors="coerce")
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=float)
        arr = arr.ravel()
        if arr.size == n:
            return arr
        if arr.size > n:
            return arr[:n]
        out = np.full(n, np.nan, dtype=float)
        out[:arr.size] = arr
        return out

    # Require enough finite morphology information in the saved table.
    fin_count = np.zeros(n, dtype=int)
    for c in MORPH_PLOT_COLS:
        if c in df.columns:
            fin_count += np.isfinite(col(c)).astype(int)
    m &= fin_count >= max(5, min(7, len(MORPH_PLOT_COLS)))
    if not np.any(m):
        return m

    def apply_quantile_band(name: str, qlo: float, qhi: float) -> None:
        nonlocal m
        if name not in df.columns:
            return
        v = col(name)
        ok = m & np.isfinite(v)
        if np.sum(ok) < 100:
            return
        lo, hi = np.nanpercentile(v[ok], [qlo, qhi])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        m &= np.isfinite(v) & (v >= float(lo)) & (v <= float(hi))

    # Conservative adaptive cuts using only persisted columns.
    apply_quantile_band("morph_concentration_r80r20", 2, 98)
    apply_quantile_band("morph_gini", 2, 98)
    apply_quantile_band("morph_roundness", 2, 98)
    apply_quantile_band("morph_peakedness_kurtosis", 1, 99)
    apply_quantile_band("morph_smoothness", 1, 98)
    apply_quantile_band("morph_edge_asym_180", 1, 95)

    if "phot_g_mean_mag" in df.columns:
        g = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy(dtype=float)
        m &= np.isfinite(g)

    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Color UMAP with morphology metrics from cutout pixels.")
    ap.add_argument("--embedding_csv", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/embedding_umap.csv")
    ap.add_argument("--dataset_root", default="/data/yn316/Codes/output/dataset_npz")
    ap.add_argument(
        "--embedding_name",
        default=None,
        help="Label used in output folder naming. Default: parent folder name of embedding CSV.",
    )
    ap.add_argument(
        "--run_label",
        default=None,
        help="Optional extra folder level under output roots (for grouping multiple embeddings).",
    )
    ap.add_argument(
        "--out_plot_root",
        default=None,
        help="Optional override for plot output root.",
    )
    ap.add_argument(
        "--out_tab_root",
        default=None,
        help="Optional override for table output root.",
    )
    ap.add_argument(
        "--input_with_morph_csv",
        default=None,
        help="If set, reuse an existing embedding_umap_with_morph.csv and skip morphology recomputation.",
    )
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--max_points", type=int, default=None, help="Optional subsample for faster debug.")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    embedding_csv = Path(args.embedding_csv)
    dataset_root = Path(args.dataset_root)
    tag = str(args.embedding_name) if args.embedding_name else embedding_csv.parent.name
    run_label = str(args.run_label) if args.run_label else None

    plot_root = Path(args.out_plot_root) if args.out_plot_root else (base / "plots" / "qa" / "embeddings" / "double_stars_8d_morph")
    tab_root = Path(args.out_tab_root) if args.out_tab_root else (base / "output" / "experiments" / "embeddings" / "double_stars_8d_morph")
    out_plot = ensure_dir(plot_root / run_label / tag) if run_label else ensure_dir(plot_root / tag)
    out_tab = ensure_dir(tab_root / run_label / tag) if run_label else ensure_dir(tab_root / tag)

    print("\n=== Gaia8D + Morph Overlay ===")
    print("embedding_csv:", embedding_csv)
    print("embedding_name:", tag)
    if run_label:
        print("run_label    :", run_label)
    print("dataset_root :", dataset_root)
    print("out_plot     :", out_plot)
    print("out_tab      :", out_tab)

    mod = load_morph_module(base)
    if args.input_with_morph_csv:
        in_csv = Path(args.input_with_morph_csv)
        print("input_with_morph_csv:", in_csv)
        df = pd.read_csv(in_csv, low_memory=False)
        before = len(df)
        need = ["source_id", "dataset_tag", "npz_file", "index_in_file", "x", "y"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"input_with_morph_csv missing columns: {miss}")
        df["index_in_file"] = pd.to_numeric(df["index_in_file"], errors="coerce")
        df = df.dropna(subset=["index_in_file"]).copy()
        df["index_in_file"] = df["index_in_file"].astype(int)
        print(f"[REUSE] loaded {len(df):,}/{before:,} rows from existing morph table")
    else:
        emb = load_embedding(embedding_csv)
        if args.max_points is not None and len(emb) > int(args.max_points):
            emb = emb.sample(n=int(args.max_points), random_state=int(args.seed)).reset_index(drop=True)
            print(f"[SAMPLE] using {len(emb):,} points")

        idx = load_source_index(dataset_root)
        df = emb.merge(idx, on="source_id", how="left")
        before = len(df)
        df = df.dropna(subset=["dataset_tag", "npz_file", "index_in_file"]).copy()
        df["index_in_file"] = pd.to_numeric(df["index_in_file"], errors="coerce").astype(int)
        print(f"[JOIN] matched sources with pixel index: {len(df):,}/{before:,}")

        df = compute_morph_for_embedding(df, dataset_root=dataset_root, mod=mod, eps=float(args.eps))

    plot_morph_panels(df, out_plot / "01_umap_morphology_panels.png")
    out_ext = ensure_dir(out_plot / "extremes_stamps")
    for metric in MORPH_PLOT_COLS:
        plot_metric_extremes_stamps(
            df=df,
            dataset_root=dataset_root,
            mod=mod,
            metric=metric,
            out_png=out_ext / f"{metric}_low5_high5_stamps.png",
            n_each=5,
            candidate_mask=None,
        )
    out_csv = out_tab / "embedding_umap_with_morph.csv"
    df.to_csv(out_csv, index=False)

    summary = {
        "embedding_csv": str(embedding_csv),
        "n_input": int(before),
        "n_output": int(len(df)),
        "morph_cols": MORPH_PLOT_COLS,
        "plots": [str(out_plot / "01_umap_morphology_panels.png")],
        "extreme_stamp_dir": str(out_ext),
        "table": str(out_csv),
    }
    with open(out_tab / "summary_morph_overlay.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE")
    print("Plot :", out_plot / "01_umap_morphology_panels.png")
    print("Table:", out_csv)
    print("Meta :", out_tab / "summary_morph_overlay.json")


if __name__ == "__main__":
    main()
