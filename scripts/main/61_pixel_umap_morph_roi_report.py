#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _eps_from_knn(xy: np.ndarray, k: int = 15, q: float = 75.0) -> float:
    if len(xy) <= (k + 1):
        return 0.25
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(xy)
    d, _ = nn.kneighbors(xy)
    kth = d[:, -1]
    kth = kth[np.isfinite(kth)]
    if kth.size == 0:
        return 0.25
    return float(np.percentile(kth, q))


def _select_tail_rois(
    df: pd.DataFrame,
    metric_col: str,
    tail: str,
    tail_q: float,
    min_roi_size: int,
    n_rois_per_tail: int,
    seed: int,
) -> pd.DataFrame:
    vals = df[metric_col].to_numpy(dtype=float)
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return pd.DataFrame()
    q_low = float(np.quantile(v, tail_q))
    q_high = float(np.quantile(v, 1.0 - tail_q))
    if tail == "high":
        sel = df[metric_col] >= q_high
    else:
        sel = df[metric_col] <= q_low
    keep_cols = ["source_id", "x", "y", metric_col, "dataset_tag", "npz_file", "index_in_file", "field_tag"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    sub = df.loc[sel, keep_cols].copy()
    if len(sub) < max(2 * min_roi_size, 40):
        return pd.DataFrame()

    xy = sub[["x", "y"]].to_numpy(dtype=float)
    eps = _eps_from_knn(xy, k=15, q=75.0)
    labels = DBSCAN(eps=eps, min_samples=max(10, min_roi_size // 6)).fit_predict(xy)
    sub["roi_cluster"] = labels.astype(int)
    sub = sub[sub["roi_cluster"] >= 0].copy()
    if sub.empty:
        return pd.DataFrame()

    stats = (
        sub.groupby("roi_cluster", as_index=False)
        .agg(
            n=("source_id", "size"),
            mean_x=("x", "mean"),
            mean_y=("y", "mean"),
            mean_metric=(metric_col, "mean"),
        )
        .sort_values(["n", "mean_metric"], ascending=[False, tail != "high"])
        .reset_index(drop=True)
    )
    stats = stats[stats["n"] >= int(min_roi_size)].head(int(n_rois_per_tail)).copy()
    if stats.empty:
        return pd.DataFrame()

    keep = sub[sub["roi_cluster"].isin(stats["roi_cluster"])].copy()
    keep = keep.merge(stats, on="roi_cluster", how="left")
    keep["metric_col"] = metric_col
    keep["tail"] = tail
    keep["tail_q"] = float(tail_q)
    keep["dbscan_eps"] = float(eps)
    keep["roi_name"] = [
        f"{metric_col}_{tail}_roi{int(i)+1}" for i in pd.factorize(keep["roi_cluster"])[0]
    ]
    return keep


def plot_metric_panels(df: pd.DataFrame, metric_cols: List[str], out_png: Path) -> None:
    n = len(metric_cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 5.3 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for i in range(nrows * ncols):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off")
            continue
        col = metric_cols[i]
        sc = ax.scatter(df["x"], df["y"], c=df[col], s=3, cmap="viridis", alpha=0.70, linewidths=0)
        ax.set_title(col)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(col)
    fig.suptitle("Pixel-UMAP colored by morphology metrics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig(out_png)


def plot_field_map(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10.8, 8.6))
    tags = sorted(df["field_tag"].dropna().astype(str).unique().tolist())
    if len(tags) <= 20:
        cmap = plt.cm.get_cmap("tab20", max(1, len(tags)))
        for i, tag in enumerate(tags):
            sub = df[df["field_tag"] == tag]
            plt.scatter(sub["x"], sub["y"], s=3, alpha=0.65, color=cmap(i), linewidths=0, label=tag)
        plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=7)
    else:
        top = df["field_tag"].value_counts().head(12).index.tolist()
        cmap = plt.cm.get_cmap("tab20", max(1, len(top) + 1))
        for i, tag in enumerate(top):
            sub = df[df["field_tag"] == tag]
            plt.scatter(sub["x"], sub["y"], s=3, alpha=0.65, color=cmap(i), linewidths=0, label=tag)
        rest = df[~df["field_tag"].isin(top)]
        plt.scatter(rest["x"], rest["y"], s=3, alpha=0.28, color="#bdbdbd", linewidths=0, label="other fields")
        plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=7)
    plt.title("Pixel-UMAP colored by field")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    savefig(out_png)


def plot_doubles_map(df: pd.DataFrame, out_png: Path, double_col: str, double_label: str) -> None:
    plt.figure(figsize=(10.8, 8.6))
    plt.scatter(df["x"], df["y"], s=3, alpha=0.33, color="#4f5d75", linewidths=0, label="all")
    d = df[df[double_col]].copy()
    if len(d):
        plt.scatter(
            d["x"],
            d["y"],
            s=15,
            alpha=0.95,
            facecolors="none",
            edgecolors="#d62828",
            linewidths=0.5,
            label=double_label,
        )
    plt.title(f"Pixel-UMAP with overlay: {double_label}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best", frameon=True)
    savefig(out_png)


def plot_metric_with_roi(df: pd.DataFrame, roi_stats: pd.DataFrame, metric_col: str, out_png: Path) -> None:
    plt.figure(figsize=(10.8, 8.6))
    sc = plt.scatter(df["x"], df["y"], c=df[metric_col], s=3, cmap="viridis", alpha=0.62, linewidths=0)
    cb = plt.colorbar(sc)
    cb.set_label(metric_col)
    ax = plt.gca()
    colors = {"high": "#d62828", "low": "#1d3557"}
    for _, r in roi_stats[roi_stats["metric_col"] == metric_col].iterrows():
        cx = float(r["mean_x"])
        cy = float(r["mean_y"])
        rr = max(0.14, float(r["roi_radius"]))
        circ = plt.Circle((cx, cy), rr, fill=False, ec=colors.get(str(r["tail"]), "#111111"), lw=2.1, alpha=0.95)
        ax.add_patch(circ)
        ax.text(cx, cy, str(r["roi_name"]), fontsize=8, color=colors.get(str(r["tail"]), "#111111"))
    plt.title(f"Pixel-UMAP {metric_col} with high/low morphology ROIs")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    savefig(out_png)


def _resolve_stamp_rows(roi_rows: pd.DataFrame, dataset_root: Path) -> Tuple[np.ndarray, List[int]]:
    out_imgs: List[np.ndarray] = []
    out_sid: List[int] = []
    grouped = roi_rows.groupby(["dataset_tag", "npz_file"], sort=False)
    for (dataset_tag, npz_file), g in grouped:
        npz_path = dataset_root / str(dataset_tag) / str(npz_file)
        if not npz_path.exists():
            continue
        try:
            with np.load(npz_path, mmap_mode="r") as d:
                if "X" not in d:
                    continue
                x = d["X"]
                idxs = pd.to_numeric(g["index_in_file"], errors="coerce").dropna().astype(int).to_numpy()
                sid = pd.to_numeric(g["source_id"], errors="coerce").dropna().astype(np.int64).to_numpy()
                ok = (idxs >= 0) & (idxs < x.shape[0])
                idxs = idxs[ok]
                sid = sid[ok]
                for ii, ss in zip(idxs, sid):
                    out_imgs.append(np.array(x[int(ii)], dtype=np.float32))
                    out_sid.append(int(ss))
        except Exception:
            continue
    if not out_imgs:
        return np.empty((0, 21, 21), dtype=np.float32), []
    return np.stack(out_imgs, axis=0), out_sid


def save_montage(imgs: np.ndarray, sids: List[int], title: str, out_png: Path, n_cols: int = 8) -> None:
    n = int(len(imgs))
    if n <= 0:
        return
    cols = max(1, int(n_cols))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n:
            continue
        im = imgs[i]
        lo = float(np.nanpercentile(im, 1.0))
        hi = float(np.nanpercentile(im, 99.0))
        ax.imshow(im, cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="nearest")
        if i < len(sids):
            ax.set_title(str(sids[i]), fontsize=7)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding_csv", required=True, help="embedding_umap.csv from 34_embed_double_stars_pixels.py")
    ap.add_argument("--dataset_root", default="/data/yn316/Codes/output/dataset_npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--multipeak_labels_csv",
        default="",
        help="Optional labels_multipeak.csv to overlay true_multipeak_flag as doubles.",
    )
    ap.add_argument(
        "--morph_cols",
        default="morph_ellipticity_e2,morph_concentration_r80r20,morph_edge_asym_180,morph_asym_180,morph_peak_to_total",
    )
    ap.add_argument("--tail_q", type=float, default=0.10, help="Tail quantile for high/low ROI (e.g., 0.10 => top/bottom 10 percent).")
    ap.add_argument("--n_rois_per_tail", type=int, default=3)
    ap.add_argument("--min_roi_size", type=int, default=60)
    ap.add_argument("--n_samples_per_roi", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "roi_samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "roi_montages").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.embedding_csv), low_memory=False)
    need = ["source_id", "x", "y", "field_tag", "dataset_tag", "npz_file", "index_in_file", "is_double"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column in embedding CSV: {c}")
    for c in ["source_id", "x", "y", "index_in_file"]:
        df[c] = _safe_num(df[c])
    df["is_double"] = df["is_double"].astype(bool)
    df = df.dropna(subset=["source_id", "x", "y", "index_in_file"]).copy()
    df["source_id"] = df["source_id"].astype(np.int64)
    df["index_in_file"] = df["index_in_file"].astype(int)

    double_col = "is_double"
    double_label = "WDS doubles"
    if str(args.multipeak_labels_csv).strip():
        mp = pd.read_csv(Path(str(args.multipeak_labels_csv)), low_memory=False)
        if {"source_id", "true_multipeak_flag"}.issubset(set(mp.columns)):
            mp["source_id"] = _safe_num(mp["source_id"])
            mp["true_multipeak_flag"] = _safe_num(mp["true_multipeak_flag"])
            mp = mp.dropna(subset=["source_id", "true_multipeak_flag"]).copy()
            mp["source_id"] = mp["source_id"].astype(np.int64)
            mp["true_multipeak_flag"] = mp["true_multipeak_flag"].astype(int)
            mp = mp.drop_duplicates(subset=["source_id"], keep="first")
            df = df.merge(mp[["source_id", "true_multipeak_flag"]], on="source_id", how="left")
            df["true_multipeak_flag"] = df["true_multipeak_flag"].fillna(0).astype(int)
            df["true_multipeak_flag"] = df["true_multipeak_flag"] > 0
            double_col = "true_multipeak_flag"
            double_label = "Euclid true multipeak"

    morph_cols = [x.strip() for x in str(args.morph_cols).split(",") if x.strip()]
    morph_cols = [c for c in morph_cols if c in df.columns]
    if not morph_cols:
        raise RuntimeError("No requested morphology columns found in embedding CSV.")
    for c in morph_cols:
        df[c] = _safe_num(df[c])

    plot_metric_panels(df, morph_cols, out_dir / "umap_morphology_panels.png")
    plot_field_map(df, out_dir / "umap_colored_by_field.png")
    plot_doubles_map(df, out_dir / "umap_with_doubles_overlay.png", double_col=double_col, double_label=double_label)

    roi_rows_all: List[pd.DataFrame] = []
    roi_stats_rows: List[Dict[str, object]] = []
    for metric_col in morph_cols:
        for tail in ["high", "low"]:
            roi_points = _select_tail_rois(
                df=df,
                metric_col=metric_col,
                tail=tail,
                tail_q=float(args.tail_q),
                min_roi_size=int(args.min_roi_size),
                n_rois_per_tail=int(args.n_rois_per_tail),
                seed=int(args.seed),
            )
            if roi_points.empty:
                continue
            for roi_name, rr in roi_points.groupby("roi_name", sort=False):
                rr = rr.copy()
                rr = rr.merge(
                    df[["source_id", double_col, "phot_g_mean_mag"]].drop_duplicates(subset=["source_id"], keep="first"),
                    on="source_id",
                    how="left",
                )
                rr["roi_name"] = str(roi_name)
                rr["metric_col"] = metric_col
                rr["tail"] = tail
                roi_rows_all.append(rr)
                xy = rr[["x", "y"]].to_numpy(dtype=float)
                if len(xy) > 1:
                    cx, cy = float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1]))
                    rad = float(np.percentile(np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2), 85))
                else:
                    cx, cy, rad = float(xy[0, 0]), float(xy[0, 1]), 0.2
                roi_stats_rows.append(
                    {
                        "roi_name": str(roi_name),
                        "metric_col": metric_col,
                        "tail": tail,
                        "n_points": int(len(rr)),
                        "mean_x": cx,
                        "mean_y": cy,
                        "roi_radius": rad,
                        "mean_metric": float(np.nanmean(rr[metric_col].to_numpy(dtype=float))),
                        "double_rate": float(np.nanmean(rr[double_col].to_numpy(dtype=float))),
                        "mean_gmag": float(np.nanmean(pd.to_numeric(rr.get("phot_g_mean_mag", np.nan), errors="coerce"))),
                    }
                )

    if not roi_rows_all:
        raise RuntimeError("No morphology-tail ROIs found. Try lowering --min_roi_size or increasing --tail_q.")

    roi_df = pd.concat(roi_rows_all, ignore_index=True)
    roi_df = roi_df.drop_duplicates(subset=["source_id", "roi_name"]).reset_index(drop=True)
    roi_stats = pd.DataFrame(roi_stats_rows).drop_duplicates(subset=["roi_name"]).reset_index(drop=True)
    roi_df.to_csv(out_dir / "roi_points.csv", index=False)
    roi_stats.to_csv(out_dir / "roi_stats.csv", index=False)

    for metric_col in morph_cols:
        plot_metric_with_roi(df, roi_stats, metric_col, out_dir / f"umap_{metric_col}_with_rois.png")

    dataset_root = Path(args.dataset_root)
    seed_rows = []
    for _, r in roi_stats.sort_values(["metric_col", "tail", "n_points"], ascending=[True, True, False]).iterrows():
        roi_name = str(r["roi_name"])
        sub = roi_df[roi_df["roi_name"] == roi_name].copy()
        n_take = min(int(args.n_samples_per_roi), len(sub))
        if n_take <= 0:
            continue
        take_idx = rng.choice(np.arange(len(sub)), size=n_take, replace=False)
        picked = sub.iloc[take_idx].copy().reset_index(drop=True)
        picked.to_csv(out_dir / "roi_samples" / f"{roi_name}_samples.csv", index=False)
        imgs, sids = _resolve_stamp_rows(picked, dataset_root=dataset_root)
        save_montage(
            imgs=imgs,
            sids=sids,
            title=f"{roi_name}  n={len(sub)}  metric={str(r['metric_col'])} ({str(r['tail'])})",
            out_png=out_dir / "roi_montages" / f"{roi_name}_montage.png",
            n_cols=8,
        )
        for sid in pd.to_numeric(picked["source_id"], errors="coerce").dropna().astype(np.int64).tolist():
            seed_rows.append(
                {
                    "source_id": int(sid),
                    "roi_name": roi_name,
                    "metric_col": str(r["metric_col"]),
                    "tail": str(r["tail"]),
                }
            )

    pd.DataFrame(seed_rows).drop_duplicates().to_csv(out_dir / "roi_labeling_seed_list.csv", index=False)

    summary = {
        "embedding_csv": str(args.embedding_csv),
        "dataset_root": str(dataset_root),
        "n_points": int(len(df)),
        "morph_cols": morph_cols,
        "tail_q": float(args.tail_q),
        "n_rois_per_tail": int(args.n_rois_per_tail),
        "min_roi_size": int(args.min_roi_size),
        "n_rois_found": int(len(roi_stats)),
        "outputs": {
            "roi_points_csv": str(out_dir / "roi_points.csv"),
            "roi_stats_csv": str(out_dir / "roi_stats.csv"),
            "roi_samples_dir": str(out_dir / "roi_samples"),
            "roi_montages_dir": str(out_dir / "roi_montages"),
            "roi_labeling_seed_list_csv": str(out_dir / "roi_labeling_seed_list.csv"),
            "umap_morphology_panels_png": str(out_dir / "umap_morphology_panels.png"),
            "umap_colored_by_field_png": str(out_dir / "umap_colored_by_field.png"),
            "umap_with_doubles_overlay_png": str(out_dir / "umap_with_doubles_overlay.png"),
        },
        "double_overlay": {
            "double_col": str(double_col),
            "double_label": str(double_label),
            "double_rate_global": float(np.nanmean(df[double_col].to_numpy(dtype=float))),
            "n_double_global": int(np.sum(df[double_col].to_numpy(dtype=bool))),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
