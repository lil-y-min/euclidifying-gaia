#!/usr/bin/env python3
"""
Plot WDS separations for double-star candidates by discovered UMAP cluster.

Inputs:
- output/experiments/embeddings/double_stars_8d/umap_standard/double_candidates_ranked_by_local_enrichment_umap.csv

Outputs:
- output/experiments/embeddings/double_stars_8d/umap_standard/wds_separation_by_cluster/
- plots/qa/embeddings/double_stars_8d/wds_separation_by_cluster/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def load_input(csv_path: Path, include_noise: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"double_cluster", "wds_lstsep"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df["double_cluster"] = pd.to_numeric(df["double_cluster"], errors="coerce")
    df["wds_lstsep"] = pd.to_numeric(df["wds_lstsep"], errors="coerce")
    df = df.dropna(subset=["double_cluster", "wds_lstsep"]).copy()
    df["double_cluster"] = df["double_cluster"].astype(int)
    df = df[df["wds_lstsep"] > 0].copy()

    if not include_noise:
        df = df[df["double_cluster"] >= 0].copy()

    df["cluster_type"] = np.where(df["double_cluster"].isin([0, 1]), "A_0plus1", "B_2_or_other")
    return df


def summarize_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, sub in df.groupby("double_cluster", sort=True):
        vals = sub["wds_lstsep"].to_numpy(dtype=float)
        rows.append(
            {
                "double_cluster": int(c),
                "n": int(vals.size),
                "sep_mean_arcsec": float(np.mean(vals)),
                "sep_median_arcsec": float(np.median(vals)),
                "sep_p25_arcsec": float(np.percentile(vals, 25)),
                "sep_p75_arcsec": float(np.percentile(vals, 75)),
                "sep_min_arcsec": float(np.min(vals)),
                "sep_max_arcsec": float(np.max(vals)),
            }
        )
    return pd.DataFrame(rows).sort_values("double_cluster").reset_index(drop=True)


def summarize_by_type(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, sub in df.groupby("cluster_type", sort=False):
        vals = sub["wds_lstsep"].to_numpy(dtype=float)
        rows.append(
            {
                "cluster_type": t,
                "n": int(vals.size),
                "sep_mean_arcsec": float(np.mean(vals)),
                "sep_median_arcsec": float(np.median(vals)),
                "sep_p25_arcsec": float(np.percentile(vals, 25)),
                "sep_p75_arcsec": float(np.percentile(vals, 75)),
                "sep_min_arcsec": float(np.min(vals)),
                "sep_max_arcsec": float(np.max(vals)),
            }
        )
    return pd.DataFrame(rows)


def plot_box_by_cluster(df: pd.DataFrame, out_png: Path) -> None:
    clusters = sorted(df["double_cluster"].unique().tolist())
    data = [df.loc[df["double_cluster"] == c, "wds_lstsep"].to_numpy(dtype=float) for c in clusters]

    plt.figure(figsize=(8.4, 5.2))
    bp = plt.boxplot(data, tick_labels=[str(c) for c in clusters], patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set(facecolor="#90caf9", alpha=0.75)
    plt.ylabel("WDS LSTSEP [arcsec]")
    plt.xlabel("Double cluster ID")
    plt.title("WDS separations by UMAP double cluster")
    plt.yscale("log")
    savefig(out_png)


def plot_ecdf_by_cluster(df: pd.DataFrame, out_png: Path) -> None:
    clusters = sorted(df["double_cluster"].unique().tolist())
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(8.4, 5.2))
    for i, c in enumerate(clusters):
        vals = np.sort(df.loc[df["double_cluster"] == c, "wds_lstsep"].to_numpy(dtype=float))
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size + 1, dtype=float) / vals.size
        plt.step(vals, y, where="post", color=cmap(i % 10), linewidth=2.0, label=f"cluster {c} (n={vals.size})")
    plt.xscale("log")
    plt.ylim(0, 1.0)
    plt.xlabel("WDS LSTSEP [arcsec]")
    plt.ylabel("ECDF")
    plt.title("Cumulative separation distribution by cluster")
    plt.legend(frameon=True, fontsize=9, loc="lower right")
    savefig(out_png)


def plot_type_violin(df: pd.DataFrame, out_png: Path) -> None:
    order = ["A_0plus1", "B_2_or_other"]
    data = [df.loc[df["cluster_type"] == t, "wds_lstsep"].to_numpy(dtype=float) for t in order]

    plt.figure(figsize=(7.8, 5.0))
    vp = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    for body in vp["bodies"]:
        body.set_facecolor("#ffcc80")
        body.set_edgecolor("#e65100")
        body.set_alpha(0.75)
    plt.xticks([1, 2], order)
    plt.ylabel("WDS LSTSEP [arcsec]")
    plt.title("WDS separations by cluster type")
    plt.yscale("log")
    savefig(out_png)


def plot_umap_colored_by_sep(df: pd.DataFrame, out_png: Path) -> None:
    if ("x" not in df.columns) or ("y" not in df.columns):
        return

    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")
    sep = pd.to_numeric(df["wds_lstsep"], errors="coerce")
    ok = x.notna() & y.notna() & sep.notna() & (sep > 0)
    if ok.sum() == 0:
        return

    xv = x[ok].to_numpy(dtype=float)
    yv = y[ok].to_numpy(dtype=float)
    sv = sep[ok].to_numpy(dtype=float)
    log_sep = np.log10(sv)

    plt.figure(figsize=(9.0, 7.2))
    sc = plt.scatter(
        xv,
        yv,
        s=56,
        c=log_sep,
        cmap="viridis",
        alpha=0.95,
        edgecolors="white",
        linewidths=0.45,
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("log10(WDS LSTSEP [arcsec])")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP doubles colored by WDS separation")
    savefig(out_png)


def plot_umap_sep_panels_by_type(df: pd.DataFrame, out_png: Path) -> None:
    if ("x" not in df.columns) or ("y" not in df.columns):
        return

    d = df.copy()
    d["x"] = pd.to_numeric(d["x"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d["wds_lstsep"] = pd.to_numeric(d["wds_lstsep"], errors="coerce")
    d = d.dropna(subset=["x", "y", "wds_lstsep"]).copy()
    d = d[d["wds_lstsep"] > 0].copy()
    if d.empty:
        return

    d["cluster_type"] = np.where(d["double_cluster"].isin([0, 1]), "A_0plus1", "B_2_or_other")
    d["log_sep"] = np.log10(d["wds_lstsep"].to_numpy(dtype=float))

    vmin = float(np.nanpercentile(d["log_sep"], 2))
    vmax = float(np.nanpercentile(d["log_sep"], 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharex=True, sharey=True)
    order = ["A_0plus1", "B_2_or_other"]
    last_sc = None
    for ax, typ in zip(axes, order):
        sub = d[d["cluster_type"] == typ].copy()
        ax.scatter(d["x"], d["y"], s=8, c="0.88", alpha=0.28, linewidths=0)
        if not sub.empty:
            last_sc = ax.scatter(
                sub["x"],
                sub["y"],
                s=62,
                c=sub["log_sep"],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                alpha=0.96,
                edgecolors="white",
                linewidths=0.45,
            )
        ax.set_title(f"{typ} (n={len(sub)})")
        ax.set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label("log10(WDS LSTSEP [arcsec])")
    fig.suptitle("UMAP doubles: WDS separation by cluster type", fontsize=13)
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot WDS separation distribution by double cluster.")
    ap.add_argument(
        "--input_csv",
        default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/double_candidates_ranked_by_local_enrichment_umap.csv",
    )
    ap.add_argument(
        "--out_tab_dir",
        default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/wds_separation_by_cluster",
    )
    ap.add_argument(
        "--out_plot_dir",
        default="/data/yn316/Codes/plots/qa/embeddings/double_stars_8d/wds_separation_by_cluster",
    )
    ap.add_argument("--include_noise_cluster", action="store_true", help="Include DBSCAN noise cluster (-1).")
    args = ap.parse_args()

    out_tab_dir = ensure_dir(Path(args.out_tab_dir))
    out_plot_dir = ensure_dir(Path(args.out_plot_dir))

    df = load_input(Path(args.input_csv), include_noise=bool(args.include_noise_cluster))
    if df.empty:
        raise RuntimeError("No rows available after filtering. Check input CSV and filters.")

    by_cluster = summarize_by_cluster(df)
    by_type = summarize_by_type(df)
    by_cluster.to_csv(out_tab_dir / "wds_separation_summary_by_cluster.csv", index=False)
    by_type.to_csv(out_tab_dir / "wds_separation_summary_by_cluster_type.csv", index=False)

    plot_box_by_cluster(df, out_plot_dir / "01_wds_separation_boxplot_by_cluster.png")
    plot_ecdf_by_cluster(df, out_plot_dir / "02_wds_separation_ecdf_by_cluster.png")
    plot_type_violin(df, out_plot_dir / "03_wds_separation_violin_by_cluster_type.png")
    plot_umap_colored_by_sep(df, out_plot_dir / "04_umap_doubles_colored_by_wds_separation.png")
    plot_umap_sep_panels_by_type(df, out_plot_dir / "05_umap_doubles_wds_separation_panels_by_type.png")

    print("Done.")
    print("  Input:", Path(args.input_csv))
    print("  Rows used:", len(df))
    print("  Summary by cluster:", out_tab_dir / "wds_separation_summary_by_cluster.csv")
    print("  Summary by type   :", out_tab_dir / "wds_separation_summary_by_cluster_type.csv")
    print("  Plots             :", out_plot_dir)


if __name__ == "__main__":
    main()
