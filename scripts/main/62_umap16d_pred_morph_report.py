#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRED_METRICS = [
    "m_concentration_r2_r6_pred",
    "m_asymmetry_180_pred",
    "m_ellipticity_pred",
    "m_peak_sep_pix_pred",
    "m_edge_flux_frac_pred",
]


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def plot_pred_metric_panels(df: pd.DataFrame, metric_cols: List[str], out_png: Path) -> None:
    n = len(metric_cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.2, 5.1 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for i in range(nrows * ncols):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off")
            continue
        c = metric_cols[i]
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(vals)
        if np.any(m):
            lo, hi = np.nanpercentile(vals[m], [1, 99])
            vals = np.clip(vals, lo, hi)
        sc = ax.scatter(df["x"], df["y"], c=vals, s=4, cmap="viridis", alpha=0.70, linewidths=0)
        d = df[df["is_double"] == True]  # noqa: E712
        if len(d):
            ax.scatter(d["x"], d["y"], s=12, facecolors="none", edgecolors="#d62828", linewidths=0.45)
        ax.set_title(c)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(c)
    fig.suptitle("16D Gaia UMAP colored by predicted morphology metrics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig(out_png)


def plot_field_colored(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10.6, 8.5))
    tags = sorted(df["field_tag"].dropna().astype(str).unique().tolist())
    if len(tags) <= 20:
        cmap = plt.get_cmap("tab20", max(1, len(tags)))
        for i, t in enumerate(tags):
            sub = df[df["field_tag"] == t]
            plt.scatter(sub["x"], sub["y"], s=4, alpha=0.66, color=cmap(i), linewidths=0, label=t)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=7)
    else:
        top = df["field_tag"].value_counts().head(12).index.tolist()
        cmap = plt.get_cmap("tab20", max(1, len(top) + 1))
        for i, t in enumerate(top):
            sub = df[df["field_tag"] == t]
            plt.scatter(sub["x"], sub["y"], s=4, alpha=0.66, color=cmap(i), linewidths=0, label=t)
        rest = df[~df["field_tag"].isin(top)]
        plt.scatter(rest["x"], rest["y"], s=4, alpha=0.24, color="#bdbdbd", linewidths=0, label="other fields")
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=7)
    d = df[df["is_double"] == True]  # noqa: E712
    if len(d):
        plt.scatter(d["x"], d["y"], s=12, facecolors="none", edgecolors="black", linewidths=0.45, label="WDS doubles")
    plt.title("16D Gaia UMAP colored by field (WDS doubles overlaid)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    savefig(out_png)


def plot_wds_doubles(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10.6, 8.5))
    plt.scatter(df["x"], df["y"], s=4, alpha=0.30, color="#4f5d75", linewidths=0, label="all")
    d = df[df["is_double"] == True]  # noqa: E712
    if len(d):
        plt.scatter(d["x"], d["y"], s=16, facecolors="none", edgecolors="#d62828", linewidths=0.55, label="WDS doubles")
    plt.title("16D Gaia UMAP with WDS doubles overlay")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best", frameon=True)
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding_csv", required=True, help="16D embedding_umap.csv with x,y,field_tag,is_double.")
    ap.add_argument("--pred_morph_csv", required=True, help="morph_pred_vs_true_rows.csv from script 50.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = pd.read_csv(Path(args.embedding_csv), low_memory=False)
    need_emb = ["source_id", "x", "y", "field_tag", "is_double"]
    for c in need_emb:
        if c not in emb.columns:
            raise RuntimeError(f"Missing required embedding column: {c}")
    emb["source_id"] = _safe_num(emb["source_id"])
    emb["x"] = _safe_num(emb["x"])
    emb["y"] = _safe_num(emb["y"])
    emb = emb.dropna(subset=["source_id", "x", "y"]).copy()
    emb["source_id"] = emb["source_id"].astype(np.int64)
    emb["is_double"] = emb["is_double"].astype(bool)
    emb = emb.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)

    pm = pd.read_csv(Path(args.pred_morph_csv), low_memory=False)
    need_pm = ["source_id", *PRED_METRICS]
    miss = [c for c in need_pm if c not in pm.columns]
    if miss:
        raise RuntimeError(f"pred_morph_csv missing columns: {miss}")
    pm["source_id"] = _safe_num(pm["source_id"])
    for c in PRED_METRICS:
        pm[c] = _safe_num(pm[c])
    pm = pm.dropna(subset=["source_id"]).copy()
    pm["source_id"] = pm["source_id"].astype(np.int64)
    pm = pm.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)

    df = emb.merge(pm[["source_id", *PRED_METRICS]], on="source_id", how="left")

    plot_pred_metric_panels(df, PRED_METRICS, out_dir / "umap16d_pred_morph_panels.png")
    plot_field_colored(df, out_dir / "umap16d_color_by_field_with_wds_doubles.png")
    plot_wds_doubles(df, out_dir / "umap16d_wds_doubles_overlay.png")

    coverage = float(np.mean(np.isfinite(pd.to_numeric(df["m_ellipticity_pred"], errors="coerce").to_numpy(dtype=float))))
    summary = {
        "embedding_csv": str(args.embedding_csv),
        "pred_morph_csv": str(args.pred_morph_csv),
        "n_embedding_points": int(len(df)),
        "n_wds_doubles": int(np.sum(df["is_double"].to_numpy(dtype=bool))),
        "pred_morph_coverage_fraction": coverage,
        "outputs": {
            "pred_morph_panels_png": str(out_dir / "umap16d_pred_morph_panels.png"),
            "field_png": str(out_dir / "umap16d_color_by_field_with_wds_doubles.png"),
            "wds_doubles_png": str(out_dir / "umap16d_wds_doubles_overlay.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv(out_dir / "umap16d_with_pred_morph.csv", index=False)
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
