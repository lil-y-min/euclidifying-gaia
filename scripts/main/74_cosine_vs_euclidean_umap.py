#!/usr/bin/env python3
"""
Cosine vs Euclidean pixel UMAP comparison.

Both runs use identical settings (standard scaling, integral_pos normalisation,
n_neighbours=15, min_dist=0.1, 240k+ points) — only the UMAP metric differs.

For normalised stamps (sum=1) cosine similarity measures shape similarity
independent of scale, while euclidean is sensitive to the full flux pattern.

Outputs: report/model_decision/2026-03-05_umap_improvements/03_cosine_pixel_umap/
"""
from __future__ import annotations
import json, shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE    = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "report" / "model_decision" / "2026-03-05_umap_improvements" / "03_cosine_pixel_umap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_EUC = BASE / "output" / "experiments" / "embeddings" / "double_stars_pixels" / \
    "email_pixels_all_filtered" / "umap_standard" / "embedding_umap.csv"
CSV_COS = BASE / "output" / "experiments" / "embeddings" / "double_stars_pixels" / \
    "cosine_all_filtered" / "umap_standard" / "embedding_umap.csv"

PLOTS_EUC = BASE / "plots" / "qa" / "embeddings" / "double_stars_pixels" / \
    "email_pixels_all_filtered" / "umap_standard"
PLOTS_COS = BASE / "plots" / "qa" / "embeddings" / "double_stars_pixels" / \
    "cosine_all_filtered" / "umap_standard"

MORPH_COLS = [
    "morph_asym_180", "morph_concentration_r80r20", "morph_ellipticity_e2",
    "morph_peakedness_kurtosis", "morph_roundness", "morph_gini",
    "morph_m20", "morph_smoothness", "morph_edge_asym_180",
    "morph_peak_to_total", "morph_texture_laplacian",
]
MORPH_LABELS = {
    "morph_asym_180": "Asymmetry 180", "morph_concentration_r80r20": "Concentration r80/r20",
    "morph_ellipticity_e2": "Ellipticity e2", "morph_peakedness_kurtosis": "Kurtosis",
    "morph_roundness": "Roundness", "morph_gini": "Gini", "morph_m20": "M20",
    "morph_smoothness": "Smoothness", "morph_edge_asym_180": "Edge Asymmetry",
    "morph_peak_to_total": "Peak/Total", "morph_texture_laplacian": "Laplacian Texture",
}
FEAT_COLS = [
    "feat_log10_snr", "feat_ruwe", "feat_astrometric_excess_noise",
    "feat_ipd_frac_multi_peak", "feat_c_star", "feat_pm_significance",
]
FEAT_LABELS = {
    "feat_log10_snr": "log10 SNR", "feat_ruwe": "RUWE",
    "feat_astrometric_excess_noise": "Astrometric Excess Noise",
    "feat_ipd_frac_multi_peak": "IPD Frac Multi-Peak",
    "feat_c_star": "C*", "feat_pm_significance": "PM Significance",
}


def savefig(path: Path, dpi: int = 200) -> None:
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, low_memory=False)
    return df.dropna(subset=["x", "y"]).copy()


def make_field_colormap(df_a, df_b):
    all_tags = sorted(set(
        df_a["field_tag"].dropna().astype(str).tolist() +
        df_b["field_tag"].dropna().astype(str).tolist()
    ))
    cmap = plt.cm.get_cmap("tab20", max(len(all_tags), 1))
    return {t: cmap(i) for i, t in enumerate(all_tags)}, all_tags


def plot_field_side_by_side(df_euc, df_cos, field_cmap, all_tags, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, df, title in zip(axes, [df_euc, df_cos],
                              ["Euclidean metric  (20×20)", "Cosine metric  (20×20)"]):
        top_tags = df["field_tag"].value_counts().head(17).index.tolist()
        for tag in sorted(df["field_tag"].dropna().astype(str).unique()):
            sub = df[df["field_tag"] == tag]
            color = field_cmap.get(tag, "#aaaaaa")
            kw = dict(s=2, linewidths=0, rasterized=True, alpha=0.55)
            if tag in top_tags:
                ax.scatter(sub["x"], sub["y"], color=color, label=tag, **kw)
            else:
                ax.scatter(sub["x"], sub["y"], color="#cccccc", alpha=0.2, **kw)
        ax.set_title(f"Pixel UMAP — {title}\nn={len(df):,}", fontsize=12)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([0],[0], marker='o', color='w',
                           markerfacecolor=field_cmap.get(t,"#aaa"), markersize=6, label=t)
               for t in all_tags]
    fig.legend(handles=handles[:20], loc="center right", bbox_to_anchor=(1.01, 0.5),
               frameon=True, fontsize=7)
    fig.suptitle("Euclidean vs Cosine Pixel UMAP — Field Clustering", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    savefig(out_png)


def plot_continuous_side_by_side(df_euc, df_cos, col, label, out_png, cmap="viridis"):
    vals_all = pd.concat([
        pd.to_numeric(df_euc[col], errors="coerce"),
        pd.to_numeric(df_cos[col], errors="coerce"),
    ]).dropna()
    if len(vals_all) < 10:
        print(f"  [SKIP] {col} — no data"); return
    vmin, vmax = float(np.percentile(vals_all, 1)), float(np.percentile(vals_all, 99))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, df, title in zip(axes, [df_euc, df_cos],
                              ["Euclidean metric", "Cosine metric"]):
        vals = np.clip(pd.to_numeric(df[col], errors="coerce").to_numpy(float), vmin, vmax)
        sc = ax.scatter(df["x"], df["y"], c=vals, s=2, cmap=cmap, alpha=0.65,
                        linewidths=0, vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(f"{title}  n={len(df):,}", fontsize=12)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label=label)
    fig.suptitle(f"Euclidean vs Cosine — {label} (shared scale)", fontsize=13)
    savefig(out_png)


def plot_morph_panels_side_by_side(df_euc, df_cos, cols, labels, out_png):
    cols_ok = [c for c in cols if c in df_euc.columns and c in df_cos.columns]
    n = len(cols_ok)
    if not n: return
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1: axes = axes[np.newaxis, :]
    for i, col in enumerate(cols_ok):
        vals_all = pd.concat([
            pd.to_numeric(df_euc[col], errors="coerce"),
            pd.to_numeric(df_cos[col], errors="coerce"),
        ]).dropna()
        if len(vals_all) < 10: continue
        vmin, vmax = float(np.percentile(vals_all, 1)), float(np.percentile(vals_all, 99))
        for j, (ax, df) in enumerate(zip(axes[i], [df_euc, df_cos])):
            vals = np.clip(pd.to_numeric(df[col], errors="coerce").to_numpy(float), vmin, vmax)
            sc = ax.scatter(df["x"], df["y"], c=vals, s=2, cmap="viridis", alpha=0.6,
                            linewidths=0, vmin=vmin, vmax=vmax, rasterized=True)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
            title = ("Euclidean" if j == 0 else "Cosine") + f" — {labels.get(col, col)}"
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Euclidean vs Cosine — morphology panels", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig(out_png)


def main():
    print("[1/4] Loading embeddings...")
    df_euc = load(CSV_EUC)
    df_cos = load(CSV_COS)
    print(f"  Euclidean: {len(df_euc):,} rows")
    print(f"  Cosine:    {len(df_cos):,} rows")

    field_cmap, all_tags = make_field_colormap(df_euc, df_cos)

    print("[2/4] Side-by-side field comparison...")
    plot_field_side_by_side(df_euc, df_cos, field_cmap, all_tags,
                            OUT_DIR / "01_field_euclidean_vs_cosine.png")

    print("[3/4] Continuous-color comparisons...")
    plot_continuous_side_by_side(df_euc, df_cos, "phot_g_mean_mag", "G magnitude",
                                 OUT_DIR / "02_gmag_euclidean_vs_cosine.png", cmap="magma_r")
    plot_continuous_side_by_side(df_euc, df_cos, "morph_asym_180", "Asymmetry 180",
                                 OUT_DIR / "03_asym_euclidean_vs_cosine.png")
    plot_continuous_side_by_side(df_euc, df_cos, "morph_ellipticity_e2", "Ellipticity e2",
                                 OUT_DIR / "04_ellipticity_euclidean_vs_cosine.png", cmap="plasma")
    plot_continuous_side_by_side(df_euc, df_cos, "feat_ruwe", "RUWE",
                                 OUT_DIR / "05_ruwe_euclidean_vs_cosine.png", cmap="hot_r")
    plot_continuous_side_by_side(df_euc, df_cos, "feat_ipd_frac_multi_peak", "IPD Frac Multi-Peak",
                                 OUT_DIR / "06_ipd_multipeak_euclidean_vs_cosine.png", cmap="hot_r")

    print("[4/4] Morphology panel grid...")
    plot_morph_panels_side_by_side(df_euc, df_cos, MORPH_COLS, MORPH_LABELS,
                                   OUT_DIR / "07_morphology_panels_euclidean_vs_cosine.png")

    # Copy individual standalone plots for reference
    for name in ["05_umap_field_tag_colored.png", "06_umap_morphology_panels.png"]:
        for src, tag in [(PLOTS_EUC, "euclidean"), (PLOTS_COS, "cosine")]:
            p = src / name
            if p.exists():
                shutil.copy(p, OUT_DIR / f"{tag}_{name}")
                print(f"  copied {tag}_{name}")

    summary = {
        "euclidean": {"csv": str(CSV_EUC), "n_points": len(df_euc), "metric": "euclidean"},
        "cosine":    {"csv": str(CSV_COS), "n_points": len(df_cos), "metric": "cosine"},
        "note": "Both runs: standard scaling, integral_pos normalisation, n_neighbours=15, min_dist=0.1",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== DONE ===")
    print("Out dir:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("*.png")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
