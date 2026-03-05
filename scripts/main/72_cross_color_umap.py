#!/usr/bin/env python3
"""
Cross-color UMAP: show that Gaia feature space and pixel morphology space agree.

Panel A — Gaia 16D UMAP colored by morphology metrics
  Take the existing Gaia 16D embedding, merge morphology from the pixel
  embedding CSV by source_id, and produce colored scatter plots.

Panel B — Pixel UMAP colored by all 16 Gaia features
  Take the existing pixel embedding (already has 8D features), merge the
  remaining 8 quality features from the Gaia CSV, and produce colored panels.

The key result: if Gaia-space clusters align with morphology gradients,
that is direct evidence that Gaia residual diagnostics encode morphology.

Outputs: report/model_decision/2026-03-05_umap_improvements/02_cross_color/
"""
from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]

GAIA_EMB_CSV = BASE / "output" / "experiments" / "embeddings" / \
    "double_stars_gaia_16d_all_20260216" / "embedding_umap.csv"

PIXEL_EMB_CSV = BASE / "output" / "experiments" / "embeddings" / \
    "double_stars_pixels" / "email_pixels_all_filtered" / "umap_standard" / "embedding_umap.csv"

OUT_DIR = BASE / "report" / "model_decision" / "2026-03-05_umap_improvements" / "02_cross_color"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Column lists ──────────────────────────────────────────────────────────────
MORPH_COLS = [
    "morph_asym_180",
    "morph_concentration_r80r20",
    "morph_ellipticity_e2",
    "morph_peakedness_kurtosis",
    "morph_roundness",
    "morph_gini",
    "morph_m20",
    "morph_smoothness",
    "morph_edge_asym_180",
    "morph_peak_to_total",
    "morph_texture_laplacian",
    "morph_mirror_asym_lr",
]

MORPH_LABELS = {
    "morph_asym_180": "Asymmetry 180",
    "morph_concentration_r80r20": "Concentration r80/r20",
    "morph_ellipticity_e2": "Ellipticity e2",
    "morph_peakedness_kurtosis": "Kurtosis",
    "morph_roundness": "Roundness",
    "morph_gini": "Gini",
    "morph_m20": "M20",
    "morph_smoothness": "Smoothness",
    "morph_edge_asym_180": "Edge Asymmetry",
    "morph_peak_to_total": "Peak/Total",
    "morph_texture_laplacian": "Laplacian Texture",
    "morph_mirror_asym_lr": "Mirror Asymmetry LR",
}

FEAT_8D = [
    "feat_log10_snr",
    "feat_ruwe",
    "feat_astrometric_excess_noise",
    "feat_parallax_over_error",
    "feat_visibility_periods_used",
    "feat_ipd_frac_multi_peak",
    "feat_c_star",
    "feat_pm_significance",
]

FEAT_EXTRA_8D = [
    "feat_astrometric_excess_noise_sig",
    "feat_ipd_gof_harmonic_amplitude",
    "feat_ipd_gof_harmonic_phase",
    "feat_ipd_frac_odd_win",
    "feat_phot_bp_n_contaminated_transits",
    "feat_phot_bp_n_blended_transits",
    "feat_phot_rp_n_contaminated_transits",
    "feat_phot_rp_n_blended_transits",
]

FEAT_LABELS = {
    "feat_log10_snr": "log10 SNR",
    "feat_ruwe": "RUWE",
    "feat_astrometric_excess_noise": "Astrometric Excess Noise",
    "feat_parallax_over_error": "Parallax / Error",
    "feat_visibility_periods_used": "Visibility Periods",
    "feat_ipd_frac_multi_peak": "IPD Frac Multi-Peak",
    "feat_c_star": "C*",
    "feat_pm_significance": "PM Significance",
    "feat_astrometric_excess_noise_sig": "AEN Significance",
    "feat_ipd_gof_harmonic_amplitude": "IPD GoF Harmonic Amp",
    "feat_ipd_gof_harmonic_phase": "IPD GoF Harmonic Phase",
    "feat_ipd_frac_odd_win": "IPD Frac Odd Window",
    "feat_phot_bp_n_contaminated_transits": "BP Contaminated Transits",
    "feat_phot_bp_n_blended_transits": "BP Blended Transits",
    "feat_phot_rp_n_contaminated_transits": "RP Contaminated Transits",
    "feat_phot_rp_n_blended_transits": "RP Blended Transits",
}


def savefig(path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def scatter_panels(df, x_col, y_col, cols, labels, out_png, suptitle,
                   ncols=3, s=3, alpha=0.65, cmap="viridis", clip_pct=(1, 99)):
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        print(f"  [WARN] No columns found for {out_png.name}")
        return
    n = len(cols_present)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.3 * nrows))
    axes = np.array(axes).ravel()
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    for i, col in enumerate(cols_present):
        ax = axes[i]
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() > 10:
            lo, hi = np.nanpercentile(vals[finite], list(clip_pct))
            vals = np.clip(vals, lo, hi)
        sc = ax.scatter(x, y, c=vals, s=s, cmap=cmap, alpha=alpha, linewidths=0, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(labels.get(col, col), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(out_png)
    print(f"  saved {out_png.name}")


def plot_density_with_morph_contours(gaia_df, out_png):
    """Hexbin density of Gaia UMAP with RMSE-high sources highlighted."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hexbin(gaia_df["x"], gaia_df["y"], gridsize=180, bins="log",
              cmap="Greys", mincnt=1)
    # Overlay sources with high morphology complexity (high asym or high ellipticity)
    for col, label, color in [
        ("morph_asym_180",       "High Asymmetry (p90)",   "#e63946"),
        ("morph_ellipticity_e2", "High Ellipticity (p90)", "#457b9d"),
    ]:
        if col not in gaia_df.columns:
            continue
        vals = pd.to_numeric(gaia_df[col], errors="coerce")
        thresh = vals.quantile(0.90)
        hi = gaia_df[vals >= thresh]
        ax.scatter(hi["x"], hi["y"], s=6, color=color, alpha=0.5,
                   linewidths=0, label=label, rasterized=True)
    ax.set_xlabel("Gaia 16D UMAP-1", fontsize=11)
    ax.set_ylabel("Gaia 16D UMAP-2", fontsize=11)
    ax.set_title("Gaia 16D UMAP — high-morphology sources overlaid", fontsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    savefig(out_png)
    print(f"  saved {out_png.name}")


def main():
    # ── Load CSVs ─────────────────────────────────────────────────────────────
    print("[1/4] Loading Gaia 16D embedding...")
    gaia_df = pd.read_csv(GAIA_EMB_CSV, low_memory=False)
    gaia_df["source_id"] = pd.to_numeric(gaia_df["source_id"], errors="coerce")
    gaia_df = gaia_df.dropna(subset=["source_id", "x", "y"]).copy()
    gaia_df["source_id"] = gaia_df["source_id"].astype(np.int64)
    print(f"  Gaia 16D UMAP: {len(gaia_df):,} rows")

    print("[2/4] Loading pixel embedding...")
    pix_df = pd.read_csv(PIXEL_EMB_CSV, low_memory=False)
    pix_df["source_id"] = pd.to_numeric(pix_df["source_id"], errors="coerce")
    pix_df = pix_df.dropna(subset=["source_id", "x", "y"]).copy()
    pix_df["source_id"] = pix_df["source_id"].astype(np.int64)
    print(f"  Pixel UMAP:    {len(pix_df):,} rows")

    # ── Panel A: Gaia UMAP + morphology ──────────────────────────────────────
    print("[3/4] Building Panel A: Gaia UMAP colored by morphology...")
    morph_cols_avail = [c for c in MORPH_COLS if c in pix_df.columns]
    merge_cols = ["source_id"] + morph_cols_avail
    gaia_morph = gaia_df.merge(
        pix_df[merge_cols].drop_duplicates("source_id"),
        on="source_id", how="left"
    )
    n_matched = gaia_morph[morph_cols_avail[0]].notna().sum() if morph_cols_avail else 0
    print(f"  Matched {n_matched:,}/{len(gaia_morph):,} rows with morphology data")

    scatter_panels(
        gaia_morph, "x", "y", MORPH_COLS, MORPH_LABELS,
        OUT_DIR / "A_gaia_umap_colored_by_morphology.png",
        suptitle="Gaia 16D UMAP — colored by pixel morphology metrics\n"
                 "(sources matched by source_id; ~" + f"{n_matched/len(gaia_morph)*100:.0f}% coverage)",
        ncols=3, s=2, alpha=0.6,
    )

    plot_density_with_morph_contours(gaia_morph, OUT_DIR / "A2_gaia_umap_highmorphology_overlay.png")

    # Rename x/y to avoid collision before saving
    gaia_morph_save = gaia_morph[["source_id", "x", "y", "field_tag"] + morph_cols_avail].copy()
    gaia_morph_save.rename(columns={"x": "gaia_umap_x", "y": "gaia_umap_y"}, inplace=True)

    # ── Panel B: Pixel UMAP + full 16D Gaia features ─────────────────────────
    print("[4/4] Building Panel B: Pixel UMAP colored by full 16D Gaia features...")
    extra_feat_avail = [c for c in FEAT_EXTRA_8D if c in gaia_df.columns]
    if extra_feat_avail:
        pix_16d = pix_df.merge(
            gaia_df[["source_id"] + extra_feat_avail].drop_duplicates("source_id"),
            on="source_id", how="left"
        )
        print(f"  Merged {len(extra_feat_avail)} extra quality features into pixel embedding")
    else:
        pix_16d = pix_df.copy()
        print("  No extra features to merge (8D features already present)")

    all_feat_cols = [c for c in (FEAT_8D + FEAT_EXTRA_8D) if c in pix_16d.columns]
    scatter_panels(
        pix_16d, "x", "y", all_feat_cols, FEAT_LABELS,
        OUT_DIR / "B_pixel_umap_colored_by_gaia_features.png",
        suptitle="Pixel UMAP — colored by all 16 Gaia features",
        ncols=4, s=2, alpha=0.6, cmap="plasma",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_overlap = len(set(gaia_df["source_id"].tolist()) & set(pix_df["source_id"].tolist()))
    print(f"\n  source_id overlap between Gaia and pixel embeddings: {n_overlap:,}")
    print("\n=== DONE ===")
    print("Out dir:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("*.png")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
