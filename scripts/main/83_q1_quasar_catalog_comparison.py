"""
83_q1_quasar_catalog_comparison.py
=====================================

Compares our ERO quasar detections (WISE color-cut and Quaia) against the
official Euclid Q1 spectroscopic quasar catalog from:

  Euclid Collaboration, Lustig et al. 2025 (arxiv:2512.08803)
  ~3,500 bright quasars from Euclid NISP spectroscopy, 0 < z < 4.8

STATUS: Catalog not yet publicly available on CDS (listed as "in preparation"
as of 2026-03). Check: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/697/A4

Once the catalog is available, download it (FITS or CSV) to:
  data/q1_quasar_catalog.fits   (or .csv)
and update Q1_CATALOG_PATH below.

Expected columns (from paper):
  source_id       Gaia DR3 source_id (int64) — enables direct join
  ra, dec         J2000 coordinates
  redshift        spectroscopic redshift from NISP
  field           Euclid field name

Outputs (plots/ml_runs/quasar_q1_comparison/):
  01_venn_overlap.png            WISE / Quaia / Q1 overlap diagram
  02_q1_on_wise_color_color.png  Q1 quasars on W1-W2 vs W2-W3 diagram
  03_q1_umap_overlay.png         Q1 quasars on 15D UMAP
  04_q1_score_distribution.png   Our XGB score for Q1 sources
  05_completeness_purity.png     Recovery stats per method
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3        # pip install matplotlib-venn if needed

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ======================================================================
BASE     = Path(__file__).resolve().parents[2]
OUT_DIR  = BASE / "output" / "ml_runs" / "quasar_q1_comparison"
PLOT_DIR = BASE / "plots"  / "ml_runs" / "quasar_q1_comparison"

Q1_CATALOG_PATH = BASE / "data" / "q1_quasar_catalog.fits"

WISE_SCORES  = BASE / "output" / "ml_runs" / "quasar_wise_clf" / "scores_all.csv"
QUAIA_SCORES = BASE / "output" / "ml_runs" / "quaia_clf"       / "scores_all.csv"
WISE_CSV     = BASE / "output" / "ml_runs" / "quasar_wise_clf" / "wise_phot.csv"

EMB_16D = (BASE / "output" / "experiments" / "embeddings"
           / "umap16d_manualv8_filtered" / "embedding_umap.csv")

WISE_MIN_SNR             = 3.0
WISE_W12_LO, WISE_W12_HI = 0.6, 1.6
WISE_W23_LO, WISE_W23_HI = 2.0, 4.2

C_Q1    = "#7b2d8b"   # purple for Q1 spectroscopic
C_WISE  = "#e8671b"   # orange for WISE color-cut
C_QUAIA = "#ff007f"   # magenta for Quaia


# ======================================================================

def load_q1(path: Path) -> pd.DataFrame:
    """Load Q1 quasar catalog. Supports FITS and CSV."""
    if path.suffix.lower() == ".fits":
        from astropy.io import fits
        with fits.open(path, memmap=True) as hdul:
            data = hdul[1].data
            def native(arr):
                a = np.asarray(arr)
                return a.astype(a.dtype.newbyteorder("="), copy=False)
            df = pd.DataFrame({
                "source_id": native(data["source_id"]).astype(np.int64),
            })
            for col in ["ra", "dec", "redshift", "field"]:
                if col in data.names:
                    df[col] = native(data[col])
    else:
        df = pd.read_csv(path, low_memory=False)
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype(np.int64)
    print(f"  Q1 catalog: {len(df):,} quasars")
    return df


def main() -> None:
    if not Q1_CATALOG_PATH.exists():
        print(
            f"ERROR: Q1 catalog not found at {Q1_CATALOG_PATH}\n"
            "The Euclid Q1 spectroscopic quasar catalog (arxiv:2512.08803) is not yet\n"
            "publicly available on CDS (in preparation as of 2026-03).\n\n"
            "Once released, download it and place it at:\n"
            f"  {Q1_CATALOG_PATH}\n"
            "Then re-run this script."
        )
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load catalogs ----
    print("Loading Q1 quasar catalog...")
    q1 = load_q1(Q1_CATALOG_PATH)
    q1["source_id"] = q1["source_id"].astype("Int64")

    print("Loading WISE classifier scores...")
    wise_scores = pd.read_csv(WISE_SCORES, low_memory=False)
    wise_scores["source_id"] = pd.to_numeric(wise_scores["source_id"], errors="coerce").astype("Int64")

    print("Loading Quaia classifier scores...")
    quaia_scores = pd.read_csv(QUAIA_SCORES, low_memory=False)
    quaia_scores["source_id"] = pd.to_numeric(quaia_scores["source_id"], errors="coerce").astype("Int64")

    # ERO source_ids in each selection
    ero_ids      = set(wise_scores["source_id"].dropna().tolist())
    wise_ids     = set(wise_scores.loc[wise_scores["label"] == 1, "source_id"].dropna().tolist())
    quaia_ids    = set(quaia_scores.loc[quaia_scores["label"] == 1, "source_id"].dropna().tolist())
    q1_ids_all   = set(q1["source_id"].dropna().tolist())
    q1_in_ero    = q1_ids_all & ero_ids

    print(f"\nERO source coverage: {len(ero_ids):,} sources")
    print(f"WISE color-cut:  {len(wise_ids)} quasar-locus sources")
    print(f"Quaia catalog:   {len(quaia_ids)} confirmed quasars")
    print(f"Q1 catalog (ERO overlap): {len(q1_in_ero)} / {len(q1_ids_all):,} Q1 sources")

    # ---- Overlap statistics ----
    w_only   = len(wise_ids - quaia_ids - q1_in_ero)
    q_only   = len(quaia_ids - wise_ids - q1_in_ero)
    q1_only  = len(q1_in_ero - wise_ids - quaia_ids)
    wq       = len(wise_ids & quaia_ids - q1_in_ero)
    wq1      = len(wise_ids & q1_in_ero - quaia_ids)
    qq1      = len(quaia_ids & q1_in_ero - wise_ids)
    all3     = len(wise_ids & quaia_ids & q1_in_ero)

    print(f"\nVenn overlap:")
    print(f"  WISE only:           {w_only}")
    print(f"  Quaia only:          {q_only}")
    print(f"  Q1 only (in ERO):    {q1_only}")
    print(f"  WISE ∩ Quaia:        {wq}")
    print(f"  WISE ∩ Q1:           {wq1}")
    print(f"  Quaia ∩ Q1:          {qq1}")
    print(f"  All three:           {all3}")

    # Recovery rates for Q1
    q1_wise_recovery  = len(q1_in_ero & wise_ids)  / max(len(q1_in_ero), 1)
    q1_quaia_recovery = len(q1_in_ero & quaia_ids) / max(len(q1_in_ero), 1)
    print(f"\nQ1 recovery rates (of {len(q1_in_ero)} Q1 quasars in ERO footprint):")
    print(f"  WISE color-cut:  {q1_wise_recovery*100:.0f}%")
    print(f"  Quaia catalog:   {q1_quaia_recovery*100:.0f}%")

    # ---- Plot 01: Venn diagram ----
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        venn3(
            subsets=(w_only, q_only, wq, q1_only, wq1, qq1, all3),
            set_labels=("WISE color-cut", "Quaia", "Q1 spectroscopic"),
            set_colors=(C_WISE, C_QUAIA, C_Q1),
            alpha=0.5, ax=ax,
        )
        ax.set_title("Quasar selection overlap: WISE / Quaia / Q1 (ERO footprint)", fontsize=10)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "01_venn_overlap.png", dpi=150)
        plt.close(fig)
        print("  saved 01_venn_overlap.png")
    except Exception as e:
        print(f"  Venn diagram skipped ({e}). Install matplotlib-venn if needed.")

    # ---- Plot 02: Q1 on WISE color-color ----
    if WISE_CSV.exists():
        wise = pd.read_csv(WISE_CSV, low_memory=False)
        for c in ["w1mpro", "w2mpro", "w3mpro", "w1snr", "w2snr", "w3snr"]:
            wise[c] = pd.to_numeric(wise[c], errors="coerce")
        snr_ok = ((wise["w1snr"] >= WISE_MIN_SNR) &
                  (wise["w2snr"] >= WISE_MIN_SNR) &
                  (wise["w3snr"] >= WISE_MIN_SNR))
        wise = wise[snr_ok].copy()
        wise["w1_w2"] = wise["w1mpro"] - wise["w2mpro"]
        wise["w2_w3"] = wise["w2mpro"] - wise["w3mpro"]
        wise["source_id"] = pd.to_numeric(wise["source_id"], errors="coerce").astype("Int64")
        wise["is_q1"] = wise["source_id"].isin(q1_in_ero)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(wise.loc[~wise["is_q1"], "w2_w3"],
                   wise.loc[~wise["is_q1"], "w1_w2"],
                   s=3, alpha=0.15, color="#aaaaaa", label="ERO (WISE S/N≥3)")
        ax.scatter(wise.loc[wise["is_q1"], "w2_w3"],
                   wise.loc[wise["is_q1"], "w1_w2"],
                   s=30, alpha=1.0, color=C_Q1, zorder=5,
                   edgecolors="white", linewidths=0.4,
                   label=f"Q1 spectroscopic (n={int(wise['is_q1'].sum())})")
        rect_x = [WISE_W23_LO, WISE_W23_HI, WISE_W23_HI, WISE_W23_LO, WISE_W23_LO]
        rect_y = [WISE_W12_LO, WISE_W12_LO, WISE_W12_HI, WISE_W12_HI, WISE_W12_LO]
        ax.plot(rect_x, rect_y, "b--", lw=1.5, label="WISE color-cut box")
        ax.set_xlabel("W2 − W3 [Vega mag]", fontsize=11)
        ax.set_ylabel("W1 − W2 [Vega mag]", fontsize=11)
        ax.set_title("Q1 spectroscopic quasars on WISE colour-colour", fontsize=11)
        ax.legend(fontsize=9, markerscale=2)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "02_q1_on_wise_color_color.png", dpi=150)
        plt.close(fig)
        print("  saved 02_q1_on_wise_color_color.png")

    # ---- Plot 03: Q1 on 15D UMAP ----
    if EMB_16D.exists():
        umap = pd.read_csv(EMB_16D, usecols=["source_id", "x", "y"], low_memory=False)
        umap["source_id"] = pd.to_numeric(umap["source_id"], errors="coerce").astype("Int64")
        umap["is_q1"] = umap["source_id"].isin(q1_in_ero)
        # Attach our XGB score (from WISE classifier)
        umap = umap.merge(
            wise_scores[["source_id", "xgb_score"]].rename(columns={"xgb_score": "wise_score"}),
            on="source_id", how="left",
        )
        in_q1 = umap["is_q1"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes[0]
        ax.scatter(umap.loc[~in_q1, "x"], umap.loc[~in_q1, "y"],
                   s=0.5, alpha=0.06, color="#555555", rasterized=True)
        ax.scatter(umap.loc[in_q1, "x"], umap.loc[in_q1, "y"],
                   s=30, alpha=1.0, color=C_Q1, rasterized=True, zorder=5,
                   edgecolors="white", linewidths=0.4,
                   label=f"Q1 spectroscopic (n={in_q1.sum()})")
        ax.set_title("(a) Q1 spectroscopic quasars", fontsize=10)
        ax.legend(markerscale=1.5, fontsize=8, framealpha=0.85, edgecolor="none")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1]
        has_score = umap["wise_score"].notna()
        ax.scatter(umap.loc[~has_score, "x"], umap.loc[~has_score, "y"],
                   s=0.5, alpha=0.06, color="#555555", rasterized=True)
        sc = ax.scatter(umap.loc[has_score, "x"], umap.loc[has_score, "y"],
                        s=3, c=umap.loc[has_score, "wise_score"],
                        cmap="RdPu", alpha=0.8, vmin=0, vmax=1, rasterized=True)
        ax.scatter(umap.loc[in_q1, "x"], umap.loc[in_q1, "y"],
                   s=30, alpha=1.0, color=C_Q1, rasterized=True, zorder=6,
                   edgecolors="white", linewidths=0.4,
                   label="Q1 spectroscopic")
        plt.colorbar(sc, ax=ax, label="XGB P(WISE quasar)")
        ax.set_title("(b) WISE XGB score + Q1 overlay", fontsize=10)
        ax.legend(markerscale=1.5, fontsize=8, framealpha=0.85, edgecolor="none")
        ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle("15D Gaia UMAP — Q1 spectroscopic quasars", fontsize=12)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "03_q1_umap_overlay.png", dpi=150)
        plt.close(fig)
        print("  saved 03_q1_umap_overlay.png")

    # ---- Plot 04: XGB score distribution for Q1 sources ----
    q1_in_ero_list = list(q1_in_ero)
    q1_wise_merged = wise_scores[wise_scores["source_id"].isin(q1_in_ero_list)].copy()
    q1_quaia_merged = quaia_scores[quaia_scores["source_id"].isin(q1_in_ero_list)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins_p = np.linspace(0, 1, 35)
    for ax, df_m, label, color in [
        (axes[0], q1_wise_merged,  "WISE 15D classifier", C_WISE),
        (axes[1], q1_quaia_merged, "Quaia 15D classifier", C_QUAIA),
    ]:
        if len(df_m) > 0:
            ax.hist(df_m["xgb_score"], bins=bins_p, color=color, alpha=0.8, edgecolor="white")
            ax.axvline(0.5, color="k", lw=0.8, linestyle="--")
            high = (df_m["xgb_score"] >= 0.5).sum()
            ax.set_title(f"{label}\n(n={len(df_m)} Q1 sources, {high} with p≥0.5)", fontsize=9)
        ax.set_xlabel("XGB P(quasar)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)

    fig.suptitle("Our XGB scores for Q1 spectroscopic quasars in ERO footprint", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_q1_score_distribution.png", dpi=150)
    plt.close(fig)
    print("  saved 04_q1_score_distribution.png")

    print(f"\nOutputs -> {OUT_DIR}")
    print(f"Plots   -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
