"""
80_quasar_stamp_catalog.py
===========================

Visual stamp catalog of WISE-selected quasar locus sources from the
Euclid ERO dataset, sorted by descending XGB quasar probability.

Inputs:
  output/ml_runs/quasar_wise_clf/scores_all.csv   (source_id, label, xgb_score, w1_w2, w2_w3)
  output/ml_runs/quasar_wise_clf/wise_phot.csv    (source_id, field_tag, sep_arcsec, ...)
  output/dataset_npz/{field}/metadata_16d.csv     (npz_file, index_in_file, ra, dec, ...)
  output/dataset_npz/{field}/stamps_*.npz         (X array of raw stamps)

Output:
  plots/ml_runs/quasar_wise_clf/
    08_quasar_stamp_catalog_p{N}.png   (one page per STAMPS_PER_PAGE stamps)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ======================================================================
# CONFIG
# ======================================================================

BASE         = Path(__file__).resolve().parents[2]
DATASET_ROOT = BASE / "output" / "dataset_npz"
QSO_DIR      = BASE / "output" / "ml_runs" / "quasar_wise_clf"
PLOT_DIR     = BASE / "plots"  / "ml_runs" / "quasar_wise_clf"

SCORES_CSV   = QSO_DIR / "scores_all.csv"
WISE_CSV     = QSO_DIR / "wise_phot.csv"
META_NAME    = "metadata_16d.csv"

STAMP_PIX    = 20          # expected stamp size
STAMPS_PER_PAGE = 80       # stamps per output PNG (8 cols × 10 rows)
NCOLS        = 8
MIN_XGB_SCORE = 0.0        # show all label=1 sources; raise to filter by score

# Normalisation for display: clip to [lo_pct, hi_pct] of positive pixels
NORM_LO_PCT  = 1.0
NORM_HI_PCT  = 99.5


# ======================================================================
# HELPERS
# ======================================================================

def load_stamps_for_sources(df_sel: pd.DataFrame) -> dict[int, np.ndarray]:
    """
    Load raw Euclid VIS stamps for selected sources.
    Returns {source_id: stamp_2d_array}.
    Groups by field_tag + npz_file to minimise file opens.
    """
    need = {"source_id", "field_tag", "npz_file", "index_in_file"}
    missing = need - set(df_sel.columns)
    if missing:
        raise RuntimeError(f"Missing columns in selection df: {missing}")

    out: dict[int, np.ndarray] = {}
    groups = df_sel.groupby(["field_tag", "npz_file"], sort=False)
    for (field_tag, npz_file), grp in groups:
        npz_path = DATASET_ROOT / str(field_tag) / str(npz_file)
        if not npz_path.exists():
            print(f"  [WARN] missing {npz_path.name} in {field_tag}")
            continue
        with np.load(npz_path) as d:
            if "X" not in d:
                print(f"  [WARN] no 'X' key in {npz_path.name}")
                continue
            Xall = d["X"]   # shape (N, H*W) or (N, H, W)
        for row in grp.itertuples(index=False):
            ii = int(row.index_in_file)
            if not (0 <= ii < Xall.shape[0]):
                continue
            stamp = np.asarray(Xall[ii], dtype=np.float32)
            if stamp.ndim == 1:
                side = int(round(stamp.shape[0] ** 0.5))
                stamp = stamp.reshape(side, side)
            out[int(row.source_id)] = stamp
    return out


def norm_stamp(stamp: np.ndarray) -> np.ndarray:
    """Clip and normalise to [0, 1] for display."""
    lo = np.nanpercentile(stamp, NORM_LO_PCT)
    hi = np.nanpercentile(stamp, NORM_HI_PCT)
    if hi <= lo:
        return np.zeros_like(stamp)
    return np.clip((stamp - lo) / (hi - lo), 0, 1)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load scores and select quasar locus sources ----
    print("Loading QSO scores...")
    scores = pd.read_csv(SCORES_CSV, low_memory=False)
    scores["source_id"] = pd.to_numeric(scores["source_id"], errors="coerce").astype("Int64")

    wise = pd.read_csv(WISE_CSV, usecols=["source_id", "field_tag", "sep_arcsec"],
                       low_memory=False)
    wise["source_id"] = pd.to_numeric(wise["source_id"], errors="coerce").astype("Int64")

    qso = scores[scores["label"] == 1].copy()
    qso = qso[qso["xgb_score"] >= MIN_XGB_SCORE]
    qso = qso.merge(wise, on="source_id", how="left")
    qso = qso.sort_values("xgb_score", ascending=False).reset_index(drop=True)
    print(f"  Quasar locus sources: {len(qso)}")

    # ---- Load metadata to get npz_file / index_in_file / ra / dec ----
    print("Loading metadata for stamp locations...")
    field_dirs = sorted([
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir() and (p / META_NAME).exists()
    ])
    meta_frames = []
    for fdir in field_dirs:
        df = pd.read_csv(
            fdir / META_NAME,
            usecols=lambda c: c in {"source_id", "npz_file", "index_in_file",
                                     "ra", "dec", "phot_g_mean_mag", "field_tag"},
            low_memory=False,
        )
        if "field_tag" not in df.columns:
            df["field_tag"] = fdir.name
        meta_frames.append(df)
    meta = pd.concat(meta_frames, ignore_index=True)
    meta["source_id"] = pd.to_numeric(meta["source_id"], errors="coerce").astype("Int64")
    meta = meta.drop_duplicates("source_id")

    # Merge scores with metadata
    df_sel = qso.merge(
        meta[["source_id", "npz_file", "index_in_file", "ra", "dec",
              "phot_g_mean_mag", "field_tag"]],
        on="source_id", how="left", suffixes=("", "_meta"),
    )
    # Prefer field_tag from metadata when available
    if "field_tag_meta" in df_sel.columns:
        df_sel["field_tag"] = df_sel["field_tag_meta"].combine_first(df_sel["field_tag"])
        df_sel = df_sel.drop(columns=["field_tag_meta"])

    df_sel = df_sel.dropna(subset=["npz_file", "index_in_file"]).copy()
    df_sel["index_in_file"] = df_sel["index_in_file"].astype(int)
    print(f"  Sources with stamp location: {len(df_sel)}")

    # ---- Load stamps ----
    print("Loading stamps from NPZ files...")
    stamp_map = load_stamps_for_sources(df_sel)
    print(f"  Stamps loaded: {len(stamp_map)}")

    # ---- Build pages ----
    n_total = len(df_sel)
    n_pages = max(1, int(np.ceil(n_total / STAMPS_PER_PAGE)))
    nrows   = int(np.ceil(STAMPS_PER_PAGE / NCOLS))

    print(f"  Generating {n_pages} page(s) of {STAMPS_PER_PAGE} stamps each...")

    for page in range(n_pages):
        i0 = page * STAMPS_PER_PAGE
        i1 = min(i0 + STAMPS_PER_PAGE, n_total)
        batch = df_sel.iloc[i0:i1]

        fig = plt.figure(figsize=(NCOLS * 1.8, nrows * 2.1), facecolor="black")
        fig.suptitle(
            f"Euclid VIS stamps — WISE quasar locus  "
            f"(page {page+1}/{n_pages}, sources {i0+1}–{i1}, ranked by XGB score)",
            color="white", fontsize=9, y=0.995,
        )
        gs = gridspec.GridSpec(nrows, NCOLS, figure=fig,
                               hspace=0.55, wspace=0.08,
                               top=0.965, bottom=0.01, left=0.01, right=0.99)

        for k, row in enumerate(batch.itertuples(index=False)):
            r_idx = k // NCOLS
            c_idx = k  % NCOLS
            ax = fig.add_subplot(gs[r_idx, c_idx])
            ax.set_facecolor("black")

            sid = int(row.source_id) if hasattr(row, "source_id") else -1
            stamp = stamp_map.get(sid)

            if stamp is not None:
                ax.imshow(norm_stamp(stamp), cmap="gray",
                          origin="lower", interpolation="nearest",
                          vmin=0, vmax=1)
            else:
                ax.imshow(np.zeros((STAMP_PIX, STAMP_PIX)), cmap="gray",
                          origin="lower", vmin=0, vmax=1)
                ax.text(0.5, 0.5, "N/A", color="red", fontsize=6,
                        ha="center", va="center", transform=ax.transAxes)

            # Annotation
            field = str(getattr(row, "field_tag", "?")).replace("ERO-", "")
            gmag  = getattr(row, "phot_g_mean_mag", np.nan)
            score = float(row.xgb_score)
            w12   = float(row.w1_w2) if hasattr(row, "w1_w2") else np.nan
            w23   = float(row.w2_w3) if hasattr(row, "w2_w3") else np.nan

            title_line1 = f"{field}  G={gmag:.1f}" if np.isfinite(gmag) else field
            title_line2 = f"p={score:.2f}  W12={w12:.1f}  W23={w23:.1f}"

            ax.set_title(f"{title_line1}\n{title_line2}",
                         color="white", fontsize=4.5, pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

        # blank unused cells
        for k in range(len(batch), nrows * NCOLS):
            r_idx = k // NCOLS
            c_idx = k  % NCOLS
            ax = fig.add_subplot(gs[r_idx, c_idx])
            ax.set_visible(False)

        out_png = PLOT_DIR / f"08_quasar_stamp_catalog_p{page+1:02d}.png"
        fig.savefig(out_png, dpi=180, facecolor="black")
        plt.close(fig)
        print(f"  saved {out_png.name}")

    print(f"\nDone. {n_pages} catalog page(s) -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
