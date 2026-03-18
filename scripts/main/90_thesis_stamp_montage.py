#!/usr/bin/env python3
"""
90_thesis_stamp_montage.py
============================

Regenerate the three-row Euclid VIS stamp montage (Figure 4.13) with
log10 contrast stretching to reveal faint extended morphology.

Rows (10 stamps each):
  Top    (red)   — spectroscopically confirmed Quaia quasars
  Middle (blue)  — sources from galaxy cluster fields (Abell2390/Abell2764/Perseus)
  Bottom (green) — DESI spectroscopically confirmed galaxies, spanning a range
                   of XGBoost scores / morphological extents

Each stamp is 20x20 pixels (2"x2").
Stretch: log10(1 + 9*x) after background subtraction and percentile normalisation.
Stamps that are broken (near-zero signal) or overfilled (source fills the frame)
are screened out and replaced from a larger candidate pool.

Output:
  report/phd-thesis-template-2.4/Chapter4/Figs/Raster/23_quasar_cluster_stamps.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).resolve().parents[2]
DATASET_ROOT = BASE / "output" / "dataset_npz"
QUAIA_DIR    = BASE / "output" / "ml_runs" / "quaia_clf"
DESI_DIR     = BASE / "output" / "ml_runs" / "galaxy_desi_clf"
THESIS_OUT   = (BASE / "report" / "phd-thesis-template-2.4"
                / "Chapter4" / "Figs" / "Raster"
                / "23_quasar_cluster_stamps.png")
THESIS_OUT.parent.mkdir(parents=True, exist_ok=True)

N_STAMPS        = 10
CLUSTER_FIELDS  = ["ERO-Abell2390", "ERO-Abell2764", "ERO-Perseus"]

# Quality thresholds
MIN_PEAK_SIGNAL  = 60.0   # (hi_pct - lo_pct) in ADU; below this → broken
MAX_FILL_FRAC    = 0.14   # fraction above half-peak; above this → overfilled (DESI/quasar)
MAX_FILL_CLUSTER = 0.12   # (kept for reference; cluster now uses multi-source selection)


# ── stamp loading ─────────────────────────────────────────────────────────────

def load_stamp(field_tag: str, npz_file: str, idx: int) -> np.ndarray | None:
    """Return a (20, 20) float32 stamp or None if file missing / index OOB."""
    npz_path = DATASET_ROOT / field_tag / npz_file
    if not npz_path.exists():
        return None
    data = np.load(str(npz_path), allow_pickle=False)
    X = data["X"]
    if idx >= len(X):
        return None
    stamp = np.asarray(X[idx], dtype=np.float32)
    if stamp.ndim == 1:
        s = int(round(stamp.shape[0] ** 0.5))
        stamp = stamp.reshape(s, s)
    return stamp


# ── quality filter ────────────────────────────────────────────────────────────

def stamp_is_bad(stamp: np.ndarray, max_fill: float = MAX_FILL_FRAC) -> bool:
    """
    Return True if the stamp should be rejected:
      - Broken / near-empty: dynamic range below threshold.
      - Chip gap or bad column: any row or column is uniformly dark (stripe artifact).
      - Overfilled: source fills too much of the frame (half-peak fraction too high).
    """
    lo   = np.nanpercentile(stamp, 1.0)
    hi   = np.nanpercentile(stamp, 99.5)
    peak = hi - lo

    if peak < MIN_PEAK_SIGNAL:
        return True   # broken / empty

    # Chip gap / dead-pixel check:
    # Any stamp where > 8% of pixels are at or below 0 straddles a detector
    # gap or dead region and should be rejected.
    if np.sum(stamp <= 0) / stamp.size > 0.08:
        return True   # chip gap / off-detector region

    # Dark-stripe check: any row or column whose median is below 40% of the
    # overall stamp median flags a bad column or bleed trail.
    p50 = np.nanmedian(stamp)
    if p50 > 0:
        row_meds = np.median(stamp, axis=1)
        col_meds = np.median(stamp, axis=0)
        if np.any(row_meds < 0.4 * p50) or np.any(col_meds < 0.4 * p50):
            return True   # dark stripe artifact

    # Source extent check: fraction of pixels above half-peak
    img       = np.clip(stamp - lo, 0, None)
    half_frac = float(np.sum(img > 0.5 * peak)) / img.size
    if half_frac > max_fill:
        return True   # source fills the frame

    return False


def is_overexposed(stamp: np.ndarray, threshold: float = 0.42) -> bool:
    """
    Return True if the stamp looks visually overexposed after log10 stretch:
    mean stretched value > threshold.
    threshold=0.42 for single-source rows (quasar, DESI).
    threshold=0.55 for cluster rows where two bright sources raise the mean.
    """
    lo    = np.nanpercentile(stamp, 1.0)
    hi    = np.nanpercentile(stamp, 99.5)
    scale = hi - lo
    if scale <= 0:
        return False
    img_norm = np.clip(stamp - lo, 0, scale) / scale
    img_log  = np.log10(1.0 + 9.0 * img_norm)
    return bool(np.mean(img_log) > threshold)


def count_distinct_peaks(
    stamp: np.ndarray,
    thresh: float = 0.60,
    min_sep: float = 3.0,
) -> tuple[int, float]:
    """
    Count distinct bright local maxima in the log-stretched stamp.
    Returns (n_peaks, mean_peak_brightness).
    Peaks must be above thresh and separated by ≥ min_sep pixels from each other.
    """
    lo    = np.nanpercentile(stamp, 1.0)
    hi    = np.nanpercentile(stamp, 99.5)
    scale = hi - lo
    if scale <= 0:
        return 0, 0.0
    img = np.clip(stamp - lo, 0, scale) / scale
    img = np.log10(1.0 + 9.0 * img)

    rows, cols = img.shape
    peaks: list[tuple[int, int, float]] = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            v = float(img[r, c])
            if v < thresh:
                continue
            nb = img[r-1:r+2, c-1:c+2].copy()
            nb[1, 1] = -1.0
            if v <= nb.max():
                continue   # not a local maximum
            # Must be sufficiently far from all existing peaks
            if any(((r - pr)**2 + (c - pc)**2)**0.5 < min_sep
                   for pr, pc, _ in peaks):
                continue
            peaks.append((r, c, v))

    if not peaks:
        return 0, 0.0
    return len(peaks), float(np.mean([v for _, _, v in peaks]))


def has_separated_pair(
    stamp: np.ndarray,
    min_sep: float = 3.0,
    primary_thresh: float = 0.85,
    secondary_thresh: float = 0.60,
    max_bright_frac: float = 0.03,
) -> bool:
    """
    Return True if the log-stretched stamp contains two distinct, well-separated
    bright sources without either being a bloated saturated PSF.

    Steps:
      1. Compute log10 stretch (same as draw_row).
      2. Require a primary source: brightest pixel ≥ primary_thresh (0.80).
      3. Reject bloated PSFs: no more than max_bright_frac of pixels may exceed 0.85
         (a point source affects ~1-4 pixels; a saturated star affects far more).
      4. Require a secondary source: at least one pixel ≥ secondary_thresh (0.50)
         located ≥ min_sep pixels from the primary.
    """
    lo    = np.nanpercentile(stamp, 1.0)
    hi    = np.nanpercentile(stamp, 99.5)
    scale = hi - lo
    if scale <= 0:
        return False
    img = np.clip(stamp - lo, 0, scale) / scale
    img = np.log10(1.0 + 9.0 * img)

    br, bc = np.unravel_index(np.argmax(img), img.shape)
    if img[br, bc] < primary_thresh:
        return False   # no genuinely bright primary

    if np.sum(img > 0.85) / img.size > max_bright_frac:
        return False   # primary (or secondary) is bloated / saturated

    rows, cols = img.shape
    for r in range(rows):
        for c in range(cols):
            if img[r, c] < secondary_thresh:
                continue
            if ((r - br) ** 2 + (c - bc) ** 2) ** 0.5 >= min_sep:
                return True   # distinct secondary found
    return False


def pick_good_stamps(
    df: pd.DataFrame,
    n: int,
    label: str,
) -> list[np.ndarray]:
    """
    Load stamps from rows in df, skip bad ones, return first n good stamps.
    Raises RuntimeError if fewer than n good stamps are found.
    """
    good: list[np.ndarray] = []
    for row in df.itertuples(index=False):
        if len(good) >= n:
            break
        stamp = load_stamp(row.field_tag, row.npz_file, int(row.index_in_file))
        if stamp is None:
            continue
        if stamp_is_bad(stamp):
            continue
        good.append(stamp)

    if len(good) < n:
        print(f"  Warning: only {len(good)}/{n} good {label} stamps found.")
    return good


# ── log10 contrast stretch ────────────────────────────────────────────────────

def log_stretch(stamp: np.ndarray) -> np.ndarray:
    """
    Log10 contrast stretch:
      1. Subtract background (1st percentile).
      2. Clip to 99.5th percentile range.
      3. Apply log10(1 + 9*x): maps [0, 1] -> [0, 1] logarithmically.
         This compresses bright peaks and reveals faint outer structure.
    """
    lo    = np.nanpercentile(stamp, 1.0)
    hi    = np.nanpercentile(stamp, 99.5)
    scale = hi - lo
    if scale <= 0:
        return np.zeros_like(stamp)
    img = np.clip(stamp - lo, 0, scale) / scale    # normalise to [0, 1]
    img = np.log10(1.0 + 9.0 * img)                # log10(1..10) -> [0, 1]
    return img


# ── population selection ──────────────────────────────────────────────────────

def select_quasars(n: int = N_STAMPS, pool: int = 300) -> list[np.ndarray]:
    """
    Top-pool Quaia confirmed quasars by XGBoost score.  Collect all quality
    candidates, require a clearly visible bright source (log-stretch max ≥ 0.85),
    then return the n with the highest primary peak.
    """
    hits   = pd.read_csv(QUAIA_DIR / "quaia_hits.csv")
    scores = pd.read_csv(QUAIA_DIR / "scores_all.csv")
    df = (hits[hits.label == 1]
          .merge(scores[["source_id", "xgb_score"]], on="source_id", how="left")
          .dropna(subset=["npz_file", "index_in_file", "xgb_score"])
          .sort_values("xgb_score", ascending=False)
          .head(pool))

    candidates: list[tuple[float, np.ndarray]] = []
    for row in df.itertuples(index=False):
        stamp = load_stamp(row.field_tag, row.npz_file, int(row.index_in_file))
        if stamp is None or stamp_is_bad(stamp):
            continue
        if is_overexposed(stamp):
            continue
        lo    = np.nanpercentile(stamp, 1.0)
        hi    = np.nanpercentile(stamp, 99.5)
        scale = max(hi - lo, 1.0)
        img   = np.clip(stamp - lo, 0, scale) / scale
        img   = np.log10(1.0 + 9.0 * img)
        primary     = float(img.max())
        bright_frac = float(np.sum(img > 0.85) / img.size)
        if primary < 0.85:
            continue   # no clearly visible point source
        if bright_frac > 0.025:
            continue   # PSF too bloated / artefact (>10 pixels above 0.85)
        # Primary must not be at stamp edge (partially-cut sources)
        br, bc = np.unravel_index(np.argmax(img), img.shape)
        if br < 2 or br > 17 or bc < 2 or bc > 17:
            continue
        candidates.append((primary, stamp))

    candidates.sort(key=lambda x: x[0], reverse=True)
    c = [s for _, s in candidates]
    print(f"  Quasar: {len(c)} candidates")

    if len(c) >= 21:
        # Keep: 1,3,4,6,7,9,10. Replace broken 2,5,8 with c[15,16,17].
        good = [
            c[0],   # position  1  (keep)
            c[15],  # position  2  (new)
            c[11],  # position  3  (keep)
            c[12],  # position  4  (keep)
            c[20],  # position  5  (new)
            c[13],  # position  6  (keep)
            c[14],  # position  7  (keep)
            c[19],  # position  8  (new)
            c[8],   # position  9  (keep)
            c[9],   # position 10  (keep)
        ]
    else:
        good = c[:n]

    if len(good) < n:
        print(f"  Warning: only {len(good)}/{n} good quasar stamps found.")
    return good


def select_cluster_sources(n: int = N_STAMPS, pool: int = 10000) -> list[np.ndarray]:
    """
    Stamps from galaxy cluster fields (Abell2390/Abell2764/Perseus) that
    contain at least two distinct point sources, conveying the crowded cluster
    environment.  Basic quality checks (broken, chip gap, dark stripe) are
    applied but no fill-fraction cut is used.
    """
    quaia_ids = set(
        pd.read_csv(QUAIA_DIR / "quaia_hits.csv")
        .query("label == 1")["source_id"].values
    )
    rows = []
    for field in CLUSTER_FIELDS:
        meta_path = DATASET_ROOT / field / "metadata_16d.csv"
        if not meta_path.exists():
            continue
        meta = pd.read_csv(
            meta_path,
            usecols=["source_id", "npz_file", "index_in_file", "split_code"],
        )
        meta["field_tag"] = field
        sub = meta[
            (meta.split_code == 2) & (~meta.source_id.isin(quaia_ids))
        ]
        rows.append(sub)

    total_rows = sum(len(r) for r in rows)
    pool_df = (pd.concat(rows, ignore_index=True)
               .sample(n=min(pool, total_rows), random_state=42))

    # Collect ALL candidates that pass quality + separated-pair check, then
    # return the top n with the FEWEST overexposed pixels (most compact pairs).
    candidates: list[tuple[float, np.ndarray]] = []
    for row in pool_df.itertuples(index=False):
        stamp = load_stamp(row.field_tag, row.npz_file, int(row.index_in_file))
        if stamp is None:
            continue
        if stamp_is_bad(stamp, max_fill=0.50):
            continue
        if not has_separated_pair(stamp):
            continue
        lo    = np.nanpercentile(stamp, 1.0)
        hi    = np.nanpercentile(stamp, 99.5)
        scale = max(hi - lo, 1.0)
        img   = np.clip(stamp - lo, 0, scale) / scale
        img   = np.log10(1.0 + 9.0 * img)
        primary_peak = float(img.max())
        bright_frac  = float(np.sum(img > 0.85) / img.size)
        score = primary_peak - 10.0 * bright_frac
        candidates.append((score, stamp))

    candidates.sort(key=lambda x: x[0], reverse=True)
    c = [s for _, s in candidates]
    print(f"  Cluster: {len(c)} candidates")

    if len(c) >= 26:
        good = [
            c[21],  # position  1
            c[16],  # position  2
            c[2],   # position  3
            c[22],  # position  4
            c[23],  # position  5
            c[13],  # position  6
            c[19],  # position  7
            c[24],  # position  8
            c[20],  # position  9
            c[25],  # position 10
        ]
    else:
        good = c[:n]

    if len(good) < n:
        print(f"  Warning: only {len(good)}/{n} good cluster stamps found.")
    return good


def select_desi_galaxies(n: int = N_STAMPS) -> list[np.ndarray]:
    """
    DESI confirmed galaxies spanning a range of XGBoost scores, quality-screened.
    Sources are sorted by score (descending), screened, and n equally-spaced
    representatives are drawn from the surviving pool.
    """
    hits   = pd.read_csv(DESI_DIR / "desi_hits_ero.csv")
    scores = pd.read_csv(DESI_DIR / "scores_all.csv")

    df = (hits
          .merge(scores[["source_id", "xgb_score"]], on="source_id", how="left")
          .dropna(subset=["npz_file", "index_in_file", "xgb_score"])
          .sort_values("xgb_score", ascending=False)
          .reset_index(drop=True))

    # Pre-screen: keep only quality stamps (structural + single-source brightness)
    good_rows = []
    good_stamps = []
    for row in df.itertuples(index=False):
        stamp = load_stamp(row.field_tag, row.npz_file, int(row.index_in_file))
        if stamp is None or stamp_is_bad(stamp):
            continue
        if is_overexposed(stamp):
            continue
        good_rows.append(row)
        good_stamps.append(stamp)

    total = len(good_stamps)
    print(f"  DESI: {total} good stamps available from {len(df)} candidates")

    if total <= n:
        return good_stamps

    # Pick n equally-spaced indices offset by 2 from each end.
    idxs = list(np.round(np.linspace(2, total - 3, n)).astype(int))
    # Nudge stamp 4 (position index 3) by +5 to avoid near-duplicate with stamp 3.
    idxs[3] = min(idxs[3] + 5, total - 1)
    # Nudge stamp 9 (position index 8) by +2 to draw a different source.
    idxs[8] = min(idxs[8] + 2, total - 1)
    return [good_stamps[i] for i in idxs]


# ── figure assembly ───────────────────────────────────────────────────────────

def draw_row(
    axes: list[plt.Axes],
    stamps: list[np.ndarray],
    row_label: str,
    color: str,
) -> None:
    """Fill one row of axes with log-stretched stamps."""
    for k, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        if k < len(stamps):
            ax.imshow(log_stretch(stamps[k]), cmap="gray",
                      origin="lower", vmin=0, vmax=1,
                      interpolation="nearest")
        else:
            ax.set_facecolor("white")
        if k == 0:
            ax.set_ylabel(row_label, fontsize=13, color=color, labelpad=6,
                          rotation=90, va="center")


def make_montage(
    quasar_stamps:  list[np.ndarray],
    cluster_stamps: list[np.ndarray],
    desi_stamps:    list[np.ndarray],
    out_path: Path,
) -> None:
    n = N_STAMPS
    fig = plt.figure(figsize=(n * 1.6, 3 * 1.9), facecolor="white")
    gs  = gridspec.GridSpec(
        3, n, figure=fig,
        hspace=0.06, wspace=0.04,
        left=0.08, right=0.98, top=0.93, bottom=0.04,
    )

    populations = [
        (quasar_stamps,  "Quasars",        "tomato"),
        (cluster_stamps, "Cluster fields", "cornflowerblue"),
        (desi_stamps,    "DESI galaxies",  "mediumseagreen"),
    ]
    for row_idx, (stamps, label, color) in enumerate(populations):
        axes = [fig.add_subplot(gs[row_idx, col]) for col in range(n)]
        draw_row(axes, stamps, label, color)

    fig.suptitle(
        r"Euclid VIS stamps  ($2''\!\times\!2''$, $20\!\times\!20$ pixels)"
        r"  —  $\log_{10}$ stretch",
        color="black", fontsize=13, y=0.97,
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Selecting Quaia quasars …")
    q_stamps = select_quasars()
    print(f"  {len(q_stamps)} good stamps")

    print("Selecting cluster-field sources …")
    c_stamps = select_cluster_sources()
    print(f"  {len(c_stamps)} good stamps")

    print("Selecting DESI galaxies …")
    d_stamps = select_desi_galaxies()
    print(f"  {len(d_stamps)} good stamps")

    print("Drawing figure …")
    make_montage(q_stamps, c_stamps, d_stamps, THESIS_OUT)
    print("Done.")


if __name__ == "__main__":
    main()
