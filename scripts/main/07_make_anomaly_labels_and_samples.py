"""
07_make_anomaly_labels_and_samples.py

Creates anomaly labels from metadata.csv and produces:
- metadata_labeled.csv (adds flag columns + union label)
- sample CSVs for quick inspection
- (optional) stamp grid PNGs per group

Labels included:
- ruwe_strong: ruwe > RUWE_THRESH
- ipd_strong: ipd_frac_multi_peak > IPD_THRESH (works well if it's 0/1)
- excess_strong: data-driven high phot_bp_rp_excess_factor relative to bp_rp bins
- any_strong: union of the above

Run:
  python -u scripts/07_make_anomaly_labels_and_samples.py
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# PATHS
# =========================
BASE = Path(__file__).resolve().parents[1]  # .../Codes
DATASET_ROOT = BASE / "output" / "dataset_npz"

# Each field folder will contain:
# - metadata.csv
# - stamps_*.npz
# and we will write:
# - metadata_labeled.csv
# - label_samples/*.csv
# Plots go to:
# - plots/labels/<tag>/*.png
GRIDS_ROOT = BASE / "plots" / "labels"
GRIDS_ROOT.mkdir(parents=True, exist_ok=True)


# =========================
# LABEL SETTINGS
# =========================
RUWE_THRESH = 1.4
IPD_THRESH = 0.0  # if ipd_frac_multi_peak is 0/1, >0.0 flags the "multi-peak" ones

# phot_bp_rp_excess_factor labeling (data-driven):
BP_RP_BIN_WIDTH = 0.2
MIN_BIN_N = 30
EXCESS_QUANTILE = 0.95  # top 5% (within each bp_rp bin) flagged as excess_strong

# =========================
# SAMPLING SETTINGS
# =========================
RANDOM_SEED = 0
SAMPLE_N_PER_GROUP = 200

# Optional stamp grids
EXPORT_GRIDS = True
GRID_N = 36          # 6x6
GRID_SIDE = 6
MARK_CENTER = True
MARK_BRIGHTEST = True


# =========================
# Helpers
# =========================
def safe_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_excess_strong(df: pd.DataFrame) -> pd.Series:
    """
    Flag high phot_bp_rp_excess_factor relative to bp_rp.
    Uses per-bin quantile threshold; bins with too few points fall back to global threshold.
    """
    if ("phot_bp_rp_excess_factor" not in df.columns) or ("bp_rp" not in df.columns):
        return pd.Series(False, index=df.index)

    e = df["phot_bp_rp_excess_factor"].astype(float)
    c = df["bp_rp"].astype(float)

    ok = np.isfinite(e) & np.isfinite(c)
    out = pd.Series(False, index=df.index)

    if ok.sum() == 0:
        return out

    c_ok = c[ok]
    e_ok = e[ok]

    # Define bp_rp bins
    cmin = float(np.nanmin(c_ok))
    cmax = float(np.nanmax(c_ok))
    # Expand a bit so max falls in
    edges = np.arange(cmin, cmax + BP_RP_BIN_WIDTH * 1.01, BP_RP_BIN_WIDTH)
    if len(edges) < 3:
        # fallback: global threshold only
        thr = float(np.nanquantile(e_ok, EXCESS_QUANTILE))
        out.loc[ok] = e_ok > thr
        return out

    bin_id = np.digitize(c_ok, edges) - 1  # 0..nbin-1
    nbin = len(edges) - 1

    # Global fallback
    global_thr = float(np.nanquantile(e_ok, EXCESS_QUANTILE))

    strong = np.zeros_like(e_ok, dtype=bool)

    # Per-bin thresholds
    for b in range(nbin):
        m = bin_id == b
        if m.sum() < MIN_BIN_N:
            thr = global_thr
        else:
            thr = float(np.nanquantile(e_ok[m], EXCESS_QUANTILE))
        strong[m] = e_ok[m] > thr

    out.loc[ok] = strong
    return out


def print_overlap_table(df):
    flags = ["ruwe_strong", "ipd_strong", "excess_strong", "any_strong"]
    counts = {f: int(df[f].sum()) for f in flags}
    print("\n=== Label counts ===")
    for k, v in counts.items():
        print(f"{k:>13}: {v}")

    print("\n=== Overlaps (pairwise) ===")
    pairs = [("ruwe_strong", "ipd_strong"), ("ruwe_strong", "excess_strong"), ("ipd_strong", "excess_strong")]
    for a, b in pairs:
        both = int((df[a] & df[b]).sum())
        print(f"{a} & {b}: {both}")

    only_ruwe = int((df["ruwe_strong"] & ~df["ipd_strong"] & ~df["excess_strong"]).sum())
    only_ipd = int((~df["ruwe_strong"] & df["ipd_strong"] & ~df["excess_strong"]).sum())
    only_exc = int((~df["ruwe_strong"] & ~df["ipd_strong"] & df["excess_strong"]).sum())
    clean = int((~df["any_strong"]).sum())

    print("\n=== Exclusive groups ===")
    print("ruwe_only  :", only_ruwe)
    print("ipd_only   :", only_ipd)
    print("excess_only:", only_exc)
    print("clean      :", clean)


def save_samples(df: pd.DataFrame, samples_dir: Path):
    rng = np.random.default_rng(RANDOM_SEED)

    samples_dir.mkdir(parents=True, exist_ok=True)

    groups = {
        "ruwe_only": df["ruwe_strong"] & ~df["ipd_strong"] & ~df["excess_strong"],
        "ipd_only": ~df["ruwe_strong"] & df["ipd_strong"] & ~df["excess_strong"],
        "excess_only": ~df["ruwe_strong"] & ~df["ipd_strong"] & df["excess_strong"],
        "any_strong": df["any_strong"],
        "clean_control": ~df["any_strong"],
    }

    print("\n=== Saving sample CSVs ===")
    for name, mask in groups.items():
        sub = df.loc[mask].copy()
        n = len(sub)
        if n == 0:
            print(f"{name}: 0 rows (skip)")
            continue

        take = min(SAMPLE_N_PER_GROUP, n)
        sub = sub.sample(n=take, random_state=RANDOM_SEED)

        out = samples_dir / f"{name}.csv"
        sub.to_csv(out, index=False)
        print(f"{name}: saved {take}/{n} -> {out}")


def make_grid(df: pd.DataFrame, name: str, field_dir: Path, grids_dir: Path):
    """
    Build a GRID_N stamp grid using npz_file + index_in_file.
    Loads NPZ from the per-field folder (field_dir), not from the dataset root.
    """
    if not {"npz_file", "index_in_file"}.issubset(df.columns):
        print(f"[grid:{name}] missing npz_file/index_in_file -> skip")
        return

    if len(df) == 0:
        print(f"[grid:{name}] 0 rows -> skip")
        return

    grids_dir.mkdir(parents=True, exist_ok=True)

    sub = df.sample(n=min(GRID_N, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    stamps = []
    for _, r in sub.iterrows():
        npz_path = field_dir / str(r["npz_file"])
        idx = int(r["index_in_file"])
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        X = d["X"]
        if idx < 0 or idx >= X.shape[0]:
            continue
        stamps.append(X[idx])

    if len(stamps) == 0:
        print(f"[grid:{name}] no stamps loaded -> skip")
        return

    side = GRID_SIDE
    fig, axes = plt.subplots(side, side, figsize=(10, 10))
    axes = np.ravel(axes)

    for i in range(side * side):
        ax = axes[i]
        ax.axis("off")
        if i >= len(stamps):
            continue

        img = stamps[i]
        vals = img[np.isfinite(img)]
        if vals.size > 10:
            vmin, vmax = np.percentile(vals, [5, 99.5])
        else:
            vmin, vmax = np.nanmin(img), np.nanmax(img)

        ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

        h, w = img.shape
        cx, cy = (w - 1) / 2, (h - 1) / 2
        if MARK_CENTER:
            ax.plot([cx], [cy], marker="+", markersize=8)
        if MARK_BRIGHTEST:
            yy, xx = np.unravel_index(np.nanargmax(img), img.shape)
            ax.plot([xx], [yy], marker="x", markersize=6)

    fig.suptitle(name)
    plt.tight_layout()
    out = grids_dir / f"{name}_grid.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[grid:{name}] saved -> {out}")


def list_field_dirs(dataset_root: Path):
    """
    Supports both:
    - New layout: output/dataset_npz/<tag>/metadata.csv
    - Legacy layout: output/dataset_npz/metadata.csv
    """
    if (dataset_root / "metadata.csv").exists():
        return [dataset_root]
    dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / "metadata.csv").exists()])
    return dirs


def main():
    if not DATASET_ROOT.exists():
        raise RuntimeError(f"dataset_npz folder not found: {DATASET_ROOT}")

    field_dirs = list_field_dirs(DATASET_ROOT)
    if len(field_dirs) == 0:
        raise RuntimeError(f"No field datasets found under: {DATASET_ROOT}")

    print("Found field datasets:", len(field_dirs))
    for d in field_dirs:
        print(" -", d)

    for field_dir in field_dirs:
        tag = field_dir.name if field_dir != DATASET_ROOT else "LEGACY_OR_SINGLE"
        meta_in = field_dir / "metadata.csv"
        if not meta_in.exists():
            print(f"[SKIP] metadata.csv not found: {meta_in}")
            continue

        out_labeled = field_dir / "metadata_labeled.csv"
        samples_dir = field_dir / "label_samples"
        grids_dir = GRIDS_ROOT / tag

        print("\n" + "=" * 70)
        print("FIELD:", tag)
        print("META :", meta_in)
        print("OUT  :", out_labeled)
        print("SAMP :", samples_dir)
        print("GRID :", grids_dir)
        print("=" * 70)

        df = pd.read_csv(meta_in)

        # Ensure numeric types
        df = safe_numeric(df, ["ruwe", "ipd_frac_multi_peak", "phot_bp_rp_excess_factor", "bp_rp"])

        # Basic flags
        df["ruwe_strong"] = np.isfinite(df["ruwe"]) & (df["ruwe"] > RUWE_THRESH)
        df["ipd_strong"] = np.isfinite(df["ipd_frac_multi_peak"]) & (df["ipd_frac_multi_peak"] > IPD_THRESH)

        # Data-driven BP/RP excess flag
        df["excess_strong"] = compute_excess_strong(df)

        # Union
        df["any_strong"] = df["ruwe_strong"] | df["ipd_strong"] | df["excess_strong"]

        # Print summary
        print_overlap_table(df)

        # Save labeled metadata
        df.to_csv(out_labeled, index=False)
        print("\nSaved labeled metadata:", out_labeled)

        # Save sample CSVs
        save_samples(df, samples_dir)

        # Optional stamp grids
        if EXPORT_GRIDS:
            print("\n=== Exporting stamp grids ===")
            make_grid(df.loc[df["ruwe_strong"]], "ruwe_strong", field_dir=field_dir, grids_dir=grids_dir)
            make_grid(df.loc[df["ipd_strong"]], "ipd_strong", field_dir=field_dir, grids_dir=grids_dir)
            make_grid(df.loc[df["excess_strong"]], "excess_strong", field_dir=field_dir, grids_dir=grids_dir)
            make_grid(df.loc[df["any_strong"]], "any_strong", field_dir=field_dir, grids_dir=grids_dir)
            make_grid(df.loc[~df["any_strong"]], "clean_control", field_dir=field_dir, grids_dir=grids_dir)

        print("\nDONE field:", tag)

    print("\nALL FIELDS DONE.")
    print("DATASET ROOT:", DATASET_ROOT)
    print("GRIDS ROOT  :", GRIDS_ROOT)


if __name__ == "__main__":
    main()
