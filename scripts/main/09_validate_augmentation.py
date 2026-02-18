import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# CONFIG
# =========================
BASE = Path(__file__).resolve().parents[1]  # .../Codes

# layout: output/dataset_npz/<tag>/aug_rot/{metadata_aug.csv, stamps_*.npz}
DATASET_ROOT = BASE / "output" / "dataset_npz"
AUG_SUBDIR = "aug_rot"

PLOT_ROOT = BASE / "plots" / "qa" / "augmentation_checks"
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 84
N_SOURCES_PLOT = 4
EXPECTED_PER_ORIG = 9
SAMPLE_FOR_BORDER = 4000

BORDER_W = 2


# =========================
# Helpers
# =========================
def list_npz_files(folder: Path):
    return sorted(glob.glob(str(folder / "stamps_*.npz")))


def infer_stamp_pix_from_npz(npz_path: str) -> int:
    d = np.load(npz_path)
    X = d["X"]
    if X.ndim != 3 or X.shape[1] != X.shape[2]:
        raise RuntimeError(f"Unexpected X shape in {npz_path}: {X.shape}")
    return int(X.shape[1])


def load_stamps_by_rows(meta_rows: pd.DataFrame, in_dir: Path):
    out = []
    for npz_name, g in meta_rows.groupby("npz_file", sort=False):
        path = in_dir / npz_name
        d = np.load(path)
        X = d["X"]
        for _, r in g.iterrows():
            idx = int(r["index_in_file"])
            out.append((X[idx].astype(np.float32), r))
    return out


def border_ratio(stamps: np.ndarray, border_w: int = 2) -> np.ndarray:
    eps = 1e-12
    H, W = stamps.shape[1], stamps.shape[2]
    m = np.zeros((H, W), dtype=bool)
    m[:border_w, :] = True
    m[-border_w:, :] = True
    m[:, :border_w] = True
    m[:, -border_w:] = True

    border = np.mean(np.abs(stamps[:, m]), axis=1)
    interior = np.mean(np.abs(stamps[:, ~m]), axis=1)
    return border / (interior + eps)


def brightest_offsets(stamps: np.ndarray):
    """
    Return dx, dy of brightest pixel relative to stamp center (in pixels).
    For even sizes (20), "center" is between pixels, so use (W/2, H/2).
    """
    n, h, w = stamps.shape
    cx = w / 2.0
    cy = h / 2.0
    dx = np.zeros(n, dtype=float)
    dy = np.zeros(n, dtype=float)
    for i in range(n):
        im = stamps[i]
        yy, xx = np.unravel_index(np.nanargmax(im), im.shape)
        dx[i] = float(xx - cx)
        dy[i] = float(yy - cy)
    return dx, dy


def list_field_dirs(dataset_root: Path):
    """
    New layout: output/dataset_npz/<tag>/metadata.csv
    Legacy layout: output/dataset_npz/metadata.csv
    """
    if (dataset_root / "metadata.csv").exists():
        return [dataset_root]
    return sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / "metadata.csv").exists()])


def pct(v, p):
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if v.size else np.nan


# =========================
# Main validations
# =========================
def validate_one_field(tag: str, in_dir: Path, meta_path: Path, plot_dir: Path):
    np.random.seed(RANDOM_SEED)

    plot_dir.mkdir(parents=True, exist_ok=True)
    PLOT_GRID = plot_dir / "rotation_grid.png"
    PLOT_BORDER = plot_dir / "border_ratio_hist.png"
    PLOT_BRIGHT_OFF = plot_dir / "brightest_offset_hist.png"

    npz_files = list_npz_files(in_dir)
    if not npz_files:
        print(f"[SKIP] No stamps_*.npz found in {in_dir}")
        return

    stamp_pix_npz = infer_stamp_pix_from_npz(npz_files[0])
    print("\n=== Augmented dataset definition ===")
    print("FIELD:", tag)
    print("IN_DIR:", in_dir)
    print("META  :", meta_path)
    print("NPZ files:", len(npz_files))
    print("NPZ inferred stamp_pix:", stamp_pix_npz)

    meta = pd.read_csv(meta_path)
    print("metadata_aug rows:", len(meta))

    for col in ["stamp_arcsec", "stamp_pix", "pixscale_arcsec_per_pix", "aug_method", "big_stamp_pix"]:
        if col in meta.columns:
            uniq = sorted(meta[col].dropna().unique().tolist())
            print(f"metadata {col} unique:", uniq[:10], ("..." if len(uniq) > 10 else ""))

    if all(c in meta.columns for c in ["stamp_pix", "pixscale_arcsec_per_pix"]):
        implied = float(meta["stamp_pix"].iloc[0]) * float(meta["pixscale_arcsec_per_pix"].iloc[0])
        print(f'implied stamp arcsec = stamp_pix * pixscale = {implied:.5f}"')

    print("\n=== Completeness + invariants per original ===")
    need = ["orig_npz_file", "orig_index_in_file", "aug_id", "split_code", "source_id"]
    missing = [c for c in need if c not in meta.columns]
    if missing:
        raise RuntimeError(f"metadata_aug is missing columns: {missing}")

    g = meta.groupby(["orig_npz_file", "orig_index_in_file"], sort=False)
    counts = g.size().to_numpy()
    print("per-original count: min =", int(np.min(counts)), "max =", int(np.max(counts)))
    bad_count = int(np.sum(counts != EXPECTED_PER_ORIG))
    print(f"originals with count != {EXPECTED_PER_ORIG}:", bad_count)

    bad_split = int(np.sum(g["split_code"].nunique().to_numpy() != 1))
    bad_sid = int(np.sum(g["source_id"].nunique().to_numpy() != 1))
    print("originals with split_code not constant:", bad_split)
    print("originals with source_id not constant:", bad_sid)

    # Feature invariance (sample)
    feat_cols = [c for c in meta.columns if c.startswith("feat_")]
    if feat_cols and len(g) > 0:
        sample_groups = min(500, len(g))
        keys = list(g.groups.keys())
        pick = np.random.choice(len(keys), size=sample_groups, replace=False)
        max_ranges = {c: 0.0 for c in feat_cols}
        for i in pick:
            sub = g.get_group(keys[i])
            for c in feat_cols:
                v = sub[c].to_numpy(dtype=float)
                if np.all(np.isfinite(v)):
                    r = float(np.max(v) - np.min(v))
                    if r > max_ranges[c]:
                        max_ranges[c] = r
        worst = sorted(max_ranges.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("\nFeature invariance (sampled groups): worst max range across augments")
        for name, rng in worst:
            print(f"  {name}: {rng:.3e}")

    # Border artefact check
    print("\n=== Border artefact quick check ===")
    n = len(meta)
    if n == 0:
        print("[SKIP] metadata_aug has 0 rows.")
        return

    sample_n = min(SAMPLE_FOR_BORDER, n)
    sample_idx = np.random.choice(n, size=sample_n, replace=False)
    samp = meta.iloc[sample_idx].copy().reset_index(drop=True)

    stamps = []
    aug_id = samp["aug_id"].to_numpy(dtype=int)
    for npz_name, sub in samp.groupby("npz_file", sort=False):
        path = in_dir / npz_name
        d = np.load(path)
        X = d["X"]
        idxs = sub["index_in_file"].to_numpy(dtype=int)
        stamps.append(X[idxs].astype(np.float32))
    stamps = np.concatenate(stamps, axis=0)

    ratios = border_ratio(stamps, border_w=BORDER_W)
    r0 = ratios[aug_id == 0]
    rR = ratios[aug_id != 0]

    print("border_ratio = mean(|border|)/mean(|interior|), BORDER_W =", BORDER_W)
    print("originals: N =", int(np.sum(aug_id == 0)),
          "| median =", pct(r0, 50), "| 95% =", pct(r0, 95))
    print("rotated:   N =", int(np.sum(aug_id != 0)),
          "| median =", pct(rR, 50), "| 95% =", pct(rR, 95))

    plt.figure(figsize=(8, 4))
    plt.hist(r0[np.isfinite(r0)], bins=60, alpha=0.6, label="aug_id=0 (original)")
    plt.hist(rR[np.isfinite(rR)], bins=60, alpha=0.6, label="aug_id>0 (rotated)")
    plt.xlabel("border_ratio")
    plt.ylabel("count")
    plt.title("Border ratio hist: original vs rotated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_BORDER, dpi=200)
    plt.close()
    print("Saved:", PLOT_BORDER)

    # brightest pixel offset debug plot
    dx, dy = brightest_offsets(stamps)
    dx0, dy0 = dx[aug_id == 0], dy[aug_id == 0]
    dxR, dyR = dx[aug_id != 0], dy[aug_id != 0]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(dx0[np.isfinite(dx0)], bins=40, alpha=0.6, label="orig")
    plt.hist(dxR[np.isfinite(dxR)], bins=40, alpha=0.6, label="rot")
    plt.title("Brightest-pixel dx from center")
    plt.xlabel("dx (pix)")
    plt.ylabel("count")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(dy0[np.isfinite(dy0)], bins=40, alpha=0.6, label="orig")
    plt.hist(dyR[np.isfinite(dyR)], bins=40, alpha=0.6, label="rot")
    plt.title("Brightest-pixel dy from center")
    plt.xlabel("dy (pix)")
    plt.ylabel("count")
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_BRIGHT_OFF, dpi=200)
    plt.close()
    print("Saved:", PLOT_BRIGHT_OFF)

    # Visual rotation grid
    print("\n=== Visual rotation grid (with center + brightest pixel markers) ===")
    keys = list(g.groups.keys())
    if len(keys) == 0:
        print("[SKIP] No groups to plot.")
        return

    n_plot = min(N_SOURCES_PLOT, len(keys))
    pick = np.random.choice(len(keys), size=n_plot, replace=False)

    rows = []
    for i in pick:
        sub = g.get_group(keys[i]).sort_values("aug_id", ascending=True).copy()
        sub = sub.head(EXPECTED_PER_ORIG)
        rows.append(sub)

    to_load = pd.concat(rows, axis=0).reset_index(drop=True)
    loaded = load_stamps_by_rows(to_load, in_dir=in_dir)

    fig, axes = plt.subplots(n_plot, EXPECTED_PER_ORIG, figsize=(2.2 * EXPECTED_PER_ORIG, 2.2 * n_plot))
    if n_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    ptr = 0
    for r in range(n_plot):
        orig_stamp = loaded[ptr][0]
        vmin, vmax = np.nanpercentile(orig_stamp, [5, 99.5])

        for c in range(EXPECTED_PER_ORIG):
            ax = axes[r, c]
            ax.axis("off")

            stamp, meta_row = loaded[ptr]
            ax.imshow(stamp, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

            h, w = stamp.shape
            ax.plot([w / 2.0], [h / 2.0], marker="+", markersize=7, color="blue")

            yy, xx = np.unravel_index(np.nanargmax(stamp), stamp.shape)
            ax.plot([xx], [yy], marker="o", markersize=3, color="red")

            ang = float(meta_row.get("rot_angle_deg", 0.0))
            aid = int(meta_row.get("aug_id", -1))
            if r == 0:
                ax.set_title(f"aug={aid}\n{ang:.0f}°", fontsize=9)

            ptr += 1

    plt.tight_layout()
    plt.savefig(PLOT_GRID, dpi=200)
    plt.close(fig)
    print("Saved:", PLOT_GRID)

    print("\nDONE FIELD:", tag)
    print("=========================")


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

        in_dir = field_dir / AUG_SUBDIR
        meta_path = in_dir / "metadata_aug.csv"

        if not in_dir.exists():
            print(f"[SKIP] No augmentation folder for field {tag}: {in_dir}")
            continue
        if not meta_path.exists():
            print(f"[SKIP] No metadata_aug.csv for field {tag}: {meta_path}")
            continue

        plot_dir = PLOT_ROOT / tag
        validate_one_field(tag=tag, in_dir=in_dir, meta_path=meta_path, plot_dir=plot_dir)

    print("\nALL FIELDS DONE.")
    print("DATASET ROOT:", DATASET_ROOT)
    print("PLOT ROOT   :", PLOT_ROOT)


if __name__ == "__main__":
    main()
