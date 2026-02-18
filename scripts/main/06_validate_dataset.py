from pathlib import Path
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


# =========================
# CONFIG — PATHS
# =========================
BASE = Path(__file__).resolve().parents[1]  # .../Codes

# Root where your per-field datasets live:
# output/dataset_npz/<tag>/metadata.csv
OUTDIR = BASE / "output" / "dataset_npz"

# Root for plots (each field gets its own subfolder under this)
PLOT_ROOT = BASE / "plots" / "qa" / "dataset_checks"
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

EUCLID_RA_COL = "v_alpha_j2000"
EUCLID_DEC_COL = "v_delta_j2000"
GAIA_RA_COL = "ra"
GAIA_DEC_COL = "dec"

RANDOM_SEED = 42

N_SHOW = 36

N_OVERLAY = 25
OVERLAY_REGION_PIX = 2000

CHECKD_CENTER_PIX = None
CHECKD_CENTER_RADEC_DEG = None
CHECKD_CENTER_SOURCE_ID = None

CHECKE_DOWNSAMPLE = 16
CHECKE_MAX_POINTS = 20000
CHECKE_FIGSIZE = (12, 12)
CHECKE_DPI = 250


# -------------------------
# Helpers
# -------------------------
def open_fits_first_2d(path: str):
    hdul = fits.open(path, memmap=True)
    hdu_idx = None
    for i, h in enumerate(hdul):
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            hdu_idx = i
            break
    if hdu_idx is None:
        hdul.close()
        raise RuntimeError("No 2D image HDU found in FITS.")
    data = hdul[hdu_idx].data
    wcs = WCS(hdul[hdu_idx].header)
    return hdul, data, wcs


def infer_stamp_pix_from_npz(outdir: Path) -> int:
    files = sorted(glob.glob(str(outdir / "stamps_*.npz")))
    if not files:
        raise RuntimeError(f"No stamps_*.npz found in: {outdir}")
    d = np.load(files[0])
    X = d["X"]
    if X.ndim != 3 or X.shape[1] != X.shape[2]:
        raise RuntimeError(f"Unexpected X shape in {files[0]}: {X.shape}")
    return int(X.shape[1])


def choose_fits_path_from_metadata(meta: pd.DataFrame, fallback: str = None) -> str:
    """
    Prefer the fits_path column if present (your exporter writes it).
    If multiple FITS appear, we use the first one but print a warning.
    """
    if "fits_path" in meta.columns:
        paths = [p for p in meta["fits_path"].dropna().astype(str).unique().tolist() if len(p) > 0]
        if len(paths) == 0:
            return fallback
        if len(paths) > 1:
            print("[WARN] metadata contains multiple fits_path values. Using the first for overlay/overview plots.")
            for p in paths[:5]:
                print("  -", p)
        return paths[0]
    return fallback


def print_dataset_definition(field_dir: Path, metadata_csv: Path):
    print("\n=== Dataset definition sanity ===")
    stamp_pix_npz = infer_stamp_pix_from_npz(field_dir)
    print("Field dir:", field_dir)
    print("NPZ inferred stamp_pix:", stamp_pix_npz)

    meta = pd.read_csv(metadata_csv)
    for col in ["stamp_arcsec", "stamp_pix", "pixscale_arcsec_per_pix"]:
        if col not in meta.columns:
            print(f"metadata missing column: {col}")
            return

    uniq_arcsec = sorted(meta["stamp_arcsec"].dropna().unique().tolist())
    uniq_pix = sorted(meta["stamp_pix"].dropna().unique().tolist())
    uniq_scale = sorted(meta["pixscale_arcsec_per_pix"].dropna().unique().tolist())

    print("metadata stamp_arcsec unique:", uniq_arcsec[:10], ("..." if len(uniq_arcsec) > 10 else ""))
    print("metadata stamp_pix unique:", uniq_pix[:10], ("..." if len(uniq_pix) > 10 else ""))
    print("metadata pixscale unique:", uniq_scale[:10], ("..." if len(uniq_scale) > 10 else ""))

    # derived arcsec check (use first row)
    r0 = meta.iloc[0]
    implied = float(r0["stamp_pix"]) * float(r0["pixscale_arcsec_per_pix"])
    print(f"implied stamp arcsec = stamp_pix * pixscale = {implied:.5f}\"")
    print("metadata first row stamp_arcsec:", float(r0["stamp_arcsec"]))


# -------------------------
# Checks
# -------------------------
def quick_grid_plot_npz(field_dir: Path, out_png: Path):
    np.random.seed(RANDOM_SEED)

    files = sorted(glob.glob(str(field_dir / "stamps_*.npz")))
    if not files:
        raise RuntimeError(f"No stamps_*.npz found in: {field_dir}")

    counts = []
    for f in files:
        d = np.load(f)
        counts.append(d["X"].shape[0])
    total = sum(counts)

    print("\n=== Quick stamp grid ===")
    print("NPZ files:", len(files), "| total stamps:", total)

    probs = np.array(counts, dtype=float) / float(total)
    pick_file = np.random.choice(files, p=probs)
    d = np.load(pick_file)
    X = d["X"]

    n = min(N_SHOW, X.shape[0])
    idx = np.random.choice(X.shape[0], size=n, replace=False)
    stamps = X[idx]

    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(10, 10))
    axes = np.array(axes).reshape(side, side)

    for i in range(side * side):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n:
            continue
        im = stamps[i]
        ax.imshow(im, origin="lower", cmap="gray")
        h, w = im.shape
        ax.plot([w / 2.0], [h / 2.0], marker="+", markersize=8)
        yy, xx = np.unravel_index(np.nanargmax(im), im.shape)
        ax.plot([xx], [yy], marker="o", markersize=3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print("Saved:", out_png)


def checkA_dxdy_distribution(meta: pd.DataFrame, out_png: Path):
    dx = meta["dx_subpix"].to_numpy(dtype=float)
    dy = meta["dy_subpix"].to_numpy(dtype=float)

    plt.figure(figsize=(8, 4))
    plt.hist(dx[np.isfinite(dx)], bins=60, alpha=0.6, label="dx")
    plt.hist(dy[np.isfinite(dy)], bins=60, alpha=0.6, label="dy")
    plt.axvline(-0.5, linestyle="--")
    plt.axvline(+0.5, linestyle="--")
    plt.title("Subpixel dx/dy (should lie mostly in [-0.5, 0.5])")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved:", out_png)


def checkB_gaia_vs_euclid_sep(meta: pd.DataFrame):
    eu_ra = meta[EUCLID_RA_COL].to_numpy(dtype=float)
    eu_de = meta[EUCLID_DEC_COL].to_numpy(dtype=float)
    ga_ra = meta[GAIA_RA_COL].to_numpy(dtype=float)
    ga_de = meta[GAIA_DEC_COL].to_numpy(dtype=float)

    ok = np.isfinite(eu_ra) & np.isfinite(eu_de) & np.isfinite(ga_ra) & np.isfinite(ga_de)
    if not np.any(ok):
        print("Check B: no finite Gaia/Euclid coords found.")
        return

    c1 = SkyCoord(eu_ra[ok] * u.deg, eu_de[ok] * u.deg, frame="icrs")
    c2 = SkyCoord(ga_ra[ok] * u.deg, ga_de[ok] * u.deg, frame="icrs")
    sep = c1.separation(c2).arcsec

    print("\n=== Check B: Gaia–Euclid separation ===")
    print("N:", sep.size)
    print("median:", float(np.median(sep)))
    print("95%:", float(np.percentile(sep, 95)))
    print("max:", float(np.max(sep)))


def checkC_split_and_leakage(meta: pd.DataFrame):
    print("\n=== Check C: split counts ===")
    split = meta["split_code"].to_numpy(dtype=int)
    for code, name in [(0, "train"), (1, "val"), (2, "test")]:
        print(f"{name}: {int(np.sum(split == code))}")

    # leakage: source_id appears in >1 split
    grp = meta.groupby("source_id")["split_code"].nunique()
    leaking = grp[grp > 1]
    leak_n = int(len(leaking))
    print("Leakage count (source_id in multiple splits):", leak_n)

    if leak_n > 0:
        print("\nLeak examples (first 10):")
        ex_ids = leaking.index.to_list()[:10]
        for sid in ex_ids:
            sub = meta.loc[meta["source_id"] == sid, ["split_code", "x_pix_round", "y_pix_round", "dist_match"]]
            splits = sorted(sub["split_code"].unique().tolist())
            print(f"  source_id={sid} splits={splits} rows={len(sub)}")
            print(sub.head(5).to_string(index=False))


def checkD_overlay_boxes(meta: pd.DataFrame, fits_path: str, out_png: Path, stamp_pix: int):
    np.random.seed(RANDOM_SEED)

    hdul, data, wcs = open_fits_first_2d(fits_path)
    ny, nx = data.shape

    # choose center
    chosen = "random"
    if CHECKD_CENTER_PIX is not None:
        cx, cy = CHECKD_CENTER_PIX
        chosen = "center_pix"
    elif CHECKD_CENTER_RADEC_DEG is not None:
        ra0, dec0 = CHECKD_CENTER_RADEC_DEG
        sky0 = SkyCoord(ra0 * u.deg, dec0 * u.deg, frame="icrs")
        cx, cy = wcs.world_to_pixel(sky0)
        chosen = "center_radec"
    elif CHECKD_CENTER_SOURCE_ID is not None:
        row = meta.loc[meta["source_id"] == CHECKD_CENTER_SOURCE_ID]
        if len(row) == 0:
            raise RuntimeError(f"source_id not found in metadata: {CHECKD_CENTER_SOURCE_ID}")
        cx = float(row["x_pix_round"].iloc[0])
        cy = float(row["y_pix_round"].iloc[0])
        chosen = "center_source_id"
    else:
        cx = float(meta["x_pix_round"].sample(1, random_state=RANDOM_SEED).iloc[0])
        cy = float(meta["y_pix_round"].sample(1, random_state=RANDOM_SEED).iloc[0])

    print("\n=== Check D debug ===")
    print("Chosen center mode:", chosen)
    print(f"Center (pix): ({cx:.1f}, {cy:.1f})")
    print("STAMP_PIX:", stamp_pix, "| REGION_PIX:", OVERLAY_REGION_PIX)

    x = meta["x_pix_round"].to_numpy(dtype=float)
    y = meta["y_pix_round"].to_numpy(dtype=float)

    m = (np.abs(x - cx) < OVERLAY_REGION_PIX / 2.0) & (np.abs(y - cy) < OVERLAY_REGION_PIX / 2.0)
    sub = meta.loc[m].copy().reset_index(drop=True)
    print("Candidates in region:", len(sub))

    if len(sub) == 0:
        raise RuntimeError("No sources found in region. Increase OVERLAY_REGION_PIX or change center.")

    if len(sub) > N_OVERLAY:
        sub = sub.sample(N_OVERLAY, random_state=RANDOM_SEED).reset_index(drop=True)

    # cutout bounds
    x0 = int(max(0, cx - OVERLAY_REGION_PIX / 2))
    x1 = int(min(nx, cx + OVERLAY_REGION_PIX / 2))
    y0 = int(max(0, cy - OVERLAY_REGION_PIX / 2))
    y1 = int(min(ny, cy + OVERLAY_REGION_PIX / 2))
    print(f"Cutout bounds x[{x0},{x1}) y[{y0},{y1})")

    cut = data[y0:y1, x0:x1]
    vmin, vmax = np.nanpercentile(cut, [5, 99.5])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(cut, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    half = stamp_pix / 2.0
    for i in range(len(sub)):
        xi = float(sub["x_pix_round"].iloc[i]) - x0
        yi = float(sub["y_pix_round"].iloc[i]) - y0
        rect = plt.Rectangle((xi - half, yi - half), stamp_pix, stamp_pix, fill=False, linewidth=1, color="red")
        ax.add_patch(rect)

    ax.set_title(f"Overlay {len(sub)} stamp boxes (stamp_pix={stamp_pix})")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close(fig)
    hdul.close()
    print("Saved:", out_png)


def checkE_fullframe_overview_points(meta: pd.DataFrame, fits_path: str, out_png: Path, stamp_pix: int):
    np.random.seed(RANDOM_SEED)
    hdul, data, wcs = open_fits_first_2d(fits_path)

    img = data[::CHECKE_DOWNSAMPLE, ::CHECKE_DOWNSAMPLE]
    vmin, vmax = np.nanpercentile(img, [5, 99.5])

    x = meta["x_pix_round"].to_numpy(dtype=float)
    y = meta["y_pix_round"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    meta2 = meta.loc[ok].copy().reset_index(drop=True)

    n0 = len(x)
    if n0 > CHECKE_MAX_POINTS:
        idx = np.random.choice(n0, size=CHECKE_MAX_POINTS, replace=False)
        x = x[idx]; y = y[idx]
        meta2 = meta2.loc[idx].copy().reset_index(drop=True)

    xs = x / CHECKE_DOWNSAMPLE
    ys = y / CHECKE_DOWNSAMPLE

    print("\n=== Check E debug ===")
    print("Full image shape:", data.shape, "| downsample:", CHECKE_DOWNSAMPLE, "| shown image shape:", img.shape)
    print("Points: original=", n0, "| plotted=", len(xs))
    if "split_code" in meta2.columns:
        for code, name in [(0, "train"), (1, "val"), (2, "test")]:
            print(f"  plotted {name}: {int(np.sum(meta2['split_code'].to_numpy(dtype=int) == code))}")

    fig = plt.figure(figsize=CHECKE_FIGSIZE)
    ax = fig.add_subplot(111)
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    if "split_code" in meta2.columns:
        split = meta2["split_code"].to_numpy(dtype=int)
        for code, label in [(0, "train"), (1, "val"), (2, "test")]:
            m = (split == code)
            if np.any(m):
                ax.scatter(xs[m], ys[m], s=4, marker=".", alpha=0.6, label=label)
        ax.legend(loc="upper right", frameon=True)
    else:
        ax.scatter(xs, ys, s=4, marker=".", alpha=0.6, label="stamp centers")
        ax.legend(loc="upper right", frameon=True)

    ax.set_title(f"Full-frame overview (downsample={CHECKE_DOWNSAMPLE}, N={len(xs)}, stamp_pix={stamp_pix})")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=CHECKE_DPI)
    plt.close(fig)
    hdul.close()
    print("Saved:", out_png)


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
    if not OUTDIR.exists():
        raise RuntimeError(f"Dataset folder not found: {OUTDIR}")

    field_dirs = list_field_dirs(OUTDIR)
    if len(field_dirs) == 0:
        raise RuntimeError(f"No field datasets found under: {OUTDIR}")

    print("Found field datasets:", len(field_dirs))
    for d in field_dirs:
        print(" -", d)

    for field_dir in field_dirs:
        tag = field_dir.name if field_dir != OUTDIR else "LEGACY_OR_SINGLE"

        metadata_csv = field_dir / "metadata.csv"
        if not metadata_csv.exists():
            print(f"[SKIP] No metadata.csv in {field_dir}")
            continue

        plot_dir = PLOT_ROOT / tag
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_grid = plot_dir / "quick_visual_check.png"
        plot_dxdy = plot_dir / "dxdy_hist.png"
        plot_checkd = plot_dir / "overlay_boxes.png"
        plot_checke = plot_dir / "fullframe_overview.png"

        print("\n" + "=" * 70)
        print(f"VALIDATING FIELD: {tag}")
        print("DATA :", field_dir)
        print("META :", metadata_csv)
        print("PLOTS:", plot_dir)
        print("=" * 70)

        print_dataset_definition(field_dir, metadata_csv)

        meta = pd.read_csv(metadata_csv)

        # Per-field FITS path
        fits_path = choose_fits_path_from_metadata(meta, fallback=None)
        if fits_path is None:
            print("[WARN] No fits_path found in metadata; skipping checks D/E that require the FITS image.")
        else:
            if not Path(fits_path).exists():
                print("[WARN] fits_path from metadata does not exist on disk; skipping checks D/E.")
                print("  fits_path:", fits_path)
                fits_path = None

        quick_grid_plot_npz(field_dir, plot_grid)
        checkA_dxdy_distribution(meta, plot_dxdy)
        checkB_gaia_vs_euclid_sep(meta)
        checkC_split_and_leakage(meta)

        stamp_pix = infer_stamp_pix_from_npz(field_dir)

        if fits_path is not None:
            checkD_overlay_boxes(meta, fits_path, plot_checkd, stamp_pix=stamp_pix)
            checkE_fullframe_overview_points(meta, fits_path, plot_checke, stamp_pix=stamp_pix)

        print("\nDONE field:", tag)

    print("\nALL FIELDS DONE.")
    print("ROOT OUTDIR:", OUTDIR)
    print("PLOT ROOT :", PLOT_ROOT)


if __name__ == "__main__":
    main()
