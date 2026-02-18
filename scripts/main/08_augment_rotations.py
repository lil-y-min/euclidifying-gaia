import csv
import math
import numpy as np
import pandas as pd
from pathlib import Path

from astropy.io import fits
from scipy.ndimage import affine_transform


# =========================
# CONFIG
# =========================
BASE = Path(__file__).resolve().parents[1]  # .../Codes

# root dataset folder (contains per-field subfolders)
DATASET_ROOT = BASE / "output" / "dataset_npz"

# augmentation output folder name inside each field folder
AUG_SUBDIR = "aug_rot"
# Only run augmentation for these fields (folder names under output/dataset_npz/)
ONLY_TAGS = {
    "ERO-IC342",
    "ERO-Messier78",
    "ERO-NGC2403",
    "ERO-NGC6254",
    "ERO-NGC6397",
    "ERO-NGC6744",
    "ERO-NGC6822",
    "ERO-Perseus",
    "ERO-Taurus",
}

N_AUG = 8
RANDOM_SEED = 42
NPZ_CHUNK_SIZE = 5000

ANGLE_MIN = 0.0
ANGLE_MAX = 360.0

CLIP_NEGATIVE_TO_ZERO = False

Y_FEATURE_COLS_10D = [
    "feat_log10_snr",
    "feat_ruwe",
    "feat_astrometric_excess_noise",
    "feat_parallax_over_error",
    "feat_visibility_periods_used",
    "feat_ipd_frac_multi_peak",
    "feat_c_star",
    "feat_pm_significance",
    "feat_phot_g_mean_mag",
    "feat_bp_rp",
]

NEEDED_META_COLS = [
    "npz_file", "index_in_file",
    "fits_path",
    "x_pix_round", "y_pix_round",
    "dx_subpix", "dy_subpix",
    "stamp_pix",
    "source_id",
    "split_code",
] + Y_FEATURE_COLS_10D


# =========================
# FITS helpers
# =========================
def open_fits_first_2d(path: str):
    hdul = fits.open(path, memmap=True)
    hdu_idx = None
    for i, h in enumerate(hdul):
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            hdu_idx = i
            break
    if hdu_idx is None:
        hdul.close()
        raise RuntimeError(f"No 2D image HDU found in FITS: {path}")
    return hdul, hdul[hdu_idx].data


def extract_cutout_centered_on_int(data: np.ndarray, cxi: int, cyi: int, size: int):
    half = size // 2
    x0 = cxi - half
    y0 = cyi - half
    x1 = x0 + size
    y1 = y0 + size
    if x0 < 0 or y0 < 0 or x1 > data.shape[1] or y1 > data.shape[0]:
        return None
    cut = data[y0:y1, x0:x1]
    if cut.shape != (size, size):
        return None
    return cut.astype(np.float32)


def compute_big_pix(final_pix: int) -> int:
    # keep it simple: minimal enclosing square, but force ODD so the middle index is clean
    big = int(math.ceil(math.sqrt(2) * final_pix))
    if big % 2 == 0:
        big += 1
    return big


def extract_stamp_from_cutout(cut: np.ndarray, cx: float, cy: float, stamp_pix: int):
    """
    Round(center) and take a stamp_pix square.
    cx, cy are in cutout coordinates.
    """
    half = stamp_pix // 2
    cxi = int(round(cx))
    cyi = int(round(cy))
    x0 = cxi - half
    y0 = cyi - half
    x1 = x0 + stamp_pix
    y1 = y0 + stamp_pix
    if x0 < 0 or y0 < 0 or x1 > cut.shape[1] or y1 > cut.shape[0]:
        return None
    out = cut[y0:y1, x0:x1]
    if out.shape != (stamp_pix, stamp_pix):
        return None
    return out


def rotate_about_subpixel_center(img: np.ndarray, angle_deg: float, cy: float, cx: float):
    """
    Rotate img by angle_deg (CCW) around (cy, cx) in *subpixel* coordinates.
    Uses affine_transform with output->input mapping.
    Coordinates are in (y, x).
    """
    theta = np.deg2rad(angle_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    # matrix mapping output coords -> input coords for rotation by +theta:
    # use rotation by -theta in (y,x) ordering
    M = np.array([[c, -s],
                  [s,  c]], dtype=float)

    center = np.array([cy, cx], dtype=float)
    offset = center - M @ center

    rot = affine_transform(
        img,
        matrix=M,
        offset=offset,
        output_shape=img.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    if CLIP_NEGATIVE_TO_ZERO:
        rot = np.where(rot < 0, 0, rot)

    return rot.astype(np.float32)


# =========================
# Dataset folder helper
# =========================
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


# =========================
# Main
# =========================
def main():
    np.random.seed(RANDOM_SEED)

    if not DATASET_ROOT.exists():
        raise RuntimeError(f"dataset_npz folder not found: {DATASET_ROOT}")

    field_dirs = list_field_dirs(DATASET_ROOT)
    
    if len(field_dirs) == 0:
        raise RuntimeError(f"No field datasets found under: {DATASET_ROOT}")

    if ONLY_TAGS is not None:
        field_dirs = [d for d in field_dirs if d.name in ONLY_TAGS]

    print("Found field datasets:", len(field_dirs))
    for d in field_dirs:
        print(" -", d)

    for field_dir in field_dirs:
        tag = field_dir.name if field_dir != DATASET_ROOT else "LEGACY_OR_SINGLE"

        IN_META = field_dir / "metadata.csv"
        OUT_DIR = field_dir / AUG_SUBDIR
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_META = OUT_DIR / "metadata_aug.csv"

        print("\n" + "=" * 70)
        print("FIELD:", tag)
        print("IN_META:", IN_META)
        print("OUT_DIR:", OUT_DIR)
        print("OUT_META:", OUT_META)
        print("=" * 70)

        meta = pd.read_csv(IN_META)
        missing = [c for c in NEEDED_META_COLS if c not in meta.columns]
        if missing:
            raise RuntimeError(f"{IN_META} missing required columns: {missing}")

        stamp_pix_vals = sorted(meta["stamp_pix"].dropna().unique().tolist())
        if len(stamp_pix_vals) != 1:
            raise RuntimeError(f"[{tag}] Expected one stamp_pix, got {stamp_pix_vals}")
        final_pix = int(stamp_pix_vals[0])

        big_pix = compute_big_pix(final_pix)

        print("Final stamp_pix:", final_pix)
        print("Big rotate_pix:", big_pix, f"(odd, >= ceil(sqrt(2)*{final_pix}))")
        print("N_AUG:", N_AUG, "(per original)")

        # clean outputs (PER FIELD)
        if OUT_META.exists():
            OUT_META.unlink()
        for p in OUT_DIR.glob("stamps_*.npz"):
            try:
                p.unlink()
            except Exception:
                pass

        out_cols = list(meta.columns) + [
            "orig_npz_file", "orig_index_in_file",
            "aug_id", "rot_angle_deg",
            "aug_method", "big_stamp_pix",
        ]

        meta_f = open(OUT_META, "w", newline="")
        writer = csv.DictWriter(meta_f, fieldnames=out_cols)
        writer.writeheader()

        fits_cache = {}

        def get_data(path: str):
            if path not in fits_cache:
                hdul, data = open_fits_first_2d(path)
                fits_cache[path] = (hdul, data)
            return fits_cache[path][1]

        buf_X, buf_y, buf_ids, buf_meta = [], [], [], []
        chunk_idx = 0
        total_out = 0

        drop_big_edge = 0
        drop_crop_fail = 0

        big_half = big_pix // 2  # integer middle index since big_pix is odd

        for _, row in meta.iterrows():
            fits_path = str(row["fits_path"])
            data = get_data(fits_path)

            # integer center + subpixel offsets
            cxi = int(row["x_pix_round"])
            cyi = int(row["y_pix_round"])
            dx = float(row["dx_subpix"])
            dy = float(row["dy_subpix"])

            if not np.isfinite(dx) or not np.isfinite(dy):
                drop_crop_fail += 1
                continue

            big_cut = extract_cutout_centered_on_int(data, cxi, cyi, big_pix)
            if big_cut is None:
                drop_big_edge += 1
                continue

            # true center in cutout coordinates (subpixel)
            cx0 = big_half + dx
            cy0 = big_half + dy

            yvec = np.array([float(row[c]) for c in Y_FEATURE_COLS_10D], dtype=np.float32)
            sid = int(row["source_id"])

            angles = [0.0] + list(np.random.uniform(ANGLE_MIN, ANGLE_MAX, size=N_AUG))

            for aug_id, ang in enumerate(angles):
                if aug_id == 0:
                    rot_big = big_cut
                    rot_angle = 0.0
                else:
                    rot_big = rotate_about_subpixel_center(big_cut, float(ang), cy=cy0, cx=cx0)
                    rot_angle = float(ang)

                stamp = extract_stamp_from_cutout(rot_big, cx=cx0, cy=cy0, stamp_pix=final_pix)
                if stamp is None:
                    drop_crop_fail += 1
                    continue

                buf_X.append(stamp.astype(np.float32))
                buf_y.append(yvec)
                buf_ids.append(sid)

                out_row = dict(row)
                out_row["orig_npz_file"] = str(row["npz_file"])
                out_row["orig_index_in_file"] = int(row["index_in_file"])
                out_row["aug_id"] = int(aug_id)
                out_row["rot_angle_deg"] = float(rot_angle)
                out_row["aug_method"] = "fits_bigcut_rotate_about_subpixel_center"
                out_row["big_stamp_pix"] = int(big_pix)

                out_row["npz_file"] = ""
                out_row["index_in_file"] = ""
                buf_meta.append(out_row)

                if len(buf_X) >= NPZ_CHUNK_SIZE:
                    out_name = f"stamps_{chunk_idx:05d}.npz"
                    out_path = OUT_DIR / out_name

                    Xo = np.stack(buf_X, axis=0)
                    Yo = np.stack(buf_y, axis=0)
                    ido = np.array(buf_ids, dtype=np.int64)

                    np.savez_compressed(out_path, X=Xo, y=Yo, source_id=ido)

                    for j, rrow in enumerate(buf_meta):
                        rrow["npz_file"] = out_name
                        rrow["index_in_file"] = j
                        writer.writerow(rrow)

                    total_out += len(ido)
                    print(f"[SAVE] {out_name} | N={len(ido)} | total_out={total_out}")

                    buf_X.clear(); buf_y.clear(); buf_ids.clear(); buf_meta.clear()
                    chunk_idx += 1

        if len(buf_X) > 0:
            out_name = f"stamps_{chunk_idx:05d}.npz"
            out_path = OUT_DIR / out_name

            Xo = np.stack(buf_X, axis=0)
            Yo = np.stack(buf_y, axis=0)
            ido = np.array(buf_ids, dtype=np.int64)

            np.savez_compressed(out_path, X=Xo, y=Yo, source_id=ido)

            for j, rrow in enumerate(buf_meta):
                rrow["npz_file"] = out_name
                rrow["index_in_file"] = j
                writer.writerow(rrow)

            total_out += len(ido)
            print(f"[FINAL SAVE] {out_name} | N={len(ido)} | total_out={total_out}")

        meta_f.close()

        for path, (hdul, _) in fits_cache.items():
            try:
                hdul.close()
            except Exception:
                pass

        print("\nDONE field:", tag)
        print("OUT_DIR:", OUT_DIR)
        print("OUT_META:", OUT_META)
        print("Dropped originals (big cutout hit edge):", drop_big_edge)
        print("Dropped samples (crop failed):", drop_crop_fail)

    print("\nALL FIELDS DONE.")
    print("DATASET ROOT:", DATASET_ROOT)


if __name__ == "__main__":
    main()
