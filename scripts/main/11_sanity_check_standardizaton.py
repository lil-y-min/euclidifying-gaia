"""
11_sanity_check_standardization.py

Sanity-check your planned standardization:
1) Images: normalize each stamp by its (positive) integral
2) Gaia features: apply (Y - min) / IQR using a precomputed scaler NPZ

Outputs (per field, if MAKE_PLOTS=True):
- plots/standardization_checks/<FIELD>/integral_normalization_<FIELD>.png
- plots/standardization_checks/<FIELD>/stamp_grid_normalization_<FIELD>.png

Run:
  python -u scripts/11_sanity_check_standardization.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
BASE = Path(__file__).resolve().parents[1]  # .../Codes

DATASET_ROOT = BASE / "output" / "dataset_npz"
SCALER_NPZ = BASE / "output" / "scalers" / "y_feature_iqr.npz"  # your scaler file

USE_AUG = False
AUG_SUBDIR = "aug_rot"  # your real augmentation folder name
META_NAME = "metadata_aug.csv" if USE_AUG else "metadata.csv"

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

TRAIN_SPLIT_CODE = 0

EPS_INTEGRAL = 1e-12
USE_POSITIVE_ONLY = True

MAX_STAMPS_PER_FIELD_FOR_IMG_STATS = 50_000
MAX_STAMPS_PER_FIELD_FOR_Y_STATS = 200_000
RANDOM_SEED = 123

MAKE_PLOTS = True
PLOT_ROOT = BASE / "plots" / "qa" / "standardization_checks"

MAKE_STAMP_GRID = True
GRID_N = 36
GRID_SIDE = 6
GRID_DPI = 200


# =========================
# Helpers
# =========================
def list_field_dirs(dataset_root: Path):
    """
    New layout: output/dataset_npz/<tag>/metadata.csv
    Legacy layout: output/dataset_npz/metadata.csv
    """
    # Legacy / single
    if (dataset_root / META_NAME).exists():
        return [dataset_root]

    # Per-field
    out = []
    for p in sorted(dataset_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / META_NAME).exists():
            out.append(p)
    return out


def load_scaler(npz_path: Path):
    if not npz_path.exists():
        raise RuntimeError(f"Scaler file not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    y_min = d["y_min"].astype(float)
    y_iqr = d["y_iqr"].astype(float)

    name_to_idx = {n: i for i, n in enumerate(names)}
    missing = [c for c in Y_FEATURE_COLS_10D if c not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing feature columns in scaler file: {missing}")

    idxs = np.array([name_to_idx[c] for c in Y_FEATURE_COLS_10D], dtype=int)
    return dict(y_min=y_min[idxs], y_iqr=y_iqr[idxs])


def get_dataset_paths(field_dir: Path):
    in_dir = (field_dir / AUG_SUBDIR) if USE_AUG else field_dir
    meta_path = in_dir / META_NAME
    return in_dir, meta_path


def positive_integral_per_stamp(X: np.ndarray):
    if USE_POSITIVE_ONLY:
        Xp = np.clip(X, 0.0, np.inf)
        return np.sum(Xp, axis=(1, 2))
    return np.sum(X, axis=(1, 2))


def normalize_by_integral(X: np.ndarray, integrals: np.ndarray):
    Xn = X.astype(np.float32, copy=True)
    ok = np.isfinite(integrals) & (integrals > EPS_INTEGRAL)
    Xn[ok] /= integrals[ok, None, None]
    return Xn, ok


def robust_summary(v: np.ndarray, name=""):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return f"{name}: N=0"
    p = np.percentile(v, [0, 1, 5, 50, 95, 99, 100])
    return (
        f"{name}: N={v.size}, "
        f"min={p[0]:.6g}, p1={p[1]:.6g}, p5={p[2]:.6g}, "
        f"med={p[3]:.6g}, p95={p[4]:.6g}, p99={p[5]:.6g}, max={p[6]:.6g}"
    )


def load_stamps_by_meta_rows(meta_rows: pd.DataFrame, in_dir: Path):
    """
    meta_rows must contain npz_file and index_in_file.
    Loads stamps from all involved npz files and returns stacked X subset.
    """
    stamps = []
    for npz_name, sub in meta_rows.groupby("npz_file", sort=False):
        path = in_dir / str(npz_name)
        if not path.exists():
            continue
        with np.load(path) as d:
            X = d["X"]
            idxs = pd.to_numeric(sub["index_in_file"], errors="coerce").to_numpy(dtype=int)
            good = (idxs >= 0) & (idxs < X.shape[0])
            idxs = idxs[good]
            if idxs.size == 0:
                continue
            stamps.append(X[idxs].astype(np.float32))

    if len(stamps) == 0:
        return None
    return np.concatenate(stamps, axis=0)


def safe_hist(ax, data, bins=60, density=True, label=None, alpha=0.6):
    """
    Histogram that won't crash when data has ~zero range (e.g. nearly constant).
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    mn = float(np.min(x))
    mx = float(np.max(x))
    if not (np.isfinite(mn) and np.isfinite(mx)):
        return

    if (mx - mn) < 1e-9:
        eps = max(1e-6, abs(mn) * 1e-6)
        ax.hist(x, bins=[mn - eps, mn + eps], density=density, alpha=alpha, label=label)
    else:
        ax.hist(x, bins=bins, density=density, alpha=alpha, label=label)


def make_stamp_grid(before: np.ndarray, after: np.ndarray, out_png: Path, title: str):
    """
    2*side by side grid:
    - top: BEFORE
    - bottom: AFTER
    Each stamp is contrast-stretched independently for visibility (so “after” won’t look black).
    """
    n = min(GRID_N, before.shape[0], after.shape[0])
    side = GRID_SIDE
    if n < side * side:
        side = int(np.ceil(np.sqrt(n)))

    fig, axes = plt.subplots(2 * side, side, figsize=(10, 20))
    axes = np.array(axes)

    def stretch(img):
        v = img[np.isfinite(img)]
        if v.size < 10:
            return float(np.nanmin(img)), float(np.nanmax(img))
        vmin, vmax = np.percentile(v, [5, 99.5])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
            vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))
        return float(vmin), float(vmax)

    for i in range(side * side):
        r = i // side
        c = i % side

        ax1 = axes[r, c]
        ax1.axis("off")
        if i < n:
            vminb, vmaxb = stretch(before[i])
            ax1.imshow(before[i], origin="lower", cmap="gray", vmin=vminb, vmax=vmaxb)
            h, w = before[i].shape
            ax1.plot([w / 2.0], [h / 2.0], marker="+", color="red", markersize=6)

        ax2 = axes[r + side, c]
        ax2.axis("off")
        if i < n:
            vmina, vmaxa = stretch(after[i])
            ax2.imshow(after[i], origin="lower", cmap="gray", vmin=vmina, vmax=vmaxa)
            h, w = after[i].shape
            ax2.plot([w / 2.0], [h / 2.0], marker="+", color="red", markersize=6)

    fig.suptitle(title + "\nNOTE: each stamp is independently contrast-stretched for visibility")
    plt.tight_layout()
    plt.savefig(out_png, dpi=GRID_DPI)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    if not DATASET_ROOT.exists():
        raise RuntimeError(f"Dataset root not found: {DATASET_ROOT}")

    scaler = load_scaler(SCALER_NPZ)
    y_min = scaler["y_min"]
    y_iqr = scaler["y_iqr"]

    print("\n=== Loaded scaler info ===")
    print("SCALER:", SCALER_NPZ)
    for name, mn, iq in zip(Y_FEATURE_COLS_10D, y_min, y_iqr):
        print(f"{name:>28}: min={mn: .6g} | iqr={iq: .6g}")

    field_dirs = list_field_dirs(DATASET_ROOT)
    if len(field_dirs) == 0:
        raise RuntimeError(f"No field datasets found under: {DATASET_ROOT}")

    print("\nFound field datasets:", len(field_dirs))
    for d in field_dirs:
        print(" -", d)

    if MAKE_PLOTS:
        PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    for field_dir in field_dirs:
        tag = field_dir.name if field_dir != DATASET_ROOT else "LEGACY_OR_SINGLE"
        in_dir, meta_path = get_dataset_paths(field_dir)

        if not meta_path.exists():
            print(f"[SKIP] {tag}: metadata not found: {meta_path}")
            continue

        print("\n" + "=" * 80)
        print("FIELD:", tag)
        print("IN_DIR:", in_dir)
        print("META  :", meta_path)
        print("=" * 80)

        # read metadata (only columns we need)
        head = pd.read_csv(meta_path, nrows=1)
        need = ["npz_file", "index_in_file", "split_code"] + Y_FEATURE_COLS_10D
        missing = [c for c in need if c not in head.columns]
        if missing:
            raise RuntimeError(f"{tag}: missing columns in metadata: {missing}")

        meta = pd.read_csv(meta_path, usecols=need, low_memory=False)

        # enforce numeric
        meta["split_code"] = pd.to_numeric(meta["split_code"], errors="coerce")
        meta["index_in_file"] = pd.to_numeric(meta["index_in_file"], errors="coerce")
        for c in Y_FEATURE_COLS_10D:
            meta[c] = pd.to_numeric(meta[c], errors="coerce")

        n_all = len(meta)
        n_train = int(np.sum(meta["split_code"].to_numpy(dtype=float) == float(TRAIN_SPLIT_CODE)))
        print(f"Rows: all={n_all} | train={n_train}")

        # ---------- Image stats ----------
        meta_ok = meta.loc[meta["npz_file"].notna() & meta["index_in_file"].notna()].copy()
        if len(meta_ok) == 0:
            print(f"[WARN] {tag}: no valid rows with npz_file/index_in_file.")
            continue

        take_img = min(MAX_STAMPS_PER_FIELD_FOR_IMG_STATS, len(meta_ok))
        samp_img = meta_ok.sample(n=take_img, random_state=RANDOM_SEED).reset_index(drop=True)

        X = load_stamps_by_meta_rows(samp_img, in_dir=in_dir)
        if X is None:
            print(f"[WARN] {tag}: could not load stamps for image stats.")
            continue

        integrals = positive_integral_per_stamp(X)
        Xn, ok_norm = normalize_by_integral(X, integrals)
        integrals_after = positive_integral_per_stamp(Xn)

        print("\n=== Image integrals summary ===")
        print(robust_summary(integrals, name="Before normalization"))
        print(robust_summary(integrals_after, name="After normalization"))
        frac_bad = 1.0 - (np.sum(ok_norm) / float(len(ok_norm)))
        print(f"fraction with integral <= {EPS_INTEGRAL} or non-finite: {frac_bad:.6g}")

        if np.any(ok_norm):
            v = integrals_after[ok_norm]
            print("normalized stamps: median integral_after =", float(np.median(v)),
                  "| p95 =", float(np.percentile(v, 95.0)))

        # ---------- y scaling stats (train only) ----------
        train = meta.loc[meta["split_code"] == float(TRAIN_SPLIT_CODE)].copy()
        train = train.dropna(subset=Y_FEATURE_COLS_10D)
        if len(train) > 0:
            take_y = min(MAX_STAMPS_PER_FIELD_FOR_Y_STATS, len(train))
            if take_y < len(train):
                train = train.sample(n=take_y, random_state=RANDOM_SEED).reset_index(drop=True)

            Y = train[Y_FEATURE_COLS_10D].to_numpy(dtype=float)
            Y_scaled = (Y - y_min[None, :]) / y_iqr[None, :]

            print(f"\n=== Gaia feature values summary (scaled, train only, n={len(train)}) ===")
            for j, name in enumerate(Y_FEATURE_COLS_10D):
                col = Y_scaled[:, j]
                col = col[np.isfinite(col)]
                if col.size == 0:
                    print(f"{name:>28}: N=0")
                    continue
                p = np.percentile(col, [0, 1, 5, 50, 95, 99, 100])
                print(
                    f"{name:>28}: N={col.size} | "
                    f"min={p[0]:.6g}, p1={p[1]:.6g}, p5={p[2]:.6g}, "
                    f"med={p[3]:.6g}, p95={p[4]:.6g}, p99={p[5]:.6g}, max={p[6]:.6g}"
                )
        else:
            print(f"\n[WARN] {tag}: No valid train rows with finite features.")

        # ---------- Plots ----------
        if MAKE_PLOTS:
            plot_dir = PLOT_ROOT / tag
            plot_dir.mkdir(parents=True, exist_ok=True)

            # --- Integral plots ---
            out_int = plot_dir / f"integral_normalization_{tag}.png"
            fig = plt.figure(figsize=(12, 4))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            b = integrals[np.isfinite(integrals)]
            a = integrals_after[np.isfinite(integrals_after)]

            # Panel 1: show BEFORE distribution in log10; show AFTER as a vertical line at 0
            b_log = np.log10(b[b > 0]) if b.size else np.array([])
            safe_hist(ax1, b_log, bins=60, density=True, label="before (log10)", alpha=0.6)
            ax1.axvline(0.0, linewidth=2, color="orange", label="after (log10 ≈ 0)")
            ax1.set_xlabel("log10(positive integral)")
            ax1.set_ylabel("density")
            ax1.set_title("Log-scale view")
            ax1.legend()

            # Panel 2: show numerical spread of AFTER around 1 (ppm)
            if a.size:
                delta_ppm = (a - 1.0) * 1e6
                safe_hist(ax2, delta_ppm, bins=60, density=True,
                          label="after: (integral_after - 1) in ppm", alpha=0.6)
            ax2.axvline(0.0, linewidth=1, color="k")
            ax2.set_xlabel("(integral_after - 1) [ppm]")
            ax2.set_ylabel("density")
            ax2.set_title("After: tiny numerical spread")
            ax2.legend()

            plt.tight_layout()
            plt.savefig(out_int, dpi=200)
            plt.close(fig)
            print("Saved:", out_int)

            # --- Stamp grid ---
            if MAKE_STAMP_GRID:
                out_grid = plot_dir / f"stamp_grid_normalization_{tag}.png"
                idx_pool = np.where(ok_norm)[0]
                if idx_pool.size >= GRID_N:
                    pick = rng.choice(idx_pool, size=GRID_N, replace=False)
                else:
                    pick = rng.choice(np.arange(X.shape[0]), size=min(GRID_N, X.shape[0]), replace=False)

                make_stamp_grid(
                    before=X[pick],
                    after=Xn[pick],
                    out_png=out_grid,
                    title=f"{tag}: Stamp grid before/after normalization",
                )
                print("Saved:", out_grid)

    print("\n=== DONE ===")
    print("Dataset root:", DATASET_ROOT)
    if MAKE_PLOTS:
        print("Plots root:", PLOT_ROOT)


if __name__ == "__main__":
    main()
