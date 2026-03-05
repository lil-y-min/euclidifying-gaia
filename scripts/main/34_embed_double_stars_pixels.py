#!/usr/bin/env python3
"""
Embed normalized stamp pixels into 2D with UMAP and overlay WDS double labels.

Inputs:
- output/dataset_npz/*/metadata.csv
- output/dataset_npz/*/*.npz  (expects key "X": [N, H, W])
- output/crossmatch/wds/wds_xmatch/*_wds_best.csv

Outputs:
- plots/qa/embeddings/double_stars_pixels/umap_<scaling>/
- output/experiments/embeddings/double_stars_pixels/umap_<scaling>/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:
    gaussian_filter = None


FEATURE_COLS_8D = [
    "feat_log10_snr",
    "feat_ruwe",
    "feat_astrometric_excess_noise",
    "feat_parallax_over_error",
    "feat_visibility_periods_used",
    "feat_ipd_frac_multi_peak",
    "feat_c_star",
    "feat_pm_significance",
]

MORPH_FEATURE_COLS = [
    "morph_centroid_dx",
    "morph_centroid_dy",
    "morph_centroid_dr",
    "morph_asym_180",
    "morph_mirror_asym_lr",
    "morph_concentration_ratio_r1r2",
    "morph_concentration_r80r20",
    "morph_mxx",
    "morph_myy",
    "morph_mxy",
    "morph_ellipticity_e1",
    "morph_ellipticity_e2",
    "morph_axis_ratio_q",
    "morph_theta_rad",
    "morph_roundness",
    "morph_peakedness_kurtosis",
    "morph_peak_to_total",
    "morph_texture_laplacian",
    "morph_texture_gradient",
    "morph_gini",
    "morph_m20",
    "morph_smoothness",
    "morph_edge_asym_180",
]

MORPH_PLOT06_COLS = [
    "morph_asym_180",
    "morph_mirror_asym_lr",
    "morph_concentration_r80r20",
    "morph_ellipticity_e2",
    "morph_peakedness_kurtosis",
    "morph_roundness",
    "morph_texture_laplacian",
    "morph_gini",
    "morph_m20",
    "morph_smoothness",
    "morph_edge_asym_180",
]

MORPH_PLOT06_LABELS = {
    "morph_asym_180": "Asymmetry 180",
    "morph_mirror_asym_lr": "Mirror Asymmetry",
    "morph_concentration_r80r20": "Flux Ratio / Concentration",
    "morph_ellipticity_e2": "Ellipticity",
    "morph_peakedness_kurtosis": "Kurtosis",
    "morph_roundness": "Roundness",
    "morph_texture_laplacian": "High-Frequency Artifacts",
    "morph_gini": "Gini Coefficient",
    "morph_m20": "M20",
    "morph_smoothness": "Smoothness",
    "morph_edge_asym_180": "Edge Asymmetry",
}


def load_bad_quality_source_ids(
    quality_csv: Path,
    id_col: str,
    flag_col: str,
    bits_mask: int,
) -> np.ndarray:
    df = pd.read_csv(quality_csv, usecols=[id_col, flag_col], low_memory=False)
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df[flag_col] = pd.to_numeric(df[flag_col], errors="coerce")
    df = df.dropna(subset=[id_col, flag_col]).copy()
    if len(df) == 0:
        return np.array([], dtype=np.int64)
    sid = df[id_col].to_numpy(dtype=np.int64)
    qf = df[flag_col].to_numpy(dtype=np.int64)
    bad = (qf & int(bits_mask)) != 0
    return np.unique(sid[bad]).astype(np.int64)


def load_psf_source_ids(labels_csv: Path, label_keep: int) -> np.ndarray:
    df = pd.read_csv(labels_csv, usecols=["source_id", "label"], low_memory=False)
    df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["source_id", "label"]).copy()
    df["source_id"] = df["source_id"].astype(np.int64)
    df["label"] = df["label"].astype(np.int64)
    keep = df.loc[df["label"] == int(label_keep), "source_id"].drop_duplicates().to_numpy(dtype=np.int64)
    if keep.size == 0:
        raise RuntimeError(f"No source_id found in {labels_csv} for label={label_keep}")
    return keep


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def fmt_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "n/a"
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def infer_field_tag(path_str: str) -> str:
    try:
        p = Path(path_str)
        return p.parent.name if p.parent.name else "unknown"
    except Exception:
        return "unknown"


def load_candidates(
    dataset_root: Path,
    g_mag_min: Optional[float],
    g_mag_max: Optional[float],
    metadata_name: str = "metadata.csv",
) -> pd.DataFrame:
    usecols = ["source_id", "phot_g_mean_mag", "fits_path", "npz_file", "index_in_file", *FEATURE_COLS_8D]
    dfs: List[pd.DataFrame] = []
    metas = sorted(dataset_root.glob(f"*/{metadata_name}"))
    if not metas:
        raise RuntimeError(f"No {metadata_name} found under {dataset_root}")

    for i, meta_path in enumerate(metas, start=1):
        df = pd.read_csv(meta_path, usecols=lambda c: c in set(usecols), low_memory=False)
        for c in ["source_id", "phot_g_mean_mag", "index_in_file", *FEATURE_COLS_8D]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dataset_tag"] = meta_path.parent.name
        if "fits_path" in df.columns:
            df["field_tag"] = df["fits_path"].astype(str).map(infer_field_tag)
        else:
            df["field_tag"] = meta_path.parent.name

        m = (
            df["source_id"].notna()
            & df["npz_file"].notna()
            & df["index_in_file"].notna()
        )
        if g_mag_min is not None:
            m &= df["phot_g_mean_mag"].notna() & (df["phot_g_mean_mag"] >= float(g_mag_min))
        if g_mag_max is not None:
            m &= df["phot_g_mean_mag"].notna() & (df["phot_g_mean_mag"] <= float(g_mag_max))
        keep_cols = ["source_id", "phot_g_mean_mag", "field_tag", "dataset_tag", "npz_file", "index_in_file", *FEATURE_COLS_8D]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df.loc[m, keep_cols].copy()
        df["source_id"] = df["source_id"].astype(np.int64)
        dfs.append(df)
        print(f"[META] {i:02d}/{len(metas):02d} {meta_path.parent.name}: kept {len(df):,} rows")

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    print(f"[META] unique sources after de-dup: {len(out):,}")
    return out


def normalize_stamps(stamps: np.ndarray, mode: str, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    s = stamps.astype(np.float32, copy=False)
    if mode == "integral_pos":
        denom = np.sum(np.clip(s, 0.0, np.inf), axis=(1, 2), dtype=np.float32)
    elif mode == "integral":
        denom = np.sum(s, axis=(1, 2), dtype=np.float32)
    elif mode == "l2":
        denom = np.sqrt(np.sum(s * s, axis=(1, 2), dtype=np.float32))
    else:
        raise ValueError("normalize mode must be one of: integral_pos, integral, l2")
    ok = np.isfinite(denom) & (denom > float(eps))
    s = s.copy()
    if np.any(ok):
        s[ok] /= denom[ok, None, None]
    return s, ok


def _estimate_bkg_sigma(img: np.ndarray, r2: np.ndarray, r_in: float, r_out: float) -> Tuple[float, float]:
    m = (r2 >= (r_in * r_in)) & (r2 <= (r_out * r_out))
    vals = img[m]
    if vals.size == 0:
        vals = img.ravel()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    return med, sigma


def _ellipticity_from_moments(mxx: float, myy: float, mxy: float) -> Tuple[float, float, float, float]:
    M = np.array([[mxx, mxy], [mxy, myy]], dtype=float)
    w, v = np.linalg.eigh(M)
    lam2, lam1 = float(w[0]), float(w[1])
    if (lam1 <= 0) or (lam2 < 0):
        return np.nan, np.nan, np.nan, np.nan
    a = np.sqrt(lam1)
    b = np.sqrt(max(lam2, 0.0))
    e1 = (a - b) / (a + b) if (a + b) > 0 else np.nan
    e2 = (lam1 - lam2) / (lam1 + lam2) if (lam1 + lam2) > 0 else np.nan
    q = b / a if a > 0 else np.nan
    vx, vy = float(v[0, 1]), float(v[1, 1])
    theta = float(np.arctan2(vy, vx))
    return float(e1), float(e2), float(q), theta


def _gini_nonneg(values: np.ndarray) -> float:
    f = np.asarray(values, dtype=float)
    f = f[np.isfinite(f)]
    n = f.size
    if n < 2:
        return np.nan
    fs = np.sort(f)
    mf = float(np.mean(fs))
    if mf <= 0:
        return np.nan
    i = np.arange(1, n + 1, dtype=float)
    G = np.sum((2.0 * i - n - 1.0) * fs) / (mf * n * (n - 1.0))
    return float(G)


def _smooth_image(Ipos: np.ndarray) -> np.ndarray:
    if gaussian_filter is not None:
        return gaussian_filter(Ipos, sigma=1.0, mode="nearest")
    # Fallback smoothing if scipy is unavailable.
    return (
        4.0 * Ipos
        + np.roll(Ipos, 1, axis=0)
        + np.roll(Ipos, -1, axis=0)
        + np.roll(Ipos, 1, axis=1)
        + np.roll(Ipos, -1, axis=1)
    ) / 8.0


def compute_morphology_features(stamps: np.ndarray, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    n, ny, nx = stamps.shape
    x0 = 0.5 * (nx - 1.0)
    y0 = 0.5 * (ny - 1.0)
    X, Y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    i = X.astype(int)
    j = Y.astype(int)
    x0i = int(np.round(x0))
    y0i = int(np.round(y0))
    i2_rot = 2 * x0i - i
    j2_rot = 2 * y0i - j
    inside_rot = (i2_rot >= 0) & (i2_rot < nx) & (j2_rot >= 0) & (j2_rot < ny)
    i2_mir = 2 * x0i - i
    inside_mir = (i2_mir >= 0) & (i2_mir < nx)

    dx0 = X - x0
    dy0 = Y - y0
    r2 = dx0 * dx0 + dy0 * dy0
    r = np.sqrt(r2)
    Rmax = max(2.0, 0.5 * min(nx, ny) - 0.5)
    R = 0.9 * Rmax
    R1_edge = 0.6 * R
    R2_edge = R
    r1 = 0.35 * R
    r2c = R
    M = r <= R
    Mann = (r >= R1_edge) & (r <= R2_edge)

    out: Dict[str, np.ndarray] = {k: np.full(n, np.nan, dtype=np.float32) for k in MORPH_FEATURE_COLS}
    radii = np.linspace(0.0, R, 48)

    for k in range(n):
        img = stamps[k].astype(np.float32, copy=False)
        bkg, _ = _estimate_bkg_sigma(img, r2=r2, r_in=0.7 * Rmax, r_out=Rmax)
        Ib = img - float(bkg)
        Ipos = np.maximum(Ib, 0.0)

        F = float(np.sum(Ipos[M]))
        if not np.isfinite(F) or (F <= eps):
            continue

        # Centroid + offset from Gaia (assumed projected at cutout center).
        xc = float(np.sum(X[M] * Ipos[M]) / F)
        yc = float(np.sum(Y[M] * Ipos[M]) / F)
        dx = xc - x0
        dy = yc - y0
        dr = float(np.hypot(dx, dy))
        out["morph_centroid_dx"][k] = dx
        out["morph_centroid_dy"][k] = dy
        out["morph_centroid_dr"][k] = dr

        # 180 rotational asymmetry.
        I180 = np.zeros_like(Ipos)
        I180[inside_rot] = Ipos[j2_rot[inside_rot], i2_rot[inside_rot]]
        den = float(np.sum(np.abs(Ipos[M])))
        if den > eps:
            out["morph_asym_180"][k] = float(np.sum(np.abs(Ipos[M] - I180[M])) / den)

        # Mirror asymmetry (left-right).
        Imir = np.zeros_like(Ipos)
        Imir[inside_mir] = Ipos[j[inside_mir], i2_mir[inside_mir]]
        den = float(np.sum(np.abs(Ipos[M])))
        if den > eps:
            out["morph_mirror_asym_lr"][k] = float(np.sum(np.abs(Ipos[M] - Imir[M])) / den)

        # Concentration ratio and r80/r20.
        F1 = float(np.sum(Ib[r <= r1]))
        F2 = float(np.sum(Ib[r <= r2c]))
        if np.abs(F2) > eps:
            out["morph_concentration_ratio_r1r2"][k] = float(F1 / F2)
        curve = np.array([float(np.sum(Ib[r <= rr])) for rr in radii], dtype=float)
        Ftot = curve[-1]
        if np.isfinite(Ftot) and (Ftot > eps):
            fn = curve / Ftot
            r20 = float(np.interp(0.2, fn, radii))
            r80 = float(np.interp(0.8, fn, radii))
            if (r20 > eps) and (r80 > eps):
                out["morph_concentration_r80r20"][k] = float(5.0 * np.log10(r80 / r20))

        # Second moments.
        dxm = X[M] - xc
        dym = Y[M] - yc
        wm = Ipos[M]
        mxx = float(np.sum((dxm * dxm) * wm) / F)
        myy = float(np.sum((dym * dym) * wm) / F)
        mxy = float(np.sum((dxm * dym) * wm) / F)
        out["morph_mxx"][k] = mxx
        out["morph_myy"][k] = myy
        out["morph_mxy"][k] = mxy

        e1, e2, q, theta = _ellipticity_from_moments(mxx, myy, mxy)
        out["morph_ellipticity_e1"][k] = e1
        out["morph_ellipticity_e2"][k] = e2
        out["morph_axis_ratio_q"][k] = q
        out["morph_theta_rad"][k] = theta
        if np.isfinite(q):
            out["morph_roundness"][k] = float(1.0 - q)

        # Peakedness / peak-to-total.
        r2c_arr = (X[M] - xc) ** 2 + (Y[M] - yc) ** 2
        r2_mean = float(np.sum(r2c_arr * wm) / F)
        r4_mean = float(np.sum((r2c_arr ** 2) * wm) / F)
        if r2_mean > eps:
            out["morph_peakedness_kurtosis"][k] = float(r4_mean / (r2_mean ** 2))
        total_ib = float(np.sum(Ib[M]))
        if np.abs(total_ib) > eps:
            out["morph_peak_to_total"][k] = float(np.max(Ib[M]) / total_ib)

        # Texture.
        lap = (
            -4.0 * Ib
            + np.roll(Ib, 1, axis=0) + np.roll(Ib, -1, axis=0)
            + np.roll(Ib, 1, axis=1) + np.roll(Ib, -1, axis=1)
        )
        gx = 0.5 * (np.roll(Ib, -1, axis=1) - np.roll(Ib, 1, axis=1))
        gy = 0.5 * (np.roll(Ib, -1, axis=0) - np.roll(Ib, 1, axis=0))
        grad2 = gx * gx + gy * gy
        den_tex = float(np.sum((Ipos[M]) ** 2))
        if den_tex > eps:
            out["morph_texture_laplacian"][k] = float(np.sum((lap[M]) ** 2) / den_tex)
            out["morph_texture_gradient"][k] = float(np.sum(grad2[M]) / den_tex)

        # Gini and M20.
        out["morph_gini"][k] = _gini_nonneg(Ipos[M].ravel())
        f = Ipos[M].ravel().astype(float)
        if f.size > 0:
            Xm = X[M].ravel().astype(float)
            Ym = Y[M].ravel().astype(float)
            r2m = (Xm - xc) ** 2 + (Ym - yc) ** 2
            Mtot = float(np.sum(f * r2m))
            if Mtot > eps:
                idx = np.argsort(f)[::-1]
                fs = f[idx]
                r2s = r2m[idx]
                cumsum = np.cumsum(fs)
                cut = int(np.searchsorted(cumsum, 0.2 * np.sum(fs), side="left"))
                cut = max(cut, 0)
                M20num = float(np.sum(fs[: cut + 1] * r2s[: cut + 1]))
                if M20num > eps:
                    out["morph_m20"][k] = float(np.log10(M20num / Mtot))

        # Smoothness.
        Is = _smooth_image(Ipos)
        den_s = float(np.sum(Ipos[M]))
        if den_s > eps:
            out["morph_smoothness"][k] = float(np.sum(np.abs(Ipos[M] - Is[M])) / den_s)

        # Edge asymmetry.
        den_e = float(np.sum(np.abs(Ipos[Mann])))
        if den_e > eps:
            out["morph_edge_asym_180"][k] = float(np.sum(np.abs(Ipos[Mann] - I180[Mann])) / den_e)

    return out


def build_pixel_matrix(
    rows: pd.DataFrame,
    dataset_root: Path,
    normalize: str,
    eps: float,
    crop_center: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    out_rows: List[pd.DataFrame] = []
    chunks: List[np.ndarray] = []

    t0 = time.time()
    groups_field = list(rows.groupby("dataset_tag", sort=True))
    n_field = len(groups_field)
    for fi, (dataset_tag, dfg) in enumerate(groups_field, start=1):
        field_dir = dataset_root / dataset_tag
        npz_groups = list(dfg.groupby("npz_file", sort=False))
        print(f"[NPZ] field {fi}/{n_field} {dataset_tag}: {len(npz_groups)} files")
        for npz_name, sub in npz_groups:
            npz_path = field_dir / str(npz_name)
            if not npz_path.exists():
                continue
            with np.load(npz_path) as dnpz:
                if "X" not in dnpz:
                    continue
                Xall = dnpz["X"]

            idx = sub["index_in_file"].to_numpy(dtype=int)
            ok_idx = (idx >= 0) & (idx < Xall.shape[0])
            if not np.any(ok_idx):
                continue

            sub2 = sub.iloc[np.where(ok_idx)[0]].copy()
            idx2 = idx[ok_idx]

            stamps = Xall[idx2].astype(np.float32, copy=False)
            if stamps.ndim != 3:
                continue

            # Crop to central N×N before normalization (background diagnostic).
            if crop_center > 0:
                _, H, W = stamps.shape
                if crop_center < H and crop_center < W:
                    h0 = (H - crop_center) // 2
                    w0 = (W - crop_center) // 2
                    stamps = stamps[:, h0:h0 + crop_center, w0:w0 + crop_center]

            stamps_n, ok_norm = normalize_stamps(stamps, mode=normalize, eps=eps)
            if not np.any(ok_norm):
                continue

            sub3 = sub2.iloc[np.where(ok_norm)[0]].copy()
            stamps_keep = stamps_n[ok_norm]
            pix = stamps_keep.reshape(np.sum(ok_norm), -1).astype(np.float32)
            morph = compute_morphology_features(stamps_keep, eps=eps)
            for cname, cvals in morph.items():
                sub3[cname] = cvals

            out_rows.append(sub3)
            chunks.append(pix)

    if not chunks:
        raise RuntimeError("No valid normalized stamps produced. Try changing --normalize or --eps.")

    out_df = pd.concat(out_rows, ignore_index=True)
    X = np.vstack(chunks).astype(np.float32)
    dt = time.time() - t0
    print(f"[NPZ] built pixel matrix: rows={len(out_df):,} shape={X.shape} elapsed={fmt_duration(dt)}")
    return out_df, X


def load_wds_labels(wds_dir: Path, threshold_arcsec: float, threshold_mode: str = "le") -> pd.DataFrame:
    usecols = ["source_id", "wds_dist_arcsec", "wds_lstsep", "wds_comp", "wds_disc", "field_tag"]
    dfs: List[pd.DataFrame] = []
    files = sorted(wds_dir.glob("*_wds_best.csv"))
    if not files:
        raise RuntimeError(f"No *_wds_best.csv found under {wds_dir}")
    for p in files:
        df = pd.read_csv(p, usecols=lambda c: c in set(usecols), low_memory=False)
        if "source_id" not in df.columns:
            continue
        for c in ["source_id", "wds_dist_arcsec", "wds_lstsep"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["source_id"]).copy()
        df["source_id"] = df["source_id"].astype(np.int64)
        if threshold_mode == "lt":
            sel = df["wds_dist_arcsec"] < threshold_arcsec
        else:
            sel = df["wds_dist_arcsec"] <= threshold_arcsec
        df["is_double"] = df["wds_dist_arcsec"].notna() & sel
        dfs.append(df)
    w = pd.concat(dfs, ignore_index=True)
    w = w.sort_values(["source_id", "wds_dist_arcsec"], ascending=[True, True], na_position="last")
    w = w.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return w[["source_id", "is_double", "wds_dist_arcsec", "wds_lstsep", "wds_comp", "wds_disc"]]


def scale_features(
    X: np.ndarray,
    mode: str,
    sigma_floor_quantile: float = 20.0,
    small_sigma_policy: str = "zero",
) -> np.ndarray:
    mode = mode.lower().strip()
    if mode == "standard":
        return StandardScaler().fit_transform(X)
    if mode == "standard_sigma_floor":
        q = float(sigma_floor_quantile)
        if not (0.0 <= q < 100.0):
            raise ValueError("sigma_floor_quantile must be in [0, 100)")
        policy = str(small_sigma_policy).lower().strip()
        if policy not in {"zero", "floor"}:
            raise ValueError("small_sigma_policy must be one of: zero, floor")

        mu = np.mean(X, axis=0, dtype=np.float64)
        sigma = np.std(X, axis=0, dtype=np.float64)
        pos_sigma = sigma[np.isfinite(sigma) & (sigma > 0.0)]
        sigma_floor = float(np.percentile(pos_sigma, q)) if pos_sigma.size else 0.0
        small = (~np.isfinite(sigma)) | (sigma <= sigma_floor)

        Xc = X.astype(np.float32, copy=False) - mu.astype(np.float32, copy=False)
        out = np.zeros_like(Xc, dtype=np.float32)
        if policy == "zero":
            good = ~small
            if np.any(good):
                out[:, good] = Xc[:, good] / sigma[good].astype(np.float32, copy=False)
        else:
            denom = sigma.copy()
            denom[~np.isfinite(denom)] = 0.0
            denom = np.maximum(denom, sigma_floor)
            denom[denom <= 0.0] = 1.0
            out = Xc / denom.astype(np.float32, copy=False)

        print(
            "[SCALE] standard_sigma_floor:"
            f" q={q:.1f} floor={sigma_floor:.3e}"
            f" small_dims={int(np.sum(small))}/{int(X.shape[1])}"
            f" policy={policy}"
        )
        return out
    if mode == "minmax":
        return MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(X)
    if mode == "none":
        return X
    raise ValueError("scale mode must be one of: standard, standard_sigma_floor, minmax, none")


def build_embedding(X: np.ndarray, seed: int, n_neighbors: int, min_dist: float, metric: str = "euclidean") -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        init="spectral",
    )
    return reducer.fit_transform(X)


def parse_seed_list(text: str) -> List[int]:
    t = (text or "").strip()
    if not t:
        return []
    out = []
    for part in t.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    seen = set()
    uniq = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def knn_overlap_fraction(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    k: int = 50,
    sample_n: int = 15000,
    seed: int = 123,
) -> float:
    n = emb_a.shape[0]
    if n < 3:
        return float("nan")
    k_eff = int(max(2, min(k, n - 1)))
    if sample_n is not None and n > int(sample_n):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=int(sample_n), replace=False))
        a = emb_a[idx]
        b = emb_b[idx]
    else:
        a = emb_a
        b = emb_b
    nn_a = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(a)
    nn_b = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(b)
    ia = nn_a.kneighbors(a, return_distance=False)[:, 1:]
    ib = nn_b.kneighbors(b, return_distance=False)[:, 1:]
    frac = []
    for xa, xb in zip(ia, ib):
        sa = set(xa.tolist())
        sb = set(xb.tolist())
        frac.append(len(sa.intersection(sb)) / float(k_eff))
    return float(np.mean(frac))


def compute_local_double_fraction(emb: np.ndarray, is_double: np.ndarray, k: int = 50) -> np.ndarray:
    k_eff = int(max(5, min(k, len(emb) - 1)))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(emb)
    idx = nn.kneighbors(emb, return_distance=False)
    neigh = idx[:, 1:]
    return is_double[neigh].mean(axis=1)


def cluster_doubles_dbscan(emb: np.ndarray, is_double: np.ndarray) -> np.ndarray:
    labels = np.full(len(emb), -1, dtype=int)
    d_idx = np.where(is_double)[0]
    if len(d_idx) < 8:
        return labels
    xd = emb[d_idx]
    nn = NearestNeighbors(n_neighbors=min(6, len(xd)), metric="euclidean").fit(xd)
    dists, _ = nn.kneighbors(xd)
    eps = max(float(np.quantile(dists[:, -1], 0.60)), 1e-3)
    min_samples = int(np.clip(round(np.sqrt(len(xd)) / 2), 4, 20))
    cl = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(xd)
    labels[d_idx] = cl
    return labels


def plot_main_map(df: pd.DataFrame, out_png: Path, threshold_text: str) -> None:
    plt.figure(figsize=(10, 8))
    plt.hexbin(df["x"], df["y"], gridsize=160, bins="log", cmap="Greys", mincnt=1)
    d = df[df["is_double"]]
    plt.scatter(d["x"], d["y"], s=16, c="#e63946", alpha=0.95, edgecolors="white", linewidths=0.2, label=f"WDS double ({threshold_text})")
    plt.xlabel("Embedding X")
    plt.ylabel("Embedding Y")
    plt.title("UMAP projection of normalized stamp pixels")
    cbar = plt.colorbar()
    cbar.set_label("Background density (log counts)")
    plt.legend(loc="best", frameon=True)
    savefig(out_png)


def plot_local_enrichment(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(df["x"], df["y"], c=df["local_double_frac_k50"], s=5, cmap="viridis", alpha=0.8, linewidths=0)
    d = df[df["is_double"]]
    plt.scatter(d["x"], d["y"], s=14, facecolors="none", edgecolors="#f94144", linewidths=0.4, label="doubles")
    plt.xlabel("Embedding X")
    plt.ylabel("Embedding Y")
    plt.title("Local enrichment of doubles (k=50)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Fraction doubles in local neighborhood")
    plt.legend(loc="best", frameon=True)
    savefig(out_png)


def plot_gmag(df: pd.DataFrame, out_png: Path) -> None:
    vals = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(vals)
    if np.any(m):
        lo, hi = np.nanpercentile(vals[m], [1, 99])
        vals = np.clip(vals, lo, hi)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(df["x"], df["y"], c=vals, s=5, cmap="magma_r", alpha=0.8, linewidths=0)
    d = df[df["is_double"]]
    plt.scatter(d["x"], d["y"], s=14, facecolors="none", edgecolors="white", linewidths=0.35)
    plt.xlabel("Embedding X")
    plt.ylabel("Embedding Y")
    plt.title("UMAP colored by Gaia G magnitude")
    cbar = plt.colorbar(sc)
    cbar.set_label("phot_g_mean_mag (clipped 1-99 pct)")
    savefig(out_png)


def plot_feature_color_panels(
    df: pd.DataFrame,
    feature_cols: List[str],
    out_png: Path,
    title_map: Optional[Dict[str, str]] = None,
    suptitle: Optional[str] = None,
) -> None:
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        return
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, c in zip(axes, cols):
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if np.any(mask):
            lo, hi = np.nanpercentile(vals[mask], [1, 99])
            vals = np.clip(vals, lo, hi)
        sc = ax.scatter(df["x"], df["y"], c=vals, s=5, cmap="viridis", alpha=0.75, linewidths=0)
        d = df[df["is_double"]]
        ax.scatter(d["x"], d["y"], s=12, facecolors="none", edgecolors="#f94144", linewidths=0.35)
        ax.set_title(title_map.get(c, c) if title_map else c, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    savefig(out_png)


def plot_field_tag_colors(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 8))
    tags = sorted(df["field_tag"].dropna().astype(str).unique().tolist())
    cmap = plt.get_cmap("tab20", max(len(tags), 1))
    for i, tag in enumerate(tags):
        sub = df[df["field_tag"] == tag]
        plt.scatter(sub["x"], sub["y"], s=5, color=cmap(i), alpha=0.6, linewidths=0, label=tag)
    d = df[df["is_double"]]
    plt.scatter(d["x"], d["y"], s=14, facecolors="none", edgecolors="black", linewidths=0.4, label="doubles")
    plt.xlabel("Embedding X")
    plt.ylabel("Embedding Y")
    plt.title("Pixel-UMAP color-coded by field")
    if len(tags) <= 20:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="UMAP on normalized stamp pixels with WDS overlays.")
    ap.add_argument("--method", default="umap", choices=["umap"])
    ap.add_argument("--dataset_root", default="/data/yn316/Codes/output/dataset_npz")
    ap.add_argument("--wds_dir", default="/data/yn316/Codes/output/crossmatch/wds/wds_xmatch")
    ap.add_argument("--metadata_name", default="metadata.csv", help="Per-field metadata filename to read (e.g. metadata_16d.csv)")
    ap.add_argument("--out_tag", default="", help="Optional subfolder tag to isolate outputs (e.g. 16d_phase3)")
    ap.add_argument("--max_points", type=int, default=120000)
    ap.add_argument("--force_include_doubles", action="store_true", help="When sampling, always include all WDS doubles then fill with non-doubles.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n_neighbors", type=int, default=30)
    ap.add_argument("--min_dist", type=float, default=0.08)
    ap.add_argument("--scaling", default="standard", choices=["standard", "standard_sigma_floor", "minmax", "none"])
    ap.add_argument(
        "--sigma_floor_quantile",
        type=float,
        default=20.0,
        help="For scaling=standard_sigma_floor: percentile of per-dim sigma used as floor (e.g. 10 or 20).",
    )
    ap.add_argument(
        "--small_sigma_policy",
        default="zero",
        choices=["zero", "floor"],
        help="For scaling=standard_sigma_floor: zero small-sigma dims, or divide by floored sigma.",
    )
    ap.add_argument("--normalize", default="integral_pos", choices=["integral_pos", "integral", "l2"])
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--g_mag_min", type=float, default=None)
    ap.add_argument("--g_mag_max", type=float, default=None)
    ap.add_argument("--double_threshold_arcsec", type=float, default=1.0)
    ap.add_argument("--double_condition", default="le", choices=["le", "lt"])
    ap.add_argument("--shuffle_before_umap", action="store_true", help="Shuffle row order before fitting UMAP (sanity check for order effects)")
    ap.add_argument("--shuffle_seed", type=int, default=123, help="Seed used for optional pre-UMAP shuffle")
    ap.add_argument("--stability_seeds", default="", help="Comma-separated extra UMAP seeds for stability checks, e.g. '7,42,123'")
    ap.add_argument("--stability_k", type=int, default=50, help="k for kNN-overlap stability metric")
    ap.add_argument("--stability_sample_n", type=int, default=15000, help="Max sample size for stability kNN-overlap")
    ap.add_argument("--quality_flags_csv", default="", help="Optional quality flag CSV for source-level exclusion.")
    ap.add_argument("--quality_id_col", default="source_id")
    ap.add_argument("--quality_flag_col", default="quality_flag")
    ap.add_argument("--quality_bits_mask", type=int, default=0, help="Exclude rows with (quality_flag & mask) != 0.")
    ap.add_argument("--psf_labels_csv", default="", help="Optional labels CSV with source_id,label to keep only one class.")
    ap.add_argument("--psf_label_keep", type=int, default=-1, help="0=psf_like, 1=non_psf_like")
    ap.add_argument("--crop_center", type=int, default=0, help="If >0, crop each stamp to central N×N pixels before normalization (e.g. 10 for 10x10). 0=no crop.")
    ap.add_argument("--umap_metric", default="euclidean", help="UMAP distance metric (e.g. euclidean, cosine). Default: euclidean.")
    args = ap.parse_args()
    args.stability_seeds_parsed = parse_seed_list(args.stability_seeds)

    base = Path(__file__).resolve().parents[2]
    dataset_root = Path(args.dataset_root)
    wds_dir = Path(args.wds_dir)

    out_tag = str(args.out_tag).strip()
    if out_tag:
        out_plot_base = base / "plots" / "qa" / "embeddings" / "double_stars_pixels" / out_tag
        out_tab_base = base / "output" / "experiments" / "embeddings" / "double_stars_pixels" / out_tag
    else:
        out_plot_base = base / "plots" / "qa" / "embeddings" / "double_stars_pixels"
        out_tab_base = base / "output" / "experiments" / "embeddings" / "double_stars_pixels"

    out_plot = ensure_dir(out_plot_base / f"umap_{args.scaling}")
    out_tab = ensure_dir(out_tab_base / f"umap_{args.scaling}")

    print("\n=== PIXEL UMAP RUN ===")
    print("dataset_root:", dataset_root)
    print("wds_dir     :", wds_dir)
    print("max_points  :", args.max_points)
    print("scaling     :", args.scaling)
    print("normalize   :", args.normalize)
    print("metadata    :", args.metadata_name)
    print("out_tag     :", out_tag if out_tag else "(default)")
    print("g_mag_min   :", args.g_mag_min)
    print("g_mag_max   :", args.g_mag_max)

    cand = load_candidates(
        dataset_root,
        g_mag_min=args.g_mag_min,
        g_mag_max=args.g_mag_max,
        metadata_name=str(args.metadata_name),
    )
    if str(args.psf_labels_csv).strip():
        if int(args.psf_label_keep) not in (0, 1):
            raise RuntimeError("--psf_label_keep must be 0 or 1 when --psf_labels_csv is provided.")
        keep_ids = load_psf_source_ids(Path(str(args.psf_labels_csv)), int(args.psf_label_keep))
        cand = cand.loc[cand["source_id"].isin(keep_ids)].copy()
        print(f"PSF label filter applied: keep_label={int(args.psf_label_keep)} kept={len(cand):,} rows after merge.")
    if str(args.quality_flags_csv).strip():
        if int(args.quality_bits_mask) <= 0:
            raise RuntimeError("--quality_bits_mask must be >0 when --quality_flags_csv is provided.")
        bad_ids = load_bad_quality_source_ids(
            quality_csv=Path(str(args.quality_flags_csv)),
            id_col=str(args.quality_id_col),
            flag_col=str(args.quality_flag_col),
            bits_mask=int(args.quality_bits_mask),
        )
        if bad_ids.size > 0:
            cand = cand.loc[~cand["source_id"].isin(bad_ids)].copy()
        print(f"Quality filter applied: mask={int(args.quality_bits_mask)} removed={int(bad_ids.size)} sources (from quality CSV).")
    w = load_wds_labels(wds_dir, threshold_arcsec=float(args.double_threshold_arcsec), threshold_mode=str(args.double_condition))
    if len(cand) > int(args.max_points):
        if bool(args.force_include_doubles):
            c2 = cand.merge(w[["source_id", "is_double"]], on="source_id", how="left")
            c2["is_double"] = c2["is_double"].fillna(False).astype(bool)
            idx_double = np.where(c2["is_double"].to_numpy(dtype=bool))[0]
            idx_non = np.where(~c2["is_double"].to_numpy(dtype=bool))[0]
            n_keep_non = max(0, min(len(idx_non), int(args.max_points) - len(idx_double)))
            rng = np.random.default_rng(int(args.seed))
            keep_non = rng.choice(idx_non, size=n_keep_non, replace=False) if n_keep_non < len(idx_non) else idx_non
            keep = np.unique(np.concatenate([idx_double, keep_non]))
            cand = c2.iloc[keep].drop(columns=["is_double"]).reset_index(drop=True)
            print(f"[SAMPLE] forced doubles: sampled to {len(cand):,} rows (doubles_kept={len(idx_double):,})")
        else:
            cand = cand.sample(n=int(args.max_points), random_state=args.seed).reset_index(drop=True)
            print(f"[SAMPLE] sampled to {len(cand):,} rows")

    rows, X = build_pixel_matrix(cand, dataset_root=dataset_root, normalize=args.normalize, eps=args.eps, crop_center=int(args.crop_center))

    dfe = rows.merge(w, on="source_id", how="left")
    dfe["is_double"] = dfe["is_double"].fillna(False).astype(bool)

    Xs = scale_features(
        X,
        mode=args.scaling,
        sigma_floor_quantile=float(args.sigma_floor_quantile),
        small_sigma_policy=str(args.small_sigma_policy),
    ).astype(np.float32, copy=False)
    if bool(args.shuffle_before_umap):
        rng_shuffle = np.random.default_rng(int(args.shuffle_seed))
        perm = rng_shuffle.permutation(len(dfe))
        dfe = dfe.iloc[perm].reset_index(drop=True)
        Xs = Xs[perm]
        print(f"[SHUFFLE] permuted rows before UMAP using shuffle_seed={args.shuffle_seed}")
    print(f"Fitting UMAP on {len(dfe):,} rows and {Xs.shape[1]} pixel dims...")
    t0 = time.time()
    emb = build_embedding(Xs, seed=int(args.seed), n_neighbors=int(args.n_neighbors), min_dist=float(args.min_dist), metric=str(args.umap_metric))
    print(f"UMAP done in {fmt_duration(time.time() - t0)}")

    dfe["x"] = emb[:, 0]
    dfe["y"] = emb[:, 1]
    dfe["local_double_frac_k50"] = compute_local_double_fraction(emb, dfe["is_double"].to_numpy(dtype=bool), k=50)
    dfe["double_cluster"] = cluster_doubles_dbscan(emb, dfe["is_double"].to_numpy(dtype=bool))

    stability_rows = []
    for s in args.stability_seeds_parsed:
        if int(s) == int(args.seed):
            continue
        print(f"[STABILITY] fitting additional seed={s} ...")
        emb_s = build_embedding(
            Xs,
            seed=int(s),
            n_neighbors=int(args.n_neighbors),
            min_dist=float(args.min_dist),
        )
        ov = knn_overlap_fraction(
            emb,
            emb_s,
            k=int(args.stability_k),
            sample_n=int(args.stability_sample_n),
            seed=int(args.seed),
        )
        stability_rows.append({"seed_ref": int(args.seed), "seed_alt": int(s), "k": int(args.stability_k), "sample_n": int(args.stability_sample_n), "mean_knn_overlap": ov})
        print(f"[STABILITY] seed {s}: mean_kNN_overlap@{args.stability_k}={ov:.4f}")

    threshold_text = f"{'<' if args.double_condition == 'lt' else '<='}{args.double_threshold_arcsec:.3g}\""

    plot_main_map(dfe, out_plot / "01_umap_main_map_density_plus_doubles.png", threshold_text=threshold_text)
    plot_local_enrichment(dfe, out_plot / "02_umap_local_enrichment_k50.png")
    plot_gmag(dfe, out_plot / "03_umap_colored_by_gmag.png")
    plot_feature_color_panels(
        dfe,
        FEATURE_COLS_8D,
        out_plot / "04_umap_features_colored_panels.png",
        suptitle="Pixel-UMAP color-coded by Gaia 8D features",
    )
    plot_field_tag_colors(dfe, out_plot / "05_umap_field_tag_colored.png")
    plot_feature_color_panels(
        dfe,
        MORPH_PLOT06_COLS,
        out_plot / "06_umap_morphology_panels.png",
        title_map=MORPH_PLOT06_LABELS,
        suptitle="Pixel-UMAP color-coded by selected morphology metrics",
    )

    keep = [
        "source_id",
        "field_tag",
        "dataset_tag",
        "phot_g_mean_mag",
        "npz_file",
        "index_in_file",
        *FEATURE_COLS_8D,
        *MORPH_FEATURE_COLS,
        "is_double",
        "wds_dist_arcsec",
        "wds_lstsep",
        "wds_comp",
        "wds_disc",
        "local_double_frac_k50",
        "double_cluster",
        "x",
        "y",
    ]
    out_csv = out_tab / "embedding_umap.csv"
    dfe[keep].to_csv(out_csv, index=False)

    summary = {
        "n_points_embedded": int(len(dfe)),
        "n_double_selected": int(dfe["is_double"].sum()),
        "frac_double": float(dfe["is_double"].mean()) if len(dfe) else 0.0,
        "pixel_dim": int(X.shape[1]),
        "crop_center": int(args.crop_center),
        "normalize": args.normalize,
        "scaling": args.scaling,
        "sigma_floor_quantile": float(args.sigma_floor_quantile),
        "small_sigma_policy": str(args.small_sigma_policy),
        "g_mag_min": args.g_mag_min,
        "g_mag_max": args.g_mag_max,
        "umap_n_neighbors": int(args.n_neighbors),
        "umap_min_dist": float(args.min_dist),
        "umap_metric": str(args.umap_metric),
        "double_threshold_arcsec": float(args.double_threshold_arcsec),
        "double_condition": str(args.double_condition),
        "morphology_feature_cols": MORPH_FEATURE_COLS,
        "shuffle_before_umap": bool(args.shuffle_before_umap),
        "shuffle_seed": int(args.shuffle_seed),
    }
    if stability_rows:
        stab_csv = out_tab / "umap_seed_stability.csv"
        pd.DataFrame(stability_rows).to_csv(stab_csv, index=False)
        summary["seed_stability_csv"] = str(stab_csv)
        summary["seed_stability_n_runs"] = int(len(stability_rows))
        summary["seed_stability_k"] = int(args.stability_k)
        summary["seed_stability_sample_n"] = int(args.stability_sample_n)
        print("[STABILITY] wrote:", stab_csv)
    with open(out_tab / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DONE ===")
    print("Embedding CSV:", out_csv)
    print("Summary      :", out_tab / "summary_stats.json")
    print("Plots        :", out_plot)


if __name__ == "__main__":
    main()
