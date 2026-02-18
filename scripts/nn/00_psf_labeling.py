#!/usr/bin/env python3
"""
Create weak labels for PSF-like vs non-PSF-like sources from stamp morphology.

Output CSV columns:
- source_id
- label (0=psf_like, 1=non_psf_like)
- confidence (0..1)
- split_code
- field_tag
- score_psf_like
- score_non_psf_like

The script is conservative by default:
- high-confidence tails are labeled
- middle band is dropped as ambiguous
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Cfg:
    base_dir: Path = Path(__file__).resolve().parents[2]
    dataset_root: Path = None
    out_dir: Path = None
    low_quantile: float = 0.30
    high_quantile: float = 0.70
    min_confidence: float = 0.55

    def __post_init__(self) -> None:
        if self.dataset_root is None:
            self.dataset_root = self.base_dir / "output" / "dataset_npz"
        if self.out_dir is None:
            self.out_dir = self.base_dir / "output" / "ml_runs" / "nn_psf_labels"


def _safe_percentile(a: np.ndarray, q: float) -> float:
    v = a[np.isfinite(a)]
    if v.size == 0:
        return 0.0
    return float(np.percentile(v, q))


def extract_morphology(stamp: np.ndarray) -> Dict[str, float]:
    img = np.array(stamp, dtype=np.float64)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    bg = _safe_percentile(img, 10.0)
    ip = img - bg
    ip[ip < 0] = 0.0

    h, w = ip.shape
    yy, xx = np.indices((h, w))
    flux_sum = float(np.sum(ip))
    peak = float(np.max(ip)) if ip.size else 0.0

    if flux_sum <= 0:
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        ell = 0.0
        a = 0.0
        b = 0.0
    else:
        cx = float(np.sum(ip * xx) / flux_sum)
        cy = float(np.sum(ip * yy) / flux_sum)
        dx = xx - cx
        dy = yy - cy
        qxx = float(np.sum(ip * dx * dx) / flux_sum)
        qyy = float(np.sum(ip * dy * dy) / flux_sum)
        qxy = float(np.sum(ip * dx * dy) / flux_sum)
        tr = qxx + qyy
        det = qxx * qyy - qxy * qxy
        disc = max(tr * tr - 4.0 * det, 0.0)
        lam1 = max((tr + np.sqrt(disc)) * 0.5, 0.0)
        lam2 = max((tr - np.sqrt(disc)) * 0.5, 0.0)
        a = float(np.sqrt(lam1))
        b = float(np.sqrt(lam2))
        ell = float((a - b) / (a + b + 1e-9))

    flat = ip.ravel()
    if flat.size >= 2:
        idx2 = np.argpartition(flat, -2)[-2:]
        vals2 = flat[idx2]
        ord2 = np.argsort(vals2)[::-1]
        i1 = int(idx2[ord2[0]])
        i2 = int(idx2[ord2[1]])
        y1, x1 = divmod(i1, w)
        y2, x2 = divmod(i2, w)
        peak_ratio = float(flat[i2] / (flat[i1] + 1e-9))
        peak_sep = float(np.hypot(x2 - x1, y2 - y1))
    else:
        peak_ratio = 0.0
        peak_sep = 0.0

    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    f_r2 = float(np.sum(ip[rr <= 2.0]))
    f_r6 = float(np.sum(ip[rr <= 6.0]))
    concentration = float(f_r2 / (f_r6 + 1e-9))

    border = np.zeros_like(ip, dtype=bool)
    border[:2, :] = True
    border[-2:, :] = True
    border[:, :2] = True
    border[:, -2:] = True
    edge_flux_frac = float(np.sum(ip[border]) / (flux_sum + 1e-9))

    rot180 = np.flipud(np.fliplr(ip))
    asym = float(np.sum(np.abs(ip - rot180)) / (np.sum(np.abs(ip)) + 1e-9))

    return {
        "m_flux_sum": flux_sum,
        "m_flux_peak": peak,
        "m_peak_ratio_2over1": peak_ratio,
        "m_peak_sep_pix": peak_sep,
        "m_ellipticity": ell,
        "m_size_a_pix": a,
        "m_size_b_pix": b,
        "m_concentration_r2_r6": concentration,
        "m_edge_flux_frac": edge_flux_frac,
        "m_asymmetry_180": asym,
    }


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad + 1e-9
    return (x - med) / scale


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    # PSF-like: concentrated, symmetric, single-peaked, round, centered.
    z_conc = robust_z(df["m_concentration_r2_r6"].to_numpy(dtype=float))
    z_asym = robust_z(df["m_asymmetry_180"].to_numpy(dtype=float))
    z_ell = robust_z(df["m_ellipticity"].to_numpy(dtype=float))
    z_sep = robust_z(df["m_peak_sep_pix"].to_numpy(dtype=float))
    z_edge = robust_z(df["m_edge_flux_frac"].to_numpy(dtype=float))
    z_ratio = robust_z(df["m_peak_ratio_2over1"].to_numpy(dtype=float))

    score_psf = (
        +1.3 * z_conc
        -1.0 * z_asym
        -0.8 * z_ell
        -1.1 * z_sep
        -0.6 * z_edge
        -0.5 * z_ratio
    )
    score_non = -score_psf

    out = df.copy()
    out["score_psf_like"] = score_psf
    out["score_non_psf_like"] = score_non
    return out


def load_fields(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / "metadata.csv").exists()])


def build_feature_table(cfg: Cfg) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    fields = load_fields(cfg.dataset_root)
    if not fields:
        raise RuntimeError(f"No field directories with metadata.csv under {cfg.dataset_root}")

    for fdir in fields:
        meta = pd.read_csv(fdir / "metadata.csv", low_memory=False)
        keep = [c for c in ["source_id", "split_code", "npz_file", "index_in_file", "ruwe", "ipd_frac_multi_peak", "phot_bp_rp_excess_factor"] if c in meta.columns]
        meta = meta[keep].copy()
        meta = meta.dropna(subset=["source_id", "npz_file", "index_in_file"])
        meta["source_id"] = pd.to_numeric(meta["source_id"], errors="coerce").astype("Int64")
        meta["index_in_file"] = pd.to_numeric(meta["index_in_file"], errors="coerce").astype("Int64")
        meta = meta.dropna(subset=["source_id", "index_in_file"]).copy()

        cache: Dict[str, np.ndarray] = {}
        for r in meta.itertuples(index=False):
            source_id = int(r.source_id)
            npz_file = str(r.npz_file)
            idx = int(r.index_in_file)
            npz_path = fdir / npz_file
            if not npz_path.exists():
                continue
            if npz_file not in cache:
                arr = np.load(npz_path)
                if "X" not in arr.files:
                    continue
                cache[npz_file] = np.array(arr["X"], dtype=np.float32)
            x = cache[npz_file]
            if idx < 0 or idx >= x.shape[0]:
                continue
            feats = extract_morphology(x[idx])
            out = {
                "source_id": source_id,
                "field_tag": fdir.name,
                "split_code": int(getattr(r, "split_code", 0)) if hasattr(r, "split_code") and pd.notna(getattr(r, "split_code", 0)) else 0,
                "ruwe": float(getattr(r, "ruwe", np.nan)) if hasattr(r, "ruwe") else np.nan,
                "ipd_frac_multi_peak": float(getattr(r, "ipd_frac_multi_peak", np.nan)) if hasattr(r, "ipd_frac_multi_peak") else np.nan,
                "phot_bp_rp_excess_factor": float(getattr(r, "phot_bp_rp_excess_factor", np.nan)) if hasattr(r, "phot_bp_rp_excess_factor") else np.nan,
            }
            out.update(feats)
            rows.append(out)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable stamps found while building PSF labels")
    return df


def assign_labels(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    d = compute_scores(df)

    lo = float(np.nanquantile(d["score_psf_like"], cfg.low_quantile))
    hi = float(np.nanquantile(d["score_psf_like"], cfg.high_quantile))

    # Gaia anomaly proxy: tends to correlate with non-point-like/contaminated cases.
    gaia_non_psf = (
        (pd.to_numeric(d.get("ruwe", np.nan), errors="coerce") > 1.4)
        | (pd.to_numeric(d.get("ipd_frac_multi_peak", np.nan), errors="coerce") > 0.1)
    )

    is_psf = d["score_psf_like"] >= hi
    is_non = (d["score_psf_like"] <= lo) | gaia_non_psf.fillna(False)

    out = d.copy()
    out["label"] = np.nan
    out.loc[is_psf, "label"] = 0
    out.loc[is_non, "label"] = 1

    # resolve collisions conservatively: if both, keep non-PSF only if Gaia anomaly triggered strongly
    both = is_psf & is_non
    out.loc[both, "label"] = np.where(gaia_non_psf[both], 1, np.nan)

    mid = 0.5 * (hi + lo)
    dist = np.abs(out["score_psf_like"] - mid)
    dist = dist / (np.abs(hi - lo) + 1e-9)
    out["confidence"] = np.clip(dist, 0.0, 1.0)

    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    out = out[out["confidence"] >= cfg.min_confidence].copy()

    # One source_id can appear multiple times; keep highest-confidence label per source.
    out = out.sort_values(["source_id", "confidence"], ascending=[True, False]).drop_duplicates(subset=["source_id"], keep="first")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="", help="Path to output/dataset_npz")
    ap.add_argument("--out_dir", default="", help="Output directory")
    ap.add_argument("--low_quantile", type=float, default=0.30)
    ap.add_argument("--high_quantile", type=float, default=0.70)
    ap.add_argument("--min_confidence", type=float, default=0.55)
    args = ap.parse_args()

    cfg = Cfg(
        dataset_root=Path(args.dataset_root) if str(args.dataset_root).strip() else None,
        out_dir=Path(args.out_dir) if str(args.out_dir).strip() else None,
        low_quantile=float(args.low_quantile),
        high_quantile=float(args.high_quantile),
        min_confidence=float(args.min_confidence),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    feat = build_feature_table(cfg)
    lab = assign_labels(feat, cfg)

    out_csv = cfg.out_dir / "labels_psf_weak.csv"
    cols = [
        "source_id",
        "label",
        "confidence",
        "split_code",
        "field_tag",
        "score_psf_like",
        "score_non_psf_like",
        "m_concentration_r2_r6",
        "m_asymmetry_180",
        "m_ellipticity",
        "m_peak_sep_pix",
        "m_edge_flux_frac",
        "m_peak_ratio_2over1",
        "ruwe",
        "ipd_frac_multi_peak",
    ]
    cols = [c for c in cols if c in lab.columns]
    lab[cols].to_csv(out_csv, index=False)

    counts = lab["label"].value_counts().to_dict()
    print("Saved:", out_csv)
    print("Counts:", counts)
    print("Total labeled:", len(lab))


if __name__ == "__main__":
    main()
