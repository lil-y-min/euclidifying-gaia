#!/usr/bin/env python3
"""
Build quality bitmask flags from WSDB/forensics-style catalogs.

Bit definitions:
  1   non_converged_fit
  2   pathological_noisearea
  4   low_morph_confidence
  8   bad_cross_match
  16  edge_source
  32  flux_mismatch
  64  wsdb_corner_or_chip
  128 wsdb_artifact_or_mask
  256 seam_gradient_discontinuity
  512 zero_pixel_contamination
  1024 manual_border_confirmed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BIT_INFO = {
    1: "non_converged_fit",
    2: "pathological_noisearea",
    4: "low_morph_confidence",
    8: "bad_cross_match",
    16: "edge_source",
    32: "flux_mismatch",
    64: "wsdb_corner_or_chip",
    128: "wsdb_artifact_or_mask",
    256: "seam_gradient_discontinuity",
    512: "zero_pixel_contamination",
    1024: "manual_border_confirmed",
}


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _first_present(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Input merged catalog/forensics csv.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--field_col", default="field_tag")
    ap.add_argument("--source_id_col", default="source_id")

    ap.add_argument("--niter_col", default="v_niter_model")
    ap.add_argument("--niter_global_threshold", type=float, default=np.nan, help="Optional hard threshold. If omitted, only field p-quantile applies.")
    ap.add_argument("--niter_field_quantile", type=float, default=0.95)

    ap.add_argument("--noisearea_col", default="v_noisearea_model")
    ap.add_argument("--noisearea_min", type=float, default=5.0)
    ap.add_argument("--noisearea_max", type=float, default=500.0)

    ap.add_argument("--spread_col", default="v_spread_model")
    ap.add_argument("--spreaderr_col", default="v_spreaderr_model")
    ap.add_argument("--spread_snr_min", type=float, default=2.0)

    ap.add_argument("--dist_match_col", default="dist_match")
    ap.add_argument("--dist_match_max", type=float, default=0.5)

    ap.add_argument("--x_col", default="v_x_image")
    ap.add_argument("--y_col", default="v_y_image")
    ap.add_argument("--stamp_width", type=float, default=20.0)
    ap.add_argument("--stamp_height", type=float, default=20.0)
    ap.add_argument("--edge_distance_min_px", type=float, default=3.0)

    ap.add_argument("--flux_auto_col", default="v_flux_auto")
    ap.add_argument("--flux_model_col", default="v_flux_model")
    ap.add_argument("--flux_mismatch_max_frac", type=float, default=0.3)
    ap.add_argument("--flux_eps", type=float, default=1e-12)

    ap.add_argument(
        "--corner_cols",
        default="wsdb_corner_flag,corner_flag,v_corner_flag",
        help="Comma-separated candidate boolean/int columns for corner/chip issues.",
    )
    ap.add_argument(
        "--artifact_cols",
        default="wsdb_artifact_flag,artifact_flag,mask_flag,v_mask_flag",
        help="Comma-separated candidate boolean/int columns for artifact/mask issues.",
    )
    # Optional pixel-stamp artifact bits (requires npz_relpath + index_in_file + dataset_root)
    ap.add_argument("--dataset_root", default="", help="Root path for dataset npz files (e.g. .../output/dataset_npz).")
    ap.add_argument("--npz_relpath_col", default="npz_relpath")
    ap.add_argument("--index_in_file_col", default="index_in_file")
    ap.add_argument("--stamp_key", default="X")
    ap.add_argument("--enable_bit256_gradient", action="store_true", help="Enable gradient seam discontinuity bit.")
    ap.add_argument("--enable_bit512_zero", action="store_true", help="Enable zero-pixel contamination bit.")
    ap.add_argument("--zero_pixel_threshold", type=float, default=1e-6)
    ap.add_argument("--zero_frac_min", type=float, default=0.05)
    ap.add_argument(
        "--bit512_mode",
        choices=["frac", "line", "line_or_frac"],
        default="frac",
        help="Bit512 strategy: global zero fraction, row/col zero-line, or union.",
    )
    ap.add_argument(
        "--zero_line_frac_min",
        type=float,
        default=0.5,
        help="For bit512 line mode: flag if any row/col has >= this zero fraction.",
    )
    ap.add_argument(
        "--zero_line_include_diag",
        action="store_true",
        help="For bit512 line mode: also test diagonals.",
    )
    ap.add_argument("--gradient_eps", type=float, default=1e-6)
    ap.add_argument("--gradient_ratio_min", type=float, default=np.nan, help="If finite, fixed threshold for bit256 on max_gradient/median_abs_intensity.")
    ap.add_argument("--gradient_quantile", type=float, default=0.99, help="Quantile threshold for bit256 when --gradient_ratio_min is not set.")
    ap.add_argument("--gradient_per_field", action="store_true", help="Compute bit256 gradient quantile threshold per field.")
    ap.add_argument("--manual_bit1024_csv", default="", help="Optional CSV containing manually confirmed artifact IDs to set bit1024.")
    ap.add_argument("--manual_bit1024_id_col", default="test_index", help="ID column name used to match manual_bit1024_csv to input rows.")
    args = ap.parse_args()

    in_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv, low_memory=False)
    n = len(df)
    qflag = np.zeros(n, dtype=np.uint16)
    bit_masks: Dict[int, np.ndarray] = {}
    missing_checks: List[str] = []
    active_checks: List[str] = []

    field_col = str(args.field_col)
    if field_col not in df.columns:
        field_col = ""

    # Bit 1: non-converged fit
    niter_col = str(args.niter_col)
    if niter_col in df.columns:
        x = _safe_num(df[niter_col]).to_numpy(dtype=float)
        m = np.zeros(n, dtype=bool)
        if field_col:
            tmp = pd.DataFrame({"field": df[field_col].astype(str), "x": x})
            q_by_field = tmp.groupby("field")["x"].quantile(float(args.niter_field_quantile))
            q_map = tmp["field"].map(q_by_field).to_numpy(dtype=float)
            m |= np.isfinite(x) & np.isfinite(q_map) & (x >= q_map)
        if np.isfinite(float(args.niter_global_threshold)):
            m |= np.isfinite(x) & (x >= float(args.niter_global_threshold))
        bit_masks[1] = m
        qflag[m] |= 1
        active_checks.append("bit1_non_converged_fit")
    else:
        missing_checks.append("bit1_non_converged_fit")

    # Bit 2: pathological noise area
    na_col = str(args.noisearea_col)
    if na_col in df.columns:
        x = _safe_num(df[na_col]).to_numpy(dtype=float)
        m = np.isfinite(x) & ((x < float(args.noisearea_min)) | (x > float(args.noisearea_max)))
        bit_masks[2] = m
        qflag[m] |= 2
        active_checks.append("bit2_pathological_noisearea")
    else:
        missing_checks.append("bit2_pathological_noisearea")

    # Bit 4: low morphology confidence
    sp_col = str(args.spread_col)
    se_col = str(args.spreaderr_col)
    if (sp_col in df.columns) and (se_col in df.columns):
        sp = _safe_num(df[sp_col]).to_numpy(dtype=float)
        se = _safe_num(df[se_col]).to_numpy(dtype=float)
        good = np.isfinite(sp) & np.isfinite(se) & (np.abs(se) > 0)
        snr = np.full(n, np.nan, dtype=float)
        snr[good] = np.abs(sp[good] / se[good])
        m = np.isfinite(snr) & (snr < float(args.spread_snr_min))
        bit_masks[4] = m
        qflag[m] |= 4
        active_checks.append("bit4_low_morph_confidence")
    else:
        missing_checks.append("bit4_low_morph_confidence")

    # Bit 8: bad cross-match
    dm_col = str(args.dist_match_col)
    if dm_col in df.columns:
        x = _safe_num(df[dm_col]).to_numpy(dtype=float)
        m = np.isfinite(x) & (x > float(args.dist_match_max))
        bit_masks[8] = m
        qflag[m] |= 8
        active_checks.append("bit8_bad_cross_match")
    else:
        missing_checks.append("bit8_bad_cross_match")

    # Bit 16: edge source
    x_col = str(args.x_col)
    y_col = str(args.y_col)
    if (x_col in df.columns) and (y_col in df.columns):
        x = _safe_num(df[x_col]).to_numpy(dtype=float)
        y = _safe_num(df[y_col]).to_numpy(dtype=float)
        ex = np.minimum(x, float(args.stamp_width) - x)
        ey = np.minimum(y, float(args.stamp_height) - y)
        d = np.minimum(ex, ey)
        m = np.isfinite(d) & (d < float(args.edge_distance_min_px))
        bit_masks[16] = m
        qflag[m] |= 16
        active_checks.append("bit16_edge_source")
    else:
        missing_checks.append("bit16_edge_source")

    # Bit 32: flux mismatch
    fa_col = str(args.flux_auto_col)
    fm_col = str(args.flux_model_col)
    if (fa_col in df.columns) and (fm_col in df.columns):
        fa = _safe_num(df[fa_col]).to_numpy(dtype=float)
        fm = _safe_num(df[fm_col]).to_numpy(dtype=float)
        den = np.maximum(np.abs(fa), float(args.flux_eps))
        frac = np.abs(fa - fm) / den
        m = np.isfinite(frac) & (frac > float(args.flux_mismatch_max_frac))
        bit_masks[32] = m
        qflag[m] |= 32
        active_checks.append("bit32_flux_mismatch")
    else:
        missing_checks.append("bit32_flux_mismatch")

    # Bit 64: corner/chip flags
    corner_candidates = [x.strip() for x in str(args.corner_cols).split(",") if x.strip()]
    corner_col = _first_present(df, corner_candidates)
    if corner_col:
        c = _safe_num(df[corner_col]).to_numpy(dtype=float)
        m = np.isfinite(c) & (c > 0)
        bit_masks[64] = m
        qflag[m] |= 64
        active_checks.append(f"bit64_wsdb_corner_or_chip({corner_col})")
    else:
        missing_checks.append("bit64_wsdb_corner_or_chip")

    # Bit 128: artifact/mask flags
    artifact_candidates = [x.strip() for x in str(args.artifact_cols).split(",") if x.strip()]
    art_col = _first_present(df, artifact_candidates)
    if art_col:
        a = _safe_num(df[art_col]).to_numpy(dtype=float)
        m = np.isfinite(a) & (a > 0)
        bit_masks[128] = m
        qflag[m] |= 128
        active_checks.append(f"bit128_wsdb_artifact_or_mask({art_col})")
    else:
        missing_checks.append("bit128_wsdb_artifact_or_mask")

    # Bits 256/512: pixel-level artifact bits from stamps
    need_pix_bits = bool(args.enable_bit256_gradient or args.enable_bit512_zero)
    if need_pix_bits:
        dataset_root = Path(str(args.dataset_root).strip()) if str(args.dataset_root).strip() else None
        rel_col = str(args.npz_relpath_col)
        idx_col = str(args.index_in_file_col)
        have_cols = (rel_col in df.columns) and (idx_col in df.columns)
        if (dataset_root is None) or (not have_cols):
            if args.enable_bit256_gradient:
                missing_checks.append("bit256_seam_gradient_discontinuity")
            if args.enable_bit512_zero:
                missing_checks.append("bit512_zero_pixel_contamination")
        else:
            rel = df[rel_col].astype(str)
            idx = pd.to_numeric(df[idx_col], errors="coerce")
            valid = rel.notna() & rel.ne("") & np.isfinite(idx)
            idx_i = np.full(n, -1, dtype=np.int64)
            idx_i[valid.to_numpy()] = idx[valid].astype(np.int64).to_numpy()
            bit512 = np.zeros(n, dtype=bool)
            grad_score = np.full(n, np.nan, dtype=float)
            bad_files = 0
            used_rows = 0

            # Group by npz file to avoid repeated loads.
            grp = pd.DataFrame({"i": np.arange(n), "rel": rel, "idx": idx_i})
            grp = grp[(grp["idx"] >= 0) & grp["rel"].notna()].copy()
            for npz_rel, sub in grp.groupby("rel", sort=False):
                npz_path = dataset_root / str(npz_rel)
                if not npz_path.exists():
                    bad_files += 1
                    continue
                rows_i = sub["i"].to_numpy(dtype=np.int64)
                take_idx = sub["idx"].to_numpy(dtype=np.int64)
                try:
                    with np.load(npz_path, allow_pickle=False) as z:
                        if str(args.stamp_key) not in z:
                            bad_files += 1
                            continue
                        X = z[str(args.stamp_key)]
                        ok = (take_idx >= 0) & (take_idx < X.shape[0])
                        if not np.any(ok):
                            continue
                        rows_i = rows_i[ok]
                        take_idx = take_idx[ok]
                        stamps = X[take_idx].astype(np.float32, copy=False)  # (m,H,W)
                except Exception:
                    bad_files += 1
                    continue

                used_rows += int(stamps.shape[0])
                if args.enable_bit512_zero:
                    zmask = stamps <= float(args.zero_pixel_threshold)
                    # Fraction mode
                    zf = np.mean(zmask, axis=(1, 2))
                    m_frac = zf > float(args.zero_frac_min)
                    # Line mode: contiguous row/col zero-line signatures.
                    m_line = (
                        (np.max(np.mean(zmask, axis=2), axis=1) >= float(args.zero_line_frac_min))
                        | (np.max(np.mean(zmask, axis=1), axis=1) >= float(args.zero_line_frac_min))
                    )
                    if bool(args.zero_line_include_diag):
                        n = zmask.shape[1]
                        for k in range(-(n - 1), n):
                            d1 = np.diagonal(zmask, offset=k, axis1=1, axis2=2)
                            d2 = np.diagonal(np.flip(zmask, axis=2), offset=k, axis1=1, axis2=2)
                            # d1/d2 shape: (N, L)
                            if d1.shape[1] >= 2:
                                m_line |= (np.mean(d1, axis=1) >= float(args.zero_line_frac_min))
                            if d2.shape[1] >= 2:
                                m_line |= (np.mean(d2, axis=1) >= float(args.zero_line_frac_min))

                    mode = str(args.bit512_mode)
                    if mode == "frac":
                        bit512[rows_i] = m_frac
                    elif mode == "line":
                        bit512[rows_i] = m_line
                    else:
                        bit512[rows_i] = m_frac | m_line

                if args.enable_bit256_gradient:
                    gx = np.abs(np.diff(stamps, axis=2))
                    gy = np.abs(np.diff(stamps, axis=1))
                    gmax = np.maximum(np.max(gx, axis=(1, 2)), np.max(gy, axis=(1, 2)))
                    med_abs = np.median(np.abs(stamps), axis=(1, 2))
                    score = gmax / np.maximum(med_abs, float(args.gradient_eps))
                    grad_score[rows_i] = score

            if args.enable_bit512_zero:
                bit_masks[512] = bit512
                qflag[bit512] |= 512
                active_checks.append("bit512_zero_pixel_contamination")

            if args.enable_bit256_gradient:
                finite = np.isfinite(grad_score)
                bit256 = np.zeros(n, dtype=bool)
                if np.any(finite):
                    if np.isfinite(float(args.gradient_ratio_min)):
                        thr = float(args.gradient_ratio_min)
                        bit256 = finite & (grad_score > thr)
                    else:
                        q = float(args.gradient_quantile)
                        q = min(max(q, 0.5), 0.999999)
                        if bool(args.gradient_per_field) and field_col:
                            tmp = pd.DataFrame({"field": df[field_col].astype(str), "s": grad_score})
                            q_by_field = tmp.groupby("field")["s"].quantile(q)
                            q_map = tmp["field"].map(q_by_field).to_numpy(dtype=float)
                            bit256 = finite & np.isfinite(q_map) & (grad_score > q_map)
                        else:
                            thr = float(np.nanquantile(grad_score[finite], q))
                            bit256 = finite & (grad_score > thr)
                bit_masks[256] = bit256
                qflag[bit256] |= 256
                active_checks.append("bit256_seam_gradient_discontinuity")

            if bad_files > 0:
                missing_checks.append(f"pixel_bits_missing_npz_files={bad_files}")
            if used_rows == 0:
                missing_checks.append("pixel_bits_no_rows_processed")

    # Bit 1024: manual-confirmed border artifacts
    manual_csv = str(args.manual_bit1024_csv).strip()
    manual_id_col = str(args.manual_bit1024_id_col).strip()
    if manual_csv:
        p = Path(manual_csv)
        if (not p.exists()) or (manual_id_col not in df.columns):
            missing_checks.append("bit1024_manual_border_confirmed")
        else:
            dm = pd.read_csv(p, low_memory=False)
            if manual_id_col not in dm.columns:
                missing_checks.append("bit1024_manual_border_confirmed")
            else:
                ids_in = pd.to_numeric(df[manual_id_col], errors="coerce")
                ids_manual = pd.to_numeric(dm[manual_id_col], errors="coerce")
                ids_manual = ids_manual[np.isfinite(ids_manual)].astype(np.int64).to_numpy()
                m = np.zeros(n, dtype=bool)
                ok = np.isfinite(ids_in.to_numpy(dtype=float))
                if np.any(ok) and ids_manual.size > 0:
                    m[ok] = np.isin(ids_in.to_numpy(dtype=float)[ok].astype(np.int64), ids_manual, assume_unique=False)
                bit_masks[1024] = m
                qflag[m] |= 1024
                active_checks.append("bit1024_manual_border_confirmed")

    df_out = df.copy()
    df_out["quality_flag"] = qflag.astype(np.uint16)
    df_out["quality_flag_any"] = (qflag > 0).astype(np.uint8)

    out_csv = out_dir / "quality_flagged_catalog.csv"
    df_out.to_csv(out_csv, index=False)

    # bit summary
    rows = []
    for b, name in BIT_INFO.items():
        m = bit_masks.get(b, np.zeros(n, dtype=bool))
        rows.append(
            {
                "bit": int(b),
                "name": name,
                "n_set": int(np.sum(m)),
                "frac_set": float(np.mean(m)) if n > 0 else float("nan"),
            }
        )
    bit_df = pd.DataFrame(rows).sort_values("bit")
    bit_df.to_csv(out_dir / "quality_flag_bit_counts.csv", index=False)

    # per-field summary
    if field_col:
        pf_rows = []
        for f, sub in df_out.groupby(field_col, dropna=False):
            qv = sub["quality_flag"].to_numpy(dtype=np.uint16)
            row = {
                "field_tag": f,
                "n": int(len(sub)),
                "frac_any": float(np.mean(qv > 0)),
            }
            for b in sorted(BIT_INFO.keys()):
                row[f"frac_bit_{b}"] = float(np.mean((qv & b) > 0))
            pf_rows.append(row)
        pd.DataFrame(pf_rows).to_csv(out_dir / "quality_flag_by_field.csv", index=False)

    summary = {
        "input_csv": str(in_csv),
        "output_csv": str(out_csv),
        "n_rows": int(n),
        "n_any_flag": int(np.sum(qflag > 0)),
        "frac_any_flag": float(np.mean(qflag > 0)) if n > 0 else float("nan"),
        "active_checks": active_checks,
        "missing_checks": missing_checks,
        "bit_definitions": {str(k): v for k, v in BIT_INFO.items()},
    }
    (out_dir / "quality_flag_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print("out_dir:", out_dir)
    print("output_csv:", out_csv)


if __name__ == "__main__":
    main()
