#!/usr/bin/env python3
"""
Audit PSF weak-label pipeline with exact stage counts and score diagnostics.

This script re-runs the same logic as scripts/nn/00_psf_labeling.py but records
intermediate counts and score-component statistics for decision-making.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


WEIGHTS = {
    "m_concentration_r2_r6": +1.3,
    "m_asymmetry_180": -1.0,
    "m_ellipticity": -0.8,
    "m_peak_sep_pix": -1.1,
    "m_edge_flux_frac": -0.6,
    "m_peak_ratio_2over1": -0.5,
}


def _load_psf_module(base_dir: Path):
    mod_path = base_dir / "scripts" / "nn" / "00_psf_labeling.py"
    module_name = "psf_labeling_mod"
    spec = importlib.util.spec_from_file_location(module_name, str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def robust_z_with_params(x: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    x = np.asarray(x, dtype=float)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    scale = float(1.4826 * mad + 1e-9)
    z = (x - med) / scale
    return z, med, mad, scale


def qstats(v: np.ndarray) -> Dict[str, float]:
    x = np.asarray(v, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "q01": np.nan,
            "q10": np.nan,
            "q50": np.nan,
            "q90": np.nan,
            "q99": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q01": float(np.quantile(x, 0.01)),
        "q10": float(np.quantile(x, 0.10)),
        "q50": float(np.quantile(x, 0.50)),
        "q90": float(np.quantile(x, 0.90)),
        "q99": float(np.quantile(x, 0.99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--low_quantile", type=float, default=0.30)
    ap.add_argument("--high_quantile", type=float, default=0.70)
    ap.add_argument("--min_confidence", type=float, default=0.55)
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    mod = _load_psf_module(base_dir)

    cfg = mod.Cfg(
        dataset_root=Path(args.dataset_root) if str(args.dataset_root).strip() else None,
        out_dir=Path(args.out_dir) if str(args.out_dir).strip() else None,
        low_quantile=float(args.low_quantile),
        high_quantile=float(args.high_quantile),
        min_confidence=float(args.min_confidence),
    )

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base_dir / "report" / "model_decision" / "20260220_psf_labeling_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building morphology feature table...")
    df = mod.build_feature_table(cfg)

    print("[2/4] Computing score components and thresholds...")
    z_conc, med_conc, mad_conc, scl_conc = robust_z_with_params(df["m_concentration_r2_r6"].to_numpy(dtype=float))
    z_asym, med_asym, mad_asym, scl_asym = robust_z_with_params(df["m_asymmetry_180"].to_numpy(dtype=float))
    z_ell, med_ell, mad_ell, scl_ell = robust_z_with_params(df["m_ellipticity"].to_numpy(dtype=float))
    z_sep, med_sep, mad_sep, scl_sep = robust_z_with_params(df["m_peak_sep_pix"].to_numpy(dtype=float))
    z_edge, med_edge, mad_edge, scl_edge = robust_z_with_params(df["m_edge_flux_frac"].to_numpy(dtype=float))
    z_ratio, med_ratio, mad_ratio, scl_ratio = robust_z_with_params(df["m_peak_ratio_2over1"].to_numpy(dtype=float))

    score_psf = (
        +1.3 * z_conc
        -1.0 * z_asym
        -0.8 * z_ell
        -1.1 * z_sep
        -0.6 * z_edge
        -0.5 * z_ratio
    )
    score_non = -score_psf

    d = df.copy()
    d["score_psf_like"] = score_psf
    d["score_non_psf_like"] = score_non

    lo = float(np.nanquantile(d["score_psf_like"], float(args.low_quantile)))
    hi = float(np.nanquantile(d["score_psf_like"], float(args.high_quantile)))

    ruwe = pd.to_numeric(d.get("ruwe", np.nan), errors="coerce")
    ipd = pd.to_numeric(d.get("ipd_frac_multi_peak", np.nan), errors="coerce")
    gaia_non_psf = (ruwe > 1.4) | (ipd > 0.1)

    is_psf = d["score_psf_like"] >= hi
    is_non_score = d["score_psf_like"] <= lo
    is_non = is_non_score | gaia_non_psf.fillna(False)
    both = is_psf & is_non

    out = d.copy()
    out["label"] = np.nan
    out.loc[is_psf, "label"] = 0
    out.loc[is_non, "label"] = 1
    out.loc[both, "label"] = np.where(gaia_non_psf[both], 1, np.nan)

    mid = 0.5 * (hi + lo)
    dist = np.abs(out["score_psf_like"] - mid)
    dist = dist / (np.abs(hi - lo) + 1e-9)
    out["confidence"] = np.clip(dist, 0.0, 1.0)

    print("[3/4] Building exact stage counts...")
    n_total_rows = int(len(out))
    n_unique_source_total = int(pd.to_numeric(out["source_id"], errors="coerce").dropna().astype(np.int64).nunique())

    s0_tail_psf = int(np.sum(is_psf.to_numpy(dtype=bool)))
    s0_tail_non_score = int(np.sum(is_non_score.to_numpy(dtype=bool)))
    s0_gaia_non = int(np.sum(gaia_non_psf.fillna(False).to_numpy(dtype=bool)))
    s0_is_non_union = int(np.sum(is_non.to_numpy(dtype=bool)))
    s0_both = int(np.sum(both.to_numpy(dtype=bool)))

    labeled_pre_conf = out.dropna(subset=["label"]).copy()
    n_labeled_pre_conf = int(len(labeled_pre_conf))
    n_labeled_pre_conf_psf = int((labeled_pre_conf["label"] == 0).sum())
    n_labeled_pre_conf_non = int((labeled_pre_conf["label"] == 1).sum())

    labeled_post_conf = labeled_pre_conf[labeled_pre_conf["confidence"] >= float(args.min_confidence)].copy()
    n_labeled_post_conf = int(len(labeled_post_conf))
    n_labeled_post_conf_psf = int((labeled_post_conf["label"] == 0).sum())
    n_labeled_post_conf_non = int((labeled_post_conf["label"] == 1).sum())

    final = labeled_post_conf.sort_values(["source_id", "confidence"], ascending=[True, False]).drop_duplicates(subset=["source_id"], keep="first")
    final["label"] = final["label"].astype(int)
    n_final = int(len(final))
    n_final_psf = int((final["label"] == 0).sum())
    n_final_non = int((final["label"] == 1).sum())

    n_removed_by_conf = int(n_labeled_pre_conf - n_labeled_post_conf)
    n_removed_by_dedup = int(n_labeled_post_conf - n_final)

    by_split_final = pd.DataFrame()
    if "split_code" in final.columns:
        by_split_final = (
            final.groupby(["split_code", "label"]).size().reset_index(name="n").sort_values(["split_code", "label"])
        )
        by_split_final.to_csv(out_dir / "audit_final_counts_by_split.csv", index=False)

    print("[4/4] Writing score/morphology diagnostics...")
    comp_rows: List[Dict[str, float]] = []
    components = [
        ("m_concentration_r2_r6", z_conc, med_conc, mad_conc, scl_conc, +1.3),
        ("m_asymmetry_180", z_asym, med_asym, mad_asym, scl_asym, -1.0),
        ("m_ellipticity", z_ell, med_ell, mad_ell, scl_ell, -0.8),
        ("m_peak_sep_pix", z_sep, med_sep, mad_sep, scl_sep, -1.1),
        ("m_edge_flux_frac", z_edge, med_edge, mad_edge, scl_edge, -0.6),
        ("m_peak_ratio_2over1", z_ratio, med_ratio, mad_ratio, scl_ratio, -0.5),
    ]
    for name, z, med, mad, scale, w in components:
        raw_stats = qstats(d[name].to_numpy(dtype=float))
        z_stats = qstats(z)
        contrib = w * z
        c_stats = qstats(contrib)
        comp_rows.append(
            {
                "metric": name,
                "weight": float(w),
                "raw_median": float(med),
                "raw_mad": float(mad),
                "robust_scale": float(scale),
                "raw_q10": raw_stats["q10"],
                "raw_q50": raw_stats["q50"],
                "raw_q90": raw_stats["q90"],
                "z_q10": z_stats["q10"],
                "z_q50": z_stats["q50"],
                "z_q90": z_stats["q90"],
                "contrib_q10": c_stats["q10"],
                "contrib_q50": c_stats["q50"],
                "contrib_q90": c_stats["q90"],
            }
        )
    pd.DataFrame(comp_rows).to_csv(out_dir / "audit_score_components.csv", index=False)

    # Morphology summaries by final label
    morph_cols = [c for c in [
        "m_concentration_r2_r6",
        "m_asymmetry_180",
        "m_ellipticity",
        "m_peak_sep_pix",
        "m_edge_flux_frac",
        "m_peak_ratio_2over1",
        "score_psf_like",
        "score_non_psf_like",
        "confidence",
        "ruwe",
        "ipd_frac_multi_peak",
    ] if c in final.columns]

    ms_rows: List[Dict[str, float]] = []
    for lab in [0, 1]:
        sub = final[final["label"] == lab]
        for c in morph_cols:
            stats = qstats(pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float))
            ms_rows.append({"label": int(lab), "metric": c, **stats})
    pd.DataFrame(ms_rows).to_csv(out_dir / "audit_metric_stats_by_label.csv", index=False)

    summary = {
        "params": {
            "low_quantile": float(args.low_quantile),
            "high_quantile": float(args.high_quantile),
            "min_confidence": float(args.min_confidence),
            "gaia_non_psf_rule": "(ruwe > 1.4) OR (ipd_frac_multi_peak > 0.1)",
            "score_weights": WEIGHTS,
        },
        "thresholds": {
            "score_psf_like_low": lo,
            "score_psf_like_high": hi,
            "score_psf_like_q10": float(np.nanquantile(d["score_psf_like"], 0.10)),
            "score_psf_like_q50": float(np.nanquantile(d["score_psf_like"], 0.50)),
            "score_psf_like_q90": float(np.nanquantile(d["score_psf_like"], 0.90)),
        },
        "counts": {
            "n_rows_feature_table": n_total_rows,
            "n_unique_source_feature_table": n_unique_source_total,
            "tail_psf_count": s0_tail_psf,
            "tail_nonpsf_by_score_count": s0_tail_non_score,
            "gaia_nonpsf_override_count": s0_gaia_non,
            "nonpsf_union_count": s0_is_non_union,
            "psf_and_non_overlap_count": s0_both,
            "labeled_pre_conf_total": n_labeled_pre_conf,
            "labeled_pre_conf_psf": n_labeled_pre_conf_psf,
            "labeled_pre_conf_nonpsf": n_labeled_pre_conf_non,
            "labeled_post_conf_total": n_labeled_post_conf,
            "labeled_post_conf_psf": n_labeled_post_conf_psf,
            "labeled_post_conf_nonpsf": n_labeled_post_conf_non,
            "removed_by_confidence_filter": n_removed_by_conf,
            "final_after_dedup_total": n_final,
            "final_after_dedup_psf": n_final_psf,
            "final_after_dedup_nonpsf": n_final_non,
            "removed_by_dedup": n_removed_by_dedup,
        },
        "artifacts": {
            "final_counts_by_split_csv": str(out_dir / "audit_final_counts_by_split.csv"),
            "score_components_csv": str(out_dir / "audit_score_components.csv"),
            "metric_stats_by_label_csv": str(out_dir / "audit_metric_stats_by_label.csv"),
        },
    }

    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save final labels for direct compare with existing labels file.
    keep_cols = [c for c in [
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
    ] if c in final.columns]
    final[keep_cols].to_csv(out_dir / "labels_psf_weak_recomputed.csv", index=False)

    print("DONE")
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
