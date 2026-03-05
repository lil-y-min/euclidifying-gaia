#!/usr/bin/env python3
"""
Assess impact of quality flags on outliers and chi2 metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
}


def _q(v: np.ndarray, q: float) -> float:
    vv = np.asarray(v, dtype=float)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return float("nan")
    return float(np.quantile(vv, q))


def _metrics(df: pd.DataFrame, chi_col: str) -> Dict[str, float]:
    x = pd.to_numeric(df[chi_col], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return {
        "n_sources": int(x.size),
        "chi2_median": _q(x, 0.50),
        "chi2_p90": _q(x, 0.90),
        "chi2_p99": _q(x, 0.99),
    }


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged_csv", required=True, help="Output from 43_quality_flag_builder.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--chi2_col", default="chi2nu")
    ap.add_argument("--is_outlier_col", default="is_outlier")
    ap.add_argument("--field_col", default="field_tag")
    ap.add_argument("--fornax_name", default="ERO-Fornax")
    ap.add_argument("--severe_bits", default="1,2,16", help="Comma-separated bit values used for severe exclusion level.")
    args = ap.parse_args()

    in_csv = Path(args.flagged_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv, low_memory=False)
    if "quality_flag" not in df.columns:
        raise RuntimeError("Missing quality_flag column.")
    if args.chi2_col not in df.columns:
        raise RuntimeError(f"Missing chi2 column: {args.chi2_col}")
    if args.field_col not in df.columns:
        raise RuntimeError(f"Missing field column: {args.field_col}")

    qf = pd.to_numeric(df["quality_flag"], errors="coerce").fillna(0).astype(np.uint16)
    df["quality_flag"] = qf

    # outlier mask
    if args.is_outlier_col in df.columns:
        outlier = pd.to_numeric(df[args.is_outlier_col], errors="coerce").fillna(0).astype(int) > 0
    else:
        thr = _q(pd.to_numeric(df[args.chi2_col], errors="coerce").to_numpy(dtype=float), 0.99)
        outlier = pd.to_numeric(df[args.chi2_col], errors="coerce").to_numpy(dtype=float) >= thr
    df["is_outlier_eval"] = outlier

    severe_bits = [int(x.strip()) for x in str(args.severe_bits).split(",") if x.strip()]
    severe_mask = np.zeros(len(df), dtype=np.uint16)
    for b in severe_bits:
        severe_mask = severe_mask | np.uint16(b)

    # overlap with outliers
    out_df = df[df["is_outlier_eval"]].copy()
    rows = []
    n_out = max(1, len(out_df))
    for b, name in BIT_INFO.items():
        m = ((out_df["quality_flag"].to_numpy(dtype=np.uint16) & np.uint16(b)) > 0)
        rows.append(
            {
                "bit": int(b),
                "name": name,
                "n_outliers_with_bit": int(np.sum(m)),
                "frac_outliers_with_bit": float(np.sum(m) / n_out),
            }
        )
    overlap_df = pd.DataFrame(rows).sort_values("bit")
    overlap_df.to_csv(out_dir / "flag_overlap_with_outliers.csv", index=False)

    # impact levels
    base = df.copy()
    sev = df[(df["quality_flag"].to_numpy(dtype=np.uint16) & severe_mask) == 0].copy()
    allf = df[df["quality_flag"].to_numpy(dtype=np.uint16) == 0].copy()

    m0 = _metrics(base, args.chi2_col)
    m1 = _metrics(sev, args.chi2_col)
    m2 = _metrics(allf, args.chi2_col)

    impact = pd.DataFrame(
        [
            {"exclusion_level": "baseline_all", **m0},
            {"exclusion_level": "severe_only_excluded", **m1},
            {"exclusion_level": "all_flags_excluded", **m2},
        ]
    )
    b_p99 = float(m0["chi2_p99"])
    impact["p99_delta_vs_baseline_pct"] = 100.0 * (impact["chi2_p99"] / b_p99 - 1.0)
    impact["p99_improvement_vs_baseline_pct"] = 100.0 * (1.0 - impact["chi2_p99"] / b_p99)
    impact.to_csv(out_dir / "flag_impact_on_metrics.csv", index=False)

    # Fornax vs non-Fornax bit rates
    is_f = df[args.field_col].astype(str) == str(args.fornax_name)
    grp = [("fornax", is_f), ("non_fornax", ~is_f)]
    bar_rows = []
    for gname, gm in grp:
        q = df.loc[gm, "quality_flag"].to_numpy(dtype=np.uint16)
        n = max(1, q.size)
        for b in sorted(BIT_INFO.keys()):
            bar_rows.append(
                {
                    "group": gname,
                    "bit": int(b),
                    "name": BIT_INFO[b],
                    "frac_set": float(np.sum((q & np.uint16(b)) > 0) / n),
                }
            )
    bar_df = pd.DataFrame(bar_rows)
    bar_df.to_csv(out_dir / "fornax_vs_nonfornax_bit_rates.csv", index=False)

    # Plot Fornax distribution
    bits = sorted(BIT_INFO.keys())
    x = np.arange(len(bits))
    fvals = [float(bar_df[(bar_df["group"] == "fornax") & (bar_df["bit"] == b)]["frac_set"].iloc[0]) for b in bits]
    nvals = [float(bar_df[(bar_df["group"] == "non_fornax") & (bar_df["bit"] == b)]["frac_set"].iloc[0]) for b in bits]
    w = 0.38
    plt.figure(figsize=(10, 5.5))
    plt.bar(x - w / 2, fvals, width=w, label="Fornax")
    plt.bar(x + w / 2, nvals, width=w, label="Non-Fornax")
    plt.xticks(x, [str(b) for b in bits])
    plt.xlabel("Quality flag bit")
    plt.ylabel("Fraction set")
    plt.title("Fornax vs Non-Fornax quality flag rates")
    plt.legend()
    _savefig(out_dir / "fornax_flag_distribution.png")

    # Plot outlier overlap
    plt.figure(figsize=(10, 5.5))
    plt.bar(overlap_df["bit"].astype(str), overlap_df["frac_outliers_with_bit"])
    plt.xlabel("Quality flag bit")
    plt.ylabel("Fraction of outliers with bit set")
    plt.title("Outlier overlap by quality bit")
    _savefig(out_dir / "outlier_flag_overlap.png")

    summary = {
        "input_csv": str(in_csv),
        "n_total": int(len(df)),
        "n_outliers_eval": int(np.sum(df["is_outlier_eval"])),
        "severe_bits": severe_bits,
        "baseline": m0,
        "severe_only_excluded": m1,
        "all_flags_excluded": m2,
    }
    (out_dir / "flag_impact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Quality Flag Impact Summary")
    lines.append("")
    lines.append(f"- n_total: {int(len(df))}")
    lines.append(f"- n_outliers_eval: {int(np.sum(df['is_outlier_eval']))}")
    lines.append(f"- severe bits: {severe_bits}")
    lines.append("")
    lines.append("## p99 impact")
    for _, r in impact.iterrows():
        lines.append(
            f"- {r['exclusion_level']}: n={int(r['n_sources'])}, "
            f"p99={float(r['chi2_p99']):.6g}, "
            f"improvement_vs_baseline={float(r['p99_improvement_vs_baseline_pct']):+.2f}%"
        )
    (out_dir / "flag_impact_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()

