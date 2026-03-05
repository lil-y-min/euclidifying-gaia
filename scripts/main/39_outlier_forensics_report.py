#!/usr/bin/env python3
"""
Outlier forensics report for reconstruction runs.

Builds a p99-focused diagnostic package:
- joins test chi2 with trace/source_id
- joins weak labels + morphology features
- computes frozen gate p_nonpsf from gate package
- outputs slice tables, ranked feature diagnostics, and plots
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


def _q(v: np.ndarray, q: float) -> float:
    vv = np.asarray(v, dtype=float)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return float("nan")
    return float(np.quantile(vv, q))


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def compute_gate_p(df: pd.DataFrame, gate_pkg: Dict[str, object]) -> np.ndarray:
    feat_order = [str(x) for x in gate_pkg["feature_order"]]
    coef = np.asarray(gate_pkg["coef"], dtype=np.float64)
    intercept = float(gate_pkg["intercept"])
    z_clip = float(gate_pkg.get("z_clip", 8.0))
    scaler = gate_pkg["scaler"]

    X = np.zeros((len(df), len(feat_order)), dtype=np.float64)
    ok = np.ones(len(df), dtype=bool)
    for j, f in enumerate(feat_order):
        if f not in df.columns:
            ok &= False
            continue
        x = _safe_num(df[f]).to_numpy(dtype=np.float64)
        med = float(scaler[f]["median"])
        sc = float(scaler[f]["scale"])
        if (not np.isfinite(sc)) or sc <= 0:
            sc = 1.0
        z = (x - med) / sc
        z = np.clip(z, -z_clip, z_clip)
        X[:, j] = z
        ok &= np.isfinite(z)

    lin = intercept + np.dot(X, coef)
    p = sigmoid(lin)
    p[~ok] = np.nan
    return p.astype(np.float64)


def numeric_feature_ranking(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    out = df[df["is_outlier"]].copy()
    ref = df[~df["is_outlier"]].copy()
    for c in feature_cols:
        if c not in df.columns:
            continue
        xo = _safe_num(out[c]).to_numpy(dtype=float)
        xr = _safe_num(ref[c]).to_numpy(dtype=float)
        xo = xo[np.isfinite(xo)]
        xr = xr[np.isfinite(xr)]
        if xo.size < 20 or xr.size < 20:
            continue
        mo = float(np.mean(xo))
        mr = float(np.mean(xr))
        so = float(np.std(xo))
        sr = float(np.std(xr))
        sp = float(np.sqrt(max(1e-12, 0.5 * (so * so + sr * sr))))
        effect = (mo - mr) / sp
        dmed = float(np.median(xo) - np.median(xr))
        qo = _q(xo, 0.90)
        qr = _q(xr, 0.90)
        rows.append(
            {
                "feature": c,
                "n_outlier": int(xo.size),
                "n_ref": int(xr.size),
                "mean_outlier": mo,
                "mean_ref": mr,
                "median_delta": dmed,
                "p90_delta": float(qo - qr),
                "effect_size_z": float(effect),
                "abs_effect_size_z": float(abs(effect)),
            }
        )
    out_df = pd.DataFrame(rows)
    if len(out_df) == 0:
        return out_df
    return out_df.sort_values("abs_effect_size_z", ascending=False).reset_index(drop=True)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="base_v11_16d")
    ap.add_argument("--labels_csv", default="output/ml_runs/nn_psf_labels/labels_psf_weak.csv")
    ap.add_argument("--gate_package_json", default="report/model_decision/20260220_psf_gate_frozen_v11_sf1e3/gate_package.json")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--outlier_quantile", type=float, default=0.99)
    ap.add_argument("--top_n_export", type=int, default=300)
    ap.add_argument("--exclude_flux_features", action="store_true", default=True)
    ap.add_argument("--include_flux_features", action="store_true", default=False)
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    run_name = str(args.run_name)
    run_plot_dir = base_dir / "plots" / "ml_runs" / run_name
    run_trace_csv = base_dir / "output" / "ml_runs" / run_name / "trace" / "trace_test.csv"
    labels_csv = (base_dir / str(args.labels_csv)).resolve() if not str(args.labels_csv).startswith("/") else Path(str(args.labels_csv))
    gate_json = (base_dir / str(args.gate_package_json)).resolve() if not str(args.gate_package_json).startswith("/") else Path(str(args.gate_package_json))
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base_dir / "report" / "model_decision" / f"20260224_outlier_forensics_{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = run_plot_dir / "test_chi2_metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(metrics_csv)
    if not run_trace_csv.exists():
        raise FileNotFoundError(run_trace_csv)
    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)
    if not gate_json.exists():
        raise FileNotFoundError(gate_json)

    metrics = pd.read_csv(metrics_csv)
    for c in ["chi2nu", "sigma_bg", "logF_true", "logF_pred", "dlogF"]:
        if c in metrics.columns:
            metrics[c] = _safe_num(metrics[c])
    metrics["test_index"] = np.arange(len(metrics), dtype=np.int64)

    trace = pd.read_csv(run_trace_csv)
    trace["test_index"] = pd.to_numeric(trace["test_index"], errors="coerce").astype("Int64")
    trace["source_id"] = pd.to_numeric(trace["source_id"], errors="coerce").astype("Int64")
    trace = trace.dropna(subset=["test_index"]).copy()
    trace["test_index"] = trace["test_index"].astype(np.int64)

    labels = pd.read_csv(labels_csv)
    labels["source_id"] = pd.to_numeric(labels["source_id"], errors="coerce").astype("Int64")
    labels = labels.dropna(subset=["source_id"]).copy()
    labels["source_id"] = labels["source_id"].astype(np.int64)
    labels = labels.drop_duplicates(subset=["source_id"], keep="first")

    gate_pkg = json.loads(gate_json.read_text(encoding="utf-8"))

    df = metrics.merge(trace, on="test_index", how="left")
    df = df.merge(labels, on="source_id", how="left", suffixes=("", "_label"))

    df["p_nonpsf_gate"] = compute_gate_p(df, gate_pkg)
    q = float(args.outlier_quantile)
    thr = _q(df["chi2nu"].to_numpy(dtype=float), q)
    df["is_outlier"] = _safe_num(df["chi2nu"]) >= thr

    # Gate routing buckets
    p = _safe_num(df["p_nonpsf_gate"]).to_numpy(dtype=float)
    gate_bucket = np.full(len(df), "unknown", dtype=object)
    gate_bucket[np.isfinite(p) & (p <= 0.3)] = "psf_like(<=0.3)"
    gate_bucket[np.isfinite(p) & (p > 0.3) & (p < 0.7)] = "mixed(0.3-0.7)"
    gate_bucket[np.isfinite(p) & (p >= 0.7)] = "nonpsf_like(>=0.7)"
    df["gate_bucket"] = gate_bucket

    df.to_csv(out_dir / "forensics_full_joined.csv", index=False)

    # Top outlier export
    top_n = int(max(1, args.top_n_export))
    top = df.sort_values("chi2nu", ascending=False).head(top_n).copy()
    top.to_csv(out_dir / "forensics_top_outliers.csv", index=False)

    # Slice tables
    slice_cols = ["field_tag", "dataset_kind", "label", "gate_bucket", "split_code"]
    for col in slice_cols:
        if col not in df.columns:
            continue
        t_all = df[col].fillna("NA").value_counts(dropna=False).rename("n_all").to_frame()
        t_out = df.loc[df["is_outlier"], col].fillna("NA").value_counts(dropna=False).rename("n_outlier").to_frame()
        t = t_all.join(t_out, how="outer").fillna(0.0).reset_index().rename(columns={"index": col})
        t["n_all"] = t["n_all"].astype(int)
        t["n_outlier"] = t["n_outlier"].astype(int)
        t["frac_all"] = t["n_all"] / max(1, len(df))
        t["frac_outlier"] = t["n_outlier"] / max(1, int(df["is_outlier"].sum()))
        t["lift"] = t["frac_outlier"] / np.maximum(t["frac_all"], 1e-12)
        t = t.sort_values("lift", ascending=False)
        t.to_csv(out_dir / f"slice_{col}_lift.csv", index=False)

    # Ranked feature diagnostics
    candidate_features = [
        "sigma_bg", "dlogF", "logF_true", "logF_pred",
        "confidence", "score_psf_like", "score_non_psf_like",
        "m_concentration_r2_r6", "m_asymmetry_180", "m_ellipticity",
        "m_peak_sep_pix", "m_edge_flux_frac", "m_peak_ratio_2over1",
        "ruwe", "ipd_frac_multi_peak",
        "feat_log10_snr", "feat_ruwe", "feat_astrometric_excess_noise",
        "feat_parallax_over_error", "feat_visibility_periods_used",
        "feat_ipd_frac_multi_peak", "feat_c_star", "feat_pm_significance",
        "feat_astrometric_excess_noise_sig", "feat_ipd_gof_harmonic_amplitude",
        "feat_ipd_gof_harmonic_phase", "feat_ipd_frac_odd_win",
        "feat_phot_bp_n_contaminated_transits", "feat_phot_bp_n_blended_transits",
        "feat_phot_rp_n_contaminated_transits", "feat_phot_rp_n_blended_transits",
        "p_nonpsf_gate",
    ]
    use_flux = bool(args.include_flux_features) and (not bool(args.exclude_flux_features))
    if not use_flux:
        candidate_features = [c for c in candidate_features if c not in {"dlogF", "logF_true", "logF_pred"}]
    rank_df = numeric_feature_ranking(df, candidate_features)
    rank_df.to_csv(out_dir / "candidate_feature_rankings.csv", index=False)

    # Summary JSON
    summary = {
        "run_name": run_name,
        "n_test": int(len(df)),
        "outlier_quantile": q,
        "outlier_threshold_chi2nu": float(thr),
        "n_outliers": int(df["is_outlier"].sum()),
        "frac_outliers": float(df["is_outlier"].mean()),
        "chi2nu_median": _q(df["chi2nu"].to_numpy(dtype=float), 0.50),
        "chi2nu_p90": _q(df["chi2nu"].to_numpy(dtype=float), 0.90),
        "chi2nu_p99": _q(df["chi2nu"].to_numpy(dtype=float), 0.99),
        "label_coverage_frac": float(np.mean(pd.notna(df.get("label", np.nan)))),
        "gate_p_coverage_frac": float(np.mean(np.isfinite(_safe_num(df["p_nonpsf_gate"]).to_numpy(dtype=float)))),
    }
    (out_dir / "forensics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots
    vals = _safe_num(df["chi2nu"]).to_numpy(dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    plt.figure(figsize=(8.3, 4.8))
    plt.hist(np.log10(vals), bins=120)
    if np.isfinite(thr) and thr > 0:
        plt.axvline(np.log10(thr), color="red", linestyle="--", linewidth=1.2, label=f"q{int(100*q)} threshold")
    plt.xlabel("log10(chi2nu)")
    plt.ylabel("Count")
    plt.title(f"{run_name}: chi2nu distribution with outlier threshold")
    plt.legend()
    savefig(out_dir / "forensics_01_chi2_distribution.png")

    if "field_tag" in df.columns:
        sf = pd.read_csv(out_dir / "slice_field_tag_lift.csv")
        sf = sf[sf["n_outlier"] > 0].head(15).copy()
        plt.figure(figsize=(8.8, 6.0))
        plt.barh(sf["field_tag"], sf["lift"])
        plt.gca().invert_yaxis()
        plt.xlabel("Outlier lift vs global")
        plt.title("Top fields enriched in outliers")
        savefig(out_dir / "forensics_02_field_lift.png")

    pg = _safe_num(df["p_nonpsf_gate"]).to_numpy(dtype=float)
    m_out = np.isfinite(pg) & df["is_outlier"].to_numpy(dtype=bool)
    m_ref = np.isfinite(pg) & (~df["is_outlier"].to_numpy(dtype=bool))
    if np.any(m_out) and np.any(m_ref):
        plt.figure(figsize=(8.2, 4.8))
        plt.hist(pg[m_ref], bins=50, alpha=0.6, density=True, label="non-outlier")
        plt.hist(pg[m_out], bins=50, alpha=0.6, density=True, label="outlier")
        plt.xlabel("p_nonpsf_gate")
        plt.ylabel("Density")
        plt.title("Gate probability: outlier vs non-outlier")
        plt.legend()
        savefig(out_dir / "forensics_03_gate_prob_density.png")

    if len(rank_df) > 0:
        r = rank_df.head(12).iloc[::-1]
        plt.figure(figsize=(8.8, 6.2))
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in r["effect_size_z"]]
        plt.barh(r["feature"], r["effect_size_z"], color=colors)
        plt.axvline(0.0, color="black", linewidth=1)
        plt.xlabel("Effect size z (outlier - non-outlier)")
        plt.title("Top candidate diagnostic features")
        savefig(out_dir / "forensics_04_feature_effects.png")

    # Markdown summary
    lines = []
    lines.append("# Outlier Forensics Summary")
    lines.append("")
    lines.append(f"- Run: `{run_name}`")
    lines.append(f"- n_test: {summary['n_test']}")
    lines.append(f"- Outlier quantile: q={q:.3f}")
    lines.append(f"- Outlier threshold chi2nu: {summary['outlier_threshold_chi2nu']:.6g}")
    lines.append(f"- Outliers: {summary['n_outliers']} ({100.0*summary['frac_outliers']:.2f}%)")
    lines.append(f"- Label coverage in test-joined rows: {100.0*summary['label_coverage_frac']:.2f}%")
    lines.append(f"- Gate p coverage: {100.0*summary['gate_p_coverage_frac']:.2f}%")
    if len(rank_df) > 0:
        lines.append("")
        lines.append("## Top ranked candidate features (by abs effect size)")
        for _, row in rank_df.head(10).iterrows():
            lines.append(
                f"- `{row['feature']}`: effect_z={float(row['effect_size_z']):+.3f}, "
                f"median_delta={float(row['median_delta']):+.4g}, p90_delta={float(row['p90_delta']):+.4g}"
            )
    (out_dir / "forensics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
