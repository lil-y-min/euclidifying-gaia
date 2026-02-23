#!/usr/bin/env python3
"""
Build tables and visualizations for a weighted XGB sweep.

Expected input per run:
  plots/ml_runs/<run_name>/test_chi2_metrics.csv

Outputs under --out_dir:
  weighted_sweep_metrics.csv
  weighted_sweep_delta_vs_baseline.csv
  weighted_sweep_summary.md
  weighted_sweep_01_chi2_vs_weight.png
  weighted_sweep_02_delta_vs_baseline.png
  weighted_sweep_03_gateA_passfail.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_run_spec(spec: str) -> Tuple[str, float]:
    parts = str(spec).split("=")
    if len(parts) != 2:
        raise ValueError(f"Invalid --run_spec '{spec}', expected name=weight")
    name = parts[0].strip()
    weight = float(parts[1].strip())
    if not name:
        raise ValueError(f"Invalid --run_spec '{spec}': empty run name")
    return name, weight


def _q(v: np.ndarray, q: float) -> float:
    vv = np.asarray(v, dtype=float)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return float("nan")
    return float(np.quantile(vv, q))


def load_run_metrics(base_dir: Path, run_name: str) -> Dict[str, float]:
    csv_path = base_dir / "plots" / "ml_runs" / run_name / "test_chi2_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV for run '{run_name}': {csv_path}")
    df = pd.read_csv(csv_path, usecols=["chi2nu"])
    chi2 = pd.to_numeric(df["chi2nu"], errors="coerce").dropna().to_numpy(dtype=float)
    if chi2.size == 0:
        raise RuntimeError(f"No finite chi2nu values in {csv_path}")
    return {
        "run_name": run_name,
        "n_test": int(chi2.size),
        "chi2nu_median": _q(chi2, 0.50),
        "chi2nu_p90": _q(chi2, 0.90),
        "chi2nu_p99": _q(chi2, 0.99),
        "chi2nu_mean": float(np.mean(chi2)),
    }


def plot_chi2_vs_weight(df: pd.DataFrame, out_png: Path) -> None:
    d = df.sort_values("weight")
    x = d["weight"].to_numpy(dtype=float)
    plt.figure(figsize=(9, 5.5))
    plt.plot(x, d["chi2nu_median"], marker="o", label="median")
    plt.plot(x, d["chi2nu_p90"], marker="o", label="p90")
    plt.plot(x, d["chi2nu_p99"], marker="o", label="p99")
    plt.xlabel("Non-PSF weight")
    plt.ylabel("chi2nu (lower is better)")
    plt.title("Weighted Sweep: chi2nu vs non-PSF weight")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_delta_vs_baseline(df_delta: pd.DataFrame, out_png: Path) -> None:
    d = df_delta.sort_values("weight")
    x = np.arange(len(d), dtype=float)
    labels = [f"{rn}\n(w={w:g})" for rn, w in zip(d["run_name"], d["weight"])]
    wbar = 0.26
    plt.figure(figsize=(10, 5.8))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.bar(x - wbar, d["median_change_pct"], width=wbar, label="median change %")
    plt.bar(x, d["p90_improvement_pct"], width=wbar, label="p90 improvement %")
    plt.bar(x + wbar, d["p99_improvement_pct"], width=wbar, label="p99 improvement %")
    plt.xticks(x, labels)
    plt.ylabel("Percent vs baseline (%)")
    plt.title("Weighted Sweep: Delta vs baseline")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_gate_a(df_delta: pd.DataFrame, out_png: Path, target_p99: float, median_guard: float) -> None:
    d = df_delta.copy()
    colors = np.where(d["gateA_pass"], "#2ca02c", "#d62728")
    plt.figure(figsize=(7.2, 6.0))
    plt.axvline(target_p99, color="#1f77b4", linestyle="--", linewidth=1, label=f"p99 target >= {target_p99:.1f}%")
    plt.axhline(median_guard, color="#ff7f0e", linestyle="--", linewidth=1, label=f"median guard <= {median_guard:.1f}%")
    plt.scatter(d["p99_improvement_pct"], d["median_change_pct"], c=colors, s=90)
    for _, r in d.iterrows():
        plt.text(
            float(r["p99_improvement_pct"]) + 0.2,
            float(r["median_change_pct"]) + 0.2,
            f"{r['run_name']} (w={r['weight']:g})",
            fontsize=8,
        )
    plt.xlabel("p99 improvement (%)")
    plt.ylabel("median degradation (%)")
    plt.title("Gate-A View: Tail gain vs median cost")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--run_spec",
        action="append",
        required=True,
        help="Run spec as run_name=weight ; pass multiple times.",
    )
    ap.add_argument("--baseline_run", required=True, help="Run name used as baseline reference.")
    ap.add_argument("--target_p99_improvement", type=float, default=10.0, help="Gate-A p99 improvement target (%%).")
    ap.add_argument("--median_guardrail_pct", type=float, default=2.0, help="Gate-A median degradation limit (%%).")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [parse_run_spec(s) for s in args.run_spec]
    if len({r for r, _ in run_specs}) != len(run_specs):
        raise RuntimeError("Duplicate run_name in --run_spec.")

    rows: List[Dict[str, float]] = []
    for run_name, weight in run_specs:
        m = load_run_metrics(base_dir=base_dir, run_name=run_name)
        m["weight"] = float(weight)
        rows.append(m)

    df = pd.DataFrame(rows)
    if args.baseline_run not in set(df["run_name"].tolist()):
        raise RuntimeError("--baseline_run must be included in --run_spec.")
    b = df.loc[df["run_name"] == args.baseline_run].iloc[0]
    b_med = float(b["chi2nu_median"])
    b_p90 = float(b["chi2nu_p90"])
    b_p99 = float(b["chi2nu_p99"])

    d = df.copy()
    d["median_change_pct"] = 100.0 * (d["chi2nu_median"] / b_med - 1.0)
    d["p90_improvement_pct"] = 100.0 * (1.0 - d["chi2nu_p90"] / b_p90)
    d["p99_improvement_pct"] = 100.0 * (1.0 - d["chi2nu_p99"] / b_p99)
    d["gateA_pass"] = (
        (d["p99_improvement_pct"] >= float(args.target_p99_improvement))
        & (d["median_change_pct"] <= float(args.median_guardrail_pct))
    )

    df.sort_values("weight").to_csv(out_dir / "weighted_sweep_metrics.csv", index=False)
    d.sort_values("weight").to_csv(out_dir / "weighted_sweep_delta_vs_baseline.csv", index=False)

    plot_chi2_vs_weight(df, out_dir / "weighted_sweep_01_chi2_vs_weight.png")
    plot_delta_vs_baseline(d, out_dir / "weighted_sweep_02_delta_vs_baseline.png")
    plot_gate_a(
        d,
        out_dir / "weighted_sweep_03_gateA_passfail.png",
        target_p99=float(args.target_p99_improvement),
        median_guard=float(args.median_guardrail_pct),
    )

    d_nonbase = d.loc[d["run_name"] != args.baseline_run].copy()
    d_nonbase = d_nonbase.sort_values(["gateA_pass", "p99_improvement_pct", "median_change_pct"], ascending=[False, False, True])
    winner = d_nonbase.iloc[0] if len(d_nonbase) > 0 else None

    lines: List[str] = []
    lines.append("# Weighted sweep summary")
    lines.append("")
    lines.append(f"- Baseline run: `{args.baseline_run}`")
    lines.append(f"- Baseline chi2nu: median={b_med:.6g}, p90={b_p90:.6g}, p99={b_p99:.6g}")
    lines.append(f"- Gate-A target: p99 improvement >= {float(args.target_p99_improvement):.1f}%")
    lines.append(f"- Gate-A guardrail: median degradation <= {float(args.median_guardrail_pct):.1f}%")
    if winner is not None:
        lines.append("")
        lines.append("## Top candidate by sort rule")
        lines.append(
            f"- `{winner['run_name']}` (w={float(winner['weight']):g}), "
            f"pass={bool(winner['gateA_pass'])}, "
            f"p99_improvement={float(winner['p99_improvement_pct']):.3f}%, "
            f"median_change={float(winner['median_change_pct']):.3f}%"
        )

    (out_dir / "weighted_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
