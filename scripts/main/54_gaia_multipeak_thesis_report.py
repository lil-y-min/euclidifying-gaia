#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_run_metrics(run_dir: Path, op_tag: str) -> Dict[str, float]:
    ms = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    bvx = pd.read_csv(run_dir / "baseline_vs_xgb.csv")
    ops = pd.read_csv(run_dir / "operating_points.csv")

    heu = bvx[bvx["method"] == "heuristic_ruwe_or_ipd"].iloc[0]
    xgb_method = f"xgb_{op_tag}"
    if not np.any(bvx["method"] == xgb_method):
        # fallback to precision_floor, then target_rate
        if np.any(bvx["method"] == "xgb_precision_floor"):
            xgb_method = "xgb_precision_floor"
        else:
            xgb_method = "xgb_target_rate"
    xg = bvx[bvx["method"] == xgb_method].iloc[0]

    op_tag_use = str(xgb_method).replace("xgb_", "")
    op_row = ops[ops["op_tag"] == op_tag_use]
    if op_row.empty:
        op_row = ops.iloc[[0]]
    op = op_row.iloc[0]

    return {
        "n_test": float(ms["n_test"]),
        "base_rate": float(ms["class_balance"]["test_base_rate"]),
        "auroc": float(ms["xgb"]["test_auroc"]),
        "auprc": float(ms["xgb"]["test_auprc"]),
        "heuristic_precision": float(heu["precision"]),
        "heuristic_recall": float(heu["recall"]),
        "heuristic_f1": float(heu["f1"]),
        "heuristic_pred_rate": float(heu["pred_pos_rate"]),
        "xgb_method": xgb_method,
        "xgb_threshold": float(op["threshold"]),
        "xgb_precision": float(xg["precision"]),
        "xgb_recall": float(xg["recall"]),
        "xgb_f1": float(xg["f1"]),
        "xgb_pred_rate": float(xg["pred_pos_rate"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered_run_dir", required=True)
    ap.add_argument("--expanded_run_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--op_tag", default="precision_floor")
    args = ap.parse_args()

    filtered_dir = Path(args.filtered_run_dir)
    expanded_dir = Path(args.expanded_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f = load_run_metrics(filtered_dir, op_tag=str(args.op_tag))
    e = load_run_metrics(expanded_dir, op_tag=str(args.op_tag))

    table = pd.DataFrame(
        [
            {
                "dataset": "filtered",
                **f,
            },
            {
                "dataset": "expanded",
                **e,
            },
        ]
    )
    table.to_csv(out_dir / "comparison_table.csv", index=False)

    # Panel 1: PR curves
    pr_f = pd.read_csv(filtered_dir / "pr_curve.csv")
    pr_e = pd.read_csv(expanded_dir / "pr_curve.csv")
    # Drop NaN thresholds rows duplicate endpoints are okay, keep full curve.

    # Panel 2: precision/recall bar chart for baseline vs XGB op.
    labels = [
        "Filt-Heu P",
        "Filt-Heu R",
        "Filt-XGB P",
        "Filt-XGB R",
        "Exp-Heu P",
        "Exp-Heu R",
        "Exp-XGB P",
        "Exp-XGB R",
    ]
    vals = [
        f["heuristic_precision"],
        f["heuristic_recall"],
        f["xgb_precision"],
        f["xgb_recall"],
        e["heuristic_precision"],
        e["heuristic_recall"],
        e["xgb_precision"],
        e["xgb_recall"],
    ]
    cols = ["#9C9A40", "#9C9A40", "#1A6F8A", "#1A6F8A", "#B56D1F", "#B56D1F", "#0E4D64", "#0E4D64"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    ax = axes[0]
    ax.plot(pr_f["recall"], pr_f["precision"], lw=2.0, label=f"Filtered (AUPRC={f['auprc']:.3f})", color="#1A6F8A")
    ax.plot(pr_e["recall"], pr_e["precision"], lw=2.0, label=f"Expanded (AUPRC={e['auprc']:.3f})", color="#B56D1F")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    x = np.arange(len(labels))
    ax.bar(x, vals, color=cols)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs XGB Operating Point")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Gaia-First Multi-peak: Filtered vs Expanded", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_figure.png", dpi=180)
    plt.close(fig)

    # Extra calibration/probability histogram on expanded manual subsets if available.
    pred_path = expanded_dir / "test_predictions.csv"
    if pred_path.exists():
        d = pd.read_csv(pred_path)
        if "manual_removed" in d.columns:
            kept = d[d["manual_removed"] == 0]
            rem = d[d["manual_removed"] == 1]
            plt.figure(figsize=(8.0, 5.2))
            bins = np.linspace(0.0, 1.0, 41)
            if len(kept) > 0:
                plt.hist(kept["p_xgb"], bins=bins, histtype="step", linewidth=2.0, density=True, label="manual_kept")
            if len(rem) > 0:
                plt.hist(rem["p_xgb"], bins=bins, histtype="step", linewidth=2.0, density=True, label="manual_removed")
            plt.xlabel("Predicted probability")
            plt.ylabel("Density")
            plt.title("Probability Distribution by Manual-Quality Subset (expanded)")
            plt.grid(alpha=0.25)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(out_dir / "expanded_prob_hist_manual_kept_vs_removed.png", dpi=180)
            plt.close()

    summary = {
        "filtered_run_dir": str(filtered_dir),
        "expanded_run_dir": str(expanded_dir),
        "op_tag_requested": str(args.op_tag),
        "outputs": {
            "comparison_table_csv": str(out_dir / "comparison_table.csv"),
            "comparison_figure_png": str(out_dir / "comparison_figure.png"),
            "expanded_prob_hist_png": str(out_dir / "expanded_prob_hist_manual_kept_vs_removed.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
