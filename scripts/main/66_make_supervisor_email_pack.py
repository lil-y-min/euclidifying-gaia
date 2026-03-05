#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def cp(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_classifier_curves(
    multi_pred_csv: Path,
    nonpsf_pred_csv: Path,
    out_dir: Path,
) -> pd.DataFrame:
    dm = pd.read_csv(multi_pred_csv, low_memory=False)
    dn = pd.read_csv(nonpsf_pred_csv, low_memory=False)
    for d in [dm, dn]:
        d["y_true"] = pd.to_numeric(d["y_true"], errors="coerce")
        d["p_xgb"] = pd.to_numeric(d["p_xgb"], errors="coerce")
        d.dropna(subset=["y_true", "p_xgb"], inplace=True)
        d["y_true"] = d["y_true"].astype(int)

    rows = []
    for name, d in [("multipeak", dm), ("nonpsf", dn)]:
        y = d["y_true"].to_numpy(dtype=int)
        p = d["p_xgb"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        roc_auc = float(auc(fpr, tpr))
        pr_auc = float(auc(rec, prec))
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(out_dir / f"roc_{name}.csv", index=False)
        pd.DataFrame({"recall": rec, "precision": prec}).to_csv(out_dir / f"pr_{name}.csv", index=False)
        rows.append({"classifier": name, "n": int(len(d)), "roc_auc": roc_auc, "pr_auc": pr_auc, "base_rate": float(np.mean(y))})

    # Combined figure
    fpr_m = pd.read_csv(out_dir / "roc_multipeak.csv")
    fpr_n = pd.read_csv(out_dir / "roc_nonpsf.csv")
    pr_m = pd.read_csv(out_dir / "pr_multipeak.csv")
    pr_n = pd.read_csv(out_dir / "pr_nonpsf.csv")
    sm = pd.DataFrame(rows).set_index("classifier")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    ax.plot(fpr_m["fpr"], fpr_m["tpr"], label=f"multipeak AUC={sm.loc['multipeak','roc_auc']:.3f}", lw=2)
    ax.plot(fpr_n["fpr"], fpr_n["tpr"], label=f"nonpsf AUC={sm.loc['nonpsf','roc_auc']:.3f}", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(frameon=True)

    ax = axes[1]
    ax.plot(pr_m["recall"], pr_m["precision"], label=f"multipeak AUPRC={sm.loc['multipeak','pr_auc']:.3f}", lw=2)
    ax.plot(pr_n["recall"], pr_n["precision"], label=f"nonpsf AUPRC={sm.loc['nonpsf','pr_auc']:.3f}", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves")
    ax.legend(frameon=True)
    fig.tight_layout()
    savefig(out_dir / "roc_pr_both_classifiers.png")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "classifier_auc_summary.csv", index=False)
    return out


def make_morph_pearson_plot(summary_csv: Path, out_png: Path) -> None:
    d = pd.read_csv(summary_csv)
    d = d.sort_values("pearson_r", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(d["metric"], d["pearson_r"], color="#4e79a7")
    plt.axhline(0.0, color="k", lw=1)
    plt.ylim(min(-0.1, float(np.nanmin(d["pearson_r"])) - 0.05), 1.0)
    plt.ylabel("Pearson r (true vs predicted)")
    plt.title("Morphology Sanity Check: True vs Predicted Metrics")
    savefig(out_png)


def build_true_vs_pred_from_full_morph(morph_csv: Path, out_dir: Path) -> pd.DataFrame:
    d = pd.read_csv(morph_csv, low_memory=False)
    metric_map = {
        "ellipticity": ("m_ellipticity_true", "m_ellipticity_pred"),
        "concentration": ("m_concentration_r2_r6_true", "m_concentration_r2_r6_pred"),
        "edge_flux_frac": ("m_edge_flux_frac_true", "m_edge_flux_frac_pred"),
        "asymmetry": ("m_asymmetry_180_true", "m_asymmetry_180_pred"),
    }
    rows = []
    for m, (tcol, pcol) in metric_map.items():
        if tcol not in d.columns or pcol not in d.columns:
            continue
        t = pd.to_numeric(d[tcol], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(d[pcol], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(t) & np.isfinite(p)
        n = int(np.sum(mask))
        if n < 3:
            rows.append({"metric": m, "pearson_r": np.nan, "spearman_r": np.nan, "rmse": np.nan, "collapse_ratio": np.nan, "n": n})
            continue
        tt = t[mask]
        pp = p[mask]
        pear = float(np.corrcoef(tt, pp)[0, 1]) if (np.std(tt) > 0 and np.std(pp) > 0) else np.nan
        spea = float(pd.Series(tt).rank().corr(pd.Series(pp).rank(), method="pearson"))
        rmse = float(np.sqrt(np.mean((tt - pp) ** 2)))
        collapse = float(np.std(pp) / (np.std(tt) + 1e-12))
        rows.append({"metric": m, "pearson_r": pear, "spearman_r": spea, "rmse": rmse, "collapse_ratio": collapse, "n": n})
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "true_vs_pred_morph_summary.csv", index=False)
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/data/yn316/Codes/report/model_decision/20260303_supervisor_email_pack")
    ap.add_argument("--checkpoint_dir", default="/data/yn316/Codes/report/model_decision/20260303_supervisor_checkpoint_unified")
    ap.add_argument("--multi_run_dir", default="/data/yn316/Codes/report/model_decision/20260302_gaia_first_multipeak_xgb_base_v16_clean_final_manualv8")
    ap.add_argument("--nonpsf_run_dir", default="/data/yn316/Codes/report/model_decision/20260302_gaia_first_nonpsf_xgb_base_v16_clean_final_manualv8")
    ap.add_argument("--morph_csv", default="/data/yn316/Codes/report/model_decision/20260227_morph_sanity_base_v16_clean_final_manualv8/morph_pred_vs_true_rows.csv")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sec1 = out / "01_reconstruction_morph_sanity"
    sec2 = out / "02_classifier_curves"
    sec3 = out / "03_top_probability_stamps"
    sec4 = out / "04_umap_pixels_prediction_colored"
    sec5 = out / "05_umap_16d_gaia_main"
    for s in [sec1, sec2, sec3, sec4, sec5]:
        s.mkdir(parents=True, exist_ok=True)

    chk = Path(args.checkpoint_dir)
    corr = chk / "01_correlations"
    # Section 1
    build_true_vs_pred_from_full_morph(Path(args.morph_csv), sec1)
    make_morph_pearson_plot(sec1 / "true_vs_pred_morph_summary.csv", sec1 / "morph_true_vs_pred_pearson_bar.png")
    for m in ["ellipticity", "concentration", "edge_flux_frac", "asymmetry"]:
        for pref in ["true_vs_pred_scatter", "residual", "hist_true_pred"]:
            src = corr / f"{pref}_{m}.png"
            if src.exists():
                cp(src, sec1 / src.name)

    # Section 2
    auc_df = build_classifier_curves(
        multi_pred_csv=Path(args.multi_run_dir) / "test_predictions.csv",
        nonpsf_pred_csv=Path(args.nonpsf_run_dir) / "test_predictions.csv",
        out_dir=sec2,
    )
    # also keep original PR curve PNGs
    src_pr1 = Path(args.multi_run_dir) / "pr_curve.png"
    src_pr2 = Path(args.nonpsf_run_dir) / "pr_curve.png"
    if src_pr1.exists():
        cp(src_pr1, sec2 / "pr_curve_multipeak_original.png")
    if src_pr2.exists():
        cp(src_pr2, sec2 / "pr_curve_nonpsf_original.png")

    # Section 3
    tops = chk / "top_prob_stamps"
    for fn in ["top_multipeak_stamps.png", "top_nonpsf_stamps.png", "top_multipeak_probs.csv", "top_nonpsf_probs.csv"]:
        src = tops / fn
        if src.exists():
            cp(src, sec3 / fn)

    # Section 4
    for fn in [
        "01_umappix_p_multipeak.png",
        "02_umappix_p_nonpsf.png",
        "03_umappix_field.png",
        "04_umappix_wds_doubles.png",
        "05_umappix_pred_morph_panels.png",
    ]:
        src = chk / "umap_pixels" / fn
        if src.exists():
            cp(src, sec4 / fn)
    # include test-split versions too
    for fn in [
        "01_umappix_test_p_multipeak.png",
        "02_umappix_test_p_nonpsf.png",
        "05_umappix_test_pred_morph_panels.png",
    ]:
        src = chk / "umap_pixels_testsplit" / fn
        if src.exists():
            cp(src, sec4 / fn)

    # Section 5 (main thesis maps in Gaia feature-space)
    for fn in ["01_umap16_p_multipeak.png", "02_umap16_p_nonpsf.png", "06_umap16_nuisance.png"]:
        src = chk / "umap_16d" / fn
        if src.exists():
            cp(src, sec5 / fn)
    for fn in ["01_umap16_test_p_multipeak.png", "02_umap16_test_p_nonpsf.png", "06_umap16_test_nuisance.png"]:
        src = chk / "umap_16d_testsplit" / fn
        if src.exists():
            cp(src, sec5 / fn)

    manifest = {
        "out_dir": str(out),
        "sections": {
            "01_reconstruction_morph_sanity": str(sec1),
            "02_classifier_curves": str(sec2),
            "03_top_probability_stamps": str(sec3),
            "04_umap_pixels_prediction_colored": str(sec4),
            "05_umap_16d_gaia_main": str(sec5),
        },
        "classifier_auc_summary": auc_df.to_dict(orient="records"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out)


if __name__ == "__main__":
    main()
