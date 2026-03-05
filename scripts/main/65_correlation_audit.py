#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRUE_METRIC_MAP = {
    "ellipticity": "m_ellipticity_true",
    "concentration": "m_concentration_r2_r6_true",
    "edge_flux_frac": "m_edge_flux_frac_true",
    "asymmetry": "m_asymmetry_180_true",
}

PRED_METRIC_MAP = {
    "ellipticity": "m_ellipticity_pred",
    "concentration": "m_concentration_r2_r6_pred",
    "edge_flux_frac": "m_edge_flux_frac_pred",
    "asymmetry": "m_asymmetry_180_pred",
}


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def corr_pair(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(m))
    if n < 3:
        return np.nan, np.nan, n
    xx = x[m]
    yy = y[m]
    sx = np.std(xx)
    sy = np.std(yy)
    if sx <= 0 or sy <= 0:
        pear = np.nan
    else:
        pear = float(np.corrcoef(xx, yy)[0, 1])
    spea = float(pd.Series(xx).rank().corr(pd.Series(yy).rank(), method="pearson"))
    return pear, spea, n


def rmse_pair(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(m))
    if n == 0:
        return np.nan, n
    return float(np.sqrt(np.mean((x[m] - y[m]) ** 2))), n


def plot_heatmap(df: pd.DataFrame, value_col: str, out_png: Path, title: str) -> None:
    p = df.pivot(index="gaia_feature", columns="target_metric", values=value_col)
    rows = sorted(p.index.tolist())
    cols = sorted(p.columns.tolist())
    arr = p.loc[rows, cols].to_numpy(dtype=float)
    plt.figure(figsize=(1.2 * max(6, len(cols)), 0.35 * max(10, len(rows))))
    im = plt.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label=value_col)
    plt.xticks(np.arange(len(cols)), cols, rotation=40, ha="right", fontsize=8)
    plt.yticks(np.arange(len(rows)), rows, fontsize=8)
    plt.title(title)
    savefig(out_png)


def plot_scatter_with_trend(df: pd.DataFrame, xcol: str, ycol: str, out_png: Path, title: str) -> None:
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 10:
        return
    xx = x[m]
    yy = y[m]
    plt.figure(figsize=(5.6, 4.8))
    plt.scatter(xx, yy, s=8, alpha=0.28)
    try:
        b1, b0 = np.polyfit(xx, yy, 1)
        xs = np.linspace(np.nanpercentile(xx, 1), np.nanpercentile(xx, 99), 100)
        ys = b1 * xs + b0
        plt.plot(xs, ys, "r-", lw=1.5)
    except Exception:
        pass
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gaia_csv",
        default="/data/yn316/Codes/output/experiments/embeddings/umap16d_manualv8_filtered/embedding_umap.csv",
    )
    ap.add_argument(
        "--morph_csv",
        default="/data/yn316/Codes/report/model_decision/20260227_morph_sanity_base_v16_clean_final_manualv8/morph_pred_vs_true_rows.csv",
    )
    ap.add_argument(
        "--robust_peaksep_csv",
        default="/data/yn316/Codes/report/model_decision/20260227_robust_peaksep_base_v16_clean_final_manualv8_relaxed_v2/peaksep_rows.csv",
    )
    ap.add_argument(
        "--out_dir",
        default="/data/yn316/Codes/report/model_decision/20260303_supervisor_checkpoint_unified/01_correlations",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gaia = pd.read_csv(Path(args.gaia_csv), low_memory=False)
    morph = pd.read_csv(Path(args.morph_csv), low_memory=False)
    peak = pd.read_csv(Path(args.robust_peaksep_csv), low_memory=False)

    # Basic normalization / dedup
    for df in [gaia, morph, peak]:
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
        df.dropna(subset=["source_id"], inplace=True)
        df["source_id"] = df["source_id"].astype(np.int64)
        df.drop_duplicates(subset=["source_id"], keep="first", inplace=True)

    # Build merged table for audit.
    gaia_cols = [c for c in gaia.columns if c.startswith("feat_")]
    use_morph = ["source_id", *TRUE_METRIC_MAP.values(), *PRED_METRIC_MAP.values()]
    use_morph = [c for c in use_morph if c in morph.columns]
    use_peak = ["source_id", "new_true_flag", "new_pred_flag", "new_true_sep", "new_pred_sep", "old_pred_sep"]
    use_peak = [c for c in use_peak if c in peak.columns]
    merged = gaia[["source_id", *gaia_cols]].merge(morph[use_morph], on="source_id", how="inner")
    merged = merged.merge(peak[use_peak], on="source_id", how="left")
    merged["true_multipeak_flag"] = pd.to_numeric(merged.get("new_true_flag", np.nan), errors="coerce")
    merged["true_peak_sep_pix"] = pd.to_numeric(merged.get("new_true_sep", np.nan), errors="coerce")
    merged["pred_multipeak_flag"] = pd.to_numeric(merged.get("new_pred_flag", np.nan), errors="coerce")
    merged["pred_peak_sep_pix_robust"] = pd.to_numeric(merged.get("new_pred_sep", np.nan), errors="coerce")
    merged["pred_peak_sep_pix_old"] = pd.to_numeric(merged.get("old_pred_sep", np.nan), errors="coerce")

    # 1A Gaia features <-> true morphology targets
    target_cols = {
        "ellipticity": "m_ellipticity_true",
        "concentration": "m_concentration_r2_r6_true",
        "edge_flux_frac": "m_edge_flux_frac_true",
        "asymmetry": "m_asymmetry_180_true",
        "true_multipeak_flag": "true_multipeak_flag",
        "true_peak_sep_pix": "true_peak_sep_pix",
    }
    rows = []
    for tname, tcol in target_cols.items():
        if tcol not in merged.columns:
            continue
        y = pd.to_numeric(merged[tcol], errors="coerce").to_numpy(dtype=float)
        for g in gaia_cols:
            x = pd.to_numeric(merged[g], errors="coerce").to_numpy(dtype=float)
            pear, spea, n = corr_pair(x, y)
            rows.append(
                {
                    "target_metric": tname,
                    "gaia_feature": g,
                    "pearson_r": pear,
                    "spearman_r": spea,
                    "n": n,
                }
            )
    corr_df = pd.DataFrame(rows).sort_values(["target_metric", "spearman_r"], ascending=[True, False]).reset_index(drop=True)
    corr_df.to_csv(out / "gaia_vs_morph_corr.csv", index=False)
    plot_heatmap(corr_df, "pearson_r", out / "heatmap_gaia_vs_morph_pearson.png", "Gaia features vs morphology (Pearson)")
    plot_heatmap(corr_df, "spearman_r", out / "heatmap_gaia_vs_morph_spearman.png", "Gaia features vs morphology (Spearman)")

    # Top-3 strongest pairs per target (by abs spearman)
    scat_dir = out / "scatters_top3"
    scat_dir.mkdir(parents=True, exist_ok=True)
    for tname, tcol in target_cols.items():
        sub = corr_df[corr_df["target_metric"] == tname].copy()
        if sub.empty:
            continue
        sub["abs_s"] = sub["spearman_r"].abs()
        top = sub.sort_values("abs_s", ascending=False).head(3)
        for _, r in top.iterrows():
            g = str(r["gaia_feature"])
            plot_scatter_with_trend(
                merged,
                xcol=g,
                ycol=tcol,
                out_png=scat_dir / f"scatter_{tname}__{g}.png",
                title=f"{tname} vs {g}",
            )

    # 1B True vs predicted summary
    summ_rows = []
    for k, tcol in TRUE_METRIC_MAP.items():
        pcol = PRED_METRIC_MAP.get(k, "")
        if (tcol not in merged.columns) or (pcol not in merged.columns):
            continue
        t = pd.to_numeric(merged[tcol], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(merged[pcol], errors="coerce").to_numpy(dtype=float)
        pear, spea, n = corr_pair(t, p)
        rm, _ = rmse_pair(t, p)
        m = np.isfinite(t) & np.isfinite(p)
        collapse = float(np.nanstd(p[m]) / (np.nanstd(t[m]) + 1e-12)) if np.sum(m) else np.nan
        summ_rows.append(
            {
                "metric": k,
                "pearson_r": pear,
                "spearman_r": spea,
                "rmse": rm,
                "collapse_ratio": collapse,
                "n": n,
            }
        )

        # per-metric plots
        plot_scatter_with_trend(
            merged.rename(columns={tcol: "m_true", pcol: "m_pred"}),
            xcol="m_true",
            ycol="m_pred",
            out_png=out / f"true_vs_pred_scatter_{k}.png",
            title=f"true vs pred: {k}",
        )
        mm = np.isfinite(t) & np.isfinite(p)
        if np.sum(mm) >= 10:
            rr = p[mm] - t[mm]
            plt.figure(figsize=(5.6, 4.8))
            plt.scatter(t[mm], rr, s=8, alpha=0.28)
            plt.axhline(0.0, color="r", ls="--", lw=1.2)
            plt.xlabel("true")
            plt.ylabel("pred-true")
            plt.title(f"residual: {k}")
            savefig(out / f"residual_{k}.png")

            plt.figure(figsize=(5.6, 4.8))
            plt.hist(t[mm], bins=50, alpha=0.55, label="true")
            plt.hist(p[mm], bins=50, alpha=0.55, label="pred")
            plt.xlabel(k)
            plt.ylabel("count")
            plt.title(f"hist true/pred: {k}")
            plt.legend()
            savefig(out / f"hist_true_pred_{k}.png")

    pd.DataFrame(summ_rows).to_csv(out / "true_vs_pred_morph_summary.csv", index=False)

    # 1B robust peak-sep variants
    y_true = pd.to_numeric(merged["true_peak_sep_pix"], errors="coerce").to_numpy(dtype=float)
    p_new = pd.to_numeric(merged["pred_peak_sep_pix_robust"], errors="coerce").to_numpy(dtype=float)
    p_old = pd.to_numeric(merged["pred_peak_sep_pix_old"], errors="coerce").to_numpy(dtype=float)
    f_true = pd.to_numeric(merged["true_multipeak_flag"], errors="coerce").to_numpy(dtype=float) > 0
    f_pred = pd.to_numeric(merged["pred_multipeak_flag"], errors="coerce").to_numpy(dtype=float) > 0

    var_rows = []
    # tp_only
    m_tp = f_true & f_pred & np.isfinite(y_true) & np.isfinite(p_new)
    pear, spea, n = corr_pair(y_true[m_tp], p_new[m_tp])
    rm, _ = rmse_pair(y_true[m_tp], p_new[m_tp])
    var_rows.append({"variant": "tp_only", "pearson": pear, "spearman": spea, "rmse": rm, "n": int(n)})

    # true_cond_best_effort
    best_eff = p_new.copy()
    miss = ~np.isfinite(best_eff)
    best_eff[miss] = p_old[miss]
    m_tc = f_true & np.isfinite(y_true) & np.isfinite(best_eff)
    pear, spea, n = corr_pair(y_true[m_tc], best_eff[m_tc])
    rm, _ = rmse_pair(y_true[m_tc], best_eff[m_tc])
    var_rows.append({"variant": "true_cond_best_effort", "pearson": pear, "spearman": spea, "rmse": rm, "n": int(n)})

    # true_cond_fallback_constant
    const_v = float(np.nanmedian(best_eff[np.isfinite(best_eff)])) if np.any(np.isfinite(best_eff)) else 1.0
    pred_const = np.full_like(y_true, const_v, dtype=float)
    m_c = f_true & np.isfinite(y_true)
    pear, spea, n = corr_pair(y_true[m_c], pred_const[m_c])
    rm, _ = rmse_pair(y_true[m_c], pred_const[m_c])
    var_rows.append({"variant": "true_cond_fallback_constant", "pearson": pear, "spearman": spea, "rmse": rm, "n": int(n)})

    pd.DataFrame(var_rows).to_csv(out / "peaksep_conditional_variants.csv", index=False)

    # 1C morphology redundancy (true metrics)
    red_cols = [
        "m_ellipticity_true",
        "m_concentration_r2_r6_true",
        "m_edge_flux_frac_true",
        "m_asymmetry_180_true",
        "true_multipeak_flag",
        "true_peak_sep_pix",
    ]
    red_cols = [c for c in red_cols if c in merged.columns]
    red_rows = []
    for i, a in enumerate(red_cols):
        for b in red_cols[i:]:
            x = pd.to_numeric(merged[a], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(merged[b], errors="coerce").to_numpy(dtype=float)
            pear, spea, n = corr_pair(x, y)
            red_rows.append({"metric_a": a, "metric_b": b, "pearson_r": pear, "spearman_r": spea, "n": n})
            if a != b:
                red_rows.append({"metric_a": b, "metric_b": a, "pearson_r": pear, "spearman_r": spea, "n": n})
    red_df = pd.DataFrame(red_rows)
    red_df.to_csv(out / "morph_vs_morph_corr.csv", index=False)

    # heatmap from pearson
    piv = red_df.pivot(index="metric_a", columns="metric_b", values="pearson_r")
    rr = sorted(piv.index.tolist())
    cc = sorted(piv.columns.tolist())
    arr = piv.loc[rr, cc].to_numpy(dtype=float)
    plt.figure(figsize=(1.3 * len(cc), 0.6 * len(rr)))
    im = plt.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="pearson_r")
    plt.xticks(np.arange(len(cc)), cc, rotation=35, ha="right", fontsize=8)
    plt.yticks(np.arange(len(rr)), rr, fontsize=8)
    plt.title("True morphology metric redundancy (Pearson)")
    savefig(out / "heatmap_morph_vs_morph.png")

    # Save merged analysis frame for reproducibility
    merged.to_csv(out / "audit_merged_rows.csv", index=False)
    print("DONE")
    print("out_dir:", out)


if __name__ == "__main__":
    main()
