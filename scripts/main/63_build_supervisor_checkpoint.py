#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import PowerNorm


PRED_METRICS = [
    "m_concentration_r2_r6_pred",
    "m_asymmetry_180_pred",
    "m_ellipticity_pred",
    "m_peak_sep_pix_pred",
    "m_edge_flux_frac_pred",
]


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def load_xgb_model_and_features(run_dir: Path) -> Tuple[xgb.Booster, List[str]]:
    model_path = run_dir / "model_xgb.json"
    feat_path = run_dir / "feature_cols.json"
    if not model_path.exists() or not feat_path.exists():
        raise RuntimeError(f"Missing model or feature list in {run_dir}")
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    feat_cols = json.loads(feat_path.read_text(encoding="utf-8"))
    return booster, feat_cols


def predict_probs(df: pd.DataFrame, booster: xgb.Booster, feat_cols: List[str]) -> np.ndarray:
    miss = [c for c in feat_cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"Embedding missing required features: {miss}")
    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    med = x.median(numeric_only=True)
    x = x.fillna(med)
    dm = xgb.DMatrix(x.to_numpy(dtype=np.float32), feature_names=feat_cols)
    nround = booster.best_iteration + 1 if booster.best_iteration is not None and booster.best_iteration >= 0 else 0
    if nround > 0:
        p = booster.predict(dm, iteration_range=(0, nround))
    else:
        p = booster.predict(dm)
    return np.asarray(p, dtype=np.float32)


def plot_prob_map(
    df: pd.DataFrame,
    col: str,
    title: str,
    out_png: Path,
    cmap: str | LinearSegmentedColormap = "viridis",
    color_mask: np.ndarray | None = None,
    q_low: float = 2.0,
    q_high: float = 98.0,
    gamma: float = 0.50,
    quantile_stretch: bool = True,
) -> None:
    plt.figure(figsize=(10.8, 8.6))
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if color_mask is None:
        color_mask = np.ones(len(df), dtype=bool)
    else:
        color_mask = np.asarray(color_mask, dtype=bool)
        if color_mask.shape[0] != len(df):
            raise ValueError("color_mask length must match df rows")
    base_alpha = 0.20 if np.any(~color_mask) else 0.06
    plt.scatter(df["x"], df["y"], s=4, color="#bdbdbd", alpha=base_alpha, linewidths=0)
    sc = None
    mask = color_mask & np.isfinite(vals)
    if np.any(mask):
        cvals = vals[mask]
        lo, hi = np.nanpercentile(cvals, [q_low, q_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(cvals)), float(np.nanmax(cvals))
        norm = None
        cplot = cvals
        cbar_label = col
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            cvals_clip = np.clip(cvals, lo, hi)
            if quantile_stretch:
                # Map values through empirical CDF so colors are used more uniformly.
                q = np.nanpercentile(cvals_clip, np.linspace(0, 100, 256))
                q = np.asarray(q, dtype=float)
                uq = np.unique(q)
                if uq.size >= 2:
                    y = np.linspace(0.0, 1.0, uq.size)
                    cplot = np.interp(cvals_clip, uq, y)
                    cbar_label = f"{col} (quantile-scaled)"
                else:
                    cplot = cvals_clip
                    norm = PowerNorm(gamma=gamma, vmin=lo, vmax=hi)
            else:
                cplot = cvals_clip
                norm = PowerNorm(gamma=gamma, vmin=lo, vmax=hi)
        sc = plt.scatter(
            df.loc[mask, "x"],
            df.loc[mask, "y"],
            c=cplot,
            s=4,
            cmap=cmap,
            norm=norm,
            alpha=0.82,
            linewidths=0,
        )
    if sc is not None:
        cbar = plt.colorbar(sc)
        cbar.set_label(cbar_label)
    if "is_double" in df.columns:
        d = df[df["is_double"] == True]  # noqa: E712
        if len(d):
            plt.scatter(d["x"], d["y"], s=12, facecolors="none", edgecolors="#d62828", linewidths=0.45)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    savefig(out_png)


def plot_field_map(df: pd.DataFrame, title: str, out_png: Path) -> None:
    plt.figure(figsize=(10.8, 8.6))
    tags = sorted(df["field_tag"].dropna().astype(str).unique().tolist())
    if len(tags) <= 20:
        cmap = plt.get_cmap("tab20", max(1, len(tags)))
        for i, t in enumerate(tags):
            sub = df[df["field_tag"] == t]
            plt.scatter(sub["x"], sub["y"], s=4, color=cmap(i), alpha=0.65, linewidths=0, label=t)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=7)
    else:
        top = df["field_tag"].value_counts().head(12).index.tolist()
        cmap = plt.get_cmap("tab20", max(1, len(top) + 1))
        for i, t in enumerate(top):
            sub = df[df["field_tag"] == t]
            plt.scatter(sub["x"], sub["y"], s=4, color=cmap(i), alpha=0.65, linewidths=0, label=t)
        rest = df[~df["field_tag"].isin(top)]
        plt.scatter(rest["x"], rest["y"], s=4, color="#bdbdbd", alpha=0.24, linewidths=0, label="other fields")
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=7)
    if "is_double" in df.columns:
        d = df[df["is_double"] == True]  # noqa: E712
        if len(d):
            plt.scatter(d["x"], d["y"], s=12, facecolors="none", edgecolors="black", linewidths=0.45)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    savefig(out_png)


def plot_numeric_map(
    df: pd.DataFrame,
    col: str,
    title: str,
    out_png: Path,
    cmap: str | LinearSegmentedColormap = "viridis",
    q_low: float = 2.0,
    q_high: float = 98.0,
) -> None:
    plt.figure(figsize=(10.8, 8.6))
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(vals)
    plt.scatter(df["x"], df["y"], s=4, color="#d0d0d0", alpha=0.12, linewidths=0)
    sc = None
    if np.any(m):
        lo, hi = np.nanpercentile(vals[m], [q_low, q_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(vals[m])), float(np.nanmax(vals[m]))
        cvals = vals[m]
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            cvals = np.clip(cvals, lo, hi)
        sc = plt.scatter(df.loc[m, "x"], df.loc[m, "y"], c=cvals, s=4, cmap=cmap, alpha=0.80, linewidths=0)
    if sc is not None:
        cbar = plt.colorbar(sc)
        cbar.set_label(col)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    savefig(out_png)


def plot_doubles_overlay(df: pd.DataFrame, title: str, out_png: Path) -> None:
    plt.figure(figsize=(10.8, 8.6))
    plt.scatter(df["x"], df["y"], s=4, color="#4f5d75", alpha=0.30, linewidths=0)
    if "is_double" in df.columns:
        d = df[df["is_double"] == True]  # noqa: E712
        if len(d):
            plt.scatter(d["x"], d["y"], s=16, facecolors="none", edgecolors="#d62828", linewidths=0.55)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    savefig(out_png)


def plot_pred_morph_panels(df: pd.DataFrame, prefix: str, out_png: Path) -> None:
    cols = [
        "m_concentration_r2_r6_pred",
        "m_asymmetry_180_pred",
        "m_ellipticity_pred",
        "m_peak_sep_pix_pred",
        "m_edge_flux_frac_pred",
    ]
    cols = [c for c in cols if c in df.columns]
    n = len(cols)
    if n == 0:
        return
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.2, 5.1 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for i in range(nrows * ncols):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off")
            continue
        c = cols[i]
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(vals)
        if np.any(m):
            lo, hi = np.nanpercentile(vals[m], [1, 99])
            vals = np.clip(vals, lo, hi)
        sc = ax.scatter(df["x"], df["y"], c=vals, s=4, cmap="viridis", alpha=0.70, linewidths=0)
        if "is_double" in df.columns:
            d = df[df["is_double"] == True]  # noqa: E712
            if len(d):
                ax.scatter(d["x"], d["y"], s=11, facecolors="none", edgecolors="#d62828", linewidths=0.4)
        ax.set_title(c)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(c)
    fig.suptitle(f"{prefix} UMAP colored by predicted morphology metrics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig(out_png)


def write_manuscript(
    out_md: Path,
    summary_rows: pd.DataFrame,
    coverage_rows: pd.DataFrame,
    outputs_root: Path,
    cfg: Dict[str, str],
) -> None:
    def _df_block(df: pd.DataFrame) -> str:
        return "```\n" + df.to_string(index=False) + "\n```"

    lines: List[str] = []
    lines.append("# Supervisor Checkpoint Report")
    lines.append("")
    lines.append("## Objective")
    lines.append("Provide a complete, reproducible status checkpoint of the Gaia->Euclid morphology pipeline,")
    lines.append("including training setup, classifier performance, UMAP construction choices, and supporting figures.")
    lines.append("")
    lines.append("## What Is Trained")
    lines.append("- Reconstruction model (`base_v16_clean_final_manualv8`) trained on all quality-filtered objects (not non-PSF-only).")
    lines.append("- Gaia-first classifier 1: `true_multipeak_flag` (XGBoost).")
    lines.append("- Gaia-first classifier 2: `true_nonpsf_flag` (XGBoost).")
    lines.append("")
    lines.append("## Why")
    lines.append("- Multi-peak from reconstructed stamps under-recovers recall due to smoothness; Gaia-first classification is stronger.")
    lines.append("- Non-PSF classifier extends the scope beyond doubles/blends to broader morphology.")
    lines.append("")
    lines.append("## UMAPs Included")
    lines.append("- 16D Gaia UMAP (`umap16d_manualv8_filtered`): field-colored, WDS doubles overlays, and full-coverage classifier probability maps.")
    lines.append("- Pixel-stamp UMAP (`pixels_umap_standard_manualv8_filtered`): field-colored, WDS doubles overlays, and projected classifier probability maps.")
    lines.append("- Predicted-morphology overlays (`*_pred`) are included where prediction rows exist.")
    lines.append("")
    lines.append("## Core Performance Summary")
    lines.append(_df_block(summary_rows))
    lines.append("")
    lines.append("## Coverage/UMAP Join Summary")
    lines.append(_df_block(coverage_rows))
    lines.append("")
    lines.append("## Key Output Folders")
    lines.append(f"- Unified checkpoint: `{outputs_root}`")
    lines.append(f"- 16D panel folder: `{outputs_root / 'umap_16d'}`")
    lines.append(f"- Pixel panel folder: `{outputs_root / 'umap_pixels'}`")
    lines.append("")
    lines.append("## Reproducibility Inputs")
    lines.append(f"- multipeak model dir: `{cfg['multi_run_dir']}`")
    lines.append(f"- nonpsf model dir: `{cfg['nonpsf_run_dir']}`")
    lines.append(f"- 16D embedding: `{cfg['embedding16_csv']}`")
    lines.append(f"- pixel embedding: `{cfg['embeddingpix_csv']}`")
    lines.append(f"- predicted morph table: `{cfg['pred_morph_csv']}`")
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("- WDS doubles are sparse in these filtered UMAP samples; overlays are for localization, not standalone model validation.")
    lines.append("- Predicted morphology is diagnostic/secondary; primary thesis evidence is Gaia-first classifier skill.")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--embedding16_csv",
        default="/data/yn316/Codes/output/experiments/embeddings/umap16d_manualv8_filtered/embedding_umap.csv",
    )
    ap.add_argument(
        "--embeddingpix_csv",
        default="/data/yn316/Codes/output/experiments/embeddings/double_stars_pixels/pixels_umap_standard_manualv8_filtered/umap_standard/embedding_umap.csv",
    )
    ap.add_argument(
        "--multi_run_dir",
        default="/data/yn316/Codes/report/model_decision/20260302_gaia_first_multipeak_xgb_base_v16_clean_final_manualv8",
    )
    ap.add_argument(
        "--nonpsf_run_dir",
        default="/data/yn316/Codes/report/model_decision/20260302_gaia_first_nonpsf_xgb_base_v16_clean_final_manualv8",
    )
    ap.add_argument(
        "--pred_morph_csv",
        default="/data/yn316/Codes/report/model_decision/20260227_morph_sanity_base_v16_clean_final_manualv8/morph_pred_vs_true_rows.csv",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    umap16_dir = out_dir / "umap_16d"
    umappix_dir = out_dir / "umap_pixels"
    umap16_test_dir = out_dir / "umap_16d_testsplit"
    umappix_test_dir = out_dir / "umap_pixels_testsplit"
    umap16_dir.mkdir(parents=True, exist_ok=True)
    umappix_dir.mkdir(parents=True, exist_ok=True)
    umap16_test_dir.mkdir(parents=True, exist_ok=True)
    umappix_test_dir.mkdir(parents=True, exist_ok=True)

    emb16 = pd.read_csv(Path(args.embedding16_csv), low_memory=False)
    embpix = pd.read_csv(Path(args.embeddingpix_csv), low_memory=False)
    for df in [emb16, embpix]:
        for c in ["source_id", "x", "y"]:
            df[c] = _safe_num(df[c])
        df.dropna(subset=["source_id", "x", "y"], inplace=True)
        df["source_id"] = df["source_id"].astype(np.int64)
        if "is_double" in df.columns:
            df["is_double"] = df["is_double"].astype(bool)
        df.drop_duplicates(subset=["source_id"], keep="first", inplace=True)
        df.reset_index(drop=True, inplace=True)

    booster_multi, feat_multi = load_xgb_model_and_features(Path(args.multi_run_dir))
    booster_nonpsf, feat_nonpsf = load_xgb_model_and_features(Path(args.nonpsf_run_dir))
    emb16["p_multipeak_xgb"] = predict_probs(emb16, booster_multi, feat_multi)
    emb16["p_nonpsf_xgb"] = predict_probs(emb16, booster_nonpsf, feat_nonpsf)

    embpix = embpix.merge(
        emb16[["source_id", "p_multipeak_xgb", "p_nonpsf_xgb"]],
        on="source_id",
        how="left",
    )

    pm = pd.read_csv(Path(args.pred_morph_csv), low_memory=False)
    pm_cols = [c for c in PRED_METRICS if c in pm.columns]
    pm["source_id"] = _safe_num(pm["source_id"])
    pm = pm.dropna(subset=["source_id"]).copy()
    pm["source_id"] = pm["source_id"].astype(np.int64)
    pm = pm.drop_duplicates(subset=["source_id"], keep="first")
    emb16 = emb16.merge(pm[["source_id", *pm_cols]], on="source_id", how="left")
    embpix = embpix.merge(pm[["source_id", *pm_cols]], on="source_id", how="left")
    test_source: set[int] = set()
    test_pred_csv = Path(args.multi_run_dir) / "test_predictions.csv"
    if test_pred_csv.exists():
        tp = pd.read_csv(test_pred_csv, usecols=["source_id"], low_memory=False)
        tp["source_id"] = _safe_num(tp["source_id"])
        tp = tp.dropna(subset=["source_id"]).copy()
        tp["source_id"] = tp["source_id"].astype(np.int64)
        test_source = set(tp["source_id"].tolist())
    if not test_source:
        # Fallback to prediction/morph table ids when test predictions are unavailable.
        test_source = set(pm["source_id"].astype(np.int64).tolist())
    emb16_test = emb16[emb16["source_id"].isin(test_source)].copy()
    embpix_test = embpix[embpix["source_id"].isin(test_source)].copy()

    cmap_nonpsf = LinearSegmentedColormap.from_list(
        "nonpsf_gbp",
        ["#edf8e9", "#66c2a4", "#2f6bff", "#4b0082"],
    )
    cmap_multipeak = cmap_nonpsf

    # 16D plots
    plot_prob_map(
        emb16,
        "p_multipeak_xgb",
        "16D UMAP colored by p(multipeak)",
        umap16_dir / "01_umap16_p_multipeak.png",
        cmap=cmap_multipeak,
    )
    plot_prob_map(
        emb16,
        "p_nonpsf_xgb",
        "16D UMAP colored by p(nonpsf)",
        umap16_dir / "02_umap16_p_nonpsf.png",
        cmap=cmap_nonpsf,
    )
    plot_field_map(emb16, "16D UMAP colored by field", umap16_dir / "03_umap16_field.png")
    plot_doubles_overlay(emb16, "16D UMAP with WDS doubles overlay", umap16_dir / "04_umap16_wds_doubles.png")
    plot_pred_morph_panels(emb16, "16D", umap16_dir / "05_umap16_pred_morph_panels.png")
    nuisance_col_16d = "phot_g_mean_mag" if "phot_g_mean_mag" in emb16.columns else ("feat_log10_snr" if "feat_log10_snr" in emb16.columns else "")
    if nuisance_col_16d:
        plot_numeric_map(
            emb16,
            nuisance_col_16d,
            f"16D UMAP colored by nuisance: {nuisance_col_16d}",
            umap16_dir / "06_umap16_nuisance.png",
            cmap="cividis",
        )

    # Pixel plots
    plot_prob_map(
        embpix,
        "p_multipeak_xgb",
        "Pixel UMAP colored by p(multipeak)",
        umappix_dir / "01_umappix_p_multipeak.png",
        cmap=cmap_multipeak,
    )
    plot_prob_map(
        embpix,
        "p_nonpsf_xgb",
        "Pixel UMAP colored by p(nonpsf)",
        umappix_dir / "02_umappix_p_nonpsf.png",
        cmap=cmap_nonpsf,
    )
    plot_field_map(embpix, "Pixel UMAP colored by field", umappix_dir / "03_umappix_field.png")
    plot_doubles_overlay(embpix, "Pixel UMAP with WDS doubles overlay", umappix_dir / "04_umappix_wds_doubles.png")
    plot_pred_morph_panels(embpix, "Pixel", umappix_dir / "05_umappix_pred_morph_panels.png")

    # Test-split-only views (source_id present in pred_morph table).
    if len(emb16_test):
        plot_prob_map(
            emb16_test,
            "p_multipeak_xgb",
            "16D UMAP (test-split) p(multipeak)",
            umap16_test_dir / "01_umap16_test_p_multipeak.png",
            cmap=cmap_multipeak,
        )
        plot_prob_map(
            emb16_test,
            "p_nonpsf_xgb",
            "16D UMAP (test-split) p(nonpsf)",
            umap16_test_dir / "02_umap16_test_p_nonpsf.png",
            cmap=cmap_nonpsf,
        )
        plot_field_map(emb16_test, "16D UMAP (test-split) field", umap16_test_dir / "03_umap16_test_field.png")
        plot_doubles_overlay(emb16_test, "16D UMAP (test-split) WDS doubles", umap16_test_dir / "04_umap16_test_wds_doubles.png")
        plot_pred_morph_panels(emb16_test, "16D (test-split)", umap16_test_dir / "05_umap16_test_pred_morph_panels.png")
        if nuisance_col_16d:
            plot_numeric_map(
                emb16_test,
                nuisance_col_16d,
                f"16D UMAP (test-split) nuisance: {nuisance_col_16d}",
                umap16_test_dir / "06_umap16_test_nuisance.png",
                cmap="cividis",
            )
    if len(embpix_test):
        plot_prob_map(
            embpix_test,
            "p_multipeak_xgb",
            "Pixel UMAP (test-split) p(multipeak)",
            umappix_test_dir / "01_umappix_test_p_multipeak.png",
            cmap=cmap_multipeak,
        )
        plot_prob_map(
            embpix_test,
            "p_nonpsf_xgb",
            "Pixel UMAP (test-split) p(nonpsf)",
            umappix_test_dir / "02_umappix_test_p_nonpsf.png",
            cmap=cmap_nonpsf,
        )
        plot_field_map(embpix_test, "Pixel UMAP (test-split) field", umappix_test_dir / "03_umappix_test_field.png")
        plot_doubles_overlay(embpix_test, "Pixel UMAP (test-split) WDS doubles", umappix_test_dir / "04_umappix_test_wds_doubles.png")
        plot_pred_morph_panels(embpix_test, "Pixel (test-split)", umappix_test_dir / "05_umappix_test_pred_morph_panels.png")

    # Tables
    comp_multi = pd.read_csv(Path(args.multi_run_dir).parent / "20260302_gaia_multipeak_thesis_report" / "comparison_table.csv")
    nonpsf_metrics = json.loads((Path(args.nonpsf_run_dir) / "metrics_summary.json").read_text(encoding="utf-8"))
    summary_rows = pd.DataFrame(
        [
            {
                "task": "multipeak_filtered",
                "base_rate": float(comp_multi.loc[comp_multi["dataset"] == "filtered", "base_rate"].iloc[0]),
                "auroc": float(comp_multi.loc[comp_multi["dataset"] == "filtered", "auroc"].iloc[0]),
                "auprc": float(comp_multi.loc[comp_multi["dataset"] == "filtered", "auprc"].iloc[0]),
            },
            {
                "task": "multipeak_expanded",
                "base_rate": float(comp_multi.loc[comp_multi["dataset"] == "expanded", "base_rate"].iloc[0]),
                "auroc": float(comp_multi.loc[comp_multi["dataset"] == "expanded", "auroc"].iloc[0]),
                "auprc": float(comp_multi.loc[comp_multi["dataset"] == "expanded", "auprc"].iloc[0]),
            },
            {
                "task": "nonpsf_filtered",
                "base_rate": float(nonpsf_metrics["class_balance"]["test_base_rate"]),
                "auroc": float(nonpsf_metrics["xgb"]["test_auroc"]),
                "auprc": float(nonpsf_metrics["xgb"]["test_auprc"]),
            },
        ]
    )
    summary_rows.to_csv(out_dir / "model_performance_summary.csv", index=False)

    cov16 = float(np.mean(np.isfinite(pd.to_numeric(emb16.get("m_ellipticity_pred", np.nan), errors="coerce"))))
    covpix = float(np.mean(np.isfinite(pd.to_numeric(embpix.get("m_ellipticity_pred", np.nan), errors="coerce"))))
    coverage_rows = pd.DataFrame(
        [
            {
                "umap_type": "16d",
                "n_points": int(len(emb16)),
                "n_wds_doubles": int(np.sum(emb16["is_double"].to_numpy(dtype=bool))) if "is_double" in emb16.columns else 0,
                "pred_morph_coverage": cov16,
                "p_multipeak_mean": float(np.nanmean(emb16["p_multipeak_xgb"].to_numpy(dtype=float))),
                "p_nonpsf_mean": float(np.nanmean(emb16["p_nonpsf_xgb"].to_numpy(dtype=float))),
            },
            {
                "umap_type": "pixel",
                "n_points": int(len(embpix)),
                "n_wds_doubles": int(np.sum(embpix["is_double"].to_numpy(dtype=bool))) if "is_double" in embpix.columns else 0,
                "pred_morph_coverage": covpix,
                "p_multipeak_mean": float(np.nanmean(pd.to_numeric(embpix["p_multipeak_xgb"], errors="coerce"))),
                "p_nonpsf_mean": float(np.nanmean(pd.to_numeric(embpix["p_nonpsf_xgb"], errors="coerce"))),
            },
        ]
    )
    coverage_rows.to_csv(out_dir / "umap_coverage_summary.csv", index=False)

    emb16.to_csv(out_dir / "umap16_with_probs_and_predmorph.csv", index=False)
    embpix.to_csv(out_dir / "umappix_with_probs_and_predmorph.csv", index=False)
    emb16_test.to_csv(out_dir / "umap16_testsplit_with_probs_and_predmorph.csv", index=False)
    embpix_test.to_csv(out_dir / "umappix_testsplit_with_probs_and_predmorph.csv", index=False)

    cfg = {
        "multi_run_dir": str(args.multi_run_dir),
        "nonpsf_run_dir": str(args.nonpsf_run_dir),
        "embedding16_csv": str(args.embedding16_csv),
        "embeddingpix_csv": str(args.embeddingpix_csv),
        "pred_morph_csv": str(args.pred_morph_csv),
    }
    write_manuscript(
        out_md=out_dir / "MANUSCRIPT_REPORT.md",
        summary_rows=summary_rows,
        coverage_rows=coverage_rows,
        outputs_root=out_dir,
        cfg=cfg,
    )

    summary = {
        "out_dir": str(out_dir),
        "plots_16d_dir": str(umap16_dir),
        "plots_pixel_dir": str(umappix_dir),
        "tables": [
            str(out_dir / "model_performance_summary.csv"),
            str(out_dir / "umap_coverage_summary.csv"),
            str(out_dir / "umap16_with_probs_and_predmorph.csv"),
            str(out_dir / "umappix_with_probs_and_predmorph.csv"),
            str(out_dir / "umap16_testsplit_with_probs_and_predmorph.csv"),
            str(out_dir / "umappix_testsplit_with_probs_and_predmorph.csv"),
        ],
        "manuscript_report": str(out_dir / "MANUSCRIPT_REPORT.md"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
