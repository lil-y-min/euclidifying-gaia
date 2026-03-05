#!/usr/bin/env python3
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
from sklearn.cluster import MiniBatchKMeans


def load_manifest(run_root: Path) -> dict:
    d = np.load(run_root / "manifest_arrays.npz", allow_pickle=True)
    out = {}
    for k in d.files:
        v = d[k]
        if np.ndim(v) == 0:
            v = v.item()
        out[k] = v
    return out


def open_test_arrays(manifest: dict):
    n_test = int(manifest["n_test"])
    n_feat = int(manifest["n_features"])
    D = int(manifest["D"])
    stamp_pix = int(manifest["stamp_pix"])
    X_test = np.memmap(Path(str(manifest["X_test_path"])), dtype="float32", mode="r", shape=(n_test, n_feat))
    Yshape_test = np.memmap(Path(str(manifest["Yshape_test_path"])), dtype="float32", mode="r", shape=(n_test, D))
    Yflux_test = np.memmap(Path(str(manifest["Yflux_test_path"])), dtype="float32", mode="r", shape=(n_test,))
    return X_test, Yshape_test, Yflux_test, stamp_pix


def build_mag_map(dataset_root: Path, cache_csv: Path) -> pd.DataFrame:
    if cache_csv.exists():
        return pd.read_csv(cache_csv)
    rows: List[pd.DataFrame] = []
    for p in sorted(dataset_root.glob("*/metadata_16d.csv")):
        try:
            d = pd.read_csv(p, usecols=["source_id", "phot_g_mean_mag"], low_memory=False)
            rows.append(d)
        except Exception:
            continue
    if not rows:
        out = pd.DataFrame(columns=["source_id", "phot_g_mean_mag"])
    else:
        out = pd.concat(rows, ignore_index=True)
        out["source_id"] = pd.to_numeric(out["source_id"], errors="coerce")
        out = out.dropna(subset=["source_id"]).copy()
        out["source_id"] = out["source_id"].astype(np.int64)
        out = out.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    out.to_csv(cache_csv, index=False)
    return out


def save_large_montage(I: np.ndarray, idx: np.ndarray, title: str, out_png: Path, n_show: int, n_cols: int) -> None:
    n = min(int(n_show), len(idx))
    if n <= 0:
        return
    cols = max(1, int(n_cols))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n:
            continue
        img = I[int(idx[i])]
        lo = np.nanpercentile(img, 1.0)
        hi = np.nanpercentile(img, 99.0)
        ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _safe_percentile(a: np.ndarray, q: float) -> float:
    v = a[np.isfinite(a)]
    if v.size == 0:
        return 0.0
    return float(np.percentile(v, q))


def extract_morph(stamp: np.ndarray) -> Dict[str, float]:
    img = np.array(stamp, dtype=np.float64)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    bg = _safe_percentile(img, 10.0)
    ip = img - bg
    ip[ip < 0] = 0.0
    h, w = ip.shape
    yy, xx = np.indices((h, w))
    flux_sum = float(np.sum(ip))
    if flux_sum <= 0:
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        ell = 0.0
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
        "m_ellipticity": ell,
        "m_concentration_r2_r6": concentration,
        "m_edge_flux_frac": edge_flux_frac,
        "m_asymmetry_180": asym,
    }


def draw_cluster_ellipse(ax, x: np.ndarray, y: np.ndarray, color: str, lw: float = 2.0):
    if len(x) < 6:
        return
    mx, my = float(np.mean(x)), float(np.mean(y))
    cov = np.cov(np.vstack([x, y]))
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-8)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w = 2.8 * np.sqrt(vals[0])
    h = 2.8 * np.sqrt(vals[1])
    from matplotlib.patches import Ellipse

    ell = Ellipse((mx, my), width=w, height=h, angle=theta, fill=False, ec=color, lw=lw, alpha=0.95)
    ax.add_patch(ell)


def select_roi_clusters(cluster_stats: pd.DataFrame, n_roi_each: int) -> pd.DataFrame:
    valid = cluster_stats[cluster_stats["n"] >= 120].copy()
    if valid.empty:
        valid = cluster_stats.copy()
    out = []
    used: set[int] = set()
    for tag, col, asc in [
        ("high_prob", "mean_p_xgb", False),
        ("high_fp_rate", "fp_rate", False),
        ("high_fn_rate", "fn_rate", False),
    ]:
        ranked = valid.sort_values(col, ascending=asc).copy()
        picked = []
        for _, r in ranked.iterrows():
            cid = int(r["cluster_id"])
            if cid in used:
                continue
            picked.append(r)
            used.add(cid)
            if len(picked) >= int(n_roi_each):
                break
        if len(picked) < int(n_roi_each):
            # Backfill if unique pool is exhausted.
            for _, r in ranked.iterrows():
                if len(picked) >= int(n_roi_each):
                    break
                cid = int(r["cluster_id"])
                if any(int(p["cluster_id"]) == cid for p in picked):
                    continue
                picked.append(r)
        sub = pd.DataFrame(picked).copy()
        sub["roi_group"] = tag
        out.append(sub)
    out_df = pd.concat(out, ignore_index=True)
    out_df = out_df.drop_duplicates(subset=["cluster_id", "roi_group"]).reset_index(drop=True)
    return out_df


def select_morph_roi_clusters(cluster_stats: pd.DataFrame, n_roi_each: int) -> pd.DataFrame:
    valid = cluster_stats[cluster_stats["n"] >= 120].copy()
    if valid.empty:
        valid = cluster_stats.copy()
    out = []
    used: set[int] = set()
    for tag, col, asc in [
        ("high_ellipticity", "mean_m_ellipticity", False),
        ("high_edge_flux", "mean_m_edge_flux_frac", False),
        ("high_asymmetry", "mean_m_asymmetry_180", False),
        ("low_concentration", "mean_m_concentration_r2_r6", True),
    ]:
        ranked = valid.sort_values(col, ascending=asc).copy()
        picked = []
        for _, r in ranked.iterrows():
            cid = int(r["cluster_id"])
            if cid in used:
                continue
            picked.append(r)
            used.add(cid)
            if len(picked) >= int(n_roi_each):
                break
        if len(picked) < int(n_roi_each):
            for _, r in ranked.iterrows():
                if len(picked) >= int(n_roi_each):
                    break
                cid = int(r["cluster_id"])
                if any(int(p["cluster_id"]) == cid for p in picked):
                    continue
                picked.append(r)
        sub = pd.DataFrame(picked).copy()
        sub["roi_group"] = tag
        out.append(sub)
    out_df = pd.concat(out, ignore_index=True)
    out_df = out_df.drop_duplicates(subset=["cluster_id", "roi_group"]).reset_index(drop=True)
    return out_df


def plot_full_umap_metric(df: pd.DataFrame, color_col: str, cbar_label: str, title: str, out_png: Path) -> None:
    plt.figure(figsize=(10.5, 8.6))
    sc = plt.scatter(df["umap1"], df["umap2"], c=df[color_col], s=4, cmap="viridis", alpha=0.65, linewidths=0)
    cb = plt.colorbar(sc)
    cb.set_label(cbar_label)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_full_umap_with_rois(df: pd.DataFrame, roi_df: pd.DataFrame, color_col: str, cbar_label: str, title: str, out_png: Path) -> None:
    plt.figure(figsize=(10.5, 8.6))
    sc = plt.scatter(df["umap1"], df["umap2"], c=df[color_col], s=4, cmap="viridis", alpha=0.65, linewidths=0)
    cb = plt.colorbar(sc)
    cb.set_label(cbar_label)
    colors = [
        "#D32F2F",
        "#1976D2",
        "#F57C00",
        "#388E3C",
        "#7B1FA2",
        "#0097A7",
        "#5D4037",
        "#C2185B",
    ]
    group_colors = {g: colors[i % len(colors)] for i, g in enumerate(sorted(roi_df["roi_group"].unique()))}
    for _, r in roi_df.iterrows():
        cid = int(r["cluster_id"])
        grp = str(r["roi_group"])
        nm = str(r["roi_name"])
        sub = df[df["cluster_id"] == cid]
        draw_cluster_ellipse(plt.gca(), sub["umap1"].to_numpy(), sub["umap2"].to_numpy(), color=group_colors.get(grp, "#111111"), lw=2.0)
        plt.text(float(r["mean_umap1"]), float(r["mean_umap2"]), nm, fontsize=8, color=group_colors.get(grp, "#111111"))
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--science_products_dir", required=True, help="Output dir from script 55.")
    ap.add_argument("--pred_run_dir", required=True, help="Output dir from script 53 (predictions + operating points).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_clusters", type=int, default=18)
    ap.add_argument("--n_roi_each", type=int, default=4)
    ap.add_argument("--n_samples_per_roi", type=int, default=80)
    ap.add_argument("--montage_cols", type=int, default=8)
    ap.add_argument("--balanced_op_tag", default="precision_floor")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path("/data/yn316/Codes")
    run_root = base / "output" / "ml_runs" / str(args.run_name)
    sci_dir = Path(args.science_products_dir)
    pred_dir = Path(args.pred_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "roi_samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "roi_montages").mkdir(parents=True, exist_ok=True)

    emb = pd.read_csv(sci_dir / "umap_embedding.csv")
    pred = pd.read_csv(pred_dir / "test_predictions.csv")
    ops = pd.read_csv(pred_dir / "operating_points.csv")
    trace = pd.read_csv(run_root / "trace" / "trace_test.csv", usecols=["test_index", "source_id"], low_memory=False).sort_values("test_index")

    bal_row = ops[ops["op_tag"] == str(args.balanced_op_tag)]
    if bal_row.empty:
        bal_row = ops.iloc[[0]]
    thr_bal = float(bal_row.iloc[0]["threshold"])

    df = emb.merge(pred[["test_index", "p_xgb", "y_true"]], on="test_index", how="inner")
    df = df.merge(trace, on="test_index", how="left")
    df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
    df = df.dropna(subset=["source_id"]).copy()
    df["source_id"] = df["source_id"].astype(np.int64)

    manifest = load_manifest(run_root)
    feature_cols = list(manifest["feature_cols"])
    idx_ruwe = feature_cols.index("feat_ruwe") if "feat_ruwe" in feature_cols else None
    idx_ipd = feature_cols.index("feat_ipd_frac_multi_peak") if "feat_ipd_frac_multi_peak" in feature_cols else None
    X_test, Yshape_test, Yflux_test, stamp_pix = open_test_arrays(manifest)
    X = np.asarray(X_test, dtype=np.float32)
    if idx_ruwe is not None:
        df["feat_ruwe"] = X[df["test_index"].to_numpy(dtype=int), idx_ruwe]
    else:
        df["feat_ruwe"] = np.nan
    if idx_ipd is not None:
        df["feat_ipd_frac_multi_peak"] = X[df["test_index"].to_numpy(dtype=int), idx_ipd]
    else:
        df["feat_ipd_frac_multi_peak"] = np.nan

    mag_map = build_mag_map(Path(str(manifest["dataset_root"])), out_dir / "source_mag_cache.csv")
    mag_map["source_id"] = pd.to_numeric(mag_map["source_id"], errors="coerce")
    mag_map = mag_map.dropna(subset=["source_id"]).copy()
    mag_map["source_id"] = mag_map["source_id"].astype(np.int64)
    df = df.merge(mag_map, on="source_id", how="left")

    # KMeans regions on full UMAP.
    kmeans = MiniBatchKMeans(n_clusters=int(args.n_clusters), random_state=int(args.seed), batch_size=4096, n_init=10)
    labels = kmeans.fit_predict(df[["umap1", "umap2"]].to_numpy(dtype=float))
    df["cluster_id"] = labels.astype(int)

    y_true = df["y_true"].to_numpy(dtype=bool)
    y_hat = df["p_xgb"].to_numpy(dtype=float) >= thr_bal
    df["y_hat_bal"] = y_hat.astype(int)
    df["is_fp"] = ((~y_true) & y_hat).astype(int)
    df["is_fn"] = (y_true & (~y_hat)).astype(int)

    I_true = (np.asarray(Yshape_test, dtype=np.float32) * (10.0 ** np.asarray(Yflux_test, dtype=np.float32)[:, None])).reshape((-1, stamp_pix, stamp_pix))
    morph_rows = []
    for ti in df["test_index"].to_numpy(dtype=int):
        morph_rows.append(extract_morph(I_true[ti]))
    morph_df = pd.DataFrame(morph_rows)
    df = pd.concat([df.reset_index(drop=True), morph_df.reset_index(drop=True)], axis=1)

    cstats = (
        df.groupby("cluster_id", as_index=False)
        .agg(
            n=("test_index", "size"),
            mean_umap1=("umap1", "mean"),
            mean_umap2=("umap2", "mean"),
            mean_p_xgb=("p_xgb", "mean"),
            true_rate=("y_true", "mean"),
            pred_rate=("y_hat_bal", "mean"),
            fp_rate=("is_fp", "mean"),
            fn_rate=("is_fn", "mean"),
            mean_ruwe=("feat_ruwe", "mean"),
            mean_ipd_frac_multi_peak=("feat_ipd_frac_multi_peak", "mean"),
            mean_phot_g_mean_mag=("phot_g_mean_mag", "mean"),
            mean_m_ellipticity=("m_ellipticity", "mean"),
            mean_m_concentration_r2_r6=("m_concentration_r2_r6", "mean"),
            mean_m_edge_flux_frac=("m_edge_flux_frac", "mean"),
            mean_m_asymmetry_180=("m_asymmetry_180", "mean"),
        )
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    cstats.to_csv(out_dir / "umap_cluster_stats.csv", index=False)

    # Select candidate ROI clusters.
    roi_df = select_roi_clusters(cstats, n_roi_each=int(args.n_roi_each))
    # stable ROI naming
    rn = []
    for g in ["high_prob", "high_fp_rate", "high_fn_rate"]:
        sub = roi_df[roi_df["roi_group"] == g].copy().sort_values("cluster_id")
        for i, cid in enumerate(sub["cluster_id"].tolist(), start=1):
            rn.append((g, cid, f"{g}_{i}"))
    name_map = {(g, cid): n for g, cid, n in rn}
    roi_df["roi_name"] = [name_map[(g, int(c))] for g, c in zip(roi_df["roi_group"], roi_df["cluster_id"])]
    roi_df.to_csv(out_dir / "roi_clusters_selected.csv", index=False)

    # Morphology-driven ROI selection and plots.
    morph_roi_df = select_morph_roi_clusters(cstats, n_roi_each=int(args.n_roi_each))
    rn2 = []
    for g in ["high_ellipticity", "high_edge_flux", "high_asymmetry", "low_concentration"]:
        sub = morph_roi_df[morph_roi_df["roi_group"] == g].copy().sort_values("cluster_id")
        for i, cid in enumerate(sub["cluster_id"].tolist(), start=1):
            rn2.append((g, cid, f"{g}_{i}"))
    name_map2 = {(g, cid): n for g, cid, n in rn2}
    morph_roi_df["roi_name"] = [name_map2[(g, int(c))] for g, c in zip(morph_roi_df["roi_group"], morph_roi_df["cluster_id"])]
    morph_roi_df.to_csv(out_dir / "roi_clusters_morph_selected.csv", index=False)

    # Full UMAP metric maps.
    plot_full_umap_metric(df, "p_xgb", "p(multipeak)", "Full UMAP colored by p(multipeak)", out_dir / "umap_full_prob.png")
    plot_full_umap_metric(df, "phot_g_mean_mag", "phot_g_mean_mag", "Full UMAP colored by magnitude", out_dir / "umap_full_mag.png")
    plot_full_umap_metric(df, "m_ellipticity", "ellipticity", "Full UMAP colored by Euclid ellipticity", out_dir / "umap_full_morph_ellipticity.png")
    plot_full_umap_metric(df, "m_concentration_r2_r6", "concentration r2/r6", "Full UMAP colored by Euclid concentration", out_dir / "umap_full_morph_concentration.png")
    plot_full_umap_metric(df, "m_edge_flux_frac", "edge flux fraction", "Full UMAP colored by Euclid edge flux fraction", out_dir / "umap_full_morph_edge_flux.png")
    plot_full_umap_metric(df, "m_asymmetry_180", "asymmetry 180", "Full UMAP colored by Euclid asymmetry", out_dir / "umap_full_morph_asymmetry.png")

    # Overlays for both ROI sets.
    plot_full_umap_with_rois(
        df=df,
        roi_df=roi_df,
        color_col="p_xgb",
        cbar_label="p(multipeak)",
        title="Full UMAP with error/prob ROI clusters",
        out_png=out_dir / "umap_full_with_rois_prob.png",
    )
    plot_full_umap_with_rois(
        df=df,
        roi_df=roi_df,
        color_col="phot_g_mean_mag",
        cbar_label="phot_g_mean_mag",
        title="Full UMAP with error/prob ROIs (magnitude nuisance)",
        out_png=out_dir / "umap_full_with_rois_mag.png",
    )
    plot_full_umap_with_rois(
        df=df,
        roi_df=morph_roi_df,
        color_col="m_ellipticity",
        cbar_label="ellipticity",
        title="Full UMAP with morphology-driven ROI clusters",
        out_png=out_dir / "umap_full_with_rois_morph.png",
    )

    # Export ROI samples and larger stamp montages.
    rng = np.random.default_rng(int(args.seed))
    rows_lab = []
    both_roi = pd.concat([roi_df.assign(roi_set="error_prob"), morph_roi_df.assign(roi_set="morphology")], ignore_index=True)
    for _, r in both_roi.iterrows():
        cid = int(r["cluster_id"])
        roi_name = str(r["roi_name"])
        grp = str(r["roi_group"])
        roi_set = str(r["roi_set"])
        sub = df[df["cluster_id"] == cid].copy()
        n_take = min(int(args.n_samples_per_roi), len(sub))
        chosen = sub.sample(n=n_take, random_state=int(args.seed)).sort_values("p_xgb", ascending=False)
        chosen.to_csv(out_dir / "roi_samples" / f"{roi_set}_{roi_name}_samples.csv", index=False)
        idx = chosen["test_index"].to_numpy(dtype=int)
        save_large_montage(
            I=I_true,
            idx=idx,
            title=f"{roi_set}:{roi_name} ({grp})  n={len(sub)}",
            out_png=out_dir / "roi_montages" / f"{roi_set}_{roi_name}_montage.png",
            n_show=n_take,
            n_cols=int(args.montage_cols),
        )
        for sid in chosen["source_id"].tolist():
            rows_lab.append({"source_id": sid, "roi_set": roi_set, "roi_name": roi_name, "roi_group": grp})

    pd.DataFrame(rows_lab).drop_duplicates().to_csv(out_dir / "roi_labeling_seed_list.csv", index=False)

    summary = {
        "run_name": str(args.run_name),
        "balanced_threshold": thr_bal,
        "n_clusters": int(args.n_clusters),
        "n_roi_each": int(args.n_roi_each),
        "roi_groups": ["high_prob", "high_fp_rate", "high_fn_rate"],
        "outputs": {
            "cluster_stats_csv": str(out_dir / "umap_cluster_stats.csv"),
            "roi_clusters_selected_csv": str(out_dir / "roi_clusters_selected.csv"),
            "roi_clusters_morph_selected_csv": str(out_dir / "roi_clusters_morph_selected.csv"),
            "umap_full_prob_png": str(out_dir / "umap_full_prob.png"),
            "umap_full_mag_png": str(out_dir / "umap_full_mag.png"),
            "umap_full_morph_ellipticity_png": str(out_dir / "umap_full_morph_ellipticity.png"),
            "umap_full_morph_concentration_png": str(out_dir / "umap_full_morph_concentration.png"),
            "umap_full_morph_edge_flux_png": str(out_dir / "umap_full_morph_edge_flux.png"),
            "umap_full_morph_asymmetry_png": str(out_dir / "umap_full_morph_asymmetry.png"),
            "umap_full_with_rois_prob_png": str(out_dir / "umap_full_with_rois_prob.png"),
            "umap_full_with_rois_mag_png": str(out_dir / "umap_full_with_rois_mag.png"),
            "umap_full_with_rois_morph_png": str(out_dir / "umap_full_with_rois_morph.png"),
            "roi_samples_dir": str(out_dir / "roi_samples"),
            "roi_montages_dir": str(out_dir / "roi_montages"),
            "roi_labeling_seed_list_csv": str(out_dir / "roi_labeling_seed_list.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
