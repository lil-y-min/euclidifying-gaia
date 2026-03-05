#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


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


def save_montage(I: np.ndarray, idx: np.ndarray, title: str, out_png: Path, n_show: int = 200) -> None:
    n = min(int(n_show), len(idx))
    if n == 0:
        return
    cols = 10
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n:
            continue
        img = I[idx[i]]
        lo = np.nanpercentile(img, 1.0)
        hi = np.nanpercentile(img, 99.0)
        ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--science_products_dir", required=True, help="Output dir from script 55 (contains umap_embedding.csv)")
    ap.add_argument("--pred_run_dir", required=True, help="Output dir from script 53 (contains test_predictions.csv)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_top", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path("/data/yn316/Codes")
    run_root = base / "output" / "ml_runs" / str(args.run_name)
    sci_dir = Path(args.science_products_dir)
    pred_dir = Path(args.pred_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = pd.read_csv(sci_dir / "umap_embedding.csv")
    pred = pd.read_csv(pred_dir / "test_predictions.csv")
    trace = pd.read_csv(run_root / "trace" / "trace_test.csv", usecols=["test_index", "source_id"], low_memory=False)
    df = emb.merge(pred[["test_index", "y_true", "p_xgb"]], on="test_index", how="inner").merge(trace, on="test_index", how="left")

    manifest = load_manifest(run_root)
    X_test, Yshape_test, Yflux_test, stamp_pix = open_test_arrays(manifest)
    X = np.asarray(X_test, dtype=np.float32)
    I_true = (np.asarray(Yshape_test, dtype=np.float32) * (10.0 ** np.asarray(Yflux_test, dtype=np.float32)[:, None])).reshape((-1, stamp_pix, stamp_pix))

    # LOF on UMAP coordinates
    um = emb[["umap1", "umap2"]].to_numpy(dtype=float)
    lof_um = LocalOutlierFactor(n_neighbors=35, contamination=0.01, novelty=False)
    _ = lof_um.fit_predict(um)
    score_um = -lof_um.negative_outlier_factor_
    df["outlier_score_umap_lof"] = score_um
    top_um = df.sort_values("outlier_score_umap_lof", ascending=False).head(int(args.n_top)).copy()
    top_um.to_csv(out_dir / "outliers_umap_top200.csv", index=False)

    # IsolationForest on original Gaia feature space
    iso = IsolationForest(
        n_estimators=400,
        contamination=0.01,
        random_state=int(args.seed),
        n_jobs=1,
    )
    iso.fit(X)
    score_feat = -iso.score_samples(X)
    df["outlier_score_feature_iforest"] = score_feat
    top_feat = df.sort_values("outlier_score_feature_iforest", ascending=False).head(int(args.n_top)).copy()
    top_feat.to_csv(out_dir / "outliers_feature_top200.csv", index=False)

    # Ambiguous and confident-wrong slices
    df["dist_to_0p5"] = np.abs(df["p_xgb"] - 0.5)
    amb = df.sort_values("dist_to_0p5", ascending=True).head(int(args.n_top)).copy()
    amb.to_csv(out_dir / "outliers_ambiguous_prob_top200.csv", index=False)

    df["is_fp_confident"] = (df["y_true"] == 0) & (df["p_xgb"] >= 0.9)
    df["is_fn_confident"] = (df["y_true"] == 1) & (df["p_xgb"] <= 0.1)
    fp_conf = df[df["is_fp_confident"]].sort_values("p_xgb", ascending=False).head(int(args.n_top)).copy()
    fn_conf = df[df["is_fn_confident"]].sort_values("p_xgb", ascending=True).head(int(args.n_top)).copy()
    fp_conf.to_csv(out_dir / "outliers_confident_fp_top200.csv", index=False)
    fn_conf.to_csv(out_dir / "outliers_confident_fn_top200.csv", index=False)

    # overlap
    set_um = set(top_um["test_index"].astype(int).tolist())
    set_feat = set(top_feat["test_index"].astype(int).tolist())
    overlap = sorted(set_um & set_feat)
    pd.DataFrame({"test_index": overlap}).to_csv(out_dir / "outliers_overlap_umap_and_feature.csv", index=False)

    # montages
    save_montage(I_true, top_um["test_index"].to_numpy(dtype=int), "Top UMAP-density outliers (LOF)", out_dir / "montage_outliers_umap_top200.png", n_show=int(args.n_top))
    save_montage(I_true, top_feat["test_index"].to_numpy(dtype=int), "Top feature-space outliers (IsolationForest)", out_dir / "montage_outliers_feature_top200.png", n_show=int(args.n_top))
    save_montage(I_true, amb["test_index"].to_numpy(dtype=int), "Top probability-ambiguous cases", out_dir / "montage_outliers_ambiguous_top200.png", n_show=int(args.n_top))
    if len(fp_conf) > 0:
        save_montage(I_true, fp_conf["test_index"].to_numpy(dtype=int), "Confident false positives", out_dir / "montage_outliers_confident_fp_top200.png", n_show=int(args.n_top))
    if len(fn_conf) > 0:
        save_montage(I_true, fn_conf["test_index"].to_numpy(dtype=int), "Confident false negatives", out_dir / "montage_outliers_confident_fn_top200.png", n_show=int(args.n_top))

    summary = {
        "run_name": str(args.run_name),
        "n_top": int(args.n_top),
        "counts": {
            "overlap_umap_feature": int(len(overlap)),
            "confident_fp_found": int(len(fp_conf)),
            "confident_fn_found": int(len(fn_conf)),
        },
        "outputs": {
            "outliers_umap_csv": str(out_dir / "outliers_umap_top200.csv"),
            "outliers_feature_csv": str(out_dir / "outliers_feature_top200.csv"),
            "outliers_ambiguous_csv": str(out_dir / "outliers_ambiguous_prob_top200.csv"),
            "outliers_confident_fp_csv": str(out_dir / "outliers_confident_fp_top200.csv"),
            "outliers_confident_fn_csv": str(out_dir / "outliers_confident_fn_top200.csv"),
            "overlap_csv": str(out_dir / "outliers_overlap_umap_and_feature.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
