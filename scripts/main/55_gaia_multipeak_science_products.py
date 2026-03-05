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
import umap
from scipy.stats import zscore


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


def bin_metrics(y_true: np.ndarray, y_hat01: np.ndarray) -> Dict[str, float]:
    t = y_true.astype(bool)
    p = y_hat01.astype(bool)
    tp = float(np.sum(t & p))
    fp = float(np.sum((~t) & p))
    fn = float(np.sum(t & (~p)))
    tn = float(np.sum((~t) & (~p)))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_rate": float(np.mean(p)),
    }


def threshold_for_precision(val_df: pd.DataFrame, target_precision: float) -> float:
    grid = np.unique(np.quantile(val_df["p_xgb"].to_numpy(dtype=float), np.linspace(0.0, 1.0, 2000)))
    best_thr = float(np.max(grid))
    for thr in np.sort(grid):
        m = bin_metrics(val_df["y_true"].to_numpy(dtype=bool), val_df["p_xgb"].to_numpy(dtype=float) >= thr)
        if m["precision"] >= target_precision:
            best_thr = float(thr)
            break
    return best_thr


def threshold_for_recall(val_df: pd.DataFrame, target_recall: float) -> float:
    grid = np.unique(np.quantile(val_df["p_xgb"].to_numpy(dtype=float), np.linspace(0.0, 1.0, 2000)))
    best_thr = float(np.min(grid))
    for thr in np.sort(grid)[::-1]:
        m = bin_metrics(val_df["y_true"].to_numpy(dtype=bool), val_df["p_xgb"].to_numpy(dtype=float) >= thr)
        if m["recall"] >= target_recall:
            best_thr = float(thr)
            break
    return best_thr


def build_mag_map(dataset_root: Path, cache_csv: Path) -> pd.DataFrame:
    if cache_csv.exists():
        return pd.read_csv(cache_csv)
    rows: List[pd.DataFrame] = []
    for p in sorted(dataset_root.glob("*/metadata_16d.csv")):
        try:
            d = pd.read_csv(p, usecols=["source_id", "phot_g_mean_mag", "feat_log10_snr"], low_memory=False)
            rows.append(d)
        except Exception:
            continue
    if not rows:
        out = pd.DataFrame(columns=["source_id", "phot_g_mean_mag", "feat_log10_snr"])
    else:
        out = pd.concat(rows, ignore_index=True)
        out["source_id"] = pd.to_numeric(out["source_id"], errors="coerce")
        out = out.dropna(subset=["source_id"]).copy()
        out["source_id"] = out["source_id"].astype(np.int64)
        out = out.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    out.to_csv(cache_csv, index=False)
    return out


def plot_umap_prob(emb: np.ndarray, probs: np.ndarray, out_png: Path) -> None:
    plt.figure(figsize=(8.2, 6.8))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=probs, s=4, cmap="viridis", alpha=0.65, linewidths=0)
    cb = plt.colorbar(sc)
    cb.set_label("p(multipeak)")
    plt.title("UMAP of Gaia Features Colored by Multi-peak Probability")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_umap_subsets(emb: np.ndarray, subsets: Dict[str, np.ndarray], out_png: Path) -> None:
    plt.figure(figsize=(8.2, 6.8))
    plt.scatter(emb[:, 0], emb[:, 1], s=2, c="#cfcfcf", alpha=0.35, linewidths=0, label="all")
    styles = {
        "high_purity": ("#0E4D64", "o"),
        "balanced": ("#1A6F8A", "^"),
        "high_recall": ("#B56D1F", "s"),
    }
    for k, mask in subsets.items():
        color, marker = styles.get(k, ("#333333", "o"))
        if int(np.sum(mask)) == 0:
            continue
        plt.scatter(emb[mask, 0], emb[mask, 1], s=12, c=color, alpha=0.8, marker=marker, linewidths=0, label=f"{k} (n={int(np.sum(mask))})")
    plt.title("UMAP with Candidate Subsets Highlighted")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_umap_nuisance(emb: np.ndarray, nuisance: np.ndarray, label: str, out_png: Path) -> None:
    plt.figure(figsize=(8.2, 6.8))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=nuisance, s=4, cmap="plasma", alpha=0.65, linewidths=0)
    cb = plt.colorbar(sc)
    cb.set_label(label)
    plt.title("UMAP Colored by Nuisance Variable")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def match_controls(
    cand_idx: np.ndarray,
    mag: np.ndarray,
    candidate_mask: np.ndarray,
    tol: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pool = np.where(~candidate_mask & np.isfinite(mag))[0]
    used = set()
    out = []
    for i in cand_idx:
        if not np.isfinite(mag[i]):
            continue
        m = np.abs(mag[pool] - mag[i]) <= tol
        choices = pool[m]
        if choices.size == 0:
            continue
        rng.shuffle(choices)
        pick = None
        for c in choices:
            cc = int(c)
            if cc not in used:
                pick = cc
                break
        if pick is not None:
            used.add(pick)
            out.append(pick)
    return np.asarray(out, dtype=np.int64)


def save_montage(
    I: np.ndarray,
    cand_idx: np.ndarray,
    ctrl_idx: np.ndarray,
    title: str,
    out_png: Path,
    n_each: int,
) -> None:
    n1 = min(int(n_each), len(cand_idx))
    n2 = min(int(n_each), len(ctrl_idx))
    cols = 10
    rows = 10
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 12.5))
    for ax in axes.ravel():
        ax.axis("off")

    # first 5 rows candidates
    for j in range(min(n1, 50)):
        r, c = divmod(j, cols)
        ax = axes[r, c]
        img = I[cand_idx[j]]
        lo = np.nanpercentile(img, 1.0)
        hi = np.nanpercentile(img, 99.0)
        ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.axis("off")
    # last 5 rows controls
    for j in range(min(n2, 50)):
        rr, c = divmod(j, cols)
        r = rr + 5
        ax = axes[r, c]
        img = I[ctrl_idx[j]]
        lo = np.nanpercentile(img, 1.0)
        hi = np.nanpercentile(img, 99.0)
        ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.axis("off")

    fig.suptitle(f"{title}\nTop: candidates, Bottom: mag-matched controls", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--pred_run_dir", required=True, help="Output dir from script 53 containing val/test predictions.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--balanced_op_tag", default="precision_floor")
    ap.add_argument("--target_precision", type=float, default=0.7)
    ap.add_argument("--target_recall", type=float, default=0.8)
    ap.add_argument("--n_montage_each", type=int, default=50)
    ap.add_argument("--mag_tol", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path("/data/yn316/Codes")
    run_root = base / "output" / "ml_runs" / str(args.run_name)
    pred_dir = Path(args.pred_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_path = pred_dir / "val_predictions.csv"
    test_path = pred_dir / "test_predictions.csv"
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("pred_run_dir must contain val_predictions.csv and test_predictions.csv from script 53.")

    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    ops = pd.read_csv(pred_dir / "operating_points.csv")

    # Thresholds from validation only.
    bal_row = ops[ops["op_tag"] == str(args.balanced_op_tag)]
    if bal_row.empty:
        bal_row = ops.iloc[[0]]
    thr_bal = float(bal_row.iloc[0]["threshold"])
    thr_hp = threshold_for_precision(val, target_precision=float(args.target_precision))
    thr_hr = threshold_for_recall(val, target_recall=float(args.target_recall))

    y_test = test["y_true"].to_numpy(dtype=bool)
    p_test = test["p_xgb"].to_numpy(dtype=float)
    masks = {
        "high_purity": p_test >= thr_hp,
        "balanced": p_test >= thr_bal,
        "high_recall": p_test >= thr_hr,
    }
    thrs = {
        "high_purity": thr_hp,
        "balanced": thr_bal,
        "high_recall": thr_hr,
    }

    rows_m = []
    for name in ["high_purity", "balanced", "high_recall"]:
        m = masks[name]
        bm = bin_metrics(y_test, m)
        rows_m.append(
            {
                "subset": name,
                "threshold": thrs[name],
                "n_candidates": int(np.sum(m)),
                "candidate_rate": float(np.mean(m)),
                "precision": bm["precision"],
                "recall": bm["recall"],
                "f1": bm["f1"],
                "tp": bm["tp"],
                "fp": bm["fp"],
                "fn": bm["fn"],
            }
        )
    subset_metrics = pd.DataFrame(rows_m)
    subset_metrics.to_csv(out_dir / "subset_metrics.csv", index=False)

    # Candidate rows
    cand_rows = []
    for name, m in masks.items():
        d = test.loc[m, ["test_index", "source_id", "p_xgb", "y_true"]].copy()
        d["subset"] = name
        cand_rows.append(d)
    candidates = pd.concat(cand_rows, ignore_index=True) if cand_rows else pd.DataFrame(columns=["test_index", "source_id", "p_xgb", "y_true", "subset"])
    candidates.to_csv(out_dir / "subsets.csv", index=False)

    # UMAP on Gaia features (test split)
    manifest = load_manifest(run_root)
    X_test, Yshape_test, Yflux_test, stamp_pix = open_test_arrays(manifest)
    X = np.asarray(X_test, dtype=np.float32)
    X_std = zscore(X, axis=0, nan_policy="omit")
    X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=int(args.seed))
    emb = reducer.fit_transform(X_std)
    pd.DataFrame({"test_index": np.arange(len(emb), dtype=np.int64), "umap1": emb[:, 0], "umap2": emb[:, 1]}).to_csv(out_dir / "umap_embedding.csv", index=False)

    plot_umap_prob(emb, p_test, out_dir / "umap_prob.png")
    plot_umap_subsets(emb, masks, out_dir / "umap_subsets.png")

    # Nuisance color: phot_g_mean_mag if available, else feat_log10_snr from metadata or test feature.
    trace_test = pd.read_csv(run_root / "trace" / "trace_test.csv", usecols=["test_index", "source_id"], low_memory=False).sort_values("test_index")
    mag_map = build_mag_map(Path(str(manifest["dataset_root"])), out_dir / "source_mag_cache.csv")
    t = trace_test.merge(mag_map[["source_id", "phot_g_mean_mag", "feat_log10_snr"]], on="source_id", how="left")
    mag = pd.to_numeric(t.get("phot_g_mean_mag", np.nan), errors="coerce").to_numpy(dtype=float)
    if np.sum(np.isfinite(mag)) > 0:
        nuisance = mag
        nuisance_label = "phot_g_mean_mag"
    else:
        snr = pd.to_numeric(t.get("feat_log10_snr", np.nan), errors="coerce").to_numpy(dtype=float)
        if np.sum(np.isfinite(snr)) == 0:
            snr = X[:, 0].astype(float)
        nuisance = snr
        nuisance_label = "feat_log10_snr"
    plot_umap_nuisance(emb, nuisance, nuisance_label, out_dir / "umap_nuisance.png")

    # True-stamp montages with matched controls
    I_true = (np.asarray(Yshape_test, dtype=np.float32) * (10.0 ** np.asarray(Yflux_test, dtype=np.float32)[:, None])).reshape((-1, stamp_pix, stamp_pix))
    rng = np.random.default_rng(int(args.seed))
    mag_for_match = mag if np.sum(np.isfinite(mag)) > 0 else nuisance

    for name in ["high_purity", "balanced", "high_recall"]:
        m = masks[name]
        cand_all = np.where(m)[0]
        if cand_all.size == 0:
            continue
        if cand_all.size > int(args.n_montage_each):
            cand_idx = np.sort(rng.choice(cand_all, size=int(args.n_montage_each), replace=False))
        else:
            cand_idx = cand_all
        ctrl_idx = match_controls(
            cand_idx=cand_idx,
            mag=mag_for_match,
            candidate_mask=m,
            tol=float(args.mag_tol),
            rng=rng,
        )
        save_montage(
            I=I_true,
            cand_idx=cand_idx,
            ctrl_idx=ctrl_idx,
            title=f"{name} (thr={thrs[name]:.4f})",
            out_png=out_dir / f"montage_{name}.png",
            n_each=int(args.n_montage_each),
        )

    summary = {
        "run_name": str(args.run_name),
        "pred_run_dir": str(pred_dir),
        "thresholds": {"high_purity": thr_hp, "balanced": thr_bal, "high_recall": thr_hr},
        "targets": {"target_precision": float(args.target_precision), "target_recall": float(args.target_recall)},
        "outputs": {
            "subset_metrics_csv": str(out_dir / "subset_metrics.csv"),
            "subsets_csv": str(out_dir / "subsets.csv"),
            "umap_embedding_csv": str(out_dir / "umap_embedding.csv"),
            "umap_prob_png": str(out_dir / "umap_prob.png"),
            "umap_subsets_png": str(out_dir / "umap_subsets.png"),
            "umap_nuisance_png": str(out_dir / "umap_nuisance.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
