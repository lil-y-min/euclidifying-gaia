#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


def load_manifest(run_root: Path) -> dict:
    mpath = run_root / "manifest_arrays.npz"
    if not mpath.exists():
        raise FileNotFoundError(f"Missing manifest: {mpath}")
    d = np.load(mpath, allow_pickle=True)
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
    return X_test, Yshape_test, Yflux_test, stamp_pix, D


def predict_shape(models_dir: Path, X_sub: np.ndarray, D: int) -> np.ndarray:
    dmat = xgb.DMatrix(X_sub)
    out = np.empty((X_sub.shape[0], D), dtype=np.float32)
    for j in range(D):
        b = xgb.Booster()
        b.load_model(str(models_dir / f"booster_pix_{j:04d}.json"))
        out[:, j] = b.predict(dmat).astype(np.float32, copy=False)
    return out


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor="black")
    plt.close()


def make_panel(group_df: pd.DataFrame, I_true: np.ndarray, I_pred: np.ndarray, out_png: Path, title: str):
    n = len(group_df)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(8.5, max(2.0 * n, 3.0)), facecolor="black")
    if n == 1:
        axes = np.array([axes])
    for i in range(n):
        t = I_true[i]
        p = I_pred[i]
        r = t - p
        lo = np.nanpercentile(np.r_[t.ravel(), p.ravel()], 5)
        hi = np.nanpercentile(np.r_[t.ravel(), p.ravel()], 99.5)
        rv = np.nanpercentile(np.abs(r), 99)
        row = group_df.iloc[i]
        sid = int(row["source_id"]) if np.isfinite(row["source_id"]) else -1
        ra = float(row["ra"]) if np.isfinite(row["ra"]) else np.nan
        dec = float(row["dec"]) if np.isfinite(row["dec"]) else np.nan
        hdr = f"idx={int(row['test_index'])} sid={sid} chi2={float(row['chi2nu']):.3g}\nra={ra:.6f} dec={dec:.6f}"

        for j, (img, cmap, vmin, vmax, ttxt) in enumerate(
            [
                (t, "gray", lo, hi, "TRUE"),
                (p, "gray", lo, hi, "PRED"),
                (r, "RdBu_r", -rv, rv, "RESID"),
            ]
        ):
            ax = axes[i, j]
            ax.set_facecolor("black")
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(hdr, color="white", fontsize=7)
            ax.set_title(ttxt, color="white", fontsize=8)
    fig.suptitle(title, color="white")
    _savefig(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--n_each", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    base = Path("/data/yn316/Codes")
    run_root = base / "output" / "ml_runs" / str(args.run_name)
    plots_root = base / "plots" / "ml_runs" / str(args.run_name)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base / "report" / "model_decision" / f"20260227_{args.run_name}_best_random_worst")
    out_dir.mkdir(parents=True, exist_ok=True)

    chi = pd.read_csv(plots_root / "test_chi2_metrics.csv", usecols=["chi2nu"]).reset_index().rename(columns={"index": "test_index"})
    tr = pd.read_csv(run_root / "trace" / "trace_test.csv")
    df = tr.merge(chi, on="test_index", how="inner")
    df["source_id"] = pd.to_numeric(df.get("source_id"), errors="coerce")
    df["ra"] = pd.to_numeric(df.get("ra"), errors="coerce")
    df["dec"] = pd.to_numeric(df.get("dec"), errors="coerce")
    df = df.sort_values("test_index").reset_index(drop=True)

    n_each = int(args.n_each)
    rng = np.random.default_rng(int(args.seed))
    best = df.nsmallest(n_each, "chi2nu").copy()
    worst = df.nlargest(n_each, "chi2nu").copy()
    used = set(best["test_index"].tolist()) | set(worst["test_index"].tolist())
    pool = df.loc[~df["test_index"].isin(used)]
    take = min(n_each, len(pool))
    random_df = pool.sample(n=take, random_state=int(args.seed)).copy()

    pick = pd.concat(
        [
            best.assign(group="best"),
            random_df.assign(group="random"),
            worst.assign(group="worst"),
        ],
        ignore_index=True,
    ).reset_index(drop=True)
    pick.to_csv(out_dir / "selected_samples.csv", index=False)

    manifest = load_manifest(run_root)
    X_test, Yshape_test, Yflux_test, stamp_pix, D = open_test_arrays(manifest)
    idx = pick["test_index"].to_numpy(dtype=int)
    X_sub = np.asarray(X_test[idx], dtype=np.float32)
    yshape_true = np.asarray(Yshape_test[idx], dtype=np.float32)
    yflux_true = np.asarray(Yflux_test[idx], dtype=np.float32)

    yshape_pred = predict_shape(run_root / "models", X_sub, D=D)
    I_true = yshape_true * (10.0 ** yflux_true[:, None])
    I_pred = yshape_pred * (10.0 ** yflux_true[:, None])  # true-flux mode
    I_true = I_true.reshape((-1, stamp_pix, stamp_pix))
    I_pred = I_pred.reshape((-1, stamp_pix, stamp_pix))

    # write panels
    for grp in ["best", "random", "worst"]:
        g = pick[pick["group"] == grp].reset_index(drop=True)
        gi = pick.index[pick["group"] == grp].to_numpy(dtype=int)
        make_panel(
            g,
            I_true[gi],
            I_pred[gi],
            out_dir / f"{grp}_panel.png",
            f"{args.run_name} - {grp.upper()} ({len(g)})",
        )
        g.to_csv(out_dir / f"{grp}_samples.csv", index=False)

    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
