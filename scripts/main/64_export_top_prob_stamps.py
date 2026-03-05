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


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def build_lookup(dataset_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for meta in sorted(dataset_root.glob("*/metadata_16d.csv")):
        tag = meta.parent.name
        use = ["source_id", "npz_file", "index_in_file", "phot_g_mean_mag", "ra", "dec"]
        d = pd.read_csv(meta, usecols=lambda c: c in set(use), low_memory=False)
        for c in ["source_id", "index_in_file", "phot_g_mean_mag", "ra", "dec"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["source_id", "npz_file", "index_in_file"]).copy()
        d["source_id"] = d["source_id"].astype(np.int64)
        d["index_in_file"] = d["index_in_file"].astype(int)
        d["dataset_tag"] = tag
        keep = [c for c in ["source_id", "dataset_tag", "npz_file", "index_in_file", "phot_g_mean_mag", "ra", "dec"] if c in d.columns]
        rows.append(d[keep])
    if not rows:
        raise RuntimeError(f"No metadata_16d.csv found under {dataset_root}")
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return out


def fetch_stamps(df: pd.DataFrame, dataset_root: Path) -> Tuple[List[np.ndarray], List[int]]:
    imgs: List[np.ndarray] = []
    sids: List[int] = []
    for _, r in df.iterrows():
        sid = int(r["source_id"])
        npz_path = dataset_root / str(r["dataset_tag"]) / str(r["npz_file"])
        idx = int(r["index_in_file"])
        if not npz_path.exists():
            continue
        try:
            with np.load(npz_path, mmap_mode="r") as d:
                if "X" not in d:
                    continue
                x = d["X"]
                if idx < 0 or idx >= x.shape[0]:
                    continue
                imgs.append(np.array(x[idx], dtype=np.float32))
                sids.append(sid)
        except Exception:
            continue
    return imgs, sids


def plot_panel(imgs: List[np.ndarray], rows: pd.DataFrame, sids: List[int], title: str, out_png: Path) -> None:
    n = len(imgs)
    if n == 0:
        return
    cols = 5
    rr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rr, cols, figsize=(2.8 * cols, 2.8 * rr))
    axes = np.array(axes).reshape(rr, cols)
    m = {int(r["source_id"]): r for _, r in rows.iterrows()}
    for i in range(rr * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n:
            continue
        im = imgs[i]
        lo, hi = np.nanpercentile(im, [1, 99])
        sid = int(sids[i])
        row = m.get(sid, None)
        prob_txt = ""
        if row is not None:
            if "p_multipeak_xgb" in row:
                prob_txt += f"p_multipeak={float(row['p_multipeak_xgb']):.3f} "
            if "p_nonpsf_xgb" in row:
                prob_txt += f"p_nonpsf={float(row['p_nonpsf_xgb']):.3f}"
            ra_txt = f"{float(row['ra']):.5f}" if ("ra" in row and np.isfinite(row["ra"])) else "nan"
            dec_txt = f"{float(row['dec']):.5f}" if ("dec" in row and np.isfinite(row["dec"])) else "nan"
            prob_txt = f"{prob_txt.strip()}\nRA={ra_txt} Dec={dec_txt}"
        ax.imshow(im, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"{sid}\n{prob_txt.strip()}", fontsize=7)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs_csv", required=True, help="CSV with source_id, p_multipeak_xgb, p_nonpsf_xgb.")
    ap.add_argument("--dataset_root", default="/data/yn316/Codes/output/dataset_npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_top", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = pd.read_csv(Path(args.probs_csv), low_memory=False)
    for c in ["source_id", "p_multipeak_xgb", "p_nonpsf_xgb"]:
        if c not in probs.columns:
            raise RuntimeError(f"Missing column in probs_csv: {c}")
    probs["source_id"] = pd.to_numeric(probs["source_id"], errors="coerce")
    probs["p_multipeak_xgb"] = pd.to_numeric(probs["p_multipeak_xgb"], errors="coerce")
    probs["p_nonpsf_xgb"] = pd.to_numeric(probs["p_nonpsf_xgb"], errors="coerce")
    probs = probs.dropna(subset=["source_id"]).copy()
    probs["source_id"] = probs["source_id"].astype(np.int64)
    probs = probs.drop_duplicates(subset=["source_id"], keep="first")

    lookup = build_lookup(Path(args.dataset_root))
    full = probs.merge(lookup, on="source_id", how="inner")

    top_m = full.sort_values("p_multipeak_xgb", ascending=False).head(int(args.n_top)).copy()
    top_n = full.sort_values("p_nonpsf_xgb", ascending=False).head(int(args.n_top)).copy()
    top_m.to_csv(out_dir / "top_multipeak_probs.csv", index=False)
    top_n.to_csv(out_dir / "top_nonpsf_probs.csv", index=False)

    imgs_m, sid_m = fetch_stamps(top_m, Path(args.dataset_root))
    imgs_n, sid_n = fetch_stamps(top_n, Path(args.dataset_root))
    plot_panel(imgs_m, top_m, sid_m, f"Top {int(args.n_top)} p(multipeak)", out_dir / "top_multipeak_stamps.png")
    plot_panel(imgs_n, top_n, sid_n, f"Top {int(args.n_top)} p(nonpsf)", out_dir / "top_nonpsf_stamps.png")
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
