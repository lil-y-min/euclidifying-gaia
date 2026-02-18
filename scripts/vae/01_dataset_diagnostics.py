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

from common import image_moments, load_manifest, open_split_memmaps, ensure_dirs


def quantiles(v: np.ndarray) -> Dict[str, float]:
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(vv)),
        "median": float(np.median(vv)),
        "p90": float(np.percentile(vv, 90.0)),
    }


def save_hist(v: np.ndarray, path: Path, title: str, xlabel: str) -> None:
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return
    plt.figure(figsize=(6.2, 4.2))
    plt.hist(vv, bins=80)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to manifest_arrays.npz")
    ap.add_argument("--run_name", default="cvae_diag")
    ap.add_argument("--max_per_split", type=int, default=80000)
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    base_dir = Path(__file__).resolve().parents[2]
    out_run = base_dir / "output" / "ml_runs" / "vae" / args.run_name
    plot_dir = base_dir / "plots" / "ml_runs" / "vae" / args.run_name
    rep_dir = base_dir / "report" / "model_decision" / "vae" / args.run_name
    ensure_dirs(out_run, plot_dir, rep_dir)

    summary = {
        "manifest": str(manifest.path),
        "stamp_pix": int(manifest.stamp_pix),
        "n_features": int(manifest.n_features),
        "splits": {},
    }

    for split in ("train", "val", "test"):
        X, Yshape, Yflux = open_split_memmaps(manifest, split)
        n = X.shape[0]
        k = min(n, int(args.max_per_split))
        idx = np.arange(n) if k == n else np.random.default_rng(123).choice(n, size=k, replace=False)

        yshape = np.asarray(Yshape[idx], dtype=np.float32)
        flux = np.asarray(Yflux[idx], dtype=np.float32)
        moms = image_moments(yshape, manifest.stamp_pix)

        summary["splits"][split] = {
            "n_total": int(n),
            "n_used": int(k),
            "flux_log10": quantiles(flux),
            "x0": quantiles(moms["x0"]),
            "y0": quantiles(moms["y0"]),
            "ell": quantiles(moms["ell"]),
            "sigma": quantiles(moms["sigma"]),
        }

        save_hist(moms["x0"], plot_dir / f"{split}_x0_hist.png", f"{split} centroid x0", "x0 [pix]")
        save_hist(moms["y0"], plot_dir / f"{split}_y0_hist.png", f"{split} centroid y0", "y0 [pix]")
        save_hist(moms["ell"], plot_dir / f"{split}_ell_hist.png", f"{split} ellipticity", "ell")
        save_hist(moms["sigma"], plot_dir / f"{split}_sigma_hist.png", f"{split} size sigma", "sigma [pix]")

    with open(out_run / "dataset_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(rep_dir / "dataset_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_run / "dataset_diagnostics.json")
    print("Plots:", plot_dir)
    print("Report copy:", rep_dir / "dataset_diagnostics.json")


if __name__ == "__main__":
    main()
