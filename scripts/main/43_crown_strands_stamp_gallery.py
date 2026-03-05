#!/usr/bin/env python3
"""
Build crown-region stamp galleries from pixel UMAP embedding.

What it does:
- Load embedding_umap.csv from pixel UMAP run.
- Select ROI (region of interest) with x < 0 and configurable y-range.
- Split ROI into 5 groups:
  - 4 user-defined strand boxes
  - bottom = remaining ROI points not in any strand box
- Sample 10 sources per part.
- For each sampled source, load TRUE stamp from npz, normalize per-stamp
  with integral_pos, predict stamp pixels from XGB shape boosters, and plot
  TRUE / PRED / RESID triplets.
- Annotate each row with source_id, field_tag, RA, Dec.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import PowerNorm


FEATURE_COLS_8D = [
    "feat_log10_snr",
    "feat_ruwe",
    "feat_astrometric_excess_noise",
    "feat_parallax_over_error",
    "feat_visibility_periods_used",
    "feat_ipd_frac_multi_peak",
    "feat_c_star",
    "feat_pm_significance",
]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def normalize_integral_pos(stamp: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.asarray(stamp, dtype=np.float32)
    den = float(np.sum(np.clip(s, 0.0, np.inf)))
    if not np.isfinite(den) or den <= eps:
        return np.full_like(s, np.nan, dtype=np.float32)
    return (s / den).astype(np.float32, copy=False)


def load_shape_models(model_dir: Path, D: int = 400) -> List[xgb.Booster]:
    boosters: List[xgb.Booster] = []
    for j in range(D):
        p = model_dir / f"booster_pix_{j:04d}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing pixel model: {p}")
        b = xgb.Booster()
        b.load_model(str(p))
        boosters.append(b)
    return boosters


def predict_shape(boosters: List[xgb.Booster], X: np.ndarray) -> np.ndarray:
    dmat = xgb.DMatrix(np.asarray(X, dtype=np.float32, order="C"))
    out = np.empty((X.shape[0], len(boosters)), dtype=np.float32)
    for j, b in enumerate(boosters):
        out[:, j] = b.predict(dmat).astype(np.float32, copy=False)
    return out


def load_ra_dec_for_sources(dataset_root: Path, source_ids: np.ndarray) -> pd.DataFrame:
    sid_set = set(int(x) for x in source_ids.tolist())
    rows: List[pd.DataFrame] = []
    metas = sorted(dataset_root.glob("*/metadata.csv"))
    for p in metas:
        usecols = ["source_id", "ra", "dec", "field_tag", "fits_path"]
        df = pd.read_csv(p, usecols=lambda c: c in set(usecols), low_memory=False)
        if "source_id" not in df.columns:
            continue
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
        df = df[df["source_id"].notna()].copy()
        df["source_id"] = df["source_id"].astype(np.int64)
        df = df[df["source_id"].isin(sid_set)].copy()
        if df.empty:
            continue
        if "field_tag" not in df.columns:
            if "fits_path" in df.columns:
                df["field_tag"] = df["fits_path"].astype(str).map(lambda s: Path(s).parent.name if str(s).strip() else p.parent.name)
            else:
                df["field_tag"] = p.parent.name
        rows.append(df[["source_id", "ra", "dec", "field_tag"]])
    if not rows:
        return pd.DataFrame(columns=["source_id", "ra", "dec", "field_tag"])
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("source_id").drop_duplicates("source_id", keep="first")
    return out


def select_crown_parts(df: pd.DataFrame, y_min: float, y_max: float) -> pd.DataFrame:
    # Global hard constraint requested by user.
    roi = df[(df["x"] >= -4.0) & (df["x"] < -0.5) & (df["y"] >= y_min) & (df["y"] <= y_max)].copy()
    if len(roi) < 100:
        raise RuntimeError(f"ROI too small ({len(roi)} rows). Try adjusting bounds.")

    roi["crown_part"] = "outside"

    # Strand boxes from user guidance, clipped to x < 0.
    m1 = (roi["x"] >= -4.0) & (roi["x"] <= -2.7) & (roi["y"] >= 16.5) & (roi["y"] <= 17.5)
    m2 = (roi["x"] >= -3.0) & (roi["x"] <= -1.8) & (roi["y"] >= 17.1) & (roi["y"] <= 18.2)
    m3 = (roi["x"] >= -2.1) & (roi["x"] <= -1.5) & (roi["y"] >= 17.45) & (roi["y"] <= 18.5)
    m4 = (roi["x"] >= -1.27) & (roi["x"] <= -1.0) & (roi["y"] >= 17.5) & (roi["y"] <= 18.4)

    # Assign strands first; later strand masks override earlier ones.
    roi.loc[m1, "crown_part"] = "strand_1"
    roi.loc[m2, "crown_part"] = "strand_2"
    roi.loc[m3, "crown_part"] = "strand_3"
    roi.loc[m4, "crown_part"] = "strand_4"

    # Bottom box, excluding anything already claimed by strands.
    m_bottom_box = (roi["x"] > -2.2) & (roi["x"] <= -0.7) & (roi["y"] >= 15.5) & (roi["y"] < 17.0)
    m_any_strand = roi["crown_part"].isin(["strand_1", "strand_2", "strand_3", "strand_4"])
    m_bottom = m_bottom_box & (~m_any_strand)
    roi.loc[m_bottom, "crown_part"] = "bottom"

    # Keep only selected crown parts.
    roi = roi[roi["crown_part"].isin(["bottom", "strand_1", "strand_2", "strand_3", "strand_4"])].copy()
    return roi


def sample_per_part(roi: pd.DataFrame, n_per_part: int, seed: int) -> pd.DataFrame:
    out: List[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    for part in ["bottom", "strand_1", "strand_2", "strand_3", "strand_4"]:
        sub = roi[roi["crown_part"] == part].copy()
        if sub.empty:
            continue
        n = min(n_per_part, len(sub))
        idx = rng.choice(len(sub), size=n, replace=False)
        ss = sub.iloc[np.sort(idx)].copy()
        ss["part_rank"] = np.arange(1, len(ss) + 1)
        out.append(ss)
    if not out:
        raise RuntimeError("No crown parts sampled.")
    return pd.concat(out, ignore_index=True)


def fetch_true_stamps(dataset_root: Path, sel: pd.DataFrame) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    groups = sel.groupby(["dataset_tag", "npz_file"], sort=False)
    for (dataset_tag, npz_file), g in groups:
        npz_path = dataset_root / str(dataset_tag) / str(npz_file)
        if not npz_path.exists():
            continue
        with np.load(npz_path) as d:
            if "X" not in d:
                continue
            Xall = d["X"]
        for r in g.itertuples(index=False):
            ii = int(r.index_in_file)
            if 0 <= ii < Xall.shape[0]:
                out[int(r.source_id)] = np.asarray(Xall[ii], dtype=np.float32)
    return out


def plot_part_gallery(df_part: pd.DataFrame, true_map: Dict[int, np.ndarray], pred_map: Dict[int, np.ndarray], out_png: Path) -> None:
    n = len(df_part)
    fig, axes = plt.subplots(
        n,
        4,
        figsize=(12.5, 2.15 * n),
        gridspec_kw={"width_ratios": [2.6, 1.0, 1.0, 1.0]},
    )
    if n == 1:
        axes = np.asarray([axes])

    for r, row in enumerate(df_part.itertuples(index=False)):
        sid = int(row.source_id)
        t = true_map.get(sid)
        p = pred_map.get(sid)
        if t is None or p is None or not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)):
            t = np.full((20, 20), np.nan, dtype=np.float32)
            p = np.full((20, 20), np.nan, dtype=np.float32)
        z = p - t

        at, a0, a1, a2 = axes[r, 0], axes[r, 1], axes[r, 2], axes[r, 3]
        at.axis("off")
        for ax in (a0, a1, a2):
            ax.set_xticks([])
            ax.set_yticks([])

        vmin = float(np.nanpercentile(np.concatenate([t.ravel(), p.ravel()]), 2))
        vmax = float(np.nanpercentile(np.concatenate([t.ravel(), p.ravel()]), 99.8))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = -1e-6, 1e-6

        rz = float(np.nanpercentile(np.abs(z.ravel()), 99))
        rz = rz if np.isfinite(rz) and rz > 0 else 1e-6

        # Gamma<1 boosts faint structure for better visual readability.
        norm_tp = PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax, clip=True)
        im0 = a0.imshow(t, origin="lower", cmap="gray", norm=norm_tp)
        im1 = a1.imshow(p, origin="lower", cmap="gray", norm=norm_tp)
        im2 = a2.imshow(z, origin="lower", cmap="RdBu_r", vmin=-rz, vmax=+rz)

        if r == 0:
            at.set_title("Source Info", fontsize=10)
            a0.set_title("TRUE", fontsize=10)
            a1.set_title("PRED", fontsize=10)
            a2.set_title("RESID", fontsize=10)

        ra_txt = "nan" if not np.isfinite(float(row.ra)) else f"{float(row.ra):.6f}"
        dec_txt = "nan" if not np.isfinite(float(row.dec)) else f"{float(row.dec):.6f}"
        row_txt = "\n".join(
            [
                f"source_id: {sid}",
                f"field: {row.field_tag}",
                f"RA, Dec: {ra_txt}, {dec_txt}",
                f"UMAP x,y: {float(row.x):.2f}, {float(row.y):.2f}",
            ]
        )
        at.text(0.02, 0.5, row_txt, fontsize=8, va="center", ha="left", transform=at.transAxes)

    part_name = str(df_part["crown_part"].iloc[0])
    fig.suptitle(f"Crown Part: {part_name} (n={n})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98], w_pad=0.9, h_pad=0.9)
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Crown strands TRUE/PRED/RESID galleries from pixel-UMAP.")
    ap.add_argument(
        "--embedding_csv",
        default="/data/yn316/Codes/scripts/output/experiments/embeddings/double_stars_pixels/umap_standard_sigma_floor/embedding_umap.csv",
    )
    ap.add_argument("--dataset_root", default="/data/yn316/Codes/output/dataset_npz")
    ap.add_argument("--run_name", default="ml_xgb_pixelsflux_8d_augtrain_g17")
    ap.add_argument(
        "--out_dir",
        default="/data/yn316/Codes/plots/qa/embeddings/double_stars_pixels/umap_standard_sigma_floor/crown_strands_gallery",
    )
    ap.add_argument("--n_per_part", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--roi_y_min", type=float, default=16.0)
    ap.add_argument("--roi_y_max", type=float, default=18.5)
    args = ap.parse_args()

    emb_csv = Path(args.embedding_csv)
    out_dir = ensure_dir(Path(args.out_dir))
    dataset_root = Path(args.dataset_root)
    run_root = Path("/data/yn316/Codes/output/ml_runs") / str(args.run_name)
    model_dir = run_root / "models"

    dfe = pd.read_csv(emb_csv, low_memory=False)
    need = ["source_id", "field_tag", "dataset_tag", "npz_file", "index_in_file", "x", "y", *FEATURE_COLS_8D]
    miss = [c for c in need if c not in dfe.columns]
    if miss:
        raise RuntimeError(f"Missing columns in embedding CSV: {miss}")

    for c in ["source_id", "index_in_file", "x", "y", *FEATURE_COLS_8D]:
        dfe[c] = pd.to_numeric(dfe[c], errors="coerce")
    dfe = dfe.dropna(subset=["source_id", "index_in_file", "x", "y", *FEATURE_COLS_8D]).copy()
    dfe["source_id"] = dfe["source_id"].astype(np.int64)
    dfe["index_in_file"] = dfe["index_in_file"].astype(int)

    roi = select_crown_parts(dfe, y_min=float(args.roi_y_min), y_max=float(args.roi_y_max))
    sel = sample_per_part(roi, n_per_part=int(args.n_per_part), seed=int(args.seed))

    coord = load_ra_dec_for_sources(dataset_root, sel["source_id"].to_numpy(dtype=np.int64))
    sel = sel.merge(coord, on="source_id", how="left")
    if "field_tag" not in sel.columns:
        if "field_tag_x" in sel.columns:
            sel["field_tag"] = sel["field_tag_x"]
        elif "field_tag_y" in sel.columns:
            sel["field_tag"] = sel["field_tag_y"]
    sel["ra"] = pd.to_numeric(sel["ra"], errors="coerce")
    sel["dec"] = pd.to_numeric(sel["dec"], errors="coerce")

    true_raw = fetch_true_stamps(dataset_root, sel)
    true_norm: Dict[int, np.ndarray] = {}
    for sid, s in true_raw.items():
        true_norm[sid] = normalize_integral_pos(s)

    boosters = load_shape_models(model_dir=model_dir, D=400)
    Xsel = sel[FEATURE_COLS_8D].to_numpy(dtype=np.float32)
    ypred = predict_shape(boosters, Xsel).reshape(len(sel), 20, 20)
    pred_norm: Dict[int, np.ndarray] = {}
    for i, row in enumerate(sel.itertuples(index=False)):
        pred_norm[int(row.source_id)] = ypred[i]

    sel_out = out_dir / "crown_selected_sources.csv"
    keep = ["crown_part", "part_rank", "source_id", "field_tag", "dataset_tag", "npz_file", "index_in_file", "ra", "dec", "x", "y"]
    sel[keep].to_csv(sel_out, index=False)

    parts = ["bottom", "strand_1", "strand_2", "strand_3", "strand_4"]
    made = []
    for p in parts:
        sub = sel[sel["crown_part"] == p].copy().sort_values("part_rank")
        if sub.empty:
            continue
        out_png = out_dir / f"{p}_true_pred_resid_10.png"
        plot_part_gallery(sub, true_map=true_norm, pred_map=pred_norm, out_png=out_png)
        made.append(str(out_png))

    # ROI diagnostic with selected points overlaid.
    plt.figure(figsize=(7.8, 6.8))
    plt.scatter(roi["x"], roi["y"], s=3, alpha=0.28, c="#4a4a4a", linewidths=0, label="ROI points")
    colors = {"bottom": "#f4a261", "strand_1": "#e63946", "strand_2": "#457b9d", "strand_3": "#2a9d8f", "strand_4": "#6a4c93"}
    for p in parts:
        sub = sel[sel["crown_part"] == p]
        if sub.empty:
            continue
        plt.scatter(sub["x"], sub["y"], s=35, alpha=0.95, c=colors[p], edgecolors="white", linewidths=0.3, label=p)
    plt.xlabel("Embedding X")
    plt.ylabel("Embedding Y")
    plt.title("Crown ROI Partition and Selected 10-per-part Samples")
    plt.legend(loc="best", frameon=True)
    savefig(out_dir / "crown_roi_partition_selected_points.png")

    summary = {
        "embedding_csv": str(emb_csv),
        "run_name": str(args.run_name),
        "roi_rule": f"-4.0 <= x < -0.5 and {float(args.roi_y_min):.3g} <= y <= {float(args.roi_y_max):.3g}",
        "strand_rule": {
            "strand_1": "x in [-4.0,-2.7], y in [16.5,17.5]",
            "strand_2": "x in [-3.0,-1.8], y in [17.1,18.2]",
            "strand_3": "x in [-2.1,-1.5], y in [17.45,18.5]",
            "strand_4": "x in [-1.27,-1.0], y in [17.5,18.4]",
            "bottom": "x in (-2.2,-0.7], y in [15.5,17.0), excluding strand_1..4",
        },
        "n_roi": int(len(roi)),
        "n_selected_total": int(len(sel)),
        "n_per_part_target": int(args.n_per_part),
        "parts_counts": {k: int((sel["crown_part"] == k).sum()) for k in parts},
        "selected_table": str(sel_out),
        "galleries": made,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print("Output dir:", out_dir)
    print("Selected table:", sel_out)


if __name__ == "__main__":
    main()
