#!/usr/bin/env python3
"""
Visual residual audit for top outliers (stratified).

Generates:
- one PNG per selected source with 3 panels (true / pred / normalized residual)
- summary table CSV
- manual failure-mode template CSV
- Fornax vs non-Fornax comparison sheet
- markdown report
"""

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


def _q(v: np.ndarray, q: float) -> float:
    vv = np.asarray(v, dtype=float)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return float("nan")
    return float(np.quantile(vv, q))


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def load_manifest(run_root: Path) -> Dict[str, object]:
    man = np.load(run_root / "manifest_arrays.npz", allow_pickle=True)
    out = {k: man[k] for k in man.files}
    for k in ["n_test", "n_features", "stamp_pix", "D"]:
        out[k] = int(np.asarray(out[k]).reshape(()))
    for k in ["X_test_path", "Yshape_test_path", "Yflux_test_path"]:
        out[k] = str(out[k])
    return out


def load_shape_models(model_dir: Path, D: int) -> List[xgb.Booster]:
    arr: List[xgb.Booster] = []
    for j in range(D):
        p = model_dir / f"booster_pix_{j:04d}.json"
        if not p.exists():
            raise FileNotFoundError(p)
        b = xgb.Booster()
        b.load_model(str(p))
        arr.append(b)
    return arr


def predict_shape_selected(boosters: List[xgb.Booster], Xsel: np.ndarray) -> np.ndarray:
    d = xgb.DMatrix(np.asarray(Xsel, dtype=np.float32, order="C"))
    D = len(boosters)
    out = np.empty((Xsel.shape[0], D), dtype=np.float32)
    for j, b in enumerate(boosters):
        out[:, j] = b.predict(d).astype(np.float32, copy=False)
    return out


def sigma_bg_from_border_mad(I_true: np.ndarray, stamp_pix: int, w: int = 2, floor: float = 1e-6) -> np.ndarray:
    N = I_true.shape[0]
    imgs = I_true.reshape(N, stamp_pix, stamp_pix)
    mask = np.zeros((stamp_pix, stamp_pix), dtype=bool)
    mask[:w, :] = True
    mask[-w:, :] = True
    mask[:, :w] = True
    mask[:, -w:] = True
    b = imgs[:, mask]
    med = np.median(b, axis=1)
    mad = np.median(np.abs(b - med[:, None]), axis=1)
    sig = 1.4826 * mad
    sig = np.where(np.isfinite(sig), sig, floor)
    sig = np.maximum(sig, floor)
    return sig.astype(np.float32)


def edge_distance_from_peak(I_true_row: np.ndarray, stamp_pix: int) -> float:
    img = np.asarray(I_true_row, dtype=np.float32).reshape(stamp_pix, stamp_pix)
    iy, ix = np.unravel_index(int(np.nanargmax(img)), img.shape)
    d = min(ix, iy, stamp_pix - 1 - ix, stamp_pix - 1 - iy)
    return float(d)


def pick_top(df: pd.DataFrame, cond: np.ndarray, n: int, used: set) -> List[int]:
    sub = df.loc[cond].sort_values("chi2nu", ascending=False)
    out: List[int] = []
    for i in sub.index.tolist():
        tidx = int(df.loc[i, "test_index"])
        if tidx in used:
            continue
        out.append(i)
        used.add(tidx)
        if len(out) >= n:
            break
    return out


def plot_source_panel(
    out_png: Path,
    I_true: np.ndarray,
    I_pred: np.ndarray,
    Z: np.ndarray,
    title: str,
    subtitle: str,
    cmap_image: str,
    cmap_resid: str,
) -> None:
    t = np.asarray(I_true).reshape(20, 20)
    p = np.asarray(I_pred).reshape(20, 20)
    z = np.asarray(Z).reshape(20, 20)
    # Nonnegative display range for stamp intensity: black background, white flux.
    vmax = np.nanpercentile(np.maximum(np.concatenate([t.reshape(-1), p.reshape(-1)]), 0.0), 99.5)
    vmax = max(vmax, 1e-6)
    zmax = np.nanpercentile(np.abs(z), 99.0)
    zmax = max(zmax, 1e-6)

    fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.4))
    im0 = ax[0].imshow(np.maximum(t, 0.0), origin="lower", cmap=cmap_image, vmin=0.0, vmax=vmax)
    ax[0].set_title("TRUE")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.02)
    im1 = ax[1].imshow(np.maximum(p, 0.0), origin="lower", cmap=cmap_image, vmin=0.0, vmax=vmax)
    ax[1].set_title("PRED")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.02)
    im2 = ax[2].imshow(z, origin="lower", cmap=cmap_resid, vmin=-zmax, vmax=zmax)
    ax[2].set_title("(TRUE-PRED)/sigma")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.02)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    fig.suptitle(title + "\n" + subtitle, fontsize=9)
    _savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="base_v11_16d")
    ap.add_argument("--forensics_csv", default="report/model_decision/20260224_outlier_forensics_base_v11_noflux/influence_feature_table.csv")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--neighbor_col", default="neighbor_count_r8")
    ap.add_argument("--influence_col", default="influence_r20")
    ap.add_argument("--fornax_name", default="ERO-Fornax")
    ap.add_argument("--n_fornax_isolated", type=int, default=25)
    ap.add_argument("--n_nonfornax_isolated", type=int, default=25)
    ap.add_argument("--n_high_asym", type=int, default=20)
    ap.add_argument("--n_low_conc", type=int, default=20)
    ap.add_argument("--n_edge", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--cmap_image", default="gray")
    ap.add_argument("--cmap_resid", default="RdBu_r")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    run_root = base_dir / "output" / "ml_runs" / str(args.run_name)
    model_dir = run_root / "models"
    man = load_manifest(run_root)
    n_test = int(man["n_test"])
    n_feat = int(man["n_features"])
    D = int(man["D"])
    stamp_pix = int(man["stamp_pix"])

    fcsv = Path(args.forensics_csv)
    if not fcsv.is_absolute():
        fcsv = base_dir / fcsv
    if not fcsv.exists():
        raise FileNotFoundError(fcsv)

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base_dir / "report" / "model_decision" / f"20260224_visual_residual_audit_{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "outlier_visual_audit"
    img_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fcsv, low_memory=False)
    req = {"test_index", "source_id", "field_tag", "chi2nu", "m_asymmetry_180", "m_concentration_r2_r6"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise RuntimeError(f"forensics csv missing: {miss}")
    if args.neighbor_col not in df.columns:
        raise RuntimeError(f"neighbor column not found: {args.neighbor_col}")
    if args.influence_col not in df.columns:
        raise RuntimeError(f"influence column not found: {args.influence_col}")

    df["test_index"] = pd.to_numeric(df["test_index"], errors="coerce")
    df["chi2nu"] = pd.to_numeric(df["chi2nu"], errors="coerce")
    df["m_asymmetry_180"] = pd.to_numeric(df["m_asymmetry_180"], errors="coerce")
    df["m_concentration_r2_r6"] = pd.to_numeric(df["m_concentration_r2_r6"], errors="coerce")
    df[args.neighbor_col] = pd.to_numeric(df[args.neighbor_col], errors="coerce")
    df[args.influence_col] = pd.to_numeric(df[args.influence_col], errors="coerce")
    if "p_nonpsf_gate" in df.columns:
        df["p_nonpsf_gate"] = pd.to_numeric(df["p_nonpsf_gate"], errors="coerce")
    else:
        df["p_nonpsf_gate"] = np.nan
    df = df.dropna(subset=["test_index", "chi2nu"]).copy()
    df["test_index"] = df["test_index"].astype(int)
    df = df[(df["test_index"] >= 0) & (df["test_index"] < n_test)].copy()

    # Outlier set by q99
    thr99 = _q(df["chi2nu"].to_numpy(dtype=float), 0.99)
    df["is_outlier"] = df["chi2nu"] >= thr99
    out_df = df[df["is_outlier"]].copy()

    # Load arrays for outlier-derived extra features (edge distance)
    Xt = np.memmap(Path(man["X_test_path"]), dtype="float32", mode="r", shape=(n_test, n_feat))
    Ys = np.memmap(Path(man["Yshape_test_path"]), dtype="float32", mode="r", shape=(n_test, D))
    Yf = np.memmap(Path(man["Yflux_test_path"]), dtype="float32", mode="r", shape=(n_test,))

    out_idx = out_df["test_index"].to_numpy(dtype=int)
    F_true = np.power(10.0, np.asarray(Yf[out_idx], dtype=np.float32)).astype(np.float32)
    I_true_out = np.asarray(Ys[out_idx], dtype=np.float32) * F_true[:, None]
    edge_d = np.array([edge_distance_from_peak(I_true_out[i], stamp_pix=stamp_pix) for i in range(I_true_out.shape[0])], dtype=np.float32)
    out_df = out_df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    out_df["edge_dist_px"] = edge_d

    # subset thresholds (computed on outlier set)
    asym_thr = _q(out_df["m_asymmetry_180"].to_numpy(dtype=float), 0.90)
    conc_thr = _q(out_df["m_concentration_r2_r6"].to_numpy(dtype=float), 0.10)

    used: set = set()
    picks: List[Tuple[str, int]] = []
    picks += [("fornax_isolated", i) for i in pick_top(
        out_df,
        (out_df["field_tag"].astype(str) == str(args.fornax_name)) & (out_df[args.neighbor_col] <= 0),
        int(args.n_fornax_isolated),
        used,
    )]
    picks += [("nonfornax_isolated", i) for i in pick_top(
        out_df,
        (out_df["field_tag"].astype(str) != str(args.fornax_name)) & (out_df[args.neighbor_col] <= 0),
        int(args.n_nonfornax_isolated),
        used,
    )]
    picks += [("high_asymmetry", i) for i in pick_top(
        out_df,
        (out_df["m_asymmetry_180"] >= asym_thr),
        int(args.n_high_asym),
        used,
    )]
    picks += [("low_concentration", i) for i in pick_top(
        out_df,
        (out_df["m_concentration_r2_r6"] <= conc_thr),
        int(args.n_low_conc),
        used,
    )]
    picks += [("edge_case", i) for i in pick_top(
        out_df,
        (out_df["edge_dist_px"] <= 2.0),
        int(args.n_edge),
        used,
    )]

    if len(picks) == 0:
        raise RuntimeError("No rows selected for audit.")

    sel_rows = []
    for subset, i in picks:
        r = out_df.loc[i].copy()
        r["subset"] = subset
        sel_rows.append(r)
    sel = pd.DataFrame(sel_rows).reset_index(drop=True)
    sel["audit_rank"] = np.arange(len(sel), dtype=int)

    sel_idx = sel["test_index"].to_numpy(dtype=int)
    Xsel = np.asarray(Xt[sel_idx], dtype=np.float32)
    Yshape_true = np.asarray(Ys[sel_idx], dtype=np.float32)
    Yflux_true = np.asarray(Yf[sel_idx], dtype=np.float32)
    F_true_sel = np.power(10.0, Yflux_true).astype(np.float32)
    I_true = Yshape_true * F_true_sel[:, None]

    boosters = load_shape_models(model_dir, D=D)
    pred_shape = predict_shape_selected(boosters, Xsel)
    I_pred = pred_shape * F_true_sel[:, None]  # true-flux mode

    sigma_bg = sigma_bg_from_border_mad(I_true, stamp_pix=stamp_pix, w=2, floor=1e-6)
    sig2 = (sigma_bg[:, None] ** 2) + float(args.alpha) * np.maximum(I_true, 0.0) + (float(args.sigma0) ** 2)
    sig2 = np.maximum(sig2, 1e-30)
    Z = (I_true - I_pred) / np.sqrt(sig2)

    # export per-source panels + summary table
    out_rows = []
    for i in range(len(sel)):
        r = sel.loc[i]
        sid = str(r.get("source_id", "NA"))
        field = str(r.get("field_tag", "NA"))
        subset = str(r.get("subset", "NA"))
        chi2 = float(r["chi2nu"])
        pnon = float(r.get("p_nonpsf_gate", np.nan))
        nnb = float(r.get(args.neighbor_col, np.nan))
        infl = float(r.get(args.influence_col, np.nan))
        ed = float(r.get("edge_dist_px", np.nan))
        png_name = f"{int(r['audit_rank']):03d}_{subset}_sid{sid}.png"
        png_path = img_dir / png_name

        t1 = f"{subset} | source_id={sid} | field={field}"
        t2 = f"chi2={chi2:.4g} | p_nonpsf={pnon:.3f} | neighbors={nnb:.0f} | influence={infl:.3g} | edge_dist={ed:.1f}px"
        plot_source_panel(
            png_path,
            I_true[i],
            I_pred[i],
            Z[i],
            t1,
            t2,
            cmap_image=str(args.cmap_image),
            cmap_resid=str(args.cmap_resid),
        )

        out_rows.append(
            {
                "audit_rank": int(r["audit_rank"]),
                "subset": subset,
                "test_index": int(r["test_index"]),
                "source_id": sid,
                "field_tag": field,
                "chi2nu": chi2,
                "p_nonpsf_gate": pnon,
                "neighbor_count": nnb,
                "influence": infl,
                "m_asymmetry_180": float(r.get("m_asymmetry_180", np.nan)),
                "m_concentration_r2_r6": float(r.get("m_concentration_r2_r6", np.nan)),
                "edge_dist_px": ed,
                "audit_png": str(png_path),
            }
        )

    sum_df = pd.DataFrame(out_rows).sort_values("audit_rank")
    sum_df.to_csv(out_dir / "audit_summary_table.csv", index=False)

    # manual classification template
    templ = sum_df[["audit_rank", "source_id", "field_tag", "subset", "chi2nu", "audit_png"]].copy()
    templ["failure_mode"] = ""
    templ["manual_notes"] = ""
    templ.to_csv(out_dir / "audit_failure_modes.csv", index=False)

    # Fornax vs non-Fornax sheet (top 10 each by chi2)
    f = sum_df[sum_df["field_tag"].astype(str) == str(args.fornax_name)].sort_values("chi2nu", ascending=False).head(10).copy()
    nf = sum_df[sum_df["field_tag"].astype(str) != str(args.fornax_name)].sort_values("chi2nu", ascending=False).head(10).copy()
    nrow = max(len(f), len(nf), 1)
    fig, axes = plt.subplots(nrow, 6, figsize=(14, 2.2 * nrow))
    if nrow == 1:
        axes = np.asarray([axes])

    def _get_arrays_by_rank(rank: int):
        ii = int(sum_df.index[sum_df["audit_rank"] == rank][0])
        return I_true[ii].reshape(stamp_pix, stamp_pix), I_pred[ii].reshape(stamp_pix, stamp_pix), Z[ii].reshape(stamp_pix, stamp_pix)

    for r in range(nrow):
        for c in range(6):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        if r < len(f):
            row = f.iloc[r]
            t, p, z = _get_arrays_by_rank(int(row["audit_rank"]))
            vmax = max(np.nanpercentile(np.maximum(np.concatenate([t.reshape(-1), p.reshape(-1)]), 0.0), 99.5), 1e-6)
            zmax = max(np.nanpercentile(np.abs(z), 99.0), 1e-6)
            axes[r, 0].imshow(np.maximum(t, 0.0), origin="lower", cmap=str(args.cmap_image), vmin=0.0, vmax=vmax)
            axes[r, 1].imshow(np.maximum(p, 0.0), origin="lower", cmap=str(args.cmap_image), vmin=0.0, vmax=vmax)
            axes[r, 2].imshow(z, origin="lower", cmap=str(args.cmap_resid), vmin=-zmax, vmax=zmax)
            axes[r, 0].set_ylabel(f"F{r+1}\nchi2={row['chi2nu']:.2g}", fontsize=8)
        if r < len(nf):
            row = nf.iloc[r]
            t, p, z = _get_arrays_by_rank(int(row["audit_rank"]))
            vmax = max(np.nanpercentile(np.maximum(np.concatenate([t.reshape(-1), p.reshape(-1)]), 0.0), 99.5), 1e-6)
            zmax = max(np.nanpercentile(np.abs(z), 99.0), 1e-6)
            axes[r, 3].imshow(np.maximum(t, 0.0), origin="lower", cmap=str(args.cmap_image), vmin=0.0, vmax=vmax)
            axes[r, 4].imshow(np.maximum(p, 0.0), origin="lower", cmap=str(args.cmap_image), vmin=0.0, vmax=vmax)
            axes[r, 5].imshow(z, origin="lower", cmap=str(args.cmap_resid), vmin=-zmax, vmax=zmax)
            axes[r, 3].set_ylabel(f"N{r+1}\nchi2={row['chi2nu']:.2g}", fontsize=8)

    ttl = ["Fornax TRUE", "Fornax PRED", "Fornax RESID", "Non-Fornax TRUE", "Non-Fornax PRED", "Non-Fornax RESID"]
    for c, t in enumerate(ttl):
        axes[0, c].set_title(t, fontsize=9)
    fig.suptitle("Visual residual audit: Fornax vs non-Fornax (top outliers)", fontsize=11)
    _savefig(out_dir / "audit_fornax_comparison.png")

    # markdown report
    lines = []
    lines.append("# Visual Residual Audit")
    lines.append("")
    lines.append(f"- Run: `{args.run_name}`")
    lines.append(f"- q99 threshold used (from forensics table): `{thr99:.6g}`")
    lines.append(f"- Total selected sources: `{len(sum_df)}`")
    lines.append("")
    lines.append("## Selection counts")
    cnt = sum_df["subset"].value_counts().to_dict()
    for k in ["fornax_isolated", "nonfornax_isolated", "high_asymmetry", "low_concentration", "edge_case"]:
        lines.append(f"- {k}: {int(cnt.get(k, 0))}")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- `audit_summary_table.csv`")
    lines.append(f"- `audit_failure_modes.csv`")
    lines.append(f"- `audit_fornax_comparison.png`")
    lines.append(f"- `outlier_visual_audit/`")
    (out_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # json summary
    js = {
        "run_name": str(args.run_name),
        "n_selected": int(len(sum_df)),
        "q99_threshold": float(thr99),
        "selection_counts": {k: int(v) for k, v in cnt.items()},
        "out_dir": str(out_dir),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(js, indent=2), encoding="utf-8")

    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
