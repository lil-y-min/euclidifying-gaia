#!/usr/bin/env python3
"""
Compute crowding/environment diagnostics from forensics output.

Primary output plot:
  forensics_05_crowding_error_correlation.png
"""

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
from scipy.spatial import cKDTree


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


def _unit_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    c = np.cos(dec)
    x = c * np.cos(ra)
    y = c * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])


def _radius_chord(arcsec: float) -> float:
    theta = np.deg2rad(float(arcsec) / 3600.0)
    return float(2.0 * np.sin(theta / 2.0))


def neighbor_counts_by_field(df: pd.DataFrame, radius_arcsec: float) -> np.ndarray:
    out = np.full(len(df), np.nan, dtype=np.float64)
    if "field_tag" not in df.columns:
        return out
    r = _radius_chord(radius_arcsec)

    for field, idx in df.groupby("field_tag", sort=False).groups.items():
        ii = np.asarray(list(idx), dtype=np.int64)
        sub = df.iloc[ii]
        ra = pd.to_numeric(sub.get("ra"), errors="coerce").to_numpy(dtype=np.float64)
        dec = pd.to_numeric(sub.get("dec"), errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(ra) & np.isfinite(dec)
        if np.sum(ok) == 0:
            continue

        xyz = _unit_xyz(ra[ok], dec[ok])
        tree = cKDTree(xyz)
        # query_ball_point includes self, so subtract 1
        n = np.array([len(v) - 1 for v in tree.query_ball_point(xyz, r)], dtype=np.float64)

        tmp = np.full(len(sub), np.nan, dtype=np.float64)
        tmp[ok] = n
        out[ii] = tmp
    return out


def summarize_crowding(
    df: pd.DataFrame,
    out_dir: Path,
    neigh_col: str,
    chi_col: str,
    radius_arcsec: float,
) -> Dict[str, object]:
    dd = df[[neigh_col, chi_col, "field_tag"]].copy()
    dd[neigh_col] = pd.to_numeric(dd[neigh_col], errors="coerce")
    dd[chi_col] = pd.to_numeric(dd[chi_col], errors="coerce")
    dd = dd.dropna(subset=[neigh_col, chi_col]).copy()
    dd = dd[dd[chi_col] > 0].copy()

    # deciles for crowding
    q = np.linspace(0.0, 1.0, 11)
    edges = np.unique(np.quantile(dd[neigh_col].to_numpy(dtype=float), q))
    if len(edges) <= 2:
        dd["crowd_bin"] = "all"
    else:
        dd["crowd_bin"] = pd.cut(dd[neigh_col], bins=edges, include_lowest=True, duplicates="drop")
        dd["crowd_bin"] = dd["crowd_bin"].astype(str)

    chi = dd[chi_col].to_numpy(dtype=float)
    thr99 = _q(chi, 0.99)
    dd["is_outlier"] = dd[chi_col] >= thr99

    by_bin = (
        dd.groupby("crowd_bin", dropna=False)
        .agg(
            n=(chi_col, "size"),
            crowd_mean=(neigh_col, "mean"),
            chi2_median=(chi_col, "median"),
            chi2_p90=(chi_col, lambda s: np.quantile(np.asarray(s, dtype=float), 0.90)),
            chi2_p99=(chi_col, lambda s: np.quantile(np.asarray(s, dtype=float), 0.99)),
            outlier_rate=("is_outlier", "mean"),
        )
        .reset_index()
    )
    by_bin.to_csv(out_dir / "crowding_bins_summary.csv", index=False)

    # Per-field correlation
    rows: List[Dict[str, object]] = []
    for f, s in dd.groupby("field_tag", dropna=False):
        a = s[neigh_col].to_numpy(dtype=float)
        b = np.log10(s[chi_col].to_numpy(dtype=float))
        ok = np.isfinite(a) & np.isfinite(b)
        if np.sum(ok) < 20:
            continue
        rho = pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman")
        rows.append({"field_tag": f, "n": int(np.sum(ok)), "spearman_rho_neighbor_vs_logchi2": float(rho)})
    corr_df = pd.DataFrame(rows).sort_values("spearman_rho_neighbor_vs_logchi2", ascending=False)
    corr_df.to_csv(out_dir / "crowding_field_correlations.csv", index=False)

    # Global correlation
    rho_g = pd.Series(dd[neigh_col]).corr(np.log10(dd[chi_col]), method="spearman")
    out = {
        "n_with_coords": int(len(dd)),
        "radius_arcsec": float(radius_arcsec),
        "chi2_q99_threshold": float(thr99),
        "global_spearman_rho_neighbor_vs_logchi2": float(rho_g),
    }
    (out_dir / "crowding_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--forensics_csv",
        default="report/model_decision/20260224_outlier_forensics_base_v11_noflux/forensics_full_joined.csv",
    )
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--radius_arcsec", type=float, default=8.0)
    ap.add_argument("--chi2_col", default="chi2nu")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    fcsv = Path(args.forensics_csv)
    if not fcsv.is_absolute():
        fcsv = base_dir / fcsv
    if not fcsv.exists():
        raise FileNotFoundError(fcsv)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else fcsv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fcsv, low_memory=False)
    if args.chi2_col not in df.columns:
        raise RuntimeError(f"Missing chi2 column: {args.chi2_col}")
    if ("ra" not in df.columns) or ("dec" not in df.columns):
        raise RuntimeError("forensics csv must include 'ra' and 'dec'")

    radius = float(args.radius_arcsec)
    neigh_col = f"neighbor_count_{radius:g}arcsec"
    df[neigh_col] = neighbor_counts_by_field(df, radius_arcsec=radius)
    df[f"log1p_{neigh_col}"] = np.log1p(pd.to_numeric(df[neigh_col], errors="coerce"))
    df.to_csv(out_dir / "forensics_with_environment.csv", index=False)

    summary = summarize_crowding(
        df,
        out_dir=out_dir,
        neigh_col=neigh_col,
        chi_col=args.chi2_col,
        radius_arcsec=radius,
    )

    # Plot: crowding vs error heatmap
    x = pd.to_numeric(df[neigh_col], errors="coerce").to_numpy(dtype=float)
    yraw = pd.to_numeric(df[args.chi2_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(yraw) & (yraw > 0)
    x = x[ok]
    y = np.log10(yraw[ok])
    thr = float(summary["chi2_q99_threshold"])
    thr_log = np.log10(thr) if (np.isfinite(thr) and thr > 0) else np.nan

    plt.figure(figsize=(8.5, 6.0))
    hb = plt.hexbin(
        x,
        y,
        gridsize=55,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    cb = plt.colorbar(hb)
    cb.set_label("log10(count)")
    if np.isfinite(thr_log):
        plt.axhline(thr_log, color="red", linestyle="--", linewidth=1.2, label="q99 threshold")
    plt.xlabel(f"Neighbor count within {radius:g} arcsec (same field)")
    plt.ylabel("log10(chi2nu)")
    plt.title("Crowding vs reconstruction error")
    plt.legend(loc="upper right")
    _savefig(out_dir / "forensics_05_crowding_error_correlation.png")

    # Plot: outlier rate by crowding decile
    b = pd.read_csv(out_dir / "crowding_bins_summary.csv")
    plt.figure(figsize=(9.0, 4.8))
    plt.plot(np.arange(len(b)), b["outlier_rate"], marker="o")
    plt.xticks(np.arange(len(b)), b["crowd_bin"], rotation=45, ha="right")
    plt.ylabel("Outlier rate")
    plt.xlabel("Crowding bin")
    plt.title("Outlier rate vs crowding bin")
    _savefig(out_dir / "forensics_06_outlier_rate_by_crowding_bin.png")

    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
