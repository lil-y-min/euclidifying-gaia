#!/usr/bin/env python3
"""
Bright-neighbor influence and environment sweep for outlier forensics.

Core diagnostic:
  influence_r = sum_j flux_proxy_j / (dist_ij^2 + eps2), j within radius

Outputs:
  influence_feature_table.csv
  influence_radius_summary.csv
  influence_deciles_<radius>.csv
  candidate_environment_feature_rankings.csv
  forensics_05_bright_influence_error_correlation.png
  forensics_07_influence_spearman_vs_radius.png
  forensics_08_outlier_rate_by_influence_decile.png
  forensics_09_fornax_vs_nonfornax_influence_error.png
  forensics_10_brightest_neighbor_distance_hist.png
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
    plt.savefig(path, dpi=175)
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


def _chord_to_arcsec(ch: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(ch, dtype=np.float64), 0.0, 2.0)
    theta = 2.0 * np.arcsin(0.5 * x)
    return np.rad2deg(theta) * 3600.0


def _effect_size(out: np.ndarray, ref: np.ndarray) -> float:
    xo = np.asarray(out, dtype=float)
    xr = np.asarray(ref, dtype=float)
    xo = xo[np.isfinite(xo)]
    xr = xr[np.isfinite(xr)]
    if xo.size < 10 or xr.size < 10:
        return float("nan")
    mo = float(np.mean(xo))
    mr = float(np.mean(xr))
    so = float(np.std(xo))
    sr = float(np.std(xr))
    sp = float(np.sqrt(max(1e-12, 0.5 * (so * so + sr * sr))))
    return (mo - mr) / sp


def compute_field_features(
    ra: np.ndarray,
    dec: np.ndarray,
    flux: np.ndarray,
    sigma_bg: np.ndarray,
    radii_arcsec: List[float],
    eps2_arcsec2: float,
) -> Dict[str, np.ndarray]:
    n = len(ra)
    out: Dict[str, np.ndarray] = {}
    xyz = _unit_xyz(ra, dec)
    tree = cKDTree(xyz)

    radii_sorted = sorted(float(r) for r in radii_arcsec)
    max_r = max(radii_sorted)
    max_ch = _radius_chord(max_r)
    neigh_max = tree.query_ball_point(xyz, r=max_ch)

    # Prep arrays
    for r in radii_sorted:
        out[f"influence_r{r:g}"] = np.full(n, np.nan, dtype=np.float64)
        out[f"neighbor_count_r{r:g}"] = np.full(n, np.nan, dtype=np.float64)
    out[f"dist_brightest_neighbor_r{max_r:g}"] = np.full(n, np.nan, dtype=np.float64)
    out[f"brightest_neighbor_flux_r{max_r:g}"] = np.full(n, np.nan, dtype=np.float64)
    out[f"background_gradient_r{max_r:g}"] = np.full(n, np.nan, dtype=np.float64)

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    for i in range(n):
        idx = np.asarray(neigh_max[i], dtype=np.int64)
        if idx.size == 0:
            continue

        # distances from i to neighbors in arcsec (on-sphere)
        ch = np.linalg.norm(xyz[idx] - xyz[i], axis=1)
        d_arc = _chord_to_arcsec(ch)
        nonself = d_arc > 0
        idx = idx[nonself]
        d_arc = d_arc[nonself]
        if idx.size == 0:
            # valid zero-neighbor case
            for r in radii_sorted:
                out[f"influence_r{r:g}"][i] = 0.0
                out[f"neighbor_count_r{r:g}"][i] = 0.0
            continue

        # radius-specific features
        for r in radii_sorted:
            m = d_arc <= r
            if not np.any(m):
                out[f"influence_r{r:g}"][i] = 0.0
                out[f"neighbor_count_r{r:g}"][i] = 0.0
                continue
            dj = d_arc[m]
            fj = flux[idx[m]]
            valid = np.isfinite(dj) & np.isfinite(fj)
            dj = dj[valid]
            fj = fj[valid]
            if dj.size == 0:
                out[f"influence_r{r:g}"][i] = 0.0
                out[f"neighbor_count_r{r:g}"][i] = 0.0
                continue
            infl = np.sum(fj / (dj * dj + eps2_arcsec2))
            out[f"influence_r{r:g}"][i] = float(infl)
            out[f"neighbor_count_r{r:g}"][i] = float(dj.size)

        # brightest neighbor within max radius
        mmax = d_arc <= max_r
        if np.any(mmax):
            dmax = d_arc[mmax]
            fmax = flux[idx[mmax]]
            ok = np.isfinite(dmax) & np.isfinite(fmax)
            dmax = dmax[ok]
            fmax = fmax[ok]
            if dmax.size > 0:
                j = int(np.argmax(fmax))
                out[f"brightest_neighbor_flux_r{max_r:g}"][i] = float(fmax[j])
                out[f"dist_brightest_neighbor_r{max_r:g}"][i] = float(dmax[j])

                # local background gradient proxy from neighbors
                # fit sigma_bg(offset_x, offset_y) = a*x + b*y + c
                nb_idx = idx[mmax][ok]
                sb = sigma_bg[nb_idx]
                ok2 = np.isfinite(sb)
                if np.sum(ok2) >= 5:
                    nb_idx = nb_idx[ok2]
                    sb = sb[ok2]
                    dra = (ra_rad[nb_idx] - ra_rad[i]) * np.cos(dec_rad[i]) * (180.0 / np.pi) * 3600.0
                    dde = (dec_rad[nb_idx] - dec_rad[i]) * (180.0 / np.pi) * 3600.0
                    A = np.column_stack([dra, dde, np.ones_like(dra)])
                    try:
                        coef, *_ = np.linalg.lstsq(A, sb, rcond=None)
                        g = float(np.sqrt(coef[0] ** 2 + coef[1] ** 2))
                        out[f"background_gradient_r{max_r:g}"][i] = g
                    except Exception:
                        pass

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--forensics_csv",
        default="report/model_decision/20260224_outlier_forensics_base_v11_noflux/forensics_full_joined.csv",
    )
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--radii_arcsec", default="2,4,8,12,20")
    ap.add_argument("--eps2_arcsec2", type=float, default=1e-6)
    ap.add_argument("--flux_proxy_col", default="logF_true", help="Log-flux proxy column used only for neighbor influence diagnostics.")
    ap.add_argument("--chi2_col", default="chi2nu")
    ap.add_argument("--fornax_name", default="ERO-Fornax")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    fcsv = Path(args.forensics_csv)
    if not fcsv.is_absolute():
        fcsv = base_dir / fcsv
    if not fcsv.exists():
        raise FileNotFoundError(fcsv)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else fcsv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    radii = [float(x.strip()) for x in str(args.radii_arcsec).split(",") if str(x).strip()]
    radii = sorted(set(radii))
    if len(radii) == 0:
        raise RuntimeError("No valid radii provided.")
    r_plot = max(radii)

    df = pd.read_csv(fcsv, low_memory=False)
    for c in ["ra", "dec", args.chi2_col, "sigma_bg", args.flux_proxy_col]:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df[args.chi2_col] = pd.to_numeric(df[args.chi2_col], errors="coerce")
    df["sigma_bg"] = pd.to_numeric(df["sigma_bg"], errors="coerce")
    df[args.flux_proxy_col] = pd.to_numeric(df[args.flux_proxy_col], errors="coerce")

    # Flux proxy only for neighbor influence diagnostics
    flux = np.power(10.0, np.clip(df[args.flux_proxy_col].to_numpy(dtype=np.float64), -6.0, 8.0))
    # winsorize to reduce extreme leverage
    fcap = _q(flux[np.isfinite(flux)], 0.999)
    if np.isfinite(fcap) and fcap > 0:
        flux = np.clip(flux, 0.0, fcap)
    df["_flux_proxy_diag"] = flux

    # Compute features per field to avoid cross-field artificial neighbors
    feat_cols: Dict[str, np.ndarray] = {}
    for r in radii:
        feat_cols[f"influence_r{r:g}"] = np.full(len(df), np.nan, dtype=np.float64)
        feat_cols[f"neighbor_count_r{r:g}"] = np.full(len(df), np.nan, dtype=np.float64)
    feat_cols[f"dist_brightest_neighbor_r{r_plot:g}"] = np.full(len(df), np.nan, dtype=np.float64)
    feat_cols[f"brightest_neighbor_flux_r{r_plot:g}"] = np.full(len(df), np.nan, dtype=np.float64)
    feat_cols[f"background_gradient_r{r_plot:g}"] = np.full(len(df), np.nan, dtype=np.float64)

    if "field_tag" not in df.columns:
        raise RuntimeError("forensics csv must include field_tag")

    for _, idx in df.groupby("field_tag", sort=False).groups.items():
        ii = np.asarray(list(idx), dtype=np.int64)
        sub = df.iloc[ii]
        ra = sub["ra"].to_numpy(dtype=np.float64)
        dec = sub["dec"].to_numpy(dtype=np.float64)
        fl = sub["_flux_proxy_diag"].to_numpy(dtype=np.float64)
        sb = sub["sigma_bg"].to_numpy(dtype=np.float64)
        ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(fl) & np.isfinite(sb)
        if np.sum(ok) < 3:
            continue
        tmp = compute_field_features(
            ra=ra[ok],
            dec=dec[ok],
            flux=fl[ok],
            sigma_bg=sb[ok],
            radii_arcsec=radii,
            eps2_arcsec2=float(args.eps2_arcsec2),
        )
        # write back
        pos = np.where(ok)[0]
        for k, arr in tmp.items():
            tgt = feat_cols[k]
            tgt[ii[pos]] = arr

    for k, arr in feat_cols.items():
        df[k] = arr
        if k.startswith("influence_r"):
            df[f"log10_{k}"] = np.log10(np.maximum(pd.to_numeric(df[k], errors="coerce"), 1e-20))

    # Outlier definition (q99)
    thr99 = _q(df[args.chi2_col].to_numpy(dtype=float), 0.99)
    df["is_outlier"] = pd.to_numeric(df[args.chi2_col], errors="coerce") >= thr99
    df.to_csv(out_dir / "influence_feature_table.csv", index=False)

    # Radius sweep summary
    rows = []
    ylog = np.log10(np.maximum(pd.to_numeric(df[args.chi2_col], errors="coerce").to_numpy(dtype=float), 1e-20))
    for r in radii:
        x = pd.to_numeric(df[f"log10_influence_r{r:g}"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(ylog)
        if np.sum(m) < 30:
            continue
        rho = pd.Series(x[m]).corr(pd.Series(ylog[m]), method="spearman")
        # outlier lift top decile vs bottom decile
        z = x[m]
        out = df["is_outlier"].to_numpy(dtype=bool)[m]
        q10 = np.quantile(z, 0.10)
        q90 = np.quantile(z, 0.90)
        rate_lo = float(np.mean(out[z <= q10])) if np.any(z <= q10) else float("nan")
        rate_hi = float(np.mean(out[z >= q90])) if np.any(z >= q90) else float("nan")
        lift = float(rate_hi / max(1e-12, rate_lo)) if np.isfinite(rate_hi) and np.isfinite(rate_lo) else float("nan")
        rows.append(
            {
                "radius_arcsec": float(r),
                "spearman_rho_log_influence_vs_log_chi2": float(rho),
                "outlier_rate_bottom_decile": rate_lo,
                "outlier_rate_top_decile": rate_hi,
                "outlier_rate_lift_top_vs_bottom_decile": lift,
            }
        )
    rs = pd.DataFrame(rows).sort_values("radius_arcsec")
    rs.to_csv(out_dir / "influence_radius_summary.csv", index=False)

    # Choose plotting radius (max by default)
    r = r_plot
    x_raw = pd.to_numeric(df[f"influence_r{r:g}"], errors="coerce").to_numpy(dtype=float)
    x = np.log10(np.maximum(x_raw, 1e-20))
    y = ylog
    m = np.isfinite(x) & np.isfinite(y)

    # 05: influence vs error correlation
    plt.figure(figsize=(8.6, 6.0))
    hb = plt.hexbin(x[m], y[m], gridsize=58, bins="log", mincnt=1, cmap="magma")
    cb = plt.colorbar(hb)
    cb.set_label("log10(count)")
    # binned median trend
    xb = x[m]
    yb = y[m]
    bins = np.quantile(xb, np.linspace(0, 1, 16))
    bins = np.unique(bins)
    if len(bins) > 3:
        mids = []
        meds = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mm = (xb >= b0) & (xb <= b1)
            if np.sum(mm) < 30:
                continue
            mids.append(0.5 * (b0 + b1))
            meds.append(np.median(yb[mm]))
        if len(mids) > 1:
            plt.plot(mids, meds, color="cyan", linewidth=1.8, label="median trend")
    if np.isfinite(thr99) and thr99 > 0:
        plt.axhline(np.log10(thr99), color="white", linestyle="--", linewidth=1.2, label="q99 threshold")
    plt.xlabel(f"log10( influence_r{r:g} )")
    plt.ylabel("log10(chi2nu)")
    plt.title(f"Bright-neighbor influence vs error (r={r:g}\")")
    plt.legend(loc="upper left")
    _savefig(out_dir / "forensics_05_bright_influence_error_correlation.png")

    # 07: spearman vs radius
    if len(rs) > 0:
        plt.figure(figsize=(7.6, 4.8))
        plt.plot(rs["radius_arcsec"], rs["spearman_rho_log_influence_vs_log_chi2"], marker="o")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xlabel("Radius (arcsec)")
        plt.ylabel("Spearman rho")
        plt.title("Influence correlation vs radius")
        _savefig(out_dir / "forensics_07_influence_spearman_vs_radius.png")

    # 08: outlier rate by influence decile (for chosen radius)
    d = pd.DataFrame({"x": x, "is_outlier": df["is_outlier"].to_numpy(dtype=bool)})
    d = d[np.isfinite(d["x"])].copy()
    d["decile"] = pd.qcut(d["x"], q=10, duplicates="drop")
    dec = (
        d.groupby("decile")
        .agg(n=("is_outlier", "size"), outlier_rate=("is_outlier", "mean"), x_mean=("x", "mean"))
        .reset_index()
    )
    dec.to_csv(out_dir / f"influence_deciles_r{r:g}.csv", index=False)
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(np.arange(len(dec)), dec["outlier_rate"], marker="o")
    plt.xticks(np.arange(len(dec)), [str(v) for v in dec["decile"]], rotation=45, ha="right")
    plt.ylabel("Outlier rate")
    plt.xlabel(f"log10 influence decile (r={r:g}\")")
    plt.title("Outlier rate by influence decile")
    _savefig(out_dir / "forensics_08_outlier_rate_by_influence_decile.png")

    # 09: Fornax vs non-Fornax panel
    is_fornax = (df["field_tag"].astype(str) == str(args.fornax_name)).to_numpy(dtype=bool)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for ax, sel, ttl in [
        (axes[0], is_fornax, f"{args.fornax_name}"),
        (axes[1], ~is_fornax, "Non-Fornax"),
    ]:
        mm = m & sel
        if np.sum(mm) >= 20:
            hb2 = ax.hexbin(x[mm], y[mm], gridsize=40, bins="log", mincnt=1, cmap="viridis")
            # simple slope line
            xx = x[mm]
            yy = y[mm]
            if np.sum(np.isfinite(xx) & np.isfinite(yy)) >= 20:
                pfit = np.polyfit(xx, yy, 1)
                xr = np.array([np.nanmin(xx), np.nanmax(xx)])
                yr = pfit[0] * xr + pfit[1]
                ax.plot(xr, yr, color="red", linewidth=1.5, label=f"slope={pfit[0]:+.3f}")
                ax.legend(loc="upper left")
        if np.isfinite(thr99) and thr99 > 0:
            ax.axhline(np.log10(thr99), color="white", linestyle="--", linewidth=1.0)
        ax.set_title(ttl)
        ax.set_xlabel(f"log10 influence r{r:g}")
    axes[0].set_ylabel("log10(chi2nu)")
    fig.suptitle("Influence vs error: Fornax vs non-Fornax")
    _savefig(out_dir / "forensics_09_fornax_vs_nonfornax_influence_error.png")

    # 10: distance-to-brightest-neighbor (outlier vs non-outlier)
    dist_col = f"dist_brightest_neighbor_r{r:g}"
    dist = pd.to_numeric(df[dist_col], errors="coerce").to_numpy(dtype=float)
    out_mask = df["is_outlier"].to_numpy(dtype=bool)
    m1 = np.isfinite(dist) & out_mask
    m0 = np.isfinite(dist) & (~out_mask)
    if np.sum(m1) >= 10 and np.sum(m0) >= 10:
        plt.figure(figsize=(8.2, 4.8))
        bins = np.linspace(0.0, np.nanquantile(dist[np.isfinite(dist)], 0.99), 50)
        plt.hist(dist[m0], bins=bins, density=True, alpha=0.55, label="non-outlier")
        plt.hist(dist[m1], bins=bins, density=True, alpha=0.55, label="outlier")
        plt.xlabel(f"Distance to brightest neighbor within {r:g}\" (arcsec)")
        plt.ylabel("Density")
        plt.title("Brightest-neighbor distance: outlier vs non-outlier")
        plt.legend()
        _savefig(out_dir / "forensics_10_brightest_neighbor_distance_hist.png")

    # Environment feature ranking
    rank_rows = []
    cand = [f"log10_influence_r{rv:g}" for rv in radii] + [dist_col, f"brightest_neighbor_flux_r{r:g}", f"background_gradient_r{r:g}"]
    for c in cand:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        e = _effect_size(v[out_mask], v[~out_mask])
        # correlation with error
        mm = np.isfinite(v) & np.isfinite(ylog)
        rho = float(pd.Series(v[mm]).corr(pd.Series(ylog[mm]), method="spearman")) if np.sum(mm) >= 20 else float("nan")
        rank_rows.append(
            {
                "feature": c,
                "effect_size_z_outlier_vs_non": float(e),
                "abs_effect_size_z": float(abs(e)) if np.isfinite(e) else float("nan"),
                "spearman_rho_vs_logchi2": rho,
            }
        )
    rank = pd.DataFrame(rank_rows).sort_values("abs_effect_size_z", ascending=False)
    rank.to_csv(out_dir / "candidate_environment_feature_rankings.csv", index=False)

    summary = {
        "n_rows": int(len(df)),
        "outlier_threshold_q99_chi2nu": float(thr99),
        "radii_arcsec": radii,
        "plot_radius_arcsec": float(r),
        "eps2_arcsec2": float(args.eps2_arcsec2),
        "flux_proxy_col_used_for_diagnostic_influence": str(args.flux_proxy_col),
    }
    (out_dir / "influence_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()

