#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BITS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BIT_NAMES = {
    1: "bit1_nonconverged",
    2: "bit2_noisearea_pathological",
    4: "bit4_low_morph_conf",
    8: "bit8_bad_cross_match",
    16: "bit16_edge_source",
    32: "bit32_flux_mismatch",
    64: "bit64_wsdb_corner_or_chip",
    128: "bit128_wsdb_artifact_or_mask",
    256: "bit256_seam_gradient",
    512: "bit512_zero_pixel",
    1024: "bit1024_manual_border_confirmed",
    2048: "bit2048_manual_remove",
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_source_lookup(dataset_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for meta in sorted(dataset_root.glob("*/metadata_16d.csv")):
        field = meta.parent.name
        use = [
            "source_id",
            "phot_g_mean_mag",
            "log10_snr",
            "npz_file",
            "index_in_file",
        ]
        df = pd.read_csv(meta, usecols=use, low_memory=False)
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
        df["phot_g_mean_mag"] = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
        df["log10_snr"] = pd.to_numeric(df["log10_snr"], errors="coerce")
        df["index_in_file"] = pd.to_numeric(df["index_in_file"], errors="coerce")
        df = df.dropna(subset=["source_id", "npz_file", "index_in_file"]).copy()
        df["source_id"] = df["source_id"].astype(np.int64)
        df["index_in_file"] = df["index_in_file"].astype(np.int64)
        df["field_tag_meta"] = field
        df["npz_relpath"] = df.apply(lambda r: f"{field}/{r['npz_file']}", axis=1)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return out


def add_bit_columns(df: pd.DataFrame, bits: List[int], flag_col: str = "quality_flag") -> pd.DataFrame:
    q = pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(np.int64)
    for b in bits:
        df[f"b{b}"] = ((q & b) != 0).astype(np.uint8)
    return df


def plot_flag_rate_1d(df: pd.DataFrame, bits: List[int], xcol: str, bins: np.ndarray, out_png: Path, title: str) -> None:
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x)
    x = x[ok]
    if x.size == 0:
        return
    plt.figure(figsize=(8, 5))
    xc = 0.5 * (bins[:-1] + bins[1:])
    for b in bits:
        y = df.loc[ok, f"b{b}"].to_numpy(dtype=float)
        idx = np.digitize(x, bins) - 1
        m = (idx >= 0) & (idx < len(bins) - 1)
        rate = np.full(len(bins) - 1, np.nan, dtype=float)
        for i in range(len(rate)):
            mm = m & (idx == i)
            if np.any(mm):
                rate[i] = float(np.mean(y[mm]))
        label = BIT_NAMES.get(b, f"bit{b}")
        plt.plot(xc, rate, marker="o", ms=3, lw=1.4, label=label)
    plt.xlabel(xcol)
    plt.ylabel("Flag rate")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    _savefig(out_png)


def plot_overlap(df: pd.DataFrame, bits: List[int], out_png: Path, out_csv: Path) -> None:
    mat = np.zeros((len(bits), len(bits)), dtype=float)
    n = len(df)
    for i, bi in enumerate(bits):
        xi = df[f"b{bi}"].to_numpy(dtype=bool)
        for j, bj in enumerate(bits):
            xj = df[f"b{bj}"].to_numpy(dtype=bool)
            mat[i, j] = float(np.sum(xi & xj)) / float(max(n, 1))
    ov = pd.DataFrame(mat, index=[f"b{b}" for b in bits], columns=[f"b{b}" for b in bits])
    ov.to_csv(out_csv, index=True)

    plt.figure(figsize=(max(6, 0.6 * len(bits)), max(5, 0.55 * len(bits))))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar(label="Intersection fraction (of all rows)")
    plt.xticks(range(len(bits)), [f"b{b}" for b in bits], rotation=45, ha="right")
    plt.yticks(range(len(bits)), [f"b{b}" for b in bits])
    plt.title("Bit Overlap Matrix")
    for i in range(len(bits)):
        for j in range(len(bits)):
            plt.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    _savefig(out_png)


def plot_metric_hist(df: pd.DataFrame, col: str, out_png: Path, vlines: List[float], title: str, logx: bool = False) -> None:
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    plt.figure(figsize=(7, 4))
    if logx:
        x = x[x > 0]
        if x.size == 0:
            return
        plt.hist(np.log10(x), bins=80, color="#4C78A8", alpha=0.8)
        for v in vlines:
            if np.isfinite(v) and v > 0:
                plt.axvline(np.log10(v), color="crimson", ls="--", lw=1)
        plt.xlabel(f"log10({col})")
    else:
        plt.hist(x, bins=80, color="#4C78A8", alpha=0.8)
        for v in vlines:
            if np.isfinite(v):
                plt.axvline(v, color="crimson", ls="--", lw=1)
        plt.xlabel(col)
    plt.ylabel("count")
    plt.title(title)
    _savefig(out_png)


def plot_hex(df: pd.DataFrame, xcol: str, ycol: str, out_png: Path, title: str) -> None:
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) == 0:
        return
    plt.figure(figsize=(7, 5))
    plt.hexbin(x[m], y[m], gridsize=70, mincnt=1, bins="log", cmap="magma")
    plt.colorbar(label="log10(N)")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    _savefig(out_png)


def plot_spatial_flag_fraction(df: pd.DataFrame, bit: int, out_png: Path) -> None:
    ra = pd.to_numeric(df["ra_t"], errors="coerce").to_numpy(dtype=float)
    de = pd.to_numeric(df["dec_t"], errors="coerce").to_numpy(dtype=float)
    y = df[f"b{bit}"].to_numpy(dtype=float)
    m = np.isfinite(ra) & np.isfinite(de) & np.isfinite(y)
    if np.sum(m) == 0:
        return
    plt.figure(figsize=(8, 4.6))
    hb = plt.hexbin(ra[m], de[m], C=y[m], reduce_C_function=np.mean, gridsize=90, mincnt=8, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(hb, label=f"P(bit{bit}=1)")
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.title(f"Spatial flag fraction: bit{bit}")
    _savefig(out_png)


def make_montage(df: pd.DataFrame, bit: int, dataset_root: Path, out_png: Path, n_each: int = 25) -> None:
    rng = np.random.default_rng(123 + bit)
    mag_col = "phot_g_mean_mag"
    d = df.copy()
    d[mag_col] = pd.to_numeric(d[mag_col], errors="coerce")
    d = d[np.isfinite(d[mag_col])].copy()
    if len(d) == 0:
        return

    fg = d[d[f"b{bit}"] == 1].copy()
    bg = d[d[f"b{bit}"] == 0].copy()
    if len(fg) == 0 or len(bg) == 0:
        return

    fg = fg.sample(n=min(n_each, len(fg)), random_state=123 + bit)
    bg_pool = bg.copy()
    bg_sel = []
    for _, r in fg.iterrows():
        m = np.abs(bg_pool[mag_col].to_numpy(dtype=float) - float(r[mag_col]))
        if m.size == 0:
            break
        j = int(np.argmin(m))
        bg_sel.append(bg_pool.iloc[j])
        bg_pool = bg_pool.drop(index=bg_pool.index[j])
    if len(bg_sel) == 0:
        return
    bg = pd.DataFrame(bg_sel)

    sel = pd.concat([fg.assign(_grp="flagged"), bg.assign(_grp="unflagged")], ignore_index=True)
    sel = sel.sample(frac=1.0, random_state=42).reset_index(drop=True)
    sel["short_id"] = np.arange(1, len(sel) + 1, dtype=int)
    n = len(sel)
    ncols = 10
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.4), facecolor="black")
    axes = np.array(axes).reshape(nrows, ncols)
    cache: Dict[str, np.ndarray] = {}

    map_rows = []
    for i in range(nrows * ncols):
        ax = axes.flat[i]
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        r = sel.iloc[i]
        rel = str(r["npz_relpath"])
        idx = int(r["index_in_file"])
        npz_path = dataset_root / rel
        if rel not in cache:
            try:
                cache[rel] = np.load(npz_path)["X"]
            except Exception:
                ax.axis("off")
                continue
        X = cache[rel]
        if idx < 0 or idx >= X.shape[0]:
            ax.axis("off")
            continue
        im = X[idx].astype(float)
        vmin = np.percentile(im, 5)
        vmax = np.percentile(im, 99)
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        color = "cyan" if r["_grp"] == "flagged" else "orange"
        sid_txt = str(int(r["source_id"])) if pd.notna(r.get("source_id", np.nan)) else "na"
        grp_txt = "FLAG" if r["_grp"] == "flagged" else "KEEP"
        ax.set_title(
            f"{int(r['short_id']):03d} {grp_txt}\n"
            f"m={r['phot_g_mean_mag']:.1f}",
            fontsize=5.5,
            color=color,
        )
        map_rows.append(
            {
                "short_id": int(r["short_id"]),
                "group": str(r["_grp"]),
                "source_id": sid_txt,
                "phot_g_mean_mag": float(r["phot_g_mean_mag"]) if pd.notna(r["phot_g_mean_mag"]) else np.nan,
                "field_tag_meta": str(r.get("field_tag_meta", "")),
                "npz_relpath": str(r.get("npz_relpath", "")),
                "index_in_file": int(r["index_in_file"]) if pd.notna(r.get("index_in_file", np.nan)) else -1,
            }
        )

    fig.suptitle(f"Bit {bit} montage (flagged vs mag-matched unflagged)", color="white")
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, facecolor="black")
    plt.close()
    pd.DataFrame(map_rows).to_csv(out_png.with_name(out_png.stem + "_id_map.csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags_csv", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_montage_each", type=int, default=25)
    ap.add_argument("--bits", type=str, default="", help="Comma-separated bits to analyze. Default: all supported bits present in data.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    df = pd.read_csv(args.flags_csv, low_memory=False)
    if str(args.bits).strip():
        bits = [int(x.strip()) for x in str(args.bits).split(",") if x.strip()]
    else:
        bits = DEFAULT_BITS.copy()
    df = add_bit_columns(df, bits=bits, flag_col="quality_flag")

    lookup = build_source_lookup(Path(args.dataset_root))
    df = df.merge(lookup, on="source_id", how="left")

    # shared plots
    mag = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
    if np.isfinite(mag).any():
        bins_mag = np.linspace(float(np.nanpercentile(mag, 1)), float(np.nanpercentile(mag, 99)), 20)
        plot_flag_rate_1d(df, bits, "phot_g_mean_mag", bins_mag, out_dir / "shared_flag_rate_vs_mag.png", "Flag rate vs Gaia G mag")
    snr = pd.to_numeric(df["log10_snr"], errors="coerce")
    if np.isfinite(snr).any():
        bins_snr = np.linspace(float(np.nanpercentile(snr, 1)), float(np.nanpercentile(snr, 99)), 20)
        plot_flag_rate_1d(df, bits, "log10_snr", bins_snr, out_dir / "shared_flag_rate_vs_log10_snr.png", "Flag rate vs log10(SNR)")

    plot_overlap(df, bits, out_dir / "shared_bit_overlap_matrix.png", out_dir / "shared_bit_overlap_matrix.csv")

    # bit-specific
    # bit1
    if (1 in bits) and ("v_niter_model" in df.columns):
        q95 = df.groupby("field_tag")["v_niter_model"].quantile(0.95)
        q95.to_csv(out_dir / "bit1_niter_q95_by_field.csv")
        plot_metric_hist(df, "v_niter_model", out_dir / "bit1_hist_niter.png", [float(np.nanpercentile(pd.to_numeric(df["v_niter_model"], errors="coerce"), 95))], "Bit1 metric: v_niter_model")
        if "phot_g_mean_mag" in df.columns:
            plot_hex(df, "phot_g_mean_mag", "v_niter_model", out_dir / "bit1_niter_vs_mag_hex.png", "v_niter_model vs Gaia G mag")
        plot_spatial_flag_fraction(df, 1, out_dir / "bit1_spatial_flag_fraction.png")

    # bit2
    if (2 in bits) and ("v_noisearea_model" in df.columns):
        plot_metric_hist(df, "v_noisearea_model", out_dir / "bit2_hist_noisearea_log.png", [5.0, 500.0], "Bit2 metric: v_noisearea_model", logx=True)
        if "phot_g_mean_mag" in df.columns:
            plot_hex(df, "phot_g_mean_mag", "v_noisearea_model", out_dir / "bit2_noisearea_vs_mag_hex.png", "v_noisearea_model vs Gaia G mag")
        plot_spatial_flag_fraction(df, 2, out_dir / "bit2_spatial_flag_fraction.png")

    # bit4
    if (4 in bits) and ("v_spread_model" in df.columns) and ("v_spreaderr_model" in df.columns):
        sp = pd.to_numeric(df["v_spread_model"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(df["v_spreaderr_model"], errors="coerce").to_numpy(dtype=float)
        sig = np.full(len(df), np.nan, dtype=float)
        good = np.isfinite(sp) & np.isfinite(se) & (np.abs(se) > 0)
        sig[good] = sp[good] / se[good]
        df["spread_sig"] = sig
        plot_metric_hist(df, "spread_sig", out_dir / "bit4_hist_spread_sig.png", [-2.0, 2.0], "Bit4 metric: spread_sig")
        if "phot_g_mean_mag" in df.columns:
            plot_hex(df, "phot_g_mean_mag", "spread_sig", out_dir / "bit4_spreadsig_vs_mag_hex.png", "spread_sig vs Gaia G mag")
        plot_spatial_flag_fraction(df, 4, out_dir / "bit4_spatial_flag_fraction.png")

    # bit32
    if (32 in bits) and ("v_flux_auto" in df.columns) and ("v_flux_model" in df.columns):
        fa = pd.to_numeric(df["v_flux_auto"], errors="coerce").to_numpy(dtype=float)
        fm = pd.to_numeric(df["v_flux_model"], errors="coerce").to_numpy(dtype=float)
        den = np.maximum(np.abs(fa), 1e-12)
        dfrac = np.abs(fa - fm) / den
        df["deltaF_frac"] = dfrac
        plot_metric_hist(df, "deltaF_frac", out_dir / "bit32_hist_deltaF_frac.png", [0.3], "Bit32 metric: |F_auto-F_model|/|F_auto|")
        if "phot_g_mean_mag" in df.columns:
            plot_hex(df, "phot_g_mean_mag", "deltaF_frac", out_dir / "bit32_deltaF_vs_mag_hex.png", "deltaF_frac vs Gaia G mag")
        plot_spatial_flag_fraction(df, 32, out_dir / "bit32_spatial_flag_fraction.png")

    # montages
    ds_root = Path(args.dataset_root)
    for b in bits:
        make_montage(df, b, ds_root, out_dir / f"bit{b}_montage_flagged_vs_unflagged.png", n_each=int(args.n_montage_each))

    # summary
    rows = []
    n = len(df)
    for b in bits:
        nb = int(df[f"b{b}"].sum())
        rows.append({"bit": b, "name": BIT_NAMES.get(b, f"bit{b}"), "n_flagged": nb, "frac_flagged": float(nb / max(n, 1))})
    pd.DataFrame(rows).to_csv(out_dir / "bit_summary.csv", index=False)
    print("DONE")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
