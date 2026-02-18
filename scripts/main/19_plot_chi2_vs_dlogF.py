"""
19_plot_chi2_vs_dlogF.py
=======================

Reads:
  plots/ml_runs/<RUN>/patch_17bis_field_sigma/chi2nu_field_values.csv

Writes:
  plots/ml_runs/<RUN>/patch_17bis_field_sigma/diag_flux/
    - abs_dlogF_hist.png
    - chi2nu_vs_abs_dlogF_scatter.png
    - chi2nu_vs_abs_dlogF_binned_median.png
    - chi2nu_vs_abs_dlogF_scatter_by_field.png
    - chi2nu_vs_abs_dlogF_perfield_medians_top6.png
    - field_codebook.txt

No pandas required.

Run:
  python /data/yn316/Codes/scripts/19_plot_chi2_vs_dlogF.py --run ml_xgb_pixelsflux_8d_augtrain
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def encode_categories(values):
    """Map categorical strings -> integer codes [0..K-1] in stable sorted order."""
    cats = sorted(set(values))
    m = {c: i for i, c in enumerate(cats)}
    codes = np.array([m[v] for v in values], dtype=np.int32)
    return codes, cats


def save_field_codebook(out_path: Path, categories: list[str]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(categories):
            f.write(f"{i}\t{c}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run name")
    ap.add_argument("--max_points", type=int, default=200000, help="Subsample for scatter")
    ap.add_argument("--nbins", type=int, default=25, help="Bins for binned median plot")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]  # .../Codes
    patch_dir = base / "plots" / "ml_runs" / args.run / "patch_17bis_field_sigma"
    in_csv = patch_dir / "tables" / "chi2nu_field_values.csv"
    if not in_csv.exists():
        # Backward-compat: older layout had CSV directly under patch_17bis_field_sigma/
        in_csv = patch_dir / "chi2nu_field_values.csv"
    out_dir = patch_dir / "diag_flux"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_csv}")

    abs_dlogF = []
    chi2nu = []
    field_tag = []

    with open(in_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                d = float(row["dlogF"])
                c = float(row["chi2nu_field"])
            except Exception:
                continue

            ft = row.get("field_tag", "UNKNOWN") or "UNKNOWN"

            if not (np.isfinite(d) and np.isfinite(c) and c > 0):
                continue

            abs_dlogF.append(abs(d))
            chi2nu.append(c)
            field_tag.append(ft)

    abs_dlogF = np.asarray(abs_dlogF, dtype=np.float64)
    chi2nu = np.asarray(chi2nu, dtype=np.float64)
    field_tag = np.asarray(field_tag, dtype=object)

    if abs_dlogF.size == 0:
        raise RuntimeError("No usable rows found in CSV (abs_dlogF / chi2nu_field).")

    # Encode field tags -> integer codes for coloring
    field_code, field_names = encode_categories(field_tag.tolist())
    save_field_codebook(out_dir / "field_codebook.txt", field_names)

    print("Loaded rows:", abs_dlogF.size)
    print("Fields:", len(field_names))
    print("abs_dlogF range:", float(abs_dlogF.min()), float(abs_dlogF.max()))
    print("chi2nu range:", float(chi2nu.min()), float(chi2nu.max()))

    # --- histogram of |dlogF|
    plt.figure(figsize=(7.2, 4.6))
    plt.hist(abs_dlogF, bins=120)
    plt.xlabel("|dlogF| = |logF_pred - logF_true|")
    plt.ylabel("count")
    plt.title("Flux prediction error distribution")
    savefig(out_dir / "abs_dlogF_hist.png")

    # --- subsample for scatter plots
    rng = np.random.default_rng(args.seed)
    n = abs_dlogF.size
    if n > args.max_points:
        idx = rng.choice(n, size=args.max_points, replace=False)
        x = abs_dlogF[idx]
        y = chi2nu[idx]
        fc = field_code[idx]
    else:
        x = abs_dlogF
        y = chi2nu
        fc = field_code

    # --- scatter: log10(chi2nu) vs |dlogF|
    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(x, np.log10(y), s=2)
    plt.xlabel("|dlogF|")
    plt.ylabel("log10(chi2nu_field)")
    plt.title("Flux error vs chi^2 (scatter)")
    savefig(out_dir / "chi2nu_vs_abs_dlogF_scatter.png")

    # --- NEW: scatter colored by field
    plt.figure(figsize=(7.6, 5.6))
    sc = plt.scatter(x, np.log10(y), c=fc, s=3)
    plt.xlabel("|dlogF|")
    plt.ylabel("log10(chi2nu_field)")
    plt.title("Flux error vs chi^2 (colored by field)")
    cb = plt.colorbar(sc)
    cb.set_label("field code (see field_codebook.txt)")
    savefig(out_dir / "chi2nu_vs_abs_dlogF_scatter_by_field.png")

    # --- binned median trend: log10(chi2nu) vs |dlogF| (global)
    nb = int(args.nbins)
    bins = np.linspace(np.min(abs_dlogF), np.max(abs_dlogF), nb + 1)
    xmid = 0.5 * (bins[:-1] + bins[1:])

    med = np.full(nb, np.nan, dtype=np.float64)
    p16 = np.full(nb, np.nan, dtype=np.float64)
    p84 = np.full(nb, np.nan, dtype=np.float64)

    for i in range(nb):
        m = (abs_dlogF >= bins[i]) & (abs_dlogF < bins[i + 1])
        if np.sum(m) < 10:
            continue
        v = np.log10(chi2nu[m])
        v = v[np.isfinite(v)]
        if v.size < 10:
            continue
        med[i] = np.median(v)
        p16[i] = np.percentile(v, 16)
        p84[i] = np.percentile(v, 84)

    ok = np.isfinite(med)
    plt.figure(figsize=(7.4, 5.2))
    plt.plot(xmid[ok], med[ok], marker="o", linewidth=1)
    plt.fill_between(xmid[ok], p16[ok], p84[ok], alpha=0.25)
    plt.xlabel("|dlogF|")
    plt.ylabel("median log10(chi2nu_field) ± [16,84]")
    plt.title("Flux error vs chi^2 (binned median)")
    savefig(out_dir / "chi2nu_vs_abs_dlogF_binned_median.png")

    # --- OPTIONAL: per-field median curves (top 6 fields by count)
    counts = np.bincount(field_code, minlength=len(field_names))
    topN = 6
    top_fields = np.argsort(counts)[::-1][:topN]

    plt.figure(figsize=(8.2, 5.6))
    for fi in top_fields:
        mfield = (field_code == fi)
        if np.sum(mfield) < 200:
            continue
        med_f = np.full(nb, np.nan, dtype=np.float64)
        for i in range(nb):
            m = mfield & (abs_dlogF >= bins[i]) & (abs_dlogF < bins[i + 1])
            if np.sum(m) < 20:
                continue
            v = np.log10(chi2nu[m])
            v = v[np.isfinite(v)]
            if v.size < 10:
                continue
            med_f[i] = np.median(v)
        okf = np.isfinite(med_f)
        if np.sum(okf) >= 3:
            plt.plot(xmid[okf], med_f[okf], marker="o", linewidth=1, label=field_names[fi])

    plt.xlabel("|dlogF|")
    plt.ylabel("median log10(chi2nu_field)")
    plt.title("Per-field median chi^2 trend (top fields by count)")
    plt.legend(fontsize=8)
    savefig(out_dir / "chi2nu_vs_abs_dlogF_perfield_medians_top6.png")

    print("DONE.")
    print("Input :", in_csv)
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
