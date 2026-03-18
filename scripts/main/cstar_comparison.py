"""
c★ distribution comparison: ERO-matched vs Q1-matched DESI spectroscopic galaxies.

Distinguishes between:
  H1 — Population shift: Q1 galaxies more extended → CDF slope changes
  H2 — Calibration offset: photometric offset → CDF translates horizontally

Usage:
  python cstar_comparison.py                        # uses default paths
  python cstar_comparison.py ERO.csv Q1.csv OUT/    # custom paths

Output (in OUT_DIR):
  fig_cstar_comparison.pdf
  fig_cstar_comparison.png
  cstar_summary_stats.txt
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from scipy.stats import gaussian_kde

# ─── Configurable paths (edit here or pass as CLI args) ──────────────────────
DEFAULT_ERO_CSV = "/data/yn316/Codes/output/ml_runs/galaxy_desi_clf/desi_gaia_direct_ero.csv"
DEFAULT_Q1_CSV  = "/data/yn316/Codes/output/ml_runs/galaxy_desi_clf/desi_gaia_direct_q1.csv"
DEFAULT_OUT_DIR = "/data/yn316/Codes/deliverables"

# ─── Quality-cut flags ────────────────────────────────────────────────────────
G_MAG_MIN      = 17.0    # faint-end threshold (G >= 17)
REQUIRE_RUWE   = False   # True = only sources with valid RUWE
                         # NOTE: 66% of ERO DESI galaxies have NaN RUWE because
                         # Gaia PSF fitting failed on extended sources — this is
                         # a signal used by the classifier, NOT a noise source.
                         # Setting True would preferentially retain compact sources
                         # and bias the c★ comparison toward low values.

# ─── Figure aesthetics ────────────────────────────────────────────────────────
CLR_ERO = "#1f77b4"   # mpl blue
CLR_Q1  = "#ff7f0e"   # mpl orange
CSTAR_XMIN, CSTAR_XMAX = -0.5, 20.0
# NOTE: the original spec assumed [-0.5, 3.0] but DESI galaxies have c★ >> 1
# (mean ~6-7) because they ARE extended. p95 is ~15-18, so we use [-0.5, 20]
# to capture the bulk of both distributions (p1 to ~p95).

plt.rcParams.update({
    "font.family":        "serif",
    "mathtext.fontset":   "cm",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "font.size":          11,
})


# ─────────────────────────────────────────────────────────────────────────────
def load_and_filter(path: str, label: str) -> pd.Series:
    """Load CSV and return the filtered c★ Series."""
    df = pd.read_csv(path)

    # Require valid c★
    mask = df["feat_c_star"].notna()

    # G >= 17 cut
    mask &= df["phot_g_mean_mag"] >= G_MAG_MIN

    # Spectroscopic redshift available
    mask &= df["desi_z"].notna()

    # Optional: require valid RUWE
    if REQUIRE_RUWE:
        mask &= df["ruwe"].notna()

    filtered = df.loc[mask, "feat_c_star"]

    n_total  = len(df)
    n_kept   = mask.sum()
    n_nan_ruwe = df["ruwe"].isna().sum()
    print(
        f"[{label}] total={n_total}, G>=17 & valid c★={n_kept}, "
        f"NaN RUWE (excluded if REQUIRE_RUWE)={n_nan_ruwe}"
    )
    if n_kept < 30:
        warnings.warn(f"[{label}] only {n_kept} sources after cuts — results may be unreliable!")

    return filtered


def clip_to_range(s: pd.Series, xmin: float, xmax: float) -> pd.Series:
    """Clip values to plotting range for display (stats computed on full range)."""
    return s.clip(lower=xmin, upper=xmax)


def summary_stats(s: pd.Series) -> dict:
    return {
        "N":      len(s),
        "mean":   s.mean(),
        "median": s.median(),
        "std":    s.std(),
        "p10":    s.quantile(0.10),
        "p90":    s.quantile(0.90),
    }


def make_kde(s: pd.Series, xmin: float, xmax: float, n: int = 500):
    xs = np.linspace(xmin, xmax, n)
    bw = s.std() * len(s) ** (-1 / 5.0)   # Silverman's rule
    kde = gaussian_kde(s, bw_method=bw / s.std())
    ys = kde(xs)
    return xs, ys


def make_ecdf(s: pd.Series, xmin: float, xmax: float):
    arr = np.sort(s.values)
    cdf = np.arange(1, len(arr) + 1) / len(arr)
    # prepend/append for clean step
    arr = np.concatenate([[xmin], arr, [xmax]])
    cdf = np.concatenate([[0], cdf, [cdf[-1]]])
    return arr, cdf


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) == 4:
        ero_path, q1_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    elif len(sys.argv) == 1:
        ero_path, q1_path, out_dir = DEFAULT_ERO_CSV, DEFAULT_Q1_CSV, DEFAULT_OUT_DIR
    else:
        print(__doc__)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    cstar_ero = load_and_filter(ero_path, "ERO")
    cstar_q1  = load_and_filter(q1_path,  "Q1")

    # ── Statistics (full unclipped distributions) ─────────────────────────────
    st_ero = summary_stats(cstar_ero)
    st_q1  = summary_stats(cstar_q1)

    ks_stat, ks_pval = stats.ks_2samp(cstar_ero.values, cstar_q1.values)
    delta_mean = st_q1["mean"] - st_ero["mean"]

    # ── KDE and ECDF on clipped range ─────────────────────────────────────────
    cstar_ero_cl = clip_to_range(cstar_ero, CSTAR_XMIN, CSTAR_XMAX)
    cstar_q1_cl  = clip_to_range(cstar_q1,  CSTAR_XMIN, CSTAR_XMAX)

    xs_ero, ys_ero = make_kde(cstar_ero_cl, CSTAR_XMIN, CSTAR_XMAX)
    xs_q1,  ys_q1  = make_kde(cstar_q1_cl,  CSTAR_XMIN, CSTAR_XMAX)

    xe_ero, ye_ero = make_ecdf(cstar_ero_cl, CSTAR_XMIN, CSTAR_XMAX)
    xe_q1,  ye_q1  = make_ecdf(cstar_q1_cl,  CSTAR_XMIN, CSTAR_XMAX)

    # ── Build caption string ───────────────────────────────────────────────────
    caption = (
        f"Empirical $c^\\star$ distributions for Gaia-matched DESI galaxies in "
        f"the ERO (blue, $N={st_ero['N']}$) and Q1 (orange, $N={st_q1['N']}$) footprints. "
        f"Left: normalised densities with medians marked (dashed verticals). "
        f"Right: cumulative distributions. A pure horizontal shift of the "
        f"CDF would indicate a photometric calibration offset (H2); a "
        f"change in CDF slope would indicate a morphological population "
        f"shift (H1). $\\Delta\\bar{{c}}^\\star$ and KS $p$-value are annotated in the left panel."
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_kde, ax_cdf) = plt.subplots(1, 2, figsize=(7, 3.4),
                                          constrained_layout=True)

    # ─ LEFT: KDE ──────────────────────────────────────────────────────────────
    ax_kde.plot(xs_ero, ys_ero, color=CLR_ERO, lw=1.8, ls="-",
                label=f"ERO  ($N={st_ero['N']}$)")
    ax_kde.plot(xs_q1,  ys_q1,  color=CLR_Q1,  lw=1.8, ls="--",
                label=f"Q1   ($N={st_q1['N']}$)")

    # fill lightly
    ax_kde.fill_between(xs_ero, ys_ero, alpha=0.12, color=CLR_ERO)
    ax_kde.fill_between(xs_q1,  ys_q1,  alpha=0.12, color=CLR_Q1)

    # median lines
    for median, color, ls in [
        (st_ero["median"], CLR_ERO, "-"),
        (st_q1["median"],  CLR_Q1,  "--"),
    ]:
        m_clipped = np.clip(median, CSTAR_XMIN, CSTAR_XMAX)
        ax_kde.axvline(m_clipped, color=color, lw=1.0, ls=":", alpha=0.85)

    # annotation box
    pval_str = f"{ks_pval:.2e}" if ks_pval < 0.001 else f"{ks_pval:.4f}"
    annot = (
        f"$\\Delta\\bar{{c}}^\\star = {delta_mean:+.3f}$\n"
        f"KS $p = {pval_str}$"
    )
    ax_kde.text(0.97, 0.06, annot,
                transform=ax_kde.transAxes,
                ha="right", va="bottom", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))

    ax_kde.set_xlabel(r"$c^\star$")
    ax_kde.set_ylabel("Normalised density")
    ax_kde.set_xlim(CSTAR_XMIN, CSTAR_XMAX)
    ax_kde.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_kde.legend(fontsize=9.5, frameon=False, loc="upper right")

    # ─ RIGHT: CDF ─────────────────────────────────────────────────────────────
    ax_cdf.plot(xe_ero, ye_ero, color=CLR_ERO, lw=1.8, ls="-",
                drawstyle="steps-post", label="ERO")
    ax_cdf.plot(xe_q1,  ye_q1,  color=CLR_Q1,  lw=1.8, ls="--",
                drawstyle="steps-post", label="Q1")

    # median verticals (same as left)
    for median, color in [
        (st_ero["median"], CLR_ERO),
        (st_q1["median"],  CLR_Q1),
    ]:
        m_clipped = np.clip(median, CSTAR_XMIN, CSTAR_XMAX)
        ax_cdf.axvline(m_clipped, color=color, lw=1.0, ls=":", alpha=0.85)

    ax_cdf.set_xlabel(r"$c^\star$")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_xlim(CSTAR_XMIN, CSTAR_XMAX)
    ax_cdf.set_ylim(0, 1.03)
    ax_cdf.legend(fontsize=9.5, frameon=False)

    fig.suptitle("", fontsize=1)   # empty — caption goes in LaTeX

    # ── Save ──────────────────────────────────────────────────────────────────
    pdf_path = os.path.join(out_dir, "fig_cstar_comparison.pdf")
    png_path = os.path.join(out_dir, "fig_cstar_comparison.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)

    # ── Summary stats text file ────────────────────────────────────────────────
    # Interpret the result.
    # Decision rule:
    #   H2 (calibration offset) → CDF shifts horizontally (parallel curves), Δmean large
    #   H1 (population shift)   → CDF slope changes (different spread/shape), Δmean may be small
    # Use both KS p-value and relative Δmean/std as joint criteria.
    pooled_std = np.sqrt(0.5 * (st_ero["std"] ** 2 + st_q1["std"] ** 2))
    delta_mean_norm = abs(delta_mean) / pooled_std   # normalised shift
    shape_change = abs(st_q1["std"] - st_ero["std"]) / pooled_std  # relative std change

    if ks_pval >= 0.01:
        interp = "NO SIGNIFICANT DIFFERENCE: KS p >= 0.01; distributions are consistent."
    elif delta_mean_norm > 0.2 and shape_change < 0.15:
        # large mean shift, similar shape → pure offset
        interp = "LIKELY CALIBRATION OFFSET (H2): distributions shifted in mean without major shape change."
    elif delta_mean_norm > 0.2 and shape_change >= 0.15:
        # mean shift AND shape change → both effects present
        interp = ("MIXED: significant mean shift (ΔMEAN/σ={:.2f}) AND shape change (Δstd/σ={:.2f}). "
                  "Check CDF panel: parallel curves → H2, changing slope → H1."
                  ).format(delta_mean_norm, shape_change)
    else:
        interp = "LIKELY POPULATION SHIFT (H1): distributions differ in shape (KS p < 0.01) but mean shift is small."

    txt_path = os.path.join(out_dir, "cstar_summary_stats.txt")
    with open(txt_path, "w") as fh:
        fh.write("c★ distribution comparison: ERO vs Q1 DESI galaxies\n")
        fh.write("=" * 60 + "\n\n")

        for lbl, st in [("ERO", st_ero), ("Q1", st_q1)]:
            fh.write(f"{lbl}:\n")
            fh.write(f"  N            = {st['N']}\n")
            fh.write(f"  mean         = {st['mean']:.4f}\n")
            fh.write(f"  median       = {st['median']:.4f}\n")
            fh.write(f"  std          = {st['std']:.4f}\n")
            fh.write(f"  10th pctile  = {st['p10']:.4f}\n")
            fh.write(f"  90th pctile  = {st['p90']:.4f}\n\n")

        fh.write(f"KS statistic = {ks_stat:.4f}\n")
        fh.write(f"KS p-value   = {ks_pval:.4e}\n")
        fh.write(f"Δmean (Q1 - ERO) = {delta_mean:+.4f}\n\n")
        fh.write(f"Quality cuts applied:\n")
        fh.write(f"  G >= {G_MAG_MIN}\n")
        fh.write(f"  non-null feat_c_star\n")
        fh.write(f"  non-null desi_z\n")
        fh.write(f"  REQUIRE_RUWE = {REQUIRE_RUWE}\n")
        fh.write(f"  (NaN RUWE sources included: Gaia PSF fit failure is itself a morphology signal)\n\n")
        fh.write(f"Interpretation:\n  {interp}\n\n")
        fh.write(f"Caption:\n  {caption.replace(chr(10), ' ')}\n")

    print(f"Saved: {txt_path}")
    print(f"\nSummary:")
    print(f"  ERO N={st_ero['N']}, mean={st_ero['mean']:.3f}, median={st_ero['median']:.3f}")
    print(f"  Q1  N={st_q1['N']},  mean={st_q1['mean']:.3f}, median={st_q1['median']:.3f}")
    print(f"  Δmean = {delta_mean:+.3f}")
    print(f"  KS: stat={ks_stat:.4f}, p={ks_pval:.4e}")
    print(f"  → {interp}")


if __name__ == "__main__":
    main()
