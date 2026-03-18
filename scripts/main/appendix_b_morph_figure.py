"""
Generate the morphology metric illustration figure for Appendix B.

Each row shows two synthetic 20x20 stamps: low and high value for that metric.
Metric values are computed from the same formulas used in script 69.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace, maximum_filter

# ─────────────────────────────────────────────────────────────────────────────
# Metric implementations (exact copies from 69_morph_metrics_xgb.py)
# ─────────────────────────────────────────────────────────────────────────────

def flux_centroid(stamp):
    s = np.clip(stamp, 0, None)
    total = s.sum()
    if total <= 0:
        return stamp.shape[0] / 2, stamp.shape[1] / 2
    yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
    return float((s * yy).sum() / total), float((s * xx).sum() / total)

def compute_concentration(stamp, cy, cx):
    r = np.sqrt((np.mgrid[:20, :20][0] - cy)**2 + (np.mgrid[:20, :20][1] - cx)**2).flatten()
    s = np.clip(stamp, 0, None).flatten()
    total = s.sum()
    if total <= 0: return np.nan
    order = np.argsort(r)
    r_s, cum = r[order], np.cumsum(s[order])
    i20 = min(np.searchsorted(cum, 0.20 * total), len(r_s) - 1)
    i80 = min(np.searchsorted(cum, 0.80 * total), len(r_s) - 1)
    r20, r80 = r_s[i20], r_s[i80]
    if r20 <= 0: return np.nan
    return float(5.0 * np.log10(r80 / r20))

def compute_ellipticity(stamp, cy, cx):
    s = np.clip(stamp, 0, None); total = s.sum()
    if total <= 0: return np.nan
    yy, xx = np.mgrid[:20, :20]
    dy, dx = yy - cy, xx - cx
    Qxx = float((s * dx**2).sum() / total)
    Qyy = float((s * dy**2).sum() / total)
    Qxy = float((s * dx * dy).sum() / total)
    tr, det = Qxx + Qyy, Qxx * Qyy - Qxy**2
    disc = np.sqrt(max(0.0, (tr / 2)**2 - det))
    lam1, lam2 = tr / 2 + disc, tr / 2 - disc
    if lam1 <= 0: return np.nan
    a, b = np.sqrt(max(0.0, lam1)), np.sqrt(max(0.0, lam2))
    if a <= 0: return np.nan
    return float(1.0 - b / a)

def compute_asymmetry_180(stamp):
    denom = 2.0 * np.sum(np.abs(stamp))
    if denom <= 0: return np.nan
    return float(np.sum(np.abs(stamp - stamp[::-1, ::-1])) / denom)

def compute_mirror_asymmetry(stamp):
    denom = 2.0 * np.sum(np.abs(stamp))
    if denom <= 0: return np.nan
    a_lr = float(np.sum(np.abs(stamp - stamp[:, ::-1])) / denom)
    a_tb = float(np.sum(np.abs(stamp - stamp[::-1, :])) / denom)
    return 0.5 * (a_lr + a_tb)

def compute_gini(stamp, bg=0.0):
    s = np.sort(np.clip((stamp - bg).flatten(), 0, None))
    total = s.sum()
    if total <= 0: return np.nan
    n = len(s); idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(idx, s) / (n * total)) - (n + 1.0) / n)

def compute_m20(stamp, cy, cx, bg=0.0):
    s = np.clip(stamp - bg, 0, None); total = s.sum()
    if total <= 0: return np.nan
    yy, xx = np.mgrid[:20, :20]
    r2 = (yy - cy)**2 + (xx - cx)**2
    M_total = float((s * r2).sum())
    if M_total <= 0: return np.nan
    flat_s, flat_r2 = s.flatten(), r2.flatten()
    order = np.argsort(flat_s)[::-1]
    cum = np.cumsum(flat_s[order])
    idx = min(np.searchsorted(cum, 0.20 * total), len(flat_s) - 1)
    M20 = float((flat_s[order[:idx+1]] * flat_r2[order[:idx+1]]).sum())
    if M20 <= 0: return np.nan
    return float(np.log10(M20 / M_total))

def compute_kurtosis(stamp, bg=0.0):
    from scipy import stats as scipy_stats
    s = (stamp - bg).flatten(); s = s[s > 0]
    if len(s) < 4: return np.nan
    return float(scipy_stats.kurtosis(s, fisher=True))

def compute_smoothness(stamp, sigma=1.0):
    denom = 2.0 * np.sum(np.abs(stamp))
    if denom <= 0: return np.nan
    return float(np.sum(np.abs(stamp - gaussian_filter(stamp, sigma))) / denom)

def compute_multipeak(stamp, sigma_smooth=1.0, k_mad=3.0, f_flux=0.01, min_sep=2.0):
    """Gaussian pre-smoothing + MAD threshold + greedy NMS (mirrors script 53)."""
    bg = np.percentile(stamp, 10)
    ip = np.clip(stamp - bg, 0, None)
    total = ip.sum()
    if total <= 0: return np.nan
    sm = gaussian_filter(ip, sigma=sigma_smooth)
    med = np.median(sm)
    mad = float(np.median(np.abs(sm - med))) + 1e-9
    thr = med + k_mad * mad
    local_max = maximum_filter(sm, size=3)
    cand = np.argwhere((sm == local_max) & (sm > thr) & (sm / (total + 1e-12) > f_flux))
    if len(cand) == 0: return 0.0
    vals = np.array([sm[y, x] for y, x in cand])
    order = np.argsort(vals)[::-1]
    cand, vals = cand[order], vals[order]
    keep = [0]
    for i in range(1, len(cand)):
        y, x = cand[i]
        if all(np.hypot(x - cand[k][1], y - cand[k][0]) >= min_sep for k in keep):
            keep.append(i)
    if len(keep) < 2: return 0.0
    return float(vals[keep[1]] / vals[keep[0]])

def compute_hf(stamp):
    denom = float((stamp**2).sum())
    if denom <= 0: return np.nan
    return float((laplace(stamp)**2).sum() / denom)

def all_metrics(stamp):
    cy, cx = flux_centroid(stamp)
    return {
        "concentration":   compute_concentration(stamp, cy, cx),
        "ellipticity":     compute_ellipticity(stamp, cy, cx),
        "asymmetry 180°":  compute_asymmetry_180(stamp),
        "mirror asymmetry": compute_mirror_asymmetry(stamp),
        "Gini":            compute_gini(stamp),
        "M20":             compute_m20(stamp, cy, cx),
        "kurtosis":        compute_kurtosis(stamp),
        "smoothness":      compute_smoothness(stamp),
        "multi-peak ratio": compute_multipeak(stamp),
        "HF artifact":     compute_hf(stamp),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic stamp generators
# ─────────────────────────────────────────────────────────────────────────────

def gauss2d(cx=10.0, cy=10.0, sx=1.0, sy=1.0, angle=0.0, amp=1.0):
    """Centre on integer pixel by default to avoid 4-way degenerate maxima."""
    yy, xx = np.mgrid[:20, :20]
    ca, sa = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    xr = (xx - cx) * ca + (yy - cy) * sa
    yr = -(xx - cx) * sa + (yy - cy) * ca
    z = amp * np.exp(-0.5 * (xr**2 / sx**2 + yr**2 / sy**2))
    return z / z.sum()

def uniform_disk(r=5.0):
    yy, xx = np.mgrid[:20, :20]
    z = ((xx - 9.5)**2 + (yy - 9.5)**2 < r**2).astype(float)
    return z / z.sum()

def two_gaussians(sep=5.0, ratio=0.5):
    """Use integer centres to ensure one unique maximum per component."""
    c1 = round(10.0 - sep / 2)
    c2 = round(10.0 + sep / 2)
    z = gauss2d(cx=c1, cy=10, sx=1.0, sy=1.0) + \
        gauss2d(cx=c2, cy=10, sx=1.0, sy=1.0) * ratio
    return z / z.sum()

def noisy_gauss(sigma_gauss=2.0, noise_amp=0.15, seed=42):
    rng = np.random.default_rng(seed)
    z = gauss2d(sx=sigma_gauss, sy=sigma_gauss)
    z = z + noise_amp * rng.standard_normal(z.shape) * z.max()
    z = np.clip(z, 0, None)
    return z / z.sum()

def off_center_clump(offset=(3, 3), clump_amp=0.4):
    z = gauss2d(sx=1.5, sy=1.5)
    z += clump_amp * gauss2d(cx=10 + offset[1], cy=10 + offset[0], sx=0.8, sy=0.8)
    return z / z.sum()

# ─────────────────────────────────────────────────────────────────────────────
# Define the 11 rows: (metric_name, low_stamp, high_stamp, formula)
# ─────────────────────────────────────────────────────────────────────────────

rows = [
    # name, low, high, formula label
    (
        "Gini",
        uniform_disk(r=5.5),
        gauss2d(sx=0.5, sy=0.5),
        r"$G = \frac{2\sum_k k\,s_k}{N\sum s_k} - \frac{N+1}{N}$",
    ),
    (
        "Kurtosis",
        gauss2d(sx=3.0, sy=3.0),
        gauss2d(sx=0.9, sy=0.9),
        r"$\kappa = \mu_4/\mu_2^2 - 3$",
    ),
    (
        "Smoothness",
        gauss2d(sx=2.5, sy=2.5),
        noisy_gauss(sigma_gauss=1.5, noise_amp=0.6, seed=7),
        r"$S = \frac{\sum|I - \tilde{I}|}{2\sum|I|}$",
    ),
    (
        "Concentration",
        uniform_disk(r=7.0),
        # core + diffuse halo: compact 20% + extended 80% maximises r80/r20
        (lambda: (lambda z: z / z.sum())(
            gauss2d(sx=0.8, sy=0.8) * 0.6 + gauss2d(sx=4.5, sy=4.5) * 0.4))(),
        r"$C = 5\log_{10}(r_{80}/r_{20})$",
    ),
    (
        "Ellipticity",
        gauss2d(sx=1.5, sy=1.5),
        gauss2d(sx=5.0, sy=0.7, angle=30),
        r"$e = 1 - b/a$",
    ),

    (
        "Multi-peak ratio",
        gauss2d(sx=1.2, sy=1.2),
        two_gaussians(sep=6.0, ratio=0.7),
        r"$R = \tilde{f}_2^{\rm peak}/\tilde{f}_1^{\rm peak}$",
    ),
    (
        "M20",
        gauss2d(sx=1.2, sy=1.2),
        off_center_clump(offset=(4, 4), clump_amp=0.6),
        r"$M_{20} = \log_{10}(M_{20\%}/M_{\rm tot})$",
    ),
    (
        r"Asymmetry 180°",
        gauss2d(cx=9.5, cy=9.5, sx=2.0, sy=2.0),   # centred on rotation axis
        off_center_clump(offset=(3, -3), clump_amp=0.8),
        r"$A_{180} = \frac{\sum|I - I^{180}|}{2\sum|I|}$",
    ),
    (
        "Mirror asymmetry",
        gauss2d(cx=9.5, cy=9.5, sx=2.0, sy=2.0),   # centred on reflection axis
        gauss2d(cx=7, cy=10, sx=1.5, sy=1.5),
        r"$A_m = \frac{\sum|I - I^{\rm flip}|}{2\sum|I|}$",
    ),
    (
        "HF artifact",
        gauss2d(sx=2.5, sy=2.5),
        noisy_gauss(sigma_gauss=1.0, noise_amp=1.2, seed=99),
        r"$H = \sum(\nabla^2 I)^2 / \sum I^2$",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Compute metric values to annotate
# ─────────────────────────────────────────────────────────────────────────────
metric_fn = {
    "Gini":             compute_gini,
    "Kurtosis":         compute_kurtosis,
    "Smoothness":       compute_smoothness,
    "Concentration":    lambda s: compute_concentration(s, *flux_centroid(s)),
    "Ellipticity":      lambda s: compute_ellipticity(s, *flux_centroid(s)),

    "Multi-peak ratio": compute_multipeak,
    "M20":              lambda s: compute_m20(s, *flux_centroid(s)),
    r"Asymmetry 180°":  compute_asymmetry_180,
    "Mirror asymmetry": compute_mirror_asymmetry,
    "HF artifact":      compute_hf,
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
n = len(rows)
N_COLS = 2                        # metric columns side by side
n_rows = int(np.ceil(n / N_COLS))

# Each metric column: [text(3) | low(1) | high(1)], separated by a spacer(0.25)
# Total grid cols: N_COLS * 3 + (N_COLS-1) spacers
w = [3.0, 1.0, 1.0]
wr = w + [0.25] + w  # = 7 cols for N_COLS=2

fig = plt.figure(figsize=(16.0, 2.0 * n_rows + 0.8))

from matplotlib.gridspec import GridSpec
gs = GridSpec(n_rows, len(wr), figure=fig,
              width_ratios=wr,
              hspace=0.30, wspace=0.06,
              left=0.02, right=0.98, top=0.97, bottom=0.02)

col_bg = "#f8f8f8"
# col offsets for each metric column: col 0 starts at 0, col 1 starts at 4
col_offsets = [0, 4]

def draw_metric(row, col_off, name, low, high, formula, fn, is_first_row):
    val_low  = fn(low)
    val_high = fn(high)

    # name + formula
    ax_text = fig.add_subplot(gs[row, col_off])
    ax_text.set_facecolor(col_bg)
    ax_text.text(0.97, 0.68, name,
                 transform=ax_text.transAxes,
                 ha="right", va="center", fontsize=18, fontweight="bold")
    ax_text.text(0.97, 0.30, formula,
                 transform=ax_text.transAxes,
                 ha="right", va="center", fontsize=18)
    ax_text.axis("off")

    # low stamp
    ax_lo = fig.add_subplot(gs[row, col_off + 1])
    ax_lo.imshow(low, cmap="Greys_r", origin="lower",
                 vmin=0, vmax=low.max(), aspect="equal", interpolation="nearest")
    vl = f"{val_low:.2f}" if val_low is not None and np.isfinite(val_low) else "—"
    ax_lo.set_title(f"low: {vl}", fontsize=13, pad=3, color="#333333")
    ax_lo.axis("off")
    if is_first_row:
        ax_lo.text(0.5, 1.22, "Low", transform=ax_lo.transAxes,
                   ha="center", va="bottom", fontsize=14, fontweight="bold",
                   color="#555555")

    # high stamp
    ax_hi = fig.add_subplot(gs[row, col_off + 2])
    ax_hi.imshow(high, cmap="Greys_r", origin="lower",
                 vmin=0, vmax=high.max(), aspect="equal", interpolation="nearest")
    vh = f"{val_high:.2f}" if val_high is not None and np.isfinite(val_high) else "—"
    ax_hi.set_title(f"high: {vh}", fontsize=13, pad=3, color="#333333")
    ax_hi.axis("off")
    if is_first_row:
        ax_hi.text(0.5, 1.22, "High", transform=ax_hi.transAxes,
                   ha="center", va="bottom", fontsize=14, fontweight="bold",
                   color="#555555")

for i, (name, low, high, formula) in enumerate(rows):
    fn = metric_fn[name]
    mc = i % N_COLS        # which metric column (0 or 1)
    r  = i // N_COLS       # which grid row
    draw_metric(r, col_offsets[mc], name, low, high, formula, fn, r == 0)

out = ("/data/yn316/Codes/report/phd-thesis-template-2.4/"
       "Appendix2/Figs/Raster/morph_metrics_illustration.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
