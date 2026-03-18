"""
Generate the IPD harmonic illustration figure for Appendix A.

Three-panel figure:
  Left   – 2D elliptical source with six coloured scan-direction vectors
  Middle – stacked 1D along-scan profiles, one per scan angle
  Right  – IPD harmonic residual g(theta) = A cos(2theta - phi) vs scan angle
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── source parameters ────────────────────────────────────────────────────────
a        = 1.50          # semi-major axis [arb. units]
b        = 0.55          # semi-minor axis
alpha_d  = 35.0          # position angle of major axis [degrees]
alpha_r  = np.deg2rad(alpha_d)

# ── sampled scan angles ──────────────────────────────────────────────────────
theta_d  = np.array([0, 30, 60, 90, 120, 150], dtype=float)
theta_r  = np.deg2rad(theta_d)

# ── derived IPD harmonic parameters ─────────────────────────────────────────
A        = (a**2 - b**2) / 2.0          # harmonic amplitude
phi_d    = 2.0 * alpha_d                 # harmonic phase [degrees]
phi_r    = np.deg2rad(phi_d)

# ── colour palette (colourblind-safe) ────────────────────────────────────────
palette  = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]

# ─────────────────────────────────────────────────────────────────────────────
# figure layout
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 4.4))
gs  = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.1, 1.9],
                        left=0.04, right=0.97, wspace=0.38)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 – 2D source image with scan-direction vectors
# ─────────────────────────────────────────────────────────────────────────────
grid_r  = 3.2
x_arr   = np.linspace(-grid_r, grid_r, 300)
X, Y    = np.meshgrid(x_arr, x_arr)
Xr      =  X * np.cos(alpha_r) + Y * np.sin(alpha_r)
Yr      = -X * np.sin(alpha_r) + Y * np.cos(alpha_r)
Z       = np.exp(-0.5 * (Xr**2 / a**2 + Yr**2 / b**2))

ax1.imshow(Z, extent=[-grid_r, grid_r, -grid_r, grid_r],
           origin="lower", cmap="Greys_r", aspect="equal", vmin=0, vmax=1)

arrow_L = 2.6
for i, (td, tr) in enumerate(zip(theta_d, theta_r)):
    dx = arrow_L * np.cos(tr)
    dy = arrow_L * np.sin(tr)
    ax1.annotate("",
        xy=(dx, dy), xytext=(-dx, -dy),
        arrowprops=dict(arrowstyle="-|>", color=palette[i],
                        lw=1.8, mutation_scale=12))
    # angle label just outside the arrowhead
    lx = (arrow_L + 0.55) * np.cos(tr)
    ly = (arrow_L + 0.55) * np.sin(tr)
    ax1.text(lx, ly, f"${td:.0f}°$",
             ha="center", va="center", fontsize=7.5, color=palette[i])

ax1.set_xlim(-grid_r - 0.2, grid_r + 0.2)
ax1.set_ylim(-grid_r - 0.2, grid_r + 0.2)
ax1.set_xticks([]);  ax1.set_yticks([])
ax1.set_aspect("equal")
ax1.text(0.03, 0.04, "AL →", transform=ax1.transAxes,
         fontsize=8, color="white", style="italic")
ax1.set_title("(a) Source with scan directions", fontsize=9, pad=4)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 – stacked 1D along-scan profiles
# ─────────────────────────────────────────────────────────────────────────────
u         = np.linspace(-4.5, 4.5, 600)
offset_dy = 1.6

for i, (td, tr) in enumerate(zip(theta_d, theta_r)):
    sig2_al = a**2 * np.cos(tr - alpha_r)**2 + b**2 * np.sin(tr - alpha_r)**2
    sig_al  = np.sqrt(sig2_al)
    profile = np.exp(-0.5 * u**2 / sig2_al)
    base    = i * offset_dy

    ax2.plot(u, profile + base, color=palette[i], lw=1.8)
    ax2.axhline(base, color="gray", lw=0.4, ls="--", alpha=0.45)

    # FWHM indicator
    fwhm = 2.355 * sig_al
    ax2.annotate("", xy=(fwhm / 2, base + 0.5),
                 xytext=(-fwhm / 2, base + 0.5),
                 arrowprops=dict(arrowstyle="<->", color=palette[i], lw=1.0))

ax2.set_xlim(-4.5, 4.5)
ax2.set_ylim(-0.35, offset_dy * len(theta_d) + 0.2)
ax2.set_xlabel("Along-scan position $u$  [arb.]", fontsize=9)
ax2.set_yticks([])
ax2.set_title("(b) Along-scan profiles $P(u;\\,\\theta)$", fontsize=9, pad=4)
ax2.tick_params(axis="x", labelsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 – IPD harmonic residual g(theta)
# ─────────────────────────────────────────────────────────────────────────────
theta_cont   = np.linspace(0, 180, 600)
theta_cont_r = np.deg2rad(theta_cont)
g_cont       = A * np.cos(2 * theta_cont_r - phi_r)

ax3.plot(theta_cont, g_cont, "k-", lw=2.0,
         label=r"$g(\theta)=A\cos(2\theta-\phi)$")
ax3.axhline(0, color="#888888", lw=1.2, ls="--",
            label="Circular source ($A=0$)")

# sampled scan angles with small Gaussian noise (fixed seed)
rng = np.random.default_rng(42)
for i, (td, tr) in enumerate(zip(theta_d, theta_r)):
    g_val  = A * np.cos(2 * tr - phi_r)
    g_noisy = g_val + rng.normal(0, 0.04)
    ax3.scatter(td, g_noisy, color=palette[i], s=55, zorder=5, clip_on=False)

# annotate amplitude A
ax3.annotate("",
    xy  =(phi_d / 2, A),
    xytext=(phi_d / 2, 0.0),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
ax3.text(phi_d / 2 + 4, A / 2, "$A$", fontsize=11, va="center")

# annotate phase: phase zero-crossing at phi/2
zero_x = phi_d / 2 + 45          # first zero after maximum
ax3.axvline(phi_d / 2, color="#aaaaaa", lw=0.8, ls=":", alpha=0.8)
ax3.text(phi_d / 2 + 1.5, -A - 0.08,
         f"$\\phi/2 = \\alpha = {alpha_d:.0f}^\\circ$",
         fontsize=8, color="#555555", ha="left")

ax3.set_xlabel(r"Scan angle $\theta$  [degrees]", fontsize=9)
ax3.set_ylabel(r"IPD harmonic residual $g(\theta)$", fontsize=9)
ax3.set_title("(c) IPD harmonic modulation", fontsize=9, pad=4)
ax3.set_xlim(0, 180)
ax3.set_xticks(np.arange(0, 181, 30))
ax3.tick_params(labelsize=8)
ax3.legend(fontsize=8, loc="upper right", framealpha=0.85)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ─────────────────────────────────────────────────────────────────────────────
out = ("/data/yn316/Codes/report/phd-thesis-template-2.4/"
       "Appendix1/Figs/Raster/ipd_harmonic_illustration.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to {out}")
