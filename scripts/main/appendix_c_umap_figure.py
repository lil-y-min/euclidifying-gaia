"""
Generate the UMAP hyperparameter sensitivity figure for Appendix C.

Runs 6 UMAP embeddings on a 25k subsample:
  rows    = min_dist  in {0.05, 0.30}
  columns = n_neighbors in {15, 30, 100}
Default (n_neighbors=30, min_dist=0.08) is marked with a gold border.
Sources are coloured by environment type (galaxy / cluster / stellar-other).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

# ── paths ────────────────────────────────────────────────────────────────────
EMB_CSV = ("/data/yn316/Codes/output/experiments/embeddings/"
           "umap16d_manualv8_filtered/embedding_umap.csv")
OUT     = ("/data/yn316/Codes/report/phd-thesis-template-2.4/"
           "Appendix3/Figs/Raster/umap_sensitivity.png")

FEAT_COLS = [
    "feat_log10_snr", "feat_ruwe", "feat_astrometric_excess_noise",
    "feat_parallax_over_error", "feat_visibility_periods_used",
    "feat_ipd_frac_multi_peak", "feat_c_star", "feat_pm_significance",
    "feat_astrometric_excess_noise_sig", "feat_ipd_gof_harmonic_amplitude",
    "feat_ipd_gof_harmonic_phase", "feat_ipd_frac_odd_win",
    "feat_phot_bp_n_contaminated_transits", "feat_phot_bp_n_blended_transits",
    "feat_phot_rp_n_contaminated_transits", "feat_phot_rp_n_blended_transits",
]

# ── environment categorisation ───────────────────────────────────────────────
GALAXY_FIELDS  = {"IC342", "NGC2403", "NGC6744", "NGC6822", "HolmbergII", "IC10"}
CLUSTER_FIELDS = {"Abell2390", "Abell2764", "Dorado", "Fornax",
                  "NGC6254", "NGC6397", "Perseus"}

def env_label(tag: str) -> str:
    name = tag.replace("ERO-", "").replace("ERO_", "")
    if any(g in name for g in GALAXY_FIELDS):
        return "Nearby galaxy"
    if any(c in name for c in CLUSTER_FIELDS):
        return "Cluster"
    return "Stellar / star-forming"

ENV_COLORS = {
    "Nearby galaxy":       "#E69F00",
    "Cluster":             "#0072B2",
    "Stellar / star-forming": "#bbbbbb",
}
ENV_ZORDER = {
    "Nearby galaxy": 3,
    "Cluster":       2,
    "Stellar / star-forming": 1,
}

# ── load and subsample ───────────────────────────────────────────────────────
print("Loading embedding CSV …")
df = pd.read_csv(EMB_CSV)
df = df.dropna(subset=FEAT_COLS)
rng = np.random.default_rng(42)
idx = rng.choice(len(df), size=min(25000, len(df)), replace=False)
df  = df.iloc[idx].reset_index(drop=True)

X = StandardScaler().fit_transform(df[FEAT_COLS].values.astype(np.float32))
env = df["field_tag"].apply(env_label).values
print(f"Subsample: {len(df)} sources")

# ── hyperparameter grid ──────────────────────────────────────────────────────
n_neighbors_list = [15, 30, 100]
min_dist_list    = [0.08, 0.30]
DEFAULT_NN, DEFAULT_MD = 30, 0.08

# ── run embeddings ────────────────────────────────────────────────────────────
embeddings = {}
for nn in n_neighbors_list:
    for md in min_dist_list:
        key = (nn, md)
        print(f"  UMAP n_neighbors={nn}, min_dist={md} …", flush=True)
        reducer = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=md,
                            metric="euclidean", random_state=42, low_memory=False)
        embeddings[key] = reducer.fit_transform(X)
        print(f"    done.")

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    len(min_dist_list), len(n_neighbors_list),
    figsize=(13, 8.5),
    gridspec_kw=dict(hspace=0.30, wspace=0.12),
)

for row_i, md in enumerate(min_dist_list):
    for col_i, nn in enumerate(n_neighbors_list):
        ax  = axes[row_i, col_i]
        emb = embeddings[(nn, md)]
        is_default = (nn == DEFAULT_NN and abs(md - DEFAULT_MD) < 0.001)

        # plot each environment in z-order
        for label in ["Stellar / star-forming", "Cluster", "Nearby galaxy"]:
            mask = env == label
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                s=1.0, alpha=0.7, linewidths=0,
                color=ENV_COLORS[label],
                zorder=ENV_ZORDER[label],
                rasterized=True,
            )

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"$k={nn}$,  $d_{{\\min}}={md}$",
            fontsize=10, pad=4,
        )

        # gold border for the closest-to-default panel
        if is_default:
            for spine in ax.spines.values():
                spine.set_edgecolor("#DAA520")
                spine.set_linewidth(2.5)
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
                spine.set_linewidth(0.8)

        # column header on top row
        if row_i == 0:
            ax.set_xlabel(f"$k = {nn}$", fontsize=9, labelpad=2)

    # row label on left
    axes[row_i, 0].set_ylabel(
        f"$d_{{\\min}} = {md}$", fontsize=10, labelpad=6
    )

# ── shared legend ─────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ENV_COLORS[l],
           markersize=7, label=l)
    for l in ["Nearby galaxy", "Cluster", "Stellar / star-forming"]
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.01))

fig.text(0.5, 0.97,
         "UMAP hyperparameter sensitivity  "
         r"(gold border: thesis default $k=30$, $d_{\min}=0.08$)",
         ha="center", va="top", fontsize=10)

fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved: {OUT}")
