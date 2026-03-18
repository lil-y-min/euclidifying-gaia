# Euclidifying Gaia

Bachelor thesis (University of Cambridge / École Polytechnique).
Advisor: Prof. Vasily Belokurov.

**Core claim:** Gaia DR3 astrometric and photometric residuals encode recoverable
morphology signal about Euclid VIS sources. Measurement mismatch is physically
informative.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # Python 3.11.14
```

---

## Repository layout

```
.
├── scripts/
│   ├── main/           # All analysis scripts, numbered in execution order
│   │   ├── 00–12       # Data preparation: crossmatch, export NPZ, augmentation
│   │   ├── 13–20       # Baseline XGBoost reconstruction (pixel-flux model)
│   │   ├── 21–35       # UMAP embeddings and double-star / WDS analysis
│   │   ├── 36–52       # PSF/non-PSF labeling, MoE, quality flags
│   │   ├── 53–69       # Morphology metrics, feature diagnostics
│   │   ├── 70–78       # Galaxy environment classifier, radius regression
│   │   ├── 79–86       # Quasar classifiers (WISE, Quaia, DESI galaxies)
│   │   ├── 87–92       # Thesis figures (UMAP, borders, morphology, ROC, stamps)
│   │   ├── appendix_*  # Appendix figures
│   │   └── feature_schema.py   # Canonical 8D/10D/15D/16D feature definitions
│   ├── nn/             # CNN and neural-net experiments (exploratory, not in thesis)
│   ├── vae/            # cVAE morphology experiments (exploratory, not in thesis)
│   └── drafts/         # Early exploratory scripts (superseded)
│
├── projects/           # Self-contained sub-project archives (each has own README)
│   ├── crossmatching/  # Gaia × Euclid ERO crossmatch pipeline
│   ├── testing_stamps/ # Dataset export and validation
│   ├── umaps/          # UMAP embedding experiments
│   ├── psf_non_psf/    # PSF labeling, gate model, MoE, cVAE reports
│   ├── xgboost/        # XGBoost reconstruction baseline
│   └── vae/            # VAE experiment archive
│
├── output/             # Generated model outputs (gitignored)
│   ├── crossmatch/     # CSV crossmatch tables
│   ├── dataset_npz/    # 20×20 Euclid VIS stamp arrays per field
│   ├── ml_runs/        # Trained models, scores, feature importances
│   ├── experiments/    # UMAP embeddings and one-off runs
│   ├── scalers/        # Feature standardisation scalers
│   └── pca/            # PCA fits
│
├── plots/              # Generated figures (gitignored)
│   ├── ml_runs/        # Per-run diagnostic plots
│   ├── qa/             # Dataset quality checks
│   ├── labels/         # PSF label visualisations
│   └── experiments/    # Exploratory plots
│
├── data/               # Raw catalogues (gitignored)
│   ├── quaia_G20.5.fits          # Quaia quasar catalogue (Storey-Fisher+2024)
│   ├── with_desi/                # DESI × Euclid crossmatch tables
│   └── wds.sql                   # Washington Double Star catalogue query
│
├── report/             # LaTeX thesis source (gitignored)
├── deliverables/       # One-off analysis outputs (gitignored)
│
├── requirements.txt    # Pinned Python dependencies
└── .gitignore
```

---

## Key scripts

| Script | What it does |
|--------|-------------|
| `scripts/main/01_crossmatch_euclid_ero_to_gaia_dr3.py` | Build the primary Gaia × Euclid ERO crossmatch |
| `scripts/main/05_export_dataset_npz.py` | Export 20×20 VIS stamps + Gaia feature metadata |
| `scripts/main/14a_train_xgb.py` | Train per-pixel XGBoost stamp reconstructor |
| `scripts/main/16_eval_xgb_reconstruct.py` | Evaluate reconstruction (χ²_ν metrics) |
| `scripts/main/76_galaxy_environment_classifier.py` | Galaxy disk environment classifier (AUC=0.902) |
| `scripts/main/79_quasar_wise_locus_classifier.py` | WISE photometric quasar classifier (AUC=0.989) |
| `scripts/main/81_quaia_quasar_classifier.py` | Quaia spectroscopic quasar classifier (AUC=0.992) |
| `scripts/main/86_galaxy_desi_classifier.py` | DESI confirmed galaxy classifier (AUC=0.996) |
| `scripts/main/feature_schema.py` | Single source of truth for all feature sets |

---

## Classifier results summary

| Task | AUC | Leading feature |
|------|-----|----------------|
| Galaxy disk environment (16D) | 0.902 | IPD harmonic phase |
| Galaxy disk environment (15D) | 0.869 | IPD harmonic phase |
| WISE quasar locus | 0.989 | log10 SNR |
| Quaia quasars (spectroscopic) | 0.992 | pm_significance |
| ERO → Q1 transfer | 0.990 | pm_significance |
| DESI confirmed galaxies | 0.996 | c★ |

---

## Data access

Raw Euclid VIS stamps and Gaia DR3 features are stored locally and are not
committed (see `.gitignore`). The WSDB (Cambridge) database is required to
re-run the crossmatch scripts. Processed outputs are in `output/ml_runs/`.
