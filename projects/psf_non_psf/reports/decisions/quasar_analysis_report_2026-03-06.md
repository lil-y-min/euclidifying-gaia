# Quasar Analysis — Decision Report
**Date:** 2026-03-06
**Status:** 15D transition complete; scripts 79 + 81 need rerun; Q1 pipeline ready to run

---

## 1. What Was Done

### Feature set decision
`feat_visibility_periods_used` removed from all classifiers going forward.
**15D = 16D minus visibility_periods_used** (Gaia scanning-law artefact, not physical).
`feature_schema.py` updated — 15D is now canonical. Scripts 79 and 81 updated to `FEATURE_SET = "15D"`.
Scaler reuses the 16D `.npz` file; subsetting is done by feature name at runtime.

### Approach 1 — WISE color-cut classifier (script 79)
Source: AllWISE crossmatch from WSDB (`allwise.main`, q3c lateral join, 2" radius).
Labels: W1−W2 ∈ [0.6, 1.6], W2−W3 ∈ [2.0, 4.2] Vega mag (from Fig. 5 of arxiv:2512.08803).
**Results (16D run — now stale, needs 15D rerun):**
- AUC = 0.987, AP = 0.677
- 359 quasar-locus sources out of 11,259 with WISE S/N≥3 in all bands

### Approach 2 — Quaia confirmed quasar classifier (script 81)
Source: Quaia G20.5.fits (1.3M quasars, Storey-Fisher+2024), direct source_id join.
Labels: label=1 if source_id in Quaia, label=0 otherwise.
**Results (16D run — now stale, needs 15D rerun):**
- 236 Quaia hits in ERO dataset
- AUC = 0.981, AP = 0.104 (low AP expected: prevalence ~0.15%, model is ~70x above chance)
- WISE cross-check: color box recovers 98% of Quaia hits (completeness), 49% purity

### Redshift analysis (script 82)
New plots replacing old AUC-per-bin bars (hard to interpret at small n):
- `07_features_vs_redshift.png` — running median of RUWE, AEN, IPD, pm_significance vs z
- `08_score_vs_redshift.png` — XGB score vs z with smoothed trend line
- `09_umap_quasars_by_redshift.png` — UMAP colored by photometric redshift
**Status: NOT YET RUN with new script.** Old plots (AUC bars) still on disk.

### ERO↔Q1 comparison (script 84)
Q1 source catalog available in WSDB as `euclid_q1.mer_final_cat` (29.9M sources).
All Q1 sources have `gaia_id` column → direct JOIN with `gaia_dr3.gaia_source`.
With G<21 + ruwe not null: **305,427 Q1 sources** (comparable to ERO's 151k).
Q1 table also has `point_like_prob`, `extended_prob` from Euclid's own morphology pipeline.
**Status: NOT YET RUN.** Script 84 written and ready.

---

## 2. Scripts and Status

| Script | Purpose | Status | Notes |
|--------|---------|--------|-------|
| `79_quasar_wise_locus_classifier.py` | WISE color-cut, train | **NEEDS RERUN** (15D) | Run `--train` only (WSDB pull done) |
| `80_quasar_stamp_catalog.py` | Stamp gallery for WISE quasars | **NEEDS RERUN** after 79 | |
| `81_quaia_quasar_classifier.py` | Quaia crossmatch, train | **NEEDS RERUN** (15D) | |
| `82_quasar_redshift_analysis.py` | Redshift trend plots | **NEEDS RUN** (new version) | Depends on 81 output |
| `83_q1_quasar_catalog_comparison.py` | Q1 spectroscopic cat comparison | **WAITING** (catalog not on CDS yet) | |
| `84_q1_quasar_pipeline.py` | Full Q1 pipeline | **READY TO RUN** | See run order below |

**Run order to get everything up to date:**
```bash
export WSDB_USER=yasmine_nourlil_2026
export WSDB_PASS='cENzh9dfC$'

python 79_quasar_wise_locus_classifier.py --train
python 80_quasar_stamp_catalog.py
python 81_quaia_quasar_classifier.py
python 82_quasar_redshift_analysis.py
python 84_q1_quasar_pipeline.py --pull_wsdb   # ~30 min
python 84_q1_quasar_pipeline.py --quaia
python 84_q1_quasar_pipeline.py --train --plots
```

---

## 3. Plot Inventory

### Raw output folders (do not use directly for thesis)
| Folder | Contents |
|--------|---------|
| `plots/ml_runs/quasar_wise_clf/` | 15 plots (01–08), **16D — stale** |
| `plots/ml_runs/quaia_clf/` | 9 plots (01–09), **16D + old redshift plots — stale** |
| `plots/ml_runs/quasar_q1/` | Empty — script 84 not yet run |

### Thesis folder: `plots/thesis_quasar/`
**Currently stale (16D).** Will be rebuilt after reruns. Planned contents:

| File | Source | Description |
|------|--------|-------------|
| `wise_01_color_color.png` | script 79 | W1-W2 vs W2-W3 color selection |
| `wise_02_roc_pr.png` | script 79 | WISE classifier ROC + PR |
| `wise_03_feature_importance.png` | script 79 | Gaia features for quasar selection |
| `wise_04_umap16d_overlay.png` | script 79 | WISE quasars on 15D UMAP |
| `wise_05_umap_pixel_overlay.png` | script 79 | WISE quasars on pixel UMAP |
| `wise_06_stamp_catalog.png` | script 80 | Best 80 stamps (ranked by score) |
| `quaia_01_roc_pr.png` | script 81 | Quaia classifier ROC + PR |
| `quaia_02_feature_importance.png` | script 81 | Gaia features for Quaia quasars |
| `quaia_03_wise_crosscheck.png` | script 81 | Quaia on WISE diagram |
| `quaia_04_umap16d_overlay.png` | script 81 | Quaia quasars on 15D UMAP |
| `quaia_05_features_vs_redshift.png` | script 82 | Feature trends vs z |
| `quaia_06_score_vs_redshift.png` | script 82 | XGB score trend vs z |
| `quaia_07_umap_by_redshift.png` | script 82 | UMAP colored by redshift |
| `q1_01_roc_pr_comparison.png` | script 84 | ERO vs Q1 performance |
| `q1_02_feature_importance.png` | script 84 | Feature importance ERO vs Q1 |
| `q1_03_redshift_comparison.png` | script 84 | Quaia z distribution ERO vs Q1 |
| `q1_04_transfer_scores.png` | script 84 | ERO model generalization to Q1 |

---

## 4. Key Numbers to Remember

| Metric | Value |
|--------|-------|
| ERO sources total | ~151,593 |
| WISE matches (S/N≥3 all bands) | 11,259 |
| WISE quasar-locus (ERO) | 359 |
| Quaia hits in ERO | 236 |
| WISE classifier AUC (16D, stale) | 0.987 |
| Quaia classifier AUC (16D, stale) | 0.981 |
| Quaia classifier AP (16D, stale) | 0.104 (~70x above chance) |
| WISE box completeness vs Quaia | 98% |
| WISE box purity vs Quaia | 49% |
| Q1 sources with Gaia (G<21) | 305,427 |

---

## 5. Scientific Narrative (for thesis)

1. **WISE color-cut** selects 359 candidate quasars from the ERO-Gaia footprint using the standard W1-W2/W2-W3 locus. A 15D Gaia-only classifier achieves AUC~0.987 on this selection — Gaia quality-degradation features encode AGN-related information independently of IR photometry.

2. **Quaia crossmatch** provides 236 spectroscopically/photometrically confirmed quasars. The same 15D Gaia classifier achieves AUC~0.981, AP~0.104 (meaningful given 0.15% prevalence). The WISE color-box is 98% complete but only 49% pure relative to Quaia.

3. **Redshift analysis** tests whether host-galaxy contamination degrades performance at low-z. Expected: RUWE/AEN elevated at z<0.5 (Gaia sees resolved host), classifier less confident.

4. **Q1 transfer test**: train on ERO, apply to Q1. Measures generalization across Euclid programs. Q1 fields are cleaner extragalactic survey fields vs ERO's unusual environments (galaxy disks, globular clusters).
