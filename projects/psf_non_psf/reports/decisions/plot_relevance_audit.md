# Plot Relevance Audit for Thesis (Full Inventory Reviewed)

## What was reviewed
- Reviewed all project plot/image artifacts found under `plots/`, `output/`, `report/model_decision/`, and `projects/`.
- Total reviewed paths: **1,146** (excluding `.venv` and thesis-template assets).
- Full per-file inventory with rationale is in:
  - `projects/psf_non_psf/reports/decisions/plot_relevance_inventory.tsv`

## Brutal summary
- You have enough evidence for a strong thesis.
- You also have substantial plot bloat and duplication.
- Current risk: too many galleries and too few decisive comparison figures.

## Quantitative triage
- `High relevance`: 268
- `Medium relevance`: 409
- `Low relevance`: 469

Action split:
- `Promote` (main-text candidates): 268
- `Select representative only`: 156
- `Promote selected panels`: 132
- `Keep 6-12 pre-defined exemplars`: 450
- `Archive / avoid in final main narrative`: 19+ (legacy/path-artifact groups)

## Main issues to fix
1. Duplicate figure families are repeated across `output/` and `plots/`.
2. Outlier galleries dominate file count but are weak as primary evidence.
3. Some legacy files are explicitly labeled bad/legacy and should not appear in main text.
4. Classification-only diagnostics exist, but your thesis is reconstruction-first.

## Recommended chapter-to-plot mapping (core set)
Use these as main text anchors; move the rest to appendix.

### Chapter 2: Domain gap and data reliability
- `plots/experiments/preliminary_tests/01_match_distance_hist_r0.5.png`
- `plots/experiments/preliminary_tests/01_match_distance_hist_r1.0.png`
- `plots/experiments/preliminary_tests/02_euclid_vs_gaia_mag_scatter_r0.5.png`
- `plots/experiments/preliminary_tests/02_euclid_vs_gaia_mag_scatter_r1.0.png`
- `plots/experiments/stamp_inspection_outputs/stamp_sizes_r0p5.png`
- `plots/qa/standardization_checks/ERO-Fornax/integral_normalization_ERO-Fornax.png`

Why useful:
- Provides crossmatch uncertainty sensitivity and normalization validity, both mandatory evidence gates.

### Chapter 3: Reconstruction backbone
- `plots/pca/explained_variance_k032.png`
- `plots/ml_runs/base_v11_16d/01_chi2nu_hist_test.png`
- `plots/ml_runs/base_v11_16d/01b_chi2nu_hist_test_log10.png`
- `plots/ml_runs/base_v11_16d/rmse_val_per_pixel_hist.png`
- `plots/ml_runs/base_v11_16d/02_mean_maps_true_pred_absresid_raw.png`
- `plots/ml_runs/base_v11_16d/03_examples_resid_raw.png`

Why useful:
- Gives PCA justification plus central/tail reconstruction behavior in physical-evaluation framing.

### Chapter 4: Morphology-aware mitigation
- `report/model_decision/20260223_weighted_sweep_results/weighted_sweep_01_chi2_vs_weight.png`
- `report/model_decision/20260223_weighted_sweep_results/weighted_sweep_02_delta_vs_baseline.png`
- `report/model_decision/20260224_weighted_sweep_results/weighted_sweep_03_gateA_passfail.png`
- `report/model_decision/20260223_feature_importance_plots/psf_vs_nonpsf_feature_importance_pixels_top20.png`
- `report/model_decision/20260223_feature_importance_plots/nonpsf_vs_softmix_feature_importance_pixels_top20.png`
- `report/model_decision/20260220_moe_contract_runs/moe_softmix_stamps_worst_true_pred_z.png`

Why useful:
- Directly tests the mitigation hypothesis and shows trade-offs (tail gains vs stability/cost).

### Chapter 5: Representation and error topology
- `plots/qa/embeddings/double_stars_8d/umap_standard/01_umap_main_map_density_plus_doubles.png`
- `plots/qa/embeddings/double_stars_8d/umap_standard/05_umap_gaia_single_vs_multiple_r10_panels_by_type.png`
- `plots/qa/embeddings/double_stars_8d/umap_standard/08_umap_cluster_neighbor_map.png`
- `plots/qa/embeddings/umap_stamp_morph_colored/stamp_features_both_umaps_20260217/gaia16d/01_umap_morphology_panels.png`

Why useful:
- Supports the geometry-to-error interpretation instead of purely anecdotal visual statements.

### Chapter 6: Model limits vs information ceiling
- `report/model_decision/20260224_outlier_forensics_base_v11_noflux/forensics_01_chi2_distribution.png`
- `report/model_decision/20260224_outlier_forensics_base_v11_noflux/forensics_02_field_lift.png`
- `report/model_decision/20260224_outlier_forensics_base_v11_noflux/forensics_06_outlier_rate_by_crowding_bin.png`
- `report/model_decision/20260224_outlier_forensics_base_v11_noflux/forensics_08_outlier_rate_by_influence_decile.png`
- `report/model_decision/20260224_outlier_forensics_base_v11_noflux/forensics_09_fornax_vs_nonfornax_influence_error.png`

Why useful:
- Best current evidence that residual failures are regime-structured and not random underfitting noise.

## What to demote to appendix (important)
1. Bulk outlier galleries:
   - `plots/ml_runs/ml_xgb_8d/outliers_test_pages_png/*`
   - `plots/ml_runs/ml_xgb_8d/outliers_test_raw_true_pred_resid_png/*`
   - `report/model_decision/20260224_visual_residual_audit_base_v11/outlier_visual_audit/*`
2. Repeated QA per field:
   - many repeated grids in `plots/qa/dataset_checks/*`, `plots/qa/augmentation_checks/*`, `plots/qa/standardization_checks/*`
3. Duplicated run exports:
   - mirrored outputs in `output/*/eval/plots` and `plots/*`
4. Legacy artifacts:
   - `output/experiments/first_test_draft/*`
   - paths carrying Windows export artifact under `output/ml_runs/ml_xgb_8d/C:\\Users\\...`

## Missing or weak evidence to add (high priority)
1. A single ablation summary figure/table (baseline vs 3 controls).
2. One explicit crossmatch contamination/ambiguity estimate plot/table.
3. One model-capacity ladder figure to support ceiling-vs-underfitting argument.
4. Pre-registered case-selection figure protocol (worst tail, median bin, representative cluster).

## Suggested figure budget for main text
- Chapter 2: 4 figures
- Chapter 3: 5 figures
- Chapter 4: 4 figures
- Chapter 5: 3 figures
- Chapter 6: 4 figures
- Total main figures: ~20 (everything else appendix)

## Non-negotiable writing rule for these plots
For every figure:
1. state the claim it supports,
2. name the failure mode it rules out,
3. include one-line takeaway tied to chapter question.
