# Final 20-Figure Shortlist (Main Thesis)

## Usage rule
- Keep this set in main text.
- Move all other figure families to appendix unless they replace one of these directly.

| # | Chapter | Figure path | Claim supported | Helps rule out |
|---|---|---|---|---|
| 1 | Ch2 | `plots/experiments/preliminary_tests/01_match_distance_hist_r0.5.png` | Crossmatch distances are concentrated within a tight radius. | Matching is fully random/spurious. |
| 2 | Ch2 | `plots/experiments/preliminary_tests/01_match_distance_hist_r1.0.png` | Radius expansion changes association profile and ambiguity risk. | Radius choice is irrelevant. |
| 3 | Ch2 | `plots/experiments/preliminary_tests/02_euclid_vs_gaia_mag_scatter_r0.5.png` | Matched sources show physically plausible photometric correspondence. | Crossmatch does not preserve source identity. |
| 4 | Ch2 | `plots/qa/standardization_checks/ERO-Fornax/integral_normalization_ERO-Fornax.png` | Stamp normalization is stable and suitable for morphology-focused learning. | Model is driven by flux-scale artifacts only. |
| 5 | Ch3 | `plots/pca/explained_variance_k032.png` | PCA basis captures useful stamp structure with controlled dimensionality. | PCA target space is arbitrary/compressive loss too high. |
| 6 | Ch3 | `plots/ml_runs/base_v11_16d/01_chi2nu_hist_test.png` | Reconstruction error distribution has measurable central performance. | No useful reconstruction signal exists. |
| 7 | Ch3 | `plots/ml_runs/base_v11_16d/01b_chi2nu_hist_test_log10.png` | Tail behavior is substantial and must be analyzed explicitly. | Mean/median metrics alone are sufficient. |
| 8 | Ch3 | `plots/ml_runs/base_v11_16d/rmse_val_per_pixel_hist.png` | Pixel-space error shows stable bulk performance level. | Results depend only on a few lucky examples. |
| 9 | Ch3 | `plots/ml_runs/base_v11_16d/02_mean_maps_true_pred_absresid_raw.png` | Mean morphology is reconstructed with interpretable residual structure. | Model predicts noise/unstructured artifacts. |
|10| Ch3 | `plots/ml_runs/base_v11_16d/03_examples_resid_raw.png` | Qualitative reconstructions match aggregate error story. | Aggregate metrics hide obvious failure everywhere. |
|11| Ch4 | `report/model_decision/20260223_weighted_sweep_results/weighted_sweep_01_chi2_vs_weight.png` | Weighting changes tail/overall trade-off in a controlled way. | Mitigation strategies have no measurable effect. |
|12| Ch4 | `report/model_decision/20260223_weighted_sweep_results/weighted_sweep_02_delta_vs_baseline.png` | Specialized weighting yields non-uniform gains/losses vs baseline. | More complexity is always better. |
|13| Ch4 | `report/model_decision/20260223_feature_importance_plots/psf_vs_nonpsf_feature_importance_pixels_top20.png` | PSF and non-PSF regimes rely on different feature mechanisms. | One global mapping is equally adequate across regimes. |
|14| Ch4 | `report/model_decision/20260220_moe_contract_runs/moe_softmix_stamps_worst_true_pred_z.png` | MoE can mitigate selected worst cases but not universally. | Specialist models remove hard-failure regime entirely. |
|15| Ch5 | `plots/qa/embeddings/double_stars_8d/umap_standard/01_umap_main_map_density_plus_doubles.png` | Gaia feature space has organized structure linked to source types. | Embedding geometry is random visualization artifact. |
|16| Ch5 | `plots/qa/embeddings/double_stars_8d/umap_standard/05_umap_double_neighborhood_examples.png` | Local neighborhoods reveal morphology-relevant clustering behavior. | Cluster interpretations are anecdotal only. |
|17| Ch5 | `plots/qa/embeddings/umap_stamp_morph_colored/stamp_features_both_umaps_20260217/gaia16d/01_umap_morphology_panels.png` | Morphology indicators align with structured regions in embedding space. | Morphology variation is disconnected from feature geometry. |
|18| Ch6 | `report/model_decision/20260224_outlier_forensics_base_v11_noflux/forensics_01_chi2_distribution.png` | Hard-error tails persist after core modeling choices. | Remaining error is negligible/noisy. |
|19| Ch6 | `report/model_decision/20260224_outlier_forensics_base_v11_noflux/forensics_06_outlier_rate_by_crowding_bin.png` | Outlier rates are regime-dependent (crowding-linked). | Failures are uniformly distributed random misses. |
|20| Ch6 | `report/model_decision/20260224_outlier_forensics_base_v11_noflux/forensics_09_fornax_vs_nonfornax_influence_error.png` | Field/environment dependence supports structured residual limits. | Residuals are only due to generic underfitting. |

## Optional swaps (if needed)
- Replace #16 with: `plots/qa/embeddings/double_stars_8d/umap_standard/08_umap_cluster_neighbor_map.png`
- Replace #14 with: `report/model_decision/20260217_psf_split_experiment/moe_softmix_stamps_worst_true_pred_z.png`

## Missing figure still recommended to generate
1. One ablation summary plot/table combining baseline + 3 controls in a single panel.
