# Project Report: Recent Work, Decisions, Metrics, and Conclusions

## 1) Scope and datasets used
- We worked with two evaluation universes:
- `Full reconstruction test set`: `n=29,767` (from `manifest_arrays.npz`) for stamp reconstruction metrics (`chi2nu`, shape RMSE).
- `Labeled gate/test subset`: `n=18,524` for PSF/non-PSF gate metrics (`AUC`, `AP`, `logloss`).
- Final weak-labeled set size: `151,593` rows.
- Label proportions overall: `PSF=66,302 (43.74%)`, `non-PSF=85,291 (56.26%)`.

## 2) How PSF score weights were built (exact)
- Initial handcrafted score weights:
- `m_concentration_r2_r6: +1.3`
- `m_asymmetry_180: -1.0`
- `m_ellipticity: -0.8`
- `m_peak_sep_pix: -1.1`
- `m_edge_flux_frac: -0.6`
- `m_peak_ratio_2over1: -0.5`
- Calibration experiment (logistic fit against labels) produced alternative weights:
- `m_concentration_r2_r6: -2.1781`
- `m_asymmetry_180: +0.6123`
- `m_ellipticity: +1.1177`
- `m_peak_sep_pix: 0.0` (frozen)
- `m_edge_flux_frac: +0.6360`
- `m_peak_ratio_2over1: +0.6676`

## 3) Labeling pipeline results
- Raw feature rows: `273,830`.
- Labeled after quantile + Gaia rule + confidence filter: `151,593`.
- Removed by confidence filter: `22,976`.
- No dedup loss (`removed_by_dedup=0`).
- Gaia non-PSF override count: `36,587`.
- Clarification on `confidence filter` (exact definition from `scripts/nn/00_psf_labeling.py`):
- Confidence is a normalized distance from the score midpoint, not a calibrated probability.
- `mid = (hi + lo) / 2`
- `confidence = clip( abs(score_psf_like - mid) / abs(hi - lo), 0, 1 )`
- Filter used: keep only `confidence >= 0.55`.
- Interpretation: rows close to the ambiguous middle are dropped; high-tail rows are kept.
- Split balance (post-labeling):
- Train: `113,004` (`PSF 43.20%`, `non-PSF 56.80%`)
- Val: `20,065` (`PSF 44.74%`, `non-PSF 55.26%`)
- Test: `18,524` (`PSF 45.90%`, `non-PSF 54.10%`)

## 4) Gate model experiments and decision
- Models compared on gate task: logistic / RF / XGB.
- Best test AUC: logistic (`0.965152`).
- Best test logloss: XGB (`0.134655`) in stability pass comparison.
- Frozen production gate decision: **logistic v1.1 with scale floor `1e-3`**.
- Test metrics: `AUC=0.965152`, `AP=0.980478`, `logloss=0.142308`.
- Reason: stable numerics with equivalent held-out performance.
- Final frozen gate coefficients (exact):
- Intercept: `-0.8405220131`
- Coefs in feature order `[conc, asym, ell, peak_sep, edge_flux, peak_ratio]`:
- `[-1.8360597216, 0.6278905796, 1.2190703543, 1.1578589931, 0.6175777985, 0.5366239584]`

## 5) XGB vs NN reconstruction test
- Decision on 2026-02-17: keep XGBoost as primary recon model.
- XGB won most shape/flux/chi2 metrics; NN only won `chi2nu_p90` in that comparison.

## 6) PSF/non-PSF specialist experiments
- Clarification on `smoke` specialist runs:
- `Smoke` means a small-scale pipeline sanity test before full-cost training.
- In this project smoke caps were: `train=8000`, `val=2000`, `test=2000`, with shortened boosting schedule.
- Smoke is used to validate filtering/training/evaluation plumbing; it is not used as final production ranking.
- Smoke split runs validated pipeline behavior:
- PSF specialist (smoke): better median chi2 (`35.83`) than non-PSF specialist smoke (`55.16`).
- Non-PSF specialist smoke had much larger tails (`p90=3186.95`, `p99=14043.71`).
- Full specialists vs baseline by slice:
- On PSF-like slice, specialist improved strongly over baseline (`chi2 median 33.63 vs 44.45`).
- On non-PSF-like slice, specialist improved tails (`p90 3526.77 vs 4630.22`, `p99 14863.47 vs 23590.13`) but median got slightly worse (`49.08 vs 44.36`).

## 7) Softmix MoE ("soft mx MoE")
- Contract routing rule used:
- `p<0.3 -> PSF expert`, `p>0.7 -> non-PSF expert`, otherwise linear softmix.
- Softmix formula:
- `w_psf = (0.7 - p_nonpsf) / 0.4`
- `w_nonpsf = 1 - w_psf`
- `shape_pred = w_psf * shape_psf + w_nonpsf * shape_nonpsf`
- Later contract run (`20260220`): all 4 modes (`psf_only/nonpsf_only/soft_mix/hard_route`) were nearly identical to each other; deltas vs softmix were tiny (`~0.01%` to `0.04%` in tails).
- Routing proportions at `0.3/0.7` on labeled gate test (`n=18,524`):
- `PSF-only 37.82%`, `Soft 21.12%`, `non-PSF-only 41.06%`.
- Class-conditional routing:
- True PSF: `71.90%` routed PSF-only.
- True non-PSF: `73.03%` routed non-PSF-only.

## 8) Explicit comparison: 2-expert MoE vs base (no-MoE)
- We use these explicit formulas (lower metric is better):
- `tail_improvement_pct = 100 * (1 - candidate_tail / base_tail)`
- `median_change_pct = 100 * (candidate_median / base_median - 1)`
- `shape_rmse_change_pct = 100 * (candidate_rmse / base_rmse - 1)`

### 8.1 Contract-gate comparison using archived `20260217` artifacts
- Clarification on wording: previously described as `contract-like` because we applied the MoE contract pass/fail gates to saved historical artifacts rather than a fresh rerun of the full contract package.
- Base: `baseline_16d_phase3` (from `psf_split_full_vs_baseline_16d.csv`).
- Candidate: `soft_mix` (from `20260217_psf_split_experiment/moe_metrics_test_trueflux.csv`).
- Numeric comparison:
- `chi2nu_p90`: `1317.99 -> 1236.41` => `+6.19%` improvement.
- `chi2nu_p99`: `15337.42 -> 14900.61` => `+2.85%` improvement.
- `chi2nu_median`: `38.01 -> 40.88` => `+7.56%` degradation (worse).
- Contract verdict: **fail** (does not meet p99 and median gates).

### 8.2 Strict full-test recomputation for `base_v11_16d` vs `20260220` softmix
- Base metrics recomputed directly from `base_v11_16d` boosters on full test (`n=29,767`) with same chi2 definition (`sqrt_Itrue`, `alpha=1.0`, `sigma0=1e-4`, border-MAD sigma bg):
- Base (`base_v11_16d`):
- `shape_rmse_mean=0.0056328154`
- `chi2nu_median=37.9573`
- `chi2nu_p90=1316.0255`
- `chi2nu_p99=15231.6279`
- MoE candidate from `20260220_moe_contract_runs/moe_metrics_test_trueflux.csv` (`soft_mix`):
- `shape_rmse_mean=0.0131876604`
- `chi2nu_median=606.3894`
- `chi2nu_p90=1445.0598`
- `chi2nu_p99=2092.4425`
- Explicit deltas vs base:
- `chi2nu_p90`: `-9.80%` (worse tail at p90).
- `chi2nu_p99`: `+86.26%` (much better extreme tail at p99).
- `chi2nu_median`: `+1497.55%` (massive central degradation).
- `shape_rmse_mean`: `+134.12%` (much worse).
- Interpretation: current MoE candidate is not production-eligible; improvements are concentrated in extreme tail only and offset by very large central-mass degradation.

## 9) CVAE / latent-data experiments
- CVAE recon experiments (`best-of-N`) gave only modest RMSE gains and still showed morphology collapse tendencies (predicted ellipticity far below true ellipticity).
- Example (`z32 w2 fb000 b01`, n40):
- `rmse_shape_bestofN=0.0055749`, but `ell_true_mean=0.0900` vs `ell_bestofN_pred_mean=0.00987`.
- Morph + latent for gate classification:
- Best test AUC: `morph_plus_latent=0.965183` (slightly above morph-only `0.965152`).
- Best test logloss: morph-only (`0.142308`) better than morph+latent (`0.142400`).
- Conclusion: latent features add very small incremental signal and not enough to replace morphology features.

## 10) Final decisions made
- Keep **XGBoost** as primary reconstruction model.
- Keep **morph-only logistic gate** (v1.1, `scale_floor=1e-3`) as production PSF/non-PSF scorer.
- Keep **softmix MoE as experimental**, not promoted as main production path yet due weak/unstable gains.
- Keep **CVAE as diagnostic/auxiliary** (QC and latent analysis), not primary recon replacement yet.

## 11) Reproducibility checklist (explicit)
- Environment:
- Run from repo root: `/data/yn316/Codes`
- Use same Python environment as training/eval runs.

### 11.1 Regenerate MoE metrics table
```bash
python scripts/main/36_moe_softmix_eval_16d.py \
  --full_run base_v11_16d \
  --psf_run moe_v2_psf_16d \
  --nonpsf_run moe_v2_nonpsf_16d \
  --labels_csv output/ml_runs/nn_psf_labels/labels_psf_weak.csv \
  --hard_low 0.3 \
  --hard_high 0.7 \
  --out_dir report/model_decision/20260220_moe_contract_runs
```

### 11.2 Recompute base-only full-test metrics with exact chi2 settings
```bash
python - <<'PY'
# (same script used in this report)
# computes base_v11_16d shape RMSE and chi2nu quantiles on n=29,767
PY
```

### 11.3 Compute explicit percent deltas
```text
tail_improvement_pct = 100 * (1 - candidate_tail / base_tail)
median_change_pct    = 100 * (candidate_median / base_median - 1)
rmse_change_pct      = 100 * (candidate_rmse / base_rmse - 1)
```

## 12) Important continuation notes (expanded)
- Why this matters:
- We currently see mixed behavior: some extreme-tail gains can come with strong central-mass degradation; this can make a model look better on one statistic while being worse overall.
- Practical rule for continuation:
- no candidate should be promoted unless it improves tails **and** protects median/core quality under the same evaluation setup.
- Recommended next cycle (explicit):
1. Lock artifacts before training: labels CSV, manifest NPZ, gate package, split mapping, sigma settings.
2. Train candidates under fixed protocol: baseline refresh + 1 to 2 non-PSF multimodal candidates.
3. Evaluate all candidates on the same full test universe (`n=29,767`) with same chi2 settings.
4. Apply hard gates jointly: p90 gain, p99 gain, median protection, and shape-RMSE protection.
5. Run slice diagnostics with minimum support (`>=5%` test rows per slice) so decisions are not driven by tiny subsets.
6. Export failure package (same source IDs across models) to locate where regressions concentrate.
7. Produce one final decision table and explicit verdict (`promote` / `do not promote`).
- Expected benefit:
- This removes ambiguity and makes model selection reproducible, auditable, and robust against cherry-picking a single favorable metric.
