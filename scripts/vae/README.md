# VAE Workspace

Purpose:
- Move from deterministic reconstruction to probabilistic conditional modeling (cVAE) for multimodal morphology.

Why:
- Non-PSF-only deterministic diagnostic still shows strong rounding collapse.

Implementation files:
- `scripts/vae/common.py`
- `scripts/vae/01_dataset_diagnostics.py`
- `scripts/vae/02_train_flux_mlp.py`
- `scripts/vae/03_train_cvae_shape.py`
- `scripts/vae/04_eval_cvae.py`

Execution order:
1. Run dataset diagnostics (centroid spread, ellipticity/size stats)
2. Train flux baseline regressor (optional if using true flux at eval)
3. Train cVAE for shape-only stamps
4. Evaluate mean and best-of-N samples

Assumptions:
- Input is an existing `manifest_arrays.npz` produced by the XGB data pipeline.
- `Yshape_*` are normalized shape stamps.
- `Yflux_*` are `log10(F_true)` used only when reconstructing raw intensity.
- Default `flux_mode` in eval is `true` (no flux prediction).

Quick start (example with 16D g17 full run):
```bash
python -u scripts/vae/01_dataset_diagnostics.py \
  --manifest output/ml_runs/ml_xgb_pixelsflux_16d_augtrain_g17/manifest_arrays.npz \
  --run_name cvae_diag_16d_g17
```

```bash
python -u scripts/vae/02_train_flux_mlp.py \
  --manifest output/ml_runs/ml_xgb_pixelsflux_16d_augtrain_g17/manifest_arrays.npz \
  --run_name flux_mlp_16d_g17 \
  --epochs 30 --batch_size 512
```

```bash
python -u scripts/vae/03_train_cvae_shape.py \
  --manifest output/ml_runs/ml_xgb_pixelsflux_16d_augtrain_g17/manifest_arrays.npz \
  --run_name cvae_shape_16d_g17_z16 \
  --z_dim 16 --epochs 60 --batch_size 256 \
  --beta_max 1.0 --beta_warmup_frac 0.35
```

```bash
python -u scripts/vae/04_eval_cvae.py \
  --manifest output/ml_runs/ml_xgb_pixelsflux_16d_augtrain_g17/manifest_arrays.npz \
  --cvae_ckpt output/ml_runs/vae/cvae_shape_16d_g17_z16/best_cvae.pt \
  --run_name cvae_eval_16d_g17_trueflux \
  --split test --n_samples 10 --flux_mode true \
  --alpha 1.0 --sigma0 1e-4
```

If you want predicted flux in eval:
```bash
python -u scripts/vae/04_eval_cvae.py \
  --manifest output/ml_runs/ml_xgb_pixelsflux_16d_augtrain_g17/manifest_arrays.npz \
  --cvae_ckpt output/ml_runs/vae/cvae_shape_16d_g17_z16/best_cvae.pt \
  --run_name cvae_eval_16d_g17_predflux \
  --split test --n_samples 10 --flux_mode pred \
  --flux_ckpt output/ml_runs/vae/flux_mlp_16d_g17/best_flux_mlp.pt \
  --alpha 1.0 --sigma0 1e-4
```

Where outputs go:
- runs: `output/ml_runs/vae/`
- plots: `plots/ml_runs/vae/`
- notes/tables: `report/model_decision/vae/`

Reference docs:
- `report/model_decision/vae/CVAE_ROADMAP.md`
- `report/model_decision/vae/TOMORROW_CHECKLIST.md`
