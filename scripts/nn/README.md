# NN Pipeline (PSF vs Non-PSF)

This folder contains a minimal, organized neural pipeline to classify image cutouts as `psf_like` or `non_psf_like`.

## Files
- `00_psf_labeling.py`: builds weak labels from morphology proxies + optional Gaia anomaly flags.
- `01_train_cnn_psf.py`: trains a baseline CNN classifier using those labels.
- `02_train_nn_pixelsflux.py`: trains a neural stamp reconstruction model (`features -> shape + flux -> stamp`).
- `03_eval_nn_pixelsflux.py`: generates QA plots (`chi2`, residual maps/examples) and permutation feature importance.

## Typical workflow
1. Build labels:
```bash
python scripts/nn/00_psf_labeling.py
```

2. Train CNN:
```bash
python scripts/nn/01_train_cnn_psf.py \
  --labels_csv output/ml_runs/nn_psf_labels/labels_psf_weak.csv \
  --run_name nn_cnn_s_img_psfweak_v1
```

## Label semantics
- `label = 0`: `psf_like`
- `label = 1`: `non_psf_like`
- Intermediate/ambiguous objects are dropped during weak-label construction.

## Notes
- This is intentionally conservative: it favors cleaner pseudo-labels over maximum sample count.
- First baseline is image-only; metadata branch can be added once baseline is stable.

## Reconstruction Workflow (what you asked for)
Train NN stamp predictor from an existing `manifest_arrays.npz`:

```bash
.venv/bin/python -u scripts/nn/02_train_nn_pixelsflux.py \
  --manifest_npz output/ml_runs/ml_xgb_pixelsflux_8d_augtrain_g17/manifest_arrays.npz \
  --run_name nn_pixelsflux_mlp_v1
```

Live debug logs while running:
```bash
tail -f output/ml_runs/<run_id>/train.log
```

Quick smoke test (small subset):
```bash
.venv/bin/python -u scripts/nn/02_train_nn_pixelsflux.py \
  --manifest_npz output/ml_runs/ml_xgb_pixelsflux_8d_augtrain_g17/manifest_arrays.npz \
  --run_name nn_pixelsflux_smoke \
  --epochs 3 \
  --max_train_samples 20000 \
  --max_val_samples 4000 \
  --max_test_samples 4000
```

Evaluate a finished run and generate plots + importance:
```bash
.venv/bin/python -u scripts/nn/03_eval_nn_pixelsflux.py \
  --run nn_pixelsflux_smoke_YYYYMMDD_HHMMSS
```
