# Output and Plot Layout

This project now uses a cleaner split for generated artifacts.

## Output tree (`output/`)

- `output/crossmatch/gaia_euclid/`
  - `euclid_xmatch_gaia_*.csv`
  - `euclid_ero_xmatch_gaia_*.csv`
- `output/crossmatch/wds/`
  - `wds_xmatch/`, `wds_xmatch_r1/`, `wds_xmatch_r5/`, `wds_xmatch_test/`
- `output/dataset_npz/`
- `output/scalers/`
- `output/pca/`
- `output/ml_runs/`
- `output/experiments/`
  - one-off runs like `first_test_draft/`

## Plot tree (`plots/`)

- `plots/ml_runs/`
  - run-specific ML plots (including legacy `xgb_aug_8d`)
- `plots/qa/`
  - `dataset_checks/`
  - `augmentation_checks/`
  - `standardization_checks/`
- `plots/labels/`
- `plots/pca/`
- `plots/experiments/`
  - exploratory outputs like `preliminary_tests/`, `stamp_inspection_outputs/`

## Organizer script

Use this script to tidy legacy files without overwriting existing data:

- Dry-run: `python3 scripts/00_organize_outputs_and_plots.py --dry-run`
- Apply: `python3 scripts/00_organize_outputs_and_plots.py --apply`

The script only moves high-confidence top-level legacy items and skips any destination that already exists.
