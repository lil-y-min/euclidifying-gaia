#!/usr/bin/env python3
from __future__ import annotations

"""
One-time organizer for output/ and plots/.

Usage:
  python scripts/00_organize_outputs_and_plots.py --dry-run
  python scripts/00_organize_outputs_and_plots.py --apply

The script is conservative:
- It only moves known top-level legacy items.
- It keeps current pipeline roots (output/dataset_npz, output/ml_runs, output/pca, output/scalers).
- It never overwrites existing files.
"""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Move:
    src: Path
    dst: Path


def plan_moves(base: Path) -> List[Move]:
    output = base / "output"
    plots = base / "plots"

    moves: List[Move] = []

    # Output: move crossmatch CSVs into one clear sub-tree.
    cross_gaia = output / "crossmatch" / "gaia_euclid"
    for p in output.glob("euclid_xmatch_gaia_*.csv"):
        moves.append(Move(p, cross_gaia / p.name))
    for p in output.glob("euclid_ero_xmatch_gaia_*.csv"):
        moves.append(Move(p, cross_gaia / p.name))

    # Output: group WDS runs together.
    cross_wds = output / "crossmatch" / "wds"
    for name in ["wds_xmatch", "wds_xmatch_r1", "wds_xmatch_r5", "wds_xmatch_test"]:
        p = output / name
        if p.exists():
            moves.append(Move(p, cross_wds / name))

    # Output: group explicit draft/experimental run folders.
    exp_out = output / "experiments"
    first_test = output / "first_test_draft"
    if first_test.exists():
        moves.append(Move(first_test, exp_out / first_test.name))

    # Plots: move obvious legacy/one-off dirs into structured locations.
    plots_ml = plots / "ml_runs"
    xgb_aug_legacy = plots / "xgb_aug_8d"
    if xgb_aug_legacy.exists():
        moves.append(Move(xgb_aug_legacy, plots_ml / "xgb_aug_8d"))

    plots_exp = plots / "experiments"
    for name in ["preliminary_tests", "stamp_inspection_outputs"]:
        p = plots / name
        if p.exists():
            moves.append(Move(p, plots_exp / name))

    # Plots: normalize QA and diagnostic roots under plots/qa/.
    plots_qa = plots / "qa"
    qa_renames = {
        "checking_npz": "dataset_checks",
        "checking_npz_aug": "augmentation_checks",
        "standardization_checks": "standardization_checks",
    }
    for old, new in qa_renames.items():
        p = plots / old
        if p.exists():
            moves.append(Move(p, plots_qa / new))

    # Plots: normalize label grids under plots/labels/.
    label_grids = plots / "label_grids"
    if label_grids.exists():
        moves.append(Move(label_grids, plots / "labels"))

    return moves


def safe_move(m: Move, apply: bool) -> str:
    if not m.src.exists():
        return f"[SKIP] missing: {m.src}"
    if m.src.resolve() == m.dst.resolve():
        return f"[SKIP] same path: {m.src}"
    if m.dst.exists():
        return f"[SKIP] destination exists: {m.dst}"

    if not apply:
        return f"[PLAN] {m.src} -> {m.dst}"

    m.dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(m.src), str(m.dst))
    return f"[MOVED] {m.src} -> {m.dst}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Organize output/ and plots/ folders")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Show planned moves only")
    mode.add_argument("--apply", action="store_true", help="Apply planned moves")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    moves = plan_moves(base)

    if not moves:
        print("No moves needed.")
        return

    print(f"Planned moves: {len(moves)}")
    for m in moves:
        print(safe_move(m, apply=args.apply))


if __name__ == "__main__":
    main()
