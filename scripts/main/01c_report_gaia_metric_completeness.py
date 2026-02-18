"""
Report completeness of Gaia extended metrics in crossmatch CSV files.

Input:
  output/crossmatch/gaia_euclid/euclid_xmatch_gaia_*.csv

Outputs:
  output/crossmatch/gaia_euclid/gaia_metric_completeness_by_file.csv
  output/crossmatch/gaia_euclid/gaia_metric_completeness_overall.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from feature_schema import GAIA_BACKFILL_TARGET_COLS


BASE = Path(__file__).resolve().parents[1]
IN_DIR = BASE / "output" / "crossmatch" / "gaia_euclid"
FILE_GLOB = "euclid_xmatch_gaia_*.csv"
OUT_BY_FILE = IN_DIR / "gaia_metric_completeness_by_file.csv"
OUT_OVERALL = IN_DIR / "gaia_metric_completeness_overall.csv"


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def summarize_file(path: Path, metrics: list[str]) -> list[dict]:
    df = pd.read_csv(path, low_memory=False)
    n_rows = len(df)
    rows: list[dict] = []
    for m in metrics:
        v = numeric_series(df, m)
        n_finite = int(np.isfinite(v).sum())
        frac_finite = float(n_finite / n_rows) if n_rows > 0 else float("nan")
        rows.append(
            {
                "file": path.name,
                "metric": m,
                "n_rows": n_rows,
                "n_finite": n_finite,
                "frac_finite": frac_finite,
            }
        )
    return rows


def main() -> None:
    files = sorted(IN_DIR.glob(FILE_GLOB))
    if not files:
        raise RuntimeError(f"No files found: {IN_DIR / FILE_GLOB}")

    rows: list[dict] = []
    for p in files:
        rows.extend(summarize_file(p, GAIA_BACKFILL_TARGET_COLS))

    by_file = pd.DataFrame(rows)
    by_file.to_csv(OUT_BY_FILE, index=False)

    overall = (
        by_file.groupby("metric", as_index=False)[["n_rows", "n_finite"]]
        .sum()
        .sort_values("metric")
    )
    overall["frac_finite"] = overall["n_finite"] / overall["n_rows"].replace(0, np.nan)
    overall.to_csv(OUT_OVERALL, index=False)

    print("Saved:")
    print(" -", OUT_BY_FILE)
    print(" -", OUT_OVERALL)
    print("\nOverall completeness:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
