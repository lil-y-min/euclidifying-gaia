"""
Backfill additional Gaia DR3 metrics into existing crossmatch CSV files
without requiring FITS files or rerunning positional crossmatch.

It reads:
  output/crossmatch/gaia_euclid/euclid_xmatch_gaia_*.csv

Then for each file:
  - collects unique source_id values
  - queries gaia_dr3.gaia_source by source_id
  - merges requested Gaia columns back into the CSV
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import os

import numpy as np
import pandas as pd
import sqlutilpy
from feature_schema import GAIA_BACKFILL_TARGET_COLS


BASE = Path(__file__).resolve().parents[1]
IN_DIR = BASE / "output" / "crossmatch" / "gaia_euclid"
FILE_GLOB = "euclid_xmatch_gaia_*.csv"

CONN = dict(
    host="wsdb.ast.cam.ac.uk",
    db="wsdb",
    user="yasmine_nourlil_2026",
    password=(os.getenv("WSDB_PASSWORD") or os.getenv("PGPASSWORD")),
)
GAIA_TABLE = "gaia_dr3.gaia_source"

# Includes one already-present metric; script will skip existing columns unless OVERWRITE_EXISTING=True.
TARGET_GAIA_COLS = GAIA_BACKFILL_TARGET_COLS[:]

OVERWRITE_EXISTING = False


def fetch_gaia_by_source_id(source_ids: np.ndarray, cols: List[str]) -> pd.DataFrame:
    if source_ids.size == 0:
        return pd.DataFrame(columns=["source_id", *cols])

    # Keep deterministic row order by xid, then return source_id + requested cols.
    sql = f"""
    select
        m.source_id::bigint as source_id,
        {", ".join([f"g.{c}" for c in cols])}
    from mytable m
    left join {GAIA_TABLE} g
      on g.source_id = m.source_id
    order by m.xid
    """

    arrays = sqlutilpy.local_join(
        sql,
        "mytable",
        (source_ids.astype(np.int64), np.arange(source_ids.size, dtype=np.int64)),
        ("source_id", "xid"),
        **CONN,
    )
    out = pd.DataFrame({"source_id": np.array(arrays[0], dtype=np.int64)})
    for i, c in enumerate(cols, start=1):
        out[c] = np.array(arrays[i])
    return out


def main() -> None:
    files = sorted(IN_DIR.glob(FILE_GLOB))
    if not files:
        raise RuntimeError(f"No files found: {IN_DIR / FILE_GLOB}")

    print(f"Found {len(files)} files under {IN_DIR}")
    for f in files:
        print(" -", f.name)

    for path in files:
        print("\n" + "=" * 72)
        print("Processing:", path.name)
        df = pd.read_csv(path, low_memory=False)
        if "source_id" not in df.columns:
            print("  [SKIP] no source_id column")
            continue

        cols_to_add = [c for c in TARGET_GAIA_COLS if OVERWRITE_EXISTING or c not in df.columns]
        if not cols_to_add:
            print("  [OK] target columns already present")
            continue

        sid = pd.to_numeric(df["source_id"], errors="coerce")
        sid = sid.dropna().astype(np.int64)
        sid_unique = np.unique(sid.to_numpy())
        print(f"  rows={len(df):,} | unique source_id={sid_unique.size:,}")
        print(f"  columns to add={cols_to_add}")

        gdf = fetch_gaia_by_source_id(sid_unique, cols_to_add)
        # Prefix merge columns to avoid accidental collisions.
        rename_map = {c: f"__new_{c}" for c in cols_to_add}
        gdf = gdf.rename(columns=rename_map)

        merged = df.merge(gdf, how="left", on="source_id")
        for c in cols_to_add:
            merged[c] = pd.to_numeric(merged[f"__new_{c}"], errors="coerce")
            merged.drop(columns=[f"__new_{c}"], inplace=True)

        merged.to_csv(path, index=False)
        print(f"  [WRITE] updated {path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
