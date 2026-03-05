#!/usr/bin/env python3
"""
Pull Euclid WSDB quality columns into a local CSV for quality-flag analysis.

Auth is read from environment variables:
  WSDB_HOST (default: wsdb.ast.cam.ac.uk)
  WSDB_DB   (default: wsdb)
  WSDB_USER (required)
  WSDB_PASS (required)
  WSDB_PORT (default: 5432)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd

try:
    import psycopg
except Exception:
    psycopg = None


DEFAULT_COLS = [
    "source_id",
    "field_tag",
    "ra",
    "dec",
    "dist_match",
    "v_niter_model",
    "v_noisearea_model",
    "v_spread_model",
    "v_spreaderr_model",
    "v_flux_auto",
    "v_flux_model",
    "v_x_image",
    "v_y_image",
    "wsdb_corner_flag",
    "wsdb_artifact_flag",
]


def parse_cols(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True, help="WSDB schema name.")
    ap.add_argument("--table", required=True, help="WSDB table name.")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--where", default="", help="Optional SQL WHERE clause (without the word WHERE).")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--columns", default=",".join(DEFAULT_COLS))
    ap.add_argument("--chunk_size", type=int, default=200000)
    args = ap.parse_args()

    if psycopg is None:
        raise RuntimeError("psycopg is not available in this environment.")

    host = os.environ.get("WSDB_HOST", "wsdb.ast.cam.ac.uk")
    db = os.environ.get("WSDB_DB", "wsdb")
    user = os.environ.get("WSDB_USER", "")
    pw = os.environ.get("WSDB_PASS", "")
    port = int(os.environ.get("WSDB_PORT", "5432"))

    if not user or not pw:
        raise RuntimeError("Set WSDB_USER and WSDB_PASS in environment before running.")

    cols = parse_cols(args.columns)
    if len(cols) == 0:
        raise RuntimeError("No columns selected.")
    cols_sql = ", ".join(cols)

    schema = str(args.schema).strip()
    table = str(args.table).strip()
    if not schema or not table:
        raise RuntimeError("schema/table cannot be empty.")

    q = f"SELECT {cols_sql} FROM {schema}.{table}"
    if str(args.where).strip():
        q += f" WHERE {str(args.where).strip()}"
    if int(args.limit) > 0:
        q += f" LIMIT {int(args.limit)}"

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    wrote_header = False
    total = 0
    with psycopg.connect(host=host, dbname=db, user=user, password=pw, port=port, autocommit=True) as conn:
        with conn.cursor(name="wsdb_pull_cursor") as cur:
            cur.itersize = int(args.chunk_size)
            cur.execute(q)
            colnames = [d.name for d in cur.description]
            while True:
                rows = cur.fetchmany(int(args.chunk_size))
                if not rows:
                    break
                df = pd.DataFrame(rows, columns=colnames)
                df.to_csv(out_csv, mode="a", header=(not wrote_header), index=False)
                wrote_header = True
                total += len(df)
                print(f"wrote_rows={total}")

    print("DONE")
    print("out_csv:", out_csv)
    print("rows:", total)


if __name__ == "__main__":
    main()

