#!/usr/bin/env python3
"""
Match WSDB Euclid quality columns onto local forensics rows by sky position.

Inputs:
- local forensics CSV with columns:
  test_index, source_id, field_tag, v_alpha_j2000, v_delta_j2000

WSDB source:
- euclid_ero.merge_cat (or custom schema/table)

Auth from env:
- WSDB_HOST, WSDB_DB, WSDB_USER, WSDB_PASS, WSDB_PORT
"""

from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import psycopg


def chunked(rows: List[Tuple], n: int):
    for i in range(0, len(rows), n):
        yield rows[i : i + n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forensics_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--schema", default="euclid_ero")
    ap.add_argument("--table", default="merge_cat")
    ap.add_argument("--match_radius_arcsec", type=float, default=0.4)
    ap.add_argument("--batch_insert", type=int, default=5000)
    ap.add_argument(
        "--select_cols",
        default="v_alpha_j2000,v_delta_j2000,v_niter_model,v_noisearea_model,v_spread_model,v_spreaderr_model,v_flux_auto,v_flux_model,v_x_image,v_y_image,dist_match,v_objid,objid",
    )
    args = ap.parse_args()

    host = os.environ.get("WSDB_HOST", "wsdb.ast.cam.ac.uk")
    db = os.environ.get("WSDB_DB", "wsdb")
    user = os.environ.get("WSDB_USER", "")
    pw = os.environ.get("WSDB_PASS", "")
    port = int(os.environ.get("WSDB_PORT", "5432"))
    if not user or not pw:
        raise RuntimeError("Set WSDB_USER and WSDB_PASS environment variables.")

    fin = Path(args.forensics_csv)
    fout = Path(args.out_csv)
    fout.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fin, usecols=lambda c: c in {"test_index", "source_id", "field_tag", "v_alpha_j2000", "v_delta_j2000"}, low_memory=False)
    for c in ["test_index", "source_id", "v_alpha_j2000", "v_delta_j2000"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["test_index", "v_alpha_j2000", "v_delta_j2000"]).copy()
    df["test_index"] = df["test_index"].astype(int)
    if "source_id" in df.columns:
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").fillna(-1).astype("int64")
    else:
        df["source_id"] = -1
    if "field_tag" not in df.columns:
        df["field_tag"] = ""

    rows = [
        (
            int(r.test_index),
            int(r.source_id),
            str(r.field_tag),
            float(r.v_alpha_j2000),
            float(r.v_delta_j2000),
        )
        for r in df.itertuples(index=False)
    ]

    cols_sql = ", ".join([x.strip() for x in str(args.select_cols).split(",") if x.strip()])
    rad_deg = float(args.match_radius_arcsec) / 3600.0

    sql = f"""
    SELECT
      t.test_index,
      t.source_id,
      t.field_tag,
      t.ra_t,
      t.dec_t,
      m.*,
      (q3c_dist(t.ra_t, t.dec_t, m.v_alpha_j2000, m.v_delta_j2000) * 3600.0) AS wsdb_match_arcsec
    FROM tmp_forensics_targets t
    LEFT JOIN LATERAL (
      SELECT {cols_sql}
      FROM {args.schema}.{args.table}
      WHERE q3c_join(t.ra_t, t.dec_t, v_alpha_j2000, v_delta_j2000, {rad_deg})
      ORDER BY q3c_dist(t.ra_t, t.dec_t, v_alpha_j2000, v_delta_j2000) ASC
      LIMIT 1
    ) m ON TRUE
    ORDER BY t.test_index
    """

    with psycopg.connect(host=host, dbname=db, user=user, password=pw, port=port) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS tmp_forensics_targets")
            cur.execute(
                """
                CREATE TEMP TABLE tmp_forensics_targets (
                  test_index bigint,
                  source_id bigint,
                  field_tag text,
                  ra_t double precision,
                  dec_t double precision
                ) ON COMMIT DROP
                """
            )

            # Insert targets via COPY (psycopg3-safe, fast).
            for batch in chunked(rows, int(args.batch_insert)):
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerows(batch)
                buf.seek(0)
                with cur.copy(
                    "COPY tmp_forensics_targets(test_index,source_id,field_tag,ra_t,dec_t) FROM STDIN WITH (FORMAT CSV)"
                ) as cp:
                    cp.write(buf.read())

            # Query + stream to csv
            with conn.cursor(name="q_match") as s:
                s.itersize = 10000
                s.execute(sql)
                colnames = [d.name for d in s.description]
                with open(fout, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(colnames)
                    n = 0
                    while True:
                        chunk = s.fetchmany(10000)
                        if not chunk:
                            break
                        w.writerows(chunk)
                        n += len(chunk)
                        if n % 50000 == 0:
                            print(f"wrote_rows={n}")
            conn.commit()

    print("DONE")
    print("out_csv:", fout)


if __name__ == "__main__":
    main()
