#!/usr/bin/env python3
"""
Query Gaia DR3 around near-cluster non-double sources to test whether Gaia resolves
one vs two+ sources.

Default groups (by parent double cluster):
- Type A: clusters 0 and 1
- Type B: cluster 2

Inputs:
- output/experiments/embeddings/double_stars_8d/umap_standard/non_double_neighbors_near_clusters_umap.csv
- output/dataset_npz/*/metadata.csv (for source RA/Dec)

Outputs:
- output/experiments/embeddings/double_stars_8d/umap_standard/gaia_multiplicity_check_non_cluster/
- plots/qa/embeddings/double_stars_8d/gaia_multiplicity_check_non_cluster/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlutilpy
from psycopg import OperationalError

CONN_BASE = dict(host="wsdb.ast.cam.ac.uk", db="wsdb")
GAIA_TABLE = "gaia_dr3.gaia_source"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def parse_cluster_list(s: str) -> List[int]:
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def load_source_coords(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(dataset_root.glob("*/metadata.csv")):
        df = pd.read_csv(p, usecols=lambda c: c in {"source_id", "ra", "dec", "fits_path"}, low_memory=False)
        for c in ["source_id", "ra", "dec"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["source_id", "ra", "dec"]).copy()
        df["source_id"] = df["source_id"].astype(np.int64)
        if "fits_path" in df.columns:
            df["field_tag"] = df["fits_path"].astype(str).map(lambda s: Path(s).parent.name if str(s) else p.parent.name)
        else:
            df["field_tag"] = p.parent.name
        rows.append(df[["source_id", "ra", "dec", "field_tag"]])

    allc = pd.concat(rows, ignore_index=True)
    allc = allc.sort_values("source_id").drop_duplicates(subset=["source_id"], keep="first")
    return allc.reset_index(drop=True)


def query_gaia_neighbors(tab: pd.DataFrame, max_radius_arcsec: float, conn: Dict[str, str]) -> pd.DataFrame:
    sid = tab["source_id"].to_numpy(dtype=np.int64)
    ra = tab["ra"].to_numpy(dtype=float)
    dec = tab["dec"].to_numpy(dtype=float)
    grp = tab["cluster_type"].astype(str).to_numpy(dtype=np.str_)
    clid = tab["double_cluster"].to_numpy(dtype=int)
    xid = np.arange(len(tab), dtype=np.int64)

    sql = f"""
    select
      m.source_id::bigint as source_id,
      m.cluster_type::text as cluster_type,
      m.double_cluster::int as double_cluster,
      coalesce(tt.n_other_r05, 0)::int as n_other_r05,
      coalesce(tt.n_other_r10, 0)::int as n_other_r10,
      coalesce(tt.n_other_r20, 0)::int as n_other_r20,
      tt.nn_other_arcsec::float8 as nn_other_arcsec
    from mytable m
    left join lateral (
      select
        count(*) filter (
          where g.source_id <> m.source_id
            and q3c_dist(m.ra, m.dec, g.ra, g.dec) * 3600.0 <= 0.5
        ) as n_other_r05,
        count(*) filter (
          where g.source_id <> m.source_id
            and q3c_dist(m.ra, m.dec, g.ra, g.dec) * 3600.0 <= 1.0
        ) as n_other_r10,
        count(*) filter (
          where g.source_id <> m.source_id
            and q3c_dist(m.ra, m.dec, g.ra, g.dec) * 3600.0 <= 2.0
        ) as n_other_r20,
        min(
          case
            when g.source_id <> m.source_id then q3c_dist(m.ra, m.dec, g.ra, g.dec) * 3600.0
            else null
          end
        ) as nn_other_arcsec
      from {GAIA_TABLE} g
      where q3c_join(m.ra, m.dec, g.ra, g.dec, {float(max_radius_arcsec)}/3600.0)
    ) tt on true
    order by xid
    """

    out = sqlutilpy.local_join(
        sql,
        "mytable",
        (sid, ra, dec, grp, clid, xid),
        ("source_id", "ra", "dec", "cluster_type", "double_cluster", "xid"),
        **conn,
    )

    cols = ["source_id", "cluster_type", "double_cluster", "n_other_r05", "n_other_r10", "n_other_r20", "nn_other_arcsec"]
    qdf = pd.DataFrame({c: v for c, v in zip(cols, out)})

    qdf["source_id"] = pd.to_numeric(qdf["source_id"], errors="coerce").astype("Int64")
    qdf["double_cluster"] = pd.to_numeric(qdf["double_cluster"], errors="coerce").astype("Int64")
    for c in ["n_other_r05", "n_other_r10", "n_other_r20"]:
        qdf[c] = pd.to_numeric(qdf[c], errors="coerce").fillna(0).astype(int)
    qdf["nn_other_arcsec"] = pd.to_numeric(qdf["nn_other_arcsec"], errors="coerce")

    return qdf


def write_query_pack(tab: pd.DataFrame, out_tab: Path, max_radius_arcsec: float) -> None:
    inp = out_tab / "sources_to_check.csv"
    tab[["source_id", "ra", "dec", "cluster_type", "double_cluster"]].to_csv(inp, index=False)

    sql = f"""-- Run this in PostgreSQL where gaia_dr3.gaia_source is available.
-- 1) Load the CSV generated by this script:
--    CREATE TEMP TABLE mytable (
--      source_id bigint, ra double precision, dec double precision,
--      cluster_type text, double_cluster int
--    );
--    \\copy mytable FROM 'sources_to_check.csv' WITH (FORMAT csv, HEADER true);
--
-- 2) Run query:
select
  m.source_id,
  m.cluster_type,
  m.double_cluster,
  coalesce(tt.n_other_r05, 0)::int as n_other_r05,
  coalesce(tt.n_other_r10, 0)::int as n_other_r10,
  coalesce(tt.n_other_r20, 0)::int as n_other_r20,
  tt.nn_other_arcsec::float8 as nn_other_arcsec
from mytable m
left join lateral (
  select
    count(*) filter (
      where g.source_id <> m.source_id
        and q3c_dist(m.ra, m.dec, g.ra, g.dec) * 3600.0 <= 0.5
    ) as n_other_r05,
    count(*) filter (
      where g.source_id <> m.source_id
        and q3c_dist(m.ra, m.dec, g.ra, g.dec) * 3600.0 <= 1.0
    ) as n_other_r10,
    count(*) filter (
      where g.source_id <> m.source_id
        and q3c_dist(m.ra, m.dec, g.ra, g.dec) * 3600.0 <= 2.0
    ) as n_other_r20,
    min(
      case
        when g.source_id <> m.source_id then q3c_dist(m.ra, m.dec, g.ra, g.dec) * 3600.0
        else null
      end
    ) as nn_other_arcsec
  from {GAIA_TABLE} g
  where q3c_join(m.ra, m.dec, g.ra, g.dec, {float(max_radius_arcsec)}/3600.0)
) tt on true;
"""
    (out_tab / "gaia_multiplicity_query_template.sql").write_text(sql, encoding="utf-8")


def summarize_by_type(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, sub in df.groupby("cluster_type"):
        n = len(sub)
        rows.append(
            {
                "cluster_type": t,
                "n_sources": int(n),
                "frac_has_other_r05": float(np.mean(sub["n_other_r05"] >= 1)),
                "frac_has_other_r10": float(np.mean(sub["n_other_r10"] >= 1)),
                "frac_has_other_r20": float(np.mean(sub["n_other_r20"] >= 1)),
                "mean_n_other_r10": float(np.mean(sub["n_other_r10"])),
                "median_n_other_r10": float(np.median(sub["n_other_r10"])),
                "median_nn_other_arcsec": float(np.nanmedian(sub["nn_other_arcsec"])),
            }
        )
    return pd.DataFrame(rows)


def build_clear_multiplicity_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["gaia_status_r10"] = np.where(out["n_other_r10"] >= 1, "multiple_detected", "single_detected")
    out["gaia_status_r20"] = np.where(out["n_other_r20"] >= 1, "multiple_detected", "single_detected")

    rows = []
    for t, sub in out.groupby("cluster_type"):
        n = len(sub)
        n_mult10 = int((sub["n_other_r10"] >= 1).sum())
        n_mult20 = int((sub["n_other_r20"] >= 1).sum())
        rows.append(
            {
                "cluster_type": t,
                "n_sources": int(n),
                "n_single_r10": int(n - n_mult10),
                "n_multiple_r10": int(n_mult10),
                "frac_multiple_r10": float(n_mult10 / n) if n else np.nan,
                "n_single_r20": int(n - n_mult20),
                "n_multiple_r20": int(n_mult20),
                "frac_multiple_r20": float(n_mult20 / n) if n else np.nan,
            }
        )
    clear_summary = pd.DataFrame(rows)
    return out, clear_summary


def plot_presence_bars(summary: pd.DataFrame, out_png: Path) -> None:
    order = ["NC_A_0plus1", "NC_B_2"]
    s = summary.set_index("cluster_type").reindex(order).dropna(how="all").reset_index()

    x = np.arange(len(s))
    w = 0.24

    plt.figure(figsize=(8, 4.8))
    plt.bar(x - w, s["frac_has_other_r05"], width=w, label="<=0.5\"")
    plt.bar(x, s["frac_has_other_r10"], width=w, label="<=1.0\"")
    plt.bar(x + w, s["frac_has_other_r20"], width=w, label="<=2.0\"")

    plt.xticks(x, s["cluster_type"])
    plt.ylim(0, 1.0)
    plt.ylabel("Fraction with >=1 extra Gaia source")
    plt.title("Gaia multiplicity by near-cluster type")
    plt.legend(frameon=True)
    savefig(out_png)


def plot_nn_hist(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    for t, color in [("NC_A_0plus1", "#d62828"), ("NC_B_2", "#1d3557")]:
        v = pd.to_numeric(df.loc[df["cluster_type"] == t, "nn_other_arcsec"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(v) == 0:
            continue
        v = v[v <= 5.0]
        if len(v) == 0:
            continue
        plt.hist(v, bins=20, density=True, histtype="step", linewidth=2, label=t, color=color)
    plt.xlabel("Nearest other Gaia source distance [arcsec]")
    plt.ylabel("Density")
    plt.title("Nearest-neighbor Gaia distance distribution (near-cluster non-doubles)")
    plt.legend(frameon=True)
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check Gaia multiplicity around near-cluster non-double sources")
    ap.add_argument("--dataset_root", default="/data/yn316/Codes/output/dataset_npz")
    ap.add_argument("--near_csv", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/non_double_neighbors_near_clusters_umap.csv")
    ap.add_argument("--group_a", default="0,1")
    ap.add_argument("--group_b", default="2")
    ap.add_argument("--max_radius_arcsec", type=float, default=5.0)
    ap.add_argument("--db_host", default=os.getenv("GAIA_DB_HOST", CONN_BASE["host"]))
    ap.add_argument("--db_name", default=os.getenv("GAIA_DB_NAME", CONN_BASE["db"]))
    ap.add_argument("--db_user", default=os.getenv("GAIA_DB_USER", "yasmine_nourlil_2026"))
    ap.add_argument("--db_password", default=os.getenv("GAIA_DB_PASSWORD", os.getenv("PGPASSWORD", "")))
    ap.add_argument("--query_result_csv", default="", help="Optional path to external SQL result CSV to skip live DB query")
    ap.add_argument("--out_tab_dir", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/gaia_multiplicity_check_non_cluster")
    ap.add_argument("--out_plot_dir", default="/data/yn316/Codes/plots/qa/embeddings/double_stars_8d/gaia_multiplicity_check_non_cluster")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    near_csv = Path(args.near_csv)
    out_tab = ensure_dir(Path(args.out_tab_dir))
    out_plot = ensure_dir(Path(args.out_plot_dir))

    a = parse_cluster_list(args.group_a)
    b = parse_cluster_list(args.group_b)

    near = pd.read_csv(near_csv)
    near["double_cluster"] = pd.to_numeric(near["double_cluster"], errors="coerce")
    near = near.dropna(subset=["double_cluster", "source_id"]).copy()
    near["double_cluster"] = near["double_cluster"].astype(int)

    near = near[near["double_cluster"].isin(a + b)].copy()
    near["cluster_type"] = np.where(near["double_cluster"].isin(a), "NC_A_0plus1", "NC_B_2")

    coords = load_source_coords(dataset_root)
    tab = near.merge(coords[["source_id", "ra", "dec", "field_tag"]], on="source_id", how="left")
    tab = tab.dropna(subset=["ra", "dec"]).copy()

    write_query_pack(tab, out_tab, max_radius_arcsec=float(args.max_radius_arcsec))

    qdf = None
    if str(args.query_result_csv).strip():
        qpath = Path(str(args.query_result_csv))
        if not qpath.exists():
            raise FileNotFoundError(f"--query_result_csv not found: {qpath}")
        qdf = pd.read_csv(qpath)
        print(f"Loaded external query result CSV: {qpath}")
    else:
        conn = dict(host=args.db_host, db=args.db_name, user=args.db_user)
        if args.db_password:
            conn["password"] = args.db_password

        if "password" not in conn:
            print("Gaia query skipped: no DB password provided.")
            print("Set GAIA_DB_PASSWORD or pass --db_password to run the live query.")
            print("Prepared query pack:")
            print(" ", out_tab / "sources_to_check.csv")
            print(" ", out_tab / "gaia_multiplicity_query_template.sql")
            print("Then rerun this script with --query_result_csv <your_sql_output.csv>")
            return

        print(f"Querying Gaia neighbors for {len(tab)} near-cluster non-double sources...")
        try:
            qdf = query_gaia_neighbors(tab, max_radius_arcsec=float(args.max_radius_arcsec), conn=conn)
        except Exception as e:
            msg = str(e)
            if isinstance(e, OperationalError) or "failed to resolve host" in msg.lower() or "fe_sendauth" in msg.lower():
                print("Gaia query skipped due to DB connection/auth issue.")
                print(f"Connection error detail: {e}")
                print("Prepared query pack:")
                print(" ", out_tab / "sources_to_check.csv")
                print(" ", out_tab / "gaia_multiplicity_query_template.sql")
                print("Then rerun this script with --query_result_csv <your_sql_output.csv>")
                return
            raise

    # sanitize / cast expected columns for both live and external query paths
    expected = ["source_id", "cluster_type", "double_cluster", "n_other_r05", "n_other_r10", "n_other_r20", "nn_other_arcsec"]
    missing_cols = [c for c in expected if c not in qdf.columns]
    if missing_cols:
        raise ValueError(f"query result is missing required columns: {missing_cols}")
    qdf = qdf[expected].copy()
    qdf["source_id"] = pd.to_numeric(qdf["source_id"], errors="coerce").astype("Int64")
    qdf["double_cluster"] = pd.to_numeric(qdf["double_cluster"], errors="coerce").astype("Int64")
    for c in ["n_other_r05", "n_other_r10", "n_other_r20"]:
        qdf[c] = pd.to_numeric(qdf[c], errors="coerce").fillna(0).astype(int)
    qdf["nn_other_arcsec"] = pd.to_numeric(qdf["nn_other_arcsec"], errors="coerce")

    out = tab.merge(qdf, on=["source_id", "cluster_type", "double_cluster"], how="left")
    clear_per_source, clear_summary = build_clear_multiplicity_tables(out)
    summary = summarize_by_type(out)

    out.to_csv(out_tab / "gaia_neighbor_counts_per_source.csv", index=False)
    summary.to_csv(out_tab / "gaia_neighbor_summary_by_cluster_type.csv", index=False)
    clear_per_source.to_csv(out_tab / "gaia_multiplicity_per_source_clear.csv", index=False)
    clear_summary.to_csv(out_tab / "gaia_multiplicity_summary_clear.csv", index=False)

    plot_presence_bars(summary, out_plot / "01_fraction_with_extra_gaia_sources.png")
    plot_nn_hist(out, out_plot / "02_nearest_other_gaia_distance_hist.png")

    print("Done.")
    print("  Per-source:", out_tab / "gaia_neighbor_counts_per_source.csv")
    print("  Summary   :", out_tab / "gaia_neighbor_summary_by_cluster_type.csv")
    print("  Clear per-source:", out_tab / "gaia_multiplicity_per_source_clear.csv")
    print("  Clear summary   :", out_tab / "gaia_multiplicity_summary_clear.csv")
    print("  Plots     :", out_plot)


if __name__ == "__main__":
    main()
