#!/usr/bin/env python3
"""
Common-proper-motion (CPM) check for WDS doubles vs near-cluster non-doubles.

This script compares proper-motion consistency using Gaia nearest neighbors:
- WDS-labeled doubles
- near-cluster non-doubles (optionally only Gaia-multiple at r10)
- control non-double sample

Inputs (defaults):
- output/experiments/embeddings/double_stars_8d/umap_standard/double_candidates_ranked_by_local_enrichment_umap.csv
- output/experiments/embeddings/double_stars_8d/umap_standard/gaia_multiplicity_check_non_cluster/gaia_multiplicity_per_source_clear.csv
- output/experiments/embeddings/double_stars_8d/umap_standard/embedding_umap.csv
- output/dataset_npz/*/metadata.csv

Outputs:
- output/experiments/embeddings/double_stars_8d/umap_standard/cpm_check/
- plots/qa/embeddings/double_stars_8d/cpm_check/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
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


def build_samples(
    doubles_csv: Path,
    near_csv: Path,
    embedding_csv: Path,
    seed: int,
    control_n: int,
    near_require_multiple_r10: bool,
) -> pd.DataFrame:
    d = pd.read_csv(doubles_csv)
    d["source_id"] = pd.to_numeric(d["source_id"], errors="coerce")
    d["double_cluster"] = pd.to_numeric(d.get("double_cluster"), errors="coerce")
    d = d.dropna(subset=["source_id"]).copy()
    d["source_id"] = d["source_id"].astype(np.int64)
    d["sample_type"] = "WDS_DOUBLE"
    d["cluster_label"] = d["double_cluster"].fillna(-99).astype(int).astype(str).map(lambda s: f"WDS_cluster_{s}")
    d = d[["source_id", "sample_type", "cluster_label"]].drop_duplicates(subset=["source_id"])

    n = pd.read_csv(near_csv)
    n["source_id"] = pd.to_numeric(n["source_id"], errors="coerce")
    n = n.dropna(subset=["source_id"]).copy()
    n["source_id"] = n["source_id"].astype(np.int64)
    if near_require_multiple_r10 and "gaia_status_r10" in n.columns:
        n = n[n["gaia_status_r10"].astype(str) == "multiple_detected"].copy()
    if "cluster_type" not in n.columns:
        n["cluster_type"] = "NEAR_unknown"
    n["sample_type"] = "NEAR_NONDOUBLE"
    n["cluster_label"] = n["cluster_type"].astype(str)
    n = n[["source_id", "sample_type", "cluster_label"]].drop_duplicates(subset=["source_id"])

    e = pd.read_csv(embedding_csv, usecols=lambda c: c in {"source_id", "is_double", "double_cluster"})
    e["source_id"] = pd.to_numeric(e["source_id"], errors="coerce")
    e["is_double"] = e["is_double"].fillna(False).astype(bool)
    if "double_cluster" in e.columns:
        e["double_cluster"] = pd.to_numeric(e["double_cluster"], errors="coerce")
    e = e.dropna(subset=["source_id"]).copy()
    e["source_id"] = e["source_id"].astype(np.int64)

    exclude = set(d["source_id"].tolist()) | set(n["source_id"].tolist())
    ctrl_pool = e[~e["is_double"] & ~e["source_id"].isin(exclude)].copy()
    if len(ctrl_pool) == 0:
        ctrl = pd.DataFrame(columns=["source_id", "sample_type", "cluster_label"])
    else:
        n_take = min(control_n if control_n > 0 else len(n), len(ctrl_pool))
        ctrl = ctrl_pool.sample(n=n_take, random_state=seed).copy()
        ctrl["sample_type"] = "CONTROL_NONDOUBLE"
        ctrl["cluster_label"] = "CONTROL"
        ctrl = ctrl[["source_id", "sample_type", "cluster_label"]]

    all_samples = pd.concat([d, n, ctrl], ignore_index=True)
    all_samples = all_samples.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return all_samples


def query_gaia_nearest_neighbor_pm(tab: pd.DataFrame, conn: Dict[str, str], max_radius_arcsec: float) -> pd.DataFrame:
    sid = tab["source_id"].to_numpy(dtype=np.int64)
    ra = tab["ra"].to_numpy(dtype=float)
    dec = tab["dec"].to_numpy(dtype=float)
    st = tab["sample_type"].astype(str).to_numpy(dtype=np.str_)
    cl = tab["cluster_label"].astype(str).to_numpy(dtype=np.str_)
    xid = np.arange(len(tab), dtype=np.int64)

    sql = f"""
    select
      m.source_id::bigint as source_id,
      m.sample_type::text as sample_type,
      m.cluster_label::text as cluster_label,
      self.pmra::float8 as pmra,
      self.pmdec::float8 as pmdec,
      self.pmra_error::float8 as pmra_error,
      self.pmdec_error::float8 as pmdec_error,
      nn.nn_source_id::bigint as nn_source_id,
      nn.nn_pmra::float8 as nn_pmra,
      nn.nn_pmdec::float8 as nn_pmdec,
      nn.nn_pmra_error::float8 as nn_pmra_error,
      nn.nn_pmdec_error::float8 as nn_pmdec_error,
      nn.nn_dist_arcsec::float8 as nn_dist_arcsec
    from mytable m
    left join {GAIA_TABLE} self
      on self.source_id = m.source_id
    left join lateral (
      select
        g.source_id as nn_source_id,
        g.pmra as nn_pmra,
        g.pmdec as nn_pmdec,
        g.pmra_error as nn_pmra_error,
        g.pmdec_error as nn_pmdec_error,
        q3c_dist(m.ra, m.dec, g.ra, g.dec) * 3600.0 as nn_dist_arcsec
      from {GAIA_TABLE} g
      where g.source_id <> m.source_id
        and q3c_join(m.ra, m.dec, g.ra, g.dec, {float(max_radius_arcsec)}/3600.0)
      order by q3c_dist(m.ra, m.dec, g.ra, g.dec) asc
      limit 1
    ) nn on true
    order by xid
    """

    out = sqlutilpy.local_join(
        sql,
        "mytable",
        (sid, ra, dec, st, cl, xid),
        ("source_id", "ra", "dec", "sample_type", "cluster_label", "xid"),
        **conn,
    )

    cols = [
        "source_id",
        "sample_type",
        "cluster_label",
        "pmra",
        "pmdec",
        "pmra_error",
        "pmdec_error",
        "nn_source_id",
        "nn_pmra",
        "nn_pmdec",
        "nn_pmra_error",
        "nn_pmdec_error",
        "nn_dist_arcsec",
    ]
    qdf = pd.DataFrame({c: v for c, v in zip(cols, out)})

    for c in [
        "source_id",
        "nn_source_id",
        "pmra",
        "pmdec",
        "pmra_error",
        "pmdec_error",
        "nn_pmra",
        "nn_pmdec",
        "nn_pmra_error",
        "nn_pmdec_error",
        "nn_dist_arcsec",
    ]:
        if c in qdf.columns:
            qdf[c] = pd.to_numeric(qdf[c], errors="coerce")
    qdf["source_id"] = qdf["source_id"].astype("Int64")
    qdf["nn_source_id"] = qdf["nn_source_id"].astype("Int64")
    return qdf


def write_query_pack(tab: pd.DataFrame, out_tab: Path, max_radius_arcsec: float) -> None:
    inp = out_tab / "sources_to_check_cpm.csv"
    tab[["source_id", "ra", "dec", "sample_type", "cluster_label"]].to_csv(inp, index=False)

    sql = f"""-- Run this in PostgreSQL where gaia_dr3.gaia_source is available.
-- 1) CREATE TEMP TABLE mytable (
--      source_id bigint, ra double precision, dec double precision,
--      sample_type text, cluster_label text
--    );
-- 2) \\copy mytable FROM 'sources_to_check_cpm.csv' WITH (FORMAT csv, HEADER true);
-- 3) Run this query:
select
  m.source_id::bigint as source_id,
  m.sample_type::text as sample_type,
  m.cluster_label::text as cluster_label,
  self.pmra::float8 as pmra,
  self.pmdec::float8 as pmdec,
  self.pmra_error::float8 as pmra_error,
  self.pmdec_error::float8 as pmdec_error,
  nn.nn_source_id::bigint as nn_source_id,
  nn.nn_pmra::float8 as nn_pmra,
  nn.nn_pmdec::float8 as nn_pmdec,
  nn.nn_pmra_error::float8 as nn_pmra_error,
  nn.nn_pmdec_error::float8 as nn_pmdec_error,
  nn.nn_dist_arcsec::float8 as nn_dist_arcsec
from mytable m
left join {GAIA_TABLE} self
  on self.source_id = m.source_id
left join lateral (
  select
    g.source_id as nn_source_id,
    g.pmra as nn_pmra,
    g.pmdec as nn_pmdec,
    g.pmra_error as nn_pmra_error,
    g.pmdec_error as nn_pmdec_error,
    q3c_dist(m.ra, m.dec, g.ra, g.dec) * 3600.0 as nn_dist_arcsec
  from {GAIA_TABLE} g
  where g.source_id <> m.source_id
    and q3c_join(m.ra, m.dec, g.ra, g.dec, {float(max_radius_arcsec)}/3600.0)
  order by q3c_dist(m.ra, m.dec, g.ra, g.dec) asc
  limit 1
) nn on true;
"""
    (out_tab / "gaia_cpm_query_template.sql").write_text(sql, encoding="utf-8")


def compute_cpm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in [
        "pmra",
        "pmdec",
        "pmra_error",
        "pmdec_error",
        "nn_pmra",
        "nn_pmdec",
        "nn_pmra_error",
        "nn_pmdec_error",
        "nn_dist_arcsec",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["delta_pmra"] = out["pmra"] - out["nn_pmra"]
    out["delta_pmdec"] = out["pmdec"] - out["nn_pmdec"]
    out["delta_mu"] = np.sqrt(out["delta_pmra"] ** 2 + out["delta_pmdec"] ** 2)

    sigma1 = np.sqrt(np.clip(out["pmra_error"], 0, None) ** 2 + np.clip(out["pmdec_error"], 0, None) ** 2)
    sigma2 = np.sqrt(np.clip(out["nn_pmra_error"], 0, None) ** 2 + np.clip(out["nn_pmdec_error"], 0, None) ** 2)
    out["sigma_delta_mu"] = np.sqrt(sigma1**2 + sigma2**2)
    out["delta_mu_sigma"] = out["delta_mu"] / out["sigma_delta_mu"].replace(0, np.nan)

    mu1 = np.sqrt(out["pmra"] ** 2 + out["pmdec"] ** 2)
    mu2 = np.sqrt(out["nn_pmra"] ** 2 + out["nn_pmdec"] ** 2)
    dot = out["pmra"] * out["nn_pmra"] + out["pmdec"] * out["nn_pmdec"]
    denom = mu1 * mu2
    cosang = (dot / denom).clip(-1, 1)
    out["delta_mu_angle_deg"] = np.degrees(np.arccos(cosang))
    out.loc[~np.isfinite(out["delta_mu_angle_deg"]), "delta_mu_angle_deg"] = np.nan

    return out


def summarize_by_sample(df: pd.DataFrame, cpm_dmu_thr: float, cpm_sigma_thr: float) -> pd.DataFrame:
    rows = []
    for s, sub in df.groupby("sample_type", sort=False):
        vals = pd.to_numeric(sub["delta_mu"], errors="coerce")
        sig = pd.to_numeric(sub["delta_mu_sigma"], errors="coerce")
        dist = pd.to_numeric(sub["nn_dist_arcsec"], errors="coerce")
        ok = vals.notna()
        n = int(ok.sum())
        rows.append(
            {
                "sample_type": s,
                "n_with_delta_mu": n,
                "delta_mu_median_masyr": float(np.nanmedian(vals)) if n else np.nan,
                "delta_mu_p25_masyr": float(np.nanpercentile(vals, 25)) if n else np.nan,
                "delta_mu_p75_masyr": float(np.nanpercentile(vals, 75)) if n else np.nan,
                "nn_dist_median_arcsec": float(np.nanmedian(dist)) if n else np.nan,
                "frac_cpm_delta_mu_lt_thr": float(np.nanmean(vals < cpm_dmu_thr)) if n else np.nan,
                "frac_cpm_sigma_lt_thr": float(np.nanmean(sig < cpm_sigma_thr)) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_hist_delta_mu(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(8.4, 5.2))
    for sample, color in [("WDS_DOUBLE", "#d62828"), ("NEAR_NONDOUBLE", "#1d3557"), ("CONTROL_NONDOUBLE", "#2a9d8f")]:
        v = pd.to_numeric(df.loc[df["sample_type"] == sample, "delta_mu"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(v) == 0:
            continue
        v = v[(v > 0) & np.isfinite(v)]
        if len(v) == 0:
            continue
        plt.hist(v, bins=np.logspace(np.log10(max(1e-3, np.min(v))), np.log10(max(v)), 35), histtype="step", linewidth=2, color=color, label=f"{sample} (n={len(v)})")
    plt.xscale("log")
    plt.xlabel("Delta mu [mas/yr]")
    plt.ylabel("Count")
    plt.title("Nearest-neighbor proper-motion difference")
    plt.legend(frameon=True, fontsize=9)
    savefig(out_png)


def plot_cdf_delta_mu(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(8.4, 5.2))
    for sample, color in [("WDS_DOUBLE", "#d62828"), ("NEAR_NONDOUBLE", "#1d3557"), ("CONTROL_NONDOUBLE", "#2a9d8f")]:
        v = pd.to_numeric(df.loc[df["sample_type"] == sample, "delta_mu"], errors="coerce").dropna().to_numpy(dtype=float)
        v = np.sort(v[(v > 0) & np.isfinite(v)])
        if len(v) == 0:
            continue
        y = np.arange(1, len(v) + 1, dtype=float) / len(v)
        plt.step(v, y, where="post", color=color, linewidth=2, label=f"{sample} (n={len(v)})")
    plt.xscale("log")
    plt.ylim(0, 1.0)
    plt.xlabel("Delta mu [mas/yr]")
    plt.ylabel("ECDF")
    plt.title("Cumulative delta mu comparison")
    plt.legend(frameon=True, fontsize=9, loc="lower right")
    savefig(out_png)


def plot_delta_mu_vs_sep(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(8.4, 5.2))
    for sample, color in [("WDS_DOUBLE", "#d62828"), ("NEAR_NONDOUBLE", "#1d3557"), ("CONTROL_NONDOUBLE", "#2a9d8f")]:
        sub = df[df["sample_type"] == sample].copy()
        x = pd.to_numeric(sub["nn_dist_arcsec"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["delta_mu"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if m.sum() == 0:
            continue
        plt.scatter(x[m], y[m], s=18, alpha=0.55, color=color, linewidths=0, label=f"{sample} (n={int(m.sum())})")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Nearest neighbor separation [arcsec]")
    plt.ylabel("Delta mu [mas/yr]")
    plt.title("Delta mu vs angular separation")
    plt.legend(frameon=True, fontsize=8)
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="CPM check for WDS doubles vs near non-doubles")
    ap.add_argument("--dataset_root", default="/data/yn316/Codes/output/dataset_npz")
    ap.add_argument("--doubles_csv", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/double_candidates_ranked_by_local_enrichment_umap.csv")
    ap.add_argument("--near_csv", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/gaia_multiplicity_check_non_cluster/gaia_multiplicity_per_source_clear.csv")
    ap.add_argument("--embedding_csv", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/embedding_umap.csv")
    ap.add_argument("--near_require_multiple_r10", action="store_true", help="Keep only near non-doubles with gaia_status_r10=multiple_detected.")
    ap.add_argument("--control_n", type=int, default=90, help="Control sample size.")
    ap.add_argument("--max_radius_arcsec", type=float, default=5.0)
    ap.add_argument("--db_host", default=os.getenv("GAIA_DB_HOST", CONN_BASE["host"]))
    ap.add_argument("--db_name", default=os.getenv("GAIA_DB_NAME", CONN_BASE["db"]))
    ap.add_argument("--db_user", default=os.getenv("GAIA_DB_USER", "yasmine_nourlil_2026"))
    ap.add_argument("--db_password", default=os.getenv("GAIA_DB_PASSWORD", os.getenv("PGPASSWORD", "")))
    ap.add_argument("--query_result_csv", default="", help="Optional path to external SQL result CSV to skip live DB query")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpm_dmu_thr", type=float, default=2.0, help="CPM-like threshold in mas/yr for delta_mu.")
    ap.add_argument("--cpm_sigma_thr", type=float, default=3.0, help="CPM-like threshold for delta_mu_sigma.")
    ap.add_argument("--out_tab_dir", default="/data/yn316/Codes/output/experiments/embeddings/double_stars_8d/umap_standard/cpm_check")
    ap.add_argument("--out_plot_dir", default="/data/yn316/Codes/plots/qa/embeddings/double_stars_8d/cpm_check")
    args = ap.parse_args()

    out_tab = ensure_dir(Path(args.out_tab_dir))
    out_plot = ensure_dir(Path(args.out_plot_dir))

    samples = build_samples(
        doubles_csv=Path(args.doubles_csv),
        near_csv=Path(args.near_csv),
        embedding_csv=Path(args.embedding_csv),
        seed=int(args.seed),
        control_n=int(args.control_n),
        near_require_multiple_r10=bool(args.near_require_multiple_r10),
    )
    coords = load_source_coords(Path(args.dataset_root))
    tab = samples.merge(coords[["source_id", "ra", "dec", "field_tag"]], on="source_id", how="left")
    tab = tab.dropna(subset=["ra", "dec"]).copy()
    tab.to_csv(out_tab / "cpm_input_samples_with_coords.csv", index=False)

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
            print("CPM query skipped: no DB password provided.")
            print("Prepared query pack:")
            print(" ", out_tab / "sources_to_check_cpm.csv")
            print(" ", out_tab / "gaia_cpm_query_template.sql")
            print("Then rerun this script with --query_result_csv <your_sql_output.csv>")
            return

        print(f"Querying Gaia nearest-neighbor PM for {len(tab)} sources...")
        try:
            qdf = query_gaia_nearest_neighbor_pm(tab, conn=conn, max_radius_arcsec=float(args.max_radius_arcsec))
        except Exception as e:
            msg = str(e)
            if isinstance(e, OperationalError) or "failed to resolve host" in msg.lower() or "fe_sendauth" in msg.lower():
                print("CPM query skipped due to DB connection/auth issue.")
                print(f"Connection error detail: {e}")
                print("Prepared query pack:")
                print(" ", out_tab / "sources_to_check_cpm.csv")
                print(" ", out_tab / "gaia_cpm_query_template.sql")
                print("Then rerun this script with --query_result_csv <your_sql_output.csv>")
                return
            raise

    expected = [
        "source_id",
        "sample_type",
        "cluster_label",
        "pmra",
        "pmdec",
        "pmra_error",
        "pmdec_error",
        "nn_source_id",
        "nn_pmra",
        "nn_pmdec",
        "nn_pmra_error",
        "nn_pmdec_error",
        "nn_dist_arcsec",
    ]
    miss = [c for c in expected if c not in qdf.columns]
    if miss:
        raise ValueError(f"query result is missing required columns: {miss}")

    qdf = qdf[expected].copy()
    out = compute_cpm_metrics(qdf)
    summary = summarize_by_sample(out, cpm_dmu_thr=float(args.cpm_dmu_thr), cpm_sigma_thr=float(args.cpm_sigma_thr))

    out.to_csv(out_tab / "cpm_per_source.csv", index=False)
    summary.to_csv(out_tab / "cpm_summary_by_sample.csv", index=False)

    plot_hist_delta_mu(out, out_plot / "01_delta_mu_hist_by_sample.png")
    plot_cdf_delta_mu(out, out_plot / "02_delta_mu_ecdf_by_sample.png")
    plot_delta_mu_vs_sep(out, out_plot / "03_delta_mu_vs_sep_by_sample.png")

    print("Done.")
    print("  Per-source:", out_tab / "cpm_per_source.csv")
    print("  Summary   :", out_tab / "cpm_summary_by_sample.csv")
    print("  Plots     :", out_plot)


if __name__ == "__main__":
    main()
