#!/usr/bin/env python3
"""
Build a compact PSF-split experiment report with decision-ready tables.

Inputs:
- weak labels CSV (from scripts/nn/00_psf_labeling.py)
- optional MoE directory with gate probabilities + metrics
- optional run names for feature-importance comparison (full/psf/nonpsf)

Outputs (under --out_dir):
- labels_summary.json
- labels_by_split.csv
- gate_mixing_summary.json (if gate probabilities found)
- gate_routing_by_label.csv (if y_true available)
- feature_importance_compare_flux.csv (if run names provided)
- feature_importance_compare_pixels.csv (if run names provided)
- decision_summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


def _q(v: np.ndarray, q: float) -> float:
    vv = np.asarray(v, dtype=float)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return float("nan")
    return float(np.quantile(vv, q))


def _fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{100.0 * x:.2f}%"


def _feature_key_to_idx(k: str) -> Optional[int]:
    if not isinstance(k, str) or not k.startswith("f"):
        return None
    try:
        return int(k[1:])
    except Exception:
        return None


def _score_to_vec(score: Dict[str, float], n_features: int) -> np.ndarray:
    out = np.zeros(n_features, dtype=np.float64)
    for k, val in score.items():
        idx = _feature_key_to_idx(k)
        if idx is None or idx < 0 or idx >= n_features:
            continue
        out[idx] = float(val)
    return out


def _load_manifest(run_root: Path) -> Dict[str, object]:
    man = np.load(run_root / "manifest_arrays.npz", allow_pickle=True)
    return {
        "feature_cols": [str(x) for x in man["feature_cols"].tolist()],
        "n_features": int(np.asarray(man["n_features"]).reshape(())),
        "D": int(np.asarray(man["D"]).reshape(())),
    }


def _load_booster(path: Path) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(str(path))
    return b


def summarize_labels(labels_csv: Path, out_dir: Path, q_low: float, q_high: float) -> Dict[str, object]:
    df = pd.read_csv(labels_csv)
    req = {"source_id", "label", "confidence", "score_psf_like"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise RuntimeError(f"labels csv missing columns: {miss}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["score_psf_like"] = pd.to_numeric(df["score_psf_like"], errors="coerce")
    if "split_code" in df.columns:
        df["split_code"] = pd.to_numeric(df["split_code"], errors="coerce")

    df = df.dropna(subset=["label", "confidence", "score_psf_like"]).copy()
    df["label"] = df["label"].astype(int)

    counts = df["label"].value_counts().to_dict()
    n_total = int(len(df))

    conf_by_label = (
        df.groupby("label")["confidence"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    by_split = pd.DataFrame()
    if "split_code" in df.columns:
        by_split = (
            df.groupby(["split_code", "label"])
            .size()
            .reset_index(name="n")
            .sort_values(["split_code", "label"])
        )
        by_split.to_csv(out_dir / "labels_by_split.csv", index=False)

    lo_est = _q(df["score_psf_like"].to_numpy(), q_low)
    hi_est = _q(df["score_psf_like"].to_numpy(), q_high)

    gaia_override = None
    if "ruwe" in df.columns or "ipd_frac_multi_peak" in df.columns:
        ruwe = pd.to_numeric(df.get("ruwe", np.nan), errors="coerce")
        ipd = pd.to_numeric(df.get("ipd_frac_multi_peak", np.nan), errors="coerce")
        mask = (ruwe > 1.4) | (ipd > 0.1)
        tmp = df.copy()
        tmp["gaia_nonpsf_proxy"] = mask.fillna(False)
        gaia_override = (
            tmp.groupby("label")["gaia_nonpsf_proxy"].mean().to_dict()
        )

    out = {
        "labels_csv": str(labels_csv),
        "n_total_labeled": n_total,
        "n_psf_label0": int(counts.get(0, 0)),
        "n_nonpsf_label1": int(counts.get(1, 0)),
        "frac_psf_label0": float(counts.get(0, 0) / max(1, n_total)),
        "frac_nonpsf_label1": float(counts.get(1, 0) / max(1, n_total)),
        "confidence_by_label": conf_by_label.to_dict(orient="records"),
        "score_psf_like_quantile_low": float(q_low),
        "score_psf_like_quantile_high": float(q_high),
        "score_psf_like_threshold_est_low": lo_est,
        "score_psf_like_threshold_est_high": hi_est,
        "score_psf_like_global_q10": _q(df["score_psf_like"].to_numpy(), 0.10),
        "score_psf_like_global_q50": _q(df["score_psf_like"].to_numpy(), 0.50),
        "score_psf_like_global_q90": _q(df["score_psf_like"].to_numpy(), 0.90),
    }
    if gaia_override is not None:
        out["gaia_nonpsf_proxy_rate_by_label"] = {str(k): float(v) for k, v in gaia_override.items()}

    (out_dir / "labels_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def summarize_gate(
    moe_dir: Path,
    out_dir: Path,
    hard_low: float,
    hard_high: float,
) -> Optional[Dict[str, object]]:
    gate_csv = moe_dir / "gate_test_probabilities.csv"
    if not gate_csv.exists():
        return None

    g = pd.read_csv(gate_csv)
    if "p_nonpsf" not in g.columns:
        return None
    p = pd.to_numeric(g["p_nonpsf"], errors="coerce").dropna().to_numpy(dtype=float)
    if p.size == 0:
        return None

    n = p.size
    frac_psf = float(np.mean(p <= hard_low))
    frac_non = float(np.mean(p >= hard_high))
    frac_mix = float(np.mean((p > hard_low) & (p < hard_high)))

    out = {
        "gate_csv": str(gate_csv),
        "n_gate_test": int(n),
        "hard_low": float(hard_low),
        "hard_high": float(hard_high),
        "route_frac_psf_only": frac_psf,
        "route_frac_nonpsf_only": frac_non,
        "route_frac_soft_mix": frac_mix,
        "p_nonpsf_q10": _q(p, 0.10),
        "p_nonpsf_q50": _q(p, 0.50),
        "p_nonpsf_q90": _q(p, 0.90),
    }

    if "y_true" in g.columns:
        y = pd.to_numeric(g["y_true"], errors="coerce")
        yy = y.dropna().to_numpy(dtype=int)
        pp = pd.to_numeric(g.loc[y.notna(), "p_nonpsf"], errors="coerce").to_numpy(dtype=float)
        route = np.where(pp <= hard_low, "psf_only", np.where(pp >= hard_high, "nonpsf_only", "soft_mix"))
        by = pd.DataFrame({"y_true": yy, "route": route}).value_counts().reset_index(name="n")
        by = by.sort_values(["y_true", "route"])
        by.to_csv(out_dir / "gate_routing_by_label.csv", index=False)

    (out_dir / "gate_mixing_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def summarize_feature_importance(
    base_dir: Path,
    out_dir: Path,
    full_run: str,
    psf_run: str,
    nonpsf_run: str,
) -> Dict[str, str]:
    run_roots = {
        "full": base_dir / "output" / "ml_runs" / full_run,
        "psf": base_dir / "output" / "ml_runs" / psf_run,
        "nonpsf": base_dir / "output" / "ml_runs" / nonpsf_run,
    }
    for k, p in run_roots.items():
        if not p.exists():
            raise FileNotFoundError(f"Run missing ({k}): {p}")

    man = _load_manifest(run_roots["full"])
    feature_cols = man["feature_cols"]
    n_features = int(man["n_features"])
    D = int(man["D"])

    flux_tbl: Dict[str, np.ndarray] = {}
    pix_tbl: Dict[str, np.ndarray] = {}

    for role, root in run_roots.items():
        flux_b = _load_booster(root / "models" / "booster_flux.json")
        flux_tbl[role] = _score_to_vec(flux_b.get_score(importance_type="gain"), n_features)

        pix_paths = sorted((root / "models").glob("booster_pix_*.json"))
        if len(pix_paths) == 0:
            pix_tbl[role] = np.zeros(n_features, dtype=np.float64)
        else:
            if len(pix_paths) != D:
                print(f"[WARN] {role}: expected D={D} pixel models, found {len(pix_paths)}")
            rows = []
            for p in pix_paths:
                b = _load_booster(p)
                rows.append(_score_to_vec(b.get_score(importance_type="gain"), n_features))
            pix_tbl[role] = np.vstack(rows).mean(axis=0)

    df_flux = pd.DataFrame({
        "feature": feature_cols,
        "gain_full": flux_tbl["full"],
        "gain_psf": flux_tbl["psf"],
        "gain_nonpsf": flux_tbl["nonpsf"],
    })
    df_flux["delta_psf_vs_full"] = df_flux["gain_psf"] - df_flux["gain_full"]
    df_flux["delta_nonpsf_vs_full"] = df_flux["gain_nonpsf"] - df_flux["gain_full"]
    df_flux = df_flux.sort_values("gain_full", ascending=False)
    df_flux.to_csv(out_dir / "feature_importance_compare_flux.csv", index=False)

    df_pix = pd.DataFrame({
        "feature": feature_cols,
        "gain_full": pix_tbl["full"],
        "gain_psf": pix_tbl["psf"],
        "gain_nonpsf": pix_tbl["nonpsf"],
    })
    df_pix["delta_psf_vs_full"] = df_pix["gain_psf"] - df_pix["gain_full"]
    df_pix["delta_nonpsf_vs_full"] = df_pix["gain_nonpsf"] - df_pix["gain_full"]
    df_pix = df_pix.sort_values("gain_full", ascending=False)
    df_pix.to_csv(out_dir / "feature_importance_compare_pixels.csv", index=False)

    return {
        "flux_csv": str(out_dir / "feature_importance_compare_flux.csv"),
        "pixels_csv": str(out_dir / "feature_importance_compare_pixels.csv"),
    }


def build_decision_md(
    out_dir: Path,
    labels_summary: Dict[str, object],
    gate_summary: Optional[Dict[str, object]],
    fi_paths: Optional[Dict[str, str]],
) -> None:
    lines: List[str] = []
    lines.append("# PSF split decision summary")
    lines.append("")
    lines.append("## Label stats")
    lines.append(f"- Total labeled: {int(labels_summary['n_total_labeled'])}")
    lines.append(
        f"- PSF (label=0): {int(labels_summary['n_psf_label0'])} ({_fmt_pct(float(labels_summary['frac_psf_label0']))})"
    )
    lines.append(
        f"- non-PSF (label=1): {int(labels_summary['n_nonpsf_label1'])} ({_fmt_pct(float(labels_summary['frac_nonpsf_label1']))})"
    )
    lines.append(
        f"- Score thresholds (est): q{int(round(100*float(labels_summary['score_psf_like_quantile_low'])))}={labels_summary['score_psf_like_threshold_est_low']:.6g}, "
        f"q{int(round(100*float(labels_summary['score_psf_like_quantile_high'])))}={labels_summary['score_psf_like_threshold_est_high']:.6g}"
    )

    if gate_summary is not None:
        lines.append("")
        lines.append("## Routing / mixing stats")
        lines.append(f"- Gate test rows: {int(gate_summary['n_gate_test'])}")
        lines.append(f"- PSF-only route (p<=low): {_fmt_pct(float(gate_summary['route_frac_psf_only']))}")
        lines.append(f"- non-PSF-only route (p>=high): {_fmt_pct(float(gate_summary['route_frac_nonpsf_only']))}")
        lines.append(f"- soft-mix route (between): {_fmt_pct(float(gate_summary['route_frac_soft_mix']))}")

    if fi_paths is not None:
        lines.append("")
        lines.append("## Feature importance change tables")
        lines.append(f"- Flux importance diff: `{Path(fi_paths['flux_csv']).name}`")
        lines.append(f"- Pixel importance diff: `{Path(fi_paths['pixels_csv']).name}`")

    (out_dir / "decision_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--q_low", type=float, default=0.30)
    ap.add_argument("--q_high", type=float, default=0.70)
    ap.add_argument("--moe_dir", default="")
    ap.add_argument("--hard_low", type=float, default=0.10)
    ap.add_argument("--hard_high", type=float, default=0.90)
    ap.add_argument("--full_run", default="")
    ap.add_argument("--psf_run", default="")
    ap.add_argument("--nonpsf_run", default="")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_summary = summarize_labels(
        labels_csv=Path(args.labels_csv),
        out_dir=out_dir,
        q_low=float(args.q_low),
        q_high=float(args.q_high),
    )

    gate_summary = None
    if str(args.moe_dir).strip():
        gate_summary = summarize_gate(
            moe_dir=Path(args.moe_dir),
            out_dir=out_dir,
            hard_low=float(args.hard_low),
            hard_high=float(args.hard_high),
        )

    fi_paths = None
    if str(args.full_run).strip() and str(args.psf_run).strip() and str(args.nonpsf_run).strip():
        fi_paths = summarize_feature_importance(
            base_dir=base_dir,
            out_dir=out_dir,
            full_run=str(args.full_run),
            psf_run=str(args.psf_run),
            nonpsf_run=str(args.nonpsf_run),
        )

    build_decision_md(
        out_dir=out_dir,
        labels_summary=labels_summary,
        gate_summary=gate_summary,
        fi_paths=fi_paths,
    )

    print("DONE")
    print("Output directory:", out_dir)


if __name__ == "__main__":
    main()
