"""
Aggregate feature importance for pixels+flux XGBoost runs.

Reads:
  output/ml_runs/<RUN>/manifest_arrays.npz
  output/ml_runs/<RUN>/models/booster_flux.json
  output/ml_runs/<RUN>/models/booster_pix_*.json

Writes:
  plots/ml_runs/<RUN>/feature_importance/
    - feature_importance_flux_gain.csv
    - feature_importance_pixels_gain.csv
    - feature_importance_flux_top20.png
    - feature_importance_pixels_top20_mean_gain.png
    - feature_importance_pixels_nonzero_counts.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def load_manifest(run_root: Path) -> Dict:
    p = run_root / "manifest_arrays.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing manifest: {p}")
    d = np.load(p, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["feature_cols"] = [str(x) for x in out["feature_cols"].tolist()]
    out["n_features"] = int(out["n_features"])
    out["D"] = int(out["D"])
    return out


def feature_key_to_idx(k: str) -> int | None:
    # XGBoost uses feature ids like "f0", "f1", ...
    if not isinstance(k, str) or not k.startswith("f"):
        return None
    try:
        return int(k[1:])
    except Exception:
        return None


def score_to_vector(score: Dict[str, float], n_features: int) -> np.ndarray:
    v = np.zeros(n_features, dtype=np.float64)
    for k, val in score.items():
        idx = feature_key_to_idx(k)
        if idx is None or idx < 0 or idx >= n_features:
            continue
        v[idx] = float(val)
    return v


def load_booster(path: Path) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(str(path))
    return b


def top_bar(df: pd.DataFrame, value_col: str, out_png: Path, title: str, top_n: int = 20) -> None:
    d = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(8.5, 6.0))
    plt.barh(d["feature"], d[value_col])
    plt.xlabel(value_col)
    plt.title(title)
    savefig(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run name under output/ml_runs/")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    run_root = base / "output" / "ml_runs" / args.run
    model_dir = run_root / "models"
    out_dir = base / "plots" / "ml_runs" / args.run / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    man = load_manifest(run_root)
    feature_cols: List[str] = man["feature_cols"]
    n_features = int(man["n_features"])
    D = int(man["D"])

    # Flux model importance
    flux_path = model_dir / "booster_flux.json"
    if not flux_path.exists():
        raise FileNotFoundError(f"Missing {flux_path}")
    b_flux = load_booster(flux_path)
    flux_gain = score_to_vector(b_flux.get_score(importance_type="gain"), n_features)
    flux_weight = score_to_vector(b_flux.get_score(importance_type="weight"), n_features)
    df_flux = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": flux_gain,
            "weight": flux_weight,
        }
    ).sort_values("gain", ascending=False)
    df_flux.to_csv(out_dir / "feature_importance_flux_gain.csv", index=False)
    top_bar(
        df_flux,
        value_col="gain",
        out_png=out_dir / "feature_importance_flux_top20.png",
        title="Flux model feature importance (gain)",
        top_n=20,
    )

    # Pixel models aggregate importance
    pix_paths = sorted(model_dir.glob("booster_pix_*.json"))
    if len(pix_paths) != D:
        print(f"[WARN] expected {D} pixel models, found {len(pix_paths)}")
    all_gain = []
    all_weight = []
    for p in pix_paths:
        b = load_booster(p)
        all_gain.append(score_to_vector(b.get_score(importance_type="gain"), n_features))
        all_weight.append(score_to_vector(b.get_score(importance_type="weight"), n_features))
    G = np.vstack(all_gain) if all_gain else np.zeros((0, n_features))
    W = np.vstack(all_weight) if all_weight else np.zeros((0, n_features))

    gain_mean = G.mean(axis=0) if G.size else np.zeros(n_features)
    gain_std = G.std(axis=0) if G.size else np.zeros(n_features)
    nonzero_models = np.sum(G > 0, axis=0) if G.size else np.zeros(n_features)
    weight_mean = W.mean(axis=0) if W.size else np.zeros(n_features)

    df_pix = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain_mean": gain_mean,
            "gain_std": gain_std,
            "weight_mean": weight_mean,
            "nonzero_model_count": nonzero_models.astype(int),
            "nonzero_model_frac": (nonzero_models / max(len(pix_paths), 1)).astype(float),
        }
    ).sort_values("gain_mean", ascending=False)
    df_pix.to_csv(out_dir / "feature_importance_pixels_gain.csv", index=False)

    top_bar(
        df_pix,
        value_col="gain_mean",
        out_png=out_dir / "feature_importance_pixels_top20_mean_gain.png",
        title="Pixel models feature importance (mean gain across pixels)",
        top_n=20,
    )

    d = df_pix.sort_values("nonzero_model_count", ascending=False).head(20).iloc[::-1]
    plt.figure(figsize=(8.5, 6.0))
    plt.barh(d["feature"], d["nonzero_model_count"])
    plt.xlabel("number of pixel models where feature is used")
    plt.title("Feature usage frequency across pixel models")
    savefig(out_dir / "feature_importance_pixels_nonzero_counts.png")

    print("DONE.")
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
