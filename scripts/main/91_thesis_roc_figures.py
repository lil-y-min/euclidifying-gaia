#!/usr/bin/env python3
"""
91_thesis_roc_figures.py
=========================

Generate three thesis-quality ROC+PRC figures using saved models and scores,
without retraining.  Each figure has ROC on the left and PRC on the right;
related classifiers are overlaid on the same axes.

Outputs (saved to thesis Chapter4/Figs/Raster/):
  15_roc_galaxy.png        — galaxy disk classifier: 16D vs 15D ROC+PRC
  20_roc_quasar_combined.png — WISE quasar locus + Quaia ROC+PRC on same axes
  22_roc_q1_comparison.png  — ERO→Q1 transfer vs Q1 native ROC+PRC
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts" / "main"))
from feature_schema import get_feature_cols, scaler_stem

DATASET_ROOT  = BASE / "output" / "dataset_npz"
OUT_DIR       = BASE / "report" / "phd-thesis-template-2.4" / "Chapter4" / "Figs" / "Raster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# typography — moderate sizes for single/dual-panel wrapfigures
FS_LABEL  = 14   # axis labels
FS_TICK   = 12   # tick labels
FS_LEGEND = 12   # legend entries
FS_TITLE  = 13   # subplot titles (not used for ROC)

WISE_W12_LO, WISE_W12_HI = 0.6, 1.6
WISE_W23_LO, WISE_W23_HI = 2.0, 4.2


# ── helpers ───────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def load_scaler(feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    stem = scaler_stem("16D")
    d    = np.load(BASE / "output" / "scalers" / f"{stem}.npz", allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    idx  = np.array([names.index(c) for c in feature_cols], dtype=int)
    return d["y_min"].astype(np.float32)[idx], d["y_iqr"].astype(np.float32)[idx]


def apply_model(model_path: Path, X: np.ndarray,
                feature_cols: list[str]) -> np.ndarray:
    b = xgb.Booster()
    b.load_model(str(model_path))
    dm = xgb.DMatrix(X, feature_names=feature_cols)
    return b.predict(dm)


def roc_panel(ax: plt.Axes, curves: list[dict]) -> None:
    """
    Plot ROC curves.
    Each dict: {fpr, tpr, auc, label, color, ls}
    """
    for c in curves:
        ax.plot(c["fpr"], c["tpr"], color=c["color"], lw=2.2, ls=c.get("ls", "-"),
                label=f"{c['label']}  (AUC={c['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    ax.set_xlabel("False positive rate", fontsize=FS_LABEL)
    ax.set_ylabel("True positive rate", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc="lower right", framealpha=0.85)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def prc_panel(ax: plt.Axes, curves: list[dict]) -> None:
    """
    Plot Precision-Recall curves.
    Each dict: {precision, recall, ap, label, color, ls}
    """
    for c in curves:
        ax.plot(c["recall"], c["precision"], color=c["color"], lw=2.2,
                ls=c.get("ls", "-"),
                label=f"{c['label']}  (AP={c['ap']:.3f})")
    ax.set_xlabel("Recall", fontsize=FS_LABEL)
    ax.set_ylabel("Precision", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc="upper right", framealpha=0.85)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ── Figure 1: Galaxy disk ROC (15_roc_galaxy.png) ────────────────────────────

BIG_GALAXY_TABLE = [
    ("IC342",       56.6988,    68.0961,   10.7,  "ERO-IC342"),
    ("NGC6744",    287.4421,   -63.8575,    7.75, "ERO-NGC6744"),
    ("NGC2403",    114.2142,    65.6031,   10.95, "ERO-NGC2403"),
    ("NGC6822",    296.2404,   -14.8031,    7.75, "ERO-NGC6822"),
    ("HolmbergII", 124.7708,    70.7219,    3.95, "ERO-HolmbergII"),
    ("IC10",         5.0721,    59.3039,    3.15, "ERO-IC10"),
]
GALAXY_OUTSIDE_FIELDS = {
    "ERO-Abell2390","ERO-Abell2764","ERO-Barnard30",
    "ERO-Horsehead","ERO-Messier78","ERO-Taurus",
    "ERO-NGC6397","ERO-NGC6254","ERO-Perseus",
}

META_NAME = "metadata_16d.csv"


def load_galaxy_test_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load metadata and compute geometric galaxy labels for the test split."""
    print("  Loading galaxy metadata …")
    galaxy_tags = {g[4] for g in BIG_GALAXY_TABLE}
    all_fields  = galaxy_tags | GALAXY_OUTSIDE_FIELDS
    frames = []
    for fdir in sorted(p for p in DATASET_ROOT.iterdir()
                       if p.is_dir() and (p / META_NAME).exists()):
        if fdir.name not in all_fields:
            continue
        df = pd.read_csv(fdir / META_NAME, low_memory=False)
        if "field_tag" not in df.columns:
            df["field_tag"] = fdir.name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["label"] = 0

    src = SkyCoord(ra=df["ra"].to_numpy() * u.deg,
                   dec=df["dec"].to_numpy() * u.deg, frame="icrs")
    for name, ra, dec, r_d25, tag in BIG_GALAXY_TABLE:
        gc   = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        mask = (df["field_tag"] == tag).to_numpy()
        if not mask.any():
            continue
        seps = src[mask].separation(gc).to(u.arcmin).value
        df.loc[df.index[mask], "label"] = (seps < r_d25).astype(int)

    test = df[df["split_code"] == 2].copy()
    print(f"  Test set: {len(test):,} rows  (pos={int(test.label.sum())})")
    return test


def make_galaxy_roc() -> None:
    print("Galaxy ROC …")
    test = load_galaxy_test_data()

    model_dir = BASE / "output" / "ml_runs" / "galaxy_env_clf"
    results   = {}

    for fs_name in ("16D", "15D"):
        feat_cols = get_feature_cols(fs_name)
        fm, fi    = load_scaler(feat_cols)
        sub       = test.dropna(subset=feat_cols).copy()
        X         = (sub[feat_cols].to_numpy(dtype=np.float32) - fm) / np.where(fi > 0, fi, 1.0)
        y         = sub["label"].to_numpy(dtype=np.float32)

        suffix   = "_15d" if fs_name == "15D" else ""
        mpath    = model_dir / f"model_inside_galaxy{suffix}.json"
        prob     = apply_model(mpath, X, feat_cols)
        auc      = roc_auc_score(y, prob)
        ap       = average_precision_score(y, prob)
        fpr, tpr, _ = roc_curve(y, prob)
        prec, rec, _ = precision_recall_curve(y, prob)
        print(f"  {fs_name}: AUC={auc:.4f}  AP={ap:.4f}")
        results[fs_name] = {"fpr": fpr, "tpr": tpr, "auc": auc,
                             "precision": prec, "recall": rec, "ap": ap}

    curves = [
        {"fpr": results["16D"]["fpr"], "tpr": results["16D"]["tpr"],
         "auc": results["16D"]["auc"], "precision": results["16D"]["precision"],
         "recall": results["16D"]["recall"], "ap": results["16D"]["ap"],
         "label": "16D features", "color": "steelblue", "ls": "-"},
        {"fpr": results["15D"]["fpr"], "tpr": results["15D"]["tpr"],
         "auc": results["15D"]["auc"], "precision": results["15D"]["precision"],
         "recall": results["15D"]["recall"], "ap": results["15D"]["ap"],
         "label": "15D (no scanning-law feature)", "color": "tomato", "ls": "--"},
    ]

    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    roc_panel(ax_roc, curves)
    prc_panel(ax_prc, curves)
    ax_roc.set_title("ROC", fontsize=FS_TITLE, fontweight="bold")
    ax_prc.set_title("Precision-Recall", fontsize=FS_TITLE, fontweight="bold")
    fig.suptitle("Galaxy disk membership classifier", fontsize=FS_TITLE,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, OUT_DIR / "15_roc_galaxy.png")


# ── Figure 2: Combined quasar ROC (20_roc_quasar_combined.png) ───────────────

def make_quasar_combined_roc() -> None:
    print("Quasar combined ROC …")
    wise_sc  = pd.read_csv(BASE / "output" / "ml_runs" / "quasar_wise_clf" / "scores_all.csv")
    quaia_sc = pd.read_csv(BASE / "output" / "ml_runs" / "quaia_clf"       / "scores_all.csv")

    curves = []
    for label, sc, color, ls in [
        ("WISE quasar locus",   wise_sc,  "steelblue", "-"),
        ("Quaia (spectroscopic)", quaia_sc, "tomato",   "--"),
    ]:
        y    = sc["label"].to_numpy(dtype=np.float32)
        prob = sc["xgb_score"].to_numpy(dtype=np.float32)
        auc  = roc_auc_score(y, prob)
        ap   = average_precision_score(y, prob)
        fpr, tpr, _ = roc_curve(y, prob)
        prec, rec, _ = precision_recall_curve(y, prob)
        print(f"  {label}: AUC={auc:.4f}  AP={ap:.4f}")
        curves.append({"fpr": fpr, "tpr": tpr, "auc": auc,
                        "precision": prec, "recall": rec, "ap": ap,
                        "label": label, "color": color, "ls": ls})

    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    roc_panel(ax_roc, curves)
    prc_panel(ax_prc, curves)
    ax_roc.set_title("ROC", fontsize=FS_TITLE, fontweight="bold")
    ax_prc.set_title("Precision-Recall", fontsize=FS_TITLE, fontweight="bold")
    fig.suptitle("Quasar classifiers", fontsize=FS_TITLE, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, OUT_DIR / "20_roc_quasar_combined.png")


# ── Figure 3: Q1 transfer ROC (22_roc_q1_comparison.png) ─────────────────────

def make_q1_roc() -> None:
    print("Q1 transfer ROC …")
    q1_dir = BASE / "output" / "ml_runs" / "quasar_q1"

    feat_df  = pd.read_csv(q1_dir / "q1_gaia_features.csv")
    wise_df  = pd.read_csv(q1_dir / "q1_wise_phot.csv")

    # Merge WISE photometry
    wise_df["w1_w2"] = wise_df["w1mpro"] - wise_df["w2mpro"]
    wise_df["w2_w3"] = wise_df["w2mpro"] - wise_df["w3mpro"]
    in_locus = (
        (wise_df["w1_w2"] >= WISE_W12_LO) & (wise_df["w1_w2"] <= WISE_W12_HI) &
        (wise_df["w2_w3"] >= WISE_W23_LO) & (wise_df["w2_w3"] <= WISE_W23_HI) &
        (wise_df["w1snr"] >= 5) & (wise_df["w2snr"] >= 5) & (wise_df["w3snr"] >= 2)
    )
    wise_df["label"] = in_locus.astype(int)

    df = feat_df.merge(wise_df[["source_id", "label"]], on="source_id", how="inner")
    df = df.dropna(subset=get_feature_cols("15D") + ["label"])

    # Reproduce the train/val/test split from script 84 (random_state=42, 70/15/15)
    idx      = np.arange(len(df))
    y_all    = df["label"].to_numpy(dtype=np.float32)
    _, temp  = train_test_split(idx, test_size=0.30, random_state=42,
                                stratify=y_all)
    _, test_idx = train_test_split(temp, test_size=0.50, random_state=42,
                                   stratify=y_all[temp])
    test     = df.iloc[test_idx]
    y_te     = test["label"].to_numpy(dtype=np.float32)

    feat15   = get_feature_cols("15D")
    fm, fi   = load_scaler(feat15)
    X_te     = (test[feat15].to_numpy(dtype=np.float32) - fm) / np.where(fi > 0, fi, 1.0)

    print(f"  Q1 test set: {len(test):,} rows  (pos={int(y_te.sum())})")

    curves = []
    for label, mpath, color, ls in [
        ("ERO model (transfer)",   BASE / "output/ml_runs/quasar_wise_clf/model_quasar_wise.json",
         "steelblue", "--"),
        ("Q1 native model",        q1_dir / "model_q1_quasar.json",
         "tomato", "-"),
    ]:
        prob         = apply_model(mpath, X_te, feat15)
        auc          = roc_auc_score(y_te, prob)
        ap           = average_precision_score(y_te, prob)
        fpr, tpr, _  = roc_curve(y_te, prob)
        prec, rec, _ = precision_recall_curve(y_te, prob)
        print(f"  {label}: AUC={auc:.4f}  AP={ap:.4f}")
        curves.append({"fpr": fpr, "tpr": tpr, "auc": auc,
                        "precision": prec, "recall": rec, "ap": ap,
                        "label": label, "color": color, "ls": ls})

    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    roc_panel(ax_roc, curves)
    prc_panel(ax_prc, curves)
    ax_roc.set_title("ROC", fontsize=FS_TITLE, fontweight="bold")
    ax_prc.set_title("Precision-Recall", fontsize=FS_TITLE, fontweight="bold")
    fig.suptitle("ERO$\\to$Q1 transfer vs Q1 native", fontsize=FS_TITLE,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, OUT_DIR / "22_roc_q1_comparison.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    make_galaxy_roc()
    make_quasar_combined_roc()
    make_q1_roc()
    print("\nAll ROC figures done.")


if __name__ == "__main__":
    main()
