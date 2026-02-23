#!/usr/bin/env python3
"""
Apply frozen PSF gate and CVAE reconstruction QC.

For rows predicted as PSF-like by the gate, compute CVAE mean reconstruction error
and flag high-error anomalies by quantile threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This script requires PyTorch. Activate the environment with torch installed "
        f"and rerun. Original import error: {e}"
    )


MORPH_COLS = [
    "m_concentration_r2_r6",
    "m_asymmetry_180",
    "m_ellipticity",
    "m_peak_sep_pix",
    "m_edge_flux_frac",
    "m_peak_ratio_2over1",
]


class Encoder(nn.Module):
    def __init__(self, n_feat: int, z_dim: int, width: int = 2):
        super().__init__()
        c1 = 16 * width
        c2 = 32 * width
        xh = 64 * width
        fh = 256 * width
        self.enc_img = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.enc_x = nn.Sequential(
            nn.Linear(n_feat, xh),
            nn.ReLU(inplace=True),
            nn.Linear(xh, xh),
            nn.ReLU(inplace=True),
            nn.Linear(xh, xh),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(c2 * 5 * 5 + xh, fh),
            nn.ReLU(inplace=True),
            nn.Linear(fh, fh // 2),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(fh // 2, z_dim)
        self.logvar = nn.Linear(fh // 2, z_dim)

    def forward(self, y_flat: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y_flat.view(y_flat.shape[0], 1, 20, 20)
        hy = self.enc_img(y)
        hx = self.enc_x(x)
        h = self.fuse(torch.cat([hy, hx], dim=1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, n_feat: int, z_dim: int, width: int = 2):
        super().__init__()
        c0 = 64 * width
        c1 = 32 * width
        c2 = 16 * width
        fh = 256 * width
        xh = 64 * width
        self.cond_x = nn.Sequential(
            nn.Linear(n_feat, xh),
            nn.ReLU(inplace=True),
            nn.Linear(xh, xh),
            nn.ReLU(inplace=True),
        )
        self.inp = nn.Sequential(
            nn.Linear(xh + z_dim, fh),
            nn.ReLU(inplace=True),
            nn.Linear(fh, c0 * 5 * 5),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c0, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, 1, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xh = self.cond_x(x)
        h = self.inp(torch.cat([z, xh], dim=1))
        c0 = h.shape[1] // 25
        h = h.view(h.shape[0], c0, 5, 5)
        y = self.dec(h).view(h.shape[0], -1)
        y = nn.functional.softplus(y)
        y = y / (torch.sum(y, dim=1, keepdim=True) + 1e-12)
        return y


class CVAE(nn.Module):
    def __init__(self, n_feat: int, z_dim: int, width: int = 2):
        super().__init__()
        self.enc = Encoder(n_feat, z_dim, width=width)
        self.dec = Decoder(n_feat, z_dim, width=width)
        self.z_dim = z_dim

    def mean_pred(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros((x.shape[0], self.z_dim), device=x.device, dtype=x.dtype)
        return self.dec(z, x)


def _to_int(v) -> int:
    return int(np.asarray(v).reshape(()))


def _to_path(v) -> Path:
    return Path(str(np.asarray(v).reshape(())))


def load_manifest(path: Path) -> Dict[str, object]:
    d = np.load(path, allow_pickle=True)
    return {
        "dataset_root": _to_path(d["dataset_root"]),
        "feature_cols": [str(x) for x in d["feature_cols"].tolist()],
        "D": _to_int(d["D"]),
        "scaler_npz": _to_path(d["scaler_npz"]) if "scaler_npz" in d.files else None,
    }


def load_scaler(npz_path: Path, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    y_min = d["y_min"].astype(np.float32)
    y_iqr = d["y_iqr"].astype(np.float32)
    idx = {n: i for i, n in enumerate(names)}
    ii = np.array([idx[c] for c in feature_cols], dtype=int)
    return y_min[ii], y_iqr[ii]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_shape(stamp2d: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, bool]:
    s = np.asarray(stamp2d, dtype=np.float32)
    pos = np.clip(s, 0.0, np.inf)
    integ = float(np.sum(pos))
    if (not np.isfinite(integ)) or integ <= eps:
        return np.zeros_like(s, dtype=np.float32), False
    return (s / integ).astype(np.float32), True


def rmse_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--gate_package", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cvae_ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--rmse_quantile", type=float, default=0.99)
    ap.add_argument("--batch_size", type=int, default=2048)
    args = ap.parse_args()

    gate = json.loads(Path(args.gate_package).read_text(encoding="utf-8"))
    feat_order = list(gate["feature_order"])
    z_clip = float(gate["z_clip"])
    thr = float(gate["threshold_p_nonpsf"])
    coef = np.asarray(gate["coef"], dtype=float)
    intercept = float(gate["intercept"])

    labels = pd.read_csv(args.labels_csv)
    need_l = ["source_id", "split_code", "label"] + feat_order
    miss_l = [c for c in need_l if c not in labels.columns]
    if miss_l:
        raise RuntimeError(f"labels csv missing columns: {miss_l}")
    for c in need_l:
        labels[c] = pd.to_numeric(labels[c], errors="coerce")
    extra_l = [c for c in ["confidence", "score_psf_like"] if c in labels.columns]
    for c in extra_l:
        labels[c] = pd.to_numeric(labels[c], errors="coerce")
    labels = labels.dropna(subset=need_l).copy()
    labels["source_id"] = labels["source_id"].astype(np.int64)
    labels["split_code"] = labels["split_code"].astype(int)
    labels["label"] = labels["label"].astype(int)
    labels = labels.sort_values(["source_id"]).drop_duplicates("source_id", keep="first")

    # Apply frozen gate.
    Zm = []
    for c in feat_order:
        s = gate["scaler"][c]
        med = float(s["median"])
        sc = float(s["scale"])
        z = (labels[c].to_numpy(dtype=float) - med) / sc
        z = np.clip(z, -z_clip, z_clip)
        Zm.append(z)
    Zm = np.column_stack(Zm)
    p_non = sigmoid(Zm @ coef + intercept)
    pred_non = (p_non >= thr).astype(int)

    labels = labels.copy()
    labels["p_nonpsf_gate"] = p_non
    labels["pred_label_gate"] = pred_non
    psf_sel = labels[labels["pred_label_gate"] == 0].copy()
    if psf_sel.empty:
        raise RuntimeError("No PSF-selected rows from gate; nothing to QC")

    man = load_manifest(Path(args.manifest))
    dataset_root: Path = man["dataset_root"]
    feature_cols: List[str] = list(man["feature_cols"])
    D = int(man["D"])
    stamp_pix = int(round(np.sqrt(D)))
    if stamp_pix * stamp_pix != D or D != 400:
        raise RuntimeError(f"Unexpected D={D}; this script assumes 20x20")
    if man["scaler_npz"] is None:
        raise RuntimeError("Manifest missing scaler_npz")
    smin, siqr = load_scaler(Path(man["scaler_npz"]), feature_cols)

    fields = sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / "metadata_16d.csv").exists()])
    source_set = set(psf_sel["source_id"].tolist())

    meta_rows = []
    for f in fields:
        p = f / "metadata_16d.csv"
        usecols = ["source_id", "split_code", "npz_file", "index_in_file"] + feature_cols
        m = pd.read_csv(p, usecols=usecols, low_memory=False)
        m["source_id"] = pd.to_numeric(m["source_id"], errors="coerce")
        m["split_code"] = pd.to_numeric(m["split_code"], errors="coerce")
        m["index_in_file"] = pd.to_numeric(m["index_in_file"], errors="coerce")
        m = m[m["source_id"].isin(source_set)].dropna(subset=["source_id", "split_code", "npz_file", "index_in_file"] + feature_cols).copy()
        if m.empty:
            continue
        m["source_id"] = m["source_id"].astype(np.int64)
        m["split_code"] = m["split_code"].astype(int)
        m["index_in_file"] = m["index_in_file"].astype(int)
        m["field_tag"] = f.name
        m = m.sort_values(["source_id"]).drop_duplicates("source_id", keep="first")
        meta_rows.append(m)
    if not meta_rows:
        raise RuntimeError("No metadata rows for gate-selected PSF rows")

    meta = pd.concat(meta_rows, ignore_index=True)
    df = psf_sel.merge(meta, on=["source_id", "split_code"], how="inner", suffixes=("", "_meta"))
    if df.empty:
        raise RuntimeError("No rows after merge of gate-selected PSF with metadata")

    ckpt = torch.load(args.cvae_ckpt, map_location="cpu")
    n_feat = int(ckpt.get("n_features", len(feature_cols)))
    z_dim = int(ckpt.get("z_dim", 16))
    width = int(ckpt.get("model_width", 2))
    model = CVAE(n_feat=n_feat, z_dim=z_dim, width=width)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    X_rows: List[np.ndarray] = []
    Y_rows: List[np.ndarray] = []
    keep_idx: List[int] = []
    cache: Dict[Tuple[str, str], np.ndarray] = {}
    dropped = 0

    for i, r in enumerate(df.itertuples(index=False)):
        ftag = str(r.field_tag)
        npz_file = str(r.npz_file)
        idx = int(r.index_in_file)
        key = (ftag, npz_file)
        if key not in cache:
            pnpz = dataset_root / ftag / npz_file
            if not pnpz.exists():
                dropped += 1
                continue
            with np.load(pnpz) as npz:
                if "X" not in npz.files:
                    dropped += 1
                    continue
                cache[key] = np.array(npz["X"], dtype=np.float32)
        arr = cache[key]
        if idx < 0 or idx >= arr.shape[0]:
            dropped += 1
            continue
        stamp = arr[idx]
        if stamp.shape != (stamp_pix, stamp_pix):
            dropped += 1
            continue
        yshape, ok = normalize_shape(stamp)
        if not ok:
            dropped += 1
            continue
        xraw = np.array([float(getattr(r, c)) for c in feature_cols], dtype=np.float32)
        x = (xraw - smin) / siqr
        if not np.all(np.isfinite(x)):
            dropped += 1
            continue
        X_rows.append(x)
        Y_rows.append(yshape.reshape(-1))
        keep_idx.append(i)

    if not X_rows:
        raise RuntimeError("No valid rows for CVAE QC")

    X = np.vstack(X_rows).astype(np.float32)
    Y = np.vstack(Y_rows).astype(np.float32)
    kept = df.iloc[keep_idx].reset_index(drop=True)

    pred = np.empty_like(Y)
    bs = int(max(1, args.batch_size))
    with torch.no_grad():
        for s in range(0, X.shape[0], bs):
            e = min(X.shape[0], s + bs)
            xb = torch.from_numpy(X[s:e]).to(device)
            yp = model.mean_pred(xb).cpu().numpy().astype(np.float32)
            pred[s:e] = yp

    rmse = rmse_rows(pred, Y)
    q = float(args.rmse_quantile)
    thr_rmse = float(np.quantile(rmse, q))

    out = kept[["source_id", "split_code", "label", "p_nonpsf_gate", "pred_label_gate"] + [c for c in ["confidence", "score_psf_like"] if c in kept.columns]].copy()
    out["cvae_rmse_shape"] = rmse
    out["qc_is_anomaly"] = (rmse >= thr_rmse).astype(int)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    summary = {
        "labels_csv": str(args.labels_csv),
        "gate_package": str(args.gate_package),
        "manifest": str(args.manifest),
        "cvae_ckpt": str(args.cvae_ckpt),
        "out_csv": str(out_csv),
        "gate_threshold_p_nonpsf": float(thr),
        "n_input_labels": int(len(labels)),
        "n_psf_selected_by_gate": int(len(psf_sel)),
        "n_psf_with_qc": int(len(out)),
        "rmse_quantile": q,
        "rmse_threshold": thr_rmse,
        "n_qc_anomalies": int(out["qc_is_anomaly"].sum()),
        "dropped_rows_missing_or_invalid": int(dropped),
    }
    (out_csv.parent / (out_csv.stem + "_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print("Output:", out_csv)


if __name__ == "__main__":
    main()
