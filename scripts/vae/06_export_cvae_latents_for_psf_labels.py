#!/usr/bin/env python3
"""
Export CVAE latent means (mu) for PSF label rows.

This script aligns weak-label source_ids with 16D metadata rows, rebuilds
(X features, normalized stamp shape), runs CVAE encoder, and writes latent CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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


def _to_int(v) -> int:
    return int(np.asarray(v).reshape(()))


def _to_path(v) -> Path:
    return Path(str(np.asarray(v).reshape(())))


def load_manifest(path: Path) -> Dict[str, object]:
    d = np.load(path, allow_pickle=True)
    return {
        "dataset_root": _to_path(d["dataset_root"]),
        "feature_cols": [str(x) for x in d["feature_cols"].tolist()],
        "n_features": _to_int(d["n_features"]),
        "D": _to_int(d["D"]),
        "scaler_npz": _to_path(d["scaler_npz"]) if "scaler_npz" in d.files else None,
    }


def load_scaler(npz_path: Path, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in d["feature_names"].tolist()]
    y_min = d["y_min"].astype(np.float32)
    y_iqr = d["y_iqr"].astype(np.float32)
    idx = {n: i for i, n in enumerate(names)}
    miss = [c for c in feature_cols if c not in idx]
    if miss:
        raise RuntimeError(f"Scaler missing columns: {miss}")
    ii = np.array([idx[c] for c in feature_cols], dtype=int)
    return y_min[ii], y_iqr[ii]


def normalize_shape(stamp2d: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, bool]:
    s = np.asarray(stamp2d, dtype=np.float32)
    pos = np.clip(s, 0.0, np.inf)
    integ = float(np.sum(pos))
    if (not np.isfinite(integ)) or integ <= eps:
        return np.zeros_like(pos, dtype=np.float32), False
    return (s / integ).astype(np.float32), True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cvae_ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=2048)
    args = ap.parse_args()

    labels = pd.read_csv(args.labels_csv)
    need_l = ["source_id", "split_code", "label"]
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
    labels = labels.sort_values(["source_id", "confidence"] if "confidence" in labels.columns else ["source_id"]).drop_duplicates("source_id", keep="first")

    man = load_manifest(Path(args.manifest))
    dataset_root: Path = man["dataset_root"]
    feature_cols: List[str] = list(man["feature_cols"])
    D = int(man["D"])
    stamp_pix = int(round(np.sqrt(D)))
    if stamp_pix * stamp_pix != D or D != 400:
        raise RuntimeError(f"Unexpected stamp D={D}; this exporter assumes 20x20 flat stamps")

    if man["scaler_npz"] is None:
        raise RuntimeError("Manifest does not contain scaler_npz; cannot normalize features for CVAE")
    smin, siqr = load_scaler(Path(man["scaler_npz"]), feature_cols=feature_cols)

    fields = sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / "metadata_16d.csv").exists()])
    if not fields:
        raise RuntimeError(f"No fields with metadata_16d.csv under {dataset_root}")

    source_set = set(labels["source_id"].tolist())
    meta_rows = []
    for f in fields:
        p = f / "metadata_16d.csv"
        usecols = ["source_id", "split_code", "npz_file", "index_in_file"] + feature_cols
        m = pd.read_csv(p, usecols=usecols, low_memory=False)
        m["source_id"] = pd.to_numeric(m["source_id"], errors="coerce")
        m["split_code"] = pd.to_numeric(m["split_code"], errors="coerce")
        m["index_in_file"] = pd.to_numeric(m["index_in_file"], errors="coerce")
        m = m[m["source_id"].isin(source_set)].copy()
        m = m.dropna(subset=["source_id", "split_code", "npz_file", "index_in_file"] + feature_cols)
        if m.empty:
            continue
        m["source_id"] = m["source_id"].astype(np.int64)
        m["split_code"] = m["split_code"].astype(int)
        m["index_in_file"] = m["index_in_file"].astype(int)
        m["field_tag"] = f.name
        m = m.sort_values(["source_id"]).drop_duplicates("source_id", keep="first")
        meta_rows.append(m)

    if not meta_rows:
        raise RuntimeError("No metadata rows matched labels source_ids")

    meta = pd.concat(meta_rows, ignore_index=True)
    df = labels.merge(meta, on=["source_id", "split_code"], how="inner", suffixes=("_label", ""))
    if df.empty:
        raise RuntimeError("No rows after merge on (source_id, split_code)")

    ckpt = torch.load(args.cvae_ckpt, map_location="cpu")
    n_feat = int(ckpt.get("n_features", len(feature_cols)))
    z_dim = int(ckpt.get("z_dim", 16))
    width = int(ckpt.get("model_width", 2))
    model = CVAE(n_feat=n_feat, z_dim=z_dim, width=width)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Build encoder inputs row-by-row (cache NPZ per field+file).
    X_rows: List[np.ndarray] = []
    Y_rows: List[np.ndarray] = []
    keep_idx: List[int] = []
    cache: Dict[Tuple[str, str], np.ndarray] = {}
    dropped_bad_stamp = 0
    dropped_missing_npz = 0
    for i, r in enumerate(df.itertuples(index=False)):
        ftag = str(r.field_tag)
        npz_file = str(r.npz_file)
        idx = int(r.index_in_file)
        key = (ftag, npz_file)
        if key not in cache:
            npz_path = dataset_root / ftag / npz_file
            if not npz_path.exists():
                dropped_missing_npz += 1
                continue
            with np.load(npz_path) as npz:
                if "X" not in npz.files:
                    dropped_missing_npz += 1
                    continue
                cache[key] = np.array(npz["X"], dtype=np.float32)
        arr = cache[key]
        if idx < 0 or idx >= arr.shape[0]:
            dropped_missing_npz += 1
            continue
        stamp = arr[idx]
        if stamp.shape != (stamp_pix, stamp_pix):
            dropped_bad_stamp += 1
            continue
        yshape, ok = normalize_shape(stamp)
        if not ok:
            dropped_bad_stamp += 1
            continue
        xraw = np.array([float(getattr(r, c)) for c in feature_cols], dtype=np.float32)
        x = (xraw - smin) / siqr
        if not np.all(np.isfinite(x)):
            dropped_bad_stamp += 1
            continue
        X_rows.append(x)
        Y_rows.append(yshape.reshape(-1).astype(np.float32))
        keep_idx.append(i)

    if not X_rows:
        raise RuntimeError("No valid rows to encode")

    X = np.vstack(X_rows).astype(np.float32)
    Y = np.vstack(Y_rows).astype(np.float32)
    kept = df.iloc[keep_idx].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    z_out = np.empty((X.shape[0], z_dim), dtype=np.float32)
    bs = int(max(1, args.batch_size))
    with torch.no_grad():
        for s in range(0, X.shape[0], bs):
            e = min(X.shape[0], s + bs)
            xb = torch.from_numpy(X[s:e]).to(device)
            yb = torch.from_numpy(Y[s:e]).to(device)
            mu, _ = model.enc(yb, xb)
            z_out[s:e] = mu.cpu().numpy().astype(np.float32)

    out = kept[["source_id", "split_code", "label"] + [c for c in ["confidence", "score_psf_like"] if c in kept.columns]].copy()
    for j in range(z_dim):
        out[f"z_{j:02d}"] = z_out[:, j]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    summary = {
        "labels_csv": str(args.labels_csv),
        "manifest": str(args.manifest),
        "cvae_ckpt": str(args.cvae_ckpt),
        "out_csv": str(out_csv),
        "n_labels_input": int(len(labels)),
        "n_after_metadata_merge": int(len(df)),
        "n_encoded": int(len(out)),
        "z_dim": int(z_dim),
        "dropped_missing_npz_or_idx": int(dropped_missing_npz),
        "dropped_bad_stamp_or_feature": int(dropped_bad_stamp),
    }
    (out_csv.parent / (out_csv.stem + "_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print("Output:", out_csv)
    print("Rows:", len(out), "z_dim:", z_dim)


if __name__ == "__main__":
    main()
