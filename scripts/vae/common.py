#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Manifest:
    path: Path
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    D: int
    stamp_pix: int
    X_train_path: Path
    Yshape_train_path: Path
    Yflux_train_path: Path
    X_val_path: Path
    Yshape_val_path: Path
    Yflux_val_path: Path
    X_test_path: Path
    Yshape_test_path: Path
    Yflux_test_path: Path
    feature_cols: List[str]
    run_root: Path


def _to_int(v) -> int:
    return int(np.asarray(v).reshape(()))


def _to_path(v) -> Path:
    return Path(str(np.asarray(v).reshape(())))


def load_manifest(path: Path) -> Manifest:
    d = np.load(path, allow_pickle=True)
    feature_cols: List[str]
    if "feature_cols" in d.files:
        feature_cols = [str(x) for x in d["feature_cols"].tolist()]
    else:
        n_feat = _to_int(d["n_features"])
        feature_cols = [f"feat_{i}" for i in range(n_feat)]
    return Manifest(
        path=path,
        n_train=_to_int(d["n_train"]),
        n_val=_to_int(d["n_val"]),
        n_test=_to_int(d["n_test"]),
        n_features=_to_int(d["n_features"]),
        D=_to_int(d["D"]),
        stamp_pix=_to_int(d["stamp_pix"]),
        X_train_path=_to_path(d["X_train_path"]),
        Yshape_train_path=_to_path(d["Yshape_train_path"]),
        Yflux_train_path=_to_path(d["Yflux_train_path"]),
        X_val_path=_to_path(d["X_val_path"]),
        Yshape_val_path=_to_path(d["Yshape_val_path"]),
        Yflux_val_path=_to_path(d["Yflux_val_path"]),
        X_test_path=_to_path(d["X_test_path"]),
        Yshape_test_path=_to_path(d["Yshape_test_path"]),
        Yflux_test_path=_to_path(d["Yflux_test_path"]),
        feature_cols=feature_cols,
        run_root=path.parent,
    )


def open_split_memmaps(manifest: Manifest, split: str) -> Tuple[np.memmap, np.memmap, np.memmap]:
    if split == "train":
        n = manifest.n_train
        return (
            np.memmap(manifest.X_train_path, dtype="float32", mode="r", shape=(n, manifest.n_features)),
            np.memmap(manifest.Yshape_train_path, dtype="float32", mode="r", shape=(n, manifest.D)),
            np.memmap(manifest.Yflux_train_path, dtype="float32", mode="r", shape=(n,)),
        )
    if split == "val":
        n = manifest.n_val
        return (
            np.memmap(manifest.X_val_path, dtype="float32", mode="r", shape=(n, manifest.n_features)),
            np.memmap(manifest.Yshape_val_path, dtype="float32", mode="r", shape=(n, manifest.D)),
            np.memmap(manifest.Yflux_val_path, dtype="float32", mode="r", shape=(n,)),
        )
    if split == "test":
        n = manifest.n_test
        return (
            np.memmap(manifest.X_test_path, dtype="float32", mode="r", shape=(n, manifest.n_features)),
            np.memmap(manifest.Yshape_test_path, dtype="float32", mode="r", shape=(n, manifest.D)),
            np.memmap(manifest.Yflux_test_path, dtype="float32", mode="r", shape=(n,)),
        )
    raise ValueError(f"Unknown split: {split}")


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def raw_from_shape_flux(y_shape: np.ndarray, y_flux: np.ndarray, flux_clip: Tuple[float, float] = (-6.0, 8.0)) -> np.ndarray:
    f = np.clip(np.asarray(y_flux, dtype=np.float32), flux_clip[0], flux_clip[1])
    scale = (10.0 ** f).astype(np.float32)
    return np.asarray(y_shape, dtype=np.float32) * scale[:, None]


def image_moments(shape_flat: np.ndarray, stamp_pix: int, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    n = shape_flat.shape[0]
    img = shape_flat.reshape(n, stamp_pix, stamp_pix).astype(np.float64)

    y, x = np.mgrid[0:stamp_pix, 0:stamp_pix]
    x = x[None, :, :]
    y = y[None, :, :]

    w = np.clip(img, 0.0, np.inf)
    s = np.sum(w, axis=(1, 2)) + eps
    x0 = np.sum(w * x, axis=(1, 2)) / s
    y0 = np.sum(w * y, axis=(1, 2)) / s

    dx = x - x0[:, None, None]
    dy = y - y0[:, None, None]
    xx = np.sum(w * dx * dx, axis=(1, 2)) / s
    yy = np.sum(w * dy * dy, axis=(1, 2)) / s
    xy = np.sum(w * dx * dy, axis=(1, 2)) / s

    t = xx + yy + eps
    e1 = (xx - yy) / t
    e2 = (2.0 * xy) / t
    ell = np.sqrt(np.clip(e1 * e1 + e2 * e2, 0.0, 1e9))
    sigma = np.sqrt(np.clip(0.5 * t, 0.0, 1e9))
    return {
        "x0": x0.astype(np.float32),
        "y0": y0.astype(np.float32),
        "e1": e1.astype(np.float32),
        "e2": e2.astype(np.float32),
        "ell": ell.astype(np.float32),
        "sigma": sigma.astype(np.float32),
    }
