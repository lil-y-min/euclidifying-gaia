#!/usr/bin/env python3
"""
Train a neural network to reconstruct stamps from Gaia features.

Targets mirror the existing pixels+flux setup:
- y_shape: normalized stamp pixels (D = stamp_pix * stamp_pix)
- y_flux:  log10(positive integral)
- reconstructed raw stamp: y_shape * (10 ** y_flux)

Input data is loaded from an existing manifest + memmaps (produced by prior xgb scripts).
This script writes epoch logs both to stdout and to <run_dir>/train.log for live debugging.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Cfg:
    base_dir: Path = Path(__file__).resolve().parents[2]
    manifest_npz: Path = None
    out_root: Path = None
    run_name: str = "nn_pixelsflux_mlp"

    seed: int = 11
    batch_size: int = 512
    num_workers: int = 0
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_patience: int = 8

    hidden_dim: int = 256
    depth: int = 3
    dropout: float = 0.10

    loss_w_shape: float = 1.0
    loss_w_flux: float = 0.20
    raw_flux_clip_min: float = -6.0
    raw_flux_clip_max: float = 8.0

    max_train_samples: int = 0
    max_val_samples: int = 0
    max_test_samples: int = 0

    def __post_init__(self) -> None:
        if self.manifest_npz is None:
            self.manifest_npz = self.base_dir / "output" / "ml_runs" / "ml_xgb_pixelsflux_8d_augtrain_g17" / "manifest_arrays.npz"
        if self.out_root is None:
            self.out_root = self.base_dir / "output" / "ml_runs"


class Logger:
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class MemmapDataset(Dataset):
    def __init__(self, X: np.memmap, Yshape: np.memmap, Yflux: np.memmap, n_use: int):
        self.X = X
        self.Yshape = Yshape
        self.Yflux = Yflux
        self.n = int(n_use)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        # Copy to make arrays writable and avoid torch warning on readonly memmaps.
        x = np.asarray(self.X[idx], dtype=np.float32).copy()
        ys = np.asarray(self.Yshape[idx], dtype=np.float32).copy()
        yf = np.float32(self.Yflux[idx])
        return torch.from_numpy(x), torch.from_numpy(ys), torch.tensor(yf, dtype=torch.float32)


class ShapeFluxMLP(nn.Module):
    def __init__(self, n_feat: int, D: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        blocks: List[nn.Module] = []
        in_dim = n_feat
        for _ in range(depth):
            blocks.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.head_shape = nn.Linear(hidden_dim, D)
        self.head_flux = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        shape_logits = self.head_shape(z)
        # Shape target is normalized positive map; softmax enforces positivity + sum-to-1.
        shape = torch.softmax(shape_logits, dim=1)
        flux = self.head_flux(z).squeeze(1)
        return shape, flux


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_int(v) -> int:
    return int(np.asarray(v).reshape(()))


def _to_path(v) -> Path:
    return Path(str(np.asarray(v).reshape(())))


def load_manifest(path: Path) -> Dict[str, object]:
    d = np.load(path, allow_pickle=True)
    out = {
        "n_train": _to_int(d["n_train"]),
        "n_val": _to_int(d["n_val"]),
        "n_test": _to_int(d["n_test"]),
        "n_features": _to_int(d["n_features"]),
        "D": _to_int(d["D"]),
        "stamp_pix": _to_int(d["stamp_pix"]),
        "X_train_path": _to_path(d["X_train_path"]),
        "Yshape_train_path": _to_path(d["Yshape_train_path"]),
        "Yflux_train_path": _to_path(d["Yflux_train_path"]),
        "X_val_path": _to_path(d["X_val_path"]),
        "Yshape_val_path": _to_path(d["Yshape_val_path"]),
        "Yflux_val_path": _to_path(d["Yflux_val_path"]),
        "X_test_path": _to_path(d["X_test_path"]),
        "Yshape_test_path": _to_path(d["Yshape_test_path"]),
        "Yflux_test_path": _to_path(d["Yflux_test_path"]),
    }
    return out


def rmse_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((a - b) ** 2))


def raw_from_shape_flux(shape: torch.Tensor, flux_log10: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    f = torch.clamp(flux_log10, min=clip_min, max=clip_max)
    scale = torch.pow(torch.tensor(10.0, device=shape.device), f).unsqueeze(1)
    return shape * scale


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    w_shape: float,
    w_flux: float,
    raw_flux_clip_min: float,
    raw_flux_clip_max: float,
) -> Dict[str, float]:
    model.eval()
    losses = []
    rmses_shape = []
    rmses_flux = []
    rmses_raw = []

    with torch.no_grad():
        for xb, ys_true, yf_true in loader:
            xb = xb.to(device)
            ys_true = ys_true.to(device)
            yf_true = yf_true.to(device)

            ys_pred, yf_pred = model(xb)

            l_shape = torch.mean((ys_pred - ys_true) ** 2)
            l_flux = torch.mean((yf_pred - yf_true) ** 2)
            loss = w_shape * l_shape + w_flux * l_flux

            raw_true = raw_from_shape_flux(ys_true, yf_true, raw_flux_clip_min, raw_flux_clip_max)
            raw_pred = raw_from_shape_flux(ys_pred, yf_pred, raw_flux_clip_min, raw_flux_clip_max)

            losses.append(float(loss.item()))
            rmses_shape.append(float(rmse_torch(ys_pred, ys_true).item()))
            rmses_flux.append(float(rmse_torch(yf_pred, yf_true).item()))
            rmses_raw.append(float(rmse_torch(raw_pred, raw_true).item()))

    return {
        "loss": float(np.mean(losses) if losses else np.nan),
        "rmse_shape": float(np.mean(rmses_shape) if rmses_shape else np.nan),
        "rmse_flux": float(np.mean(rmses_flux) if rmses_flux else np.nan),
        "rmse_raw": float(np.mean(rmses_raw) if rmses_raw else np.nan),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    w_shape: float,
    w_flux: float,
) -> Dict[str, float]:
    model.train()
    losses = []
    rmses_shape = []
    rmses_flux = []

    for xb, ys_true, yf_true in loader:
        xb = xb.to(device)
        ys_true = ys_true.to(device)
        yf_true = yf_true.to(device)

        ys_pred, yf_pred = model(xb)
        l_shape = torch.mean((ys_pred - ys_true) ** 2)
        l_flux = torch.mean((yf_pred - yf_true) ** 2)
        loss = w_shape * l_shape + w_flux * l_flux

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        rmses_shape.append(float(rmse_torch(ys_pred, ys_true).item()))
        rmses_flux.append(float(rmse_torch(yf_pred, yf_true).item()))

    return {
        "loss": float(np.mean(losses) if losses else np.nan),
        "rmse_shape": float(np.mean(rmses_shape) if rmses_shape else np.nan),
        "rmse_flux": float(np.mean(rmses_flux) if rmses_flux else np.nan),
    }


def write_history_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_npz", default="")
    ap.add_argument("--run_name", default="nn_pixelsflux_mlp")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--loss_w_shape", type=float, default=1.0)
    ap.add_argument("--loss_w_flux", type=float, default=0.20)
    ap.add_argument("--raw_flux_clip_min", type=float, default=-6.0)
    ap.add_argument("--raw_flux_clip_max", type=float, default=8.0)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    args = ap.parse_args()

    cfg = Cfg(
        manifest_npz=Path(args.manifest_npz) if str(args.manifest_npz).strip() else None,
        run_name=str(args.run_name),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        loss_w_shape=float(args.loss_w_shape),
        loss_w_flux=float(args.loss_w_flux),
        raw_flux_clip_min=float(args.raw_flux_clip_min),
        raw_flux_clip_max=float(args.raw_flux_clip_max),
        max_train_samples=int(args.max_train_samples),
        max_val_samples=int(args.max_val_samples),
        max_test_samples=int(args.max_test_samples),
    )

    seed_everything(cfg.seed)

    run_id = f"{cfg.run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = cfg.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "train.log")

    try:
        logger.log(f"run_id={run_id}")
        logger.log(f"manifest={cfg.manifest_npz}")
        logger.log(f"tail command: tail -f {out_dir / 'train.log'}")

        m = load_manifest(cfg.manifest_npz)
        n_feat = int(m["n_features"])
        D = int(m["D"])
        stamp_pix = int(m["stamp_pix"])

        n_train_all = int(m["n_train"])
        n_val_all = int(m["n_val"])
        n_test_all = int(m["n_test"])

        n_train = n_train_all if cfg.max_train_samples <= 0 else min(n_train_all, cfg.max_train_samples)
        n_val = n_val_all if cfg.max_val_samples <= 0 else min(n_val_all, cfg.max_val_samples)
        n_test = n_test_all if cfg.max_test_samples <= 0 else min(n_test_all, cfg.max_test_samples)

        logger.log(f"dims: n_features={n_feat}, D={D}, stamp_pix={stamp_pix}")
        logger.log(f"rows: train={n_train}/{n_train_all}, val={n_val}/{n_val_all}, test={n_test}/{n_test_all}")

        X_train = np.memmap(m["X_train_path"], dtype="float32", mode="r", shape=(n_train_all, n_feat))
        Yshape_train = np.memmap(m["Yshape_train_path"], dtype="float32", mode="r", shape=(n_train_all, D))
        Yflux_train = np.memmap(m["Yflux_train_path"], dtype="float32", mode="r", shape=(n_train_all,))

        X_val = np.memmap(m["X_val_path"], dtype="float32", mode="r", shape=(n_val_all, n_feat))
        Yshape_val = np.memmap(m["Yshape_val_path"], dtype="float32", mode="r", shape=(n_val_all, D))
        Yflux_val = np.memmap(m["Yflux_val_path"], dtype="float32", mode="r", shape=(n_val_all,))

        X_test = np.memmap(m["X_test_path"], dtype="float32", mode="r", shape=(n_test_all, n_feat))
        Yshape_test = np.memmap(m["Yshape_test_path"], dtype="float32", mode="r", shape=(n_test_all, D))
        Yflux_test = np.memmap(m["Yflux_test_path"], dtype="float32", mode="r", shape=(n_test_all,))

        ds_train = MemmapDataset(X_train, Yshape_train, Yflux_train, n_use=n_train)
        ds_val = MemmapDataset(X_val, Yshape_val, Yflux_val, n_use=n_val)
        ds_test = MemmapDataset(X_test, Yshape_test, Yflux_test, n_use=n_test)

        dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.log(f"device={device}")

        model = ShapeFluxMLP(
            n_feat=n_feat,
            D=D,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            dropout=cfg.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_val = math.inf
        best_state = None
        wait = 0
        history: List[Dict[str, float]] = []

        for ep in range(1, cfg.epochs + 1):
            t0 = time.time()
            tr = train_one_epoch(
                model=model,
                loader=dl_train,
                device=device,
                optimizer=optimizer,
                w_shape=cfg.loss_w_shape,
                w_flux=cfg.loss_w_flux,
            )
            va = evaluate(
                model=model,
                loader=dl_val,
                device=device,
                w_shape=cfg.loss_w_shape,
                w_flux=cfg.loss_w_flux,
                raw_flux_clip_min=cfg.raw_flux_clip_min,
                raw_flux_clip_max=cfg.raw_flux_clip_max,
            )

            row = {
                "epoch": ep,
                "train_loss": tr["loss"],
                "train_rmse_shape": tr["rmse_shape"],
                "train_rmse_flux": tr["rmse_flux"],
                "val_loss": va["loss"],
                "val_rmse_shape": va["rmse_shape"],
                "val_rmse_flux": va["rmse_flux"],
                "val_rmse_raw": va["rmse_raw"],
                "epoch_seconds": time.time() - t0,
            }
            history.append(row)

            logger.log(
                f"epoch={ep:03d} "
                f"train_loss={row['train_loss']:.6f} "
                f"val_loss={row['val_loss']:.6f} "
                f"val_rmse_shape={row['val_rmse_shape']:.6f} "
                f"val_rmse_flux={row['val_rmse_flux']:.6f} "
                f"val_rmse_raw={row['val_rmse_raw']:.6f} "
                f"sec={row['epoch_seconds']:.1f}"
            )

            score = va["loss"]
            if score < best_val:
                best_val = score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                wait = 0
                torch.save(best_state, out_dir / "best_model.pt")
            else:
                wait += 1

            if wait >= cfg.early_patience:
                logger.log(f"early_stop at epoch={ep}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        te = evaluate(
            model=model,
            loader=dl_test,
            device=device,
            w_shape=cfg.loss_w_shape,
            w_flux=cfg.loss_w_flux,
            raw_flux_clip_min=cfg.raw_flux_clip_min,
            raw_flux_clip_max=cfg.raw_flux_clip_max,
        )

        write_history_csv(history, out_dir / "history.csv")
        torch.save(model.state_dict(), out_dir / "last_model.pt")

        metrics = {
            "run_id": run_id,
            "manifest_npz": str(cfg.manifest_npz),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_features": n_feat,
            "D": D,
            "stamp_pix": stamp_pix,
            "best_val_loss": float(best_val),
            "test": te,
        }

        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)

        logger.log(f"saved_run={out_dir}")
        logger.log(f"test_metrics={json.dumps(te)}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
