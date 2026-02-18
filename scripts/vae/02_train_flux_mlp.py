#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import ensure_dirs, load_manifest, open_split_memmaps


@dataclass
class Cfg:
    batch_size: int = 512
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 256
    depth: int = 3
    dropout: float = 0.1
    seed: int = 11
    num_workers: int = 0
    early_patience: int = 8


class FluxDataset(Dataset):
    def __init__(self, X: np.memmap, y: np.memmap, n_use: int):
        self.X = X
        self.y = y
        self.n = int(n_use)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        x = np.asarray(self.X[i], dtype=np.float32).copy()
        y = np.float32(self.y[i])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class FluxMLP(nn.Module):
    def __init__(self, n_feat: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        d = n_feat
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z).squeeze(1)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((a - b) ** 2))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    rmses: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yp = model(xb)
            loss = nn.functional.smooth_l1_loss(yp, yb)
            losses.append(float(loss.item()))
            rmses.append(float(rmse(yp, yb).item()))
    return {"huber": float(np.mean(losses)), "rmse": float(np.mean(rmses))}


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optim: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    losses: List[float] = []
    rmses: List[float] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yp = model(xb)
        loss = nn.functional.smooth_l1_loss(yp, yb)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        losses.append(float(loss.item()))
        rmses.append(float(rmse(yp, yb).item()))
    return {"huber": float(np.mean(losses)), "rmse": float(np.mean(rmses))}


def write_history(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run_name", default="flux_mlp_v1")
    ap.add_argument("--epochs", type=int, default=Cfg.epochs)
    ap.add_argument("--batch_size", type=int, default=Cfg.batch_size)
    ap.add_argument("--lr", type=float, default=Cfg.lr)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=Cfg.seed)
    args = ap.parse_args()

    cfg = Cfg(batch_size=int(args.batch_size), epochs=int(args.epochs), lr=float(args.lr), seed=int(args.seed))
    seed_all(cfg.seed)

    manifest = load_manifest(Path(args.manifest))
    Xtr, _, Yftr = open_split_memmaps(manifest, "train")
    Xva, _, Yfva = open_split_memmaps(manifest, "val")

    ntr = Xtr.shape[0] if int(args.max_train) <= 0 else min(Xtr.shape[0], int(args.max_train))
    nva = Xva.shape[0] if int(args.max_val) <= 0 else min(Xva.shape[0], int(args.max_val))

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = base_dir / "output" / "ml_runs" / "vae" / args.run_name
    ensure_dirs(out_dir)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(cfg),
                "manifest": str(manifest.path),
                "n_train_used": int(ntr),
                "n_val_used": int(nva),
                "feature_cols": manifest.feature_cols,
            },
            f,
            indent=2,
        )

    tr_loader = DataLoader(FluxDataset(Xtr, Yftr, ntr), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    va_loader = DataLoader(FluxDataset(Xva, Yfva, nva), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FluxMLP(manifest.n_features, cfg.hidden, cfg.depth, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(model, tr_loader, device, opt)
        va = evaluate(model, va_loader, device)
        row = {
            "epoch": ep,
            "train_huber": tr["huber"],
            "train_rmse": tr["rmse"],
            "val_huber": va["huber"],
            "val_rmse": va["rmse"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if va["rmse"] < best_val:
            best_val = va["rmse"]
            best_epoch = ep
            bad_epochs = 0
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val_rmse": best_val}, out_dir / "best_flux_mlp.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_patience:
                break

    write_history(history, out_dir / "history.csv")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_val_rmse": best_val}, f, indent=2)
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
