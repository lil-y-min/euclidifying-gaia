#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common import ensure_dirs, load_manifest, open_split_memmaps


@dataclass
class Cfg:
    z_dim: int = 16
    batch_size: int = 256
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 11
    num_workers: int = 0
    early_patience: int = 12
    beta_max: float = 1.0
    beta_warmup_frac: float = 0.35
    recon_huber_delta: float = 0.02
    free_bits_per_dim: float = 0.0
    model_width: int = 2


class ShapeDataset(Dataset):
    def __init__(self, X: np.memmap, yshape: np.memmap, n_use: int):
        self.X = X
        self.Y = yshape
        self.n = int(n_use)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        x = np.asarray(self.X[i], dtype=np.float32).copy()
        y = np.asarray(self.Y[i], dtype=np.float32).copy()
        s = float(np.sum(y))
        if s > 0:
            y /= s
        return torch.from_numpy(x), torch.from_numpy(y)


class Encoder(nn.Module):
    def __init__(self, n_feat: int, z_dim: int, width: int = 2):
        super().__init__()
        c1 = 16 * width
        c2 = 32 * width
        xh = 64 * width
        fh = 256 * width
        self.enc_img = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1),  # 10x10
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),  # 5x5
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1),  # 5x5
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
            nn.ConvTranspose2d(c0, c1, kernel_size=4, stride=2, padding=1),  # 10x10
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),  # 20x20
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

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.enc(y, x)
        z = self.reparameterize(mu, logvar)
        y_hat = self.dec(z, x)
        return y_hat, mu, logvar

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros((x.shape[0], self.enc.mu.out_features), device=x.device, dtype=x.dtype)
        return self.dec(z, x)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu * mu - 1.0 - logvar, dim=1)


def beta_at_epoch(epoch: int, total_epochs: int, beta_max: float, warmup_frac: float) -> float:
    warmup_epochs = max(1, int(round(total_epochs * warmup_frac)))
    return float(beta_max) * min(1.0, float(epoch) / float(warmup_epochs))


def recon_huber(pred: torch.Tensor, true: torch.Tensor, delta: float) -> torch.Tensor:
    return nn.functional.huber_loss(pred, true, delta=delta, reduction="mean")


def make_train_loader(
    ds: Dataset,
    batch_size: int,
    num_workers: int,
    sample_weight: Optional[np.ndarray] = None,
) -> DataLoader:
    if sample_weight is None:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.shape[0] != len(ds):
        raise RuntimeError(f"sample_weight length mismatch: {w.shape[0]} vs {len(ds)}")
    w = np.clip(w, 1e-12, np.inf)
    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(ds), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)


def eval_epoch(model: CVAE, loader: DataLoader, device: torch.device, delta: float, beta_eval: float, free_bits: float) -> Dict[str, float]:
    model.eval()
    recs: List[float] = []
    kls_raw: List[float] = []
    kls_used: List[float] = []
    tots: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yp, mu, logvar = model(yb, xb)
            rec = recon_huber(yp, yb, delta=delta)
            kl_vec_raw = kl_per_sample(mu, logvar)
            kl_raw = torch.mean(kl_vec_raw)
            kl_vec = kl_vec_raw
            if free_bits > 0.0:
                kl_vec = torch.maximum(kl_vec, torch.full_like(kl_vec, free_bits * mu.shape[1]))
            kl_used = torch.mean(kl_vec)
            tot = rec + beta_eval * kl_used
            recs.append(float(rec.item()))
            kls_raw.append(float(kl_raw.item()))
            kls_used.append(float(kl_used.item()))
            tots.append(float(tot.item()))
    return {
        "recon": float(np.mean(recs)),
        "kl_raw": float(np.mean(kls_raw)),
        "kl_used": float(np.mean(kls_used)),
        "total": float(np.mean(tots)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run_name", default="cvae_shape_v1")
    ap.add_argument("--z_dim", type=int, default=Cfg.z_dim)
    ap.add_argument("--epochs", type=int, default=Cfg.epochs)
    ap.add_argument("--batch_size", type=int, default=Cfg.batch_size)
    ap.add_argument("--lr", type=float, default=Cfg.lr)
    ap.add_argument("--beta_max", type=float, default=Cfg.beta_max)
    ap.add_argument("--beta_warmup_frac", type=float, default=Cfg.beta_warmup_frac)
    ap.add_argument("--free_bits_per_dim", type=float, default=Cfg.free_bits_per_dim)
    ap.add_argument("--model_width", type=int, default=Cfg.model_width, help="Width multiplier for CNN/MLP blocks.")
    ap.add_argument("--sample_weight_train_npy", default="", help="Optional weights for train oversampling.")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=Cfg.seed)
    args = ap.parse_args()

    cfg = Cfg(
        z_dim=int(args.z_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        beta_max=float(args.beta_max),
        beta_warmup_frac=float(args.beta_warmup_frac),
        free_bits_per_dim=float(args.free_bits_per_dim),
        model_width=int(args.model_width),
        seed=int(args.seed),
    )
    seed_all(cfg.seed)

    manifest = load_manifest(Path(args.manifest))
    Xtr, Ystr, _ = open_split_memmaps(manifest, "train")
    Xva, Ysva, _ = open_split_memmaps(manifest, "val")
    ntr = Xtr.shape[0] if int(args.max_train) <= 0 else min(Xtr.shape[0], int(args.max_train))
    nva = Xva.shape[0] if int(args.max_val) <= 0 else min(Xva.shape[0], int(args.max_val))

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = base_dir / "output" / "ml_runs" / "vae" / args.run_name
    ensure_dirs(out_dir)

    sample_weight = None
    if str(args.sample_weight_train_npy).strip():
        sample_weight = np.load(str(args.sample_weight_train_npy))
        sample_weight = np.asarray(sample_weight[:ntr], dtype=np.float64)

    ds_train = ShapeDataset(Xtr, Ystr, ntr)
    ds_val = ShapeDataset(Xva, Ysva, nva)
    train_loader = make_train_loader(ds_train, cfg.batch_size, cfg.num_workers, sample_weight=sample_weight)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(manifest.n_features, cfg.z_dim, width=cfg.model_width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(cfg),
                "manifest": str(manifest.path),
                "n_train_used": int(ntr),
                "n_val_used": int(nva),
                "feature_cols": manifest.feature_cols,
                "sample_weight_train_npy": str(args.sample_weight_train_npy),
            },
            f,
            indent=2,
        )

    history_path = out_dir / "history.csv"
    latest_path = out_dir / "latest_epoch.json"
    history_fields = [
        "epoch",
        "beta",
        "train_recon",
        "train_kl_raw",
        "train_kl_used",
        "train_total",
        "val_recon",
        "val_kl_raw",
        "val_kl_used",
        "val_total_beta1",
        "epoch_sec",
    ]
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=history_fields)
        w.writeheader()

    hist: List[Dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for ep in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        model.train()
        beta = beta_at_epoch(ep, cfg.epochs, cfg.beta_max, cfg.beta_warmup_frac)
        tr_rec: List[float] = []
        tr_kl_raw: List[float] = []
        tr_kl_used: List[float] = []
        tr_tot: List[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yp, mu, logvar = model(yb, xb)

            rec = recon_huber(yp, yb, delta=cfg.recon_huber_delta)
            kl_vec_raw = kl_per_sample(mu, logvar)
            kl_raw = torch.mean(kl_vec_raw)
            kl_vec = kl_vec_raw
            if cfg.free_bits_per_dim > 0.0:
                kl_vec = torch.maximum(kl_vec, torch.full_like(kl_vec, cfg.free_bits_per_dim * cfg.z_dim))
            kl_used = torch.mean(kl_vec)
            tot = rec + beta * kl_used

            opt.zero_grad(set_to_none=True)
            tot.backward()
            opt.step()

            tr_rec.append(float(rec.item()))
            tr_kl_raw.append(float(kl_raw.item()))
            tr_kl_used.append(float(kl_used.item()))
            tr_tot.append(float(tot.item()))

        va = eval_epoch(
            model,
            val_loader,
            device=device,
            delta=cfg.recon_huber_delta,
            beta_eval=1.0,
            free_bits=cfg.free_bits_per_dim,
        )
        row = {
            "epoch": ep,
            "beta": beta,
            "train_recon": float(np.mean(tr_rec)),
            "train_kl_raw": float(np.mean(tr_kl_raw)),
            "train_kl_used": float(np.mean(tr_kl_used)),
            "train_total": float(np.mean(tr_tot)),
            "val_recon": va["recon"],
            "val_kl_raw": va["kl_raw"],
            "val_kl_used": va["kl_used"],
            "val_total_beta1": va["total"],
            "epoch_sec": float(time.time() - ep_t0),
        }
        hist.append(row)
        print(json.dumps(row), flush=True)
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=history_fields)
            w.writerow(row)
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

        if va["recon"] < best_val:
            best_val = va["recon"]
            best_epoch = ep
            bad_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": ep,
                    "val_recon": best_val,
                    "n_features": manifest.n_features,
                    "z_dim": cfg.z_dim,
                    "model_width": cfg.model_width,
                },
                out_dir / "best_cvae.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_patience:
                break

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_val_recon": best_val}, f, indent=2)
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
