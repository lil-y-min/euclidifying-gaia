#!/usr/bin/env python3
"""
Train a baseline CNN for PSF-like (0) vs non-PSF-like (1) classification.

Expected labels CSV from `psf_labeling.py` with at least:
- source_id
- label
- split_code
"""

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
from torch.utils.data import DataLoader, Dataset


@dataclass
class Cfg:
    base_dir: Path = Path(__file__).resolve().parents[2]
    dataset_root: Path = None
    labels_csv: Path = None
    out_root: Path = None
    run_name: str = "nn_cnn_s_img_psfweak_v1"

    seed: int = 11
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_patience: int = 6

    def __post_init__(self) -> None:
        if self.dataset_root is None:
            self.dataset_root = self.base_dir / "output" / "dataset_npz"
        if self.labels_csv is None:
            self.labels_csv = self.base_dir / "output" / "ml_runs" / "nn_psf_labels" / "labels_psf_weak.csv"
        if self.out_root is None:
            self.out_root = self.base_dir / "output" / "ml_runs"


class StampDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        x = x[None, :, :]  # CHW
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_field_dirs(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / "metadata.csv").exists()])


def _to_int(x: object) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def load_source_index(dataset_root: Path) -> Dict[int, Tuple[str, str, int]]:
    idx: Dict[int, Tuple[str, str, int]] = {}
    for fdir in find_field_dirs(dataset_root):
        meta_path = fdir / "metadata.csv"
        with open(meta_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = _to_int(row.get("source_id"))
                idx_in_file = _to_int(row.get("index_in_file"))
                npz_file = str(row.get("npz_file", "")).strip()
                if sid is None or idx_in_file is None or npz_file == "":
                    continue
                if sid in idx:
                    continue
                idx[sid] = (fdir.name, npz_file, idx_in_file)
    if not idx:
        raise RuntimeError(f"No metadata.csv found under {dataset_root}")
    return idx


def load_labels(path: Path) -> Dict[int, Tuple[int, int]]:
    out: Dict[int, Tuple[int, int]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = _to_int(row.get("source_id"))
            lab = _to_int(row.get("label"))
            split = _to_int(row.get("split_code"))
            if sid is None or lab is None:
                continue
            out[sid] = (int(lab), int(split) if split is not None else 0)
    if not out:
        raise RuntimeError(f"No usable labels found in {path}")
    return out


def robust_norm(img: np.ndarray) -> np.ndarray:
    x = np.array(img, dtype=np.float32)
    p1 = np.nanpercentile(x, 1)
    p99 = np.nanpercentile(x, 99)
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, p1, p99)
    x = (x - p1) / (p99 - p1 + 1e-9)
    return x


def build_arrays(cfg: Cfg, labels: Dict[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_map = load_source_index(cfg.dataset_root)
    cache: Dict[Tuple[str, str], np.ndarray] = {}
    xs: List[np.ndarray] = []
    ys: List[int] = []
    splits: List[int] = []

    for sid, (lab, split_code) in labels.items():
        if sid not in idx_map:
            continue
        field_tag, npz_file, idx = idx_map[sid]
        key = (field_tag, npz_file)
        npz_path = cfg.dataset_root / field_tag / npz_file
        if not npz_path.exists():
            continue
        if key not in cache:
            arr = np.load(npz_path)
            if "X" not in arr.files:
                continue
            cache[key] = np.array(arr["X"], dtype=np.float32)
        Xb = cache[key]
        if idx < 0 or idx >= Xb.shape[0]:
            continue
        xs.append(robust_norm(Xb[idx]))
        ys.append(int(lab))
        splits.append(int(split_code))

    if not xs:
        raise RuntimeError("No usable samples loaded for training")

    X = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    s = np.array(splits, dtype=np.int64)
    return X, y, s


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    acc = (tp + tn) / max(1, len(y_true))

    return {
        "accuracy": float(acc),
        "precision_non_psf": float(prec),
        "recall_non_psf": float(rec),
        "f1_non_psf": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer, criterion, train: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(mode=train)
    losses = []
    all_y = []
    all_p = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        pred = torch.argmax(logits, dim=1)
        all_y.append(yb.detach().cpu().numpy())
        all_p.append(pred.detach().cpu().numpy())

    y = np.concatenate(all_y) if all_y else np.array([], dtype=int)
    p = np.concatenate(all_p) if all_p else np.array([], dtype=int)
    return float(np.mean(losses) if losses else np.nan), y, p


def write_history_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="")
    ap.add_argument("--labels_csv", default="")
    ap.add_argument("--run_name", default="nn_cnn_s_img_psfweak_v1")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    cfg = Cfg(
        dataset_root=Path(args.dataset_root) if str(args.dataset_root).strip() else None,
        labels_csv=Path(args.labels_csv) if str(args.labels_csv).strip() else None,
        run_name=str(args.run_name),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )

    seed_everything(cfg.seed)

    run_id = f"{cfg.run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = cfg.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(cfg.labels_csv)

    X, y, s = build_arrays(cfg, labels)

    tr = s == 0
    va = s == 1
    te = s == 2

    if int(va.sum()) == 0:
        va = te
    if int(te.sum()) == 0:
        te = va

    ds_tr = StampDataset(X[tr], y[tr])
    ds_va = StampDataset(X[va], y[va])
    ds_te = StampDataset(X[te], y[te])

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)

    # Class weighting to handle imbalance on non-PSF class.
    n0 = float(np.sum(y[tr] == 0))
    n1 = float(np.sum(y[tr] == 1))
    w0 = 1.0
    w1 = n0 / max(1.0, n1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = -np.inf
    best_state = None
    wait = 0
    history: List[Dict[str, float]] = []

    for ep in range(1, cfg.epochs + 1):
        tr_loss, ytr, ptr = run_epoch(model, dl_tr, device, optimizer, criterion, train=True)
        va_loss, yva, pva = run_epoch(model, dl_va, device, optimizer, criterion, train=False)

        mtr = metrics_binary(ytr, ptr)
        mva = metrics_binary(yva, pva)

        row = {
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_f1_non_psf": mtr["f1_non_psf"],
            "val_f1_non_psf": mva["f1_non_psf"],
            "val_recall_non_psf": mva["recall_non_psf"],
            "val_precision_non_psf": mva["precision_non_psf"],
        }
        history.append(row)

        score = mva["f1_non_psf"]
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        print(f"[epoch {ep:03d}] train_f1={mtr['f1_non_psf']:.4f} val_f1={mva['f1_non_psf']:.4f} val_recall={mva['recall_non_psf']:.4f}")

        if wait >= cfg.early_patience:
            print("Early stopping triggered")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, yte, pte = run_epoch(model, dl_te, device, optimizer=None, criterion=criterion, train=False)
    mte = metrics_binary(yte, pte)

    torch.save(model.state_dict(), out_dir / "model.pt")
    write_history_csv(history, out_dir / "history.csv")

    metrics = {
        "run_id": run_id,
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "class_counts_train": {
            "psf_like_0": int(np.sum(y[tr] == 0)),
            "non_psf_like_1": int(np.sum(y[tr] == 1)),
        },
        "test_metrics": mte,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    print("Saved run:", out_dir)
    print("Test metrics:", mte)


if __name__ == "__main__":
    main()
