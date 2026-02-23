#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_recon"], label="train_recon")
    ax.plot(df["epoch"], df["val_recon"], label="val_recon")
    ax.set_title("Reconstruction")
    ax.set_xlabel("epoch")
    ax.set_ylabel("huber")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_kl_raw"], label="train_kl_raw")
    ax.plot(df["epoch"], df["val_kl_raw"], label="val_kl_raw")
    ax.plot(df["epoch"], df["train_kl_used"], "--", label="train_kl_used")
    ax.plot(df["epoch"], df["val_kl_used"], "--", label="val_kl_used")
    ax.set_title("KL raw vs used")
    ax.set_xlabel("epoch")
    ax.set_ylabel("KL")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(df["epoch"], df["beta"], label="beta")
    ax.set_title("KL Weight")
    ax.set_xlabel("epoch")
    ax.set_ylabel("beta")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(df["epoch"], df["epoch_sec"], label="epoch_sec")
    ax.set_title("Epoch Runtime")
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds")
    ax.legend()

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True, help="VAE run name under output/ml_runs/vae/")
    ap.add_argument("--poll_sec", type=float, default=10.0)
    ap.add_argument("--max_wait_sec", type=float, default=0.0, help="0 means wait forever")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    run_dir = base_dir / "output" / "ml_runs" / "vae" / args.run_name
    history_csv = run_dir / "history.csv"
    out_png = run_dir / "live_training_diagnostics.png"

    t0 = time.time()
    last_n = -1

    while True:
        if history_csv.exists():
            try:
                df = pd.read_csv(history_csv)
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                n = len(df)
                if n != last_n:
                    render(df, out_png)
                    last_n = n
                    print(f"[monitor] epochs={n} updated {out_png}", flush=True)
                if args.once:
                    return
        else:
            if args.once:
                print(f"[monitor] history file not found yet: {history_csv}", flush=True)
                return

        if args.max_wait_sec > 0 and (time.time() - t0) > args.max_wait_sec:
            print("[monitor] max_wait_sec reached; exiting", flush=True)
            return
        time.sleep(max(0.5, args.poll_sec))


if __name__ == "__main__":
    main()
