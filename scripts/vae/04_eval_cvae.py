#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import ensure_dirs, image_moments, load_manifest, open_split_memmaps, raw_from_shape_flux


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

    def sample_pred(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        b = x.shape[0]
        z = torch.randn((n_samples, b, self.z_dim), device=x.device, dtype=x.dtype)
        xrep = x[None, :, :].expand(n_samples, b, x.shape[1])
        ys = []
        for k in range(n_samples):
            ys.append(self.dec(z[k], xrep[k]))
        return torch.stack(ys, dim=0)


class FluxMLP(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 256, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d = n_feat
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)


def rmse_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def zrms_rows(z: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(z * z, axis=1))


def summarize(v: np.ndarray) -> Dict[str, float]:
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {"mean": float(np.mean(vv)), "median": float(np.median(vv)), "p90": float(np.percentile(vv, 90.0))}


def predict_flux(X: np.ndarray, mode: str, flux_model_path: str, yflux_true: np.ndarray, device: torch.device) -> np.ndarray:
    if mode == "true":
        return np.asarray(yflux_true, dtype=np.float32)
    if mode != "pred":
        raise ValueError(f"Unknown flux_mode: {mode}")
    ckpt = torch.load(flux_model_path, map_location=device)
    model = FluxMLP(X.shape[1]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    out = np.empty(X.shape[0], dtype=np.float32)
    bs = 2048
    with torch.no_grad():
        for s in range(0, X.shape[0], bs):
            e = min(X.shape[0], s + bs)
            xb = torch.from_numpy(np.asarray(X[s:e], dtype=np.float32).copy()).to(device)
            out[s:e] = model(xb).cpu().numpy().astype(np.float32)
    return out


def gallery(path: Path, true_img: np.ndarray, pred_img: np.ndarray, zmap: np.ndarray, title: str, vmax_abs: float = 10.0) -> None:
    n = true_img.shape[0]
    if n == 0:
        return
    d = int(true_img.shape[1])
    pix = int(round(np.sqrt(d)))
    if pix * pix != d:
        raise RuntimeError(f"Gallery expects flattened square stamps; got D={d}")
    fig, axes = plt.subplots(n, 3, figsize=(8.5, max(2.2, n * 1.65)))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        t = true_img[i].reshape(pix, pix)
        p = pred_img[i].reshape(pix, pix)
        z = zmap[i].reshape(pix, pix)
        for j, im, ttl, cmap, vmin, vmax in [
            (0, t, "TRUE", "magma", None, None),
            (1, p, "PRED", "magma", None, None),
            (2, z, "Z", "coolwarm", -vmax_abs, vmax_abs),
        ]:
            ax = axes[i, j]
            m = ax.imshow(im, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(ttl, fontsize=10)
            fig.colorbar(m, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cvae_ckpt", required=True)
    ap.add_argument("--run_name", default="cvae_eval_v1")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n_samples", type=int, default=8, help="Number of cVAE samples for best-of-N.")
    ap.add_argument("--max_eval", type=int, default=0)
    ap.add_argument("--flux_mode", default="true", choices=["true", "pred"])
    ap.add_argument("--flux_ckpt", default="")
    ap.add_argument("--sigma_mode", default="sqrt_Itrue", choices=["sqrt_Itrue"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--sigma0", type=float, default=1e-4)
    ap.add_argument("--gallery_n", type=int, default=12)
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    Xmm, Yshape_mm, Yflux_mm = open_split_memmaps(manifest, args.split)
    n = Xmm.shape[0] if int(args.max_eval) <= 0 else min(Xmm.shape[0], int(args.max_eval))
    X = np.asarray(Xmm[:n], dtype=np.float32).copy()
    Yshape = np.asarray(Yshape_mm[:n], dtype=np.float32)
    Yflux_true = np.asarray(Yflux_mm[:n], dtype=np.float32)

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = base_dir / "output" / "ml_runs" / "vae" / args.run_name
    plot_dir = base_dir / "plots" / "ml_runs" / "vae" / args.run_name
    rep_dir = base_dir / "report" / "model_decision" / "vae" / args.run_name
    ensure_dirs(out_dir, plot_dir, rep_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = torch.load(args.cvae_ckpt, map_location=device)
    n_feat = int(c.get("n_features", manifest.n_features))
    z_dim = int(c.get("z_dim", 16))
    model_width = int(c.get("model_width", 2))
    model = CVAE(n_feat, z_dim, width=model_width).to(device)
    model.load_state_dict(c["state_dict"])
    model.eval()

    X_t = torch.from_numpy(X).to(device)
    bs = 1024
    y_mean = np.empty_like(Yshape)
    y_best = np.empty_like(Yshape)
    rmse_mean = np.empty(n, dtype=np.float32)
    rmse_best = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(n, s + bs)
            xb = X_t[s:e]
            yp_mean = model.mean_pred(xb).cpu().numpy().astype(np.float32)
            ys = model.sample_pred(xb, n_samples=int(args.n_samples)).cpu().numpy().astype(np.float32)  # [K,B,D]

            y_true_b = Yshape[s:e]
            err_mean = rmse_rows(yp_mean, y_true_b)
            err_k = np.stack([rmse_rows(ys[k], y_true_b) for k in range(ys.shape[0])], axis=0)
            kbest = np.argmin(err_k, axis=0)
            yp_best = ys[kbest, np.arange(ys.shape[1])]
            err_best = err_k[kbest, np.arange(ys.shape[1])]

            y_mean[s:e] = yp_mean
            y_best[s:e] = yp_best
            rmse_mean[s:e] = err_mean
            rmse_best[s:e] = err_best

    yflux_eval = predict_flux(X, args.flux_mode, args.flux_ckpt, Yflux_true, device=device)

    I_true = raw_from_shape_flux(Yshape, Yflux_true)
    I_mean = raw_from_shape_flux(y_mean, yflux_eval)
    I_best = raw_from_shape_flux(y_best, yflux_eval)

    resid_mean = I_mean - I_true
    sig2 = np.maximum(args.alpha * np.maximum(I_true, 0.0) + args.sigma0 ** 2, 1e-30)
    z_mean = resid_mean / np.sqrt(sig2)
    z_rms_mean = zrms_rows(z_mean)

    moms_t = image_moments(Yshape, manifest.stamp_pix)
    moms_m = image_moments(y_mean, manifest.stamp_pix)
    moms_b = image_moments(y_best, manifest.stamp_pix)

    metrics = {
        "manifest": str(manifest.path),
        "split": args.split,
        "n_eval": int(n),
        "n_samples_bestofN": int(args.n_samples),
        "flux_mode": args.flux_mode,
        "rmse_shape_mean": float(np.mean(rmse_mean)),
        "rmse_shape_bestofN": float(np.mean(rmse_best)),
        "z_rms_mean_pred": summarize(z_rms_mean),
        "ell_true": summarize(moms_t["ell"]),
        "ell_mean_pred": summarize(moms_m["ell"]),
        "ell_bestofN_pred": summarize(moms_b["ell"]),
        "sigma_true": summarize(moms_t["sigma"]),
        "sigma_mean_pred": summarize(moms_m["sigma"]),
        "sigma_bestofN_pred": summarize(moms_b["sigma"]),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(rep_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    rows = []
    for i in range(n):
        rows.append(
            {
                "idx": i,
                "rmse_shape_mean": float(rmse_mean[i]),
                "rmse_shape_bestofN": float(rmse_best[i]),
                "ell_true": float(moms_t["ell"][i]),
                "ell_mean": float(moms_m["ell"][i]),
                "ell_bestofN": float(moms_b["ell"][i]),
                "sigma_true": float(moms_t["sigma"][i]),
                "sigma_mean": float(moms_m["sigma"][i]),
                "sigma_bestofN": float(moms_b["sigma"][i]),
                "z_rms_mean": float(z_rms_mean[i]),
            }
        )
    with open(out_dir / "per_object.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["idx"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    rng = np.random.default_rng(123)
    k = min(int(args.gallery_n), n)
    # Rank galleries by sigma-normalized residual amplitude, not RMSE.
    idx_best = np.argsort(z_rms_mean)[:k]
    idx_worst = np.argsort(z_rms_mean)[-k:]
    idx_rand = rng.choice(n, size=k, replace=False) if n > 0 else np.array([], dtype=int)

    gallery(plot_dir / "best_true_pred_z.png", I_true[idx_best], I_mean[idx_best], z_mean[idx_best], "Best by z-RMS (sigma-normalized residual)", vmax_abs=10.0)
    gallery(plot_dir / "worst_true_pred_z.png", I_true[idx_worst], I_mean[idx_worst], z_mean[idx_worst], "Worst by z-RMS (sigma-normalized residual)", vmax_abs=10.0)
    gallery(plot_dir / "random_true_pred_z.png", I_true[idx_rand], I_mean[idx_rand], z_mean[idx_rand], "Random sample", vmax_abs=10.0)

    plt.figure(figsize=(6.4, 4.4))
    plt.hist(z_rms_mean[np.isfinite(z_rms_mean)], bins=90)
    plt.xlabel("z-RMS per object")
    plt.ylabel("Count")
    plt.title("Sigma-normalized residual amplitude")
    plt.tight_layout()
    plt.savefig(plot_dir / "z_rms_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.4, 4.4))
    plt.hist(moms_t["ell"], bins=80, alpha=0.45, label="true")
    plt.hist(moms_m["ell"], bins=80, alpha=0.45, label="mean_pred")
    plt.hist(moms_b["ell"], bins=80, alpha=0.45, label="best_of_N")
    plt.xlabel("Ellipticity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "ellipticity_distribution_compare.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.4, 4.4))
    plt.hist(moms_t["sigma"], bins=80, alpha=0.45, label="true")
    plt.hist(moms_m["sigma"], bins=80, alpha=0.45, label="mean_pred")
    plt.hist(moms_b["sigma"], bins=80, alpha=0.45, label="best_of_N")
    plt.xlabel("Sigma [pix]")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "sigma_distribution_compare.png", dpi=150)
    plt.close()

    print("Saved metrics:", out_dir / "metrics.json")
    print("Saved plots  :", plot_dir)
    print("Report copy  :", rep_dir / "metrics.json")


if __name__ == "__main__":
    main()
