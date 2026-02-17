#!/usr/bin/env python3
"""
CIFAR-10 NF+JEPA training script (single-GPU).

Trains a Mamba-GINR-style neural field encoder with JEPA-style
predictive embedding loss and optional reconstruction loss.

Architecture:
  - Context encoder (Mamba/Transformer): sparse (coord, value) → embeddings
  - Target encoder (EMA copy): all pixels → target embeddings
  - Predictor: context features + target coords → predicted embeddings
  - INR decoder (optional): global z + query coords → RGB

Usage:
    python train.py                                    # defaults
    python train.py --config config.yaml               # from YAML
    python train.py --obs_frac 0.1 --recon_weight 0.1  # with recon
    python train.py --use_mamba false                   # transformer fallback
"""

import argparse
import copy
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import yaml

from models import (
    image_to_coord_value,
    make_nf_jepa_model,
    sample_sparse_observations,
    sample_target_set,
    MAMBA_AVAILABLE,
)

# ---------------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, final_lr, T_max):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / max(1, self.warmup_steps)
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = float(self._step - self.warmup_steps) / max(1, self.T_max)
            new_lr = max(
                self.final_lr,
                self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )
        for g in self.optimizer.param_groups:
            g["lr"] = new_lr
        return new_lr


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # data
    data_root="./data",
    batch_size=128,
    num_workers=4,
    img_size=32,
    # sparse observation
    obs_frac=0.1,         # fraction of pixels observed by context encoder
    target_frac=0.2,      # fraction of pixels as prediction targets
    exclude_context=True,  # whether to exclude context pixels from target set
    # model
    d_model=192,
    n_layers=4,
    num_freqs=32,
    use_mamba=True,
    nhead=4,
    pred_dim=96,
    pred_layers=3,
    pred_mode="cross_attn",  # 'cross_attn' or 'pool'
    # INR decoder
    use_inr_decoder=True,
    inr_hidden=256,
    inr_layers=4,
    # loss
    jepa_weight=1.0,
    recon_weight=0.1,        # set to 0.0 for JEPA-only
    jepa_loss_type="mse",    # 'mse' or 'cosine'
    # optimization
    epochs=200,
    lr=1e-3,
    start_lr=1e-4,
    final_lr=1e-6,
    warmup=10,
    weight_decay=0.05,
    ema_start=0.996,
    ema_end=1.0,
    use_amp=False,
    # logging
    log_freq=50,
    checkpoint_freq=10,
    output_dir="./output/cifar10_nf_jepa",
    seed=42,
)


def load_config(args):
    cfg = dict(DEFAULTS)
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        if yaml_cfg:
            cfg.update(yaml_cfg)
    for k, v in vars(args).items():
        if k == "config":
            continue
        if v is not None:
            cfg[k] = v
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 NF+JEPA training")
    p.add_argument("--config", type=str, default="config.yaml")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda x: x.lower() in ("true", "1"), default=None)
        elif isinstance(v, float):
            p.add_argument(f"--{k}", type=float, default=None)
        elif isinstance(v, int):
            p.add_argument(f"--{k}", type=int, default=None)
        else:
            p.add_argument(f"--{k}", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_cifar10_loader(cfg):
    """CIFAR-10 loader with minimal augmentation (JEPA uses pixel info)."""
    transform = transforms.Compose([
        transforms.RandomCrop(cfg["img_size"], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Keep values in [0, 1] range for reconstruction; normalize coords instead.
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=cfg["data_root"], train=True, download=True, transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    return loader


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def jepa_loss_fn(pred, target, loss_type="mse"):
    """Compute JEPA embedding loss."""
    if loss_type == "cosine":
        pred_norm = F.normalize(pred, dim=-1)
        tgt_norm = F.normalize(target, dim=-1)
        return (1.0 - (pred_norm * tgt_norm).sum(dim=-1)).mean()
    else:
        return F.mse_loss(pred, target)


def reconstruction_loss_fn(pred_rgb, target_rgb):
    """MSE reconstruction loss."""
    return F.mse_loss(pred_rgb, target_rgb)


def compute_psnr(pred, target):
    """PSNR in dB (assumes [0, 1] range)."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return -10 * math.log10(mse)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args)

    seed = cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Mamba available: {MAMBA_AVAILABLE}")
    logger.info(f"Config: {cfg}")

    with open(os.path.join(cfg["output_dir"], "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # ---- Model ----
    model = make_nf_jepa_model(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        num_freqs=cfg["num_freqs"],
        use_mamba=cfg["use_mamba"],
        nhead=cfg["nhead"],
        pred_dim=cfg["pred_dim"],
        pred_layers=cfg["pred_layers"],
        pred_mode=cfg["pred_mode"],
        use_inr_decoder=cfg["use_inr_decoder"],
        inr_hidden=cfg["inr_hidden"],
        inr_layers=cfg["inr_layers"],
    ).to(device)

    # Target encoder: EMA copy of the context encoder
    target_encoder = copy.deepcopy(model.encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    logger.info(f"Total model params: {total_params:,}  Encoder params: {enc_params:,}")
    logger.info(f"Using {'Mamba' if model.encoder.use_mamba else 'Transformer (fallback)'} encoder")

    # ---- Data ----
    train_loader = make_cifar10_loader(cfg)
    ipe = len(train_loader)
    logger.info(f"Training batches per epoch: {ipe}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    T_max = cfg["epochs"] * ipe
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(cfg["warmup"] * ipe),
        start_lr=cfg["start_lr"],
        ref_lr=cfg["lr"],
        final_lr=cfg["final_lr"],
        T_max=T_max,
    )

    scaler = torch.amp.GradScaler("cuda") if cfg["use_amp"] and device.type == "cuda" else None

    # EMA schedule
    ema_s, ema_e = cfg["ema_start"], cfg["ema_end"]
    momentum_schedule = (ema_s + i * (ema_e - ema_s) / T_max for i in range(T_max + 1))

    # ---- Training loop ----
    logger.info("Starting NF+JEPA training on CIFAR-10")
    H = W = cfg["img_size"]
    N_pixels = H * W

    for epoch in range(cfg["epochs"]):
        model.train()
        loss_jepa_sum, loss_recon_sum, loss_total_sum = 0.0, 0.0, 0.0
        psnr_sum = 0.0
        n_batches = 0
        t0 = time.time()

        for itr, (imgs, _labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)  # (B, 3, H, W) in [0,1]
            B = imgs.size(0)

            # Convert images to coordinate-value pairs
            all_coords, all_values = image_to_coord_value(imgs)  # (B, N, 2), (B, N, 3)

            # Sample sparse context observations
            ctx_coords, ctx_values, ctx_idx = sample_sparse_observations(
                all_coords, all_values, cfg["obs_frac"],
            )

            # Sample target set (exclude context if configured)
            tgt_coords, tgt_values, tgt_idx = sample_target_set(
                all_coords, all_values, cfg["target_frac"],
                context_indices=ctx_idx if cfg["exclude_context"] else None,
            )

            _lr = lr_scheduler.step()

            with torch.amp.autocast("cuda", enabled=cfg["use_amp"] and device.type == "cuda"):
                # ---- Target encoder: encode ALL pixels → select target embeddings ----
                with torch.no_grad():
                    tgt_token_feats, _ = target_encoder(all_coords, all_values)
                    # Select embeddings at target positions
                    tgt_embeddings = torch.gather(
                        tgt_token_feats, 1,
                        tgt_idx.unsqueeze(-1).expand(-1, -1, tgt_token_feats.size(-1)),
                    )
                    tgt_embeddings = F.layer_norm(tgt_embeddings, (tgt_embeddings.size(-1),))

                # ---- Context encoder: encode sparse observations ----
                ctx_token_feats, z_global = model.forward_context(ctx_coords, ctx_values)

                # ---- Predictor: predict target embeddings ----
                pred_embeddings = model.forward_predict(
                    ctx_token_feats, tgt_coords, z_global,
                )

                # ---- Losses ----
                loss_jepa = jepa_loss_fn(pred_embeddings, tgt_embeddings, cfg["jepa_loss_type"])
                loss = cfg["jepa_weight"] * loss_jepa

                loss_recon = torch.tensor(0.0, device=device)
                if cfg["use_inr_decoder"] and cfg["recon_weight"] > 0:
                    # Reconstruct ALL pixels for maximum supervision
                    pred_rgb = model.forward_reconstruct(z_global, all_coords)
                    loss_recon = reconstruction_loss_fn(pred_rgb, all_values)
                    loss = loss + cfg["recon_weight"] * loss_recon

            # Backward
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # EMA update of target encoder
            with torch.no_grad():
                m = next(momentum_schedule)
                for p_q, p_k in zip(model.encoder.parameters(), target_encoder.parameters()):
                    p_k.data.mul_(m).add_((1.0 - m) * p_q.detach().data)

            # Metrics
            loss_total_sum += loss.item()
            loss_jepa_sum += loss_jepa.item()
            loss_recon_sum += loss_recon.item()
            n_batches += 1

            if cfg["use_inr_decoder"] and cfg["recon_weight"] > 0:
                with torch.no_grad():
                    psnr_sum += compute_psnr(pred_rgb.clamp(0, 1), all_values)

            if itr % cfg["log_freq"] == 0:
                log_msg = (
                    f"[ep {epoch+1}/{cfg['epochs']}, itr {itr}/{ipe}] "
                    f"loss={loss.item():.4f}  jepa={loss_jepa.item():.4f}"
                )
                if cfg["recon_weight"] > 0:
                    psnr = psnr_sum / max(n_batches, 1)
                    log_msg += f"  recon={loss_recon.item():.4f}  psnr={psnr:.1f}dB"
                log_msg += f"  lr={_lr:.2e}  ema_m={m:.5f}"
                logger.info(log_msg)

            assert not np.isnan(loss.item()), "Loss is NaN!"

        # Epoch summary
        elapsed = time.time() - t0
        avg_jepa = loss_jepa_sum / max(n_batches, 1)
        avg_recon = loss_recon_sum / max(n_batches, 1)
        avg_total = loss_total_sum / max(n_batches, 1)
        avg_psnr = psnr_sum / max(n_batches, 1)

        summary = (
            f"Epoch {epoch+1} — total={avg_total:.4f}  jepa={avg_jepa:.4f}  "
            f"recon={avg_recon:.4f}"
        )
        if cfg["recon_weight"] > 0:
            summary += f"  psnr={avg_psnr:.1f}dB"
        summary += f"  time={elapsed:.1f}s"
        logger.info(summary)

        # ---- Verify target encoder has no gradients ----
        if epoch == 0:
            tgt_grad_ok = all(p.grad is None for p in target_encoder.parameters())
            logger.info(f"Target encoder gradient check: {'PASS' if tgt_grad_ok else 'FAIL'}")

        # Always save latest checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "config": cfg,
            "loss": avg_total,
        }
        torch.save(ckpt, os.path.join(cfg["output_dir"], "checkpoint_latest.pth"))

        # Save numbered checkpoint periodically
        if (epoch + 1) % cfg["checkpoint_freq"] == 0 or (epoch + 1) == cfg["epochs"]:
            path = os.path.join(cfg["output_dir"], f"checkpoint_ep{epoch+1}.pth")
            torch.save(ckpt, path)
            logger.info(f"Saved checkpoint → {path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
