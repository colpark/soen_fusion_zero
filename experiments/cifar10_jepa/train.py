#!/usr/bin/env python3
"""
CIFAR-10 I-JEPA training script (single-GPU).

Trains a ViT encoder + predictor using the I-JEPA framework:
  - Context encoder sees masked context patches.
  - Predictor predicts target encoder embeddings at masked target positions.
  - Target encoder is an EMA copy of the context encoder.

Usage:
    python train.py                          # defaults
    python train.py --config config.yaml     # from YAML
    python train.py --epochs 100 --lr 1e-3   # CLI overrides
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
    MaskCollator,
    apply_masks,
    make_cifar10_jepa_model,
    repeat_interleave_batch,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedulers (self-contained, from ijepa)
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
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(
                self.final_lr,
                self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr


class CosineWDSchedule:
    def __init__(self, optimizer, ref_wd, final_wd, T_max):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
        new_wd = max(self.final_wd, new_wd) if self.final_wd <= self.ref_wd else min(self.final_wd, new_wd)
        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # data
    data_root="./data",
    batch_size=256,
    num_workers=4,
    crop_size=32,
    # model
    patch_size=4,
    embed_dim=192,
    depth=6,
    num_heads=3,
    pred_embed_dim=96,
    pred_depth=4,
    pred_num_heads=3,
    # mask
    enc_mask_scale=[0.85, 1.0],
    pred_mask_scale=[0.15, 0.25],
    aspect_ratio=[0.75, 1.5],
    nenc=1,
    npred=4,
    min_keep=4,
    allow_overlap=False,
    # optimization
    epochs=200,
    lr=1e-3,
    start_lr=1e-4,
    final_lr=1e-6,
    warmup=10,
    weight_decay=0.05,
    final_weight_decay=0.05,
    ema_start=0.996,
    ema_end=1.0,
    use_amp=False,
    # logging / checkpointing
    log_freq=50,
    checkpoint_freq=10,
    output_dir="./output/cifar10_jepa",
    seed=42,
)


def load_config(args):
    """Merge YAML config (if any) with CLI args and defaults."""
    cfg = dict(DEFAULTS)
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        if yaml_cfg:
            cfg.update(yaml_cfg)
    # CLI overrides (only if explicitly provided)
    for k, v in vars(args).items():
        if k == "config":
            continue
        if v is not None:
            cfg[k] = v
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 I-JEPA training")
    p.add_argument("--config", type=str, default="config.yaml")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda x: x.lower() in ("true", "1"), default=None)
        elif isinstance(v, list):
            p.add_argument(f"--{k}", type=float, nargs="+", default=None)
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

def make_cifar10_loader(cfg, mask_collator):
    """Create CIFAR-10 training dataloader with mask collation."""
    transform = transforms.Compose([
        transforms.RandomCrop(cfg["crop_size"], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=cfg["data_root"], train=True, download=True, transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=mask_collator,
    )
    return loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args)

    # Reproducibility
    seed = cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")

    # Save config
    with open(os.path.join(cfg["output_dir"], "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # ---- Model ----
    encoder, predictor = make_cifar10_jepa_model(
        img_size=cfg["crop_size"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        pred_embed_dim=cfg["pred_embed_dim"],
        pred_depth=cfg["pred_depth"],
        pred_num_heads=cfg["pred_num_heads"],
    )
    encoder.to(device)
    predictor.to(device)

    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    enc_params = sum(p.numel() for p in encoder.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())
    logger.info(f"Encoder params: {enc_params:,}  Predictor params: {pred_params:,}")

    # ---- Data ----
    mask_collator = MaskCollator(
        input_size=cfg["crop_size"],
        patch_size=cfg["patch_size"],
        enc_mask_scale=cfg["enc_mask_scale"],
        pred_mask_scale=cfg["pred_mask_scale"],
        aspect_ratio=cfg["aspect_ratio"],
        nenc=cfg["nenc"],
        npred=cfg["npred"],
        min_keep=cfg["min_keep"],
        allow_overlap=cfg["allow_overlap"],
    )
    train_loader = make_cifar10_loader(cfg, mask_collator)
    ipe = len(train_loader)
    logger.info(f"Training batches per epoch: {ipe}")

    # ---- Optimizer / Schedulers ----
    param_groups = [
        {"params": [p for n, p in encoder.named_parameters() if "bias" not in n and len(p.shape) != 1]},
        {"params": [p for n, p in predictor.named_parameters() if "bias" not in n and len(p.shape) != 1]},
        {"params": [p for n, p in encoder.named_parameters() if "bias" in n or len(p.shape) == 1],
         "WD_exclude": True, "weight_decay": 0},
        {"params": [p for n, p in predictor.named_parameters() if "bias" in n or len(p.shape) == 1],
         "WD_exclude": True, "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    T_max = int(cfg["epochs"] * ipe)
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(cfg["warmup"] * ipe),
        start_lr=cfg["start_lr"],
        ref_lr=cfg["lr"],
        final_lr=cfg["final_lr"],
        T_max=T_max,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=cfg["weight_decay"],
        final_wd=cfg["final_weight_decay"],
        T_max=T_max,
    )

    scaler = torch.amp.GradScaler("cuda") if cfg["use_amp"] and device.type == "cuda" else None

    # ---- Momentum schedule ----
    ema_start, ema_end = cfg["ema_start"], cfg["ema_end"]
    momentum_schedule = (ema_start + i * (ema_end - ema_start) / T_max for i in range(T_max + 1))

    # ---- Training loop ----
    logger.info("Starting I-JEPA training on CIFAR-10")
    for epoch in range(cfg["epochs"]):
        encoder.train()
        predictor.train()
        loss_sum, loss_count = 0.0, 0

        t0 = time.time()
        for itr, (udata, masks_enc, masks_pred) in enumerate(train_loader):
            imgs = udata[0].to(device, non_blocking=True)
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

            _lr = lr_scheduler.step()
            _wd = wd_scheduler.step()

            with torch.amp.autocast("cuda", enabled=cfg["use_amp"] and device.type == "cuda"):
                # Target
                with torch.no_grad():
                    h = target_encoder(imgs)
                    h = F.layer_norm(h, (h.size(-1),))
                    B = h.size(0)
                    h = apply_masks(h, masks_pred)
                    h = repeat_interleave_batch(h, B, repeat=cfg["nenc"])

                # Context → predictor
                z = encoder(imgs, masks_enc)
                z = predictor(z, masks_enc, masks_pred)

                loss = F.smooth_l1_loss(z, h)

            # Backward
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # EMA update
            with torch.no_grad():
                m = next(momentum_schedule)
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

            loss_val = loss.item()
            loss_sum += loss_val
            loss_count += 1

            if itr % cfg["log_freq"] == 0:
                logger.info(
                    f"[ep {epoch+1}/{cfg['epochs']}, itr {itr}/{ipe}] "
                    f"loss={loss_val:.4f}  avg_loss={loss_sum/loss_count:.4f}  "
                    f"lr={_lr:.2e}  wd={_wd:.2e}  ema_m={m:.5f}"
                )

            assert not (np.isnan(loss_val) or np.isinf(loss_val)), "Loss is NaN/Inf!"

        epoch_loss = loss_sum / max(loss_count, 1)
        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch+1} done — avg_loss={epoch_loss:.4f}  time={elapsed:.1f}s")

        # Always save latest checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "config": cfg,
            "loss": epoch_loss,
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
