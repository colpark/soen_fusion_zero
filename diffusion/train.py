"""
Train diffusion model (2D AdaLN UNet) on decimated ECEi with clear/disruption + t_disrupt-300ms conditioning.
Run from repo root: python diffusion/train.py --prebuilt-mmap-dir ./subseqs_original_mmap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion.data.dataset import DecimatedEceiMmapDataset
from diffusion.models import DDPMScheduler, UNet2DAdaLN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prebuilt-mmap-dir", type=str, default="./subseqs_original_mmap")
    p.add_argument("--decimate-factor", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-timesteps", type=int, default=1000)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--checkpoint-dir", type=str, default="./diffusion/checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_ds = DecimatedEceiMmapDataset(args.prebuilt_mmap_dir, decimate_factor=args.decimate_factor, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # (C, T) -> (1, C, T) as 2D image; or (1, 160, 7512)
    sample_x, _, _ = train_ds[0]
    C, T = sample_x.shape
    in_channels = 1
    model = UNet2DAdaLN(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4, 8),
        num_classes=2,
        time_embed_dim=128,
        cond_embed_dim=128,
    ).to(device)
    scheduler = DDPMScheduler(num_timesteps=args.num_timesteps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x, class_id, t_disrupt = batch
            # x (B, C, T) -> (B, 1, C, T) for 2D conv
            x = x.unsqueeze(1).to(device)
            x = (x - x.mean()) / (x.std() + 1e-5)
            x = torch.clamp(x, -3.0, 3.0) / 3.0
            class_id = class_id.to(device)
            t_disrupt = t_disrupt.to(device).float()
            B = x.shape[0]
            t = torch.randint(0, args.num_timesteps, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x, device=device)
            x_noisy = scheduler.q_sample(x, t, noise)
            pred_noise = model(x_noisy, t, class_id, t_disrupt)
            loss = nn.functional.mse_loss(pred_noise, noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
        avg = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg:.4e}")

        if (epoch + 1) % 10 == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": vars(args),
            }
            torch.save(ckpt, Path(args.checkpoint_dir) / f"ckpt_epoch_{epoch + 1}.pt")
    print("Done.")


if __name__ == "__main__":
    main()
