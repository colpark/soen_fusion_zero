"""
Train diffusion model (2D AdaLN UNet) on decimated ECEi with clear/disruption + t_disrupt-300ms conditioning.

Single-GPU:
  python diffusion/train.py --prebuilt-mmap-dir ./subseqs_original_mmap

Distributed (multi-GPU):
  torchrun --nproc_per_node=N diffusion/train.py --prebuilt-mmap-dir ./subseqs_original_mmap [OPTIONS]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusion.data.dataset import DecimatedEceiMmapDataset
from diffusion.models import DDPMScheduler, UNet2DAdaLN


def _save_sample_visualization(samples: np.ndarray, out_path: Path, n_show: int = 4, title_prefix: str = ""):
    """Save a grid of 2D ECEi visualizations (channel x time) for generated samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_show = min(n_show, samples.shape[0])
    if n_show <= 0:
        return
    fig, axes = plt.subplots(2, (n_show + 1) // 2, figsize=(4 * ((n_show + 1) // 2), 6))
    axes = np.atleast_2d(axes)
    vmin, vmax = float(np.percentile(samples, 2)), float(np.percentile(samples, 98))
    for k in range(n_show):
        i, j = k // axes.shape[1], k % axes.shape[1]
        x = samples[k].squeeze()
        if x.ndim == 3:
            x = x[0]
        axes[i, j].imshow(x, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i, j].set_title(f"{title_prefix} sample {k + 1}")
        axes[i, j].set_xlabel("Time")
        axes[i, j].set_ylabel("Channel")
    for k in range(n_show, axes.size):
        axes.flat[k].set_visible(False)
    plt.suptitle(title_prefix or "Generated samples")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def _is_distributed() -> bool:
    return os.environ.get("WORLD_SIZE", "1") != "1"


def _setup_ddp() -> tuple[int, int, torch.device]:
    """Initialize process group. Returns (rank, world_size, device)."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    try:
        dist.init_process_group(backend=backend)
    except RuntimeError as e:
        if "NCCL" in str(e) and backend == "nccl":
            dist.init_process_group(backend="gloo")
        else:
            raise
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, device


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
    p.add_argument("--log-every", type=int, default=0, help="Log loss every N batches (0=epoch only)")
    p.add_argument("--sample-every", type=int, default=10, help="Generate samples and save viz every N epochs (0=never)")
    p.add_argument("--num-sample-viz", type=int, default=4, help="Number of samples to plot when saving viz")
    p.add_argument("--verbose", action="store_true", help="Print config and dataset summary at start")
    p.add_argument("--no-ddp", action="store_true", help="Disable DDP even if WORLD_SIZE>1 (single process)")
    args = p.parse_args()

    use_ddp = _is_distributed() and not args.no_ddp
    if use_ddp:
        rank, world_size, device = _setup_ddp()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_main = rank == 0
    else:
        rank, world_size, local_rank = 0, 1, 0
        is_main = True
        device = torch.device(args.device)

    ckpt_dir = Path(args.checkpoint_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "samples").mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()

    train_ds = DecimatedEceiMmapDataset(args.prebuilt_mmap_dir, decimate_factor=args.decimate_factor, split="train")
    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    n_batches_per_epoch = len(train_loader)
    n_train = len(train_ds)

    sample_x, sample_c, sample_t = train_ds[0]
    T = sample_x.shape[-1]
    in_channels = 1
    if is_main and args.verbose:
        print("=== Diffusion training config ===")
        print(json.dumps(vars(args), indent=2))
        print(f"  Dataset: {n_train} samples, {n_batches_per_epoch} batches/epoch (per rank)")
        if use_ddp:
            print(f"  DDP: world_size={world_size}, effective batch={args.batch_size * world_size}")
        print(f"  Sample shape (raw): {getattr(sample_x, 'shape', '?')} -> (1, 160, {T}) for UNet")
        print(f"  Device: {device}")
        print("=================================")

    model = UNet2DAdaLN(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4, 8),
        num_classes=2,
        time_embed_dim=128,
        cond_embed_dim=128,
    ).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    scheduler = DDPMScheduler(num_timesteps=args.num_timesteps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main and args.verbose:
        print(f"  Model parameters: {n_params:,}")

    # For checkpoint/sampling we need the unwrapped model when using DDP
    model_for_ckpt = model.module if use_ddp else model

    global_step = 0
    for epoch in range(args.epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            x, class_id, t_disrupt = batch
            x = x.view(x.shape[0], -1, x.shape[-1]).unsqueeze(1).to(device)
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
            if is_main and args.log_every and (batch_idx + 1) % args.log_every == 0:
                print(f"  Epoch {epoch + 1} batch {batch_idx + 1}/{n_batches_per_epoch}  loss={loss.item():.4e}")
        avg = epoch_loss / max(n_batches, 1)
        if use_ddp:
            avg_t = torch.tensor([avg], device=device)
            dist.all_reduce(avg_t, op=dist.ReduceOp.SUM)
            avg = (avg_t / world_size).item()
        if is_main:
            print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg:.4e}  (batches={n_batches})")

        if (epoch + 1) % 10 == 0:
            if is_main:
                ckpt = {
                    "epoch": epoch + 1,
                    "model": model_for_ckpt.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": vars(args),
                }
                torch.save(ckpt, ckpt_dir / f"ckpt_epoch_{epoch + 1}.pt")
                if args.verbose:
                    print(f"  Saved checkpoint: ckpt_epoch_{epoch + 1}.pt")
            if use_ddp:
                dist.barrier()

        if args.sample_every and (epoch + 1) % args.sample_every == 0:
            model.eval()
            n_viz = min(args.num_sample_viz, 8)
            shape = (n_viz, 1, 160, T)
            with torch.no_grad():
                for label, cid, td in [("clear", 0, 0.0), ("disrupt", 1, 0.5)]:
                    cid_t = torch.full((n_viz,), cid, device=device, dtype=torch.long)
                    td_t = torch.full((n_viz,), td, device=device, dtype=torch.float32)
                    cond = {"class_id": cid_t, "t_disrupt": td_t}
                    samples = scheduler.sample(model_for_ckpt, shape, cond, device, clip_denoised=True)
                    if is_main:
                        arr = samples.cpu().numpy()
                        samples_dir = ckpt_dir / "samples"
                        out_path = samples_dir / f"epoch{epoch + 1:04d}_{label}.png"
                        _save_sample_visualization(arr, out_path, n_show=n_viz, title_prefix=label.capitalize())
                        if args.verbose:
                            print(f"  Saved sample viz: {out_path}")
            if use_ddp:
                dist.barrier()
            model.train()

    if is_main:
        print("Done.")
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
