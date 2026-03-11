"""
Sample from trained diffusion model with specified class (clear/disruption) and t_disrupt condition.
Run from repo root: python diffusion/sample.py --checkpoint diffusion/checkpoints/ckpt_epoch_100.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from diffusion.models import DDPMScheduler, UNet2DAdaLN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./diffusion/samples")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--class-id", type=int, default=0, help="0=clear, 1=disruption")
    p.add_argument("--t-disrupt", type=float, default=0.0, help="Normalised t_disrupt-300ms in [0,1]; only used if class_id=1")
    p.add_argument("--shape", type=str, default="1,160,7512", help="C,H,W as 1,160,7512")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device)
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    base_channels = config.get("base_channels", 64)
    num_timesteps = config.get("num_timesteps", 1000)
    parts = [int(x) for x in args.shape.split(",")]
    shape = (args.num_samples, *parts)

    model = UNet2DAdaLN(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        channel_mults=(1, 2, 4, 8),
        num_classes=2,
        time_embed_dim=128,
        cond_embed_dim=128,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scheduler = DDPMScheduler(num_timesteps=num_timesteps).to(device)

    class_id = torch.full((args.num_samples,), args.class_id, device=device, dtype=torch.long)
    t_disrupt = torch.full((args.num_samples,), args.t_disrupt, device=device, dtype=torch.float32)
    cond = {"class_id": class_id, "t_disrupt": t_disrupt}

    with torch.no_grad():
        samples = scheduler.sample(model, shape, cond, device, clip_denoised=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples.cpu(), out_dir / f"samples_class{args.class_id}_td{args.t_disrupt:.2f}.pt")
    print(f"Saved {args.num_samples} samples to {out_dir}")


if __name__ == "__main__":
    main()
