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

import numpy as np
import torch

from diffusion.models import DDPMScheduler, UNet2DAdaLN


def _plot_and_save_samples(samples: np.ndarray, out_path: Path, title: str = "Samples", n_show: int = 8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_show = min(n_show, samples.shape[0])
    if n_show <= 0:
        return
    n_cols = min(4, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    vmin, vmax = float(np.percentile(samples, 2)), float(np.percentile(samples, 98))
    for k in range(n_show):
        i, j = k // n_cols, k % n_cols
        x = samples[k].squeeze()
        if x.ndim == 3:
            x = x[0]
        axes[i, j].imshow(x, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i, j].set_title(f"Sample {k + 1}")
        axes[i, j].set_xlabel("Time")
        axes[i, j].set_ylabel("Channel")
    for k in range(n_show, axes.size):
        axes.flat[k].set_visible(False)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


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
    p.add_argument("--save-viz", action="store_true", default=True, help="Save PNG grid of samples")
    p.add_argument("--no-save-viz", action="store_false", dest="save_viz")
    p.add_argument("--verbose", action="store_true", help="Print config and shapes")
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

    if args.verbose:
        print("=== Diffusion sampling ===")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Config: base_channels={base_channels}, num_timesteps={num_timesteps}")
        print(f"  Output shape: {shape} (B, C, H, W)")
        print(f"  Condition: class_id={args.class_id} (0=clear, 1=disrupt), t_disrupt={args.t_disrupt}")
        print(f"  Device: {device}")
        print("=========================")

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

    if args.verbose:
        print("Sampling (reverse diffusion)...")
    with torch.no_grad():
        samples = scheduler.sample(model, shape, cond, device, clip_denoised=True)
    if args.verbose:
        print(f"  Sampled shape: {samples.shape}, min={samples.min().item():.3f}, max={samples.max().item():.3f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_name = f"samples_class{args.class_id}_td{args.t_disrupt:.2f}.pt"
    torch.save(samples.cpu(), out_dir / pt_name)
    print(f"Saved {args.num_samples} samples to {out_dir / pt_name}")

    if args.save_viz:
        label = "clear" if args.class_id == 0 else "disrupt"
        title = f"Generated ({label}, t_disrupt={args.t_disrupt:.2f})"
        png_name = f"samples_class{args.class_id}_td{args.t_disrupt:.2f}.png"
        _plot_and_save_samples(samples.cpu().numpy(), out_dir / png_name, title=title, n_show=args.num_samples)
        print(f"Saved visualization to {out_dir / png_name}")


if __name__ == "__main__":
    main()
