"""
Evaluation: validation loss, simple stats (mean/std) comparison, optional sample visualizations.
Run from repo root: python diffusion/evaluate.py --checkpoint diffusion/checkpoints/ckpt_epoch_100.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion.data.dataset import DecimatedEceiMmapDataset
from diffusion.models import DDPMScheduler, UNet2DAdaLN


def compute_stats(x: torch.Tensor) -> dict:
    """Per-channel and global mean/std for (B, C, H, W)."""
    x = x.flatten(2)
    mean = x.mean(dim=(0, 2))
    std = x.std(dim=(0, 2)) + 1e-8
    return {"mean": mean.cpu(), "std": std.cpu(), "global_mean": x.mean().item(), "global_std": x.std().item()}


def _save_eval_viz(samples: np.ndarray, out_path: Path, title: str, n_show: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_show = min(n_show, samples.shape[0])
    if n_show <= 0:
        return
    fig, axes = plt.subplots(2, (n_show + 1) // 2, figsize=(4 * ((n_show + 1) // 2), 5))
    axes = np.atleast_2d(axes)
    vmin, vmax = float(np.percentile(samples, 2)), float(np.percentile(samples, 98))
    for k in range(n_show):
        i, j = k // axes.shape[1], k % axes.shape[1]
        x = samples[k].squeeze()
        if x.ndim == 3:
            x = x[0]
        axes[i, j].imshow(x, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i, j].set_title(f"{title} #{k + 1}")
    for k in range(n_show, axes.size):
        axes.flat[k].set_visible(False)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prebuilt-mmap-dir", type=str, default="./subseqs_original_mmap")
    p.add_argument("--decimate-factor", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-timesteps", type=int, default=1000)
    p.add_argument("--num-val-batches", type=int, default=50)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--save-viz-dir", type=str, default=None, help="Save generated sample PNGs here")
    p.add_argument("--verbose", action="store_true", help="Print progress and details")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    base_channels = config.get("base_channels", 64)
    epoch = ckpt.get("epoch", "?")

    if args.verbose:
        print("=== Diffusion evaluation ===")
        print(f"  Checkpoint: {args.checkpoint} (epoch {epoch})")
        print(f"  Val batches: {args.num_val_batches}, device: {device}")
        print("===========================")

    model = UNet2DAdaLN(
        in_channels=1, out_channels=1, base_channels=base_channels,
        channel_mults=(1, 2, 4, 8), num_classes=2, time_embed_dim=128, cond_embed_dim=128,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scheduler = DDPMScheduler(num_timesteps=args.num_timesteps).to(device)

    val_ds = DecimatedEceiMmapDataset(args.prebuilt_mmap_dir, decimate_factor=args.decimate_factor, split="val")
    if len(val_ds) == 0:
        val_ds = DecimatedEceiMmapDataset(args.prebuilt_mmap_dir, decimate_factor=args.decimate_factor, split="test")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if args.verbose:
        print(f"  Validation set: {len(val_ds)} samples")

    # Validation loss (few batches)
    val_loss = 0.0
    n_val = 0
    for i, batch in enumerate(val_loader):
        if i >= args.num_val_batches:
            break
        x, class_id, t_disrupt = batch
        x = x.view(x.shape[0], -1, x.shape[-1]).unsqueeze(1).to(device)
        x = (x - x.mean()) / (x.std() + 1e-5)
        x = torch.clamp(x, -3.0, 3.0) / 3.0
        class_id = class_id.to(device)
        t_disrupt = t_disrupt.to(device).float()
        B = x.shape[0]
        t = torch.randint(0, args.num_timesteps, (B,), device=device, dtype=torch.long)
        with torch.no_grad():
            loss = scheduler.p_loss(model, x, t, {"class_id": class_id, "t_disrupt": t_disrupt})
        val_loss += loss.item()
        n_val += 1
        if args.verbose and (i + 1) % 10 == 0:
            print(f"  Val batch {i + 1}/{min(args.num_val_batches, len(val_loader))}  loss={loss.item():.4e}")
    val_loss /= max(n_val, 1)
    if args.verbose:
        print(f"  Validation loss (avg over {n_val} batches): {val_loss:.4e}")

    # Real data stats
    real_stats_clear, real_stats_disrupt = [], []
    gen_stats_clear, gen_stats_disrupt = [], []
    val_iter = iter(val_loader)
    for _ in range(max(1, args.num_samples // (args.batch_size * 2))):
        try:
            x, class_id, t_disrupt = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            x, class_id, t_disrupt = next(val_iter)
        x = x.view(x.shape[0], -1, x.shape[-1]).unsqueeze(1).to(device)
        x = (x - x.mean()) / (x.std() + 1e-5)
        x = torch.clamp(x, -3.0, 3.0) / 3.0
        mask_clear = class_id == 0
        mask_disrupt = class_id == 1
        if mask_clear.any():
            real_stats_clear.append(compute_stats(x[mask_clear]))
        if mask_disrupt.any():
            real_stats_disrupt.append(compute_stats(x[mask_disrupt]))

    if args.verbose:
        print("  Generating samples (clear and disrupt) for stats...")
    with torch.no_grad():
        for class_id_val, t_d in [(0, 0.0), (1, 0.5)]:
            class_id = torch.full((args.batch_size,), class_id_val, device=device, dtype=torch.long)
            t_disrupt = torch.full((args.batch_size,), t_d, device=device, dtype=torch.float32)
            cond = {"class_id": class_id, "t_disrupt": t_disrupt}
            samples = scheduler.sample(model, (args.batch_size, 1, 160, 7512), cond, device, clip_denoised=True)
            if class_id_val == 0:
                gen_stats_clear.append(compute_stats(samples))
            else:
                gen_stats_disrupt.append(compute_stats(samples))

    def agg_stats(stats_list):
        if not stats_list:
            return {}
        g_mean = sum(s["global_mean"] for s in stats_list) / len(stats_list)
        g_std = sum(s["global_std"] for s in stats_list) / len(stats_list)
        return {"global_mean": g_mean, "global_std": g_std}

    metrics = {
        "val_loss": val_loss,
        "real_clear": agg_stats(real_stats_clear),
        "real_disrupt": agg_stats(real_stats_disrupt),
        "gen_clear": agg_stats(gen_stats_clear),
        "gen_disrupt": agg_stats(gen_stats_disrupt),
    }

    print("--- Results ---")
    print("Validation loss:", val_loss)
    print("Real clear global mean/std:", metrics["real_clear"])
    print("Real disrupt global mean/std:", metrics["real_disrupt"])
    print("Gen clear global mean/std:", metrics["gen_clear"])
    print("Gen disrupt global mean/std:", metrics["gen_disrupt"])

    if args.save_viz_dir:
        viz_dir = Path(args.save_viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        if args.verbose:
            print(f"  Saving sample visualizations to {viz_dir}")
        with torch.no_grad():
            for class_id_val, t_d, name in [(0, 0.0, "clear"), (1, 0.5, "disrupt")]:
                cid = torch.full((4,), class_id_val, device=device, dtype=torch.long)
                td = torch.full((4,), t_d, device=device, dtype=torch.float32)
                cond = {"class_id": cid, "t_disrupt": td}
                samples = scheduler.sample(model, (4, 1, 160, 7512), cond, device, clip_denoised=True)
                _save_eval_viz(
                    samples.cpu().numpy(),
                    viz_dir / f"eval_gen_{name}.png",
                    f"Generated ({name})",
                    n_show=4,
                )
        print(f"Saved visualizations to {args.save_viz_dir}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Wrote", args.output_json)


if __name__ == "__main__":
    main()
