"""
Train WaveStitch-style diffusion on ECEi (160, T) with two conditions:
- Condition 1: binary class (0=clear, 1=disruption).
- Condition 2: t_disrupt_cond = normalised [0,1] timestep when disruption starts; 0 for clear.

Follows WaveStitch exactly: condition channels are not noised (masked as "hierarchical");
model predicts noise only for the 160 signal channels. Run from wavestitch_ecei/ so that
utils and TSImputers resolve.

  python train.py --prebuilt-mmap-dir ../subseqs_original_mmap --decimate-factor 10 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run from wavestitch_ecei so that utils and TSImputers resolve
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import DecimatedEceiMmapDataset
from training_utils import fetchModel, fetchDiffusionConfig


# Signal channels 0..159; condition channels 160 (class), 161 (t_disrupt)
SIGNAL_CHANNELS = 160
COND_CHANNELS = 2
IN_CHANNELS = SIGNAL_CHANNELS + COND_CHANNELS
OUT_CHANNELS = SIGNAL_CHANNELS


def _save_sample_visualization(samples: np.ndarray, out_path: Path, n_show: int = 4, title_prefix: str = ""):
    """Save a grid of 2D ECEi visualizations (channel x time). samples: (N, 160, T)."""
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


def _generate_samples(model, diffusion_config, device, T: int, n_samples: int, class_id: int, t_disrupt: float):
    """Run reverse diffusion; return (n_samples, 160, T) numpy array."""
    model.eval()
    alpha_bars = diffusion_config["alpha_bars"].to(device)
    alphas = diffusion_config["alphas"].to(device)
    betas = diffusion_config["betas"].to(device)
    T_steps = diffusion_config["T"]
    B = n_samples
    x = torch.randn(B, T, IN_CHANNELS, device=device)
    x[:, :, 160] = float(class_id)
    x[:, :, 161] = float(t_disrupt)
    with torch.no_grad():
        for step in range(T_steps - 1, -1, -1):
            times = torch.full((B, 1), step, device=device, dtype=torch.long)
            epsilon_pred = model(x, times)
            alpha_bar_t = alpha_bars[step]
            alpha_t = alphas[step]
            beta_t = betas[step]
            diff_coeff = beta_t / torch.sqrt(1 - alpha_bar_t + 1e-8)
            signal_ctx = x[:, :, :OUT_CHANNELS].permute(0, 2, 1)
            x_prev_signal = (signal_ctx - diff_coeff * epsilon_pred) / torch.sqrt(alpha_t + 1e-8)
            if step > 0:
                alpha_bar_prev = alpha_bars[step - 1]
                var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t + 1e-8)
                var = torch.clamp(var, min=1e-20)
                x_prev_signal = x_prev_signal + torch.sqrt(var) * torch.randn_like(x_prev_signal, device=device)
            x[:, :, :OUT_CHANNELS] = x_prev_signal.permute(0, 2, 1)
    return x[:, :, :SIGNAL_CHANNELS].permute(0, 2, 1).cpu().numpy()


def build_batch_btc(x_list, class_ids, t_disrupts, device):
    """
    WaveStitch expects (B, T, C): batch, time, channels.
    x_list: list of (C, T) tensors.
    Returns: (B, T, 162) on device; dim 2: 0..159 = signal, 160 = class, 161 = t_disrupt.
    """
    B = len(x_list)
    out = []
    for i in range(B):
        x = x_list[i]
        x_flat = x.reshape(-1, x.shape[-1])  # (160, T)
        T = x_flat.shape[1]
        # (160, T) -> (T, 160); then append cond columns -> (T, 162)
        sig = x_flat.t()  # (T, 160)
        class_col = torch.full((T, 1), float(class_ids[i].item()), dtype=torch.float32, device=x.device)
        t_col = torch.full((T, 1), float(t_disrupts[i].item()), dtype=torch.float32, device=x.device)
        out.append(torch.cat([sig, class_col, t_col], dim=1))
    return torch.stack(out, dim=0).to(device)


def main():
    p = argparse.ArgumentParser(description="WaveStitch-style diffusion on ECEi")
    p.add_argument("--prebuilt-mmap-dir", type=str, default="./subseqs_original_mmap")
    p.add_argument("--decimate-factor", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta_0", type=float, default=0.0001)
    p.add_argument("--beta_T", type=float, default=0.02)
    p.add_argument("--timesteps", "-T", type=int, default=200)
    p.add_argument("--num_res_layers", type=int, default=4)
    p.add_argument("--res_channels", type=int, default=64)
    p.add_argument("--skip_channels", type=int, default=64)
    p.add_argument("--diff_step_embed_in", type=int, default=32)
    p.add_argument("--diff_step_embed_mid", type=int, default=64)
    p.add_argument("--diff_step_embed_out", type=int, default=64)
    p.add_argument("--s4_lmax", type=int, default=100)
    p.add_argument("--s4_dstate", type=int, default=64)
    p.add_argument("--s4_dropout", type=float, default=0.0)
    p.add_argument("--s4_bidirectional", type=bool, default=True)
    p.add_argument("--s4_layernorm", type=bool, default=True)
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints_wavestitch_ecei")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--verbose", action="store_true", help="Print full config, model details, val loss, and sample paths")
    p.add_argument("--log-every", type=int, default=0, help="Log loss every N batches (0=epoch only)")
    p.add_argument("--sample-every", type=int, default=5, help="Generate samples and save viz every N epochs (0=never)")
    p.add_argument("--num-sample-viz", type=int, default=4, help="Number of samples to plot when saving viz")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = ckpt_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DecimatedEceiMmapDataset(
        args.prebuilt_mmap_dir,
        decimate_factor=args.decimate_factor,
        split="train",
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,
    )
    n_batches_per_epoch = len(train_loader)
    n_train = len(train_ds)

    sample_x, sample_c, sample_t = train_ds[0]
    T = sample_x.shape[-1]

    model = fetchModel(IN_CHANNELS, OUT_CHANNELS, args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    diffusion_config = fetchDiffusionConfig(args)
    for k, v in diffusion_config.items():
        if isinstance(v, torch.Tensor):
            diffusion_config[k] = v.to(device)

    # Always show model details at start
    print(f"Model: SSSDS4Imputer in={IN_CHANNELS} out={OUT_CHANNELS} T={T}  params={n_params:,}  device={device}")
    if args.verbose:
        print("=== WaveStitch ECEi training config ===")
        print(json.dumps(vars(args), indent=2))
        print(f"  Dataset: {n_train} samples, {n_batches_per_epoch} batches/epoch")
        print(f"  Sample shape: {tuple(sample_x.shape)} -> (B, T, 162) with T={T}")
        print(f"  Device: {device}")
        print("======================================")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Condition channels 160, 161 are not noised. Layout (B, T, 162).
    conditional_mask = torch.ones(1, 1, IN_CHANNELS, device=device)
    conditional_mask[:, :, :SIGNAL_CHANNELS] = 0

    # Validation dataset for evaluation when verbose
    val_loader = None
    if args.verbose:
        try:
            val_ds = DecimatedEceiMmapDataset(
                args.prebuilt_mmap_dir,
                decimate_factor=args.decimate_factor,
                split="test",
            )
            if len(val_ds) > 0:
                val_loader = DataLoader(
                    val_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=lambda batch: batch,
                )
        except Exception:
            val_loader = None

    def _eval_loss(loader, max_batches=None):
        model.eval()
        total_loss, n_b = 0.0, 0
        with torch.no_grad():
            for batch_tuples in loader:
                if max_batches is not None and n_b >= max_batches:
                    break
                x_list = [b[0] for b in batch_tuples]
                class_ids = torch.tensor([b[1] for b in batch_tuples], dtype=torch.long)
                t_disrupts = torch.tensor([b[2] for b in batch_tuples], dtype=torch.float32)
                batch = build_batch_btc(x_list, class_ids, t_disrupts, device)
                B, L, C = batch.shape
                sig = batch[:, :, :SIGNAL_CHANNELS]
                sig = (sig - sig.mean()) / (sig.std() + 1e-5)
                batch = torch.cat([sig, batch[:, :, SIGNAL_CHANNELS:]], dim=2)
                timesteps = torch.randint(diffusion_config["T"], size=(B,), device=device)
                sigmas = torch.randn_like(batch, device=device)
                alpha_bars = diffusion_config["alpha_bars"].to(device)
                coeff_1 = torch.sqrt(alpha_bars[timesteps]).reshape(B, 1, 1)
                coeff_2 = torch.sqrt(1 - alpha_bars[timesteps]).reshape(B, 1, 1)
                mask = conditional_mask.expand(B, L, -1)
                batch_noised = (1 - mask) * (coeff_1 * batch + coeff_2 * sigmas) + mask * batch
                sigmas_predicted = model(batch_noised, timesteps.reshape(-1, 1))
                loss = criterion(sigmas_predicted, sigmas[:, :, :OUT_CHANNELS].permute(0, 2, 1))
                total_loss += loss.item()
                n_b += 1
        model.train()
        return total_loss / max(n_b, 1) if n_b else 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_idx, batch_tuples in enumerate(train_loader):
            x_list = [b[0] for b in batch_tuples]
            class_ids = torch.tensor([b[1] for b in batch_tuples], dtype=torch.long)
            t_disrupts = torch.tensor([b[2] for b in batch_tuples], dtype=torch.float32)

            batch = build_batch_btc(x_list, class_ids, t_disrupts, device)
            B, L, C = batch.shape
            assert C == IN_CHANNELS and L == T

            # Normalise signal columns only (0..159)
            sig = batch[:, :, :SIGNAL_CHANNELS]
            sig = (sig - sig.mean()) / (sig.std() + 1e-5)
            batch = torch.cat([sig, batch[:, :, SIGNAL_CHANNELS:]], dim=2)

            timesteps = torch.randint(diffusion_config["T"], size=(B,), device=device)
            sigmas = torch.randn_like(batch, device=device)

            alpha_bars = diffusion_config["alpha_bars"].to(device)
            coeff_1 = torch.sqrt(alpha_bars[timesteps]).reshape(B, 1, 1)
            coeff_2 = torch.sqrt(1 - alpha_bars[timesteps]).reshape(B, 1, 1)
            mask = conditional_mask.expand(B, L, -1)
            batch_noised = (1 - mask) * (coeff_1 * batch + coeff_2 * sigmas) + mask * batch

            timesteps = timesteps.reshape(-1, 1)
            sigmas_predicted = model(batch_noised, timesteps)

            # Model returns (B, 160, T); sigmas for signal are (B, T, 160) -> permute to (B, 160, T)
            loss = criterion(sigmas_predicted, sigmas[:, :, :OUT_CHANNELS].permute(0, 2, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if args.log_every and (batch_idx + 1) % args.log_every == 0:
                print(f"  Epoch {epoch + 1} batch {batch_idx + 1}/{n_batches_per_epoch}  loss={loss.item():.4e}")

        avg = total_loss / max(n_batches, 1)
        val_str = ""
        if args.verbose and val_loader is not None:
            val_loss = _eval_loss(val_loader, max_batches=20)
            val_str = f"  val_loss: {val_loss:.4e}"
        print(f"epoch: {epoch + 1}, loss: {avg:.4e}{val_str}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(args),
                "T": T,
            }, ckpt_dir / f"ckpt_epoch_{epoch + 1}.pt")
            if args.verbose:
                print(f"  Saved checkpoint: ckpt_epoch_{epoch + 1}.pt")

        if args.sample_every and (epoch + 1) % args.sample_every == 0:
            n_viz = min(args.num_sample_viz, 8)
            for label, cid, td in [("clear", 0, 0.0), ("disrupt", 1, 0.5)]:
                arr = _generate_samples(model, diffusion_config, device, T, n_viz, cid, td)
                out_path = samples_dir / f"epoch{epoch + 1:04d}_{label}.png"
                _save_sample_visualization(arr, out_path, n_show=n_viz, title_prefix=label.capitalize())
                if args.verbose:
                    print(f"  Saved sample viz: {out_path}")
            model.train()

    print("Done.")


if __name__ == "__main__":
    main()
