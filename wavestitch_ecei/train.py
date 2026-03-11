"""
Train WaveStitch-style diffusion on ECEi (160, T) with two conditions:
- Condition 1: binary class (0=clear, 1=disruption).
- Condition 2: t_disrupt_cond = normalised [0,1] timestep when disruption starts; 0 for clear.

Follows WaveStitch exactly: condition channels are not noised (masked as "hierarchical");
model predicts noise only for the 160 signal channels. Run from wavestitch_ecei/ so that
utils and TSImputers resolve.

  python train.py --prebuilt-mmap-dir ../subseqs_original_mmap --decimate-factor 10
"""
from __future__ import annotations

import argparse
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


def build_batch_162(x_list, class_ids, t_disrupts, device):
    """
    x_list: list of (C, T) tensors, C can be 160 or 20*8.
    class_ids: (B,) long or int.
    t_disrupts: (B,) float.
    Returns: (B, 162, T) on device; channels 0..159 = flattened signal, 160 = class, 161 = t_disrupt.
    """
    B = len(x_list)
    out = []
    for i in range(B):
        x = x_list[i]
        x_flat = x.reshape(-1, x.shape[-1])  # (160, T) or (160, T)
        T = x_flat.shape[1]
        class_row = torch.full((1, T), float(class_ids[i].item()), dtype=torch.float32, device=x.device)
        t_row = torch.full((1, T), float(t_disrupts[i].item()), dtype=torch.float32, device=x.device)
        out.append(torch.cat([x_flat, class_row, t_row], dim=0))
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
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    sample_x, sample_c, sample_t = train_ds[0]
    T = sample_x.shape[-1]
    if args.verbose:
        print(f"Dataset: {len(train_ds)} samples, signal shape {tuple(sample_x.shape)} -> 160 x {T}")
        print(f"Conditions: class_id={sample_c}, t_disrupt_cond={sample_t:.4f}")

    model = fetchModel(IN_CHANNELS, OUT_CHANNELS, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)
    for k, v in diffusion_config.items():
        if isinstance(v, torch.Tensor):
            diffusion_config[k] = v.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Condition channels (160, 161) are not noised — same as WaveStitch hierarchical columns
    conditional_mask = torch.ones(1, IN_CHANNELS, 1, device=device)
    conditional_mask[:, :SIGNAL_CHANNELS, :] = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_tuples in train_loader:
            x_list = [b[0] for b in batch_tuples]
            class_ids = torch.tensor([b[1] for b in batch_tuples], dtype=torch.long)
            t_disrupts = torch.tensor([b[2] for b in batch_tuples], dtype=torch.float32)

            batch = build_batch_162(x_list, class_ids, t_disrupts, device)
            B, C, L = batch.shape
            assert C == IN_CHANNELS and L == T

            # Normalise signal channels only (0..159)
            sig = batch[:, :SIGNAL_CHANNELS, :]
            sig = (sig - sig.mean()) / (sig.std() + 1e-5)
            batch = torch.cat([sig, batch[:, SIGNAL_CHANNELS:, :]], dim=1)

            timesteps = torch.randint(diffusion_config["T"], size=(B,), device=device)
            sigmas = torch.randn_like(batch, device=device)

            alpha_bars = diffusion_config["alpha_bars"].to(device)
            coeff_1 = torch.sqrt(alpha_bars[timesteps]).reshape(B, 1, 1)
            coeff_2 = torch.sqrt(1 - alpha_bars[timesteps]).reshape(B, 1, 1)
            mask = conditional_mask.expand(B, -1, L)
            batch_noised = (1 - mask) * (coeff_1 * batch + coeff_2 * sigmas) + mask * batch

            timesteps = timesteps.reshape(-1, 1)
            sigmas_predicted = model(batch_noised, timesteps)

            # Loss only on predicted noise for signal channels (WaveStitch: non_hier_cols)
            loss = criterion(sigmas_predicted, sigmas[:, :OUT_CHANNELS, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"epoch: {epoch + 1}, loss: {avg:.4e}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(args),
                "T": T,
            }, ckpt_dir / f"ckpt_epoch_{epoch + 1}.pt")
            if args.verbose:
                print(f"  Saved {ckpt_dir / f'ckpt_epoch_{epoch + 1}.pt'}")

    print("Done.")


if __name__ == "__main__":
    main()
