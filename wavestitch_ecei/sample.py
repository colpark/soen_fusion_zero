"""
Sample from the trained WaveStitch-style ECEi model.
Condition channels (160=class_id, 161=t_disrupt_cond) are fixed; denoise only signal channels.

  python sample.py --checkpoint checkpoints_wavestitch_ecei/ckpt_epoch_100.pt --num-samples 4 --class-id 1 --t-disrupt 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch

from training_utils import fetchModel, fetchDiffusionConfig

SIGNAL_CHANNELS = 160
COND_CHANNELS = 2
IN_CHANNELS = 162
OUT_CHANNELS = 160


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--T", type=int, default=7512, help="Time length (decimated)")
    p.add_argument("--class-id", type=int, default=0, choices=[0, 1])
    p.add_argument("--t-disrupt", type=float, default=0.0, help="t_disrupt_cond in [0,1]")
    p.add_argument("--output-dir", type=str, default="./samples_wavestitch_ecei")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    T = ckpt.get("T", args.T)

    # Build model from config
    class Args:
        pass
    a = Args()
    for k, v in config.items():
        setattr(a, k, v)
    a.timesteps = config.get("timesteps", 200)
    a.beta_0 = config.get("beta_0", 0.0001)
    a.beta_T = config.get("beta_T", 0.02)
    a.num_res_layers = config.get("num_res_layers", 4)
    a.res_channels = config.get("res_channels", 64)
    a.skip_channels = config.get("skip_channels", 64)
    a.diff_step_embed_in = config.get("diff_step_embed_in", 32)
    a.diff_step_embed_mid = config.get("diff_step_embed_mid", 64)
    a.diff_step_embed_out = config.get("diff_step_embed_out", 64)
    a.s4_lmax = config.get("s4_lmax", 100)
    a.s4_dstate = config.get("s4_dstate", 64)
    a.s4_dropout = config.get("s4_dropout", 0.0)
    a.s4_bidirectional = config.get("s4_bidirectional", True)
    a.s4_layernorm = config.get("s4_layernorm", True)

    model = fetchModel(IN_CHANNELS, OUT_CHANNELS, a).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion_config = fetchDiffusionConfig(a)
    for k, v in diffusion_config.items():
        if isinstance(v, torch.Tensor):
            diffusion_config[k] = v.to(device)

    B = args.num_samples
    # Model expects (B, T, 162). Start from noise; fix columns 160,161 to condition.
    x = torch.randn(B, T, IN_CHANNELS, device=device)
    x[:, :, 160] = float(args.class_id)
    x[:, :, 161] = float(args.t_disrupt)

    alpha_bars = diffusion_config["alpha_bars"]
    alphas = diffusion_config["alphas"]
    betas = diffusion_config["betas"]
    T_steps = diffusion_config["T"]

    with torch.no_grad():
        for step in range(T_steps - 1, -1, -1):
            times = torch.full((B, 1), step, device=device, dtype=torch.long)
            epsilon_pred = model(x, times)
            # epsilon_pred (B, 160, T); signal in x is (B, T, 160)
            alpha_bar_t = alpha_bars[step].to(device)
            alpha_t = alphas[step].to(device)
            beta_t = betas[step].to(device)
            diff_coeff = beta_t / torch.sqrt(1 - alpha_bar_t + 1e-8)
            signal_ctx = x[:, :, :OUT_CHANNELS].permute(0, 2, 1)
            x_prev_signal = (signal_ctx - diff_coeff * epsilon_pred) / torch.sqrt(alpha_t + 1e-8)
            if step > 0:
                alpha_bar_prev = alpha_bars[step - 1].to(device)
                var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t + 1e-8)
                var = torch.clamp(var, min=1e-20)
                x_prev_signal = x_prev_signal + torch.sqrt(var) * torch.randn_like(x_prev_signal, device=device)
            x[:, :, :OUT_CHANNELS] = x_prev_signal.permute(0, 2, 1)

    samples = x[:, :, :SIGNAL_CHANNELS].permute(0, 2, 1).cpu().numpy()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"samples_class{args.class_id}_td{args.t_disrupt:.2f}.npy", samples)
    print(f"Saved {samples.shape} to {out_dir}")
    return samples


if __name__ == "__main__":
    main()
