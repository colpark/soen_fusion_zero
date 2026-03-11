"""DDPM forward (noising) and reverse (sampling) schedule."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class DDPMScheduler:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device: torch.device) -> "DDPMScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        sqrt_alpha = self.sqrt_alphas_cumprod.to(x_start.device)[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def p_loss(self, model: nn.Module, x_start: torch.Tensor, t: torch.Tensor, cond: dict, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise = model(x_noisy, t, **cond)
        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: int,
        cond: dict,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        device = x.device
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_batch, **cond)
        beta_t = self.betas.to(device)[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas.to(device)[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)[t]
        x_prev = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_alpha_cumprod * pred_noise)
        if t > 0:
            posterior_variance_t = self.posterior_variance.to(device)[t]
            x_prev = x_prev + torch.sqrt(posterior_variance_t) * torch.randn_like(x, device=device)
        if clip_denoised:
            x_prev = torch.clamp(x_prev, -1.0, 1.0)
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        cond: dict,
        device: torch.device,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t, cond, clip_denoised=clip_denoised)
        return x
