"""
2D CNN U-Net for diffusion with AdaLN conditioning.
Conditioning: class_id (0=clear, 1=disruption) + t_disrupt_cond (0-1, time of t_disrupt-300ms).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AdaLNModulation(nn.Module):
    """LayerNorm then adaptive scale/shift from conditioning (AdaLN)."""

    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        # GroupNorm has no elementwise_affine; use default. AdaLN applies extra scale/shift on top.
        self.norm = nn.GroupNorm(min(8, num_features), num_features)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_features * 2),
        )
        self.num_features = num_features

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        scale, shift = self.mlp(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        x = self.norm(x) * (1 + scale) + shift
        return x


class ResBlockAdaLN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.adaln1 = AdaLNModulation(cond_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.adaln2 = AdaLNModulation(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.adaln1(h, cond)
        h = torch.nn.functional.silu(h)
        h = self.conv2(h)
        h = self.adaln2(h, cond)
        return torch.nn.functional.silu(h) + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.res = ResBlockAdaLN(in_ch, out_ch, cond_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.res(x, cond)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlockAdaLN(in_ch + out_ch, out_ch, cond_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, cond)


class UNet2DAdaLN(nn.Module):
    """
    2D U-Net for (1, H, W) with AdaLN conditioning.
    cond: (class_embed, t_disrupt_embed) fused; also timestep t for diffusion.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        num_classes: int = 2,
        time_embed_dim: int = 128,
        cond_embed_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        chs = [base_channels * m for m in channel_mults]
        cond_dim = time_embed_dim + cond_embed_dim

        # Time embedding (diffusion timestep)
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # Class + t_disrupt conditioning
        self.class_embed = nn.Embedding(num_classes, cond_embed_dim // 2)
        self.t_disrupt_embed = nn.Sequential(
            nn.Linear(1, cond_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(cond_embed_dim // 2, cond_embed_dim // 2),
        )
        self.cond_proj = nn.Linear(cond_embed_dim, cond_embed_dim)

        self.conv_in = nn.Conv2d(in_channels, chs[0], 3, padding=1)
        self.down = nn.ModuleList([
            DownBlock(chs[0], chs[0], cond_dim),
            DownBlock(chs[0], chs[1], cond_dim),
            DownBlock(chs[1], chs[2], cond_dim),
            DownBlock(chs[2], chs[3], cond_dim),
        ])
        self.mid = ResBlockAdaLN(chs[3], chs[3], cond_dim)
        self.up = nn.ModuleList([
            UpBlock(chs[3], chs[2], cond_dim),
            UpBlock(chs[2], chs[1], cond_dim),
            UpBlock(chs[1], chs[0], cond_dim),
            UpBlock(chs[0], chs[0], cond_dim),
        ])
        self.norm_out = nn.GroupNorm(8, chs[0])
        self.conv_out = nn.Conv2d(chs[0], out_channels, 3, padding=1)

        self.time_embed_dim = time_embed_dim
        self.cond_embed_dim = cond_embed_dim

    def _embed_condition(self, class_id: torch.Tensor, t_disrupt: torch.Tensor) -> torch.Tensor:
        # class_id (B,), t_disrupt (B,) in [0,1]
        c = self.class_embed(class_id)
        t = self.t_disrupt_embed(t_disrupt.unsqueeze(-1))
        cond = torch.cat([c, t], dim=-1)
        return self.cond_proj(cond)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_id: torch.Tensor,
        t_disrupt: torch.Tensor,
    ) -> torch.Tensor:
        B = x.shape[0]
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = self.time_embed(t_emb)
        cond_emb = self._embed_condition(class_id, t_disrupt)
        cond = torch.cat([t_emb, cond_emb], dim=-1)

        x = self.conv_in(x)
        skips = []
        for block in self.down:
            x = block(x, cond)
            skips.append(x)
        x = self.mid(x, cond)
        for block, skip in zip(self.up, reversed(skips)):
            x = block(x, skip, cond)
        x = self.norm_out(x)
        x = torch.nn.functional.silu(x)
        return self.conv_out(x)
