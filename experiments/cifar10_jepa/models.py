"""
CIFAR-10 I-JEPA models.

Adapted from Meta's I-JEPA (https://github.com/facebookresearch/ijepa)
for 32×32 CIFAR-10 images with patch_size=4 → 8×8 = 64 patch tokens.

Contains:
  - VisionTransformer (encoder)
  - VisionTransformerPredictor (predictor)
  - MaskCollator (block mask generator)
  - Utility functions (positional embeddings, apply_masks, etc.)
"""

import math
from functools import partial
from multiprocessing import Value

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional embedding utilities
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---------------------------------------------------------------------------
# Weight init
# ---------------------------------------------------------------------------

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lo = norm_cdf((a - mean) / std)
        hi = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * hi - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def apply_masks(x, masks):
    """
    Gather patches specified by *masks* from *x*.

    Args:
        x:     (B, N, D)
        masks: list of tensors, each (B, K) with patch indices

    Returns: (B * len(masks), K, D)
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, B, repeat):
    """Repeat each sub-batch *repeat* times along dim-0."""
    N = len(x) // B
    x = torch.cat(
        [torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)],
        dim=0,
    )
    return x


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.drop_path = nn.Identity()  # simplified; no stochastic depth for POC

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


# ---------------------------------------------------------------------------
# Vision Transformer — encoder
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """ViT encoder for I-JEPA on CIFAR-10."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                 qkv_bias=True, norm_layer=None, init_std=0.02):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches**0.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        """
        Args:
            x:     (B, C, H, W) images
            masks: optional list of index tensors, each (B, K)
        Returns: (B [* len(masks)], K_or_N, D)
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed

        if masks is not None:
            x = apply_masks(x, masks if isinstance(masks, list) else [masks])

        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Vision Transformer — predictor
# ---------------------------------------------------------------------------

class VisionTransformerPredictor(nn.Module):
    """
    Predictor for I-JEPA.

    Takes context encoder tokens + positional information for target positions
    and predicts target encoder embeddings at those positions.
    """

    def __init__(self, num_patches=64, embed_dim=192, predictor_embed_dim=96,
                 depth=4, num_heads=3, mlp_ratio=4.0, qkv_bias=True,
                 norm_layer=None, init_std=0.02):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )
        pos = get_2d_sincos_pos_embed(predictor_embed_dim, int(num_patches**0.5), cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        self.predictor_blocks = nn.ModuleList([
            Block(predictor_embed_dim, num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.init_std = init_std
        trunc_normal_(self.mask_token, std=init_std)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, masks_x, masks):
        """
        Args:
            x:       (B_ctx, K_ctx, D_enc) — context encoder outputs
            masks_x: list of nenc tensors, each (B, K_ctx) — context indices
            masks:   list of npred tensors, each (B, K_pred) — target indices
        Returns: (B_ctx * npred, K_pred, D_enc) — predicted embeddings
        """
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        B = len(x) // len(masks_x)
        npred = len(masks)

        # Project encoder dim → predictor dim and add context positional embeddings
        x = self.predictor_embed(x)
        x_pos = self.predictor_pos_embed.repeat(B, 1, 1)
        x = x + apply_masks(x_pos, masks_x)

        _, N_ctxt, _ = x.shape

        # Create mask tokens for target positions with positional embeddings
        pos_embs = apply_masks(self.predictor_pos_embed.repeat(B, 1, 1), masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1) + pos_embs

        # Concatenate context tokens with mask tokens
        x = x.repeat(npred, 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # Return only the predictions at mask positions
        x = x[:, N_ctxt:]
        return self.predictor_proj(x)


# ---------------------------------------------------------------------------
# Block mask collator (adapted for CIFAR-10 8×8 patch grid)
# ---------------------------------------------------------------------------

class MaskCollator:
    """
    Generates block masks for I-JEPA training.

    Produces disjoint context (encoder) and target (predictor) masks at the
    patch-grid level. Designed for an 8×8 grid (CIFAR-10, patch_size=4).
    """

    def __init__(
        self,
        input_size=32,
        patch_size=4,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.25),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False,
    ):
        self.height = input_size // patch_size
        self.width = input_size // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value("i", -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            return i.value

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        h = min(h, self.height - 1)
        w = min(w, self.width - 1)
        return (max(h, 1), max(w, 1))

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            top = torch.randint(0, max(self.height - h, 1), (1,))
            left = torch.randint(0, max(self.width - w, 1), (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1
            if acceptable_regions is not None:
                N = max(int(len(acceptable_regions) - tries), 0)
                for k in range(N):
                    mask *= acceptable_regions[k]
            mask = torch.nonzero(mask.flatten())
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
        mask = mask.squeeze()

        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        return mask, mask_complement

    def __call__(self, batch):
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(g, self.pred_mask_scale, self.aspect_ratio)
        e_size = self._sample_block_size(g, self.enc_mask_scale, (1.0, 1.0))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = None if self.allow_overlap else masks_C
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # Truncate to min_keep and collate
        collated_masks_pred = [
            [cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred
        ]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        collated_masks_enc = [
            [cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc
        ]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_cifar10_jepa_model(
    img_size=32,
    patch_size=4,
    embed_dim=192,
    depth=6,
    num_heads=3,
    pred_embed_dim=96,
    pred_depth=4,
    pred_num_heads=3,
):
    """Create encoder + predictor for CIFAR-10 I-JEPA."""
    encoder = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )
    num_patches = encoder.patch_embed.num_patches
    predictor = VisionTransformerPredictor(
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=pred_num_heads,
    )
    return encoder, predictor
