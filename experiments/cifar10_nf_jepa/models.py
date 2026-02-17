"""
CIFAR-10 NF+JEPA models (Mamba-GINR-style neural field + JEPA).

Implements:
  - MambaEncoder: Mamba-based set encoder for sparse (coord, value) tokens.
  - FallbackTransformerEncoder: attention-based fallback when mamba-ssm unavailable.
  - CoordPredictor: predicts target encoder embeddings at target coordinates.
  - INRDecoder: optional implicit neural representation for RGB reconstruction.
  - Coordinate utilities: grid creation, sparse sampling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Check for mamba-ssm availability
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def make_coord_grid(H, W, device="cpu", normalize=True):
    """
    Create a 2D coordinate grid.

    Returns: (H*W, 2) tensor with (y, x) coordinates.
    If normalize=True, coordinates are in [0, 1].
    """
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    if normalize:
        ys = ys / (H - 1) if H > 1 else ys
        xs = xs / (W - 1) if W > 1 else xs
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
    return coords  # (H*W, 2)


def image_to_coord_value(images, normalize_coords=True):
    """
    Convert batch of images to coordinate-value pairs.

    Args:
        images: (B, C, H, W) normalized images
        normalize_coords: whether to normalize coords to [0, 1]

    Returns:
        coords: (B, H*W, 2)  — (y, x) per pixel
        values: (B, H*W, C)  — channel values per pixel
    """
    B, C, H, W = images.shape
    coords = make_coord_grid(H, W, device=images.device, normalize=normalize_coords)
    coords = coords.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
    values = images.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
    return coords, values


def sample_sparse_observations(coords, values, obs_frac, sort_by_raster=True):
    """
    Randomly sample a fraction of coordinate-value pairs.

    Args:
        coords: (B, N, 2)
        values: (B, N, C)
        obs_frac: fraction of pixels to observe (0, 1]

    Returns:
        obs_coords:  (B, K, 2)
        obs_values:  (B, K, C)
        obs_indices: (B, K) — indices into the original N
    """
    B, N, _ = coords.shape
    K = max(1, int(N * obs_frac))

    indices = torch.stack([torch.randperm(N, device=coords.device)[:K] for _ in range(B)])

    if sort_by_raster:
        indices, _ = indices.sort(dim=1)

    obs_coords = torch.gather(coords, 1, indices.unsqueeze(-1).expand(-1, -1, 2))
    obs_values = torch.gather(values, 1, indices.unsqueeze(-1).expand(-1, -1, values.size(-1)))
    return obs_coords, obs_values, indices


def sample_target_set(coords, values, target_frac, context_indices=None, sort_by_raster=True):
    """
    Sample target coordinates, optionally excluding context indices.

    Args:
        coords: (B, N, 2)
        values: (B, N, C)
        target_frac: fraction of pixels for target
        context_indices: (B, K) — indices to potentially exclude
        sort_by_raster: sort by raster order

    Returns:
        tgt_coords:  (B, T, 2)
        tgt_values:  (B, T, C)
        tgt_indices: (B, T)
    """
    B, N, _ = coords.shape
    T = max(1, int(N * target_frac))

    if context_indices is not None:
        # Exclude context indices: create mask, sample from remaining
        tgt_indices_list = []
        for b in range(B):
            mask = torch.ones(N, dtype=torch.bool, device=coords.device)
            mask[context_indices[b]] = False
            available = mask.nonzero(as_tuple=False).squeeze(-1)
            if len(available) < T:
                available = torch.arange(N, device=coords.device)
            perm = torch.randperm(len(available), device=coords.device)[:T]
            tgt_indices_list.append(available[perm])
        tgt_indices = torch.stack(tgt_indices_list)
    else:
        tgt_indices = torch.stack([torch.randperm(N, device=coords.device)[:T] for _ in range(B)])

    if sort_by_raster:
        tgt_indices, _ = tgt_indices.sort(dim=1)

    tgt_coords = torch.gather(coords, 1, tgt_indices.unsqueeze(-1).expand(-1, -1, 2))
    tgt_values = torch.gather(values, 1, tgt_indices.unsqueeze(-1).expand(-1, -1, values.size(-1)))
    return tgt_coords, tgt_values, tgt_indices


# ---------------------------------------------------------------------------
# Sinusoidal coordinate embedding (Fourier features)
# ---------------------------------------------------------------------------

class FourierCoordEmbed(nn.Module):
    """
    Fourier positional encoding for 2D coordinates.
    Maps (B, L, 2) → (B, L, out_dim).
    """

    def __init__(self, out_dim, num_freqs=32):
        super().__init__()
        self.num_freqs = num_freqs
        # Learnable projection from Fourier features to out_dim
        freq_dim = 2 * 2 * num_freqs + 2  # sin + cos for each coord × num_freqs + raw coords
        self.proj = nn.Sequential(
            nn.Linear(freq_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        # Fixed frequency bands
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs)  # (num_freqs,)

    def forward(self, coords):
        """coords: (B, L, 2)"""
        # Expand coords with Fourier features
        coord_proj = coords.unsqueeze(-1) * self.freqs * math.pi  # (B, L, 2, num_freqs)
        fourier = torch.cat([coord_proj.sin(), coord_proj.cos()], dim=-1)  # (B, L, 2, 2*num_freqs)
        fourier = fourier.flatten(-2)  # (B, L, 4*num_freqs)
        features = torch.cat([coords, fourier], dim=-1)  # (B, L, 4*num_freqs + 2)
        return self.proj(features)


# ---------------------------------------------------------------------------
# Mamba encoder block
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Residual Mamba block: LayerNorm → Mamba → residual."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


# ---------------------------------------------------------------------------
# Fallback: Transformer encoder block (when mamba-ssm not available)
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """Simple pre-norm Transformer encoder block."""

    def __init__(self, d_model, nhead=4, dim_ff=None, dropout=0.0):
        super().__init__()
        dim_ff = dim_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Neural Field Encoder (Mamba-based, with Transformer fallback)
# ---------------------------------------------------------------------------

class NeuralFieldEncoder(nn.Module):
    """
    Encodes sparse (coordinate, value) observations into per-token features
    and a pooled global code.

    Architecture:
      1. Fourier coord embedding + linear value embedding → token features
      2. N layers of Mamba (or Transformer) blocks
      3. LayerNorm
      4. Global pool via mean

    Args:
        d_model:    hidden dimension
        n_layers:   number of Mamba/Transformer layers
        d_value:    dimension of input values (3 for RGB)
        num_freqs:  Fourier frequency bands for coord embedding
        use_mamba:  whether to use Mamba (falls back to Transformer if unavailable)
        nhead:      attention heads for fallback Transformer
    """

    def __init__(self, d_model=192, n_layers=4, d_value=3, num_freqs=32,
                 use_mamba=True, nhead=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model

        self.coord_embed = FourierCoordEmbed(d_model, num_freqs=num_freqs)
        self.value_embed = nn.Sequential(
            nn.Linear(d_value, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.use_mamba = use_mamba and MAMBA_AVAILABLE
        if self.use_mamba:
            self.layers = nn.ModuleList([
                MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ])
        else:
            if use_mamba and not MAMBA_AVAILABLE:
                import warnings
                warnings.warn("mamba-ssm not found. Falling back to Transformer encoder.")
            self.layers = nn.ModuleList([
                TransformerEncoderBlock(d_model, nhead=nhead)
                for _ in range(n_layers)
            ])

        self.norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords, values):
        """
        Args:
            coords: (B, K, 2) — normalized coordinates
            values: (B, K, C) — pixel/patch values

        Returns:
            token_feats: (B, K, d_model) — per-token features
            z_global:    (B, d_model)    — mean-pooled global code
        """
        x = self.coord_embed(coords) + self.value_embed(values)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        z_global = x.mean(dim=1)
        return x, z_global


# ---------------------------------------------------------------------------
# Coordinate Predictor
# ---------------------------------------------------------------------------

class CoordPredictor(nn.Module):
    """
    Predicts target encoder embeddings at target coordinates given context.

    Takes context token features (pooled or per-token) and target coordinates,
    and predicts the target encoder embedding at each target coordinate.

    Two modes:
      - 'pool': uses global-pooled context z only (simpler)
      - 'cross_attn': target coord tokens cross-attend to context tokens (richer)
    """

    def __init__(self, d_model=192, d_pred=96, n_layers=3, num_freqs=32,
                 mode="cross_attn", nhead=3):
        super().__init__()
        self.d_model = d_model
        self.d_pred = d_pred
        self.mode = mode

        self.coord_embed = FourierCoordEmbed(d_pred, num_freqs=num_freqs)
        self.context_proj = nn.Linear(d_model, d_pred, bias=True)

        if mode == "cross_attn":
            self.cross_attn_layers = nn.ModuleList()
            self.self_attn_layers = nn.ModuleList()
            for _ in range(n_layers):
                self.cross_attn_layers.append(
                    nn.MultiheadAttention(d_pred, nhead, batch_first=True)
                )
                self.self_attn_layers.append(
                    nn.Sequential(
                        nn.LayerNorm(d_pred),
                        nn.Linear(d_pred, d_pred * 4),
                        nn.GELU(),
                        nn.Linear(d_pred * 4, d_pred),
                    )
                )
            self.cross_norms = nn.ModuleList([nn.LayerNorm(d_pred) for _ in range(n_layers)])
            self.kv_norms = nn.ModuleList([nn.LayerNorm(d_pred) for _ in range(n_layers)])
        else:
            # Pool mode: simple MLP on (z + coord_embed)
            self.mlp_layers = nn.ModuleList()
            for _ in range(n_layers):
                self.mlp_layers.append(nn.Sequential(
                    nn.LayerNorm(d_pred),
                    nn.Linear(d_pred, d_pred * 4),
                    nn.GELU(),
                    nn.Linear(d_pred * 4, d_pred),
                ))

        self.out_norm = nn.LayerNorm(d_pred)
        self.out_proj = nn.Linear(d_pred, d_model, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, context_tokens, target_coords, z_global=None):
        """
        Args:
            context_tokens: (B, K_ctx, d_model) — from context encoder
            target_coords:  (B, T, 2) — target coordinate positions
            z_global:        (B, d_model) — pooled context (used in pool mode)

        Returns: (B, T, d_model) — predicted embeddings at target coords
        """
        # Embed target coordinates
        tgt = self.coord_embed(target_coords)  # (B, T, d_pred)

        if self.mode == "cross_attn":
            kv = self.context_proj(context_tokens)  # (B, K_ctx, d_pred)
            for ca, ff, qnorm, kvnorm in zip(
                self.cross_attn_layers, self.self_attn_layers,
                self.cross_norms, self.kv_norms
            ):
                q = qnorm(tgt)
                k = v = kvnorm(kv)
                tgt = tgt + ca(q, k, v, need_weights=False)[0]
                tgt = tgt + ff(tgt)
        else:
            z = self.context_proj(z_global).unsqueeze(1)  # (B, 1, d_pred)
            tgt = tgt + z
            for mlp in self.mlp_layers:
                tgt = tgt + mlp(tgt)

        tgt = self.out_norm(tgt)
        return self.out_proj(tgt)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# INR Decoder (optional reconstruction head)
# ---------------------------------------------------------------------------

class INRDecoder(nn.Module):
    """
    Implicit Neural Representation decoder.

    Maps (global_code, query_coords) → RGB values.
    Used as an auxiliary reconstruction head for debugging / PSNR evaluation.

    Architecture: MLP with Fourier-embedded coordinates concatenated with z.
    """

    def __init__(self, d_model=192, hidden_dim=256, n_layers=4, num_freqs=32, out_channels=3):
        super().__init__()
        self.coord_embed = FourierCoordEmbed(hidden_dim, num_freqs=num_freqs)
        self.z_proj = nn.Linear(d_model, hidden_dim)

        layers = []
        for i in range(n_layers):
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_global, query_coords):
        """
        Args:
            z_global:     (B, d_model) — global context code
            query_coords: (B, Q, 2) — query positions

        Returns: (B, Q, out_channels) — predicted values (e.g. RGB)
        """
        B, Q, _ = query_coords.shape
        z = self.z_proj(z_global).unsqueeze(1).expand(-1, Q, -1)  # (B, Q, hidden)
        c = self.coord_embed(query_coords)  # (B, Q, hidden)
        h = z + c
        h = self.mlp(h)
        return self.out(h)  # (B, Q, C)


# ---------------------------------------------------------------------------
# Full NF+JEPA model (bundles encoder, predictor, decoder)
# ---------------------------------------------------------------------------

class NFJEPAModel(nn.Module):
    """
    Neural Field + JEPA model bundle.

    Holds the context encoder, predictor, and optional INR decoder.
    The target encoder should be created externally as a deepcopy of
    the context encoder with EMA updates.
    """

    def __init__(self, d_model=192, n_layers=4, d_value=3, num_freqs=32,
                 use_mamba=True, nhead=4,
                 pred_dim=96, pred_layers=3, pred_mode="cross_attn",
                 use_inr_decoder=True, inr_hidden=256, inr_layers=4):
        super().__init__()

        self.encoder = NeuralFieldEncoder(
            d_model=d_model, n_layers=n_layers, d_value=d_value,
            num_freqs=num_freqs, use_mamba=use_mamba, nhead=nhead,
        )
        self.predictor = CoordPredictor(
            d_model=d_model, d_pred=pred_dim, n_layers=pred_layers,
            num_freqs=num_freqs, mode=pred_mode, nhead=nhead,
        )
        self.inr_decoder = None
        if use_inr_decoder:
            self.inr_decoder = INRDecoder(
                d_model=d_model, hidden_dim=inr_hidden,
                n_layers=inr_layers, num_freqs=num_freqs, out_channels=d_value,
            )

    def forward_context(self, ctx_coords, ctx_values):
        """Encode sparse context observations."""
        return self.encoder(ctx_coords, ctx_values)

    def forward_predict(self, context_tokens, target_coords, z_global=None):
        """Predict target encoder embeddings at target coordinates."""
        return self.predictor(context_tokens, target_coords, z_global)

    def forward_reconstruct(self, z_global, query_coords):
        """Optional: reconstruct RGB at query coordinates."""
        if self.inr_decoder is None:
            raise RuntimeError("INR decoder is not enabled.")
        return self.inr_decoder(z_global, query_coords)


def make_nf_jepa_model(
    d_model=192, n_layers=4, d_value=3, num_freqs=32,
    use_mamba=True, nhead=4,
    pred_dim=96, pred_layers=3, pred_mode="cross_attn",
    use_inr_decoder=True, inr_hidden=256, inr_layers=4,
):
    """Factory for NF+JEPA model."""
    return NFJEPAModel(
        d_model=d_model, n_layers=n_layers, d_value=d_value,
        num_freqs=num_freqs, use_mamba=use_mamba, nhead=nhead,
        pred_dim=pred_dim, pred_layers=pred_layers, pred_mode=pred_mode,
        use_inr_decoder=use_inr_decoder, inr_hidden=inr_hidden, inr_layers=inr_layers,
    )
