"""JAX noise and perturbation strategies.

This module provides JAX-compatible noise injection for forward passes,
mirroring the PyTorch noise.py functionality.

Key differences from PyTorch:
1. All operations are pure functions (no internal mutable state)
2. RNG keys must be explicitly passed and split
3. Perturbations are precomputed and passed through the scan
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Noise Strategies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianNoiseConfig:
    """Configuration for per-step Gaussian noise.

    Adds fresh random noise at each timestep.

    Args:
        std: Standard deviation of the noise
        relative: If True, scales std by abs(tensor) element-wise
    """

    std: float = 0.0
    relative: bool = False


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration for fixed-per-forward perturbation.

    Draws a single offset once at the start of forward pass,
    reused across all timesteps. This models device-to-device
    or batch-to-batch variation.

    Args:
        mean: Mean of the perturbation
        std: Standard deviation of the perturbation
    """

    mean: float = 0.0
    std: float = 0.0


@dataclass(frozen=True)
class NoiseConfig:
    """Complete noise configuration for a single tensor (phi, s, g, etc.).

    Combines per-step noise and fixed perturbation.
    """

    noise: GaussianNoiseConfig | None = None
    perturbation: PerturbationConfig | None = None

    @property
    def is_trivial(self) -> bool:
        """Returns True if no noise/perturbation is configured."""
        has_noise = self.noise is not None and self.noise.std != 0.0
        has_perturb = self.perturbation is not None and (
            self.perturbation.std != 0.0 or self.perturbation.mean != 0.0
        )
        return not has_noise and not has_perturb


@dataclass(frozen=True)
class NoiseSettings:
    """Container for all noise configurations.

    Maps tensor keys to their noise configs:
        - phi: input flux noise
        - s: state noise
        - g: activation noise
        - j: connection weight noise
        - params: dict of param_name -> NoiseConfig
    """

    phi: NoiseConfig | None = None
    s: NoiseConfig | None = None
    g: NoiseConfig | None = None
    j: NoiseConfig | None = None
    # Extra parameter noise (e.g., tau, beta)
    extras: dict[str, NoiseConfig] | None = None

    def is_trivial(self) -> bool:
        """Returns True if no noise is configured anywhere."""
        for cfg in (self.phi, self.s, self.g, self.j):
            if cfg is not None and not cfg.is_trivial:
                return False
        if self.extras:
            for cfg in self.extras.values():
                if cfg is not None and not cfg.is_trivial:
                    return False
        return True


# ---------------------------------------------------------------------------
# Precomputed Perturbations
# ---------------------------------------------------------------------------


class PerturbationOffsets(NamedTuple):
    """Precomputed perturbation offsets for a forward pass.

    These are sampled once at the start and reused across all timesteps.
    Shape: [batch, dim] for each tensor that has perturbation configured.
    """

    phi: jax.Array | None = None
    s: jax.Array | None = None
    g: jax.Array | None = None
    # Per-layer perturbations: layer_idx -> {param_name: offset}
    layer_params: dict[int, dict[str, jax.Array]] | None = None


def sample_perturbation_offset(
    key: jax.Array,
    shape: tuple[int, ...],
    config: PerturbationConfig,
    dtype: jnp.dtype,
) -> jax.Array:
    """Sample a single perturbation offset.

    Args:
        key: JAX RNG key
        shape: Shape for the offset (typically [batch, dim])
        config: Perturbation configuration
        dtype: Output dtype

    Returns:
        Offset array of the given shape
    """
    if config.std == 0.0:
        return jnp.full(shape, config.mean, dtype=dtype)
    return jax.random.normal(key, shape, dtype=dtype) * config.std + config.mean


def precompute_perturbations(
    key: jax.Array,
    settings: NoiseSettings,
    batch_size: int,
    layer_dims: dict[int, int],  # layer_idx -> dim
    param_shapes: dict[int, dict[str, tuple[int, ...]]] | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> PerturbationOffsets:
    """Precompute all perturbation offsets for a forward pass.

    Called once at the start of forward pass. The offsets are then
    passed through the scan and reused at each timestep.

    Args:
        key: JAX RNG key
        settings: Noise configuration
        batch_size: Batch size
        layer_dims: Dict mapping layer_idx to output dimension
        param_shapes: Optional dict of layer_idx -> {param_name: shape}
        dtype: Output dtype

    Returns:
        PerturbationOffsets containing all sampled offsets
    """
    phi_offset = None
    s_offset = None
    g_offset = None
    layer_params: dict[int, dict[str, jax.Array]] = {}

    # Phi perturbation (applied to input, uses first layer dim)
    if settings.phi and settings.phi.perturbation:
        cfg = settings.phi.perturbation
        if cfg.std != 0.0 or cfg.mean != 0.0:
            key, subkey = jax.random.split(key)
            # Use input dim (first layer) - caller should pass this
            input_dim = list(layer_dims.values())[0] if layer_dims else 1
            phi_offset = sample_perturbation_offset(
                subkey, (batch_size, input_dim), cfg, dtype
            )

    # State and g perturbations need per-layer offsets
    for layer_idx, dim in layer_dims.items():
        layer_offsets: dict[str, jax.Array] = {}

        if settings.s and settings.s.perturbation:
            cfg = settings.s.perturbation
            if cfg.std != 0.0 or cfg.mean != 0.0:
                key, subkey = jax.random.split(key)
                layer_offsets["s"] = sample_perturbation_offset(
                    subkey, (batch_size, dim), cfg, dtype
                )

        if settings.g and settings.g.perturbation:
            cfg = settings.g.perturbation
            if cfg.std != 0.0 or cfg.mean != 0.0:
                key, subkey = jax.random.split(key)
                layer_offsets["g"] = sample_perturbation_offset(
                    subkey, (batch_size, dim), cfg, dtype
                )

        # Parameter perturbations
        if settings.extras and param_shapes and layer_idx in param_shapes:
            for param_name, ncfg_val in settings.extras.items():
                if ncfg_val and ncfg_val.perturbation:
                    pcfg = ncfg_val.perturbation
                    if pcfg.std != 0.0 or pcfg.mean != 0.0:
                        if param_name in param_shapes[layer_idx]:
                            key, subkey = jax.random.split(key)
                            layer_offsets[param_name] = sample_perturbation_offset(
                                subkey,
                                param_shapes[layer_idx][param_name],
                                pcfg,
                                dtype,
                            )

        if layer_offsets:
            layer_params[layer_idx] = layer_offsets

    return PerturbationOffsets(
        phi=phi_offset,
        s=s_offset,
        g=g_offset,
        layer_params=layer_params if layer_params else None,
    )


# ---------------------------------------------------------------------------
# Noise Application Functions
# ---------------------------------------------------------------------------


def apply_noise_step(
    key: jax.Array,
    tensor: jax.Array,
    config: NoiseConfig | None,
    perturbation_offset: jax.Array | None = None,
) -> jax.Array:
    """Apply noise and perturbation to a tensor at a single timestep.

    Args:
        key: JAX RNG key (only used if noise.std > 0)
        tensor: Input tensor [batch, dim]
        config: Noise configuration
        perturbation_offset: Precomputed perturbation offset (if any)

    Returns:
        Tensor with noise/perturbation applied
    """
    if config is None:
        return tensor

    result = tensor

    # Apply perturbation offset (fixed across timesteps)
    if perturbation_offset is not None:
        # Broadcast offset to match tensor shape if needed
        if perturbation_offset.ndim < result.ndim:
            # Expand from [B, D] to [B, T, D] if needed
            perturbation_offset = jnp.expand_dims(perturbation_offset, axis=1)
        result = result + perturbation_offset

    # Apply per-step noise
    if config.noise is not None and config.noise.std > 0.0:
        if config.noise.relative:
            scale = cast(jax.Array, config.noise.std * jnp.abs(tensor))
        else:
            scale = cast(jax.Array, jnp.asarray(config.noise.std, dtype=tensor.dtype))
        noise = jax.random.normal(key, tensor.shape, dtype=tensor.dtype) * scale
        result = result + noise

    return result


def apply_connection_noise(
    key: jax.Array,
    J: jax.Array,
    config: NoiseConfig | None,
    perturbation_offset: jax.Array | None = None,
) -> jax.Array:
    """Apply ONLY per-step noise to a connection weight matrix.

    For perturbation (fixed per forward), use precompute_connection_perturbation() instead.

    Args:
        key: JAX RNG key
        J: Weight matrix [D, F]
        config: Noise configuration for connections
        perturbation_offset: Precomputed perturbation offset [D, F] (optional)

    Returns:
        Weight matrix with noise applied [D, F]
    """
    if config is None:
        return J

    result = J

    # Apply precomputed perturbation offset if provided
    if perturbation_offset is not None:
        result = result + perturbation_offset

    # Apply per-step Gaussian noise (time-varying)
    if config.noise is not None and config.noise.std > 0.0:
        if config.noise.relative:
            scale = cast(jax.Array, config.noise.std * jnp.abs(J))
        else:
            scale = cast(jax.Array, jnp.asarray(config.noise.std, dtype=J.dtype))
        noise = jax.random.normal(key, J.shape, dtype=J.dtype) * scale
        result = result + noise

    return result


def precompute_connection_perturbation(
    key: jax.Array,
    J: jax.Array,
    config: NoiseConfig | None,
) -> jax.Array | None:
    """Precompute perturbation offset for a connection (fixed per forward pass).

    Args:
        key: JAX RNG key
        J: Weight matrix [D, F]
        config: Noise configuration

    Returns:
        Perturbation offset [D, F] or None if no perturbation configured
    """
    if config is None or config.perturbation is None:
        return None

    pcfg = config.perturbation
    if pcfg.std == 0.0 and pcfg.mean == 0.0:
        return None

    if pcfg.std == 0.0:
        return jnp.full(J.shape, pcfg.mean, dtype=J.dtype)

    return jax.random.normal(key, J.shape, dtype=J.dtype) * pcfg.std + pcfg.mean


def apply_connection_noise_layerwise(
    key: jax.Array,
    J: jax.Array,
    config: NoiseConfig | None,
    T: int,
    perturbation_offset: jax.Array | None = None,
) -> jax.Array:
    """Apply per-timestep noise to connection for layerwise mode.

    Expands J from [D, F] to [T, D, F] with different noise at each timestep.
    Perturbation (fixed offset) is added once and applies to all timesteps.

    Args:
        key: JAX RNG key
        J: Weight matrix [D, F]
        config: Noise configuration
        T: Number of timesteps
        perturbation_offset: Precomputed perturbation [D, F] (optional)

    Returns:
        Weight matrices [T, D, F] with per-timestep noise
    """
    D, F = J.shape

    # Start with base J expanded to [T, D, F]
    result = jnp.broadcast_to(J, (T, D, F))

    # Add perturbation (same offset for all timesteps)
    if perturbation_offset is not None:
        result = result + perturbation_offset[None, :, :]

    # Add per-timestep noise if configured
    if config is not None and config.noise is not None and config.noise.std > 0.0:
        if config.noise.relative:
            scale = cast(jax.Array, config.noise.std * jnp.abs(J))  # [D, F]
            scale = jnp.broadcast_to(scale, (T, D, F))
        else:
            scale = cast(jax.Array, jnp.asarray(config.noise.std, dtype=J.dtype))

        noise = jax.random.normal(key, (T, D, F), dtype=J.dtype) * scale
        result = result + noise

    return result


# ---------------------------------------------------------------------------
# Unified Precomputation (used by both layerwise and stepwise forwards)
# ---------------------------------------------------------------------------


class ForwardNoiseContext(NamedTuple):
    """All precomputed noise data needed for a forward pass.

    This consolidates the precomputation that was duplicated in
    unified_forward.py and unified_stepwise.py.
    """

    # Layer perturbation offsets (for phi, s, g)
    layer_perturbations: PerturbationOffsets | None = None
    # Connection perturbation offsets: conn_key -> offset [D, F]
    conn_perturbations: dict[str, jax.Array] | None = None
    # Whether any noise is configured (for fast path checks)
    has_layer_noise: bool = False
    has_connection_noise: bool = False


def precompute_forward_noise(
    rng_key: jax.Array | None,
    model,  # JAXModel
    noise_settings: NoiseSettings | None,
    B: int,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array | None, ForwardNoiseContext]:
    """Precompute all noise offsets for a forward pass.

    Consolidates the noise precomputation logic that was duplicated in
    unified_forward.py and unified_stepwise.py.

    Args:
        rng_key: JAX RNG key (or None if no noise)
        model: JAXModel instance
        noise_settings: Layer noise settings (or None)
        B: Batch size
        dtype: Array dtype

    Returns:
        Tuple of (remaining_rng_key, ForwardNoiseContext)
    """
    has_layer_noise = noise_settings is not None and not noise_settings.is_trivial()
    has_connection_noise = (
        model.connection_noise_settings is not None
        and len(model.connection_noise_settings) > 0
    )

    if rng_key is None:
        return None, ForwardNoiseContext(
            layer_perturbations=None,
            conn_perturbations=None,
            has_layer_noise=has_layer_noise,
            has_connection_noise=has_connection_noise,
        )

    # Precompute layer perturbation offsets
    layer_perturbations = None
    if has_layer_noise:
        layers_sorted = sorted(model.layers, key=lambda layer: layer.layer_id)
        layer_dims = {spec.layer_id: spec.dim for spec in layers_sorted}
        rng_key, perturb_key = jax.random.split(rng_key)
        layer_perturbations = precompute_perturbations(
            perturb_key, cast("NoiseSettings", noise_settings), B, layer_dims, dtype=dtype
        )

    # Precompute connection perturbation offsets (fixed per forward)
    conn_perturbations: dict[str, jax.Array] = {}
    if has_connection_noise:
        for c in model.connections:
            conn_key = f"J_{c.from_layer}_to_{c.to_layer}"
            conn_noise_cfg = model.connection_noise_settings.get(conn_key)
            if conn_noise_cfg is not None:
                rng_key, perturb_key = jax.random.split(rng_key)
                offset = precompute_connection_perturbation(perturb_key, c.J, conn_noise_cfg)
                if offset is not None:
                    conn_perturbations[conn_key] = offset

    return rng_key, ForwardNoiseContext(
        layer_perturbations=layer_perturbations,
        conn_perturbations=conn_perturbations if conn_perturbations else None,
        has_layer_noise=has_layer_noise,
        has_connection_noise=has_connection_noise,
    )


# ---------------------------------------------------------------------------
# Builder from dict config
# ---------------------------------------------------------------------------


def build_noise_settings(config: dict | None) -> NoiseSettings:
    """Build NoiseSettings from a dict configuration.

    Supports format:
    {
        "phi": 0.01,           # shorthand for std
        "phi_std": 0.01,       # explicit noise std
        "phi_perturb_std": 0.02,  # perturbation std
        "phi_perturb_mean": 0.0,
        "s": 0.005,
        "relative": True,      # applies to all noise
        ...
    }

    Args:
        config: Dict configuration or None

    Returns:
        NoiseSettings object
    """
    if config is None:
        return NoiseSettings()

    relative = config.get("relative", False)

    def make_config(key: str) -> NoiseConfig | None:
        """Create NoiseConfig for a given key."""
        # Check for noise
        noise_std = config.get(f"{key}_std", config.get(key, 0.0))
        noise = None
        if noise_std != 0.0:
            noise = GaussianNoiseConfig(std=float(noise_std), relative=relative)

        # Check for perturbation
        perturb_std = config.get(f"{key}_perturb_std", 0.0)
        perturb_mean = config.get(f"{key}_perturb_mean", 0.0)
        perturbation = None
        if perturb_std != 0.0 or perturb_mean != 0.0:
            perturbation = PerturbationConfig(
                mean=float(perturb_mean), std=float(perturb_std)
            )

        if noise is None and perturbation is None:
            return None
        return NoiseConfig(noise=noise, perturbation=perturbation)

    # Collect extras (non-standard keys)
    standard_keys = {"phi", "s", "g", "j", "relative"}
    suffixes = {"_std", "_perturb_std", "_perturb_mean"}
    extras: dict[str, NoiseConfig] = {}
    for key in config:
        base_key = key
        for suffix in suffixes:
            if key.endswith(suffix):
                base_key = key[: -len(suffix)]
                break
        if base_key not in standard_keys and base_key not in extras:
            cfg = make_config(base_key)
            if cfg is not None:
                extras[base_key] = cfg

    return NoiseSettings(
        phi=make_config("phi"),
        s=make_config("s"),
        g=make_config("g"),
        j=make_config("j"),
        extras=extras if extras else None,
    )


__all__ = [
    "GaussianNoiseConfig",
    "PerturbationConfig",
    "NoiseConfig",
    "NoiseSettings",
    "PerturbationOffsets",
    "ForwardNoiseContext",
    "precompute_perturbations",
    "precompute_forward_noise",
    "apply_noise_step",
    "apply_connection_noise",
    "precompute_connection_perturbation",
    "apply_connection_noise_layerwise",
    "build_noise_settings",
]

