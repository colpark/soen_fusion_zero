"""Iterative flux-matching weight initialization.

This module provides utilities to iteratively adjust weights based on observed
upstream states, converging towards the target flux at each node.

Two weight update modes are available:
- "node_wise": Compute a single weight per destination node (averages upstream states)
- "connection_wise": Compute individual weights per connection (uses each source state directly)

Layer Filtering:
- include_layers: Only initialize specified layers (None = all)
- exclude_layers: Skip specified layers (takes precedence over include_layers)

Per-Layer Configuration:
- layer_overrides: Dict mapping layer_id -> LayerFluxConfig for custom targets per layer
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Literal

import h5py
import torch

if TYPE_CHECKING:
    from soen_toolkit.core.soen_model_core import SOENModelCore

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Per-layer configuration
# -----------------------------------------------------------------------------

@dataclass
class LayerFluxConfig:
    """Per-layer flux matching configuration.

    Use this to specify different target phi values for specific layers.
    """

    phi_total_target: float = 0.5
    """Target total flux for this layer."""

    phi_total_target_min: float | None = None
    """Min target for symmetry breaking (optional)."""

    phi_total_target_max: float | None = None
    """Max target for symmetry breaking (optional)."""

    alpha: float | None = None
    """Step size override for this layer (None = use global)."""

    skip: bool = False
    """If True, skip this layer entirely."""


class WeightUpdateMode(str, Enum):
    """Weight update strategy for flux-matching."""

    NODE_WISE = "node_wise"
    """Compute uniform weight per destination node (averages upstream states)."""

    CONNECTION_WISE = "connection_wise"
    """Compute individual weight per connection (uses each source state directly)."""


def _default_log_fn(msg: str) -> None:
    """Default logging function for verbose output."""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


@dataclass
class FluxMatchingConfig:
    """Configuration for iterative flux-matching initialization.

    Targets phi_total (the actual flux seen by the source function), not phi_exc.
    The algorithm automatically reads each node's phi_offset and computes the
    required external flux: phi_exc_target[i] = phi_total_target[i] - phi_offset[i]

    For symmetry breaking, specify a range (phi_total_target_min, phi_total_target_max).
    Targets are uniformly distributed across dendrites within each layer.

    Layer Filtering:
        Use include_layers or exclude_layers to control which layers are initialized.
        exclude_layers takes precedence if both are specified.

    Per-Layer Overrides:
        Use layer_overrides to specify different targets for specific layers.
    """

    phi_total_target: float = 0.5
    """Target total flux per node (phi_exc + phi_offset). 0.5 is the optimal operating point."""

    phi_total_target_min: float | None = None
    """Minimum target total flux (for symmetry breaking). If None, uses phi_total_target."""

    phi_total_target_max: float | None = None
    """Maximum target total flux (for symmetry breaking). If None, uses phi_total_target."""

    num_iterations: int = 5
    """Number of flux-matching iterations."""

    batch_size: int = 32
    """Batch size for forward passes."""

    num_batches: int | None = None
    """Number of batches to use (None = use all data)."""

    min_state_clamp: float = 0.01
    """Minimum mean state to avoid division by zero."""

    alpha: float = 1.0
    """Step size for weight updates (0=no update, 1=full update, 0.5=halfway)."""

    weight_update_mode: Literal["node_wise", "connection_wise"] = "connection_wise"
    """Weight update strategy: 'node_wise' or 'connection_wise'."""

    flux_allocation_mode: Literal[
        "equal_per_incoming_connection",
        "proportional_to_fan_in",
    ] = "equal_per_incoming_connection"
    """How to split each destination node's phi_exc_target across incoming connections.

    This is separate from fan-in:
    - Incoming connections (edges): e.g. J_0_to_4 vs J_1_to_4
    - Fan-in (synapses within one edge): row-sum of that edge's mask, per destination node

    Modes:
    - equal_per_incoming_connection (default): each incoming edge gets an equal per-node share
    - proportional_to_fan_in: each incoming edge gets a per-node share proportional to its fan-in
    """

    verbose: bool = True
    """Whether to output iteration progress."""

    log_fn: Callable[[str], None] | None = None
    """Custom logging function. If None, uses stdout when verbose=True."""

    stop_check: Callable[[], bool] | None = None
    """Optional callback to check if processing should stop. Return True to cancel."""

    # -------------------------------------------------------------------------
    # Layer filtering
    # -------------------------------------------------------------------------

    include_layers: set[int] | None = None
    """Only initialize these layers. None means all layers. Ignored for input layer."""

    exclude_layers: set[int] | None = None
    """Skip these layers. Takes precedence over include_layers."""

    layer_overrides: dict[int, LayerFluxConfig] | None = None
    """Per-layer configuration overrides. Keys are layer IDs."""

    exclude_connections: set[str] | None = None
    """Connection keys to skip (e.g., {'J_0_to_1', 'J_2_to_3'}). These won't be updated."""

    def should_process_connection(self, connection_key: str) -> bool:
        """Check if a connection should be processed.

        Args:
            connection_key: The connection key (e.g., 'J_0_to_1')

        Returns:
            True if this connection should be updated
        """
        if self.exclude_connections and connection_key in self.exclude_connections:
            return False
        return True

    def should_process_layer(self, layer_id: int, is_input_layer: bool = False) -> bool:
        """Check if a layer should be processed.

        Args:
            layer_id: The layer ID to check
            is_input_layer: If True, always returns False (input layers have no weights)

        Returns:
            True if this layer should be processed
        """
        # Input layers are never processed (no incoming connections)
        if is_input_layer:
            return False

        # Check per-layer skip flag
        if self.layer_overrides:
            override = self.layer_overrides.get(layer_id)
            if override is not None and override.skip:
                return False

        # Exclude takes precedence
        if self.exclude_layers and layer_id in self.exclude_layers:
            return False

        # If include is specified, layer must be in it
        if self.include_layers is not None:
            return layer_id in self.include_layers

        return True

    def get_layer_config(self, layer_id: int) -> LayerFluxConfig:
        """Get effective config for a layer, merging overrides with defaults.

        Args:
            layer_id: The layer ID

        Returns:
            LayerFluxConfig with effective settings for this layer
        """
        # Start with global defaults
        base = LayerFluxConfig(
            phi_total_target=self.phi_total_target,
            phi_total_target_min=self.phi_total_target_min,
            phi_total_target_max=self.phi_total_target_max,
            alpha=self.alpha,
        )

        # Apply per-layer overrides if present
        if self.layer_overrides:
            override = self.layer_overrides.get(layer_id)
            if override is not None:
                # Only override if explicitly set (not None for alpha)
                if override.phi_total_target != 0.5:  # Non-default
                    base.phi_total_target = override.phi_total_target
                if override.phi_total_target_min is not None:
                    base.phi_total_target_min = override.phi_total_target_min
                if override.phi_total_target_max is not None:
                    base.phi_total_target_max = override.phi_total_target_max
                if override.alpha is not None:
                    base.alpha = override.alpha

        return base

    def get_phi_total_target_per_node(
        self, num_nodes: int, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """Get per-node target phi_total values.

        If min/max are specified, returns linearly spaced values from min to max.
        Otherwise, returns a constant tensor of phi_total_target.

        Args:
            num_nodes: Number of nodes in the layer
            device: Device to create tensor on

        Returns:
            Tensor of shape [num_nodes] with target phi_total for each node
        """
        if self.phi_total_target_min is not None and self.phi_total_target_max is not None:
            # Uniform distribution from min to max across nodes
            if num_nodes == 1:
                # Single node gets the midpoint
                mid = (self.phi_total_target_min + self.phi_total_target_max) / 2
                return torch.tensor([mid], device=device)
            return torch.linspace(
                self.phi_total_target_min,
                self.phi_total_target_max,
                num_nodes,
                device=device,
            )
        else:
            # Constant target for all nodes
            return torch.full((num_nodes,), self.phi_total_target, device=device)

    def get_phi_exc_target_per_node(
        self,
        num_nodes: int,
        phi_offset: torch.Tensor | None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Get per-node target phi_exc values, accounting for each node's phi_offset.

        Computes: phi_exc_target[i] = phi_total_target[i] - phi_offset[i]

        Args:
            num_nodes: Number of nodes in the layer
            phi_offset: Per-node phi_offset tensor [num_nodes], or None if layer has no offset
            device: Device to create tensor on

        Returns:
            Tensor of shape [num_nodes] with target phi_exc for each node
        """
        phi_total_targets = self.get_phi_total_target_per_node(num_nodes, device)

        if phi_offset is None:
            # No offset, phi_exc = phi_total
            return phi_total_targets

        # Ensure phi_offset is on correct device
        offset = phi_offset.to(device=device)
        return phi_total_targets - offset


@dataclass
class FluxMatchingResult:
    """Results from flux-matching iterations."""

    iteration_stats: list[dict[str, Any]] = field(default_factory=list)
    """Per-iteration statistics."""

    final_mean_states: dict[int, torch.Tensor] = field(default_factory=dict)
    """Final mean state per layer."""

    final_flux_per_node: dict[str, torch.Tensor] = field(default_factory=dict)
    """Final flux per destination node for each connection."""

    converged: bool = False
    """Whether the flux converged to target."""


def load_hdf5_batches(
    hdf5_path: str | Path,
    split: str = "train",
    batch_size: int = 32,
    num_batches: int | None = None,
    device: torch.device | str = "cpu",
    seq_len: int | None = None,
    feature_min: float | None = None,
    feature_max: float | None = None,
) -> list[torch.Tensor]:
    """Load batches from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        split: Dataset split ('train', 'val', 'test', or 'all_data')
        batch_size: Batch size
        num_batches: Number of batches to load (None = all)
        device: Device to load tensors to
        seq_len: Truncate/pad sequences to this length (None = use original)
        feature_min: Scale features to have this minimum value
        feature_max: Scale features to have this maximum value

    Returns:
        List of input tensors, each of shape [batch, seq_len, input_dim]
    """
    import os

    # Disable HDF5 file locking to avoid issues on network drives and macOS
    # This is safe for read-only access
    old_locking = os.environ.get("HDF5_USE_FILE_LOCKING")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    try:
        batches = []
        # Use swmr=True for single-writer-multi-reader mode if available
        # and locking='false' to disable file locking
        try:
            f = h5py.File(hdf5_path, "r", locking=False)
        except TypeError:
            # Older h5py versions don't support locking parameter
            f = h5py.File(hdf5_path, "r")

        try:
            if split in f:
                data = f[split]["data"][:]
            elif "all_data" in f:
                data = f["all_data"][:]
            else:
                available_keys = list(f.keys())
                f.close()
                msg = f"Could not find data in HDF5 file. Available keys: {available_keys}"
                raise ValueError(msg)
        finally:
            f.close()

        # Convert to tensor
        data = torch.from_numpy(data).float().to(device)

        # Truncate or pad to seq_len if specified
        if seq_len is not None and data.ndim >= 2:
            current_len = data.shape[1]
            if current_len > seq_len:
                # Truncate
                data = data[:, :seq_len, ...]
            elif current_len < seq_len:
                # Pad with zeros
                pad_shape = list(data.shape)
                pad_shape[1] = seq_len - current_len
                padding = torch.zeros(pad_shape, device=device, dtype=data.dtype)
                data = torch.cat([data, padding], dim=1)

        # Apply feature scaling if both min and max are specified
        if feature_min is not None and feature_max is not None:
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                # Scale to [feature_min, feature_max]
                data = (data - data_min) / (data_max - data_min)
                data = data * (feature_max - feature_min) + feature_min

        # Split into batches
        n_samples = data.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        if num_batches is not None:
            n_batches = min(n_batches, num_batches)

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batches.append(data[start:end])

        return batches

    finally:
        # Restore original locking setting
        if old_locking is None:
            os.environ.pop("HDF5_USE_FILE_LOCKING", None)
        else:
            os.environ["HDF5_USE_FILE_LOCKING"] = old_locking


def collect_layer_states(
    model: SOENModelCore,
    inputs: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Run forward pass and collect state values for each layer.

    Args:
        model: The SOEN model
        inputs: Input tensor of shape [batch, seq_len, input_dim]

    Returns:
        Dictionary mapping layer_id -> state tensor [batch, seq_len, dim]
    """
    # Forward pass
    with torch.no_grad():
        output = model(inputs)

    # Handle different output formats
    layer_histories = None
    if isinstance(output, tuple) and len(output) == 2:
        # Stepwise solvers return (final_output, layer_histories)
        _, layer_histories = output

    # Collect states from layer histories if available
    layer_states = {}

    if layer_histories is not None and isinstance(layer_histories, list):
        # layer_histories is a list of tensors, one per layer
        # Each has shape [batch, seq+1, dim]
        for idx, cfg in enumerate(model.layers_config):
            if idx < len(layer_histories) and isinstance(layer_histories[idx], torch.Tensor):
                hist = layer_histories[idx]
                # Exclude initial state (first timestep)
                layer_states[cfg.layer_id] = hist[:, 1:, :]
    else:
        # Fallback: try to extract from layer objects
        for cfg in model.layers_config:
            layer_idx = next(
                i for i, c in enumerate(model.layers_config) if c.layer_id == cfg.layer_id
            )
            layer = model.layers[layer_idx]

            # Try get_state_history API
            if hasattr(layer, "get_state_history"):
                state_hist = layer.get_state_history()
                if state_hist is not None:
                    layer_states[cfg.layer_id] = state_hist
                    continue

            # Check _state_history directly
            if hasattr(layer, "_state_history") and layer._state_history:
                if isinstance(layer._state_history, list) and layer._state_history:
                    if isinstance(layer._state_history[0], torch.Tensor):
                        states = torch.stack(layer._state_history, dim=1)
                        layer_states[cfg.layer_id] = states
                        continue
                elif isinstance(layer._state_history, torch.Tensor):
                    layer_states[cfg.layer_id] = layer._state_history
                    continue

    return layer_states


def compute_mean_states(
    layer_states: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Compute mean state per node, averaged over time and batch.

    Args:
        layer_states: Dict mapping layer_id -> [batch, seq_len, dim]

    Returns:
        Dict mapping layer_id -> [dim] mean state per node
    """
    mean_states = {}
    for layer_id, states in layer_states.items():
        # Average over batch and time: [B, T, D] -> [D]
        mean_states[layer_id] = states.mean(dim=(0, 1))
    return mean_states


def compute_upstream_mean_states(
    model: SOENModelCore,
    mean_states: dict[int, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute mean upstream state for each connection.

    For each connection, compute the mean state of source neurons weighted
    by the connection mask.

    Args:
        model: The SOEN model
        mean_states: Mean state per layer [layer_id -> [dim]]

    Returns:
        Dict mapping connection_key -> mean upstream state per destination node [to_dim]
    """
    upstream_means = {}

    for conn in model.connections_config:
        key = f"J_{conn.from_layer}_to_{conn.to_layer}"
        mask = model.connection_masks.get(key)

        if mask is None or conn.from_layer not in mean_states:
            continue

        source_states = mean_states[conn.from_layer]  # [from_dim]

        # For each destination node, compute weighted mean of source states
        # mask shape: [to_dim, from_dim]
        # Weighted mean: sum(mask[i,:] * source_states) / sum(mask[i,:])
        fan_in = mask.sum(dim=1).clamp(min=1.0)  # [to_dim]
        weighted_sum = (mask * source_states.unsqueeze(0)).sum(dim=1)  # [to_dim]
        mean_upstream = weighted_sum / fan_in  # [to_dim]

        upstream_means[key] = mean_upstream

    return upstream_means


def compute_flux_per_node(
    model: SOENModelCore,
    mean_states: dict[int, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute actual flux contribution per destination node for each connection.

    Args:
        model: The SOEN model
        mean_states: Mean state per layer

    Returns:
        Dict mapping connection_key -> flux per destination node [to_dim]
    """
    flux_per_conn = {}

    for conn in model.connections_config:
        key = f"J_{conn.from_layer}_to_{conn.to_layer}"

        if key not in model.connections or conn.from_layer not in mean_states:
            continue

        weights = model.connections[key].detach()
        mask = model.connection_masks.get(key, torch.ones_like(weights))
        source_states = mean_states[conn.from_layer]

        # Flux to each destination: sum_j(J_ij * s_j)
        flux = (weights * mask) @ source_states
        flux_per_conn[key] = flux

    return flux_per_conn


def compute_total_flux_per_layer(
    model: SOENModelCore,
    flux_per_conn: dict[str, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Compute total incoming flux per layer by summing all connections.

    Args:
        model: The SOEN model
        flux_per_conn: Flux per connection

    Returns:
        Dict mapping layer_id -> total flux per node [dim]
    """
    total_flux = {}

    for key, flux in flux_per_conn.items():
        # Parse connection key: J_from_to_to
        parts = key.split("_")
        to_layer = int(parts[3])

        if to_layer not in total_flux:
            total_flux[to_layer] = torch.zeros_like(flux)
        total_flux[to_layer] = total_flux[to_layer] + flux

    return total_flux


def _get_layer_phi_offset(model: SOENModelCore, layer_id: int) -> torch.Tensor | None:
    """Get phi_offset parameter from a layer, if it exists.

    Args:
        model: The SOEN model
        layer_id: Layer ID to get phi_offset from

    Returns:
        phi_offset tensor [dim] or None if layer has no phi_offset
    """
    # Find the layer by ID
    layer_idx = None
    for idx, cfg in enumerate(model.layers_config):
        if cfg.layer_id == layer_id:
            layer_idx = idx
            break

    if layer_idx is None:
        return None

    layer = model.layers[layer_idx]

    # Check for phi_offset in various places
    # 1. Direct attribute (registered parameter)
    if hasattr(layer, "phi_offset") and isinstance(layer.phi_offset, torch.Tensor):
        return layer.phi_offset.detach()

    # 2. In _parameters dict (standard PyTorch way)
    if hasattr(layer, "_parameters") and "phi_offset" in layer._parameters:
        param = layer._parameters["phi_offset"]
        if param is not None:
            return param.detach()

    # 3. In a params dict or similar
    if hasattr(layer, "params") and isinstance(layer.params, dict):
        if "phi_offset" in layer.params:
            val = layer.params["phi_offset"]
            if isinstance(val, torch.Tensor):
                return val.detach()

    return None


def update_weights_for_target_flux(
    model: SOENModelCore,
    mean_states: dict[int, torch.Tensor],
    upstream_means: dict[str, torch.Tensor],
    config: FluxMatchingConfig,
    min_state_clamp: float = 0.01,
    alpha: float = 1.0,
    mode: Literal["node_wise", "connection_wise"] = "connection_wise",
    input_layer_ids: set[int] | None = None,
) -> dict[int, bool]:
    """Update weights to achieve target phi_total given observed upstream states.

    Automatically accounts for each destination layer's phi_offset:
        phi_exc_target[i] = phi_total_target[i] - phi_offset[i]

    Two modes are available:
    - node_wise: Compute uniform weight per destination node (per incoming connection edge)
    - connection_wise: Compute individual weights per synapse

    Allocation (the "num_sources vs fan_in" split):
    - incoming connection edges: count of distinct incoming weight matrices into a destination
      layer (e.g., J_0_to_4 and J_1_to_4 means 2 incoming edges)
    - fan_in[i]: per-edge fan-in for destination node i (row-sum of that edge's mask)

    The default behavior matches the original implementation:
      phi_exc_target_for_edge[i] = phi_exc_target[i] / incoming_edge_count_total
      J_edge[i,j] = phi_exc_target_for_edge[i] / (fan_in_edge[i] * state[j])

    Args:
        model: The SOEN model (modified in-place)
        mean_states: Mean state per layer [layer_id -> [dim]]
        upstream_means: Mean upstream state per connection/destination
        config: FluxMatchingConfig with target phi_total settings
        min_state_clamp: Minimum state value to avoid division by zero
        alpha: Step size (0=no update, 1=full update, 0.5=halfway)
        mode: Weight update strategy ('node_wise' or 'connection_wise')
        input_layer_ids: Set of layer IDs that are input layers (no incoming weights)

    Returns:
        Dict mapping layer_id -> whether it was updated
    """
    if alpha <= 0:
        return {}  # No update

    input_layer_ids = input_layer_ids or set()
    layers_updated: dict[int, bool] = {}

    # Count incoming connection edges per destination layer.
    # (This matches the previous num_sources behavior exactly.)
    incoming_edge_count_total: dict[int, int] = {}
    for conn in model.connections_config:
        incoming_edge_count_total[conn.to_layer] = incoming_edge_count_total.get(conn.to_layer, 0) + 1

    def _allocate_phi_exc_per_edge(
        phi_exc_target: torch.Tensor,
        *,
        to_layer: int,
        fan_in_by_key: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not fan_in_by_key:
            return {}

        if config.flux_allocation_mode == "equal_per_incoming_connection":
            denom = incoming_edge_count_total.get(to_layer, 1)
            # Preserve old behavior: split by total number of incoming edges into to_layer,
            # not by "how many edges we're updating".
            return {k: phi_exc_target / denom for k in fan_in_by_key}

        if config.flux_allocation_mode == "proportional_to_fan_in":
            # New mode: distribute per-node target across edges proportional to per-node fan-in.
            total_fan_in = torch.zeros_like(phi_exc_target)
            for fan_in in fan_in_by_key.values():
                total_fan_in = total_fan_in + fan_in
            total_fan_in = total_fan_in.clamp(min=1.0)
            return {k: phi_exc_target * (fan_in_by_key[k] / total_fan_in) for k in fan_in_by_key}

        raise ValueError(f"Unknown flux_allocation_mode: {config.flux_allocation_mode!r}")

    # Get device from model
    device = next(model.parameters()).device if list(model.parameters()) else "cpu"

    # Group by destination layer so the logic reads like the mental model:
    # - compute per-node phi_exc_target for the destination
    # - allocate that target across incoming edges
    # - compute target weights per edge using fan-in and observed states
    conns_by_to_layer: dict[int, list[Any]] = {}
    for conn in model.connections_config:
        conns_by_to_layer.setdefault(conn.to_layer, []).append(conn)

    for to_layer, incoming_conns in conns_by_to_layer.items():
        # Check if this layer should be processed
        is_input = to_layer in input_layer_ids
        if not config.should_process_layer(to_layer, is_input_layer=is_input):
            layers_updated[to_layer] = False
            continue

        # Get per-layer config (may have overrides)
        layer_cfg = config.get_layer_config(to_layer)
        effective_alpha = layer_cfg.alpha if layer_cfg.alpha is not None else alpha

        # Gather incoming edge data
        keys: list[str] = []
        masks: dict[str, torch.Tensor] = {}
        current_weights: dict[str, torch.Tensor] = {}
        fan_in_by_key: dict[str, torch.Tensor] = {}
        from_layer_by_key: dict[str, int] = {}

        for conn in incoming_conns:
            key = f"J_{conn.from_layer}_to_{to_layer}"

            # Check if this connection should be processed
            if not config.should_process_connection(key):
                continue

            if key not in model.connections:
                continue

            w = model.connections[key]
            mask = model.connection_masks.get(key)
            if mask is None:
                raise RuntimeError(
                    f"Missing connection mask for {key}. This is a bug: flux matching requires "
                    "a connectivity mask to compute fan-in and to update only active synapses."
                )

            w_detached = w.detach()
            keys.append(key)
            masks[key] = mask
            current_weights[key] = w_detached.clone()
            fan_in_by_key[key] = mask.sum(dim=1).clamp(min=1.0)
            from_layer_by_key[key] = conn.from_layer

        if not keys:
            layers_updated[to_layer] = False
            continue

        to_dim = masks[keys[0]].shape[0]

        # Get phi_offset from destination layer (may be per-node)
        phi_offset = _get_layer_phi_offset(model, to_layer)

        # Build a temporary config with this layer's settings for phi_exc calculation
        temp_config = FluxMatchingConfig(
            phi_total_target=layer_cfg.phi_total_target,
            phi_total_target_min=layer_cfg.phi_total_target_min,
            phi_total_target_max=layer_cfg.phi_total_target_max,
        )
        phi_exc_target = temp_config.get_phi_exc_target_per_node(to_dim, phi_offset, device)

        phi_exc_target_by_key = _allocate_phi_exc_per_edge(
            phi_exc_target, to_layer=to_layer, fan_in_by_key=fan_in_by_key
        )

        any_updated = False
        for key in keys:
            mask = masks[key]
            fan_in = fan_in_by_key[key]
            old_w = current_weights[key]
            from_layer = from_layer_by_key[key]
            phi_exc_target_for_edge = phi_exc_target_by_key[key]  # [to_dim]

            if mode == "connection_wise":
                if from_layer not in mean_states:
                    continue
                raw_states = mean_states[from_layer]  # [from_dim]
                source_states = raw_states.sign() * raw_states.abs().clamp(min=min_state_clamp)

                j_target_matrix = phi_exc_target_for_edge.unsqueeze(1) / (
                    fan_in.unsqueeze(1) * source_states.unsqueeze(0)
                )
                target_weights = j_target_matrix * mask

            else:  # node_wise
                if key in upstream_means:
                    raw_upstream = upstream_means[key]
                elif from_layer in mean_states:
                    raw_upstream = mean_states[from_layer].mean().expand(mask.shape[0])
                else:
                    continue

                mean_upstream = raw_upstream.sign() * raw_upstream.abs().clamp(min=min_state_clamp)
                j_target = phi_exc_target_for_edge / (fan_in * mean_upstream)
                target_weights = j_target.unsqueeze(1).expand_as(mask) * mask

            new_weights = old_w + effective_alpha * (target_weights - old_w)
            with torch.no_grad():
                model.connections[key].copy_(new_weights)
            any_updated = True

        layers_updated[to_layer] = any_updated

    return layers_updated


def apply_inhibitory_fraction_preserving_flux(
    model: SOENModelCore,
    mean_states: dict[int, torch.Tensor],
    *,
    config: FluxMatchingConfig | None = None,
    inhibitory_fraction: float = 0.2,
    include_connections: set[str] | None = None,
    seed: int | None = None,
    max_resamples: int = 10,
    min_scale: float = 1e-6,
    max_scale: float = 1e3,
) -> None:
    """Flip a fraction of synapses inhibitory while preserving each node's target flux.

    Motivation:
        In many settings, flux matching yields mostly-positive weights (especially if
        upstream mean states are mostly positive). If you want a controlled fraction of
        inhibitory synapses, you can post-process weights by randomly flipping some
        synapses negative, and compensating by scaling the remaining synapses so that
        each destination node keeps the same target external flux.

    What this does (per incoming connection edge and per destination node i):
        - Choose a random subset S- of active synapses (mask==1) with size
          floor(inhibitory_fraction * fan_in_i).
        - Flip those weights negative: J_ij <- -J_ij for j in S-
        - Scale the remaining active weights by a single factor a_i so that the
          external flux matches the edge's allocated target:

            phi_i^{ext,target,(e)} = sum_j J_ij^{new} * s_j

          with:
            J_ij^{new} = a_i * J_ij          for j in S+
            J_ij^{new} = - J_ij             for j in S-

          Solving for a_i:
            a_i = (phi_target + sum_{j in S-}(J_ij * s_j)) / sum_{j in S+}(J_ij * s_j)

    Requirements / caveats:
        - This requires the same kind of upstream state estimate used in flux matching
          (`mean_states`). If your states shift, re-run flux matching or recompute means.
        - This is fail-fast: if it cannot find a subset yielding a valid positive scale
          within [min_scale, max_scale], it raises.

    Args:
        model: The SOEN model (modified in-place).
        mean_states: Mean state per layer [layer_id -> [dim]].
        config: FluxMatchingConfig used to compute per-layer targets and edge allocation.
        inhibitory_fraction: Desired fraction of inhibitory synapses per destination node
            (0 <= p < 1). Applied per incoming edge.
        include_connections: Optional set of connection keys to modify (e.g., {"J_0_to_1"}).
            If None, all connections are eligible. Target flux allocation is still computed
            over all incoming edges so per-edge targets remain consistent.
        seed: Optional RNG seed for deterministic flips.
        max_resamples: Max attempts per destination node to find a flip subset that yields
            a valid scale factor.
        min_scale: Minimum allowed excitatory scale factor a_i.
        max_scale: Maximum allowed excitatory scale factor a_i.
    """

    config = config or FluxMatchingConfig()
    if not (0.0 <= inhibitory_fraction < 1.0):
        raise ValueError(f"inhibitory_fraction must be in [0, 1). Got {inhibitory_fraction}.")

    device = next(model.parameters()).device if list(model.parameters()) else "cpu"
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    # Count incoming connection edges per destination layer (matches flux-matching logic).
    incoming_edge_count_total: dict[int, int] = {}
    for conn in model.connections_config:
        incoming_edge_count_total[conn.to_layer] = incoming_edge_count_total.get(conn.to_layer, 0) + 1

    # Group by destination layer
    conns_by_to_layer: dict[int, list[Any]] = {}
    for conn in model.connections_config:
        conns_by_to_layer.setdefault(conn.to_layer, []).append(conn)

    def _allocate_phi_ext_target_per_edge(
        phi_ext_target: torch.Tensor,
        *,
        to_layer: int,
        fan_in_by_key: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not fan_in_by_key:
            return {}

        if config.flux_allocation_mode == "equal_per_incoming_connection":
            denom = incoming_edge_count_total.get(to_layer, 1)
            return {k: phi_ext_target / denom for k in fan_in_by_key}

        if config.flux_allocation_mode == "proportional_to_fan_in":
            total_fan_in = torch.zeros_like(phi_ext_target)
            for fan_in in fan_in_by_key.values():
                total_fan_in = total_fan_in + fan_in
            total_fan_in = total_fan_in.clamp(min=1.0)
            return {k: phi_ext_target * (fan_in_by_key[k] / total_fan_in) for k in fan_in_by_key}

        raise ValueError(f"Unknown flux_allocation_mode: {config.flux_allocation_mode!r}")

    for to_layer, incoming_conns in conns_by_to_layer.items():
        layer_cfg = config.get_layer_config(to_layer)

        # Build the per-node target external flux for this layer from phi_total targets and phi_offset.
        # Use the same mechanism as flux matching: compute phi_ext_target = phi_total_target - phi_offset.
        phi_offset = _get_layer_phi_offset(model, to_layer)
        temp_config = FluxMatchingConfig(
            phi_total_target=layer_cfg.phi_total_target,
            phi_total_target_min=layer_cfg.phi_total_target_min,
            phi_total_target_max=layer_cfg.phi_total_target_max,
        )

        # Determine to_dim from any available incoming connection mask
        to_dim = None
        for conn in incoming_conns:
            key = f"J_{conn.from_layer}_to_{to_layer}"
            mask = model.connection_masks.get(key)
            if mask is not None:
                to_dim = int(mask.shape[0])
                break
        if to_dim is None:
            continue

        phi_ext_target = temp_config.get_phi_exc_target_per_node(to_dim, phi_offset, device)

        # Gather per-edge fan-in for ALL incoming edges (needed for allocation).
        # We will only modify a subset (include_connections), but allocation should remain
        # consistent with the original flux-matching per-edge targets.
        fan_in_by_key_all: dict[str, torch.Tensor] = {}
        mask_by_key_all: dict[str, torch.Tensor] = {}
        from_layer_by_key_all: dict[str, int] = {}
        for conn in incoming_conns:
            key = f"J_{conn.from_layer}_to_{to_layer}"
            if key not in model.connections:
                continue
            mask = model.connection_masks.get(key)
            if mask is None:
                raise RuntimeError(
                    f"Missing connection mask for {key}. This is a bug: inhibitory post-processing "
                    "requires a connectivity mask."
                )
            mask_by_key_all[key] = mask
            fan_in_by_key_all[key] = mask.sum(dim=1).clamp(min=1.0)
            from_layer_by_key_all[key] = conn.from_layer

        phi_ext_target_by_key_all = _allocate_phi_ext_target_per_edge(
            phi_ext_target, to_layer=to_layer, fan_in_by_key=fan_in_by_key_all
        )

        for key, mask in mask_by_key_all.items():
            if include_connections is not None and key not in include_connections:
                continue

            from_layer = from_layer_by_key_all[key]
            if from_layer not in mean_states:
                continue

            raw_states = mean_states[from_layer]  # [from_dim]
            # Match flux-matching behavior: preserve sign but clamp abs away from zero.
            source_states = raw_states.sign() * raw_states.abs().clamp(min=config.min_state_clamp)

            weights = model.connections[key]
            phi_target_edge = phi_ext_target_by_key_all[key]  # [to_dim]

            # Operate per destination node so we preserve each node's target flux.
            with torch.no_grad():
                for i in range(mask.shape[0]):
                    active_idx = torch.nonzero(mask[i], as_tuple=False).flatten()
                    fan_in_i = int(active_idx.numel())
                    if fan_in_i <= 0:
                        continue

                    n_flip = int(fan_in_i * inhibitory_fraction)
                    if n_flip <= 0:
                        continue
                    if n_flip >= fan_in_i:
                        raise ValueError(
                            f"inhibitory_fraction={inhibitory_fraction} implies flipping all synapses "
                            f"for node {i} in {key} (fan_in={fan_in_i}). This would require infinite scaling."
                        )

                    w_row = weights[i, active_idx]
                    s_row = source_states[active_idx]
                    target_i = phi_target_edge[i].item()

                    # Try a few random subsets to get a valid positive scale factor.
                    scale_i: float | None = None
                    flip_idx: torch.Tensor | None = None
                    for _ in range(max_resamples):
                        perm = torch.randperm(fan_in_i, generator=gen)
                        neg_sel = perm[:n_flip]
                        pos_sel = perm[n_flip:]

                        # Contributions with current weights
                        contrib = w_row * s_row
                        sum_neg = contrib[neg_sel].sum().item()
                        sum_pos = contrib[pos_sel].sum().item()

                        if abs(sum_pos) < 1e-12:
                            continue

                        # a = (phi_target + sum_neg) / sum_pos
                        a = (target_i + sum_neg) / sum_pos
                        if a <= 0:
                            continue
                        if a < min_scale or a > max_scale:
                            continue

                        scale_i = float(a)
                        flip_idx = active_idx[neg_sel]
                        break

                    if scale_i is None or flip_idx is None:
                        raise RuntimeError(
                            f"Could not find a valid inhibitory flip subset for node {i} in {key} "
                            f"with inhibitory_fraction={inhibitory_fraction}. "
                            f"Try reducing inhibitory_fraction or loosening scale bounds."
                        )

                    # Apply: flip inhibitory synapses, scale excitatory synapses
                    # We ONLY scale the positive synapses (pos_sel), and we ONLY flip
                    # the negative ones (neg_sel).
                    pos_idx = active_idx[pos_sel]
                    neg_idx = active_idx[neg_sel]

                    weights[i, pos_idx] = weights[i, pos_idx] * scale_i
                    weights[i, neg_idx] = -weights[i, neg_idx]


def _compute_phi_total_per_layer(
    model: SOENModelCore,
    total_flux: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Compute phi_total (phi_exc + phi_offset) for each layer.

    Args:
        model: The SOEN model
        total_flux: External flux per node for each layer

    Returns:
        Dict mapping layer_id -> phi_total per node [dim]
    """
    phi_total = {}
    for layer_id, phi_exc in total_flux.items():
        phi_offset = _get_layer_phi_offset(model, layer_id)
        if phi_offset is not None:
            phi_total[layer_id] = phi_exc + phi_offset.to(phi_exc.device)
        else:
            phi_total[layer_id] = phi_exc
    return phi_total


def run_flux_matching_iterations(
    model: SOENModelCore,
    data_batches: list[torch.Tensor],
    config: FluxMatchingConfig | None = None,
) -> FluxMatchingResult:
    """Run iterative flux-matching weight initialization.

    Targets phi_total (the actual operating point for the source function),
    automatically accounting for each layer's phi_offset.

    Layer filtering is controlled by config.include_layers and config.exclude_layers.
    Per-layer settings can be specified via config.layer_overrides.

    Args:
        model: The SOEN model to initialize
        data_batches: List of input tensors [batch, seq_len, input_dim]
        config: Configuration options (including layer filtering)

    Returns:
        FluxMatchingResult with iteration statistics
    """
    config = config or FluxMatchingConfig()
    result = FluxMatchingResult()

    # Set up logging function
    log_fn = config.log_fn if config.log_fn else _default_log_fn

    def log(msg: str) -> None:
        if config.verbose:
            log_fn(msg)

    # Identify input layers (layers with no incoming connections)
    layers_with_inputs = {conn.to_layer for conn in model.connections_config}
    all_layer_ids = {cfg.layer_id for cfg in model.layers_config}
    input_layer_ids = all_layer_ids - layers_with_inputs

    # Build list of layers that will be processed
    processed_layers = []
    skipped_layers = []
    for layer_id in sorted(all_layer_ids):
        is_input = layer_id in input_layer_ids
        if config.should_process_layer(layer_id, is_input_layer=is_input):
            processed_layers.append(layer_id)
        else:
            skipped_layers.append(layer_id)

    log(f"Starting flux-matching initialization ({config.num_iterations} iterations)")
    if config.phi_total_target_min is not None and config.phi_total_target_max is not None:
        log(f"  Target phi_total range: {config.phi_total_target_min} to {config.phi_total_target_max}")
        log("  (uniformly distributed across nodes for symmetry breaking)")
    else:
        log(f"  Target phi_total: {config.phi_total_target}")
    log("  (phi_exc computed per node based on each node's phi_offset)")
    log(f"  Mode: {config.weight_update_mode}")
    log(f"  Step size (alpha): {config.alpha}")
    log(f"  Using {len(data_batches)} batches")

    # Log layer filtering info
    if input_layer_ids:
        log(f"  Input layers (skipped): {sorted(input_layer_ids)}")
    if skipped_layers:
        non_input_skipped = [lid for lid in skipped_layers if lid not in input_layer_ids]
        if non_input_skipped:
            log(f"  Excluded layers: {non_input_skipped}")
    if processed_layers:
        log(f"  Processing layers: {processed_layers}")
    if config.layer_overrides:
        log(f"  Per-layer overrides: {list(config.layer_overrides.keys())}")
    if config.exclude_connections:
        log(f"  Excluded connections: {config.exclude_connections}")
    log("")

    for iteration in range(config.num_iterations):
        # Check for cancellation at start of each iteration
        if config.stop_check is not None and config.stop_check():
            log("Flux matching cancelled by user.")
            result.converged = False
            return result

        # Collect states from all batches
        all_layer_states: dict[int, list[torch.Tensor]] = {}

        for batch_idx, batch in enumerate(data_batches):
            # Check for cancellation periodically during batch processing
            if batch_idx % 50 == 0 and config.stop_check is not None and config.stop_check():
                log("Flux matching cancelled by user.")
                result.converged = False
                return result

            batch_states = collect_layer_states(model, batch)
            for layer_id, states in batch_states.items():
                if layer_id not in all_layer_states:
                    all_layer_states[layer_id] = []
                all_layer_states[layer_id].append(states)

        # Concatenate and compute mean states
        concat_states = {
            layer_id: torch.cat(states_list, dim=0)
            for layer_id, states_list in all_layer_states.items()
        }
        mean_states = compute_mean_states(concat_states)

        # Compute upstream means and current flux (phi_exc)
        upstream_means = compute_upstream_mean_states(model, mean_states)
        flux_per_conn = compute_flux_per_node(model, mean_states)
        total_flux_exc = compute_total_flux_per_layer(model, flux_per_conn)

        # Compute phi_total = phi_exc + phi_offset
        total_flux = _compute_phi_total_per_layer(model, total_flux_exc)

        # Collect iteration statistics
        iter_stats = {
            "iteration": iteration,
            "mean_states": {k: v.mean().item() for k, v in mean_states.items()},
            "flux_per_layer": {},  # This is now phi_total
            "flux_exc_per_layer": {},  # Store phi_exc separately
            "flux_error_per_layer": {},
            "layers_updated": {},
        }

        for layer_id, phi_total_layer in total_flux.items():
            # Get per-layer config for expected target
            layer_cfg = config.get_layer_config(layer_id)
            if layer_cfg.phi_total_target_min is not None and layer_cfg.phi_total_target_max is not None:
                expected_target = (layer_cfg.phi_total_target_min + layer_cfg.phi_total_target_max) / 2
            else:
                expected_target = layer_cfg.phi_total_target

            mean_flux = phi_total_layer.mean().item()
            std_flux = phi_total_layer.std().item()
            error = abs(mean_flux - expected_target)
            iter_stats["flux_per_layer"][layer_id] = {
                "mean": mean_flux,
                "std": std_flux,
                "min": phi_total_layer.min().item(),
                "max": phi_total_layer.max().item(),
            }
            # Also store phi_exc stats
            if layer_id in total_flux_exc:
                phi_exc = total_flux_exc[layer_id]
                iter_stats["flux_exc_per_layer"][layer_id] = {
                    "mean": phi_exc.mean().item(),
                    "std": phi_exc.std().item(),
                    "min": phi_exc.min().item(),
                    "max": phi_exc.max().item(),
                }
            iter_stats["flux_error_per_layer"][layer_id] = error

        result.iteration_stats.append(iter_stats)

        log(f"Iteration {iteration + 1}/{config.num_iterations}:")
        for layer_id in sorted(mean_states.keys()):
            mean_s = mean_states[layer_id].mean().item()
            is_input = layer_id in input_layer_ids
            will_update = config.should_process_layer(layer_id, is_input_layer=is_input)

            if layer_id in total_flux:
                flux_info = iter_stats["flux_per_layer"][layer_id]
                layer_cfg = config.get_layer_config(layer_id)

                # Format target string for this layer
                if layer_cfg.phi_total_target_min is not None and layer_cfg.phi_total_target_max is not None:
                    target_str = f"target={layer_cfg.phi_total_target_min:.2f}-{layer_cfg.phi_total_target_max:.2f}"
                else:
                    target_str = f"target={layer_cfg.phi_total_target}"

                status = "" if will_update else " [SKIPPED]"

                # Show min/max when using a target range to verify symmetry breaking
                if layer_cfg.phi_total_target_min is not None and layer_cfg.phi_total_target_max is not None:
                    log(
                        f"  Layer {layer_id}: mean_s={mean_s:.4f}, "
                        f"phi_total=[{flux_info['min']:.3f}, {flux_info['max']:.3f}] "
                        f"mean={flux_info['mean']:.4f} ({target_str}){status}"
                    )
                else:
                    log(
                        f"  Layer {layer_id}: mean_s={mean_s:.4f}, "
                        f"phi_total={flux_info['mean']:.4f}+/-{flux_info['std']:.4f} "
                        f"({target_str}){status}"
                    )
            else:
                log(f"  Layer {layer_id}: mean_s={mean_s:.4f} (input layer)")

        # Update weights based on observed states (except for last iteration)
        if iteration < config.num_iterations - 1:
            layers_updated = update_weights_for_target_flux(
                model,
                mean_states,
                upstream_means,
                config=config,
                min_state_clamp=config.min_state_clamp,
                alpha=config.alpha,
                mode=config.weight_update_mode,
                input_layer_ids=input_layer_ids,
            )
            iter_stats["layers_updated"] = layers_updated

        log("")

    # Store final results
    result.final_mean_states = mean_states
    result.final_flux_per_node = flux_per_conn

    # Check convergence only for processed layers
    final_errors = [
        result.iteration_stats[-1]["flux_error_per_layer"].get(lid, 0)
        for lid in processed_layers
    ]
    result.converged = all(e < 0.02 for e in final_errors) if final_errors else True  # 2% tolerance

    if result.converged:
        log("Flux-matching CONVERGED to target phi_total!")
    else:
        log(f"Final phi_total errors: {final_errors}")

    return result


def flux_matching_from_hdf5(
    model: SOENModelCore,
    hdf5_path: str | Path,
    split: str = "train",
    config: FluxMatchingConfig | None = None,
    seq_len: int | None = None,
    feature_min: float | None = None,
    feature_max: float | None = None,
) -> FluxMatchingResult:
    """Convenience function to run flux-matching from HDF5 data.

    Args:
        model: The SOEN model to initialize
        hdf5_path: Path to HDF5 dataset
        split: Dataset split to use
        config: Configuration options
        seq_len: Truncate/pad sequences to this length
        feature_min: Scale features to have this minimum value
        feature_max: Scale features to have this maximum value

    Returns:
        FluxMatchingResult with iteration statistics
    """
    config = config or FluxMatchingConfig()

    # Load data
    device = next(model.parameters()).device if list(model.parameters()) else "cpu"
    batches = load_hdf5_batches(
        hdf5_path,
        split=split,
        batch_size=config.batch_size,
        num_batches=config.num_batches,
        device=device,
        seq_len=seq_len,
        feature_min=feature_min,
        feature_max=feature_max,
    )

    return run_flux_matching_iterations(model, batches, config)


# =============================================================================
# High-Level API - Simple one-liner functions
# =============================================================================


def initialize_model_with_flux_matching(
    model: SOENModelCore,
    data: torch.Tensor | list[torch.Tensor] | str | Path,
    *,
    phi_target: float = 0.5,
    phi_target_range: tuple[float, float] | None = None,
    iterations: int = 5,
    mode: Literal["node_wise", "connection_wise"] = "connection_wise",
    alpha: float = 1.0,
    exclude_layers: set[int] | list[int] | None = None,
    include_layers: set[int] | list[int] | None = None,
    layer_targets: dict[int, float | tuple[float, float]] | None = None,
    batch_size: int = 32,
    num_batches: int | None = None,
    verbose: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> FluxMatchingResult:
    """Initialize a SOEN model's weights using iterative flux matching.

    This is the simplest way to apply flux matching. The algorithm adjusts
    connection weights so that each neuron's total input flux (phi_total)
    converges to the target value (0.5 is optimal for the rate-based source).

    Example usage:
        >>> from soen_toolkit.utils.flux_matching_init import initialize_model_with_flux_matching

        # Simple case - use all defaults
        >>> result = initialize_model_with_flux_matching(model, data_tensor)

        # With HDF5 data
        >>> result = initialize_model_with_flux_matching(model, "data.h5")

        # Exclude specific layers
        >>> result = initialize_model_with_flux_matching(
        ...     model, data,
        ...     exclude_layers={1, 3},  # Skip layers 1 and 3
        ...     phi_target=0.45,
        ... )

        # Different targets per layer
        >>> result = initialize_model_with_flux_matching(
        ...     model, data,
        ...     layer_targets={
        ...         2: 0.4,  # Layer 2 targets 0.4
        ...         3: (0.3, 0.5),  # Layer 3 uses symmetry breaking
        ...     }
        ... )

    Args:
        model: The SOEN model to initialize (modified in-place)
        data: Input data. Can be:
            - torch.Tensor: shape [batch, seq_len, input_dim] or [seq_len, input_dim]
            - list[torch.Tensor]: list of batches
            - str/Path: path to HDF5 file (will load from 'train' split)
        phi_target: Target phi_total value for all layers (default 0.5, optimal point)
        phi_target_range: Tuple (min, max) for symmetry breaking across nodes.
            If specified, overrides phi_target.
        iterations: Number of flux-matching iterations (default 5)
        mode: Weight update strategy:
            - "connection_wise": Individual weight per connection (default, finer control)
            - "node_wise": Uniform weight per destination node (faster)
        alpha: Step size for updates (0-1, default 1.0 = full correction)
        exclude_layers: Set/list of layer IDs to skip
        include_layers: Set/list of layer IDs to process (if None, process all)
        layer_targets: Dict mapping layer_id to target. Values can be:
            - float: single target value
            - tuple[float, float]: (min, max) range for symmetry breaking
        batch_size: Batch size for processing (default 32)
        num_batches: Limit number of batches (None = use all)
        verbose: Print progress (default True)
        log_fn: Custom logging function (default: print to stdout)

    Returns:
        FluxMatchingResult with iteration statistics and convergence status

    Raises:
        ValueError: If data cannot be processed
    """
    # Build layer overrides from layer_targets
    layer_overrides: dict[int, LayerFluxConfig] | None = None
    if layer_targets:
        layer_overrides = {}
        for layer_id, target in layer_targets.items():
            if isinstance(target, tuple):
                layer_overrides[layer_id] = LayerFluxConfig(
                    phi_total_target_min=target[0],
                    phi_total_target_max=target[1],
                )
            else:
                layer_overrides[layer_id] = LayerFluxConfig(
                    phi_total_target=target,
                )

    # Build configuration
    config = FluxMatchingConfig(
        phi_total_target=phi_target,
        phi_total_target_min=phi_target_range[0] if phi_target_range else None,
        phi_total_target_max=phi_target_range[1] if phi_target_range else None,
        num_iterations=iterations,
        batch_size=batch_size,
        num_batches=num_batches,
        alpha=alpha,
        weight_update_mode=mode,
        verbose=verbose,
        log_fn=log_fn,
        exclude_layers=set(exclude_layers) if exclude_layers else None,
        include_layers=set(include_layers) if include_layers else None,
        layer_overrides=layer_overrides,
    )

    # Handle different data types
    if isinstance(data, (str, Path)):
        # HDF5 file path
        return flux_matching_from_hdf5(model, data, split="train", config=config)

    if isinstance(data, torch.Tensor):
        # Single tensor - ensure it's batched
        if data.ndim == 2:
            # [seq_len, input_dim] -> [1, seq_len, input_dim]
            data = data.unsqueeze(0)
        # Split into batches
        n_samples = data.shape[0]
        batches = []
        for i in range(0, n_samples, batch_size):
            batches.append(data[i:i + batch_size])
        if num_batches is not None:
            batches = batches[:num_batches]
    elif isinstance(data, list):
        # Already a list of batches
        batches = data
        if num_batches is not None:
            batches = batches[:num_batches]
    else:
        msg = f"Unsupported data type: {type(data)}. Expected Tensor, list of Tensors, or path to HDF5."
        raise ValueError(msg)

    if not batches:
        msg = "No data batches to process"
        raise ValueError(msg)

    return run_flux_matching_iterations(model, batches, config)


# Convenience alias
flux_match = initialize_model_with_flux_matching

