"""Gradient flow computation and visualization utilities.

This module provides functionality to compute gradients w.r.t. connection weights
and map them to colors for visualization in the network graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from soen_toolkit.core import SOENModelCore


@dataclass
class GradientFlowConfig:
    """Configuration for gradient flow computation."""

    # Dataset settings
    hdf5_path: str = ""
    split: str = "train"
    seq_len: int = 100
    batch_size: int = 32
    num_batches: int = 1

    # Time settings
    dt: float = 37.0
    total_time_ns: float | None = None  # If set, overrides seq_len

    # Feature scaling
    feature_min: float | None = None
    feature_max: float | None = None

    # Class filtering (for classification datasets)
    use_class_filter: bool = False
    class_id: int | None = None
    sample_index: int = 0

    # Loss settings
    loss_fn: str = "mse"  # 'mse', 'cross_entropy', 'sum_output'

    # Aggregation settings
    time_agg: str = "mean"  # 'mean', 'min', 'max', 'abs_max'
    batch_agg: str = "mean"  # 'mean', 'min', 'max', 'abs_max'

    # Display settings
    log_scale: bool = False
    colormap: str = "RdBu_r"  # Diverging colormap for signed gradients

    # Computed gradients cache - connection gradients (dL/dJ)
    gradients: dict[str, np.ndarray] = field(default_factory=dict)
    grad_min: float = 0.0
    grad_max: float = 0.0

    # Activation gradients cache - per-layer gradients (dL/ds)
    activation_gradients: dict[int, np.ndarray] = field(default_factory=dict)
    activation_grad_min: float = 0.0
    activation_grad_max: float = 0.0

    def is_configured(self) -> bool:
        """Check if the config has valid dataset path."""
        return bool(self.hdf5_path) and Path(self.hdf5_path).exists()


class GradientFlowError(Exception):
    """Raised when gradient computation fails."""


class GradientComputationCancelled(Exception):
    """Raised when gradient computation is cancelled by user."""


def compute_connection_gradients(
    model: SOENModelCore,
    config: GradientFlowConfig,
    *,
    progress_callback: callable | None = None,
) -> dict[str, np.ndarray] | None:
    """Compute gradients w.r.t. connection weights from dataset.

    Args:
        model: The SOEN model to analyze
        config: Gradient flow configuration
        progress_callback: Optional callback(current, total, message) for progress.
            The callback should raise GradientComputationCancelled to cancel.

    Returns:
        Dictionary mapping connection key (e.g. 'J_0_to_1') to gradient array
        of same shape as the weight matrix. Returns None if cancelled.

    Raises:
        GradientFlowError: If computation fails
    """
    if not config.is_configured():
        raise GradientFlowError("Dataset path not configured or file does not exist")

    # Import dataset loader
    try:
        from soen_toolkit.training.data.dataloaders import GenericHDF5Dataset
    except ImportError as e:
        raise GradientFlowError("GenericHDF5Dataset not available") from e

    # Determine device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # Load dataset
    try:
        dataset = GenericHDF5Dataset(
            hdf5_path=config.hdf5_path,
            split=config.split if config.split else None,
            cache_in_memory=False,
            target_seq_len=config.seq_len,
            scale_min=config.feature_min,
            scale_max=config.feature_max,
        )
    except Exception as e:
        raise GradientFlowError(f"Failed to load dataset: {e}") from e

    if len(dataset) == 0:
        raise GradientFlowError("Dataset is empty")

    # Apply class filter and sample selection
    from torch.utils.data import DataLoader, Subset

    effective_dataset = dataset
    if config.use_class_filter and config.class_id is not None:
        # Filter to only include samples from the specified class
        # Read labels directly from HDF5 file (GenericHDF5Dataset doesn't expose them)
        try:
            from soen_toolkit.training.data.dataloaders import open_hdf5_with_consistent_locking

            with open_hdf5_with_consistent_locking(config.hdf5_path) as f:
                if config.split:
                    labels_ds = f[config.split]["labels"]
                else:
                    labels_ds = f["labels"]
                labels = labels_ds[:]

            class_indices = [i for i, lbl in enumerate(labels) if int(lbl) == config.class_id]
            if not class_indices:
                raise GradientFlowError(f"No samples found for class {config.class_id}")

            # Select specific sample within class if sample_index is specified
            sample_idx = min(config.sample_index, len(class_indices) - 1)
            selected_indices = [class_indices[sample_idx]]
            effective_dataset = Subset(dataset, selected_indices)
        except GradientFlowError:
            raise
        except Exception as e:
            raise GradientFlowError(f"Class filtering failed: {e}") from e
    elif config.sample_index > 0:
        # No class filter but specific sample index requested
        sample_idx = min(config.sample_index, len(dataset) - 1)
        effective_dataset = Subset(dataset, [sample_idx])

    # Create dataloader
    loader = DataLoader(
        effective_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Prepare model
    model.eval()
    model.zero_grad(set_to_none=True)

    # Accumulate gradients across batches
    accumulated_grads: dict[str, list[torch.Tensor]] = {}
    accumulated_activation_grads: dict[int, list[torch.Tensor]] = {}  # Per-layer activation grads
    batches_processed = 0
    total_batches = min(config.num_batches, len(loader))

    # Get layer IDs for activation gradient tracking
    layer_ids = [cfg.layer_id for cfg in model.layers_config]

    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= config.num_batches:
            break

        if progress_callback:
            progress_callback(batch_idx, total_batches, f"Processing batch {batch_idx + 1}/{total_batches}")

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Reset model state
        try:
            model.reset_stateful_components()
        except Exception:
            pass

        model.zero_grad(set_to_none=True)

        # Forward pass - get all layer states
        try:
            outputs, all_states = model(inputs)
        except Exception as e:
            raise GradientFlowError(f"Forward pass failed: {e}") from e

        # Compute loss from final output (last timestep)
        final_output = outputs[:, -1, :]
        loss = _compute_loss(final_output, targets, config.loss_fn)

        # Backward pass for connection gradients
        try:
            loss.backward(retain_graph=True)
        except Exception as e:
            raise GradientFlowError(f"Backward pass failed: {e}") from e

        # Extract gradients from connection parameters
        for key, param in model.connections.items():
            if param.grad is not None:
                grad = param.grad.detach().clone()
                if key not in accumulated_grads:
                    accumulated_grads[key] = []
                accumulated_grads[key].append(grad)

        # Also check for internal connectivity in layers
        for layer_idx, layer in enumerate(model.layers):
            layer_id = model.layers_config[layer_idx].layer_id
            if hasattr(layer, "connectivity") and layer.connectivity is not None:
                weight_param = layer.connectivity.weight
                if weight_param.grad is not None:
                    key = f"J_{layer_id}_to_{layer_id}"
                    grad = weight_param.grad.detach().clone()
                    if key not in accumulated_grads:
                        accumulated_grads[key] = []
                    accumulated_grads[key].append(grad)

        # Compute activation gradients (dL/ds) for each layer using autograd.grad
        # This gives us the gradient of loss w.r.t. each layer's activations
        for layer_idx, state in enumerate(all_states):
            if state.requires_grad or state.grad_fn is not None:
                try:
                    # Compute gradient of loss w.r.t. this layer's states
                    (state_grad,) = torch.autograd.grad(
                        loss,
                        state,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    if state_grad is not None:
                        # state_grad is [batch, seq_len, dim]
                        grad = state_grad.detach()
                        # Aggregate over time dimension
                        grad_time_agg = _aggregate_tensor(grad, config.time_agg, dim=1)  # [batch, dim]
                        # Mean over batch
                        grad_mean = grad_time_agg.mean(dim=0)  # [dim]

                        layer_id = layer_ids[layer_idx] if layer_idx < len(layer_ids) else layer_idx
                        if layer_id not in accumulated_activation_grads:
                            accumulated_activation_grads[layer_id] = []
                        accumulated_activation_grads[layer_id].append(grad_mean)
                except RuntimeError:
                    # Skip if gradient computation fails for this layer
                    pass

        batches_processed += 1

    if batches_processed == 0:
        raise GradientFlowError("No batches were processed")

    # Aggregate connection gradients across batches
    result = {}
    for key, grad_list in accumulated_grads.items():
        if not grad_list:
            continue

        # Stack gradients: [num_batches, *weight_shape]
        stacked = torch.stack(grad_list, dim=0)

        # Aggregate over batch dimension
        aggregated = _aggregate_tensor(stacked, config.batch_agg, dim=0)

        result[key] = aggregated.cpu().numpy()

    # Store connection gradients in config
    config.gradients = result

    # Compute global min/max for connection gradient colormap
    if result:
        all_vals = np.concatenate([g.flatten() for g in result.values()])
        config.grad_min = float(np.min(all_vals))
        config.grad_max = float(np.max(all_vals))
    else:
        config.grad_min = 0.0
        config.grad_max = 0.0

    # Aggregate activation gradients across batches
    activation_result = {}
    for layer_id, grad_list in accumulated_activation_grads.items():
        if not grad_list:
            continue

        # Stack gradients: [num_batches, dim]
        stacked = torch.stack(grad_list, dim=0)

        # Aggregate over batch dimension
        aggregated = _aggregate_tensor(stacked, config.batch_agg, dim=0)

        activation_result[layer_id] = aggregated.cpu().numpy()

    # Store activation gradients in config
    config.activation_gradients = activation_result

    # Compute global min/max for activation gradient colormap
    if activation_result:
        all_vals = np.concatenate([g.flatten() for g in activation_result.values()])
        config.activation_grad_min = float(np.min(all_vals))
        config.activation_grad_max = float(np.max(all_vals))
    else:
        config.activation_grad_min = 0.0
        config.activation_grad_max = 0.0

    if progress_callback:
        progress_callback(total_batches, total_batches, "Gradient computation complete")

    return result


def _compute_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: str,
) -> torch.Tensor:
    """Compute loss based on specified function.

    Args:
        outputs: Model outputs [batch, seq_len, dim] or [batch, dim]
        targets: Target tensor
        loss_fn: Loss function name

    Returns:
        Scalar loss tensor
    """
    # Reduce outputs to final timestep if 3D
    if outputs.dim() == 3:
        outputs = outputs[:, -1, :]  # [batch, dim]

    if loss_fn == "sum_output":
        # No target needed - just sum outputs to create gradient flow
        return outputs.sum()

    if loss_fn == "mse":
        # MSE loss
        if targets.dim() == 1:
            # Classification targets - convert to one-hot
            if outputs.shape[-1] > 1:
                targets = F.one_hot(targets.long(), num_classes=outputs.shape[-1]).float()
            else:
                targets = targets.float().unsqueeze(-1)
        return F.mse_loss(outputs, targets.float())

    if loss_fn == "cross_entropy":
        # Cross-entropy loss
        if targets.dim() > 1:
            targets = targets.argmax(dim=-1)
        return F.cross_entropy(outputs, targets.long())

    # Default to MSE
    return F.mse_loss(outputs, targets.float())


def _aggregate_tensor(tensor: torch.Tensor, method: str, dim: int) -> torch.Tensor:
    """Aggregate tensor along specified dimension.

    Args:
        tensor: Input tensor
        method: Aggregation method ('mean', 'min', 'max', 'abs_max')
        dim: Dimension to aggregate over

    Returns:
        Aggregated tensor
    """
    if method == "mean":
        return tensor.mean(dim=dim)
    if method == "min":
        return tensor.min(dim=dim).values
    if method == "max":
        return tensor.max(dim=dim).values
    if method == "abs_max":
        # Return value with maximum absolute magnitude, preserving sign
        abs_tensor = tensor.abs()
        max_indices = abs_tensor.argmax(dim=dim, keepdim=True)
        return tensor.gather(dim, max_indices).squeeze(dim)

    # Default to mean
    return tensor.mean(dim=dim)


def gradient_to_color(
    value: float,
    vmin: float,
    vmax: float,
    *,
    colormap: str = "RdBu_r",
    log_scale: bool = False,
) -> str:
    """Convert gradient value to hex color.

    Args:
        value: Gradient value
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        colormap: Matplotlib colormap name
        log_scale: Apply log scaling

    Returns:
        Hex color string (e.g. '#ff0000')
    """
    import matplotlib.pyplot as plt

    if log_scale:
        # Apply signed log transform: sign(x) * log10(1 + |x|)
        value = np.sign(value) * np.log10(1 + np.abs(value))
        if vmin != 0:
            vmin = np.sign(vmin) * np.log10(1 + np.abs(vmin))
        if vmax != 0:
            vmax = np.sign(vmax) * np.log10(1 + np.abs(vmax))

    # Normalize to [0, 1]
    if abs(vmax - vmin) < 1e-10:
        norm = 0.5  # All values equal - use middle of colormap
    else:
        norm = (value - vmin) / (vmax - vmin)
        norm = max(0.0, min(1.0, norm))

    # Get colormap and convert to hex
    try:
        cmap = plt.get_cmap(colormap)
        rgba = cmap(norm)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        # Fallback to gray
        gray = int(norm * 255)
        return f"#{gray:02x}{gray:02x}{gray:02x}"


def get_layer_pair_gradient(
    gradients: dict[str, np.ndarray],
    from_layer: int,
    to_layer: int,
    *,
    aggregation: str = "mean",
) -> float:
    """Get aggregated gradient value for a layer pair (simple view).

    Args:
        gradients: Dict of gradient arrays from compute_connection_gradients
        from_layer: Source layer ID
        to_layer: Target layer ID
        aggregation: How to aggregate ('mean', 'abs_mean', 'max', 'abs_max')

    Returns:
        Scalar gradient value for the layer pair
    """
    # Try different key formats
    possible_keys = [
        f"J_{from_layer}_to_{to_layer}",
        f"{from_layer}_{to_layer}",
        f"internal_{to_layer}" if from_layer == to_layer else None,
    ]

    grad_array = None
    for key in possible_keys:
        if key and key in gradients:
            grad_array = gradients[key]
            break

    if grad_array is None:
        return 0.0

    # Aggregate to scalar
    if aggregation == "mean":
        return float(np.mean(grad_array))
    if aggregation == "abs_mean":
        return float(np.mean(np.abs(grad_array)))
    if aggregation == "max":
        return float(np.max(grad_array))
    if aggregation == "abs_max":
        # Return value with max absolute magnitude, preserving sign
        flat = grad_array.flatten()
        if len(flat) == 0:
            return 0.0
        idx = np.argmax(np.abs(flat))
        return float(flat[idx])

    return float(np.mean(grad_array))


def get_layer_activation_gradient(
    activation_gradients: dict[int, np.ndarray],
    layer_id: int,
    *,
    aggregation: str = "abs_mean",
) -> float:
    """Get aggregated activation gradient for a layer (simple view).

    Args:
        activation_gradients: Dict of per-layer activation gradient arrays
        layer_id: Layer ID
        aggregation: How to aggregate ('mean', 'abs_mean', 'max', 'abs_max')

    Returns:
        Scalar gradient value for the layer
    """
    if layer_id not in activation_gradients:
        return 0.0

    grad_array = activation_gradients[layer_id]

    if aggregation == "mean":
        return float(np.mean(grad_array))
    if aggregation == "abs_mean":
        return float(np.mean(np.abs(grad_array)))
    if aggregation == "max":
        return float(np.max(grad_array))
    if aggregation == "abs_max":
        flat = grad_array.flatten()
        if len(flat) == 0:
            return 0.0
        idx = np.argmax(np.abs(flat))
        return float(flat[idx])

    return float(np.mean(np.abs(grad_array)))


def get_neuron_activation_gradient(
    activation_gradients: dict[int, np.ndarray],
    layer_id: int,
    neuron_idx: int,
) -> float:
    """Get activation gradient for a specific neuron (detailed view).

    Args:
        activation_gradients: Dict of per-layer activation gradient arrays
        layer_id: Layer ID
        neuron_idx: Neuron index within the layer

    Returns:
        Gradient value for the specific neuron
    """
    if layer_id not in activation_gradients:
        return 0.0

    grad_array = activation_gradients[layer_id]

    try:
        return float(grad_array[neuron_idx])
    except (IndexError, TypeError):
        return 0.0


def get_connection_gradient(
    gradients: dict[str, np.ndarray],
    from_layer: int,
    to_layer: int,
    from_neuron: int,
    to_neuron: int,
) -> float:
    """Get gradient value for a specific connection (detailed view).

    Args:
        gradients: Dict of gradient arrays from compute_connection_gradients
        from_layer: Source layer ID
        to_layer: Target layer ID
        from_neuron: Source neuron index within layer
        to_neuron: Target neuron index within layer

    Returns:
        Gradient value for the specific connection
    """
    # Try different key formats
    possible_keys = [
        f"J_{from_layer}_to_{to_layer}",
        f"{from_layer}_{to_layer}",
        f"internal_{to_layer}" if from_layer == to_layer else None,
    ]

    grad_array = None
    for key in possible_keys:
        if key and key in gradients:
            grad_array = gradients[key]
            break

    if grad_array is None:
        return 0.0

    # Weight matrix is [to_dim, from_dim]
    try:
        return float(grad_array[to_neuron, from_neuron])
    except (IndexError, TypeError):
        return 0.0


def get_gradient_stats_summary(config: GradientFlowConfig) -> dict:
    """Get summary statistics of computed gradients.

    Args:
        config: Configuration with computed gradients

    Returns:
        Dictionary with gradient statistics
    """
    if not config.gradients:
        return {"computed": False}

    all_vals = np.concatenate([g.flatten() for g in config.gradients.values()])

    return {
        "computed": True,
        "num_connections": len(config.gradients),
        "total_params": len(all_vals),
        "min": float(np.min(all_vals)),
        "max": float(np.max(all_vals)),
        "mean": float(np.mean(all_vals)),
        "std": float(np.std(all_vals)),
        "abs_mean": float(np.mean(np.abs(all_vals))),
        "nonzero_frac": float(np.count_nonzero(all_vals) / len(all_vals)),
    }

