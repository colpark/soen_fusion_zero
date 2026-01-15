"""Color computation utilities for visualization.

Provides single source of truth for computing layer/neuron/edge colors
based on gradients, polarity, and default settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .gradient_flow import (
    get_connection_gradient,
    get_layer_activation_gradient,
    get_layer_pair_gradient,
    get_neuron_activation_gradient,
    gradient_to_color,
)

if TYPE_CHECKING:
    from .gradient_flow import GradientFlowConfig
    from .settings import VisualizationSettings


def get_layer_color(
    settings: VisualizationSettings,
    gradient_config: GradientFlowConfig | None,
    layer_id: int,
) -> str:
    """Get fill color for a layer (simple view).

    Priority: gradient > default layer color.

    Args:
        settings: Visualization settings
        gradient_config: Gradient flow config with computed gradients
        layer_id: Layer ID

    Returns:
        Hex color string
    """
    if settings.show_gradients and gradient_config and gradient_config.activation_gradients:
        grad_val = get_layer_activation_gradient(
            gradient_config.activation_gradients,
            layer_id,
            aggregation="abs_mean",
        )
        return gradient_to_color(
            grad_val,
            gradient_config.activation_grad_min,
            gradient_config.activation_grad_max,
            colormap=settings.grad_colormap,
            log_scale=settings.grad_log_scale,
        )
    return settings.layer_color


def get_neuron_color(
    settings: VisualizationSettings,
    gradient_config: GradientFlowConfig | None,
    layer_id: int,
    neuron_idx: int,
    polarity: list | None = None,
) -> str:
    """Get fill color for a specific neuron (detailed view).

    Priority: gradient > polarity > default layer color.

    Args:
        settings: Visualization settings
        gradient_config: Gradient flow config with computed gradients
        layer_id: Layer ID
        neuron_idx: Neuron index within layer
        polarity: Optional polarity list for the layer

    Returns:
        Hex color string
    """
    # First priority: gradients
    if settings.show_gradients and gradient_config and gradient_config.activation_gradients:
        neuron_grad = get_neuron_activation_gradient(
            gradient_config.activation_gradients,
            layer_id,
            neuron_idx,
        )
        return gradient_to_color(
            neuron_grad,
            gradient_config.activation_grad_min,
            gradient_config.activation_grad_max,
            colormap=settings.grad_colormap,
            log_scale=settings.grad_log_scale,
        )

    # Second priority: polarity coloring
    if polarity is not None and neuron_idx < len(polarity):
        polarity_val = polarity[neuron_idx]
        if polarity_val == 1:  # Excitatory
            return "#FF6B6B"  # Red
        elif polarity_val == -1:  # Inhibitory
            return "#5D9CEC"  # Blue

    # Default
    return settings.layer_color


def get_edge_color(
    settings: VisualizationSettings,
    gradient_data: dict | None,
    gradient_config: GradientFlowConfig | None,
    from_layer: int,
    to_layer: int,
    from_neuron: int | None = None,
    to_neuron: int | None = None,
    weight_value: float | None = None,
) -> str:
    """Get color for a connection edge.

    Priority: gradient > connection polarity > default (intra/inter color).

    Args:
        settings: Visualization settings
        gradient_data: Dict of gradient arrays
        gradient_config: Gradient flow config
        from_layer: Source layer ID
        to_layer: Target layer ID
        from_neuron: Source neuron index (for detailed view)
        to_neuron: Target neuron index (for detailed view)
        weight_value: Weight value for polarity coloring

    Returns:
        Hex color string
    """
    is_intra = from_layer == to_layer

    # First priority: gradient coloring
    if settings.show_gradients and gradient_data:
        if from_neuron is not None and to_neuron is not None:
            # Per-connection gradient (detailed view)
            grad_val = get_connection_gradient(
                gradient_data,
                from_layer,
                to_layer,
                from_neuron,
                to_neuron,
            )
        else:
            # Aggregated layer pair gradient (simple view)
            grad_val = get_layer_pair_gradient(
                gradient_data,
                from_layer,
                to_layer,
                aggregation="abs_mean",
            )
        return gradient_to_color(
            grad_val,
            gradient_config.grad_min if gradient_config else 0,
            gradient_config.grad_max if gradient_config else 1,
            colormap=settings.grad_colormap,
            log_scale=settings.grad_log_scale,
        )

    # Second priority: connection polarity
    if settings.show_connection_polarity and weight_value is not None:
        if weight_value > 0:
            return "#FF6B6B"  # Red for excitatory (positive)
        elif weight_value < 0:
            return "#5D9CEC"  # Blue for inhibitory (negative)
        else:
            return "#888888"  # Gray for zero

    # Default: intra vs inter color
    return settings.intra_color if is_intra else settings.inter_color


def get_inter_color_with_contrast(settings: VisualizationSettings) -> str:
    """Get inter-edge color, adjusting for background contrast if using default.

    Args:
        settings: Visualization settings

    Returns:
        Hex color string for inter-layer edges
    """
    inter_color = settings.inter_color
    bg_col = settings.bg_color

    # If using default black, check if it needs adjustment for dark backgrounds
    if inter_color == "#000000":
        lum = _hex_luminance(bg_col)
        return "#000000" if lum > 0.5 else "#ffffff"

    return inter_color


def _hex_luminance(hex_color: str) -> float:
    """Calculate relative luminance of a hex color."""
    hex_color = hex_color.strip().lower().removeprefix("#")
    if len(hex_color) == 3:
        hex_color = "".join([ch * 2 for ch in hex_color])

    try:
        r, g, b = (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
    except (ValueError, IndexError):
        return 1.0  # Default to light

    def _lin(c: int) -> float:
        c_norm = c / 255.0
        return c_norm / 12.92 if c_norm <= 0.04045 else ((c_norm + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

