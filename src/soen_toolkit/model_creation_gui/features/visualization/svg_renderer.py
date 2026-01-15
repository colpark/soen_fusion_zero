"""SVG rendering utilities for network visualization.

This module provides helper functions for Graphviz-based SVG rendering.
The main _render_svg method remains in tab.py but uses these utilities.

Future work: Incrementally move more rendering logic here.
"""

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import VisualizationSettings


def color_hex_for_model(model_id: int) -> str:
    """Generate a stable color for a given model ID.

    Args:
        model_id: The model identifier

    Returns:
        Hex color string (e.g. '#ff5733')
    """
    try:
        h = abs(hash(int(model_id))) % 360
    except (ValueError, TypeError):
        h = 0
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.55, 0.90)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to RGB tuple.

    Args:
        hex_color: Hex color like '#ff5733' or '#f53'

    Returns:
        Tuple of (r, g, b) integers 0-255
    """
    s = hex_color.strip().lower().removeprefix("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except (ValueError, IndexError):
        return (255, 255, 255)


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Calculate relative luminance of an RGB color.

    Args:
        rgb: Tuple of (r, g, b) integers 0-255

    Returns:
        Relative luminance value 0.0-1.0
    """
    def _linearize(c: int) -> float:
        c_norm = c / 255.0
        return c_norm / 12.92 if c_norm <= 0.04045 else ((c_norm + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def get_contrasting_edge_color(inter_color: str, bg_color: str) -> str:
    """Get edge color with contrast adjustment for dark backgrounds.

    Args:
        inter_color: The configured inter-edge color
        bg_color: The background color

    Returns:
        Hex color string, adjusted for contrast if needed
    """
    if inter_color == "#000000":
        lum = relative_luminance(hex_to_rgb(bg_color))
        return "#000000" if lum > 0.5 else "#ffffff"
    return inter_color


def build_simple_node_html(
    title_html: str,
    layer_bg: str,
    width_pt: int | None = None,
) -> str:
    """Build HTML-like label for a simple view node.

    Args:
        title_html: The HTML content for the title (with <BR/> separators)
        layer_bg: Background color for the layer cell
        width_pt: Optional width in points

    Returns:
        Graphviz HTML-like label string
    """
    if width_pt is not None:
        cell = f'<TR><TD BGCOLOR="{layer_bg}" ALIGN="CENTER" WIDTH="{width_pt}">{title_html}</TD></TR>'
    else:
        cell = f'<TR><TD BGCOLOR="{layer_bg}" ALIGN="CENTER">{title_html}</TD></TR>'
    return f'<<TABLE BORDER="1" CELLBORDER="1" CELLPADDING="6" CELLSPACING="0">{cell}</TABLE>>'


def build_title_parts(
    layer_id: int,
    model_id: int,
    dim: int,
    layer_type: str | None,
    settings: VisualizationSettings,
) -> list[str]:
    """Build the title parts for a layer label.

    Args:
        layer_id: Layer identifier
        model_id: Model identifier
        dim: Layer dimension
        layer_type: Optional layer type name
        settings: Visualization settings

    Returns:
        List of title lines for the label
    """
    parts = []
    if settings.show_layer_ids:
        if settings.show_model_ids:
            parts.append(f"Layer: {layer_id} (M{model_id})")
        else:
            parts.append(f"Layer: {layer_id}")

    # When descriptions are enabled in simple view, also show the layer type
    if settings.show_desc and layer_type:
        parts.append(str(layer_type))

    parts.append(f"Dim: {dim}")
    return parts

