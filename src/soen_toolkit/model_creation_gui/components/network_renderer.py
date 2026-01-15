"""QGraphicsScene-based network renderer for reliable hit detection and interaction.
Replaces the Graphviz/SVG approach with direct rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
)

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.features.visualization.gradient_flow import GradientFlowConfig


@dataclass
class NodeInfo:
    """Information about a rendered node."""

    layer_id: int
    model_id: int
    rect: QRectF
    dim: int
    layer_type: str
    description: str
    item: QGraphicsRectItem
    row_items: list[QGraphicsItem] = None  # For detailed view
    circular_positions: list[QPointF] = None  # For circular layout


class NetworkRenderer(QObject):
    """Renders SOEN network models using QGraphicsScene for precise interaction."""

    # Signals
    node_clicked = pyqtSignal(int, int)  # layer_id, neuron_index (-1 for whole layer)

    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__()
        self.scene = scene
        self.nodes: dict[int, NodeInfo] = {}
        self.edges: list[QGraphicsItem] = []

        # Settings
        self.orientation = "LR"  # 'LR' or 'TB'
        self.simple_view = True
        self.detailed_layout = "linear"  # 'linear' or 'circular'
        self.edge_routing = "true"  # 'true' (curved), 'false' (straight), 'ortho' (L-shaped)
        self.show_intra = True
        self.show_descriptions = False
        self.show_node_ids = False
        self.show_layer_ids = True
        self.show_model_ids = True
        self.show_conn_types = False

        # Colors
        self.layer_color = QColor("#eae5be")
        self.inter_color = QColor("#000000")
        self.intra_color = QColor("#ff0000")
        self.bg_color = QColor("#ffffff")
        self.text_color = QColor("#000000")  # Text color for labels

        # Polarity settings
        self.show_neuron_polarity = False
        self.show_connection_polarity = False
        self.layer_polarity: dict[int, list] = {}  # layer_id -> polarity list

        # Gradient flow settings
        self.show_gradients = False
        self.gradient_data: dict[str, np.ndarray] | None = None
        self.gradient_config: GradientFlowConfig | None = None

        # Weight matrices for connection polarity (populated during render)
        self._weight_matrices: dict[str, np.ndarray] = {}

        # Dimensions
        self.layer_spacing = 120
        self.layer_width = 100
        self.layer_height = 80
        self.neuron_height = 8  # For detailed view

    def clear(self) -> None:
        """Clear all rendered items."""
        self.scene.clear()
        self.nodes.clear()
        self.edges.clear()

    def render_network(self, layers_config, connections_config, model=None) -> None:
        """Render the network from configuration.

        Args:
            layers_config: List of LayerConfig objects
            connections_config: List of ConnectionConfig objects
            model: Optional SOENModelCore for getting actual weights

        """
        self.clear()

        if not layers_config:
            return

        # Identify which layers have internal connections (for circular layout)
        self._layers_with_internal = set()
        if self.detailed_layout == "circular":
            for conn in connections_config:
                if conn.from_layer == conn.to_layer:
                    self._layers_with_internal.add(conn.from_layer)

        # Pre-compute dynamic widths for simple view so text fits within boxes
        widths_by_layer: dict[int, int] | None = None
        if self.simple_view:
            widths_by_layer = {}
            for cfg in layers_config:
                try:
                    widths_by_layer[cfg.layer_id] = self._compute_simple_layer_width(cfg)
                except Exception:
                    widths_by_layer[cfg.layer_id] = self.layer_width

        # Calculate positions for each layer (respect dynamic widths if provided)
        positions = self._calculate_layer_positions(layers_config, widths_by_layer)

        # Render layers
        for _i, layer_cfg in enumerate(layers_config):
            pos = positions[layer_cfg.layer_id]
            width_override = widths_by_layer.get(layer_cfg.layer_id) if widths_by_layer else None
            self._render_layer(layer_cfg, pos, model, width_override)

        # Render connections
        for conn_cfg in connections_config:
            self._render_connection(conn_cfg, model)

    # ─────────────────────────────────────────────────────────────────────────
    # Layout Constants (single source of truth)
    # ─────────────────────────────────────────────────────────────────────────
    LINEAR_WIDTH = 20  # Width of linear detailed view columns
    CIRCULAR_MIN_RADIUS = 40  # Minimum radius for circular layout
    CIRCULAR_NODE_SIZE = 12.0  # Preferred node diameter
    CIRCULAR_NODE_GAP = 2.0  # Gap between nodes in circular layout

    def _get_circular_radius(self, dim: int) -> float:
        """Calculate radius for circular layout. Single source of truth."""
        dim = max(1, dim)
        required_arc = self.CIRCULAR_NODE_SIZE + self.CIRCULAR_NODE_GAP
        min_radius_for_nodes = (required_arc * dim) / (2 * math.pi)
        return max(self.CIRCULAR_MIN_RADIUS, min_radius_for_nodes)

    def _is_circular_layer(self, layer_id: int) -> bool:
        """Check if a layer should use circular layout."""
        return (
            self.detailed_layout == "circular"
            and layer_id in getattr(self, "_layers_with_internal", set())
        )

    def _get_layer_width(self, dim: int, layer_id: int) -> float:
        """Get the width a layer occupies for positioning. Single source of truth."""
        if self.simple_view:
            return self.layer_width
        if self._is_circular_layer(layer_id):
            radius = self._get_circular_radius(dim)
            return 2 * radius + self.CIRCULAR_NODE_SIZE  # Full diameter + node size
        return self.LINEAR_WIDTH

    def _calculate_layer_positions(self, layers_config, widths_by_layer: dict[int, int] | None = None) -> dict[int, QPointF]:
        """Calculate position for each layer based on orientation.

        Positions are the LEFT EDGE x-coordinate for LR layout.
        For circular layers, this is the left edge of the bounding circle.
        """
        positions = {}

        if self.orientation == "LR":
            x_cursor = 0
            for cfg in sorted(layers_config, key=lambda x: x.layer_id):
                dim = cfg.params.get("dim", 1)
                layer_width = widths_by_layer.get(cfg.layer_id) if widths_by_layer else None
                if layer_width is None:
                    layer_width = self._get_layer_width(dim, cfg.layer_id)
                positions[cfg.layer_id] = QPointF(x_cursor, 0)
                x_cursor += layer_width + self.layer_spacing
        else:
            # Top to bottom layout
            y_cursor = 0
            for cfg in sorted(layers_config, key=lambda x: x.layer_id):
                positions[cfg.layer_id] = QPointF(0, y_cursor)
                dim = cfg.params.get("dim", 1)
                h = self.layer_height if self.simple_view else dim * self.neuron_height
                y_cursor += h + self.layer_spacing

        return positions

    def _build_layer_title(self, layer_id: int, model_id: int) -> str:
        """Build the title text for a layer label."""
        if not self.show_layer_ids:
            return ""
        title = f"Layer {layer_id}"
        if self.show_model_ids:
            title = f"{title} (M{model_id})"
        return title

    def _create_text(
        self, text: str, font: QFont, parent: QGraphicsItem | None = None
    ) -> QGraphicsTextItem:
        """Create a styled text item with correct color."""
        item = QGraphicsTextItem(text, parent)
        item.setFont(font)
        item.setDefaultTextColor(self.text_color)
        return item

    def _render_layer(self, layer_cfg, pos: QPointF, model=None, width_override: int | None = None) -> None:
        """Render a single layer."""
        layer_id = layer_cfg.layer_id
        dim = layer_cfg.params.get("dim", 1)
        model_id = getattr(layer_cfg, "model_id", 0)

        # Determine fill color - use activation gradient if enabled
        fill_color = self._get_layer_fill_color(layer_id)

        if self.simple_view:
            # Simple box representation
            box_width = width_override if width_override is not None else self.layer_width
            rect = QRectF(pos.x(), pos.y(), box_width, self.layer_height)

            # Adjust height if showing description
            if self.show_descriptions and layer_cfg.description:
                # Calculate description height
                temp_desc = QGraphicsTextItem(layer_cfg.description)
                font_desc = QFont("Arial", 8)
                font_desc.setItalic(True)
                temp_desc.setFont(font_desc)
                desc_height = temp_desc.boundingRect().height()
                rect.setHeight(self.layer_height + desc_height + 5)

            box = QGraphicsRectItem(rect)
            box.setPen(QPen(Qt.GlobalColor.black, 1))
            box.setBrush(QBrush(fill_color))
            box.setData(0, ("layer", layer_id, -1))
            self.scene.addItem(box)

            # Add layer label INSIDE the box (relative positioning)
            y_offset = 5
            title_text = self._build_layer_title(layer_id, model_id)
            if title_text:
                label = self._create_text(title_text, QFont("Arial", 10, QFont.Weight.Bold), box)
                # Center horizontally
                label_x = (box_width - label.boundingRect().width()) / 2
                label.setPos(label_x, y_offset)
                y_offset += 18

            # Optionally add type line in simple view when descriptions are shown
            if self.show_descriptions and layer_cfg.layer_type:
                type_label = self._create_text(layer_cfg.layer_type, QFont("Arial", 9), box)
                # Center horizontally
                type_x = (box_width - type_label.boundingRect().width()) / 2
                type_label.setPos(type_x, y_offset)
                y_offset += 16

            # Add dimension info INSIDE the box
            dim_label = self._create_text(f"Dim: {dim}", QFont("Arial", 9), box)
            # Center horizontally
            dim_x = (box_width - dim_label.boundingRect().width()) / 2
            dim_label.setPos(dim_x, y_offset)

            # Add description INSIDE the box, below other text
            if self.show_descriptions and layer_cfg.description:
                font_desc = QFont("Arial", 8)
                font_desc.setItalic(True)
                desc_label = self._create_text(layer_cfg.description, font_desc, box)
                # Center horizontally
                desc_x = (box_width - desc_label.boundingRect().width()) / 2
                desc_label.setPos(desc_x, self.layer_height)

            # Store node info
            self.nodes[layer_id] = NodeInfo(
                layer_id=layer_id,
                model_id=model_id,
                rect=rect,
                dim=dim,
                layer_type=layer_cfg.layer_type,
                description=layer_cfg.description or "",
                item=box,
            )
        elif self._is_circular_layer(layer_id):
            # Detailed view with circular layout (only for layers with internal connections)
            self._render_layer_circular(layer_cfg, pos, layer_id, dim, model_id)
        else:
            # Detailed view with linear layout
            self._render_layer_linear(layer_cfg, pos, layer_id, dim, model_id)

    def _render_layer_linear(
        self, layer_cfg, pos: QPointF, layer_id: int, dim: int, model_id: int
    ) -> None:
        """Render detailed view with neurons arranged in a vertical column (linear layout)."""
        total_height = dim * self.neuron_height
        layer_rect = QRectF(pos.x(), pos.y(), self.LINEAR_WIDTH, total_height)

        # Individual neuron rows (colored background)
        row_items = []
        for i in range(dim):
            y = pos.y() + i * self.neuron_height

            # Row separator line
            if i > 0:
                line = QGraphicsLineItem(pos.x(), y, pos.x() + self.LINEAR_WIDTH, y)
                line.setPen(QPen(QColor("#cccccc"), 0.5))
                self.scene.addItem(line)

            # Get per-neuron color if gradients enabled
            neuron_color = self._get_layer_fill_color(layer_id, neuron_idx=i)

            # Neuron rectangle with color
            neuron_rect = QRectF(pos.x(), y, self.LINEAR_WIDTH, self.neuron_height)
            neuron_item = QGraphicsRectItem(neuron_rect)
            neuron_item.setPen(QPen(QColor("#cccccc"), 0.5))
            neuron_item.setBrush(QBrush(neuron_color))
            neuron_item.setData(0, ("neuron", layer_id, i))
            self.scene.addItem(neuron_item)
            row_items.append(neuron_item)

            # Neuron index label (positioned at neuron location)
            if self.show_node_ids:
                idx_label = self._create_text(str(i), QFont("Arial", 6))
                idx_label.setPos(pos.x() + 2, y - 2)
                self.scene.addItem(idx_label)

        # Layer label ABOVE the detailed view
        title_text = self._build_layer_title(layer_id, model_id)
        if title_text:
            label = self._create_text(title_text, QFont("Arial", 10, QFont.Weight.Bold))
            label.setPos(pos.x(), pos.y() - 20)
            self.scene.addItem(label)

        # Description BELOW the detailed view
        if self.show_descriptions and layer_cfg.description:
            font_desc = QFont("Arial", 8)
            font_desc.setItalic(True)
            desc_label = self._create_text(layer_cfg.description, font_desc)
            desc_label.setPos(pos.x(), pos.y() + total_height + 5)
            self.scene.addItem(desc_label)

        # Invisible container for hit detection
        container = QGraphicsRectItem(layer_rect)
        container.setPen(QPen(Qt.PenStyle.NoPen))
        container.setBrush(QBrush(Qt.GlobalColor.transparent))
        container.setData(0, ("layer", layer_id, -1))
        self.scene.addItem(container)

        # Store node info
        self.nodes[layer_id] = NodeInfo(
            layer_id=layer_id,
            model_id=model_id,
            rect=layer_rect,
            dim=dim,
            layer_type=layer_cfg.layer_type,
            description=layer_cfg.description or "",
            item=container,
            row_items=row_items,
        )

    def _compute_circular_layout(self, dim: int, center: QPointF) -> tuple[list[QPointF], float]:
        """Compute circular node positions and node size.

        Args:
            dim: Number of neurons in the layer
            center: Center point of the circle

        Returns:
            Tuple of (positions, node_size) - node positions and diameter
        """
        dim = max(1, dim)
        radius = self._get_circular_radius(dim)

        # Compute actual node size based on arc length available
        arc_per_node = (2 * math.pi * radius) / dim
        node_size = max(4.0, min(self.CIRCULAR_NODE_SIZE, arc_per_node - self.CIRCULAR_NODE_GAP))

        positions = []
        for i in range(dim):
            angle = 2 * math.pi * i / dim - math.pi / 2  # Start at top
            x = center.x() + radius * math.cos(angle)
            y = center.y() + radius * math.sin(angle)
            positions.append(QPointF(x, y))
        return positions, node_size

    def _render_layer_circular(
        self, layer_cfg, pos: QPointF, layer_id: int, dim: int, model_id: int
    ) -> None:
        """Render detailed view with neurons arranged in a circle (circular layout).

        Args:
            pos: Left edge position (x is left edge, y is top)
        """
        # pos is the left edge, compute center for the circle
        radius = self._get_circular_radius(dim)
        center = QPointF(pos.x() + radius + self.CIRCULAR_NODE_SIZE / 2, pos.y() + radius)
        positions, node_size = self._compute_circular_layout(dim, center)

        # Calculate bounding rect from positions
        if positions:
            min_x = min(p.x() for p in positions) - node_size
            max_x = max(p.x() for p in positions) + node_size
            min_y = min(p.y() for p in positions) - node_size
            max_y = max(p.y() for p in positions) + node_size
        else:
            min_x, max_x = pos.x() - 50, pos.x() + 50
            min_y, max_y = pos.y() - 50, pos.y() + 50

        layer_rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

        # Render each neuron as a circle
        row_items = []
        for i, node_pos in enumerate(positions):
            # Get per-neuron color if gradients enabled
            neuron_color = self._get_layer_fill_color(layer_id, neuron_idx=i)

            # Create ellipse (circle) centered at node_pos
            ellipse_rect = QRectF(
                node_pos.x() - node_size / 2,
                node_pos.y() - node_size / 2,
                node_size,
                node_size,
            )
            neuron_item = QGraphicsEllipseItem(ellipse_rect)
            neuron_item.setPen(QPen(QColor("#666666"), 0.5))
            neuron_item.setBrush(QBrush(neuron_color))
            neuron_item.setData(0, ("neuron", layer_id, i))
            self.scene.addItem(neuron_item)
            row_items.append(neuron_item)

            # Neuron index label (positioned at center of node)
            if self.show_node_ids and node_size >= 8:
                font = QFont("Arial", max(4, int(node_size * 0.5)))
                idx_label = self._create_text(str(i), font)
                text_rect = idx_label.boundingRect()
                idx_label.setPos(
                    node_pos.x() - text_rect.width() / 2,
                    node_pos.y() - text_rect.height() / 2,
                )
                self.scene.addItem(idx_label)

        # Layer label ABOVE the circular layout
        title_text = self._build_layer_title(layer_id, model_id)
        if title_text:
            label = self._create_text(title_text, QFont("Arial", 10, QFont.Weight.Bold))
            label.setPos(center.x() - 40, min_y - 25)
            self.scene.addItem(label)

        # Description BELOW the circular layout
        if self.show_descriptions and layer_cfg.description:
            font_desc = QFont("Arial", 8)
            font_desc.setItalic(True)
            desc_label = self._create_text(layer_cfg.description, font_desc)
            desc_label.setPos(center.x() - 40, max_y + 5)
            self.scene.addItem(desc_label)

        # Invisible container for hit detection
        container = QGraphicsRectItem(layer_rect)
        container.setPen(QPen(Qt.PenStyle.NoPen))
        container.setBrush(QBrush(Qt.GlobalColor.transparent))
        container.setData(0, ("layer", layer_id, -1))
        self.scene.addItem(container)

        # Store node info with positions for connection rendering
        self.nodes[layer_id] = NodeInfo(
            layer_id=layer_id,
            model_id=model_id,
            rect=layer_rect,
            dim=dim,
            layer_type=layer_cfg.layer_type,
            description=layer_cfg.description or "",
            item=container,
            row_items=row_items,
            circular_positions=positions,
        )

    def _compute_simple_layer_width(self, layer_cfg) -> int:
        """Compute a suitable width for a simple-view layer box based on text lines."""
        title_font = QFont("Arial", 10, QFont.Weight.Bold)
        line_font = QFont("Arial", 9)
        fm_title = QFontMetrics(title_font)
        fm_line = QFontMetrics(line_font)

        max_w = 0
        # Title line
        if self.show_layer_ids:
            title_text = f"Layer {layer_cfg.layer_id}"
            if self.show_model_ids:
                title_text = f"{title_text} (M{getattr(layer_cfg, 'model_id', 0)})"
            max_w = max(max_w, fm_title.horizontalAdvance(title_text))

        # Optional type line when descriptions toggle is on
        if self.show_descriptions and getattr(layer_cfg, "layer_type", None):
            max_w = max(max_w, fm_line.horizontalAdvance(f"{layer_cfg.layer_type}"))

        # Dim line
        max_w = max(max_w, fm_line.horizontalAdvance(f"Dim: {layer_cfg.params.get('dim', 1)}"))

        # Padding inside the box plus border allowance
        required = max(self.layer_width, max_w + 20)  # 10px padding left+right
        return int(required)

    def _render_connection(self, conn_cfg, model=None) -> None:
        """Render a connection between layers."""
        from_id = conn_cfg.from_layer
        to_id = conn_cfg.to_layer

        if from_id not in self.nodes or to_id not in self.nodes:
            return

        from_node = self.nodes[from_id]
        to_node = self.nodes[to_id]

        # Determine if intra-layer
        is_intra = from_id == to_id
        if is_intra and not self.show_intra:
            return

        # Load weight matrix if connection polarity is enabled
        weight_mat = None
        if self.show_connection_polarity and model is not None:
            key = f"J_{from_id}_to_{to_id}"
            connections = getattr(model, "connections", None)
            if connections is not None and key in connections:
                weight_mat = connections[key].detach().cpu().numpy()
                self._weight_matrices[key] = weight_mat

        # Choose color - priority: gradient > polarity > default
        color = self._get_connection_color(from_id, to_id, is_intra, weight_mat)

        if self.simple_view:
            # Draw single edge between layer boxes
            if is_intra:
                # Self-loop
                rect = from_node.rect
                cy = rect.center().y()

                # Create a loop path
                path = QPainterPath()
                path.moveTo(rect.right(), cy)
                # Arc to the right and back
                ctrl1 = QPointF(rect.right() + 30, cy - 20)
                ctrl2 = QPointF(rect.right() + 30, cy + 20)
                path.cubicTo(ctrl1, ctrl2, QPointF(rect.right(), cy))

                path_item = QGraphicsPathItem(path)
                path_item.setPen(QPen(color, 1.5))
                self.scene.addItem(path_item)
                self.edges.append(path_item)
            else:
                # Edge between different layers
                from_rect = from_node.rect
                to_rect = to_node.rect

                # Calculate connection points
                if self.orientation == "LR":
                    from_pt = QPointF(from_rect.right(), from_rect.center().y())
                    to_pt = QPointF(to_rect.left(), to_rect.center().y())
                else:
                    from_pt = QPointF(from_rect.center().x(), from_rect.bottom())
                    to_pt = QPointF(to_rect.center().x(), to_rect.top())

                # Draw edge based on routing style
                self._draw_routed_edge(from_pt, to_pt, color, 1.5)

                # Add arrowhead
                self._add_arrowhead(to_pt, from_pt, color)

                # Add connection type label if enabled
                if self.show_conn_types and not is_intra:
                    label = self._create_text(conn_cfg.connection_type, QFont("Arial", 8))
                    label.setPos((from_pt.x() + to_pt.x()) / 2 - 20, (from_pt.y() + to_pt.y()) / 2 - 10)
                    self.scene.addItem(label)

        elif self.detailed_layout == "circular":
            # In circular layout mode, always use detailed per-connection rendering
            # for accurate polarity coloring (regardless of whether layers are circular)
            self._render_connection_detailed(conn_cfg, from_node, to_node, color, is_intra, model)
        else:
            # Linear detailed view: draw simple layer-to-layer connections
            self._render_connection_linear(from_node, to_node, color, is_intra)

    def _render_connection_linear(self, from_node, to_node, color, is_intra) -> None:
        """Original connection rendering for linear detailed view - single line between layers."""
        from_rect = from_node.rect
        to_rect = to_node.rect

        if is_intra:
            # Self-loop on detailed view
            cx = from_rect.right()
            cy = from_rect.center().y()

            path = QPainterPath()
            path.moveTo(cx, cy - 10)
            ctrl1 = QPointF(cx + 25, cy - 10)
            ctrl2 = QPointF(cx + 25, cy + 10)
            path.cubicTo(ctrl1, ctrl2, QPointF(cx, cy + 10))

            path_item = QGraphicsPathItem(path)
            path_item.setPen(QPen(color, 1))
            self.scene.addItem(path_item)
            self.edges.append(path_item)
        else:
            # Connection between layers
            if self.orientation == "LR":
                from_pt = QPointF(from_rect.right(), from_rect.center().y())
                to_pt = QPointF(to_rect.left(), to_rect.center().y())
            else:
                from_pt = QPointF(from_rect.center().x(), from_rect.bottom())
                to_pt = QPointF(to_rect.center().x(), to_rect.top())

            self._draw_routed_edge(from_pt, to_pt, color, 1.0)

    def _render_connection_detailed(self, conn_cfg, from_node, to_node, base_color, is_intra, model) -> None:
        """Render individual neuron-to-neuron connections for detailed view."""
        from_id = conn_cfg.from_layer
        to_id = conn_cfg.to_layer

        if model is None:
            raise ValueError("Model is required for detailed connection rendering")

        masks = getattr(model, "connection_masks", None)
        if masks is None:
            raise ValueError("Model has no connection_masks attribute")

        # All connections use J_{from}_to_{to} format (see builders.py line 615)
        key = f"J_{from_id}_to_{to_id}"
        mask = masks.get(key)

        if mask is None:
            available_keys = list(masks.keys())
            raise ValueError(
                f"Connection mask not found: key='{key}' not in connection_masks. "
                f"Available keys: {available_keys}"
            )

        mask = mask.detach().cpu().numpy()

        if mask.sum() == 0:
            # No connections in this mask - nothing to draw
            return

        # Get weight matrix for per-connection polarity coloring
        weight_mat = None
        if self.show_connection_polarity:
            connections = getattr(model, "connections", None)
            if connections is not None and key in connections:
                weight_mat = connections[key].detach().cpu().numpy()

        self._render_connections_from_mask(from_node, to_node, mask, base_color, is_intra, weight_mat)

    def _render_connections_from_mask(
        self,
        from_node,
        to_node,
        mask,
        base_color,
        is_intra,
        weight_mat: np.ndarray | None = None,
    ) -> None:
        """Render connections based on actual connectivity mask.

        Args:
            from_node: Source node info
            to_node: Target node info
            mask: Connection mask array
            base_color: Default color for connections
            is_intra: Whether this is an intra-layer connection
            weight_mat: Optional weight matrix for per-connection polarity coloring
        """
        is_circular = self.detailed_layout == "circular"

        # Get node positions - always compute fresh to ensure correctness
        from_positions = self._get_node_positions(from_node, is_circular)
        to_positions = self._get_node_positions(to_node, is_circular)

        if not from_positions or not to_positions:
            return

        # mask shape: (to_dim, from_dim) - rows are destination, cols are source
        nz = np.argwhere(mask != 0)

        # Limit connections for performance (avoid drawing thousands of lines)
        max_connections = 1000
        if len(nz) > max_connections:
            # Sample connections uniformly
            indices = np.linspace(0, len(nz) - 1, max_connections, dtype=int)
            nz = nz[indices]

        # Draw each connection
        for to_idx, from_idx in nz:
            if from_idx >= len(from_positions) or to_idx >= len(to_positions):
                continue

            from_pos = from_positions[from_idx]
            to_pos = to_positions[to_idx]

            # Determine color for this specific connection
            if self.show_connection_polarity and weight_mat is not None:
                # Per-connection polarity coloring based on weight sign
                weight_value = weight_mat[to_idx, from_idx]
                if weight_value > 0:
                    edge_color = QColor("#FF6B6B")  # Red for excitatory
                elif weight_value < 0:
                    edge_color = QColor("#5D9CEC")  # Blue for inhibitory
                else:
                    edge_color = QColor("#888888")  # Gray for zero
            else:
                edge_color = base_color

            if is_circular:
                self._draw_circular_edge(from_pos, to_pos, edge_color, is_intra)
            else:
                self._draw_linear_edge(from_pos, to_pos, edge_color, from_node, to_node)

    def _get_node_positions(self, node, is_circular: bool) -> list[QPointF]:
        """Get neuron positions for a node based on current layout."""
        if is_circular and node.circular_positions:
            return node.circular_positions

        # Linear layout - compute positions from rect
        positions = []
        rect = node.rect
        for i in range(node.dim):
            y = rect.y() + i * self.neuron_height + self.neuron_height / 2
            x = rect.center().x()
            positions.append(QPointF(x, y))
        return positions

    def _draw_circular_edge(self, from_pos: QPointF, to_pos: QPointF, color, is_intra: bool) -> None:
        """Draw an edge between two circular node positions."""
        dx = to_pos.x() - from_pos.x()
        dy = to_pos.y() - from_pos.y()
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1:
            return

        # Use thin lines for individual connections
        pen = QPen(color, 0.5)
        pen.setCosmetic(True)  # Keep line width constant regardless of zoom

        path = QPainterPath()
        path.moveTo(from_pos)

        if self.edge_routing == "false":
            # Straight line
            path.lineTo(to_pos)
        elif self.edge_routing == "ortho":
            # Orthogonal routing
            mid_x = (from_pos.x() + to_pos.x()) / 2
            mid_y = (from_pos.y() + to_pos.y()) / 2
            if abs(dx) > abs(dy):
                # Horizontal first
                path.lineTo(mid_x, from_pos.y())
                path.lineTo(mid_x, to_pos.y())
            else:
                # Vertical first
                path.lineTo(from_pos.x(), mid_y)
                path.lineTo(to_pos.x(), mid_y)
            path.lineTo(to_pos)
        else:
            # Curved ("true") - slightly different curve for intra vs inter
            mid_x = (from_pos.x() + to_pos.x()) / 2
            mid_y = (from_pos.y() + to_pos.y()) / 2
            if is_intra:
                # Curve inward toward center for internal connections
                curve_factor = 0.3
                perp_x = -dy / dist * curve_factor * dist
                perp_y = dx / dist * curve_factor * dist
                ctrl = QPointF(mid_x + perp_x * 0.5, mid_y + perp_y * 0.5)
                path.quadTo(ctrl, to_pos)
            else:
                # Standard bezier curve for inter-layer
                ctrl1 = QPointF(mid_x, from_pos.y())
                ctrl2 = QPointF(mid_x, to_pos.y())
                path.cubicTo(ctrl1, ctrl2, to_pos)

        path_item = QGraphicsPathItem(path)
        path_item.setPen(pen)
        self.scene.addItem(path_item)
        self.edges.append(path_item)

    def _draw_linear_edge(self, from_pos: QPointF, to_pos: QPointF, color, from_node, to_node) -> None:
        """Draw an edge between two linear node positions."""
        pen = QPen(color, 0.5)
        pen.setCosmetic(True)

        # Adjust x positions to edge of layer rectangles
        from_x = from_node.rect.right()
        to_x = to_node.rect.left()
        from_pt = QPointF(from_x, from_pos.y())
        to_pt = QPointF(to_x, to_pos.y())

        path = QPainterPath()
        path.moveTo(from_pt)

        if self.edge_routing == "false":
            # Straight line
            path.lineTo(to_pt)
        elif self.edge_routing == "ortho":
            # Orthogonal routing
            mid_x = (from_pt.x() + to_pt.x()) / 2
            path.lineTo(mid_x, from_pt.y())
            path.lineTo(mid_x, to_pt.y())
            path.lineTo(to_pt)
        else:
            # Curved ("true")
            mid_x = (from_pt.x() + to_pt.x()) / 2
            ctrl1 = QPointF(mid_x, from_pt.y())
            ctrl2 = QPointF(mid_x, to_pt.y())
            path.cubicTo(ctrl1, ctrl2, to_pt)

        path_item = QGraphicsPathItem(path)
        path_item.setPen(pen)
        self.scene.addItem(path_item)
        self.edges.append(path_item)

    def _draw_routed_edge(
        self, from_pt: QPointF, to_pt: QPointF, color: QColor, pen_width: float = 1.5
    ) -> None:
        """Draw an edge between two points using the configured edge routing style.

        Args:
            from_pt: Start point
            to_pt: End point
            color: Edge color
            pen_width: Line width
        """
        path = QPainterPath()
        path.moveTo(from_pt)

        if self.edge_routing == "false":
            # Straight line
            path.lineTo(to_pt)
        elif self.edge_routing == "ortho":
            # Orthogonal (L-shaped) routing
            if self.orientation == "LR":
                mid_x = (from_pt.x() + to_pt.x()) / 2
                path.lineTo(mid_x, from_pt.y())
                path.lineTo(mid_x, to_pt.y())
                path.lineTo(to_pt)
            else:
                mid_y = (from_pt.y() + to_pt.y()) / 2
                path.lineTo(from_pt.x(), mid_y)
                path.lineTo(to_pt.x(), mid_y)
                path.lineTo(to_pt)
        else:
            # Default: curved/spline ("true")
            if self.orientation == "LR":
                mid_x = (from_pt.x() + to_pt.x()) / 2
                ctrl1 = QPointF(mid_x, from_pt.y())
                ctrl2 = QPointF(mid_x, to_pt.y())
            else:
                mid_y = (from_pt.y() + to_pt.y()) / 2
                ctrl1 = QPointF(from_pt.x(), mid_y)
                ctrl2 = QPointF(to_pt.x(), mid_y)
            path.cubicTo(ctrl1, ctrl2, to_pt)

        path_item = QGraphicsPathItem(path)
        path_item.setPen(QPen(color, pen_width))
        self.scene.addItem(path_item)
        self.edges.append(path_item)

    def _add_arrowhead(self, tip: QPointF, from_pt: QPointF, color: QColor) -> None:
        """Add an arrowhead at the tip pointing from from_pt."""
        # Calculate angle
        dx = tip.x() - from_pt.x()
        dy = tip.y() - from_pt.y()
        angle = math.atan2(dy, dx)

        # Arrowhead dimensions
        arrow_length = 8
        arrow_angle = math.pi / 6  # 30 degrees

        # Calculate arrow points
        p1 = QPointF(
            tip.x() - arrow_length * math.cos(angle - arrow_angle),
            tip.y() - arrow_length * math.sin(angle - arrow_angle),
        )
        p2 = QPointF(
            tip.x() - arrow_length * math.cos(angle + arrow_angle),
            tip.y() - arrow_length * math.sin(angle + arrow_angle),
        )

        # Draw arrow as a small triangle
        path = QPainterPath()
        path.moveTo(tip)
        path.lineTo(p1)
        path.lineTo(p2)
        path.closeSubpath()

        arrow = QGraphicsPathItem(path)
        arrow.setPen(QPen(color, 1))
        arrow.setBrush(QBrush(color))
        self.scene.addItem(arrow)
        self.edges.append(arrow)

    def _get_layer_fill_color(self, layer_id: int, neuron_idx: int | None = None) -> QColor:
        """Get fill color for a layer or specific neuron.

        Priority: gradient > polarity > default layer color.

        Args:
            layer_id: The layer ID
            neuron_idx: Optional neuron index for detailed view coloring

        Returns:
            QColor for the layer/neuron based on gradient/polarity if enabled,
            otherwise the default layer color.
        """
        # First priority: gradient coloring
        if self.show_gradients and self.gradient_config is not None:
            if self.gradient_config.activation_gradients:
                from soen_toolkit.model_creation_gui.features.visualization.gradient_flow import (
                    get_layer_activation_gradient,
                    get_neuron_activation_gradient,
                    gradient_to_color,
                )

                # Get gradient value - per-neuron for detailed view, aggregated for simple view
                if neuron_idx is not None:
                    grad_val = get_neuron_activation_gradient(
                        self.gradient_config.activation_gradients,
                        layer_id,
                        neuron_idx,
                    )
                else:
                    grad_val = get_layer_activation_gradient(
                        self.gradient_config.activation_gradients,
                        layer_id,
                        aggregation="abs_mean",
                    )

                # Get color limits from config
                grad_min = self.gradient_config.activation_grad_min
                grad_max = self.gradient_config.activation_grad_max
                colormap = self.gradient_config.colormap if self.gradient_config else "RdBu_r"
                log_scale = self.gradient_config.log_scale if self.gradient_config else False

                # Convert to hex color
                hex_color = gradient_to_color(
                    grad_val,
                    grad_min,
                    grad_max,
                    colormap=colormap,
                    log_scale=log_scale,
                )
                return QColor(hex_color)

        # Second priority: polarity coloring (only for individual neurons)
        if self.show_neuron_polarity and neuron_idx is not None:
            polarity = self.layer_polarity.get(layer_id)
            if polarity is not None and neuron_idx < len(polarity):
                polarity_val = polarity[neuron_idx]
                if polarity_val == 1:  # Excitatory
                    return QColor("#FF6B6B")  # Red
                elif polarity_val == -1:  # Inhibitory
                    return QColor("#5D9CEC")  # Blue

        # Default: layer color
        return self.layer_color

    def _get_gradient_color(self, from_layer: int, to_layer: int) -> QColor:
        """Get gradient-based color for a connection between layers."""
        from soen_toolkit.model_creation_gui.features.visualization.gradient_flow import (
            get_layer_pair_gradient,
            gradient_to_color,
        )

        if self.gradient_data is None:
            return self.inter_color

        # Get aggregated gradient value for this layer pair
        grad_val = get_layer_pair_gradient(
            self.gradient_data,
            from_layer,
            to_layer,
            aggregation="abs_mean",
        )

        # Get color limits from config
        grad_min = self.gradient_config.grad_min if self.gradient_config else 0
        grad_max = self.gradient_config.grad_max if self.gradient_config else 1
        colormap = self.gradient_config.colormap if self.gradient_config else "RdBu_r"
        log_scale = self.gradient_config.log_scale if self.gradient_config else False

        # Convert to hex color
        hex_color = gradient_to_color(
            grad_val,
            grad_min,
            grad_max,
            colormap=colormap,
            log_scale=log_scale,
        )

        return QColor(hex_color)

    def _get_connection_color(
        self,
        from_layer: int,
        to_layer: int,
        is_intra: bool,
        weight_mat: np.ndarray | None = None,
    ) -> QColor:
        """Get color for a connection edge.

        Priority: gradient > connection polarity > default (intra/inter).

        Args:
            from_layer: Source layer ID
            to_layer: Target layer ID
            is_intra: Whether this is an intra-layer connection
            weight_mat: Optional weight matrix for polarity coloring

        Returns:
            QColor for the connection
        """
        # First priority: gradient coloring
        if self.show_gradients and self.gradient_data is not None:
            return self._get_gradient_color(from_layer, to_layer)

        # Second priority: connection polarity (use mean weight sign)
        if self.show_connection_polarity and weight_mat is not None:
            mean_weight = float(np.mean(weight_mat))
            if mean_weight > 0:
                return QColor("#FF6B6B")  # Red for excitatory (positive)
            elif mean_weight < 0:
                return QColor("#5D9CEC")  # Blue for inhibitory (negative)
            else:
                return QColor("#888888")  # Gray for zero

        # Default: intra vs inter color
        return self.intra_color if is_intra else self.inter_color

    def handle_click(self, scene_point: QPointF) -> tuple[int, int]:
        """Handle a click at the given scene point.

        Returns:
            (layer_id, neuron_index) or (-1, -1) if nothing clicked

        """
        items = self.scene.items(scene_point)

        for item in items:
            data = item.data(0)
            if data and isinstance(data, tuple):
                _item_type, layer_id, neuron_idx = data
                self.node_clicked.emit(layer_id, neuron_idx)
                return layer_id, neuron_idx

        return -1, -1
