"""Overlay invisible hit zones on top of SVG rendering for reliable click detection.
This preserves the beautiful Graphviz layout while adding precise interaction.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsScene


@dataclass
class HitZone:
    """A clickable area overlaid on the SVG.

    Attributes:
        layer_id: Logical ID of the layer this zone belongs to
        neuron_idx: Row index for detailed view
        -1 indicates the whole layer
        rect: Rectangle in scene pixel coordinates
        item: Backing QGraphicsRectItem stored for cleanup/highlight

    """

    layer_id: int
    neuron_idx: int  # -1 for whole layer
    rect: QRectF
    item: QGraphicsRectItem


class SVGHitOverlay(QObject):
    """Builds transparent hit zones on top of an SVG-rendered network.

    The overlay parses the SVG geometry produced by Graphviz and converts it
    into scene-space rectangles that precisely match the interactive region of
    each node. For detailed view, one zone per neuron row is created
    for simple
    view, a single zone that corresponds to the main body (excluding optional
    header/footer rows) is created.
    """

    node_clicked = pyqtSignal(int, int)  # layer_id, neuron_index

    def __init__(self) -> None:
        super().__init__()
        self.hit_zones: list[HitZone] = []
        self.scene: QGraphicsScene | None = None

    def parse_svg_and_create_overlay(
        self,
        svg_bytes: bytes,
        scene: QGraphicsScene,
        layers_config,
        simple_view: bool = True,
        debug: bool = False,
        layer_color_hex: str = "#eae5be",
        show_layer_ids: bool = True,
        show_desc: bool = False,
    ) -> dict:
        """Parse the SVG and create invisible hit zones.

        Returns a dict describing the root viewBox and render size so the caller
        can perform coordinate mapping if needed.
        """
        self.scene = scene
        self.clear_hit_zones()

        SVG_NS = "http://www.w3.org/2000/svg"
        ns = {"svg": SVG_NS}

        try:
            root = ET.fromstring(svg_bytes)
        except Exception:
            return {}

        # Get viewBox for coordinate mapping
        viewbox_info = {}
        vb = root.attrib.get("viewBox")
        if vb:
            parts = [float(p) for p in vb.replace(",", " ").split()]
            if len(parts) == 4:
                viewbox_info["viewBox"] = tuple(parts)

        # Get render dimensions (in px)
        def _parse_len(s: str) -> float | None:
            if not s:
                return None
            s = s.strip().lower().replace("pt", "").replace("px", "")
            try:
                return float(s)
            except Exception:
                return None

        width = _parse_len(root.attrib.get("width", ""))
        height = _parse_len(root.attrib.get("height", ""))
        if width is not None and height is not None:
            viewbox_info["render_size"] = (float(width), float(height))
        # Fallbacks
        if "viewBox" not in viewbox_info and "render_size" in viewbox_info:
            # Assume origin at 0,0
            w, h = viewbox_info["render_size"]
            viewbox_info["viewBox"] = (0.0, 0.0, float(w), float(h))
        if "render_size" not in viewbox_info and "viewBox" in viewbox_info:
            # Assume 1:1 mapping
            vx, vy, vw, vh = viewbox_info["viewBox"]
            viewbox_info["render_size"] = (float(vw), float(vh))

        # Prepare mapping from SVG coords -> scene (pixel) coords
        vx, vy, vw, vh = viewbox_info["viewBox"]
        rw, rh = viewbox_info["render_size"]
        sx = (rw / vw) if vw else 1.0
        sy = (rh / vh) if vh else 1.0
        if debug:
            pass

        # Helper: parse transform strings
        def _parse_transform(s: str) -> tuple[float, float, float, float, float, float]:
            # Return matrix(a,b,c,d,e,f)
            if not s:
                return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            s = s.strip()
            a, b, c, d, e, f = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            import re

            # Support matrix(), translate(), scale()
            m = re.findall(r"(matrix|translate|scale)\(([^\)]*)\)", s)
            for kind, vals in m:
                parts = [float(p) for p in re.split(r"[ ,]+", vals.strip()) if p]
                if kind == "matrix" and len(parts) == 6:
                    a, b, c, d, e, f = parts
                elif kind == "translate":
                    tx = parts[0] if len(parts) >= 1 else 0.0
                    ty = parts[1] if len(parts) >= 2 else 0.0
                    # multiply current by translate
                    e += tx
                    f += ty
                elif kind == "scale":
                    sx = parts[0] if len(parts) >= 1 else 1.0
                    sy = parts[1] if len(parts) >= 2 else sx
                    a *= sx
                    d *= sy
            return (a, b, c, d, e, f)

        def _apply_matrix_to_point(M, x, y):
            a, b, c, d, e, f = M
            return (a * x + c * y + e, b * x + d * y + f)

        def _apply_matrix_to_rect(M, r: QRectF) -> QRectF:
            # transform 4 corners and take bbox
            pts = [
                _apply_matrix_to_point(M, r.left(), r.top()),
                _apply_matrix_to_point(M, r.right(), r.top()),
                _apply_matrix_to_point(M, r.right(), r.bottom()),
                _apply_matrix_to_point(M, r.left(), r.bottom()),
            ]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

        def _mat_mul(M1, M2):
            # Compose affine matrices: result = M1 ∘ M2
            a1, b1, c1, d1, e1, f1 = M1
            a2, b2, c2, d2, e2, f2 = M2
            return (
                a1 * a2 + c1 * b2,
                b1 * a2 + d1 * b2,
                a1 * c2 + c1 * d2,
                b1 * c2 + d1 * d2,
                a1 * e2 + c1 * f2 + e1,
                b1 * e2 + d1 * f2 + f1,
            )

        # Capture root graph transform (Graphviz often flips Y)
        root_graph = root.find('.//svg:g[@id="graph0"]', ns)
        root_M = _parse_transform(root_graph.attrib.get("transform", "")) if root_graph is not None else (1, 0, 0, 1, 0, 0)

        # Normalize target layer color
        def _norm_hex(s: str) -> str:
            s = (s or "").strip().lower()
            if s.startswith("#") and len(s) in (4, 7):
                if len(s) == 4:
                    s = "#" + "".join([ch * 2 for ch in s[1:]])
                return s
            if s.startswith("rgb"):
                try:
                    import re

                    nums = [int(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", s)]
                    if len(nums) >= 3:
                        return f"#{nums[0]:02x}{nums[1]:02x}{nums[2]:02x}"
                except Exception:
                    pass
            return s

        target_color = _norm_hex(layer_color_hex)

        def _extract_fill(el) -> str:
            fill = (el.attrib.get("fill") or "").strip()
            if not fill:
                style = el.attrib.get("style") or ""
                # naive parse style="fill:#eae5be;stroke:..."
                for part in style.split(";"):
                    if ":" in part:
                        k, v = part.split(":", 1)
                        if k.strip() == "fill":
                            fill = v.strip()
                            break
            return _norm_hex(fill)

        # --- helper to detect neuron body bounds from fill rectangles ---
        def _find_neuron_body_bounds(fill_rects_px: list[QRectF], dim: int, debug: bool = False) -> QRectF | None:
            """Compute the bounding rectangle of the neuron body.

            New behaviour: If multiple rectangles with the target fill are present
            (as emitted by Graphviz for table cells in detailed view), take the
            union bounding box across them. This captures the full neuron stack
            area instead of a single row.
            """
            if not fill_rects_px:
                return None

            # If many rects are present, union them (common in detailed view)
            if len(fill_rects_px) >= 2:
                min_x = min(r.left() for r in fill_rects_px)
                min_y = min(r.top() for r in fill_rects_px)
                max_x = max(r.right() for r in fill_rects_px)
                max_y = max(r.bottom() for r in fill_rects_px)
                best_rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
            else:
                best_rect = fill_rects_px[0]

            # Shrink slightly to avoid including header/footer borders
            margin_y = min(12, best_rect.height() * 0.10)
            margin_x = min(6, best_rect.width() * 0.05)
            if best_rect.width() > 2 * margin_x and best_rect.height() > 2 * margin_y:
                best_rect = QRectF(
                    best_rect.x() + margin_x,
                    best_rect.y() + margin_y,
                    best_rect.width() - 2 * margin_x,
                    best_rect.height() - 2 * margin_y,
                )

            if debug:
                pass

            return best_rect

        # --- helper to find 90-degree corners for precise bounds ---
        def _detect_rectangular_bounds(g, M, target_color: str, debug: bool = False) -> list[QRectF]:
            """Detect rectangular regions by finding 90-degree corners in paths/polygons."""
            rectangles = []

            # Look for rectangular paths and polygons with the target fill
            for elem in g.findall(".//svg:polygon", ns) + g.findall(".//svg:rect", ns):
                fill = _extract_fill(elem)
                if not fill or fill != target_color:
                    continue

                if elem.tag.endswith("rect"):
                    # Direct rectangle - easy case
                    try:
                        x = float(elem.attrib.get("x", "0"))
                        y = float(elem.attrib.get("y", "0"))
                        w = float(elem.attrib.get("width", "0"))
                        h = float(elem.attrib.get("height", "0"))
                        rect_svg = QRectF(x, y, w, h)
                        rect_transformed = _apply_matrix_to_rect(M, rect_svg)
                        rectangles.append(rect_transformed)
                        if debug:
                            pass
                    except Exception:
                        continue

                elif elem.tag.endswith("polygon"):
                    # Polygon - check if it's rectangular
                    pts_attr = elem.attrib.get("points", "")
                    if not pts_attr:
                        continue

                    pts = []
                    for pair in pts_attr.strip().split():
                        if "," in pair:
                            try:
                                x_str, y_str = pair.split(",", 1)
                                pts.append((float(x_str), float(y_str)))
                            except Exception:
                                continue

                    if len(pts) >= 4:
                        # Check if this polygon forms a rectangle (4+ points with right angles)
                        # Transform points first
                        transformed_pts = [_apply_matrix_to_point(M, x, y) for x, y in pts]

                        # Find bounding box
                        if transformed_pts:
                            xs = [p[0] for p in transformed_pts]
                            ys = [p[1] for p in transformed_pts]
                            rect_transformed = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                            rectangles.append(rect_transformed)
                            if debug:
                                pass

            return rectangles

        # Find all node groups in SVG
        for g in root.findall('.//svg:g[@class="node"]', ns):
            title_el = g.find("svg:title", ns)
            if title_el is None or not title_el.text:
                continue

            # Parse layer ID from title
            t = title_el.text.strip()
            try:
                layer_id = int(t.split("_", 1)[1]) if t.startswith("layer_") else int(t)
            except Exception:
                continue

            # Get layer config for dimension early
            layer_cfg = next((c for c in layers_config if c.layer_id == layer_id), None)
            if not layer_cfg:
                continue
            dim = layer_cfg.params.get("dim", 1)

            # Compose group transform if present
            group_M = _parse_transform(g.attrib.get("transform", ""))
            M = _mat_mul(root_M, group_M)

            # Use the new rectangular bounds detection to find precise neuron body
            body_rects_transformed = _detect_rectangular_bounds(g, M, target_color, debug=debug)

            if not body_rects_transformed:
                # Fallback to old method if no rectangles found
                candidates: list[tuple[QRectF, str, float]] = []
                for pol in g.findall(".//svg:polygon", ns):
                    pts_attr = pol.attrib.get("points", "")
                    pts = []
                    for pair in pts_attr.strip().split():
                        if "," in pair:
                            xs, ys = pair.split(",", 1)
                            with contextlib.suppress(Exception):
                                pts.append((float(xs), float(ys)))
                    if len(pts) >= 3:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        r = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                        fill = _extract_fill(pol)
                        candidates.append((r, fill, r.width() * r.height()))
                        if fill and target_color and fill == target_color:
                            body_rects_transformed.append(_apply_matrix_to_rect(M, r))

                for relem in g.findall(".//svg:rect", ns):
                    try:
                        r = QRectF(
                            float(relem.attrib.get("x", "0")),
                            float(relem.attrib.get("y", "0")),
                            float(relem.attrib.get("width", "0")),
                            float(relem.attrib.get("height", "0")),
                        )
                        fill = _extract_fill(relem)
                        candidates.append((r, fill, r.width() * r.height()))
                        if fill and target_color and fill == target_color:
                            body_rects_transformed.append(_apply_matrix_to_rect(M, r))
                    except Exception:
                        pass

                if not body_rects_transformed and candidates:
                    # Last resort fallback
                    best = max(candidates, key=lambda t: t[2])[0]
                    body_rects_transformed = [_apply_matrix_to_rect(M, best)]

            if not body_rects_transformed:
                continue

            # Convert to pixel coordinates and find the neuron body bounds
            body_rects_px = []
            for rect_bounds in body_rects_transformed:
                rect_px = QRectF(
                    (rect_bounds.x() - vx) * sx,
                    (rect_bounds.y() - vy) * sy,
                    rect_bounds.width() * sx,
                    rect_bounds.height() * sy,
                )
                body_rects_px.append(rect_px)

            # Find the best neuron body rectangle
            neuron_body_rect = _find_neuron_body_bounds(body_rects_px, dim, debug=debug)
            if not neuron_body_rect:
                continue

            if debug:
                pass

            if simple_view:
                # Use the detected neuron body as the single clickable region
                self._create_hit_zone(layer_id, -1, neuron_body_rect, debug=debug)
            else:
                # Detailed view: divide the neuron body into equal rows
                row_height = neuron_body_rect.height() / max(1, dim)
                for i in range(dim):
                    row_rect = QRectF(
                        neuron_body_rect.x(),
                        neuron_body_rect.y() + i * row_height,
                        neuron_body_rect.width(),
                        row_height,
                    )
                    self._create_hit_zone(layer_id, i, row_rect, debug=debug)

        return viewbox_info

    def _create_hit_zone(self, layer_id: int, neuron_idx: int, rect: QRectF, debug: bool = False) -> None:
        """Create an invisible rectangle used for hit detection."""
        if not self.scene:
            return

        # Create rectangle (visible in debug mode)
        item = QGraphicsRectItem(rect)
        if debug:
            # Make hit zones visible for debugging
            item.setPen(QPen(QColor(255, 0, 0, 100), 1))
            item.setBrush(QBrush(QColor(255, 0, 0, 30)))
        else:
            # Invisible for normal use
            item.setPen(QPen(Qt.PenStyle.NoPen))
            item.setBrush(QBrush(Qt.GlobalColor.transparent))
        item.setData(0, ("hit", layer_id, neuron_idx))
        item.setZValue(1000)  # Above everything else
        self.scene.addItem(item)

        zone = HitZone(
            layer_id=layer_id,
            neuron_idx=neuron_idx,
            rect=rect,
            item=item,
        )
        self.hit_zones.append(zone)

    def clear_hit_zones(self) -> None:
        """Remove all hit zones from the scene."""
        if self.scene:
            for zone in self.hit_zones:
                try:
                    # Check if item still exists before removing
                    if zone.item.scene() == self.scene:
                        self.scene.removeItem(zone.item)
                except RuntimeError:
                    # Item was already deleted
                    pass
        self.hit_zones.clear()

    def handle_click(self, scene_point: QPointF, debug: bool = False) -> tuple[int, int]:
        """Return (layer_id, neuron_idx) for a click, or (-1, -1) if none."""
        if debug:
            pass

        for zone in self.hit_zones:
            if zone.rect.contains(scene_point):
                if debug:
                    pass
                self.node_clicked.emit(zone.layer_id, zone.neuron_idx)
                return zone.layer_id, zone.neuron_idx

        if debug and self.hit_zones:
            # Show nearest zone for debugging
            min_dist = float("inf")
            nearest = None
            for zone in self.hit_zones:
                center = zone.rect.center()
                dx = center.x() - scene_point.x()
                dy = center.y() - scene_point.y()
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest = zone
            if nearest:
                pass

        return -1, -1

    def highlight_zone(self, layer_id: int, neuron_idx: int = -1) -> None:
        """Temporarily highlight a specific zone for debugging."""
        for zone in self.hit_zones:
            if zone.layer_id == layer_id:
                if neuron_idx in (-1, zone.neuron_idx):
                    # Briefly show a colored border
                    zone.item.setPen(QPen(QColor(0, 255, 0, 100), 2))
                    zone.item.setBrush(QBrush(QColor(0, 255, 0, 30)))
                    # Could add a timer to remove highlight after 500ms
                    break
