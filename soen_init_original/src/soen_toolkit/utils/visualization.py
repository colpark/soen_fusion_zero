# FILEPATH: src/soen_toolkit/utils/visualization.py
import contextlib
import itertools
import os
import warnings

import graphviz

# Lazy import matplotlib only when the grid view is used to avoid import cost
try:
    from matplotlib.patches import (
        FancyArrowPatch as _FancyArrowPatch,
        FancyBboxPatch as _FancyBboxPatch,
        Rectangle as _Rectangle,
    )
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:  # pragma: no cover - matplotlib not always present in minimal envs
    np = None
    plt = None
    _FancyArrowPatch = None
    _Rectangle = None
    _FancyBboxPatch = None


def _inject_descriptions_svg(svg_bytes: bytes, layer_id_to_desc: dict, bg_color: str = "white") -> bytes:
    """Overlay layer descriptions as SVG text placed just below each node.
    This mirrors the GUI's approach to avoid affecting Graphviz layout.
    """
    try:
        from xml.etree import ElementTree as ET

        SVG_NS = "http://www.w3.org/2000/svg"
        ns = {"svg": SVG_NS}
        root = ET.fromstring(svg_bytes)

        # Pick a contrasting text color for overlays based on bg_color
        def _hex_to_rgb(s: str):
            s = (s or "").strip().lstrip("#")
            if len(s) == 3:
                s = "".join(ch * 2 for ch in s)
            try:
                return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
            except Exception:
                return (255, 255, 255)

        def _lum(rgb):
            r, g, b = [c / 255.0 for c in rgb]

            def _lin(x):
                return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

            rl, gl, bl = _lin(r), _lin(g), _lin(b)
            return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl

        text_fill = "#000000"
        try:
            text_fill = "#ffffff" if _lum(_hex_to_rgb(bg_color)) < 0.5 else "#000000"
        except Exception:
            text_fill = "#000000"

        if not any((d or "").strip() for d in layer_id_to_desc.values()):
            return svg_bytes

        max_text_baseline = 0.0
        for g in root.findall('.//svg:g[@class="node"]', ns):
            title_el = g.find("svg:title", ns)
            if title_el is None or not title_el.text:
                continue
            try:
                lid = int(title_el.text)
            except Exception:
                continue
            desc = (layer_id_to_desc.get(lid) or "").strip()
            if not desc:
                continue

            poly = g.find(".//svg:polygon", ns)
            if poly is None:
                continue
            points_attr = poly.attrib.get("points", "").strip()
            if not points_attr:
                continue

            xs, ys = [], []
            for pair in points_attr.split():
                if not pair or "," not in pair:
                    continue
                x_str, y_str = pair.split(",")
                try:
                    xs.append(float(x_str))
                    ys.append(float(y_str))
                except ValueError:
                    continue
            if not xs or not ys:
                continue

            min_x, max_x = min(xs), max(xs)
            max_y = max(ys)
            center_x = (min_x + max_x) / 2.0
            # Position the description baseline slightly below the node box
            baseline_y = max_y + 12.0
            max_text_baseline = max(max_text_baseline, baseline_y)

            # Draw a background mask rectangle under the description text to
            # fully cover any residual borders/edges of the layer box below.
            try:
                pad_x = 4.0
                mask_top = max_y + 1.0
                mask_bottom = baseline_y + 6.0  # a bit below the text baseline
                mask = ET.Element(
                    f"{{{SVG_NS}}}rect",
                    {
                        "x": f"{(min_x - pad_x):.2f}",
                        "y": f"{mask_top:.2f}",
                        "width": f"{(max_x - min_x + 2 * pad_x):.2f}",
                        "height": f"{(mask_bottom - mask_top):.2f}",
                        "fill": bg_color,
                        "stroke": "none",
                    },
                )
                g.append(mask)
            except Exception:
                pass

            text_el = ET.Element(
                f"{{{SVG_NS}}}text",
                {
                    "x": f"{center_x:.2f}",
                    "y": f"{baseline_y:.2f}",
                    "font-size": "10",
                    "font-style": "italic",
                    "text-anchor": "middle",
                    "fill": text_fill,
                },
            )
            text_el.text = desc
            g.append(text_el)

        try:
            vb = root.attrib.get("viewBox")
            if vb:
                parts = [float(p) for p in vb.replace(",", " ").split()]
                if len(parts) == 4:
                    _x0, y0, _w, h = parts
                    extra = max(0.0, (max_text_baseline + 14.0) - (y0 + h))
                    if extra > 0:
                        parts[3] = h + extra
                        root.set("viewBox", f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")

                        # Also bump height attribute if present
                        def _bump(attr_name: str) -> None:
                            s = root.attrib.get(attr_name)
                            if not s:
                                return
                            num = ""
                            unit = ""
                            for ch in s:
                                if ch.isdigit() or ch == "." or (ch == "-" and not num):
                                    num += ch
                                else:
                                    unit = s[s.index(ch) :]
                                    break
                            try:
                                new_val = float(num) + extra
                                root.set(attr_name, f"{new_val}{unit}")
                            except Exception:
                                pass

                        _bump("height")
                        # Insert a background rect to cover added space matching bg_color
                        try:
                            for child in list(root):
                                if child.tag.endswith("polygon") or child.tag.endswith("rect"):
                                    attrs = child.attrib
                                    if attrs.get("fill", "").lower() in ("#ffffff", "white") and attrs.get("stroke", "none") in ("none", "transparent"):
                                        root.remove(child)
                            W, H = parts[2], parts[3]
                            bg = ET.Element(
                                f"{{{SVG_NS}}}rect",
                                {
                                    "x": "0",
                                    "y": "0",
                                    "width": f"{W}",
                                    "height": f"{H}",
                                    "fill": bg_color,
                                    "stroke": "none",
                                },
                            )
                            root.insert(0, bg)
                            style = root.attrib.get("style", "")
                            if "background-color" not in style:
                                root.set("style", (style + "; " if style else "") + f"background-color:{bg_color}")
                        except Exception:
                            pass
        except Exception:
            pass

        return ET.tostring(root, encoding="utf-8")
    except Exception:
        return svg_bytes


def visualize(
    model,
    save_path: str | None = None,
    file_format: str = "png",
    dpi: int = 300,
    open_file: bool = False,
    # GUI-aligned settings
    orientation: str = "LR",
    edge_routing: str = "true",
    edge_thickness: float = 0.5,
    arrow_size: float = 0.5,
    layer_spacing: float = 1.0,
    inter_color: str = "#000000",
    intra_color: str = "#ff0000",
    layer_color: str = "#eae5be",
    show_layer_outline: bool = False,
    show_intra: bool = True,
    show_desc: bool = False,
    show_conn_type: bool = False,
    show_node_ids: bool = False,
    show_layer_ids: bool = True,
    show_neuron_polarity: bool = False,
    show_connection_polarity: bool = False,
    # Modern styling options
    theme: str = "default",  # "default" or "modern"
    font_name: str | None = None,  # e.g., "DejaVu Sans", "Helvetica"
    title_font_size: int | None = None,  # e.g., 12–14
    desc_font_size: int | None = None,  # e.g., 9–10
    nodesep: float | None = None,  # additional node separation within ranks
    # Legacy/compat parameters (ignored or mapped when provided)
    edge_color: str = "black",
    bg_color: str = "white",
    simple_view: bool = True,
    show_descriptions: bool = False,
    show_internal: bool = True,
    suppress_warnings: bool = True,
):
    """Visualize the model architecture in a Graphviz diagram, aligned with the GUI look-and-feel.

    The function displays inline in notebooks. If `save_path` is provided, it also saves to disk;
    otherwise nothing is written and the function returns None.

    Args:
        model: SOEN model instance (must provide `layers_config`, `connections_config`, and masks).
        save_path: Base path (without extension) for the output. Defaults to `Figures/Network_Diagrams/saved_network_diagram`.
        file_format: Output format: "png" (default), "svg", "jpg", or "pdf". PDFs are saved only (not displayed inline).
        dpi: Dots per inch for rasterized outputs and Graphviz rendering.
        open_file: If True, open the saved file with the default OS viewer.

        orientation: Graph layout direction. "LR" (left→right) or "TB" (top→bottom).
        edge_routing: Graphviz splines mode: "true", "false", or "ortho".
        edge_thickness: Edge line width (in points).
        arrow_size: Arrow size scaling factor for edge arrowheads.
        layer_spacing: Rank separation between layer columns (inches).
        inter_color: Color for inter-layer edges.
        intra_color: Color for intra-layer (within-layer) edges/loops.
        layer_color: Background color for the layer boxes.
        show_layer_outline: Draw a thin outline around layer boxes.
        simple_view: If True (default), compact per-layer boxes. If False, render a port-per-neuron column (detailed view).
        show_intra: Show intra-layer connections (self-loops in simple view, within-layer edges in detailed view).
        show_desc: Show layer descriptions beneath nodes (overlaid, without impacting layout).
        show_conn_type: In simple view, label edges with `connection_type` where available.
        show_node_ids: In detailed view, show neuron indices for each port row.
        show_layer_ids: Show a title row with the layer ID for each layer column/box.
        show_neuron_polarity: In detailed view, color neurons by polarity (red=excitatory, blue=inhibitory). Default: False.
        show_connection_polarity: In detailed view, color edges by weight sign (red=positive/excitatory, blue=negative/inhibitory). Default: False.

        edge_color: Backward-compat. If customized (not "black") and `inter_color` is the default, it replaces `inter_color`.
        bg_color: Background color for the canvas.
        simple_view: Primary view toggle. True → simple view
        False → detailed view.
        show_descriptions: Backward-compat alias for `show_desc`.
        show_internal: Backward-compat alias for `show_intra`.
        suppress_warnings: If True (default), hide non-actionable connectivity warnings during visualization.

    Returns:
        str: Full filepath of the saved diagram, including the extension.

    Notes:
        - When `show_desc=True` in simple view, descriptions are drawn as SVG overlays to avoid impacting layout.
        - Intra-layer loops are drawn from explicit `connections_config` and, if absent, inferred from `model.connection_masks`.
        - In notebooks, the saved artifact is displayed inline for SVG/PNG/JPG.

    """
    # If no save_path is provided, we render in-memory and do not write files.

    # Helpers
    def _hex_to_rgb(hex_color: str):
        s = hex_color.strip().lower()
        s = s.removeprefix("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        try:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (r, g, b)
        except Exception:
            return (255, 255, 255)

    def _relative_luminance(rgb):
        def _to_lin(c):
            c = c / 255.0
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = rgb
        return 0.2126 * _to_lin(r) + 0.7152 * _to_lin(g) + 0.0722 * _to_lin(b)

    def _contrast_color(bg_hex: str) -> str:
        lum = _relative_luminance(_hex_to_rgb(bg_hex))
        # threshold ~0.5 for light vs dark backgrounds
        return "#000000" if lum > 0.5 else "#ffffff"

    # Theme presets (pre‑processing before color mapping)
    if theme == "dark":
        # Only override defaults; respect explicit user overrides
        if (bg_color or "white").lower() == "white":
            bg_color = "#121212"
        if inter_color == "#000000":
            inter_color = "#e5e5e5"
        if intra_color == "#ff0000":
            intra_color = "#7aa2f7"
        if layer_color == "#eae5be":
            layer_color = "#22262b"
        if font_name is None:
            font_name = "DejaVu Sans"
        if title_font_size is None:
            title_font_size = 13
        if desc_font_size is None:
            desc_font_size = 9

    # Backward-compat mapping
    # show_descriptions flag maps to show_desc
    if show_descriptions:
        show_desc = True
    # show_internal maps to show_intra
    if show_internal is not None:
        show_intra = bool(show_internal)
    # edge_color maps to inter_color when explicitly customized
    if inter_color == "#000000" and edge_color and edge_color != "black":
        inter_color = edge_color
    # If inter_color remains default, auto-contrast it with background
    if inter_color == "#000000":
        with contextlib.suppress(Exception):
            inter_color = _contrast_color(bg_color or "white")

    # Graph init
    dot = graphviz.Digraph(format=file_format)
    # Base graph attributes (keep background white for parity with GUI)
    dot.attr(
        rankdir=orientation,
        splines=edge_routing,
        ranksep=str(layer_spacing),
        dpi=str(dpi),
        bgcolor=bg_color or "white",
        fontname=font_name or "Arial",
    )
    # Ensure nodes and edges also use the same font
    dot.node_attr.update(fontname=font_name or "Arial")
    dot.edge_attr.update(fontname=font_name or "Arial")
    # Optional modern/dark theme defaults applied to graph attributes and edges
    if theme == "modern":
        if inter_color == "#000000":
            inter_color = "#2b2b2b"
        if intra_color == "#ff0000":
            intra_color = "#b0b0b0"
        if layer_color == "#eae5be":
            layer_color = "#f7f5e7"
        if nodesep is None:
            nodesep = 0.6
        if title_font_size is None:
            title_font_size = 13
        if desc_font_size is None:
            desc_font_size = 9
        if font_name is None:
            font_name = "DejaVu Sans"
        dot.edge_attr.update(arrowhead="vee")
    elif theme == "dark":
        dot.edge_attr.update(arrowhead="vee")

    if nodesep is not None:
        dot.graph_attr.update(nodesep=str(nodesep))

    # Set global fonts if provided
    if font_name:
        dot.graph_attr.update(fontname=font_name)
        dot.node_attr.update(fontname=font_name)
        dot.edge_attr.update(fontname=font_name)

    # Extra vertical margin when showing descriptions in simple view so overlay has room
    if show_desc and simple_view:
        dot.graph_attr.update(margin="0.25,0.40")
    else:
        dot.graph_attr.update(margin="0.25")

    # Global edge styling
    dot.edge_attr.update(
        penwidth=str(edge_thickness),
        arrowsize=str(arrow_size),
    )

    out_ext = (file_format or "").lower()

    # Helper to get layer dims with fallback
    def _layer_dim(cfg) -> int:
        if hasattr(cfg, "params") and isinstance(getattr(cfg, "params", None), dict):
            dim = cfg.params.get("dim")
            if isinstance(dim, int):
                return dim
        # Fallback to model.layer_nodes if available
        try:
            return int(model.layer_nodes[cfg.layer_id])
        except Exception:
            return 0

    # SIMPLE VIEW (compact per-layer boxes)
    if simple_view:
        # Precompute description map
        layers_cfg = list(getattr(model, "layers_config", []))
        desc_map = {cfg.layer_id: getattr(cfg, "description", None) for cfg in layers_cfg}
        # Choose text colors based on theme
        title_color = None
        desc_color = None
        if theme == "dark":
            title_color = "#e8e8e8"
            desc_color = "#cfcfcf"

        for cfg in layers_cfg:
            title_parts = []
            if show_layer_ids:
                title_parts.append(f"Layer: {cfg.layer_id}")
            title_parts.append(f"Dim: {_layer_dim(cfg)}")
            # When showing descriptions, also include the layer type in the title block
            if show_desc:
                with contextlib.suppress(Exception):
                    title_parts.append(f"Type: {getattr(cfg, 'layer_type', '')}")
            _title_html = "<BR/>".join(title_parts)
            face = f' FACE="{font_name or "Arial"}"'
            size = f' POINT-SIZE="{title_font_size}"' if title_font_size else ""
            color = f' COLOR="{title_color}"' if title_color else ""
            title_html = f"<FONT{face}{size}{color}>{_title_html}</FONT>"

            desc_text = (desc_map.get(cfg.layer_id) or "").strip()
            desc_html = ""
            if show_desc and desc_text and out_ext != "svg":
                safe_desc = desc_text.replace("<", "&lt;").replace(">", "&gt;")
                sz = desc_font_size if desc_font_size is not None else 9
                color = f' COLOR="{desc_color}"' if desc_color else ""
                desc_html = f'<TR><TD BORDER="0" ALIGN="CENTER" BGCOLOR="{bg_color}"><FONT POINT-SIZE="{sz}"{face}{color}><I>{safe_desc}</I></FONT></TD></TR>'

            # Render text inside the tinted box (one-row table). This matches GUI visuals
            # and avoids the appearance of text sitting above an empty beige rectangle
            # seen in some notebook/SVG renderers.
            if desc_html:
                # Set table border color to improve contrast on dark themes
                border_color_attr = ' COLOR="#666666"' if theme == "dark" else ""
                html_label = (
                    f'<<TABLE BORDER="1" CELLBORDER="1" CELLPADDING="8" CELLSPACING="0"{border_color_attr}><TR><TD BGCOLOR="{layer_color}" ALIGN="CENTER">{title_html}</TD></TR>{desc_html}</TABLE>>'
                )
            else:
                border_color_attr = ' COLOR="#666666"' if theme == "dark" else ""
                html_label = f'<<TABLE BORDER="1" CELLBORDER="1" CELLPADDING="8" CELLSPACING="0"{border_color_attr}><TR><TD BGCOLOR="{layer_color}" ALIGN="CENTER">{title_html}</TD></TR></TABLE>>'

            outline_color = "black" if show_layer_outline else "transparent"
            node_kwargs = {"shape": "plaintext", "color": outline_color, "penwidth": "1", "margin": "0"}
            dot.node(str(cfg.layer_id), label=html_label, **node_kwargs)

        # Edges between layers from explicit configs
        seen_pairs = set()
        for conn in getattr(model, "connections_config", []):
            seen_pairs.add((conn.from_layer, conn.to_layer))
            color = intra_color if conn.from_layer == conn.to_layer else inter_color
            attrs = {"color": color}
            if conn.from_layer == conn.to_layer:
                if not show_intra:
                    continue
                attrs.update({"constraint": "false", "weight": "0", "minlen": "0"})
            elif conn.from_layer > conn.to_layer:
                attrs.update({"constraint": "false", "weight": "0", "minlen": "0"})
            # Only label inter-layer edges; skip labels on intra-layer (recurrent) loops
            if show_conn_type and (conn.from_layer != conn.to_layer):
                label = getattr(conn, "connection_type", "")
                if label:
                    attrs["label"] = label
                    attrs["fontsize"] = "12"
            if conn.from_layer == conn.to_layer:
                if orientation == "LR":
                    # Use east-side loop for better visibility
                    dot.edge(f"{conn.from_layer}:e", f"{conn.to_layer}:e", **attrs)
                else:
                    dot.edge(f"{conn.from_layer}:e", f"{conn.to_layer}:e", **attrs)
            else:
                dot.edge(f"{conn.from_layer}:e", f"{conn.to_layer}:w", **attrs)

        # Add intra-layer loops from masks if not already added via configs
        if show_intra:
            try:
                masks = getattr(model, "connection_masks", {})
                for cfg in layers_cfg:
                    lid = cfg.layer_id
                    if (lid, lid) in seen_pairs:
                        continue
                    m = masks.get(f"internal_{lid}")
                    try:
                        has_internal = (m is not None) and (m.sum().item() > 0)
                    except Exception:
                        has_internal = False
                    if not has_internal:
                        continue
                    attrs = {"color": intra_color, "constraint": "false", "weight": "0", "minlen": "0"}
                    # Always draw east-side loop for consistency and visibility
                    dot.edge(f"{lid}:e", f"{lid}:e", **attrs)
            except Exception:
                pass

    # DETAILED VIEW (one row per neuron with ports)
    else:
        layers = list(getattr(model, "layers_config", []))
        dims = {cfg.layer_id: _layer_dim(cfg) for cfg in layers}
        desc_map = {cfg.layer_id: getattr(cfg, "description", None) for cfg in layers}
        model_ids = {cfg.layer_id: getattr(cfg, "model_id", 0) for cfg in layers}

        # Load polarity data for each layer if show_neuron_polarity is enabled
        layer_polarity = {}
        if show_neuron_polarity:
            for cfg in layers:
                try:
                    if hasattr(cfg, "params") and isinstance(cfg.params, dict):
                        # Check for explicit polarity, polarity_file, or polarity_init
                        polarity_explicit = cfg.params.get("polarity")
                        polarity_file = cfg.params.get("polarity_file")
                        polarity_init = cfg.params.get("polarity_init")

                        if polarity_explicit is not None:
                            import torch
                            if isinstance(polarity_explicit, list):
                                layer_polarity[cfg.layer_id] = np.array(polarity_explicit) if np else polarity_explicit
                            elif isinstance(polarity_explicit, torch.Tensor):
                                layer_polarity[cfg.layer_id] = polarity_explicit.cpu().numpy() if np else polarity_explicit.cpu().tolist()
                            else:
                                layer_polarity[cfg.layer_id] = polarity_explicit
                        elif polarity_file:
                            from soen_toolkit.core.layers.common.connectivity_metadata import load_neuron_polarity
                            polarity_tensor = load_neuron_polarity(polarity_file)
                            layer_polarity[cfg.layer_id] = polarity_tensor.numpy() if np else polarity_tensor.tolist()
                        elif polarity_init:
                            from soen_toolkit.utils.polarity_utils import (
                                generate_alternating_polarity,
                                generate_excitatory_polarity,
                                generate_inhibitory_polarity,
                                generate_random_polarity,
                            )
                            dim = dims.get(cfg.layer_id, 0)
                            if isinstance(polarity_init, dict):
                                excitatory_ratio = polarity_init.get("excitatory_ratio", 0.8)
                                seed = polarity_init.get("seed")
                                polarity = generate_random_polarity(dim, excitatory_ratio=excitatory_ratio, seed=seed)
                            elif polarity_init in {"alternating", "50_50"}:
                                polarity = generate_alternating_polarity(dim)
                            elif polarity_init == "excitatory":
                                polarity = generate_excitatory_polarity(dim)
                            elif polarity_init == "inhibitory":
                                polarity = generate_inhibitory_polarity(dim)
                            else:
                                polarity = None
                            if polarity is not None:
                                layer_polarity[cfg.layer_id] = polarity
                except Exception:
                    # Skip polarity for this layer if loading fails
                    pass

        sorted_layer_ids = sorted(dims)
        offsets = {}
        running_total = 0
        for lid in sorted_layer_ids:
            offsets[lid] = running_total
            running_total += dims[lid]

        layer_node_ids = []
        for lid in sorted_layer_ids:
            label_rows = []
            if show_layer_ids:
                mid = model_ids.get(lid, 0)
                title = f"Layer {lid} (M{mid})"
                # Support type hint inline for context when show_desc
                if show_desc:
                    try:
                        lt = getattr(next(c for c in layers if c.layer_id == lid), "layer_type", "")
                        if lt:
                            title += f"  •  {lt}"
                    except Exception:
                        pass
                tcolor = "#e8e8e8" if theme == "dark" else None
                face = f' FACE="{font_name or "Arial"}"'
                size = f' POINT-SIZE="{title_font_size}"' if title_font_size else ""
                color = f' COLOR="{tcolor}"' if tcolor else ""
                # In TB mode, title needs to span all neuron columns
                colspan_attr = f' COLSPAN="{dims[lid]}"' if orientation == "TB" else ""
                label_rows.append(
                    f'<TR><TD{colspan_attr} BORDER="0" ALIGN="LEFT" BGCOLOR="{bg_color}">'
                    '<TABLE BORDER="0" CELLSPACING="0" CELLPADDING="0">'
                    f'<TR><TD ALIGN="LEFT"><B><FONT{face}{size}{color}>{title}</FONT></B></TD></TR>'
                    "</TABLE>"
                    "</TD></TR>",
                )
            num_color = ' COLOR="#e0e0e0"' if theme == "dark" else ""
            # Get polarity for this layer if available
            polarity = layer_polarity.get(lid) if show_neuron_polarity else None

            # Arrange neurons based on orientation
            if orientation == "TB":
                # Top-to-bottom: arrange neurons horizontally in a single row
                neuron_cells = []
                for i in range(dims[lid]):
                    port = f"p{i:04d}"
                    # Determine background color based on polarity
                    bgcolor_attr = ""
                    if polarity is not None and i < len(polarity):
                        polarity_val = polarity[i]
                        if polarity_val == 1:  # Excitatory
                            bgcolor_attr = ' BGCOLOR="#FF6B6B"'  # Red
                        elif polarity_val == -1:  # Inhibitory
                            bgcolor_attr = ' BGCOLOR="#5D9CEC"'  # Blue

                    if show_node_ids:
                        neuron_cells.append(
                            f'<TD PORT="{port}" WIDTH="18" HEIGHT="18" FIXEDSIZE="TRUE" ALIGN="CENTER"{bgcolor_attr}>'
                            f'<FONT FACE="{font_name or "Arial"}" POINT-SIZE="8"{num_color}>{i}</FONT></TD>'
                        )
                    else:
                        neuron_cells.append(f'<TD PORT="{port}" WIDTH="12" HEIGHT="12" FIXEDSIZE="TRUE"{bgcolor_attr}> </TD>')
                # Add all neurons as a single row
                label_rows.append(f'<TR>{"".join(neuron_cells)}</TR>')
            else:
                # Left-to-right: arrange neurons vertically (original behavior)
                for i in range(dims[lid]):
                    port = f"p{i:04d}"
                    # Determine background color based on polarity
                    bgcolor_attr = ""
                    if polarity is not None and i < len(polarity):
                        polarity_val = polarity[i]
                        if polarity_val == 1:  # Excitatory
                            bgcolor_attr = ' BGCOLOR="#FF6B6B"'  # Red
                        elif polarity_val == -1:  # Inhibitory
                            bgcolor_attr = ' BGCOLOR="#5D9CEC"'  # Blue

                    if show_node_ids:
                        label_rows.append(
                            f'<TR><TD PORT="{port}" WIDTH="16" HEIGHT="10" ALIGN="RIGHT"{bgcolor_attr}>'
                            f'<FONT FACE="{font_name or "Arial"}" POINT-SIZE="8"{num_color}>{i}</FONT></TD></TR>'
                        )
                    else:
                        label_rows.append(f'<TR><TD PORT="{port}" WIDTH="12" HEIGHT="6"{bgcolor_attr}> </TD></TR>')
            desc_text = (desc_map.get(lid) or "").strip()
            if show_desc and desc_text:
                safe_desc = desc_text.replace("<", "&lt;").replace(">", "&gt;")
                # Spacer row and description need COLSPAN in TB mode
                colspan_attr = f' COLSPAN="{dims[lid]}"' if orientation == "TB" else ""
                label_rows.append(f'<TR><TD{colspan_attr} BORDER="0" HEIGHT="2" BGCOLOR="{bg_color}"></TD></TR>')
                dcolor = ' COLOR="#cfcfcf"' if theme == "dark" else ""
                label_rows.append(
                    f'<TR><TD{colspan_attr} BORDER="0" ALIGN="LEFT" BGCOLOR="{bg_color}">'
                    '<TABLE BORDER="0" CELLSPACING="0" CELLPADDING="0">'
                    f'<TR><TD ALIGN="LEFT"><FONT FACE="{font_name or "Arial"}" POINT-SIZE="8"{dcolor}><I>{safe_desc}</I></FONT></TD></TR>'
                    "</TABLE>"
                    "</TD></TR>",
                )

            # Use CELLSPACING for TB orientation to ensure equal cell widths
            if orientation == "TB":
                html = f'<<TABLE BORDER="0" CELLBORDER="1" CELLPADDING="2" CELLSPACING="1" BGCOLOR="{layer_color}">{"".join(label_rows)}</TABLE>>'
            else:
                html = f'<<TABLE BORDER="0" CELLBORDER="1" CELLPADDING="2" CELLSPACING="0" BGCOLOR="{layer_color}">{"".join(label_rows)}</TABLE>>'
            outline_color = "black" if show_layer_outline else "transparent"
            with dot.subgraph(name=f"cluster_{lid}") as c:
                c.attr(style="rounded", color=outline_color, penwidth="1")
                node_id = f"layer_{lid}"
                c.node(node_id, label=html, shape="plaintext", margin="0")
                layer_node_ids.append(node_id)

        for a, b in itertools.pairwise(layer_node_ids):
            dot.edge(a, b, style="invis")

        # Edges from global connection matrix
        try:
            if suppress_warnings:
                # Suppress connectivity shape warnings that are not actionable in visualization
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*one_to_one connectivity with mismatched dimensions.*",
                        category=UserWarning,
                        module=r"soen_toolkit\.layers\.connectivity",
                    )
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module=r"soen_toolkit\.layers\.connectivity",
                    )
                    mat = model.get_global_connection_mask().detach().cpu().numpy()
            else:
                mat = model.get_global_connection_mask().detach().cpu().numpy()
        except Exception:
            mat = None

        # Get weight matrix if connection polarity is enabled
        weight_mat = None
        if show_connection_polarity and mat is not None:
            try:
                from soen_toolkit.utils.model_tools import create_global_connection_matrix
                weight_mat = create_global_connection_matrix(model).detach().cpu().numpy()
            except Exception:
                # If we can't get weights, fall back to regular coloring
                pass

        if mat is not None:
            total = mat.shape[0]
            for u in range(total):
                for v in range(total):
                    if mat[v, u] == 0:
                        continue
                    lu = next(layer_id for layer_id in sorted_layer_ids if offsets[layer_id] <= u < offsets[layer_id] + dims[layer_id])
                    lv = next(layer_id for layer_id in sorted_layer_ids if offsets[layer_id] <= v < offsets[layer_id] + dims[layer_id])
                    if not show_intra and lu == lv:
                        continue
                    src_port = f"p{u - offsets[lu]:04d}"
                    dst_port = f"p{v - offsets[lv]:04d}"
                    if lu == lv:
                        # Intra-layer: use side appropriate for orientation
                        if orientation == "TB":
                            src = f"layer_{lu}:{src_port}:s"
                            dst = f"layer_{lv}:{dst_port}:s"
                        else:
                            src = f"layer_{lu}:{src_port}:e"
                            dst = f"layer_{lv}:{dst_port}:e"
                    elif lu > lv:
                        # Backward/feedback connection: source from "beginning" side, destination at "end" side
                        if orientation == "TB":
                            src = f"layer_{lu}:{src_port}:n"
                            dst = f"layer_{lv}:{dst_port}:s"
                        else:
                            src = f"layer_{lu}:{src_port}:w"
                            dst = f"layer_{lv}:{dst_port}:e"
                    # Forward connection: use directional ports
                    elif orientation == "TB":
                        src = f"layer_{lu}:{src_port}:s"
                        dst = f"layer_{lv}:{dst_port}:n"
                    else:
                        src = f"layer_{lu}:{src_port}:e"
                        dst = f"layer_{lv}:{dst_port}:w"

                    # Determine edge color based on polarity or defaults
                    if show_connection_polarity and weight_mat is not None:
                        # Color by weight sign: red for positive (excitatory), blue for negative (inhibitory)
                        weight_value = weight_mat[v, u]
                        if weight_value > 0:
                            color = "#FF6B6B"  # Red for excitatory (positive)
                        elif weight_value < 0:
                            color = "#5D9CEC"  # Blue for inhibitory (negative)
                        else:
                            # Zero weight (shouldn't happen with mask, but handle it)
                            color = "#888888"  # Gray
                    else:
                        # Default coloring
                        color = intra_color if lu == lv else inter_color

                    attrs = {"color": color}
                    # Enhance visibility when showing polarity
                    if show_connection_polarity:
                        # Make edges slightly thicker and more opaque for better visibility
                        attrs["penwidth"] = str(float(edge_thickness) * 1.5)
                        attrs["style"] = "bold"
                    if lu == lv or lu > lv:
                        attrs.update({"constraint": "false", "weight": "0", "minlen": "0"})
                    dot.edge(src, dst, **attrs)

    # Render (optionally save) and display inline when possible
    output_filepath = None
    output_bytes = None
    if save_path:
        # For SVG in simple view with show_desc, inject overlays for exact GUI parity
        if out_ext == "svg" and show_desc and simple_view:
            svg_bytes = dot.pipe(format="svg")
            try:
                layer_id_to_desc = {cfg.layer_id: (getattr(cfg, "description", None) or "") for cfg in getattr(model, "layers_config", [])}
            except Exception:
                layer_id_to_desc = {}
            svg_bytes = _inject_descriptions_svg(svg_bytes, layer_id_to_desc, bg_color=bg_color or "white")
            final_path = f"{save_path}.svg" if not save_path.lower().endswith(".svg") else save_path
            os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
            with open(final_path, "wb") as f:
                f.write(svg_bytes)
            output_filepath = final_path
        else:
            output_filepath = dot.render(filename=save_path, cleanup=True)
    else:
        # Memory-only render, nothing written
        fmt = out_ext or file_format
        if fmt == "svg" and show_desc and simple_view:
            svg_bytes = dot.pipe(format="svg")
            try:
                layer_id_to_desc = {cfg.layer_id: (getattr(cfg, "description", None) or "") for cfg in getattr(model, "layers_config", [])}
            except Exception:
                layer_id_to_desc = {}
            output_bytes = _inject_descriptions_svg(svg_bytes, layer_id_to_desc, bg_color=bg_color or "white")
        else:
            output_bytes = dot.pipe(format=fmt)

    # Inline notebook display
    try:
        from IPython import get_ipython

        ipy = get_ipython()
        if ipy:
            from IPython.display import SVG, Image, display

            if output_bytes is not None:
                fmt = (out_ext or file_format).lower()
                if fmt == "svg":
                    display(SVG(output_bytes))
                elif fmt in {"png", "jpg", "jpeg"}:
                    display(Image(data=output_bytes))
            elif output_filepath:
                _, ext = os.path.splitext(output_filepath or "")
                ext = ext.lower()
                if ext == ".svg":
                    with open(output_filepath, "rb") as f:
                        display(SVG(f.read()))
                elif ext in {".png", ".jpg", ".jpeg"}:
                    display(Image(filename=output_filepath))
            # Skip inline display for PDFs
    except Exception:
        pass

    if open_file and output_filepath:
        try:
            if os.name == "posix":
                os.system(f"open {output_filepath}")
            else:
                os.startfile(output_filepath)
        except Exception:
            pass

    return output_filepath


# -----------------------------------------------------------------------------
# New Matplotlib visualisation: Grid-of-Grids (Layer-as-Module view)
# -----------------------------------------------------------------------------
def _compute_hierarchical_positions(
    num_nodes: int,
    levels: int,
    base_size: int,
) -> np.ndarray:
    """Compute node positions hierarchically respecting block structure.

    Uses a simple, clear algorithm: each node's position is the sum of offsets
    from each hierarchical level, where blocks at each level are arranged in a
    square-ish grid.

    Example: levels=3, base_size=5 (125 nodes):
    - Each node belongs to a block at each of 3 levels
    - Blocks are arranged in sqrt(base_size) × sqrt(base_size) grids
    - Final position = sum of (block_position × scale) at each level

    Args:
        num_nodes: Total number of nodes (must equal base_size^levels)
        levels: Number of hierarchical tiers
        base_size: Base block size

    Returns:
        np.ndarray: Node positions of shape (num_nodes, 2) with hierarchical layout
    """
    if num_nodes != base_size**levels:
        raise ValueError(f"num_nodes ({num_nodes}) must equal base_size^levels ({base_size}^{levels}={base_size**levels})")

    positions = np.zeros((num_nodes, 2), dtype=float)

    # Determine grid dimensions for arranging base_size blocks at each level
    # Use roughly square layout
    grid_cols = int(np.ceil(np.sqrt(base_size)))

    # Spacing factor to make blocks more compact (reduces whitespace between blocks)
    spacing_factor = 1.05  # Adds just 5% space between blocks for tight, compact layout

    # For each node, compute position from hierarchical address
    for node_id in range(num_nodes):
        x, y = 0.0, 0.0

        # Decompose node_id into block indices at each level
        for level in range(levels):
            # Size of blocks at this level
            block_size = base_size ** level

            # Which block (0 to base_size-1) does this node belong to at this level?
            block_index = (node_id // block_size) % base_size

            # Convert block index to 2D grid position
            block_row = block_index // grid_cols
            block_col = block_index % grid_cols

            # Add offset for this level (scaled by block size with compact spacing)
            x += block_col * block_size * spacing_factor
            y += block_row * block_size * spacing_factor

        positions[node_id] = [x, y]

    return positions


def _get_layer_hierarchical_info(connections_config: list, layer_id: int) -> tuple[int, int] | None:
    """Extract hierarchical structure info from intra-layer connection's visualization metadata.

    Args:
        connections_config: List of ConnectionConfig objects
        layer_id: Layer ID to check

    Returns:
        Tuple of (levels, base_size) if found, None otherwise
    """
    for conn in connections_config:
        if conn.from_layer == layer_id and conn.to_layer == layer_id:
            # Found intra-layer connection - check visualization_metadata
            if conn.visualization_metadata and "hierarchical" in conn.visualization_metadata:
                hier_info = conn.visualization_metadata["hierarchical"]
                levels = hier_info.get("levels")
                base_size = hier_info.get("base_size")
                if levels is not None and base_size is not None:
                    return (int(levels), int(base_size))
    return None


def _draw_block_structure(
    ax,
    all_positions: dict,
    dims: dict,
    layer_ids: list,
    layer_hierarchical_info: dict[int, tuple[int, int] | None],
    block_linewidth: float,
    block_alpha: float,
    block_colors: list | None,
    node_square_size: float,
    theme: str,
) -> None:
    """Draw hierarchical block boundaries within each module.

    Draws nested rectangles representing each tier of the hierarchical structure.
    Smaller tiers (more detailed) are drawn with higher alpha, larger tiers with lower alpha.
    Uses per-layer hierarchical info to support different structures per layer.
    """
    # For each layer/module
    for lid in layer_ids:
        # Get layer-specific hierarchical info
        hier_info = layer_hierarchical_info.get(lid)
        if hier_info is None:
            continue

        levels, base_size = hier_info
        dim = dims[lid]

        if dim <= 0:
            continue

        # Check if dimension matches hierarchical structure
        expected_dim = base_size**levels
        if dim != expected_dim:
            continue

        pos = all_positions[lid]  # shape (dim, 2)

        # Get block colors for this specific layer's level count
        layer_block_colors = block_colors
        if layer_block_colors is None:
            # Default color palette: grays for light theme, whites for dark theme
            if theme == "dark":
                layer_block_colors = ["#ffffff", "#e8e8e8", "#d0d0d0", "#b8b8b8"][:levels]
            else:
                layer_block_colors = ["#606060", "#808080", "#a0a0a0", "#c0c0c0"][:levels]

        # Draw blocks from largest tier (least detailed) to smallest tier (most detailed)
        # This way smaller blocks are drawn on top with higher z-order
        for tier_idx in range(levels - 1, -1, -1):
            tier_size = base_size ** (tier_idx + 1)
            num_blocks = dim // tier_size

            # For each block at this tier
            for block_idx in range(num_blocks):
                block_start = block_idx * tier_size
                block_end = block_start + tier_size

                # Get node positions in this block
                block_nodes = np.arange(block_start, block_end)
                block_pos = pos[block_nodes]

                # Compute bounding box from node positions
                x_coords = block_pos[:, 0]
                y_coords = block_pos[:, 1]
                x_min = float(np.min(x_coords)) - node_square_size / 2.0
                x_max = float(np.max(x_coords)) + node_square_size / 2.0
                y_min = float(np.min(y_coords)) - node_square_size / 2.0
                y_max = float(np.max(y_coords)) + node_square_size / 2.0

                width = x_max - x_min
                height = y_max - y_min

                # Choose color
                if tier_idx < len(layer_block_colors):
                    color = layer_block_colors[tier_idx]
                else:
                    color = "#808080"

                # Alpha decreases with tier (larger blocks more transparent)
                # tier_idx=0 (smallest blocks) get highest alpha, tier_idx=levels-1 get lowest
                alpha = block_alpha * (1.0 - 0.3 * (tier_idx / max(1, levels - 1)))

                if _Rectangle is not None:
                    rect = _Rectangle(
                        (x_min, y_min),
                        width,
                        height,
                        facecolor="none",
                        edgecolor=color,
                        linewidth=block_linewidth,
                        alpha=alpha,
                        zorder=1 + tier_idx,  # Smaller blocks drawn on top
                    )
                    ax.add_patch(rect)


def visualize_grid_of_grids(
    model,
    save_path: str | None = None,
    file_format: str = "png",
    dpi: int = 200,
    show: bool = True,
    *,
    bg_color: str = "white",
    module_bg_color: str = "#B8D4D8",
    module_bg_color_input: str = "#D6D9DD",
    module_bg_color_output: str = "#CFE3FF",
    module_border_color: str = "#1a3a52",
    module_border_linewidth: float = 2.0,
    module_border_alpha: float = 0.85,
    node_fill_color: str = "#C89FA3",
    node_edge_color: str = "white",
    node_edge_linewidth: float = 0.8,
    node_square_size: float = 0.36,
    intra_edge_color: str = "black",
    intra_edge_width: float = 1.0,
    intra_edge_alpha: float = 0.6,
    inter_edge_color: str = "black",
    inter_edge_width: float = 1.8,
    inter_edge_alpha: float = 0.5,
    # Edge routing
    edge_routing: str = "straight",
    edge_routing_rad_scale: float = 0.2,
    # New options for theme and labels
    theme: str = "default",
    font_name: str | None = None,
    title_font_size: float = 14.0,
    show_layer_ids: bool = False,
    show_layer_types: bool = False,
    show_dimensions: bool = False,
    show_descriptions: bool = True,
    show_node_ids: bool = False,
    # New options for block structure visualization
    show_block_structure: bool = False,
    levels: int | None = None,
    base_size: int | None = None,
    block_linewidth: float = 1.0,
    block_alpha: float = 0.4,
    block_colors: list[str] | None = None,
) -> str | None:
    """Render the network as a grid of grids where each layer is a module.

    This visualization displays each network layer as a module (colored box) arranged in a grid.
    Nodes are shown as small squares within each module, with connections drawn between them.
    Optionally supports hierarchical block structure visualization for networks created with
    create_hierarchical_mask().

    Args:
        model: SOEN model instance with layers_config and connection information.
        save_path: Base path (without extension) for saving visualization. If None, only displays inline.
        file_format: Output format - "png" (default), "svg", "jpg", or "pdf". Note: PDFs display inline.
        dpi: Dots per inch for rasterized output (default: 200).
        show: If True (default), display inline in notebooks. If False, only save to file.

    Module Styling Arguments:
        bg_color: Background color of the canvas (default: "white").
        module_bg_color: Background color for middle-tier modules (default: "#B8D4D8").
        module_bg_color_input: Background color for the input (first) layer module (default: "#D6D9DD").
        module_bg_color_output: Background color for the output (last) layer module (default: "#CFE3FF").
        module_border_color: Color of module borders (default: "#1a3a52" dark blue).
        module_border_linewidth: Width of module border lines (default: 2.0).
        module_border_alpha: Transparency of module borders (default: 0.85, range 0-1).

    Node Styling Arguments:
        node_fill_color: Fill color for node squares (default: "#C89FA3").
        node_edge_color: Color of node borders (default: "white").
        node_edge_linewidth: Width of node border lines (default: 0.8).
        node_square_size: Size of node squares in plot units (default: 0.36).
            Automatically increased to 0.8 for hierarchical block-structured layers.

    Connection Styling Arguments:
        intra_edge_color: Color for within-layer connections/loops (default: "black", "#7aa2f7" in dark theme).
        intra_edge_width: Line width for within-layer connections (default: 1.0).
        intra_edge_alpha: Transparency for within-layer connections (default: 0.6, range 0-1).
        inter_edge_color: Color for between-layer connections (default: "black", "#e5e5e5" in dark theme).
        inter_edge_width: Line width for between-layer connections (default: 1.8).
        inter_edge_alpha: Transparency for between-layer connections (default: 0.5, range 0-1).

    Edge Routing Arguments:
        edge_routing: Style for drawing edges - "straight" (default), "curved", or "orthogonal".
            - "straight": Direct lines from source to destination (default, clearest)
            - "curved": Smooth arc connections
            - "orthogonal": Right-angle paths (subway-map style)
            Default: "straight"
        edge_routing_rad_scale: Curvature scale for "curved" routing (default: 0.2).
            Higher values = more curved edges. Only used when edge_routing="curved".

    Theme and Label Arguments:
        theme: Visual theme preset - "default" or "dark" (default: "default").
        font_name: Font family for text (e.g., "DejaVu Sans", "Helvetica"). If None, uses system default.
        title_font_size: Font size for layer labels (default: 14.0).
        show_layer_ids: If True, display layer ID labels (e.g., "L0"). Default: False.
        show_layer_types: If True, display layer type (e.g., "SingleDendrite"). Default: False.
        show_dimensions: If True, display layer dimensions (e.g., "Dim: 10"). Default: False.
        show_descriptions: If True, display layer descriptions. Default: True.
        show_node_ids: If True, display node index inside each node square. Default: False.

        Note: Label elements are stacked vertically below each module in the order: IDs, types,
        dimensions, descriptions. Only enabled elements are shown.

    Hierarchical Block Structure Arguments:
        show_block_structure: If True, show hierarchical block organization with nested rectangles
            and reorganize nodes spatially within their blocks (default: False).
        levels: Optional global fallback for hierarchical tiers (e.g., 3 for base_size^3 nodes).
            Used for layers without metadata. Per-layer metadata (from structure.custom()) takes precedence.
            Default: None.
        base_size: Optional global fallback base block size (e.g., 4). Total nodes must equal base_size^levels.
            Used for layers without metadata. Per-layer metadata (from structure.custom()) takes precedence.
            Default: None.
        block_linewidth: Width of hierarchical block boundary lines (default: 1.0).
        block_alpha: Base transparency level for block boundaries, adjusted per tier (default: 0.4, range 0-1).
            Larger tiers get lower alpha so all levels remain visible.
        block_colors: List of colors for each tier (tier 0, tier 1, ...). If None, uses gray palette
            ["#606060", "#808080", "#a0a0a0", "#c0c0c0"]. Example: ["#FF0000", "#00FF00", "#0000FF"]
            for red, green, blue tiers.

        Note: Hierarchical structure can be specified per-layer via visualization_metadata:
            >>> g.connect(0, 0,
            ...           structure=structure.custom(
            ...               "mask.npz",
            ...               visualization_metadata={"hierarchical": {"levels": 3, "base_size": 4}}
            ...           ),
            ...           init=init.constant(0.1))
        This allows different layers to have different hierarchical structures and is fully
        extensible for future visualization features.

    Returns:
        str: Full file path of the saved visualization (if save_path provided), or None if only displayed inline.

    Notes:
        - Layer view treats each layer as a single module (ignores model_id/module_id)
        - Module background color indicates role: input (first) layer, output (last) layer, or middle layer
        - Within-layer edges are drawn from intra-layer connection matrix
        - Between-layer edges are drawn from inter-layer connection blocks
        - Hierarchical positioning (when show_block_structure=True):
            * Nodes are physically organized within their hierarchical blocks
            * Each block attempts square aspect ratio for clarity
            * Nested rectangles show structure of each tier level
            * Useful for visualizing networks from create_hierarchical_mask()

    Example:
        >>> # Basic visualization
        >>> g.visualize_grid_of_grids()

        >>> # Save to file
        >>> path = g.visualize_grid_of_grids(save_path="my_network")

        >>> # With hierarchical block structure
        >>> g.visualize_grid_of_grids(
        ...     show_block_structure=True,
        ...     levels=3,
        ...     base_size=4,
        ...     block_colors=["#FF6B6B", "#4ECDC4", "#95E1D3"]
        ... )

        >>> # Dark theme with custom styling
        >>> g.visualize_grid_of_grids(
        ...     theme="dark",
        ...     inter_edge_width=2.0,
        ...     module_border_linewidth=3.0,
        ...     save_path="dark_network"
        ... )

    """
    if np is None or plt is None:
        msg = "Matplotlib/Numpy are required for visualize_grid_of_grids but are not available in this environment."
        raise RuntimeError(
            msg,
        )

    # Apply theme presets (dark mode)
    if theme == "dark":
        if (bg_color or "white").lower() == "white":
            bg_color = "#121212"
        if module_bg_color == "#B8D4D8":
            module_bg_color = "#2a2e35"
        if module_bg_color_input == "#D6D9DD":
            module_bg_color_input = "#30343a"
        if module_bg_color_output == "#CFE3FF":
            module_bg_color_output = "#2f3b52"
        if inter_edge_color == "black":
            inter_edge_color = "#e5e5e5"
        if intra_edge_color == "black":
            intra_edge_color = "#7aa2f7"
        if node_fill_color == "#C89FA3":
            node_fill_color = "#4c5967"

    # Collect layer dims and polarity info in ID order
    layers_cfg = list(getattr(model, "layers_config", []))
    if not layers_cfg:
        msg = "Model has no layers_config to visualize."
        raise ValueError(msg)
    layer_ids = [cfg.layer_id for cfg in layers_cfg]
    dims = {}
    layer_polarity = {}  # Store polarity data per layer
    for cfg in layers_cfg:
        dim = None
        try:
            if hasattr(cfg, "params") and isinstance(cfg.params, dict):
                dim = cfg.params.get("dim")
                # Load polarity if specified
                polarity_explicit = cfg.params.get("polarity")
                polarity_file = cfg.params.get("polarity_file")
                polarity_init = cfg.params.get("polarity_init")

                if polarity_explicit is not None:
                    import torch
                    if isinstance(polarity_explicit, list):
                        layer_polarity[cfg.layer_id] = np.array(polarity_explicit)
                    elif isinstance(polarity_explicit, torch.Tensor):
                        layer_polarity[cfg.layer_id] = polarity_explicit.cpu().numpy()
                    else:
                        layer_polarity[cfg.layer_id] = np.array(polarity_explicit)
                elif polarity_file:
                    from soen_toolkit.core.layers.common.connectivity_metadata import load_neuron_polarity
                    layer_polarity[cfg.layer_id] = load_neuron_polarity(polarity_file).numpy()
                elif polarity_init:
                    from soen_toolkit.utils.polarity_utils import (
                        generate_alternating_polarity,
                        generate_excitatory_polarity,
                        generate_inhibitory_polarity,
                        generate_random_polarity,
                    )

                    if isinstance(polarity_init, dict):
                        excitatory_ratio = polarity_init.get("excitatory_ratio", 0.8)
                        seed = polarity_init.get("seed")
                        polarity = generate_random_polarity(dim, excitatory_ratio=excitatory_ratio, seed=seed)
                    elif polarity_init in {"alternating", "50_50"}:
                        polarity = generate_alternating_polarity(dim)
                    elif polarity_init == "excitatory":
                        polarity = generate_excitatory_polarity(dim)
                    elif polarity_init == "inhibitory":
                        polarity = generate_inhibitory_polarity(dim)
                    else:
                        polarity = None
                    if polarity is not None:
                        layer_polarity[cfg.layer_id] = polarity
        except Exception:
            dim = None
        if not isinstance(dim, int):
            try:
                dim = int(model.layer_nodes[cfg.layer_id])
            except Exception:
                dim = 0
        dims[cfg.layer_id] = int(dim)

    # Build per-layer hierarchical structure info from connection metadata
    layer_hierarchical_info = {}
    connections_cfg = getattr(model, "connections_config", [])
    for lid in layer_ids:
        # Try to get metadata from connection config
        info = _get_layer_hierarchical_info(connections_cfg, lid)
        # Fall back to global params if no metadata found and global params provided
        if info is None and levels is not None and base_size is not None:
            # Check if dimension matches global structure
            if dims[lid] == base_size**levels:
                info = (levels, base_size)
        layer_hierarchical_info[lid] = info

    if not show_block_structure:
        # Automatically enable hierarchical visualization when metadata is present
        if any(info is not None for info in layer_hierarchical_info.values()):
            show_block_structure = True

    # Build global connection matrix (rows=to, cols=from)
    try:
        global_J = model.get_global_connection_mask().detach().cpu().numpy()
    except Exception:
        global_J = None

    # Compute compact grid (w,h) for a given number of nodes
    def _grid_shape(n: int) -> tuple[int, int]:
        if n <= 0:
            return (1, 1)
        w = int(np.ceil(np.sqrt(n)))
        h = int(np.ceil(n / float(w)))
        return (h, w)

    # Layout modules (layers) on a coarse grid
    num_layers = len(layer_ids)
    cols = int(np.ceil(np.sqrt(num_layers)))
    rows = int(np.ceil(num_layers / cols))
    module_spacing = 1 + max(_grid_shape(max(dims.values()))[0], _grid_shape(max(dims.values()))[1]) + 2

    module_positions: list[tuple[float, float]] = []
    for k in range(num_layers):
        r = k // cols
        c = k % cols
        module_positions.append((c * module_spacing, (rows - 1 - r) * module_spacing))

    # Precompute per-layer node positions (global coordinates) and IO sets
    all_positions = {}
    io_nodes = {}
    module_box = {}
    for idx, lid in enumerate(layer_ids):
        dim = dims[lid]
        (h, w) = _grid_shape(dim)
        ox, oy = module_positions[idx]

        # Get layer-specific hierarchical info
        hier_info = layer_hierarchical_info.get(lid)

        # Use hierarchical positioning if requested and layer has hierarchical info
        if show_block_structure and hier_info is not None:
            levels_for_layer, base_size_for_layer = hier_info
            if dim == base_size_for_layer**levels_for_layer:
                # Use hierarchical positioning
                try:
                    hierarchical_pos = _compute_hierarchical_positions(dim, levels_for_layer, base_size_for_layer)
                    # Scale and offset hierarchical positions to module position
                    positions = []
                    for i in range(dim):
                        x = ox + hierarchical_pos[i, 0]
                        y = oy + hierarchical_pos[i, 1]
                        positions.append((x, y))
                    all_positions[lid] = np.asarray(positions, dtype=float)
                except ValueError:
                    # Fall back to regular grid if hierarchical fails
                    positions = []
                    for k in range(dim):
                        row = k // w
                        col = k % w
                        x = ox + col
                        y = oy + (h - 1 - row)
                        positions.append((x, y))
                    all_positions[lid] = np.asarray(positions, dtype=float)
            else:
                # Dimensions don't match hierarchical structure, use regular grid
                positions = []
                for k in range(dim):
                    row = k // w
                    col = k % w
                    x = ox + col
                    y = oy + (h - 1 - row)
                    positions.append((x, y))
                all_positions[lid] = np.asarray(positions, dtype=float)
        else:
            # Standard grid positioning (original behavior)
            positions = []
            for k in range(dim):
                row = k // w
                col = k % w
                x = ox + col
                y = oy + (h - 1 - row)
                positions.append((x, y))
            all_positions[lid] = np.asarray(positions, dtype=float)

        # IO nodes: inputs=top row, outputs=last row (may be partial)
        n_top = min(w, dim)
        num_rows = int(np.ceil(dim / float(w)))
        nodes_before_last = (num_rows - 1) * w
        n_last = dim - nodes_before_last
        input_nodes = list(range(n_top))
        output_nodes = list(range(nodes_before_last, nodes_before_last + n_last))
        io_nodes[lid] = (input_nodes, output_nodes)

        # Module box geometry used for drawing – derive from actual node positions
        pos_arr = all_positions[lid]
        x_min = float(np.min(pos_arr[:, 0])) - 0.4
        x_max = float(np.max(pos_arr[:, 0])) + 0.4
        y_min = float(np.min(pos_arr[:, 1])) - 0.4
        y_max = float(np.max(pos_arr[:, 1])) + 0.4
        module_box[lid] = (x_min, y_min, x_max - x_min, y_max - y_min)

    # Figure sizing and view limits from actual drawn extents
    # Use the true module boxes (plus a little padding and label allowance)
    box_xmins = []
    box_xmaxs = []
    box_ymins = []
    box_ymaxs = []
    for x_b, y_b, w_b, h_b in module_box.values():
        box_xmins.append(x_b)
        box_xmaxs.append(x_b + w_b)
        box_ymins.append(y_b)
        box_ymaxs.append(y_b + h_b)

    # Extra padding to fully include connection arcs and the layer-id label below boxes
    label_pad = 0.5 if show_layer_ids else 0.2
    edge_pad = 0.6  # accounts for curved edges/arrow heads that extend outside boxes

    x_min_bound = min(box_xmins) - edge_pad
    x_max_bound = max(box_xmaxs) + edge_pad
    y_min_bound = min(box_ymins) - (edge_pad + label_pad)
    y_max_bound = max(box_ymaxs) + edge_pad

    x_range = x_max_bound - x_min_bound
    y_range = y_max_bound - y_min_bound
    fig_w = max(8, min(18, x_range * 1.2))
    fig_h = max(6, min(18, y_range * 1.2))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    # Draw module backgrounds and borders + labels
    first_lid = min(layer_ids)
    last_lid = max(layer_ids)
    for lid in layer_ids:
        x, y, w, h = module_box[lid]
        bg = module_bg_color
        if lid == first_lid:
            bg = module_bg_color_input or module_bg_color
        elif lid == last_lid:
            bg = module_bg_color_output or module_bg_color
        if _FancyBboxPatch is not None:
            bg_patch = _FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", facecolor=bg, edgecolor="none", linewidth=0, alpha=1.0, zorder=0)
            ax.add_patch(bg_patch)
            border = _FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", facecolor="none", edgecolor=module_border_color, linewidth=module_border_linewidth, alpha=module_border_alpha, zorder=0)
            ax.add_patch(border)
        # Build label parts based on enabled options
        label_parts = []
        if show_layer_ids:
            label_parts.append(f"L{lid}")
        if show_layer_types:
            # Get layer type from config
            layer_cfg = next((cfg for cfg in layers_cfg if cfg.layer_id == lid), None)
            if layer_cfg and hasattr(layer_cfg, "layer_type"):
                label_parts.append(str(layer_cfg.layer_type))
        if show_dimensions:
            label_parts.append(f"Dim: {dims[lid]}")
        if show_descriptions:
            # Get description from config
            layer_cfg = next((cfg for cfg in layers_cfg if cfg.layer_id == lid), None)
            if layer_cfg and hasattr(layer_cfg, "description") and layer_cfg.description:
                label_parts.append(str(layer_cfg.description))

        # Render label if any parts exist
        if label_parts:
            label_color = "#e8e8e8" if theme == "dark" else "black"
            label_text = "\n".join(label_parts)
            ax.text(
                x + w / 2.0,
                y - 0.35,
                label_text,
                fontsize=title_font_size,
                ha="center",
                va="top",
                color=label_color,
                alpha=0.9,
                fontname=font_name if font_name else None,
                fontweight="bold",
            )

    # Draw hierarchical block structure if requested
    # Check if any layer has hierarchical info
    if show_block_structure and any(info is not None for info in layer_hierarchical_info.values()):
        # Use larger node size for hierarchical blocks
        hierarchical_node_size = 0.8
        _draw_block_structure(ax, all_positions, dims, layer_ids, layer_hierarchical_info, block_linewidth, block_alpha, block_colors, hierarchical_node_size, theme)

    # Helper to draw an arrow between two node centers
    def _draw_edge(ax_, p1, p2, *, color, lw, alpha, same_module: bool, routing: str = "curved", rad_scale: float = 1.0) -> None:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = float((dx * dx + dy * dy) ** 0.5)
        if dist <= 0:
            return
        dxn = dx / dist
        dyn = dy / dist
        offset = 0.18
        x1 = p1[0] + dxn * offset
        y1 = p1[1] + dyn * offset
        x2 = p2[0] - dxn * offset
        y2 = p2[1] - dyn * offset

        if _FancyArrowPatch is not None:
            # Determine connection style based on routing parameter
            if routing == "straight":
                # Direct straight line
                connectionstyle = None
            elif routing == "orthogonal":
                # Right-angle path
                connectionstyle = "arc3,rad=0"
            else:  # "curved" (default)
                # Smooth arc with distance-dependent curvature
                rad = (0.1 + (dist / (5.0 if same_module else 10.0)) * (0.25 if same_module else 0.15)) * rad_scale
                connectionstyle = f"arc3,rad={rad}"

            kwargs = {"arrowstyle": "-|>", "mutation_scale": 8, "linewidth": lw, "color": color, "alpha": alpha, "zorder": 1 if same_module else 2, "clip_on": False}

            if connectionstyle is not None:
                kwargs["connectionstyle"] = connectionstyle

            arr = _FancyArrowPatch((x1, y1), (x2, y2), **kwargs)
            ax_.add_patch(arr)

    # Draw edges (intra first, then inter)
    if global_J is not None:
        # Compute layer index offsets in global matrix order (sorted layer_ids)
        sorted_ids = sorted(layer_ids)
        start = {}
        running = 0
        for lid in sorted_ids:
            start[lid] = running
            running += dims[lid]

        # Intra-layer edges from blocks J_i_to_i
        for lid in sorted_ids:
            d = dims[lid]
            if d <= 0:
                continue
            rs = start[lid]
            cs = start[lid]
            block = global_J[rs : rs + d, cs : cs + d]
            pos = all_positions[lid]
            nz = np.argwhere(block != 0)
            # Each row is [to_idx, from_idx] in local indices
            for ti, fj in nz:
                _draw_edge(ax, pos[fj], pos[ti], color=intra_edge_color, lw=intra_edge_width, alpha=intra_edge_alpha, same_module=True, routing=edge_routing, rad_scale=edge_routing_rad_scale)

        # Inter-layer edges between u (from) and v (to): draw ALL non-zero pairs
        for u in sorted_ids:
            for v in sorted_ids:
                if u == v:
                    continue
                du, dv = dims[u], dims[v]
                if du <= 0 or dv <= 0:
                    continue
                rs = start[v]
                cs = start[u]
                block = global_J[rs : rs + dv, cs : cs + du]
                if (block == 0).all():
                    continue
                pos_u = all_positions[u]
                pos_v = all_positions[v]
                # Iterate over all non-zero pairs (ti, fj)
                nz_pairs = np.argwhere(block != 0)
                for ti, fj in nz_pairs:
                    _draw_edge(ax, pos_u[fj], pos_v[ti], color=inter_edge_color, lw=inter_edge_width, alpha=inter_edge_alpha, same_module=False, routing=edge_routing, rad_scale=edge_routing_rad_scale)

    # Draw nodes last so they appear above edges
    # Color by polarity if available: green=excitatory, red=inhibitory, default=normal
    for lid in layer_ids:
        pos = all_positions[lid]
        d = dims[lid]
        if d <= 0:
            continue

        # Get polarity for this layer if it exists
        polarity = layer_polarity.get(lid)

        # Use larger nodes for hierarchical layers
        layer_node_size = node_square_size
        if layer_hierarchical_info.get(lid) is not None:
            layer_node_size = 0.8  # Larger nodes for hierarchical block structures

        for nid in range(d):
            x, y = pos[nid]
            # Determine node color based on polarity
            if polarity is not None and nid < len(polarity):
                if polarity[nid] == 1:  # Excitatory
                    color = "#66BB6A"  # Green
                elif polarity[nid] == -1:  # Inhibitory
                    color = "#EF5350"  # Red
                else:  # Normal (0)
                    color = node_fill_color
            else:
                color = node_fill_color

            if _Rectangle is not None:
                sq = _Rectangle(
                    (x - layer_node_size / 2.0, y - layer_node_size / 2.0), layer_node_size, layer_node_size, facecolor=color, edgecolor=node_edge_color, linewidth=node_edge_linewidth, zorder=3
                )
                ax.add_patch(sq)

            # Draw node ID label if enabled
            if show_node_ids:
                # Choose text color for contrast
                label_color = "#e8e8e8" if theme == "dark" else "#333333"
                # Scale font size with node size
                font_size = max(4, min(8, layer_node_size * 10))
                ax.text(
                    x, y, str(nid),
                    fontsize=font_size,
                    ha="center", va="center",
                    color=label_color,
                    zorder=4,
                    fontname=font_name if font_name else None,
                )

    ax.axis("off")
    ax.set_aspect("equal")
    # Apply computed bounds directly so the plot is vertically centered
    ax.set_xlim(x_min_bound, x_max_bound)
    ax.set_ylim(y_min_bound, y_max_bound)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    final_path = None
    if save_path:
        _root, ext = os.path.splitext(save_path)
        final_path = save_path if ext else f"{save_path}.{file_format}"
        os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
        plt.savefig(final_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color, edgecolor="none")
    if show:
        plt.show()
    else:
        plt.close()
    return final_path
