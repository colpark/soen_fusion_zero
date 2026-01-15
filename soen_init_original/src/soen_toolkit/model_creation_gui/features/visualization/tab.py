# src/soen_toolkit/model_creation_gui/tabs/tab_visualisation.py
from __future__ import annotations

import contextlib
import os
from pathlib import Path

# Graphviz import moved to rendering methods for lazy loading
# from graphviz import Digraph
import tempfile
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QAction, QColor, QCursor, QFont, QFontMetrics, QIcon, QPixmap
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from soen_toolkit.model_creation_gui.components.gradient_colorbar import GradientColorBar
from soen_toolkit.model_creation_gui.components.network_renderer import NetworkRenderer
from soen_toolkit.model_creation_gui.components.svg_hit_overlay import SVGHitOverlay
from soen_toolkit.model_creation_gui.utils.paths import icon

from .dialogs import VisualisationSettingsDialog
from .gradient_flow import (
    get_connection_gradient,
    get_layer_activation_gradient,
    get_layer_pair_gradient,
    get_neuron_activation_gradient,
    gradient_to_color,
)
from .interaction import GraphicsViewInteractionHandler
from .settings import VisualizationSettings
from .svg_renderer import get_contrasting_edge_color

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.model_manager import ModelManager


def _get_contrast_text_color(bg_hex: str) -> str:
    """Return black or white text color based on background brightness."""
    bg_hex = bg_hex.lstrip("#")
    if len(bg_hex) == 3:
        bg_hex = "".join(c * 2 for c in bg_hex)
    try:
        r, g, b = int(bg_hex[0:2], 16), int(bg_hex[2:4], 16), int(bg_hex[4:6], 16)
        # Perceived brightness formula
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return "#000000" if brightness > 128 else "#ffffff"
    except (ValueError, IndexError):
        return "#000000"


class VisualisationTab(QWidget):
    def __init__(self, mgr: ModelManager) -> None:
        super().__init__()
        self._mgr = mgr
        self._use_native = False  # Use hybrid SVG+overlay by default
        self._svg_bytes = None
        self._orig_size = None
        self._showing_grid_of_grids = False  # Track if showing Grid-of-Grids view
        # Start at fit-to-view (no extra magnification)
        self._zoom = 1.0
        self._extra_svg_height_px = 0.0  # additional bottom padding needed for overlays
        self._hit_nodes = []  # list of {layer_id:int, polygon:[(x,y),...]}
        self._interaction_mode = "select"  # 'select' or 'pan'
        self._scroll = None  # set in _init_ui
        self._svg_viewbox = None  # (x0, y0, w, h)
        self._svg_render_size = None  # (w, h) in px from renderer
        self._svg_bytes_base = None  # last rendered SVG without debug overlays
        self._debug_clicks = False
        # Load settings from persistent storage (uses dataclass defaults for missing values)
        self._settings = VisualizationSettings.load()

        # Gradient flow config and computed gradients
        self._gradient_config = None
        self._gradient_data = None  # Dict of computed gradients
        self._panning = False
        self._pan_start = QPoint()
        self._last_press_global = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        tb = QToolBar("Visualisation")

        # Renderer toggle (optional, hidden for now)
        # act_native = QAction("Native Renderer", self)
        # act_native.setCheckable(True)
        # act_native.setChecked(False)
        # act_native.toggled.connect(self._toggle_renderer)
        # tb.addAction(act_native)
        # tb.addSeparator()

        act_render = QAction(QIcon(icon("plus")), "Render", self)
        act_render.triggered.connect(self._render)
        tb.addAction(act_render)
        act_zi = QAction(QIcon(icon("zoom-in.svg")), "Zoom In", self)
        act_zi.triggered.connect(lambda: self._zoom_by(1.25))
        tb.addAction(act_zi)
        act_zo = QAction(QIcon(icon("zoom-out.svg")), "Zoom Out", self)
        act_zo.triggered.connect(lambda: self._zoom_by(0.8))
        tb.addAction(act_zo)

        tb.addSeparator()
        # Interaction mode toggle
        try:
            from PyQt6.QtGui import QActionGroup

            act_group = QActionGroup(self)
            act_group.setExclusive(True)
            act_select = QAction(QIcon(icon("cursor.svg")), "Select", self)
            act_select.setCheckable(True)
            act_select.setChecked(True)
            act_pan = QAction(QIcon(icon("hand.svg")), "Pan", self)
            act_pan.setCheckable(True)
            act_group.addAction(act_select)
            act_group.addAction(act_pan)

            def _set_select() -> None:
                if act_select.isChecked():
                    self._interaction_mode = "select"
                    self._apply_interaction_cursor()

            def _set_pan() -> None:
                if act_pan.isChecked():
                    self._interaction_mode = "pan"
                    self._apply_interaction_cursor()

            act_select.toggled.connect(lambda _: _set_select())
            act_pan.toggled.connect(lambda _: _set_pan())
            tb.addAction(act_select)
            tb.addAction(act_pan)
        except Exception:
            pass
        tb.addSeparator()
        # Debug toggle
        act_debug = QAction("Debug Clicks", self)
        act_debug.setCheckable(True)
        act_debug.setChecked(False)

        def _toggle_debug() -> None:
            self._debug_clicks = act_debug.isChecked()

        act_debug.toggled.connect(lambda _: _toggle_debug())
        tb.addAction(act_debug)
        tb.addSeparator()
        act_settings = QAction(QIcon(icon("settings.svg")), "Settings", self)
        act_settings.triggered.connect(self._open_settings)
        tb.addAction(act_settings)
        tb.addSeparator()
        act_save = QAction(QIcon(icon("save.svg")), "Save SVG…", self)
        act_save.triggered.connect(self._save_svg)
        tb.addAction(act_save)
        tb.addSeparator()

        # Gradient Flow toggle
        self._act_gradients = QAction("Gradients", self)
        self._act_gradients.setCheckable(True)
        self._act_gradients.setChecked(False)
        self._act_gradients.setToolTip("Color edges by gradient magnitude (configure in Settings)")
        self._act_gradients.toggled.connect(self._toggle_gradient_mode)
        tb.addAction(self._act_gradients)
        layout.addWidget(tb)

        # Colorbar for gradient visualization
        self._colorbar_widget = GradientColorBar()
        self._colorbar_widget.setVisible(False)
        layout.addWidget(self._colorbar_widget)

        # Create hybrid SVG view with overlay
        self._create_hybrid_view()

        # Keep native view for future use
        self._create_native_view()

        # Start with hybrid view
        self._current_view = self._hybrid_view

        # Details sidebar (initially collapsed)
        self._details_panel = QWidget()
        details_layout = QVBoxLayout(self._details_panel)
        details_layout.addWidget(QLabel("<b>Node Details</b>"))
        self._details_view = QTextBrowser()
        self._details_view.setOpenExternalLinks(True)
        details_layout.addWidget(self._details_view, 1)
        with contextlib.suppress(Exception):
            self._details_panel.setMinimumWidth(260)

        # Splitter to host view and details
        self._splitter = QSplitter()
        self._splitter.addWidget(self._hybrid_view)
        self._splitter.addWidget(self._details_panel)
        # Start collapsed (hide details)
        self._splitter.setSizes([1, 0])
        layout.addWidget(self._splitter, 1)

        # Initial render handled by _on_model_changed

    def _show_svg_preview(self, svg_path: str, *, title: str = "Preview (SVG)", suggest_name: str = "image.svg") -> None:
        """Show an SVG in a scrollable dialog with Save As."""
        try:
            import os

            if not os.path.exists(svg_path):
                QMessageBox.warning(self, "Not found", f"SVG not found:\n{svg_path}")
                return
            dlg = QDialog(self)
            dlg.setWindowTitle(title)
            v = QVBoxLayout(dlg)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            svg_widget = QSvgWidget(svg_path)
            scroll.setWidget(svg_widget)
            v.addWidget(scroll, 1)

            btns = QDialogButtonBox()
            btn_save = btns.addButton("Save As…", QDialogButtonBox.ButtonRole.ActionRole)
            btn_close = btns.addButton(QDialogButtonBox.StandardButton.Close)

            def _save_as() -> None:
                from PyQt6.QtWidgets import QFileDialog

                start_dir = str(Path.home())
                fname, _ = QFileDialog.getSaveFileName(self, "Save SVG As", os.path.join(start_dir, suggest_name), "SVG (*.svg);;All Files (*)")
                if fname:
                    try:
                        from shutil import copyfile

                        copyfile(svg_path, fname)
                        QMessageBox.information(self, "Saved", f"Saved to:\n{fname}")
                    except Exception as e:
                        QMessageBox.critical(self, "Save failed", str(e))

            btn_save.clicked.connect(_save_as)
            btn_close.clicked.connect(dlg.close)
            v.addWidget(btns)
            dlg.resize(900, 700)
            dlg.exec()
        except Exception:
            pass

    def _show_image_preview(self, image_path: str, *, title: str = "Preview", suggest_name: str = "image.png") -> None:
        """Show a scrollable preview of an image with a Save As option."""
        try:
            import os
            from pathlib import Path

            if not os.path.exists(image_path):
                QMessageBox.warning(self, "Not found", f"Image not found:\n{image_path}")
                return
            dlg = QDialog(self)
            dlg.setWindowTitle(title)
            v = QVBoxLayout(dlg)

            # --- Zoom controls row ---
            controls_row = QHBoxLayout()
            btn_zoom_in = QPushButton("+")
            btn_zoom_in.setToolTip("Zoom In")
            btn_zoom_in.setFixedWidth(30)
            btn_zoom_out = QPushButton("-")
            btn_zoom_out.setToolTip("Zoom Out")
            btn_zoom_out.setFixedWidth(30)
            btn_zoom_reset = QPushButton("Reset Size")
            btn_zoom_reset.setToolTip("Reset zoom to 100%")
            controls_row.addWidget(btn_zoom_in)
            controls_row.addWidget(btn_zoom_out)
            controls_row.addWidget(btn_zoom_reset)
            controls_row.addStretch(1)
            v.addLayout(controls_row)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = QPixmap(image_path)
            if pix.isNull():
                QMessageBox.warning(self, "Invalid image", "Could not load generated image.")
                return
            # Keep original pixmap for high-quality rescaling
            _orig_pix = QPixmap(pix)
            _scale = 1.0

            def _apply_scale() -> None:
                nonlocal _scale
                w = max(1, int(_orig_pix.width() * _scale))
                h = max(1, int(_orig_pix.height() * _scale))
                label.setPixmap(_orig_pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

            def _zoom(factor: float) -> None:
                nonlocal _scale
                _scale = max(0.1, min(10.0, _scale * factor))
                _apply_scale()

            def _reset_zoom() -> None:
                nonlocal _scale
                _scale = 1.0
                _apply_scale()

            btn_zoom_in.clicked.connect(lambda: _zoom(1.25))
            btn_zoom_out.clicked.connect(lambda: _zoom(0.8))
            btn_zoom_reset.clicked.connect(_reset_zoom)
            _apply_scale()
            scroll.setWidget(label)
            v.addWidget(scroll, 1)

            btns = QDialogButtonBox()
            btn_save = btns.addButton("Save As…", QDialogButtonBox.ButtonRole.ActionRole)
            btn_close = btns.addButton(QDialogButtonBox.StandardButton.Close)

            def _save_as() -> None:
                from PyQt6.QtWidgets import QFileDialog

                start_dir = str(Path.home())
                fname, _ = QFileDialog.getSaveFileName(self, "Save Image As", os.path.join(start_dir, suggest_name), "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)")
                if fname:
                    try:
                        from shutil import copyfile

                        copyfile(image_path, fname)
                        QMessageBox.information(self, "Saved", f"Saved to:\n{fname}")
                    except Exception as e:
                        QMessageBox.critical(self, "Save failed", str(e))

            btn_save.clicked.connect(_save_as)
            btn_close.clicked.connect(dlg.close)
            v.addWidget(btns)
            dlg.resize(900, 700)
            dlg.exec()
        except Exception:
            pass

    def _create_hybrid_view(self) -> None:
        """Create hybrid view: SVG rendering with overlay for hit detection."""
        # Use a graphics view that can display both SVG and overlay items
        self._overlay_scene = QGraphicsScene()
        self._hybrid_view = QGraphicsView(self._overlay_scene)
        self._hybrid_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._hybrid_view.viewport().installEventFilter(self)

        # We'll add the SVG as a pixmap item to the scene
        self._svg_widget = QSvgWidget()  # Keep for rendering
        self._scroll = self._hybrid_view  # Compatibility

        # Create overlay handler
        self._hit_overlay = SVGHitOverlay()
        self._hit_overlay.node_clicked.connect(self._on_overlay_click)

        # Create interaction handler
        self._hybrid_interaction = GraphicsViewInteractionHandler(
            self._hybrid_view,
            lambda pt: self._hit_overlay.handle_click(pt, debug=self._debug_clicks),
            lambda: self._interaction_mode,
        )

    def _create_native_view(self) -> None:
        """Create the native QGraphicsScene-based view."""
        self._scene = QGraphicsScene()
        self._native_view = QGraphicsView(self._scene)
        from PyQt6.QtGui import QPainter

        self._native_view.setRenderHints(
            self._native_view.renderHints() | QPainter.RenderHint.Antialiasing,
        )
        self._native_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._native_view.viewport().installEventFilter(self)

        # Create renderer
        self._renderer = NetworkRenderer(self._scene)
        self._renderer.node_clicked.connect(self._on_node_clicked)

        # Create interaction handler
        self._native_interaction = GraphicsViewInteractionHandler(
            self._native_view,
            lambda pt: self._renderer.handle_click(pt),
            lambda: self._interaction_mode,
        )

    def _create_svg_view(self) -> None:
        """Create the SVG-based view (legacy)."""
        self._svg_widget = QSvgWidget()
        self._svg_widget.setStyleSheet("background-color: white;")
        self._svg_widget.installEventFilter(self)
        self._svg_scroll = QScrollArea()
        self._svg_scroll.setWidgetResizable(True)
        self._svg_scroll.setWidget(self._svg_widget)
        self._svg_scroll.viewport().installEventFilter(self)
        self._scroll = self._svg_scroll  # Compatibility

    def _toggle_renderer(self, use_native) -> None:
        """Switch between native and SVG renderers."""
        self._use_native = use_native

        # Swap views in splitter
        if use_native:
            self._splitter.replaceWidget(0, self._native_view)
            self._current_view = self._native_view
        else:
            self._splitter.replaceWidget(0, self._hybrid_view)
            self._current_view = self._hybrid_view

        # Re-render with new renderer
        self._render()

    def _on_overlay_click(self, layer_id: int, neuron_idx: int) -> None:
        """Handle click from overlay on SVG."""
        self._show_layer_details(layer_id, neuron_idx if neuron_idx >= 0 else None)

    def _on_node_clicked(self, layer_id: int, neuron_idx: int) -> None:
        """Handle click from native renderer."""
        self._show_layer_details(layer_id, neuron_idx if neuron_idx >= 0 else None)

    def _on_model_changed(self) -> None:
        """Slot called when the model manager signals a model change."""
        # Do not auto-render on build/load. Always clear the view so nothing
        # is shown until the user clicks "Render".
        try:
            if hasattr(self, "_overlay_scene") and self._overlay_scene is not None:
                self._overlay_scene.clear()
            if hasattr(self, "_hybrid_view") and self._hybrid_view is not None:
                self._hybrid_view.resetTransform()
        except Exception:
            pass
        try:
            if hasattr(self, "_scene") and self._scene is not None:
                self._scene.clear()
            if hasattr(self, "_native_view") and self._native_view is not None:
                self._native_view.resetTransform()
        except Exception:
            pass
        # Clear legacy SVG widget and internal caches
        with contextlib.suppress(Exception):
            self._svg_widget.load(b"")
        self._svg_bytes = None
        self._svg_bytes_base = None
        self._orig_size = None
        self._hit_nodes = []
        self._svg_viewbox = None
        self._svg_render_size = None

    def _open_settings(self) -> None:
        dlg = VisualisationSettingsDialog(self, self._settings.copy())
        if dlg.exec():
            # Update settings from dialog (returns dict, we update dataclass)
            self._settings.update_from_dict(dlg.get_settings())
            # Refresh gradients if they're enabled (always recompute on settings change)
            if self._settings.get("show_gradients", False):
                self._refresh_gradients()
            # Persist settings
            self._settings.save()
            self._render()

    def _refresh_gradients(self) -> bool:
        """Recompute gradients and update visualization. Returns True on success."""
        self._gradient_data = None
        self._gradient_config = None

        success = self._compute_gradients()
        if success:
            self._update_colorbar()
            self._colorbar_widget.setVisible(True)
        else:
            # Disable gradient mode if computation failed
            self._act_gradients.blockSignals(True)
            self._act_gradients.setChecked(False)
            self._act_gradients.blockSignals(False)
            self._settings["show_gradients"] = False
            self._colorbar_widget.setVisible(False)

        return success

    def _toggle_gradient_mode(self, enabled: bool) -> None:
        """Toggle gradient coloring mode on/off."""
        self._settings["show_gradients"] = enabled

        if enabled:
            self._refresh_gradients()
        else:
            self._colorbar_widget.setVisible(False)

        self._render()

    def _update_colorbar(self) -> None:
        """Update the colorbar with current gradient range."""
        if self._gradient_config is None:
            return

        # Use connection gradient range for edges, activation for layers
        # Show both ranges in the label
        conn_min = self._gradient_config.grad_min
        conn_max = self._gradient_config.grad_max
        act_min = self._gradient_config.activation_grad_min
        act_max = self._gradient_config.activation_grad_max

        # Use the wider range for the colorbar
        vmin = min(conn_min, act_min)
        vmax = max(conn_max, act_max)

        self._colorbar_widget.update_range(
            vmin=vmin,
            vmax=vmax,
            colormap=self._settings.get("grad_colormap", "RdBu_r"),
            log_scale=self._settings.get("grad_log_scale", False),
            label="dL/dJ (edges), dL/ds (nodes)",
        )

    def _compute_gradients(self) -> bool:
        """Compute gradients from a single sample. Returns True on success."""
        from .gradient_flow import GradientFlowConfig, GradientFlowError, compute_connection_gradients

        # Check if model exists
        if self._mgr.model is None:
            QMessageBox.warning(self, "No Model", "Build the model first before computing gradients.")
            return False

        # Check if dataset is configured
        hdf5_path = self._settings.get("grad_hdf5_path", "")
        if not hdf5_path or not Path(hdf5_path).exists():
            QMessageBox.warning(
                self,
                "No Dataset",
                "Configure a dataset in Settings > Gradient Flow tab first.",
            )
            return False

        # Determine task type
        task_type = self._settings.get("grad_task_type", "classification")
        is_classification = task_type == "classification"

        # Build config from settings - single sample mode
        config = GradientFlowConfig(
            hdf5_path=hdf5_path,
            split=self._settings.get("grad_split", "train"),
            seq_len=self._settings.get("grad_seq_len", 100),
            dt=self._settings.get("grad_dt", 37.0),
            total_time_ns=self._settings.get("grad_total_time_ns"),
            batch_size=1,  # Single sample
            num_batches=1,  # Single batch
            feature_min=self._settings.get("grad_feature_min"),
            feature_max=self._settings.get("grad_feature_max"),
            # Sample selection (always filter by class for classification)
            use_class_filter=is_classification,
            class_id=self._settings.get("grad_class_id", 0) if is_classification else None,
            sample_index=self._settings.get("grad_sample_index", 0),
            # Gradient settings
            loss_fn=self._settings.get("grad_loss_fn", "mse"),
            batch_agg="mean",  # Not used for single sample but keep for backend
            log_scale=self._settings.get("grad_log_scale", False),
            colormap=self._settings.get("grad_colormap", "RdBu_r"),
        )

        # Create progress dialog
        progress = QProgressDialog("Computing gradients...", "Cancel", 0, 1, self)
        progress.setWindowTitle("Gradient Computation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Progress callback
        cancelled = [False]

        def progress_callback(current: int, total: int, message: str) -> None:
            if progress.wasCanceled():
                cancelled[0] = True
                return
            progress.setValue(current)
            progress.setLabelText(message)
            # Process events to keep UI responsive
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()

        # Compute gradients
        try:
            self._gradient_config = config
            self._gradient_data = compute_connection_gradients(
                self._mgr.model,
                config,
                progress_callback=progress_callback,
            )

            progress.close()

            if cancelled[0]:
                self._gradient_data = None
                return False

            if not self._gradient_data:
                QMessageBox.warning(
                    self,
                    "No Gradients",
                    "No gradients were computed. Check that the model has learnable connections.",
                )
                return False

            return True

        except GradientFlowError as e:
            progress.close()
            QMessageBox.critical(self, "Gradient Computation Failed", str(e))
            return False
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Unexpected error computing gradients:\n{e}")
            return False

    def _render(self) -> None:
        detailed_layout = self._settings.get("detailed_layout", "linear")
        is_simple_view = self._settings.get("simple_view", True)

        # Grid layout uses Matplotlib (displayed in hybrid view as PNG)
        if detailed_layout == "grid" and not is_simple_view:
            # Ensure we're in hybrid view mode for displaying the PNG
            if self._use_native:
                self._use_native = False
                self._splitter.replaceWidget(0, self._hybrid_view)
                self._current_view = self._hybrid_view
            self._render_grid()
            return

        # Circular layout uses native QGraphicsScene renderer
        needs_native = detailed_layout == "circular" and not is_simple_view

        if needs_native and not self._use_native:
            # Switch to native view for circular
            self._toggle_renderer(True)
            return  # _toggle_renderer already triggered a render

        if not needs_native and self._use_native:
            # Switch back to SVG view for linear (restore original behavior)
            self._toggle_renderer(False)
            return  # _toggle_renderer already triggered a render

        if self._use_native:
            self._render_native()
        else:
            self._render_svg()

    def _render_grid(self) -> None:
        """Render using Matplotlib Grid-of-Grids visualization.

        Uses high-DPI PNG for fast zoom/pan (SVG would be too slow with thousands of elements).
        Maps shared settings to Grid-of-Grids parameters.
        """
        import os
        import tempfile

        self._showing_grid_of_grids = True

        model = getattr(self._mgr, "model", None)
        if model is None:
            self._overlay_scene.clear()
            return

        # Map shared settings to Grid-of-Grids parameters
        theme = self._settings.get("theme", "default")

        # Generate high-DPI PNG for fast rendering
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            out_path = model.visualize_grid_of_grids(
                save_path=tmp_path,
                file_format="png",
                dpi=300,  # High DPI for crisp zoom
                show=False,
                # Map shared settings
                theme=theme,
                bg_color=self._settings.get("bg_color", "#ffffff"),
                module_bg_color=self._settings.get("layer_color", "#eae5be"),
                inter_edge_color=self._settings.get("inter_color", "#555555"),
                intra_edge_color=self._settings.get("intra_color", "#888888"),
                # Label options
                show_layer_ids=self._settings.get("show_layer_ids", True),
                show_descriptions=self._settings.get("show_desc", False),
                show_node_ids=self._settings.get("show_node_ids", False),
            )

            # Load PNG into QGraphicsPixmapItem for fast zoom/pan
            from PyQt6.QtGui import QPixmap
            from PyQt6.QtWidgets import QGraphicsPixmapItem

            # Clear scene
            self._overlay_scene.clear()

            # Load pixmap from file
            pixmap = QPixmap(out_path)
            if pixmap.isNull():
                raise ValueError(f"Failed to load image: {out_path}")

            pixmap_item = QGraphicsPixmapItem(pixmap)
            pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self._overlay_scene.addItem(pixmap_item)

            # Store size for zoom
            from PyQt6.QtCore import QSize

            self._orig_size = QSize(pixmap.width(), pixmap.height())

            # Fit in view
            self._hybrid_view.fitInView(
                self._overlay_scene.itemsBoundingRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

        except Exception as e:
            QMessageBox.critical(self, "Render failed", f"Error rendering Grid view:\n{e}")
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                with contextlib.suppress(Exception):
                    os.remove(tmp_path)

    def _render_native(self) -> None:
        """Render using native QGraphicsScene."""
        # Clear grid-of-grids mode when rendering network
        self._showing_grid_of_grids = False

        if self._mgr.model is None:
            self._scene.clear()
            return

        # Apply settings to renderer
        self._renderer.orientation = self._settings["orientation"]
        self._renderer.simple_view = self._settings.get("simple_view", True)
        self._renderer.detailed_layout = self._settings.get("detailed_layout", "linear")
        self._renderer.edge_routing = self._settings.get("edge_routing", "true")
        self._renderer.show_intra = self._settings["show_intra"]
        self._renderer.show_descriptions = self._settings["show_desc"]
        self._renderer.show_conn_types = self._settings.get("show_conn_type", False)
        self._renderer.show_layer_ids = self._settings.get("show_layer_ids", True)
        self._renderer.show_node_ids = self._settings.get("show_node_ids", False)
        self._renderer.show_model_ids = self._settings.get("show_model_ids", True)

        # Colors
        self._renderer.layer_color = QColor(self._settings.get("layer_color", "#eae5be"))
        self._renderer.inter_color = QColor(self._settings["inter_color"])
        self._renderer.intra_color = QColor(self._settings["intra_color"])
        self._renderer.bg_color = QColor(self._settings.get("bg_color", "#ffffff"))
        self._renderer.text_color = QColor(self._settings.get("text_color", "#000000"))

        # Gradient flow settings
        self._renderer.show_gradients = self._settings.get("show_gradients", False)
        self._renderer.gradient_data = self._gradient_data
        self._renderer.gradient_config = self._gradient_config

        # Polarity settings
        self._renderer.show_neuron_polarity = self._settings.get("show_neuron_polarity", False)
        self._renderer.show_connection_polarity = self._settings.get("show_connection_polarity", False)

        # Load polarity data if neuron polarity is enabled
        if self._renderer.show_neuron_polarity:
            self._renderer.layer_polarity = self._load_polarity_data()
        else:
            self._renderer.layer_polarity = {}

        # Spacing
        self._renderer.layer_spacing = int(self._settings.get("layer_spacing", 1.0) * 100)

        # Render
        self._renderer.render_network(
            self._mgr.layers,
            self._mgr.connections,
            self._mgr.model,
        )

        # Set scene background
        self._scene.setBackgroundBrush(self._renderer.bg_color)

        # Fit in view
        self._native_view.fitInView(self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _render_svg(self) -> None:
        """Render using Graphviz/SVG (legacy)."""
        # Clear grid-of-grids mode when rendering network
        self._showing_grid_of_grids = False
        # Reset any extra padding from previous renders
        self._extra_svg_height_px = 0.0
        self._hit_nodes = []
        if self._mgr.model is None:
            # Clear all visual backends if no model is present
            try:
                if hasattr(self, "_overlay_scene") and self._overlay_scene is not None:
                    self._overlay_scene.clear()
                if hasattr(self, "_hybrid_view") and self._hybrid_view is not None:
                    self._hybrid_view.resetTransform()
            except Exception:
                pass
            try:
                if hasattr(self, "_scene") and self._scene is not None:
                    self._scene.clear()
                if hasattr(self, "_native_view") and self._native_view is not None:
                    self._native_view.resetTransform()
            except Exception:
                pass
            # Clear legacy SVG widget and internal caches
            with contextlib.suppress(Exception):
                self._svg_widget.load(b"")
            self._svg_bytes = None
            self._svg_bytes_base = None
            self._orig_size = None
            self._hit_nodes = []
            self._svg_viewbox = None
            self._svg_render_size = None
            return
        try:
            # Lazy import graphviz only when rendering SVG
            from graphviz import Digraph

            # Initialize Graphviz Digraph with hierarchical layout
            dot = Digraph(format="svg", engine="dot")
            dot.attr(
                rankdir=self._settings["orientation"],
                splines=self._settings["edge_routing"],
                ranksep=str(self._settings["layer_spacing"]),
                nodesep="0.02",  # Tight vertical spacing
                newrank="true",  # Better ranking algorithm
                compound="true", # Allow cluster edges
                fontname="Arial",
            )
            dot.node_attr.update(fontname="Arial")
            dot.edge_attr.update(fontname="Arial")

            # Calculate layer ranks (depths) for column assignment
            # BFS from input layers, group by model_id, outputs get final column
            layer_ids = {cfg.layer_id for cfg in self._mgr.layers}
            layer_model_ids = {cfg.layer_id: getattr(cfg, 'model_id', 0) for cfg in self._mgr.layers}

            # Build adjacency (excluding self-connections)
            forward_incoming = {lid: set() for lid in layer_ids}
            forward_outgoing = {lid: set() for lid in layer_ids}
            for conn in self._mgr.connections:
                src, dst = conn.from_layer, conn.to_layer
                if src in layer_ids and dst in layer_ids and src != dst:
                    forward_incoming[dst].add(src)
                    forward_outgoing[src].add(dst)

            # Find input layers (no incoming non-self connections)
            input_layers = {lid for lid in layer_ids if not forward_incoming[lid]}
            if not input_layers:
                input_layers = {0} if 0 in layer_ids else {min(layer_ids)}

            # BFS to compute minimum depth from inputs
            from collections import deque
            layer_ranks = {}
            queue = deque()
            for lid in input_layers:
                layer_ranks[lid] = 0
                queue.append(lid)

            while queue:
                current = queue.popleft()
                current_depth = layer_ranks[current]
                for neighbor in forward_outgoing[current]:
                    if neighbor not in layer_ranks:
                        layer_ranks[neighbor] = current_depth + 1
                        queue.append(neighbor)

            # Handle disconnected nodes
            max_rank = max(layer_ranks.values()) if layer_ranks else 0
            for lid in layer_ids:
                if lid not in layer_ranks:
                    layer_ranks[lid] = max_rank

            # Find output layers (no outgoing non-self connections)
            output_layers = {lid for lid in layer_ids if not forward_outgoing[lid]}

            # Push output layers to their own final column
            if output_layers:
                current_max = max(layer_ranks.values())
                non_outputs_at_max = [
                    lid for lid in layer_ids
                    if layer_ranks[lid] == current_max and lid not in output_layers
                ]
                if non_outputs_at_max:
                    for lid in output_layers:
                        layer_ranks[lid] = current_max + 1

            # Group layers by rank, sorted by (model_id, layer_id) for vertical grouping
            ranks = {}
            for lid, r in layer_ranks.items():
                ranks.setdefault(r, []).append(lid)
            for r, lids in ranks.items():
                lids.sort(key=lambda lid: (layer_model_ids.get(lid, 0), lid))

            # Apply canvas background
            dot.graph_attr.update(bgcolor=self._settings.get("bg_color", "white"))
            # Match widget and viewport background to canvas background to avoid white borders
            try:
                bg_css = f"background-color: {self._settings.get('bg_color', 'white')};"
                self._svg_widget.setStyleSheet(bg_css)
                self._scroll.viewport().setStyleSheet(bg_css)
            except Exception:
                pass
            # Add extra vertical margin when descriptions are shown in simple view
            # so overlaid text isn't clipped by the canvas.
            if self._settings.get("show_desc", False) and self._settings.get("simple_view", True):
                # margin values are in inches; set asymmetric via x,y string.
                # Slight vertical padding so description overlay has room but not too much.
                dot.graph_attr.update(margin="0.25,0.40")  # x=0.25in, y=0.40in
            else:
                dot.graph_attr.update(margin="0.25")
            # Apply edge style globally. Using edge_attr avoids Graphviz ignoring
            # penwidth when node-specific attrs override defaults.
            dot.edge_attr.update(
                penwidth=str(self._settings.get("edge_thickness", 1.0)),
                arrowsize=str(self._settings.get("arrow_size", 1.0)),
            )

            if self._settings.get("simple_view", True):
                # Simple view: render compact nodes
                # Choose inter edge color with contrast for dark backgrounds
                inter_color = get_contrasting_edge_color(
                    self._settings["inter_color"],
                    self._settings.get("bg_color", "white"),
                )

                # Group layers by model_id if enabled
                group_by_model = self._settings.get("group_by_model", False)
                if group_by_model:
                    from collections import defaultdict
                    model_groups = defaultdict(list)
                    for cfg in self._mgr.layers:
                        mid = getattr(cfg, "model_id", 0)
                        model_groups[mid].append(cfg)

                    # Create cluster subgraphs for each model
                    model_subgraphs = {}
                    model_subgraph_cms = {}  # Store context managers for proper cleanup
                    for mid in sorted(model_groups.keys()):
                        cluster_name = f"cluster_model_{mid}"
                        cm = dot.subgraph(name=cluster_name)
                        subgraph = cm.__enter__()  # Capture the actual subgraph from __enter__()
                        subgraph.attr(
                            label=f"Model {mid}",
                            style="rounded,dashed",
                            color="#666666",
                            fontsize="12",
                            labeljust="l",
                        )
                        model_subgraphs[mid] = subgraph
                        model_subgraph_cms[mid] = cm  # Store context manager for later __exit__()

                for cfg in self._mgr.layers:
                    mid = getattr(cfg, "model_id", 0)
                    title_parts = []
                    if self._settings.get("show_layer_ids", True):
                        if self._settings.get("show_model_ids", True):
                            title_parts.append(f"Layer: {cfg.layer_id} (M{mid})")
                        else:
                            title_parts.append(f"Layer: {cfg.layer_id}")
                    # When descriptions are enabled in simple view, also show the layer type (no prefix)
                    if self._settings.get("show_desc", False):
                        try:
                            if getattr(cfg, "layer_type", None):
                                title_parts.append(str(cfg.layer_type))
                        except Exception:
                            pass
                    title_parts.append(f"Dim: {cfg.params.get('dim')}")

                    # Ensure the cell is wide enough for the longest line to avoid spillover due to font metric mismatches
                    try:
                        title_font = QFont("Arial", 10)
                        title_font.setWeight(QFont.Weight.Bold)
                        line_font = QFont("Arial", 9)
                        fm_title = QFontMetrics(title_font)
                        fm_line = QFontMetrics(line_font)
                        max_px = 0
                        for i, line in enumerate(title_parts):
                            if i == 0 and self._settings.get("show_layer_ids", True):
                                max_px = max(max_px, fm_title.horizontalAdvance(line))
                            else:
                                max_px = max(max_px, fm_line.horizontalAdvance(line))
                        # Convert px to points for Graphviz WIDTH (~72 pt/in, ~96 px/in)
                        # Account for CELLPADDING=8 (16px total horizontal) plus margin
                        width_pt = int(max_px * 0.75) + 20
                    except Exception:
                        width_pt = None

                    # Single table: main tinted cell contains title+dim.
                    # Do NOT embed description here; it will be overlaid separately
                    # in simple view to avoid layout shifts and duplication.
                    # Determine layer background color - use activation gradient if enabled
                    if self._settings.get("show_gradients", False) and self._gradient_config and self._gradient_config.activation_gradients:
                        layer_grad = get_layer_activation_gradient(
                            self._gradient_config.activation_gradients,
                            cfg.layer_id,
                            aggregation="abs_mean",
                        )
                        layer_bg = gradient_to_color(
                            layer_grad,
                            self._gradient_config.activation_grad_min,
                            self._gradient_config.activation_grad_max,
                            colormap=self._settings.get("grad_colormap", "RdBu_r"),
                            log_scale=self._settings.get("grad_log_scale", False),
                        )
                    else:
                        layer_bg = self._settings["layer_color"]
                    # Compute contrasting text color for layer background
                    text_color = _get_contrast_text_color(layer_bg)
                    title_html = f'<FONT FACE="Arial" COLOR="{text_color}">' + "<BR/>".join(title_parts) + "</FONT>"
                    rows = []
                    # Tinted main cell with the text inside
                    if width_pt is not None:
                        rows.append(f'<TR><TD BGCOLOR="{layer_bg}" ALIGN="CENTER" WIDTH="{width_pt}">{title_html}</TD></TR>')
                    else:
                        rows.append(f'<TR><TD BGCOLOR="{layer_bg}" ALIGN="CENTER">{title_html}</TD></TR>')
                    html_label = '<<TABLE BORDER="1" CELLBORDER="1" CELLPADDING="8" CELLSPACING="0">' + "".join(rows) + "</TABLE>>"

                    outline_color = "black" if self._settings.get("show_layer_outline", True) else "transparent"

                    node_kwargs = {"shape": "plaintext", "color": outline_color, "penwidth": "1", "id": f"node_{cfg.layer_id}", "margin": "0"}

                    # Add node to model subgraph if grouping enabled, otherwise to main graph
                    if group_by_model:
                        model_subgraphs[mid].node(str(cfg.layer_id), label=html_label, **node_kwargs)
                    else:
                        dot.node(str(cfg.layer_id), label=html_label, **node_kwargs)

                # Close model subgraphs if grouping was enabled
                if group_by_model:
                    for cm in model_subgraph_cms.values():
                        cm.__exit__(None, None, None)

                # Enforce column ranks (Simple View)
                for r, lids in ranks.items():
                    # Skip if rank has only one node (no constraints needed)
                    if len(lids) < 2 and r != 0:
                        continue
                    with dot.subgraph(name=f"rank_{r}") as s:
                        if r == 0:
                            s.attr(rank="source")
                        else:
                            s.attr(rank="same")
                        for lid in lids:
                            s.node(str(lid))

                # Add invisible backbone to enforce rank ordering (left-to-right flow)
                import itertools
                sorted_ranks = sorted(ranks.keys())
                for r1, r2 in itertools.pairwise(sorted_ranks):
                    if ranks[r1] and ranks[r2]:
                        u = str(ranks[r1][0])
                        v = str(ranks[r2][0])
                        dot.edge(u, v, style="invis", weight="1000")

                # Draw connections between layers
                for conn in self._mgr.connections:
                    # Determine edge color
                    if self._settings.get("show_gradients", False) and self._gradient_data:
                        # Use gradient coloring
                        grad_val = get_layer_pair_gradient(
                            self._gradient_data,
                            conn.from_layer,
                            conn.to_layer,
                            aggregation="abs_mean",
                        )
                        color = gradient_to_color(
                            grad_val,
                            self._gradient_config.grad_min if self._gradient_config else 0,
                            self._gradient_config.grad_max if self._gradient_config else 1,
                            colormap=self._settings.get("grad_colormap", "RdBu_r"),
                            log_scale=self._settings.get("grad_log_scale", False),
                        )
                    else:
                        # Default: intra vs inter-layer color
                        color = self._settings["intra_color"] if conn.from_layer == conn.to_layer else inter_color
                    attrs = {"color": color}
                    # Handle intra-layer visibility and styling
                    if conn.from_layer == conn.to_layer:
                        if not self._settings["show_intra"]:
                            continue
                        attrs.update({"constraint": "false", "weight": "0"})
                    # For backward edges (right to left), relax ranking so the
                    # curve stays tight between the nodes instead of looping out.
                    elif conn.from_layer > conn.to_layer:
                        attrs.update({"constraint": "false", "weight": "0", "minlen": "0"})
                    # Only label inter-layer edges; skip labels on intra-layer (recurrent) loops
                    if self._settings.get("show_conn_type", False) and (conn.from_layer != conn.to_layer):
                        attrs["label"] = conn.connection_type
                        attrs["fontsize"] = "12"
                    # For self-connections, anchor both ends on the east side so the
                    # loop hugs the right edge with a small radius. For other edges,
                    # keep east-to-west anchors for straight routing across columns.
                    # We are using hierarchical layout, so ports work well
                    use_ports = True
                    if conn.from_layer == conn.to_layer:
                        attrs.update({"constraint": "false", "weight": "0", "minlen": "0"})
                        if use_ports:
                            dot.edge(f"{conn.from_layer}:e", f"{conn.to_layer}:e", **attrs)
                        else:
                            dot.edge(str(conn.from_layer), str(conn.to_layer), **attrs)
                    elif use_ports:
                        dot.edge(f"{conn.from_layer}:e", f"{conn.to_layer}:w", **attrs)
                    else:
                        dot.edge(str(conn.from_layer), str(conn.to_layer), **attrs)

                # Add intra-layer loops from masks if not already present via explicit connection
                if self._settings.get("show_intra", True) and getattr(self._mgr, "model", None) is not None:
                    try:
                        masks = getattr(self._mgr.model, "connection_masks", {})
                        explicit_self_edges = {(c.from_layer, c.to_layer) for c in self._mgr.connections if c.from_layer == c.to_layer}
                        for cfg in self._mgr.layers:
                            lid = cfg.layer_id
                            if (lid, lid) in explicit_self_edges:
                                continue
                            m = masks.get(f"internal_{lid}")
                            has_internal = False
                            if m is not None:
                                try:
                                    has_internal = m.sum().item() > 0
                                except Exception:
                                    has_internal = False
                            if not has_internal:
                                continue
                            # Determine color for internal connection
                            if self._settings.get("show_gradients", False) and self._gradient_data:
                                grad_val = get_layer_pair_gradient(
                                    self._gradient_data, lid, lid, aggregation="abs_mean"
                                )
                                intra_color = gradient_to_color(
                                    grad_val,
                                    self._gradient_config.grad_min if self._gradient_config else 0,
                                    self._gradient_config.grad_max if self._gradient_config else 1,
                                    colormap=self._settings.get("grad_colormap", "RdBu_r"),
                                    log_scale=self._settings.get("grad_log_scale", False),
                                )
                            else:
                                intra_color = self._settings["intra_color"]
                            attrs = {"color": intra_color, "constraint": "false", "weight": "0", "minlen": "0"}
                            if use_ports:
                                dot.edge(f"{lid}:e", f"{lid}:e", **attrs)
                            else:
                                dot.edge(str(lid), str(lid), **attrs)
                    except Exception:
                        pass
            else:
                # Detailed node view: each layer rendered as a single node with
                # an HTML-like TABLE label exposing one port per neuron. This
                # guarantees a stable, aesthetically ordered vertical list.
                dims = {cfg.layer_id: cfg.params.get("dim", 0) for cfg in self._mgr.layers}

                # Load polarity data if neuron polarity is enabled
                layer_polarity = self._load_polarity_data() if self._settings.get("show_neuron_polarity", False) else {}

                sorted_layer_ids = sorted(dims)
                offsets = {}
                running_total = 0
                for lid in sorted_layer_ids:
                    offsets[lid] = running_total
                    running_total += dims[lid]

                layer_node_ids = []
                for lid in sorted_layer_ids:
                    label_rows = []
                    # Optional compact title row (layer ID only), white background and zero padding to avoid indent
                    if self._settings.get("show_layer_ids", True):
                        mid = getattr(next((c for c in self._mgr.layers if c.layer_id == lid), None), "model_id", 0)
                        if self._settings.get("show_model_ids", True):
                            title = f"Layer {lid} (M{mid})"
                        else:
                            title = f"Layer {lid}"
                        # In TB mode, title needs to span all neuron columns
                        colspan_attr = f' COLSPAN="{dims[lid]}"' if self._settings["orientation"] == "TB" else ""
                        bg_color = self._settings.get("bg_color", "white")
                        title_text_color = _get_contrast_text_color(bg_color)
                        label_rows.append(
                            f'<TR><TD{colspan_attr} BORDER="0" ALIGN="LEFT" BGCOLOR="{bg_color}">'
                            '<TABLE BORDER="0" CELLSPACING="0" CELLPADDING="0">'
                            f'<TR><TD ALIGN="LEFT"><FONT FACE="Arial" COLOR="{title_text_color}"><B>{title}</B></FONT></TD></TR>'
                            "</TABLE>"
                            "</TD></TR>",
                        )
                    # Arrange neurons based on orientation
                    # Get polarity for this layer if available
                    polarity = layer_polarity.get(lid) if self._settings.get("show_neuron_polarity", False) else None

                    if self._settings["orientation"] == "TB":
                        # Top-to-bottom: arrange neurons horizontally in a single row
                        neuron_cells = []
                        for i in range(dims[lid]):
                            port = f"p{i:04d}"
                            # Determine background color - gradient > polarity > default
                            if self._settings.get("show_gradients", False) and self._gradient_config and self._gradient_config.activation_gradients:
                                neuron_grad = get_neuron_activation_gradient(
                                    self._gradient_config.activation_gradients,
                                    lid,
                                    i,
                                )
                                bgcolor = gradient_to_color(
                                    neuron_grad,
                                    self._gradient_config.activation_grad_min,
                                    self._gradient_config.activation_grad_max,
                                    colormap=self._settings.get("grad_colormap", "RdBu_r"),
                                    log_scale=self._settings.get("grad_log_scale", False),
                                )
                            elif polarity is not None and i < len(polarity):
                                polarity_val = polarity[i]
                                if polarity_val == 1:  # Excitatory
                                    bgcolor = "#FF6B6B"  # Red
                                elif polarity_val == -1:  # Inhibitory
                                    bgcolor = "#5D9CEC"  # Blue
                                else:
                                    bgcolor = self._settings["layer_color"]
                            else:
                                bgcolor = self._settings["layer_color"]

                            if self._settings.get("show_node_ids", True):
                                neuron_text_color = _get_contrast_text_color(bgcolor)
                                neuron_cells.append(
                                    f'<TD PORT="{port}" WIDTH="18" HEIGHT="18" FIXEDSIZE="TRUE" '
                                    f'ALIGN="CENTER" BGCOLOR="{bgcolor}">'
                                    f'<FONT FACE="Arial" POINT-SIZE="8" COLOR="{neuron_text_color}">{i}</FONT></TD>'
                                )
                            else:
                                neuron_cells.append(
                                    f'<TD PORT="{port}" WIDTH="12" HEIGHT="12" FIXEDSIZE="TRUE" BGCOLOR="{bgcolor}"> </TD>'
                                )
                        # Add all neurons as a single row
                        label_rows.append(f'<TR>{"".join(neuron_cells)}</TR>')
                    else:
                        # Left-to-right: arrange neurons vertically (original behavior)
                        for i in range(dims[lid]):
                            port = f"p{i:04d}"
                            # Determine background color - gradient > polarity > default
                            if self._settings.get("show_gradients", False) and self._gradient_config and self._gradient_config.activation_gradients:
                                neuron_grad = get_neuron_activation_gradient(
                                    self._gradient_config.activation_gradients,
                                    lid,
                                    i,
                                )
                                bgcolor = gradient_to_color(
                                    neuron_grad,
                                    self._gradient_config.activation_grad_min,
                                    self._gradient_config.activation_grad_max,
                                    colormap=self._settings.get("grad_colormap", "RdBu_r"),
                                    log_scale=self._settings.get("grad_log_scale", False),
                                )
                            elif polarity is not None and i < len(polarity):
                                polarity_val = polarity[i]
                                if polarity_val == 1:  # Excitatory
                                    bgcolor = "#FF6B6B"  # Red
                                elif polarity_val == -1:  # Inhibitory
                                    bgcolor = "#5D9CEC"  # Blue
                                else:
                                    bgcolor = self._settings["layer_color"]
                            else:
                                bgcolor = self._settings["layer_color"]

                            if self._settings.get("show_node_ids", True):
                                neuron_text_color = _get_contrast_text_color(bgcolor)
                                label_rows.append(
                                    f'<TR><TD PORT="{port}" WIDTH="16" HEIGHT="10" ALIGN="RIGHT" BGCOLOR="{bgcolor}">'
                                    f'<FONT FACE="Arial" POINT-SIZE="8" COLOR="{neuron_text_color}">{i}</FONT></TD></TR>',
                                )
                            else:
                                # unlabeled compact cell
                                label_rows.append(
                                    f'<TR><TD PORT="{port}" WIDTH="12" HEIGHT="6" BGCOLOR="{bgcolor}"> </TD></TR>',
                                )
                    # Omit description in detailed label; overlay will add it below.

                    # Use CELLSPACING for TB orientation to ensure equal cell widths
                    if self._settings["orientation"] == "TB":
                        html = '<<TABLE BORDER="0" CELLBORDER="1" CELLPADDING="2" CELLSPACING="1">{}</TABLE>>'.format("".join(label_rows))
                    else:
                        # Wrapper table has no background; neuron rows carry the tint
                        html = '<<TABLE BORDER="0" CELLBORDER="1" CELLPADDING="2" CELLSPACING="0">{}</TABLE>>'.format("".join(label_rows))

                    outline_color = "black" if self._settings.get("show_layer_outline", True) else "transparent"
                    node_id = f"layer_{lid}"
                    # Node acts as the container
                    dot.node(node_id, label=html, shape="plaintext", color=outline_color, penwidth="1", id=f"node_{lid}", margin="0")
                    layer_node_ids.append(node_id)

                # Enforce column ranks (Detailed View)
                # This ensures layers are aligned in columns based on depth
                for r, lids in ranks.items():
                    # Skip if rank has only one node (no constraints needed)
                    if len(lids) < 2 and r != 0:
                        continue
                    with dot.subgraph(name=f"rank_{r}") as s:
                        if r == 0:
                            s.attr(rank="source")
                        else:
                            s.attr(rank="same")
                        for lid in lids:
                            s.node(f"layer_{lid}")

                # Add invisible backbone to enforce rank ordering (left-to-right flow)
                # This prevents ranks from collapsing onto each other when constraint=false
                import itertools
                sorted_ranks = sorted(ranks.keys())
                for r1, r2 in itertools.pairwise(sorted_ranks):
                    if ranks[r1] and ranks[r2]:
                        # Connect representative nodes to force order
                        u = f"layer_{ranks[r1][0]}"
                        v = f"layer_{ranks[r2][0]}"
                        # High weight to prioritize this structure
                        dot.edge(u, v, style="invis", weight="1000")

                # Draw edges using table ports
                mat = self._mgr.model.get_global_connection_mask().detach().cpu().numpy()

                # Get weight matrix if connection polarity is enabled
                weight_mat = None
                if self._settings.get("show_connection_polarity", False):
                    try:
                        from soen_toolkit.utils.model_tools import create_global_connection_matrix
                        weight_mat = create_global_connection_matrix(self._mgr.model).detach().cpu().numpy()
                    except Exception:
                        # If we can't get weights, fall back to regular coloring
                        pass

                total = mat.shape[0]
                for u in range(total):
                    for v in range(total):
                        if mat[v, u] == 0:
                            continue
                        lu = next(layer_id for layer_id in sorted_layer_ids if offsets[layer_id] <= u < offsets[layer_id] + dims[layer_id])
                        lv = next(layer_id for layer_id in sorted_layer_ids if offsets[layer_id] <= v < offsets[layer_id] + dims[layer_id])
                        if not self._settings["show_intra"] and lu == lv:
                            continue
                        src_port = f"p{u - offsets[lu]:04d}"
                        dst_port = f"p{v - offsets[lv]:04d}"
                        # Arrange connection points based on orientation
                        if lu == lv:
                            # Intra-layer: use side appropriate for orientation
                            if self._settings["orientation"] == "TB":
                                src = f"layer_{lu}:{src_port}:s"
                                dst = f"layer_{lv}:{dst_port}:s"
                            else:
                                src = f"layer_{lu}:{src_port}:e"
                                dst = f"layer_{lv}:{dst_port}:e"
                        elif lu > lv:
                            # Backward/feedback connection: source from "beginning" side, destination at "end" side
                            if self._settings["orientation"] == "TB":
                                src = f"layer_{lu}:{src_port}:n"
                                dst = f"layer_{lv}:{dst_port}:s"
                            else:
                                src = f"layer_{lu}:{src_port}:w"
                                dst = f"layer_{lv}:{dst_port}:e"
                        # Forward connection: use directional ports
                        elif self._settings["orientation"] == "TB":
                            src = f"layer_{lu}:{src_port}:s"
                            dst = f"layer_{lv}:{dst_port}:n"
                        else:
                            src = f"layer_{lu}:{src_port}:e"
                            dst = f"layer_{lv}:{dst_port}:w"

                        # Determine edge color based on gradient, polarity, or defaults
                        if self._settings.get("show_gradients", False) and self._gradient_data:
                            # Use gradient coloring for individual connections
                            # u is source neuron (global index), v is target neuron (global index)
                            # Need to convert to local indices within their respective layers
                            from_neuron = u - offsets[lu]
                            to_neuron = v - offsets[lv]
                            grad_val = get_connection_gradient(
                                self._gradient_data,
                                lu,
                                lv,
                                from_neuron,
                                to_neuron,
                            )
                            color = gradient_to_color(
                                grad_val,
                                self._gradient_config.grad_min if self._gradient_config else 0,
                                self._gradient_config.grad_max if self._gradient_config else 1,
                                colormap=self._settings.get("grad_colormap", "RdBu_r"),
                                log_scale=self._settings.get("grad_log_scale", False),
                            )
                        elif self._settings.get("show_connection_polarity", False) and weight_mat is not None:
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
                            color = self._settings["intra_color"] if lu == lv else self._settings["inter_color"]

                        attrs = {"color": color}
                        # Enhance visibility when showing gradients or polarity
                        if self._settings.get("show_gradients", False) or self._settings.get("show_connection_polarity", False):
                            # Make edges slightly thicker for better visibility
                            attrs["penwidth"] = str(float(self._settings.get("edge_thickness", 1.0)) * 1.5)
                            attrs["style"] = "bold"

                        # ALL neuron-level edges should NOT affect layout positioning.
                        # Only the invisible backbone edges between layer nodes should
                        # control the horizontal ordering. This prevents Graphviz from
                        # creating massive vertical gaps when routing many edges.
                        attrs.update({"constraint": "false", "weight": "0"})

                        if lu == lv or lu > lv:
                            # Intra-layer and backward edges: also set minlen=0
                            attrs["minlen"] = "0"

                        dot.edge(src, dst, **attrs)
            # Generate SVG via render-to-file then load bytes. This mirrors the working save path
            # and avoids occasional blank outputs from direct piping in some environments.
            tmp_base = None
            tmp_svg_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".gv") as tmp:
                    tmp_base = tmp.name
                # Render to SVG file; graphviz will append .svg
                out_path = dot.render(filename=tmp_base, cleanup=True)
                # Overlay descriptions beneath nodes when enabled (both views)
                svg_bytes = Path(out_path).read_bytes()
                if self._settings.get("show_desc", False):
                    with contextlib.suppress(Exception):
                        svg_bytes = self._inject_descriptions_svg(svg_bytes)
                # Update view from generated SVG
                self._svg_bytes = svg_bytes
                self._svg_bytes_base = svg_bytes

                # For hybrid view, add SVG to scene with overlay
                if not self._use_native:
                    # Vector SVG in scene for crisp zoom
                    from PyQt6.QtCore import QByteArray
                    from PyQt6.QtSvg import QSvgRenderer
                    from PyQt6.QtSvgWidgets import QGraphicsSvgItem

                    # Clear scene
                    self._overlay_scene.clear()

                    # Load renderer from bytes
                    renderer = QSvgRenderer(QByteArray(svg_bytes))
                    svg_item = QGraphicsSvgItem()
                    svg_item.setSharedRenderer(renderer)
                    svg_item.setZValue(0)
                    self._overlay_scene.addItem(svg_item)
                    size = renderer.defaultSize()

                    # Create overlay hit zones
                    viewbox_info = self._hit_overlay.parse_svg_and_create_overlay(
                        svg_bytes,
                        self._overlay_scene,
                        self._mgr.layers,
                        self._settings.get("simple_view", True),
                        debug=self._debug_clicks,
                        layer_color_hex=self._settings.get("layer_color", "#eae5be"),
                        show_layer_ids=self._settings.get("show_layer_ids", True),
                        show_desc=self._settings.get("show_desc", False),
                    )

                    # Store for coordinate mapping
                    self._svg_viewbox = viewbox_info.get("viewBox")
                    self._svg_render_size = viewbox_info.get("render_size")

                    # Fit in view
                    self._hybrid_view.fitInView(
                        self._overlay_scene.itemsBoundingRect(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                    )
                else:
                    # Original SVG widget loading for pure SVG mode
                    self._svg_widget.load(svg_bytes)
                    size = self._svg_widget.renderer().defaultSize()

                # Record size for zoom logic (hybrid or pure svg)
                self._orig_size = size
                self._apply_zoom()
                tmp_svg_path = out_path
            finally:
                # Clean up temp files
                try:
                    if tmp_base and os.path.exists(tmp_base):
                        os.remove(tmp_base)
                except Exception:
                    pass
                try:
                    if tmp_svg_path and os.path.exists(tmp_svg_path):
                        os.remove(tmp_svg_path)
                except Exception:
                    pass
        except FileNotFoundError:
            QMessageBox.critical(self, "GraphViz error", "'dot' not found; install GraphViz.")
        except Exception as e:
            # Provide richer diagnostics (Graphviz often puts details in stderr)
            try:
                import traceback

                details = str(e) or repr(e)
                stderr = getattr(e, "stderr", b"")
                if stderr:
                    with contextlib.suppress(Exception):
                        details += "\n\n" + stderr.decode("utf-8", errors="ignore")
                details += "\n\n" + traceback.format_exc()
            except Exception:
                details = str(e) or "Unknown error"
            QMessageBox.critical(self, "Render failed", details)
            # Clear SVG on error so the view doesn't show stale output
            self._svg_widget.load(b"")
            self._svg_bytes = None
            self._orig_size = None

    def _zoom_by(self, f) -> None:
        # Always allow zoom with hybrid/native views
        self._zoom *= f
        self._apply_zoom()

    def _apply_zoom(self) -> None:
        if not self._use_native and hasattr(self, "_hybrid_view"):
            # Hybrid zoom: fit content first, then apply extra zoom factor
            try:
                br = self._overlay_scene.itemsBoundingRect()
                if br and not br.isNull():
                    self._hybrid_view.resetTransform()
                    self._hybrid_view.fitInView(br, Qt.AspectRatioMode.KeepAspectRatio)
                    if abs(self._zoom - 1.0) > 1e-6:
                        self._hybrid_view.scale(self._zoom, self._zoom)
                return
            except Exception:
                self._hybrid_view.resetTransform()
                self._hybrid_view.scale(self._zoom, self._zoom)
                return

        if self._use_native:
            # Native zoom: fit content, then apply extra zoom factor
            try:
                br = self._scene.itemsBoundingRect()
                if br and not br.isNull():
                    self._native_view.resetTransform()
                    self._native_view.fitInView(br, Qt.AspectRatioMode.KeepAspectRatio)
                    if abs(self._zoom - 1.0) > 1e-6:
                        self._native_view.scale(self._zoom, self._zoom)
                return
            except Exception:
                # Fallback to simple scale
                self._native_view.resetTransform()
                self._native_view.scale(self._zoom, self._zoom)
                return

        if not self._orig_size:
            return
        w = int(self._orig_size.width() * self._zoom)
        # Add any extra height requested by SVG overlay step (scaled by zoom)
        h = int((self._orig_size.height() + self._extra_svg_height_px) * self._zoom)
        self._svg_widget.setFixedSize(w, h)

    def _save_svg(self) -> None:
        if not self._svg_bytes:
            QMessageBox.warning(self, "Nothing to save", "Click Render first.")
        return
        fname, _ = QFileDialog.getSaveFileName(self, "Save SVG", str(Path.home()), "SVG files (*.svg)")
        if fname:
            Path(fname).write_bytes(self._svg_bytes)
        QMessageBox.information(self, "Saved", f"SVG saved to:\n{fname}")

    def eventFilter(self, source, event):
        # Use unified handlers for graphics views
        if not self._use_native and hasattr(self, "_hybrid_interaction"):
            if self._hybrid_interaction.filter_event(source, event):
                return False  # Allow default handling after our processing

        if self._use_native and hasattr(self, "_native_interaction"):
            if self._native_interaction.filter_event(source, event):
                return False  # Allow default handling after our processing

        # Legacy SVG widget handling
        if (source is self._svg_widget) or (self._scroll is not None and source is self._scroll.viewport()):
            # Normalize acquisition of global pos across event types
            if event.type() == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
                gp = event.globalPosition().toPoint()
                self._handle_click_at_global_pos(gp)
                return True
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._pan_start = event.globalPosition().toPoint()
                self._last_press_global = self._pan_start
                if self._interaction_mode == "pan" and self._scroll is not None:
                    self._panning = True
                    self._orig_scroll = QPoint(
                        self._scroll.horizontalScrollBar().value(),
                        self._scroll.verticalScrollBar().value(),
                    )
                    with contextlib.suppress(Exception):
                        self._scroll.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                else:
                    self._panning = False
                return True
            if event.type() == QEvent.Type.MouseMove:
                if self._interaction_mode == "pan" and self._panning and self._scroll is not None:
                    delta = event.globalPosition().toPoint() - self._pan_start
                    self._scroll.horizontalScrollBar().setValue(self._orig_scroll.x() - delta.x())
                    self._scroll.verticalScrollBar().setValue(self._orig_scroll.y() - delta.y())
                    return True
            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                gp = event.globalPosition().toPoint()
                if self._interaction_mode == "pan":
                    # End pan, optionally treat as click if minimal motion
                    was_click = False
                    try:
                        if self._last_press_global is not None:
                            dx = gp.x() - self._last_press_global.x()
                            dy = gp.y() - self._last_press_global.y()
                            was_click = (dx * dx + dy * dy) <= 9
                    except Exception:
                        was_click = False
                    self._panning = False
                    try:
                        if self._scroll is not None:
                            self._scroll.viewport().setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                    except Exception:
                        pass
                    if was_click:
                        self._handle_click_at_global_pos(gp)
                    return True
                # Select mode: treat as click when small motion
                was_click = False
                try:
                    if self._last_press_global is not None:
                        dx = gp.x() - self._last_press_global.x()
                        dy = gp.y() - self._last_press_global.y()
                        was_click = (dx * dx + dy * dy) <= 9
                except Exception:
                    was_click = False
                if was_click:
                    self._handle_click_at_global_pos(gp)
                return True
        return super().eventFilter(source, event)

    def _apply_interaction_cursor(self) -> None:
        try:
            if self._interaction_mode == "pan":
                self._scroll.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            else:
                self._scroll.viewport().unsetCursor()
        except Exception:
            pass

    def _handle_click_at_global_pos(self, global_pt) -> None:
        if not self._svg_bytes or not self._hit_nodes:
            return
        try:
            # Map global position to SVG widget coordinates
            local = self._svg_widget.mapFromGlobal(global_pt)
            # Ignore clicks outside widget bounds
            if local.x() < 0 or local.y() < 0 or local.x() > self._svg_widget.width() or local.y() > self._svg_widget.height():
                return
            # Convert to SVG coordinate space:
            # 1) undo zoom scaling
            lx = local.x() / max(self._zoom, 1e-6)
            ly = local.y() / max(self._zoom, 1e-6)
            # 2) map widget pixels to SVG viewBox coordinates
            if self._svg_viewbox and self._svg_render_size:
                vb_x, vb_y, vb_w, vb_h = self._svg_viewbox
                r_w, r_h = self._svg_render_size
                sx = vb_x + lx * (vb_w / max(r_w, 1e-6))
                sy = vb_y + ly * (vb_h / max(r_h, 1e-6))
            else:
                # Fallback: assume 1:1 (may fail on some platforms)
                sx, sy = lx, ly
            x_svg, y_svg = sx, sy
            if self._debug_clicks:
                with contextlib.suppress(Exception):
                    self._overlay_debug_marker(x_svg, y_svg)
            # Hit test polygons (iterate in reverse so topmost wins if overlapping)
            hit_any = False
            for item in reversed(self._hit_nodes):
                inside = self._point_in_polygon(x_svg, y_svg, item["polygon"])
                # Apply a small margin in SVG units based on ~6px in current scale
                if not inside:
                    margin_x = 6.0 * (self._svg_viewbox[2] / max(self._svg_render_size[0], 1e-6)) if (self._svg_viewbox and self._svg_render_size) else 0.0
                    margin_y = 6.0 * (self._svg_viewbox[3] / max(self._svg_render_size[1], 1e-6)) if (self._svg_viewbox and self._svg_render_size) else 0.0
                    inside = self._point_near_polygon_bbox(x_svg, y_svg, item["polygon"], margin_x, margin_y)
                if inside:
                    hit_any = True
                    if self._debug_clicks:
                        with contextlib.suppress(Exception):
                            self._overlay_polygon(item["polygon"], color="#00aa00")
                    # In detailed view, attempt to estimate neuron index from vertical position
                    neuron_idx = None
                    if not self._settings.get("simple_view", True):
                        try:
                            row_bounds = item.get("row_bounds") or []
                            for idx, (y0, y1) in enumerate(row_bounds):
                                if y0 <= y_svg <= y1:
                                    neuron_idx = idx
                                    break
                            # Fallback to bbox proportional estimate
                            if neuron_idx is None and row_bounds:
                                min_y = row_bounds[0][0]
                                max_y = row_bounds[-1][1]
                                dim = len(row_bounds)
                                if max_y > min_y:
                                    rel = (y_svg - min_y) / (max_y - min_y)
                                    neuron_idx = max(0, min(dim - 1, int(rel * dim)))
                        except Exception:
                            neuron_idx = None
                    self._show_layer_details(item["layer_id"], neuron_idx)
                    return
            if not hit_any:
                # No direct hit. In debug, outline nearest to aid calibration; otherwise, do nothing.
                try:
                    if self._debug_clicks:
                        nearest = None
                        best_d2 = 1e12
                        for item in self._hit_nodes:
                            poly = item["polygon"]
                            cx = sum(p[0] for p in poly) / len(poly)
                            cy = sum(p[1] for p in poly) / len(poly)
                            d2 = (cx - x_svg) ** 2 + (cy - y_svg) ** 2
                            if d2 < best_d2:
                                best_d2 = d2
                                nearest = item
                        if nearest is not None:
                            self._overlay_polygon(nearest["polygon"], color="#aa0000")
                except Exception:
                    pass
        except Exception:
            return

    def _point_in_polygon(self, x, y, poly):
        # Ray casting algorithm for point-in-polygon
        inside = False
        n = len(poly)
        if n < 3:
            return False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
            if intersects:
                inside = not inside
            j = i
        return inside

    def _point_near_polygon_bbox(self, x: float, y: float, poly, margin_x: float, margin_y: float) -> bool:
        if not poly:
            return False
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        min_x, max_x = min(xs) - margin_x, max(xs) + margin_x
        min_y, max_y = min(ys) - margin_y, max(ys) + margin_y
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)

    def _build_hit_nodes(self, svg_bytes: bytes) -> None:
        from xml.etree import ElementTree as ET

        SVG_NS = "http://www.w3.org/2000/svg"
        ns = {"svg": SVG_NS}
        root = ET.fromstring(svg_bytes)
        # Cache viewBox and render size for coordinate mapping
        try:
            vb = root.attrib.get("viewBox")
            if vb:
                parts = [float(p) for p in vb.replace(",", " ").split()]
                if len(parts) == 4:
                    self._svg_viewbox = (parts[0], parts[1], parts[2], parts[3])
            else:
                self._svg_viewbox = None
        except Exception:
            self._svg_viewbox = None
        try:
            if self._orig_size:
                self._svg_render_size = (float(self._orig_size.width()), float(self._orig_size.height()))
            else:
                self._svg_render_size = None
        except Exception:
            self._svg_render_size = None
        hit_list = []
        for g in root.findall('.//svg:g[@class="node"]', ns):
            title_el = g.find("svg:title", ns)
            if title_el is None or not title_el.text:
                continue
            t = title_el.text.strip()
            try:
                layer_id = int(t.split("_", 1)[1]) if t.startswith("layer_") else int(t)
            except Exception:
                # If <title> is not a number, check id attribute from our emitted nodes
                node_id_attr = g.attrib.get("id") or ""
                if node_id_attr.startswith("node_"):
                    try:
                        layer_id = int(node_id_attr.split("_", 1)[1])
                    except Exception:
                        continue
                else:
                    continue
            # Prefer polygon bounds
            poly_el = g.find(".//svg:polygon", ns)
            points = []
            if poly_el is not None and "points" in poly_el.attrib:
                pts = poly_el.attrib["points"].strip().split()
                for pair in pts:
                    if "," in pair:
                        xs, ys = pair.split(",", 1)
                        with contextlib.suppress(ValueError):
                            points.append((float(xs), float(ys)))
            # Fallback to rect
            if not points:
                rect_el = g.find(".//svg:rect", ns)
                if rect_el is not None:
                    try:
                        x = float(rect_el.attrib.get("x", "0"))
                        y = float(rect_el.attrib.get("y", "0"))
                        w = float(rect_el.attrib.get("width", "0"))
                        h = float(rect_el.attrib.get("height", "0"))
                        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    except Exception:
                        points = []
            # Compute per-row y-buckets to enable exact neuron hit
            row_bounds = []
            try:
                # Find inner TD rows; Graphviz nests <table> elements. Collect all TDs with PORT attr
                for td in g.findall(".//svg:td", ns):
                    port = td.attrib.get("port") or td.attrib.get("PORT")
                    if not port:
                        continue
                    # Try to get surrounding rect to estimate y; fallback to use polygon bbox split
                    # ElementTree loses layout, but Graphviz often emits a <polygon> per node; we approximate by equal splits
                if points:
                    ys = sorted(p[1] for p in points)
                    min_y, max_y = (ys[0], ys[-1]) if ys else (0.0, 0.0)
                    dim = next((c.params.get("dim") for c in self._mgr.layers if c.layer_id == layer_id), 0)
                    if dim and max_y > min_y:
                        step = (max_y - min_y) / max(1, dim)
                        for i in range(dim):
                            y0 = min_y + i * step
                            y1 = min_y + (i + 1) * step
                            row_bounds.append((y0, y1))
            except Exception:
                row_bounds = []
            if points:
                hit_list.append({"layer_id": layer_id, "polygon": points, "row_bounds": row_bounds})
        self._hit_nodes = hit_list

    def _show_layer_details(self, layer_id: int, neuron_index: int | None = None) -> None:
        cfg = next((c for c in self._mgr.layers if c.layer_id == layer_id), None)
        if cfg is None:
            return
        # Build HTML summary
        simple_view = self._settings.get("simple_view", True)
        parts = []
        parts.append(f"<h3>Layer {cfg.layer_id}</h3>")
        parts.append(f"<p><b>Type:</b> {cfg.layer_type}</p>")
        if not simple_view and neuron_index is not None:
            parts.append(f"<p><b>Approx. neuron index:</b> {neuron_index}</p>")
            # Fan-in / Fan-out for this neuron using global connection mask (not weights)
            try:
                if getattr(self._mgr, "model", None) is not None:
                    # Use mask to count actual connections, not weight values
                    mat = self._mgr.model.get_global_connection_mask().detach().cpu().numpy()
                    dims = {c.layer_id: c.params.get("dim", 0) for c in self._mgr.layers}
                    sorted_layer_ids = sorted(dims)
                    offsets = {}
                    running = 0
                    for lid in sorted_layer_ids:
                        offsets[lid] = running
                        running += dims[lid]
                    g_idx = offsets.get(cfg.layer_id, 0) + int(neuron_index)
                    if 0 <= g_idx < mat.shape[0]:
                        import numpy as np

                        fan_in = int(np.count_nonzero(mat[g_idx, :]))
                        fan_out = int(np.count_nonzero(mat[:, g_idx]))
                        parts.append(f"<p><b>Fan-in:</b> {fan_in} &nbsp;&nbsp; <b>Fan-out:</b> {fan_out}</p>")
            except Exception:
                pass
        if cfg.description:
            parts.append(f"<p><b>Description:</b> {cfg.description}</p>")
        # Parameters table
        try:
            params_rows = "".join([f"<tr><td><code>{k}</code></td><td>{v}</td></tr>" for k, v in cfg.params.items()])
            parts.append("<h4>Parameters</h4>")
            parts.append(f"<table border='1' cellspacing='0' cellpadding='4'>{params_rows}</table>")
        except Exception:
            pass
        # Live parameter values from model layer (vectors)
        try:
            if getattr(self._mgr, "model", None) is not None:
                model = self._mgr.model
                # Map layer id -> module index
                lid_to_idx = {c.layer_id: i for i, c in enumerate(model.layers_config)}
                idx = lid_to_idx.get(cfg.layer_id)
                if idx is not None and 0 <= idx < len(model.layers):
                    layer_mod = model.layers[idx]
                    named = dict(layer_mod.named_parameters())
                    vector_keys = [k for k, v in named.items() if v.ndim == 1 and v.numel() == cfg.params.get("dim")]
                    if neuron_index is not None and vector_keys:
                        rows = []
                        for k in sorted(vector_keys):
                            try:
                                val = named[k].detach().cpu().flatten()[neuron_index].item()
                                rows.append(f"<tr><td><code>{k}[{neuron_index}]</code></td><td>{val:.6g}</td></tr>")
                            except Exception:
                                continue
                        if rows:
                            parts.append("<h4>Node Values</h4>")
                            parts.append("<table border='1' cellspacing='0' cellpadding='4'>" + "".join(rows) + "</table>")
        except Exception:
            pass
        # Connections summary (more useful in simple view)
        try:
            incoming = [c for c in self._mgr.connections if c.to_layer == layer_id]
            outgoing = [c for c in self._mgr.connections if c.from_layer == layer_id]
            parts.append("<h4>Connections</h4>")
            parts.append(f"<p><b>Incoming:</b> {len(incoming)} | <b>Outgoing:</b> {len(outgoing)}</p>")
            if simple_view:
                if incoming:
                    parts.append("<p><b>Incoming from:</b> " + ", ".join(str(c.from_layer) for c in incoming) + "</p>")
                if outgoing:
                    parts.append("<p><b>Outgoing to:</b> " + ", ".join(str(c.to_layer) for c in outgoing) + "</p>")
        except Exception:
            pass
        html = "\n".join(parts)
        self._details_view.setHtml(html)
        # Open the sidebar; if unavailable, show a temporary message box
        try:
            self._open_sidebar()
        except Exception:
            with contextlib.suppress(Exception):
                QMessageBox.information(self, "Selection", f"Layer {cfg.layer_id} selected.")

    def _load_polarity_data(self) -> dict[int, list]:
        """Load polarity data for all layers.

        Returns:
            Dict mapping layer_id to polarity list.
        """
        layer_polarity = {}
        dims = {cfg.layer_id: cfg.params.get("dim", 0) for cfg in self._mgr.layers}

        for cfg in self._mgr.layers:
            try:
                # Check for explicit polarity, polarity_file, or polarity_init
                polarity_explicit = cfg.params.get("polarity")
                polarity_file = cfg.params.get("polarity_file")
                polarity_init = cfg.params.get("polarity_init")

                if polarity_explicit is not None:
                    import torch
                    if isinstance(polarity_explicit, list):
                        layer_polarity[cfg.layer_id] = polarity_explicit
                    elif isinstance(polarity_explicit, torch.Tensor):
                        layer_polarity[cfg.layer_id] = polarity_explicit.cpu().tolist()
                    else:
                        layer_polarity[cfg.layer_id] = polarity_explicit
                elif polarity_file:
                    from soen_toolkit.core.layers.common.connectivity_metadata import load_neuron_polarity
                    polarity_tensor = load_neuron_polarity(polarity_file)
                    layer_polarity[cfg.layer_id] = polarity_tensor.tolist()
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

        return layer_polarity

    def _open_sidebar(self) -> None:
        """Show the details sidebar with robust fallback sizing."""
        total_w = self._splitter.size().width() if self._splitter else 0
        right_min = max(260, self._details_panel.minimumWidth() if self._details_panel else 260)
        left = max(200, total_w - right_min) if total_w else 600
        right = right_min if total_w else 320
        with contextlib.suppress(Exception):
            self._splitter.setSizes([left, right])
            self._details_panel.show()

    def _overlay_polygon(self, poly, color="#00aa00") -> None:
        if not self._svg_bytes_base:
            return
        from xml.etree import ElementTree as ET

        SVG_NS = "http://www.w3.org/2000/svg"
        root = ET.fromstring(self._svg_bytes_base)
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in poly)
        pg = ET.Element(
            f"{{{SVG_NS}}}polygon",
            {
                "points": pts,
                "fill": "none",
                "stroke": color,
                "stroke-width": "1.5",
                "stroke-dasharray": "3,2",
            },
        )
        root.append(pg)
        new_bytes = ET.tostring(root, encoding="utf-8")
        self._svg_widget.load(new_bytes)
        self._orig_size = self._svg_widget.renderer().defaultSize()
        self._apply_zoom()

    def _overlay_debug_marker(self, x_svg: float, y_svg: float) -> None:
        """Reload the SVG with a small marker at svg coords for visual verification."""
        if not self._svg_bytes_base:
            return
        from xml.etree import ElementTree as ET

        SVG_NS = "http://www.w3.org/2000/svg"
        root = ET.fromstring(self._svg_bytes_base)

        # Create a small crosshair
        def _line(x1, y1, x2, y2):
            return ET.Element(
                f"{{{SVG_NS}}}line",
                {
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                    "stroke": "#ff0000",
                    "stroke-width": "1.5",
                },
            )

        size = 6.0
        root.append(_line(x_svg - size, y_svg, x_svg + size, y_svg))
        root.append(_line(x_svg, y_svg - size, x_svg, y_svg + size))
        # Also add a small circle
        circ = ET.Element(
            f"{{{SVG_NS}}}circle",
            {
                "cx": f"{x_svg:.2f}",
                "cy": f"{y_svg:.2f}",
                "r": "2.5",
                "fill": "none",
                "stroke": "#ff0000",
                "stroke-width": "1.5",
            },
        )
        root.append(circ)
        new_bytes = ET.tostring(root, encoding="utf-8")
        # Preserve original base for subsequent overlays (polygon/next clicks)
        self._svg_bytes = new_bytes
        self._svg_widget.load(new_bytes)
        # Preserve zoom and size
        self._orig_size = self._svg_widget.renderer().defaultSize()
        self._apply_zoom()

    def _inject_descriptions_svg(self, svg_bytes: bytes) -> bytes:
        """Overlay layer descriptions as SVG text within each node group.
        This avoids changing Graphviz layout/geometry while still showing text.
        """
        from xml.etree import ElementTree as ET

        SVG_NS = "http://www.w3.org/2000/svg"
        ns = {"svg": SVG_NS}
        root = ET.fromstring(svg_bytes)
        # reset extra padding; will be recomputed
        self._extra_svg_height_px = 0.0

        # Map layer_id -> description
        desc_map = {cfg.layer_id: (cfg.description or "").strip() for cfg in self._mgr.layers}
        if not any(desc_map.values()):
            return svg_bytes

        # Iterate over node groups produced by Graphviz
        max_text_baseline = 0.0
        for g in root.findall('.//svg:g[@class="node"]', ns):
            # Robustly resolve layer id from <title> or the group's id attribute.
            title_el = g.find("svg:title", ns)
            lid = None
            t = title_el.text.strip() if title_el is not None and title_el.text else ""
            if t:
                try:
                    # Simple view often uses numeric titles
                    lid = int(t)
                except Exception:
                    # Detailed view uses node name like "layer_3"; also support "node_3"
                    if t.startswith(("layer_", "node_")):
                        try:
                            lid = int(t.split("_", 1)[1])
                        except Exception:
                            lid = None
            if lid is None:
                node_id_attr = g.attrib.get("id") or ""
                if node_id_attr.startswith("node_"):
                    try:
                        lid = int(node_id_attr.split("_", 1)[1])
                    except Exception:
                        lid = None
            if lid is None:
                continue

            desc = desc_map.get(lid, "")
            if not desc:
                continue

            # Compute a union bounding box across ALL polygons/rects in the node group.
            xs, ys = [], []
            for pol in g.findall(".//svg:polygon", ns):
                points_attr = pol.attrib.get("points", "").strip()
                if not points_attr:
                    continue
                for pair in points_attr.split():
                    if not pair or "," not in pair:
                        continue
                    x_str, y_str = pair.split(",", 1)
                    try:
                        xs.append(float(x_str))
                        ys.append(float(y_str))
                    except ValueError:
                        continue
            for rect_el in g.findall(".//svg:rect", ns):
                try:
                    x = float(rect_el.attrib.get("x", "0"))
                    y = float(rect_el.attrib.get("y", "0"))
                    w = float(rect_el.attrib.get("width", "0"))
                    h = float(rect_el.attrib.get("height", "0"))
                    xs.extend([x, x + w])
                    ys.extend([y, y + h])
                except Exception:
                    continue
            if not xs or not ys:
                continue

            min_x, max_x = min(xs), max(xs)
            max_y = max(ys)
            center_x = (min_x + max_x) / 2.0
            baseline_y = max_y + 12.0  # a bit below the box
            max_text_baseline = max(max_text_baseline, baseline_y)

            # Create <text> element with contrasting color
            bg_color = self._settings.get("bg_color", "#ffffff")
            desc_text_color = _get_contrast_text_color(bg_color)
            text_el = ET.Element(
                f"{{{SVG_NS}}}text",
                {
                    "x": f"{center_x:.2f}",
                    "y": f"{baseline_y:.2f}",
                    "font-size": "10",
                    "font-style": "italic",
                    "text-anchor": "middle",
                    "fill": desc_text_color,
                },
            )
            text_el.text = desc
            g.append(text_el)

        # If added text extends below the original canvas, expand the SVG viewBox/height
        try:
            vb = root.attrib.get("viewBox")
            if vb:
                parts = [float(p) for p in vb.replace(",", " ").split()]
                if len(parts) == 4:
                    _x0, y0, w, h = parts
                    extra = max(0.0, (max_text_baseline + 14.0) - (y0 + h))  # 14px for text height + padding
                    if extra > 0:
                        # record extra so widget size and background can grow even if renderer ignores it
                        self._extra_svg_height_px = max(self._extra_svg_height_px, extra)
                        parts[3] = h + extra
                        root.set("viewBox", f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")

                        # Also bump the explicit height attribute if present (preserving unit)
                        def _bump(attr_name: str) -> None:
                            s = root.attrib.get(attr_name)
                            if not s:
                                return
                            # Split numeric prefix and unit suffix
                            num = ""
                            unit = ""
                            for ch in s:
                                if ch.isdigit() or ch == "." or (ch == "-" and not num):
                                    num += ch
                                else:
                                    unit += s[s.index(ch) :]
                                    break
                            try:
                                new_val = float(num) + extra
                                root.set(attr_name, f"{new_val}{unit}")
                            except Exception:
                                pass

                        _bump("height")
                        # Ensure a full-canvas white background so added space isn't transparent
                        try:
                            # Remove any existing top-level background polygon/rect we can't easily resize
                            for child in list(root):
                                if child.tag.endswith("polygon") or child.tag.endswith("rect"):
                                    # Heuristic: Graphviz background often has no id and stroke="none" or fill="white"
                                    attrs = child.attrib
                                    if attrs.get("fill", "").lower() in ("#ffffff", "white") and attrs.get("stroke", "none") in ("none", "transparent"):
                                        root.remove(child)
                            # Insert our own background rect
                            W, H = parts[2], parts[3]
                            bg = ET.Element(
                                f"{{{SVG_NS}}}rect",
                                {
                                    "x": "0",
                                    "y": "0",
                                    "width": f"{W}",
                                    "height": f"{H}",
                                    "fill": "#ffffff",
                                    "stroke": "none",
                                },
                            )
                            root.insert(0, bg)
                            # As a fallback, also set root background via style
                            style = root.attrib.get("style", "")
                            if "background-color" not in style:
                                root.set("style", (style + "; " if style else "") + f"background-color:{self._settings.get('bg_color', 'white')}")
                        except Exception:
                            pass

        except Exception:
            # If anything goes wrong, return the SVG with text anyway
            pass

        return ET.tostring(root, encoding="utf-8")
