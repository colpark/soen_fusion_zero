# src/soen_toolkit/model_creation_gui/tabs/tab_histograms.py
from __future__ import annotations

import logging

# import pyqtgraph as pg # No longer needed for heatmap
# --- Matplotlib Imports (lazy-loaded on first use) ---
# import matplotlib
# matplotlib.use('QtAgg') # Ensure Qt backend is used
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# from mpl_toolkits.axes_grid1 import make_axes_locatable # Import for colorbar sizing
# ---
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg  # Re-add pyqtgraph import
import torch  # <-- Add torch import

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.model_manager import ModelManager

log = logging.getLogger(__name__)

# Define base figure size constants
BASE_HEATMAP_FIG_WIDTH = 8
BASE_HEATMAP_FIG_HEIGHT = 6
MIN_HEATMAP_SCALE = 0.2
COMBO_MIN_HEIGHT = 34
COMBO_STYLE = """
QComboBox {
    padding: 6px 10px;
    min-height: 34px;
}
QComboBox QAbstractItemView {
    padding: 6px 8px;
}
"""


class HistogramsTab(QWidget):
    """Displays histograms for model layer parameters and connection weights."""

    def __init__(self, manager: ModelManager) -> None:
        super().__init__()
        self._mgr = manager
        # Initialize figure scale factor
        self._heatmap_fig_scale = 1.0
        self._init_ui()
        # Connection to manager.model_changed handled by MainWindow

    def _style_combo(self, combo: QComboBox) -> None:
        """Ensure dropdowns are tall enough to read selections comfortably."""
        combo.setMinimumHeight(COMBO_MIN_HEIGHT)
        combo.setStyleSheet(COMBO_STYLE)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # --- Toolbar for view selection ---
        self.view_toolbar = QToolBar("Histogram Views")
        self.view_action_group = QActionGroup(self)
        self.view_action_group.setExclusive(True)

        self.act_layers = QAction("Layers", self)
        self.act_layers.setCheckable(True)
        self.act_conn_hist = QAction("Connection Histogram", self)
        self.act_conn_hist.setCheckable(True)
        self.act_conn_map = QAction("Connection Heatmap", self)
        self.act_conn_map.setCheckable(True)

        for act in [self.act_layers, self.act_conn_hist, self.act_conn_map]:
            self.view_toolbar.addAction(act)
            self.view_action_group.addAction(act)

        main_layout.addWidget(self.view_toolbar)

        # --- Stacked Widget for Views ---
        self.view_stack = QStackedWidget()
        main_layout.addWidget(self.view_stack)

        # --- View 1: Layer Histogram ---
        layer_view_widget = QWidget()
        layer_layout = QVBoxLayout(layer_view_widget)
        layer_controls_form = QFormLayout()
        self.layer_combo = QComboBox()
        self.layer_combo.setMinimumContentsLength(25)
        self._style_combo(self.layer_combo)
        layer_controls_form.addRow("Select Layer:", self.layer_combo)
        self.param_combo = QComboBox()
        self.param_combo.setMinimumContentsLength(25)
        self._style_combo(self.param_combo)
        layer_controls_form.addRow("Select Parameter:", self.param_combo)
        layer_layout.addLayout(layer_controls_form)
        self.layer_refresh_btn = QPushButton("Plot Layer Histogram")
        layer_layout.addWidget(self.layer_refresh_btn)
        self.layer_plot = pg.PlotWidget()
        layer_layout.addWidget(self.layer_plot, 1)
        self.view_stack.addWidget(layer_view_widget)

        # --- View 2: Connection Histogram ---
        conn_hist_view_widget = QWidget()
        conn_hist_layout = QVBoxLayout(conn_hist_view_widget)
        conn_hist_controls_form = QFormLayout()
        self.conn_combo_hist = QComboBox()
        self.conn_combo_hist.setMinimumContentsLength(30)
        self._style_combo(self.conn_combo_hist)
        conn_hist_controls_form.addRow("Select Connection:", self.conn_combo_hist)
        self.exclude_zero_chk = QCheckBox("Exclude Zero Weights")
        self.exclude_zero_chk.setChecked(True)
        conn_hist_controls_form.addRow(self.exclude_zero_chk)
        conn_hist_layout.addLayout(conn_hist_controls_form)
        self.conn_refresh_hist_btn = QPushButton("Refresh Connection Histogram")
        conn_hist_layout.addWidget(self.conn_refresh_hist_btn)
        self.conn_plot = pg.PlotWidget()
        conn_hist_layout.addWidget(self.conn_plot, 1)
        self.view_stack.addWidget(conn_hist_view_widget)

        # --- View 3: Connection Heatmap (Using Matplotlib) ---
        conn_map_view_widget = QWidget()
        conn_map_layout = QVBoxLayout(conn_map_view_widget)

        # Controls Row (Combo, Refresh, Scale, Adjacency Toggle)
        controls_row = QHBoxLayout()
        conn_map_controls_form = QFormLayout()
        self.conn_combo_map = QComboBox()
        self.conn_combo_map.setMinimumContentsLength(30)
        self._style_combo(self.conn_combo_map)
        conn_map_controls_form.addRow("Select Connection:", self.conn_combo_map)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("Weights", "weights")
        self.view_mode_combo.addItem("Masks", "mask")
        self._style_combo(self.view_mode_combo)
        conn_map_controls_form.addRow("View Mode:", self.view_mode_combo)
        controls_row.addLayout(conn_map_controls_form)

        self.conn_refresh_map_btn = QPushButton("Refresh")
        controls_row.addWidget(self.conn_refresh_map_btn)

        # Use scale terminology for tooltips if possible
        self.scale_up_btn = QPushButton("+")
        self.scale_up_btn.setToolTip("Make Plot Larger")
        self.scale_up_btn.setFixedWidth(30)
        controls_row.addWidget(self.scale_up_btn)

        self.scale_down_btn = QPushButton("-")
        self.scale_down_btn.setToolTip("Make Plot Smaller")
        self.scale_down_btn.setFixedWidth(30)
        controls_row.addWidget(self.scale_down_btn)

        # Optional Reset Button
        self.reset_scale_btn = QPushButton("Reset Size")
        self.reset_scale_btn.setToolTip("Reset plot size to default")
        controls_row.addWidget(self.reset_scale_btn)

        # Add Adjacency Toggle Checkbox
        self.adjacency_chk = QCheckBox("Show Adjacency (Binary)")
        self.adjacency_chk.setToolTip("Show 1 for non-zero weights, 0 otherwise")
        controls_row.addWidget(self.adjacency_chk)

        # Add Log Scale Toggle Checkbox
        self.log_scale_chk = QCheckBox("Log Scale")
        self.log_scale_chk.setToolTip("Display weight magnitudes in log10 scale")
        controls_row.addWidget(self.log_scale_chk)

        controls_row.addStretch()
        conn_map_layout.addLayout(controls_row)

        # Create ScrollArea
        self.heatmap_scroll_area = QScrollArea()
        self.heatmap_scroll_area.setWidgetResizable(False)
        # Ensure scrollbars appear when needed
        self.heatmap_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.heatmap_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Initialize canvas placeholder
        self.heatmap_canvas = None
        self.heatmap_fig = None
        self.heatmap_ax = None
        self._heatmap_colorbar = None

        conn_map_layout.addWidget(self.heatmap_scroll_area, 1)
        self.view_stack.addWidget(conn_map_view_widget)

        # --- Connect Signals/Slots ---
        # Toolbar actions change stacked widget index
        self.act_layers.triggered.connect(lambda: self.view_stack.setCurrentIndex(0))
        self.act_conn_hist.triggered.connect(lambda: self.view_stack.setCurrentIndex(1))
        self.act_conn_map.triggered.connect(lambda: self.view_stack.setCurrentIndex(2))

        # Layer view controls
        self.layer_combo.currentIndexChanged.connect(self._update_param_combo)
        self.layer_refresh_btn.clicked.connect(self._plot_layer_histogram)

        # Connection histogram view controls
        # Keep combo selections separate for now
        # self.conn_combo_hist.currentIndexChanged.connect(self._sync_conn_combos)
        self.conn_refresh_hist_btn.clicked.connect(self._plot_conn_histogram)

        # Connection heatmap view controls
        self.conn_refresh_map_btn.clicked.connect(self._plot_conn_heatmap)
        self.view_mode_combo.currentIndexChanged.connect(self._plot_conn_heatmap)
        # Connect to scaling slots
        self.scale_up_btn.clicked.connect(self._scale_heatmap_larger)
        self.scale_down_btn.clicked.connect(self._scale_heatmap_smaller)
        self.reset_scale_btn.clicked.connect(self._reset_heatmap_scale)
        # Connect adjacency toggle
        self.adjacency_chk.stateChanged.connect(self._plot_conn_heatmap)
        # Connect log scale toggle
        self.log_scale_chk.stateChanged.connect(self._plot_conn_heatmap)

        # Set initial state
        self.act_layers.setChecked(True)
        self.view_stack.setCurrentIndex(0)

    def _remove_old_heatmap_canvas(self) -> None:
        """Safely removes the old Matplotlib canvas from the scroll area."""
        if self.heatmap_canvas is not None:
            # Remove the event filter before deleting
            try:
                self.heatmap_canvas.removeEventFilter(self)
            except RuntimeError:  # Filter might already be removed
                pass
            old_widget = self.heatmap_scroll_area.takeWidget()
            if old_widget:
                old_widget.deleteLater()  # Schedule for deletion
            # Clear references
            self.heatmap_canvas = None
            self.heatmap_ax = None
            self.heatmap_fig = None
            self._heatmap_colorbar = None

    def _on_model_changed(self) -> None:
        """Slot called when the model manager signals a model change."""
        log.debug("HistogramsTab reacting to model change.")
        # Debug mask count
        if self._mgr.model is not None:
            mask_count = len(getattr(self._mgr.model, "connection_masks", {}))
            conn_count = len(getattr(self._mgr.model, "connections", {}))
            log.debug("HistogramsTab: model has %d connections, %d masks", conn_count, mask_count)
        # Update all combo boxes
        self._update_layer_combo()
        self._update_conn_combo()
        # Clear potentially stale plots
        self.layer_plot.clear()
        self.conn_plot.clear()
        # Clear Matplotlib heatmap plot by removing the canvas
        self._remove_old_heatmap_canvas()

    def _update_layer_combo(self) -> None:
        """Populate layer selection combo with current model layers."""
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        if self._mgr.model is not None:
            model = self._mgr.model
            for idx, cfg in enumerate(model.layers_config):
                name = f"Layer {cfg.layer_id}" if hasattr(cfg, "layer_id") else f"Layer {idx}"
                self.layer_combo.addItem(name, idx)
        self.layer_combo.blockSignals(False)
        self._update_param_combo()

    def _update_conn_combo(self) -> None:
        """Populate BOTH connection selection combos including Global Matrix option."""
        # Block signals
        self.conn_combo_hist.blockSignals(True)
        self.conn_combo_map.blockSignals(True)
        # Clear
        self.conn_combo_hist.clear()
        self.conn_combo_map.clear()
        # Populate with individual connections
        if self._mgr.model is not None:
            model = self._mgr.model
            keys = sorted(model.connections.keys())
            for key in keys:
                self.conn_combo_hist.addItem(str(key), key)
                self.conn_combo_map.addItem(str(key), key)

            # Add Global Matrix option if the method exists
            if hasattr(model, "get_global_connection_matrix"):
                # Add separator for clarity (optional)
                # self.conn_combo_hist.insertSeparator(self.conn_combo_hist.count())
                # self.conn_combo_map.insertSeparator(self.conn_combo_map.count())
                global_key = "__GLOBAL__"
                global_text = "Global Matrix"
                self.conn_combo_hist.addItem(global_text, global_key)
                self.conn_combo_map.addItem(global_text, global_key)

        # Unblock signals
        self.conn_combo_hist.blockSignals(False)
        self.conn_combo_map.blockSignals(False)
        if self.conn_combo_map.count() > 0:
            index = self.conn_combo_map.findData(global_key) if "global_key" in locals() else -1
            if index >= 0:
                self.conn_combo_map.setCurrentIndex(index)
                self.conn_combo_hist.setCurrentIndex(index)

    def _update_param_combo(self) -> None:
        """Populate parameter selection combo for the selected layer."""
        self.param_combo.blockSignals(True)  # Prevent trigger during clear/populate
        self.param_combo.clear()
        idx = self.layer_combo.currentData()
        if self._mgr.model is not None and idx is not None:
            try:
                layer = self._mgr.model.layers[idx]
                param_names = [name for name, _ in layer.named_parameters()]
                if param_names:
                    for name in sorted(param_names):
                        display_name = name.removeprefix("log_")
                        self.param_combo.addItem(display_name, name)
                else:
                    self.param_combo.addItem("No parameters", None)
            except IndexError:
                log.exception(f"Layer index {idx} out of range when updating parameter combo.")
            except Exception as e:
                log.exception(f"Error updating parameter combo for layer {idx}: {e}")
        self.param_combo.blockSignals(False)

    def _plot_layer_histogram(self) -> None:
        """Plot histogram of parameters for selected layer."""
        if self._mgr.model is None:
            QMessageBox.warning(self, "No model", "Build or load a model first.")
            return
        idx = self.layer_combo.currentData()
        param_name = self.param_combo.currentData()
        if idx is None or param_name is None:
            log.warning("Layer histogram plot skipped: Invalid layer index or parameter name.")
            self.layer_plot.clear()
            return
        layer = self._mgr.model.layers[idx]
        try:  # Get parameter tensor
            param = dict(layer.named_parameters())[param_name]
        except KeyError:
            QMessageBox.warning(self, "Parameter error", f"Parameter '{param_name}' not found in layer {idx}.")
            self.layer_plot.clear()
            return
        arr = param.detach().cpu().numpy().ravel()
        if param_name.startswith("log_"):
            arr = np.exp(arr)
        self.layer_plot.clear()
        if arr.size > 0:
            # Plot histogram bars
            try:
                counts, bins = np.histogram(arr, bins=50)
                if len(bins) > 1:
                    widths = bins[1:] - bins[:-1]
                    bg = pg.BarGraphItem(x=bins[:-1], height=counts, width=widths, brush="steelblue")
                    self.layer_plot.addItem(bg)
                else:
                    log.warning(f"Could not plot histogram for {param_name} in layer {idx}: Not enough bins.")
            except Exception as e:
                log.exception(f"Error creating histogram for {param_name} in layer {idx}: {e}")
                QMessageBox.warning(self, "Plotting error", f"Could not generate histogram: {e}")
        label_name = param_name.removeprefix("log_")
        self.layer_plot.setLabel("bottom", f"{label_name} Value")
        self.layer_plot.setLabel("left", "Count")

    def _plot_conn_histogram(self) -> None:
        """Plot histogram of weights for selected connection OR Global Matrix."""
        if self._mgr.model is None:
            QMessageBox.warning(self, "No model", "Build or load a model first.")
            return

        key = self.conn_combo_hist.currentData()
        if key is None:
            log.warning("Connection histogram plot skipped: No connection selected.")
            self.conn_plot.clear()
            return

        try:
            if key == "__GLOBAL__":
                if not hasattr(self._mgr.model, "get_global_connection_matrix"):
                    QMessageBox.warning(self, "Error", "Model does not support global matrix retrieval.")
                    self.conn_plot.clear()
                    return
                param_or_matrix = self._mgr.model.get_global_connection_matrix()
            else:
                param_or_matrix = self._mgr.model.connections[key]

        except (KeyError, AttributeError) as e:
            log.exception(f"Error accessing connection/matrix '{key}': {e}")
            QMessageBox.critical(self, "Internal Error", f"Could not access data for '{key}'.")
            self.conn_plot.clear()
            return

        # Ensure we have a tensor before detaching
        if not isinstance(param_or_matrix, torch.Tensor):
            log.error(f"Data for '{key}' is not a Tensor (type: {type(param_or_matrix)}). Cannot plot histogram.")
            QMessageBox.critical(self, "Data Error", f"Data for '{key}' is not a plottable Tensor.")
            self.conn_plot.clear()
            return

        arr = param_or_matrix.detach().cpu().numpy().ravel()

        if self.exclude_zero_chk.isChecked():
            arr = arr[arr != 0]

        # Plot histogram
        self.conn_plot.clear()
        if arr.size > 0:
            try:
                counts, bins = np.histogram(arr, bins=50)
                if len(bins) > 1:
                    widths = bins[1:] - bins[:-1]
                    bg = pg.BarGraphItem(x=bins[:-1], height=counts, width=widths, brush="orange")
                    self.conn_plot.addItem(bg)
                else:
                    log.warning(f"Could not plot histogram for connection {key}: Not enough bins.")
            except Exception as e:
                log.exception(f"Error creating histogram for connection {key}: {e}")
                QMessageBox.warning(self, "Plotting error", f"Could not generate histogram for {key}: {e}")

        self.conn_plot.setLabel("bottom", "Weight Value")
        self.conn_plot.setLabel("left", "Count")

    def _plot_conn_heatmap(self) -> None:
        """Plot heatmap, toggling between weights and adjacency."""
        # Remove any previous canvas first
        self._remove_old_heatmap_canvas()

        if self._mgr.model is None:
            # No need to explicitly clear if canvas is already removed
            # QMessageBox.warning(self, "No model", "Build or load a model first.")
            return

        key = self.conn_combo_map.currentData()
        if key is None:
            # log.warning("Connection heatmap plot skipped: No connection selected.")
            return

        try:
            if key == "__GLOBAL__":
                if not hasattr(self._mgr.model, "get_global_connection_matrix") or not hasattr(self._mgr.model, "layers_config"):
                    QMessageBox.warning(self, "Error", "Model missing required attributes for global matrix plot.")
                    return
                if self.view_mode_combo.currentData() == "mask":
                    if not hasattr(self._mgr.model, "get_global_connection_mask"):
                        QMessageBox.warning(self, "Error", "Model does not provide connection masks.")
                        return
                    param_or_matrix = self._mgr.model.get_global_connection_mask()
                else:
                    param_or_matrix = self._mgr.model.get_global_connection_matrix()
                layers_config = self._mgr.model.layers_config  # Get layer info for boundaries
            elif self.view_mode_combo.currentData() == "mask":
                param_or_matrix = self._mgr.model.connection_masks.get(key)
                if param_or_matrix is None:
                    QMessageBox.warning(self, "Error", f"No mask stored for connection '{key}'.")
                    return
            else:
                param_or_matrix = self._mgr.model.connections[key]

        except (KeyError, AttributeError) as e:
            log.exception(f"Error accessing connection/matrix '{key}': {e}")
            QMessageBox.critical(self, "Internal Error", f"Could not access data for '{key}'.")
            return  # Canvas already removed

        # Ensure we have a tensor before detaching
        if not isinstance(param_or_matrix, torch.Tensor):
            log.error(f"Data for '{key}' is not a Tensor (type: {type(param_or_matrix)}). Cannot plot heatmap.")
            QMessageBox.critical(self, "Data Error", f"Data for '{key}' is not a plottable Tensor.")
            return  # Canvas already removed

        mat = param_or_matrix.detach().cpu().numpy()

        # Check matrix validity before proceeding
        if mat.ndim != 2:
            log.warning(f"Skipping heatmap for {key}: Matrix is not 2D (shape: {mat.shape}).")
            return
        # Allow empty matrix plot for adjacency view, but not weights? Or just return?
        # Let's allow empty adjacency for now.
        is_adjacency_view = self.adjacency_chk.isChecked()
        showing_mask = self.view_mode_combo.currentData() == "mask"
        if mat.size == 0 and not is_adjacency_view:
            log.warning(f"Skipping weights heatmap for {key}: Matrix is empty.")
            return

        try:
            # --- Create new Figure/Canvas with current scale ---
            fig_w = BASE_HEATMAP_FIG_WIDTH * self._heatmap_fig_scale
            fig_h = BASE_HEATMAP_FIG_HEIGHT * self._heatmap_fig_scale

            # Lazy-load matplotlib here (only when actually needed)
            import matplotlib as mpl

            mpl.use("QtAgg")  # Ensure Qt backend is used
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            from matplotlib.figure import Figure
            from mpl_toolkits.axes_grid1 import (
                make_axes_locatable,  # Import for colorbar sizing
            )

            self.heatmap_fig = Figure(figsize=(fig_w, fig_h), dpi=100)
            self.heatmap_ax = self.heatmap_fig.add_subplot(111)
            self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
            self._heatmap_colorbar = None
            # --- Plotting Start ---

            # Clear previous axes content and remove previous colorbar if exists
            self.heatmap_ax.cla()
            if hasattr(self, "_heatmap_colorbar") and self._heatmap_colorbar is not None:
                try:
                    self._heatmap_colorbar.remove()
                except Exception as e:
                    log.warning(f"Error removing previous heatmap colorbar: {e}")
                finally:
                    self._heatmap_colorbar = None

            # --- Prepare data and plot parameters based on toggle ---
            if showing_mask:
                plot_mat = mat
                cmap = "binary"
                vmin, vmax = 0, 1
                add_colorbar = False
            elif is_adjacency_view:
                plot_mat = (mat != 0).astype(float)
                cmap = "binary"  # Or 'gray_r' for black=connection
                vmin, vmax = 0, 1
                add_colorbar = False
                if mat.size == 0:
                    # Handle empty case for adjacency
                    plot_mat = np.zeros((1, 1))  # Plot a single empty pixel
            else:  # Show weights
                # Use RdBu_r so positive (excitatory) -> red, negative (inhibitory) -> blue
                cmap = "RdBu_r"
                show_log_scale = self.log_scale_chk.isChecked()
                if show_log_scale:
                    # Log transform preserving sign: sign(w) * log10(|w| + 1)
                    # Adding 1 avoids log(0) = -inf and keeps small values near zero
                    signs = np.sign(mat)
                    plot_mat = signs * np.log10(np.abs(mat) + 1)
                    # Symmetric limits around zero
                    abs_max = np.max(np.abs(plot_mat))
                    if abs_max == 0:
                        abs_max = 1
                    vmin, vmax = -abs_max, abs_max
                else:
                    plot_mat = mat
                    min_val, max_val = mat.min(), mat.max()
                    abs_max = max(abs(min_val), abs(max_val))
                    vmin, vmax = -abs_max, abs_max
                    if abs_max == 0:
                        vmin, vmax = -1, 1  # Avoid error in imshow/colorbar
                add_colorbar = True
            # --------------------------------------------------------

            # Plot the heatmap
            ax = self.heatmap_ax
            im = ax.imshow(plot_mat, cmap=cmap, interpolation="nearest", aspect="equal", vmin=vmin, vmax=vmax)

            # Add colorbar ONLY if showing weights
            if add_colorbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.4)
                self._heatmap_colorbar = self.heatmap_fig.colorbar(im, cax=cax)
                # Label colorbar based on log scale setting
                cbar_label = "sign(w) * log10(|w|+1)" if show_log_scale else "Weight"
                self._heatmap_colorbar.set_label(cbar_label, rotation=270, labelpad=15)
            else:
                self._heatmap_colorbar = None  # Ensure it's None if not added

            # Set labels and title - SWAPPED LABELS to match matrix structure
            ax.set_xlabel("Source Neuron Index")  # X-axis is the second dim: source
            ax.set_ylabel("Destination Neuron Index")  # Y-axis is the first dim: destination
            if showing_mask:
                mode_label = "Mask"
            elif is_adjacency_view:
                mode_label = "Adjacency"
            elif show_log_scale:
                mode_label = "Weights (log)"
            else:
                mode_label = "Weights"
            ax.set_title(f"Connection Matrix ({mode_label}): {key}")

            # Set explicit limits AFTER plotting
            # Limits should be based on original matrix shape, even for empty adjacency
            ax.set_xlim(-0.5, mat.shape[1] - 0.5 if mat.ndim == 2 else 0.5)
            ax.set_ylim(-0.5, mat.shape[0] - 0.5 if mat.ndim == 2 else 0.5)

            # --- Add Layer Boundary Lines (Only for Global Matrix) ---
            if key == "__GLOBAL__" and layers_config:
                num_layers = len(layers_config)
                layer_ids = [cfg.layer_id for cfg in layers_config]
                layer_dims = [cfg.params.get("dim", 0) for cfg in layers_config]
                boundaries = np.cumsum([0, *layer_dims])  # Start/end indices [0, d1, d1+d2, ...]

                # Iterate through layers STARTING FROM THE SECOND LAYER (index 1)
                for i in range(1, num_layers + 1):
                    # Determine color based on the PREVIOUS layer (i-1)
                    prev_layer_idx = i - 1
                    if prev_layer_idx == 0:
                        line_color = "green"  # Input layer ended
                    elif prev_layer_idx == num_layers - 1:
                        line_color = "red"  # Output layer started (line drawn after last hidden)
                    else:
                        line_color = "blue"  # Hidden layer ended

                    # Boundary index marks the START of layer i (end of layer i-1)
                    boundary_idx = boundaries[i]

                    # Don't draw lines at the very edge (0 or total size)
                    # Draw line separating layer i-1 from layer i using color of layer i-1
                    if 0 < boundary_idx < mat.shape[1]:
                        # Source axis boundary (Vertical line)
                        self.heatmap_ax.axvline(
                            x=boundary_idx - 0.5,
                            color=line_color,
                            linestyle="--",
                            linewidth=1.5,
                            zorder=3,
                        )
                    if 0 < boundary_idx < mat.shape[0]:
                        # Destination axis boundary (Horizontal line)
                        self.heatmap_ax.axhline(
                            y=boundary_idx - 0.5,
                            color=line_color,
                            linestyle="--",
                            linewidth=1.5,
                            zorder=3,
                        )

                # --- Add Layer ID Labels on Top and Right ---
                # Create secondary axes for layer labels
                ax_top = ax.secondary_xaxis("top")
                ax_right = ax.secondary_yaxis("right")

                # Compute tick positions (center of each layer's range)
                layer_centers = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(num_layers)]
                layer_labels = [str(lid) for lid in layer_ids]

                # Set ticks and labels for top axis (Source layers)
                ax_top.set_xticks(layer_centers)
                ax_top.set_xticklabels(layer_labels, fontsize=8)
                ax_top.set_xlabel("Source Layer ID", fontsize=9)

                # Set ticks and labels for right axis (Destination layers)
                ax_right.set_yticks(layer_centers)
                ax_right.set_yticklabels(layer_labels, fontsize=8)
                ax_right.set_ylabel("Dest Layer ID", fontsize=9)
            # --- End Layer Boundary Lines ---

            # --- Add new canvas to scroll area, install filter, and draw ---
            self.heatmap_scroll_area.setWidget(self.heatmap_canvas)
            self.heatmap_canvas.installEventFilter(self)
            self.heatmap_canvas.draw()

        except Exception as e:
            log.exception(f"Error plotting heatmap for connection {key} using Matplotlib: {e}")
            QMessageBox.critical(self, "Heatmap Plot Error", f"Failed to render heatmap: {e}")
            # Clean up the failed canvas attempt
            self._remove_old_heatmap_canvas()

    def eventFilter(self, source, event):
        """Handle wheel events over the heatmap canvas for scrolling."""
        if source is self.heatmap_canvas and event.type() == QEvent.Type.Wheel:
            # Get the vertical scrollbar
            scrollbar = self.heatmap_scroll_area.verticalScrollBar()
            # Calculate new scroll value based on wheel delta
            # The delta is usually +/- 120 per notch
            delta = event.angleDelta().y()
            new_value = scrollbar.value() - int(delta / 4)  # Adjust sensitivity factor (4) as needed
            scrollbar.setValue(new_value)
            return True  # Event handled

        # Pass event through for other cases
        return super().eventFilter(source, event)

    def _scale_heatmap(self, factor) -> None:
        """Helper to change scale factor and replot."""
        new_scale = self._heatmap_fig_scale * factor
        # Prevent scaling down too much
        self._heatmap_fig_scale = max(MIN_HEATMAP_SCALE, new_scale)
        self._plot_conn_heatmap()  # Replot with the new scale

    def _scale_heatmap_larger(self) -> None:
        self._scale_heatmap(1.25)  # Increase size

    def _scale_heatmap_smaller(self) -> None:
        self._scale_heatmap(0.8)  # Decrease size (1 / 1.25)

    def _reset_heatmap_scale(self) -> None:
        """Resets the heatmap scale to default and replots."""
        self._heatmap_fig_scale = 1.0
        self._plot_conn_heatmap()

    # ... (eventFilter for wheel scroll) ...
