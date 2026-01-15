"""Visualization settings dialog."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QColor, QFontMetrics
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class VisualisationSettingsDialog(QDialog):
    """Dialog to configure visualization settings in organized tabs."""

    def __init__(self, parent, settings: dict) -> None:
        super().__init__(parent)
        self.setWindowTitle("Visualization Settings")
        self._settings = settings
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # --- General tab ---
        gen_tab = QWidget()
        gen_form = QFormLayout(gen_tab)
        # Group layers by model ID
        self.group_by_model_chk = QCheckBox()
        self.group_by_model_chk.setChecked(self._settings.get("group_by_model", False))
        self.group_by_model_chk.setToolTip(
            "Group layers with the same model_id into visual clusters.\n"
            "Useful for multi-module architectures."
        )
        gen_form.addRow("Group by Model ID:", self.group_by_model_chk)
        # Layout direction
        self.orient_cb = QComboBox()
        self.orient_cb.addItems(["Left→Right", "Top→Bottom"])
        self.orient_cb.setCurrentIndex(0 if self._settings["orientation"] == "LR" else 1)
        # Autosize to contents so text is not clipped
        self.orient_cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.orient_cb.setToolTip("Direction of the layout: Left→Right or Top→Bottom")
        gen_form.addRow("Layout Direction:", self.orient_cb)
        # Edge routing
        self.routing_cb = QComboBox()
        # Keep only robust, distinct options
        self.routing_cb.addItems(["true", "false", "ortho"])
        self.routing_cb.setCurrentText(self._settings["edge_routing"])
        self.routing_cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        gen_form.addRow("Edge Routing:", self.routing_cb)
        # Ensure popup widths are wide enough for items
        try:
            self._autosize_combo(self.orient_cb)
            self._autosize_combo(self.routing_cb)
        except Exception:
            pass
        # Arrow thickness
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.1, 10.0)
        self.thickness_spin.setSingleStep(0.1)
        self.thickness_spin.setDecimals(1)
        self.thickness_spin.setValue(self._settings["edge_thickness"])
        gen_form.addRow("Arrow Thickness:", self.thickness_spin)
        # Arrow size
        self.arrow_size_spin = QDoubleSpinBox()
        self.arrow_size_spin.setRange(0.2, 3.0)
        self.arrow_size_spin.setSingleStep(0.1)
        self.arrow_size_spin.setDecimals(1)
        self.arrow_size_spin.setValue(self._settings.get("arrow_size", 1.0))
        gen_form.addRow("Arrow Size:", self.arrow_size_spin)
        # Layer spacing
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.0, 10.0)
        self.spacing_spin.setSingleStep(0.1)
        self.spacing_spin.setDecimals(1)
        self.spacing_spin.setValue(self._settings.get("layer_spacing", 1.0))
        gen_form.addRow("Layer Spacing:", self.spacing_spin)
        tabs.addTab(gen_tab, "General")

        # --- Appearance tab ---
        app_tab = QWidget()
        app_form = QFormLayout(app_tab)
        # Theme selector (applies preset colors)
        self.theme_cb = QComboBox()
        self.theme_cb.addItems(["default", "dark"])
        self.theme_cb.setCurrentText(str(self._settings.get("theme", "default")))
        app_form.addRow("Theme:", self.theme_cb)
        self.inter_btn = QPushButton(self._settings["inter_color"])
        self.inter_btn.setStyleSheet(f"background-color: {self._settings['inter_color']}")
        self.inter_btn.clicked.connect(lambda: self._pick_color("inter_color", self.inter_btn))
        app_form.addRow("Inter-edge Color:", self.inter_btn)
        self.intra_btn = QPushButton(self._settings["intra_color"])
        self.intra_btn.setStyleSheet(f"background-color: {self._settings['intra_color']}")
        self.intra_btn.clicked.connect(lambda: self._pick_color("intra_color", self.intra_btn))
        app_form.addRow("Intra-edge Color:", self.intra_btn)
        # Layer background color
        self.layer_btn = QPushButton(self._settings.get("layer_color", "#eae5be"))
        self.layer_btn.setStyleSheet(f"background-color: {self._settings.get('layer_color', '#eae5be')}")
        self.layer_btn.clicked.connect(lambda: self._pick_color("layer_color", self.layer_btn))
        app_form.addRow("Layer Color:", self.layer_btn)
        # Canvas background color
        self.bg_btn = QPushButton(self._settings.get("bg_color", "#ffffff"))
        self.bg_btn.setStyleSheet(f"background-color: {self._settings.get('bg_color', '#ffffff')}")
        self.bg_btn.clicked.connect(lambda: self._pick_color("bg_color", self.bg_btn))
        app_form.addRow("Canvas Background:", self.bg_btn)
        # Text color
        self.text_btn = QPushButton(self._settings.get("text_color", "#000000"))
        self.text_btn.setStyleSheet(f"background-color: {self._settings.get('text_color', '#000000')}")
        self.text_btn.clicked.connect(lambda: self._pick_color("text_color", self.text_btn))
        app_form.addRow("Text Color:", self.text_btn)
        # Toggle layer outline visibility
        self.layer_outline_chk = QCheckBox()
        self.layer_outline_chk.setChecked(self._settings.get("show_layer_outline", True))
        app_form.addRow("Show Layer Outline:", self.layer_outline_chk)
        tabs.addTab(app_tab, "Appearance")

        # Apply theme presets to color controls when theme changes
        self.theme_cb.currentTextChanged.connect(self._apply_theme_to_controls)

        # --- Details tab ---
        det_tab = QWidget()
        det_form = QFormLayout(det_tab)
        # Simple view toggle (when unchecked, detailed view)
        self.simple_view_chk = QCheckBox()
        self.simple_view_chk.setChecked(self._settings.get("simple_view", True))
        det_form.addRow("Simple View (compact):", self.simple_view_chk)
        # Detailed view layout selector (only applies when simple_view is off)
        self.detailed_layout_cb = QComboBox()
        self.detailed_layout_cb.addItems(["Linear", "Circular", "Grid"])
        current_layout = self._settings.get("detailed_layout", "linear")
        self.detailed_layout_cb.setCurrentText(current_layout.capitalize())
        self.detailed_layout_cb.setToolTip(
            "Layout for detailed view: Linear (vertical column), Circular (nodes in a circle), or Grid (weight matrix view)"
        )
        det_form.addRow("Detailed Layout:", self.detailed_layout_cb)
        # Show intra-layer connections toggle
        self.intra_chk = QCheckBox()
        self.intra_chk.setChecked(self._settings["show_intra"])
        det_form.addRow("Show Intra-layer:", self.intra_chk)
        # Show neuron indices inside layer columns
        self.node_id_chk = QCheckBox()
        self.node_id_chk.setChecked(self._settings.get("show_node_ids", False))
        det_form.addRow("Show Neuron Indices:", self.node_id_chk)
        # Show layer IDs at the top of each layer column
        self.layer_id_chk = QCheckBox()
        self.layer_id_chk.setChecked(self._settings.get("show_layer_ids", True))
        det_form.addRow("Show Layer IDs:", self.layer_id_chk)
        # Show connection descriptions on edges
        self.desc_chk = QCheckBox()
        self.desc_chk.setChecked(self._settings["show_desc"])
        det_form.addRow("Show Descriptions:", self.desc_chk)
        # Show connection type labels on edges (simple view only)
        self.conn_type_chk = QCheckBox()
        self.conn_type_chk.setChecked(self._settings.get("show_conn_type", False))
        det_form.addRow("Show Connection Type:", self.conn_type_chk)
        # Show model IDs toggle (like show layer IDs)
        self.show_model_ids_chk = QCheckBox()
        self.show_model_ids_chk.setChecked(self._settings.get("show_model_ids", True))
        det_form.addRow("Show Model IDs:", self.show_model_ids_chk)
        # Show neuron polarity toggle (detailed view only)
        self.neuron_polarity_chk = QCheckBox()
        self.neuron_polarity_chk.setChecked(self._settings.get("show_neuron_polarity", False))
        det_form.addRow("Show Neuron Polarity:", self.neuron_polarity_chk)
        # Show connection polarity toggle (detailed view only)
        self.connection_polarity_chk = QCheckBox()
        self.connection_polarity_chk.setChecked(self._settings.get("show_connection_polarity", False))
        det_form.addRow("Show Connection Polarity:", self.connection_polarity_chk)
        tabs.addTab(det_tab, "Details")

        # --- Gradient Flow tab ---
        grad_tab = QWidget()
        grad_layout = QVBoxLayout(grad_tab)

        # Dataset settings group
        data_group = QGroupBox("Dataset")
        data_form = QFormLayout(data_group)

        # HDF5 file path
        data_row = QHBoxLayout()
        self.grad_data_path = QLineEdit()
        self.grad_data_path.setText(self._settings.get("grad_hdf5_path", ""))
        self.grad_data_path.setPlaceholderText("Select HDF5 data file...")
        data_row.addWidget(self.grad_data_path, 1)
        self.grad_browse_btn = QPushButton("Browse...")
        self.grad_browse_btn.clicked.connect(self._browse_grad_data)
        data_row.addWidget(self.grad_browse_btn)
        data_form.addRow("Data file:", data_row)

        # Data split
        self.grad_split_cb = QComboBox()
        self.grad_split_cb.addItems(["train", "val", "test", "all_data"])
        self.grad_split_cb.setCurrentText(self._settings.get("grad_split", "train"))
        data_form.addRow("Split:", self.grad_split_cb)

        # Sequence length
        self.grad_seq_len_spin = QSpinBox()
        self.grad_seq_len_spin.setRange(1, 100000)
        self.grad_seq_len_spin.setValue(self._settings.get("grad_seq_len", 100))
        self.grad_seq_len_spin.valueChanged.connect(self._update_grad_time_info)
        data_form.addRow("Seq length:", self.grad_seq_len_spin)

        # Time mode row
        time_row = QHBoxLayout()
        self.grad_time_mode_cb = QComboBox()
        self.grad_time_mode_cb.addItems(["Specify dt", "Specify total time (ns)"])
        self.grad_time_mode_cb.setCurrentIndex(
            0 if self._settings.get("grad_time_mode", "dt") == "dt" else 1
        )
        self.grad_time_mode_cb.setMinimumWidth(140)
        self.grad_time_mode_cb.currentTextChanged.connect(self._on_grad_time_mode_changed)
        time_row.addWidget(self.grad_time_mode_cb)

        self.grad_dt_label = QLabel("dt:")
        time_row.addWidget(self.grad_dt_label)
        self.grad_dt_edit = QLineEdit()
        self.grad_dt_edit.setText(str(self._settings.get("grad_dt", 37.0)))
        self.grad_dt_edit.setMaximumWidth(80)
        self.grad_dt_edit.setToolTip("Dimensionless timestep (default 37 ≈ 1ns)")
        self.grad_dt_edit.textChanged.connect(self._update_grad_time_info)
        time_row.addWidget(self.grad_dt_edit)

        self.grad_total_time_label = QLabel("Total (ns):")
        self.grad_total_time_label.setVisible(False)
        time_row.addWidget(self.grad_total_time_label)
        self.grad_total_time_edit = QLineEdit()
        self.grad_total_time_edit.setMaximumWidth(80)
        self.grad_total_time_edit.setVisible(False)
        self.grad_total_time_edit.setToolTip("Total simulation time in nanoseconds")
        self.grad_total_time_edit.textChanged.connect(self._update_grad_time_info)
        total_time_val = self._settings.get("grad_total_time_ns")
        if total_time_val is not None:
            self.grad_total_time_edit.setText(str(total_time_val))
        time_row.addWidget(self.grad_total_time_edit)

        time_row.addStretch()
        data_form.addRow("Time:", time_row)

        # Time info label
        self.grad_time_info_label = QLabel()
        self.grad_time_info_label.setStyleSheet("color: #666; font-size: 10px;")
        data_form.addRow("", self.grad_time_info_label)

        # Apply initial time mode visibility
        self._on_grad_time_mode_changed(self.grad_time_mode_cb.currentText())
        self._update_grad_time_info()

        # Feature scaling
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Min:"))
        self.grad_feature_min = QLineEdit()
        self.grad_feature_min.setPlaceholderText("auto")
        fmin = self._settings.get("grad_feature_min")
        self.grad_feature_min.setText(str(fmin) if fmin is not None else "")
        self.grad_feature_min.setMaximumWidth(80)
        scale_row.addWidget(self.grad_feature_min)
        scale_row.addWidget(QLabel("Max:"))
        self.grad_feature_max = QLineEdit()
        self.grad_feature_max.setPlaceholderText("auto")
        fmax = self._settings.get("grad_feature_max")
        self.grad_feature_max.setText(str(fmax) if fmax is not None else "")
        self.grad_feature_max.setMaximumWidth(80)
        scale_row.addWidget(self.grad_feature_max)
        scale_row.addStretch()
        data_form.addRow("Feature scale:", scale_row)

        # Task type selector
        task_row = QHBoxLayout()
        self.grad_task_type_cb = QComboBox()
        self.grad_task_type_cb.addItems(["Classification", "Seq2Seq (regression)"])
        is_seq2seq = self._settings.get("grad_task_type", "classification") == "seq2seq"
        self.grad_task_type_cb.setCurrentText(
            "Seq2Seq (regression)" if is_seq2seq else "Classification"
        )
        self.grad_task_type_cb.currentTextChanged.connect(self._on_grad_task_type_changed)
        task_row.addWidget(self.grad_task_type_cb)
        task_row.addStretch()
        data_form.addRow("Task type:", task_row)

        # Sample selection row
        sample_row = QHBoxLayout()
        self.grad_class_label = QLabel("Class:")
        sample_row.addWidget(self.grad_class_label)
        self.grad_class_cb = QComboBox()
        self.grad_class_cb.setEditable(True)
        self.grad_class_cb.setMinimumWidth(80)
        self.grad_class_cb.currentIndexChanged.connect(self._on_grad_class_changed)
        sample_row.addWidget(self.grad_class_cb)
        self.grad_sample_label = QLabel("Sample:")
        sample_row.addWidget(self.grad_sample_label)
        self.grad_sample_spin = QSpinBox()
        self.grad_sample_spin.setRange(0, 999999)
        self.grad_sample_spin.setValue(self._settings.get("grad_sample_index", 0))
        sample_row.addWidget(self.grad_sample_spin)
        self.grad_new_sample_btn = QPushButton("New Random Sample")
        self.grad_new_sample_btn.clicked.connect(self._grad_random_sample)
        sample_row.addWidget(self.grad_new_sample_btn)
        sample_row.addStretch()
        data_form.addRow("Sample:", sample_row)

        # Populate classes if dataset is already loaded
        self._grad_labels = None  # Will store labels from HDF5 for class filtering
        self._populate_grad_classes()

        # Set initial class value if saved
        if self._settings.get("grad_class_id") is not None:
            self.grad_class_cb.setCurrentText(str(self._settings.get("grad_class_id", 0)))

        # Apply initial task type visibility
        self._on_grad_task_type_changed(self.grad_task_type_cb.currentText())

        grad_layout.addWidget(data_group)

        # Gradient settings group
        grad_group = QGroupBox("Gradient Settings")
        grad_form = QFormLayout(grad_group)

        # Loss function
        self.grad_loss_cb = QComboBox()
        self.grad_loss_cb.addItems(["mse", "cross_entropy", "sum_output"])
        self.grad_loss_cb.setCurrentText(self._settings.get("grad_loss_fn", "mse"))
        grad_form.addRow("Loss function:", self.grad_loss_cb)

        # Log scale
        self.grad_log_scale_chk = QCheckBox()
        self.grad_log_scale_chk.setChecked(self._settings.get("grad_log_scale", False))
        self.grad_log_scale_chk.setToolTip("Apply log10(1 + |x|) transform preserving sign")
        grad_form.addRow("Log scale:", self.grad_log_scale_chk)

        # Colormap
        self.grad_colormap_cb = QComboBox()
        self.grad_colormap_cb.addItems(
            ["RdBu_r", "coolwarm", "seismic", "viridis", "plasma", "magma"]
        )
        self.grad_colormap_cb.setCurrentText(self._settings.get("grad_colormap", "RdBu_r"))
        grad_form.addRow("Colormap:", self.grad_colormap_cb)

        grad_layout.addWidget(grad_group)
        grad_layout.addStretch()

        tabs.addTab(grad_tab, "Gradient Flow")

        layout.addWidget(tabs)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _autosize_combo(self, cb: QComboBox) -> None:
        """Ensure combo popup is wide enough for items."""
        metrics = QFontMetrics(cb.font())
        max_w = 0
        for i in range(cb.count()):
            max_w = max(max_w, metrics.horizontalAdvance(cb.itemText(i)))
        # Add padding for checkmark/scrollbar
        popup_w = max_w + 40
        cb.view().setMinimumWidth(popup_w)
        # Also widen the closed control a bit
        cb.setMinimumContentsLength(max(10, int(max_w / metrics.horizontalAdvance("X"))))

    def _apply_theme_to_controls(self, theme: str) -> None:
        """Apply theme preset colors to controls."""
        if theme == "dark":
            inter = "#e5e5e5"
            intra = "#7aa2f7"
            layer = "#d0d0d0"  # Light gray for layers in dark theme
            bg = "#1e1e1e"    # Dark gray background
            text = "#1e1e1e"  # Dark text (unused - contrast is auto-computed)
        else:
            inter = "#000000"
            intra = "#ff0000"
            layer = "#eae5be"
            bg = "#ffffff"
            text = "#000000"
        for btn, val, key in [
            (self.inter_btn, inter, "inter_color"),
            (self.intra_btn, intra, "intra_color"),
            (self.layer_btn, layer, "layer_color"),
            (self.bg_btn, bg, "bg_color"),
            (self.text_btn, text, "text_color"),
        ]:
            btn.setText(val)
            btn.setStyleSheet(f"background-color: {val}")
            self._settings[key] = val

    def _pick_color(self, key, button) -> None:
        initial = QColor(self._settings[key])
        c = QColorDialog.getColor(initial, self, "Pick Color")
        if c.isValid():
            self._settings[key] = c.name()
            button.setStyleSheet(f"background-color: {c.name()}")
            button.setText(c.name())

    def _browse_grad_data(self) -> None:
        """Browse for HDF5 data file for gradient computation."""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 Data File",
            str(Path.home()),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if fname:
            self.grad_data_path.setText(fname)
            self._populate_grad_classes()

    def _populate_grad_classes(self) -> None:
        """Populate class selector from dataset by reading labels directly from HDF5."""
        # Guard: don't run if widgets not yet created
        if not hasattr(self, "grad_data_path") or not hasattr(self, "grad_class_cb"):
            return

        path_str = self.grad_data_path.text().strip()
        if not path_str or not Path(path_str).exists():
            return

        try:
            from soen_toolkit.training.data.dataloaders import open_hdf5_with_consistent_locking

            split = self.grad_split_cb.currentText()
            if split == "all_data":
                split = None

            # Load labels directly from HDF5 to get accurate class info
            with open_hdf5_with_consistent_locking(path_str) as f:
                labels_ds = f[split]["labels"] if split else f["labels"]
                self._grad_labels = labels_ds[:]  # Store labels for filtering

            # Get unique classes
            unique_classes = sorted(set(int(lbl) for lbl in self._grad_labels))

            # Populate class dropdown
            self.grad_class_cb.clear()
            for c in unique_classes:
                self.grad_class_cb.addItem(str(c))

            # Update sample range
            self._update_grad_sample_range()

        except Exception:
            # Non-fatal - just leave class selector empty
            pass

    def _on_grad_class_changed(self) -> None:
        """Handle class selection change."""
        if not hasattr(self, "grad_sample_spin"):
            return
        self._update_grad_sample_range()

    def _on_grad_task_type_changed(self, task_text: str) -> None:
        """Handle task type change - show/hide classification controls."""
        if not hasattr(self, "grad_class_label"):
            return
        is_classification = "classification" in task_text.lower()
        self.grad_class_label.setVisible(is_classification)
        self.grad_class_cb.setVisible(is_classification)
        self.grad_sample_label.setVisible(is_classification)
        self.grad_sample_spin.setVisible(is_classification)
        self.grad_new_sample_btn.setVisible(is_classification)

    def _get_grad_indices_for_class(self, class_id: int) -> list[int]:
        """Get indices of samples for a given class from stored labels."""
        if not hasattr(self, "_grad_labels") or self._grad_labels is None:
            return []

        # Check task type - seq2seq uses all samples
        if hasattr(self, "grad_task_type_cb"):
            is_seq2seq = "seq2seq" in self.grad_task_type_cb.currentText().lower()
            if is_seq2seq:
                return list(range(len(self._grad_labels)))

        # Classification - filter by class
        return [i for i, lbl in enumerate(self._grad_labels) if int(lbl) == class_id]

    def _get_grad_class_id(self) -> int:
        """Get current class ID from UI."""
        try:
            return int(self.grad_class_cb.currentText())
        except ValueError:
            return 0

    def _grad_random_sample(self) -> None:
        """Select a random sample."""
        import random

        if not hasattr(self, "_grad_labels") or self._grad_labels is None:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset first.")
            return

        class_id = self._get_grad_class_id()
        indices = self._get_grad_indices_for_class(class_id)

        if not indices:
            QMessageBox.warning(self, "No Samples", f"No samples found for class {class_id}")
            return

        self.grad_sample_spin.setValue(random.randint(0, len(indices) - 1))

    def _update_grad_sample_range(self) -> None:
        """Update sample spinner range based on selected class."""
        if not hasattr(self, "_grad_labels") or self._grad_labels is None:
            return

        class_id = self._get_grad_class_id()
        indices = self._get_grad_indices_for_class(class_id)

        if indices:
            max_idx = len(indices) - 1
            self.grad_sample_spin.setMaximum(max_idx)
            # Clamp current value if it exceeds new max
            if self.grad_sample_spin.value() > max_idx:
                self.grad_sample_spin.setValue(max_idx)
        else:
            self.grad_sample_spin.setMaximum(0)

    def _on_grad_time_mode_changed(self, mode_text: str) -> None:
        """Toggle visibility of dt vs total time controls."""
        # Guard: don't run if widgets not yet created
        if not hasattr(self, "grad_dt_label"):
            return
        is_dt_mode = "dt" in mode_text.lower()
        self.grad_dt_label.setVisible(is_dt_mode)
        self.grad_dt_edit.setVisible(is_dt_mode)
        self.grad_total_time_label.setVisible(not is_dt_mode)
        self.grad_total_time_edit.setVisible(not is_dt_mode)
        self._update_grad_time_info()

    def _update_grad_time_info(self) -> None:
        """Update the time info label with computed values."""
        # Guard: don't run if label not yet created
        if not hasattr(self, "grad_time_info_label"):
            return
        try:
            seq_len = self.grad_seq_len_spin.value()
            is_dt_mode = "dt" in self.grad_time_mode_cb.currentText().lower()

            if is_dt_mode:
                dt_text = self.grad_dt_edit.text().strip()
                if dt_text:
                    dt = float(dt_text)
                    # dt of 37 ≈ 1ns, so total_time_ns = seq_len * dt / 37
                    total_ns = seq_len * dt / 37.0
                    self.grad_time_info_label.setText(
                        f"Total time: {total_ns:.2f} ns ({seq_len} steps × dt={dt})"
                    )
                else:
                    self.grad_time_info_label.setText("")
            else:
                total_text = self.grad_total_time_edit.text().strip()
                if total_text:
                    total_ns = float(total_text)
                    # Compute effective dt: dt = total_ns * 37 / seq_len
                    dt = total_ns * 37.0 / seq_len if seq_len > 0 else 0
                    self.grad_time_info_label.setText(
                        f"Effective dt: {dt:.2f} ({seq_len} steps over {total_ns} ns)"
                    )
                else:
                    self.grad_time_info_label.setText("")
        except ValueError:
            self.grad_time_info_label.setText("")

    def get_settings(self):
        """Return the updated settings dictionary."""
        # General settings
        self._settings["group_by_model"] = self.group_by_model_chk.isChecked()
        self._settings["orientation"] = "LR" if self.orient_cb.currentIndex() == 0 else "TB"
        self._settings["edge_routing"] = self.routing_cb.currentText()
        self._settings["edge_thickness"] = float(self.thickness_spin.value())
        self._settings["arrow_size"] = float(self.arrow_size_spin.value())
        self._settings["layer_spacing"] = float(self.spacing_spin.value())
        # Appearance colors
        self._settings["inter_color"] = self.inter_btn.text()
        self._settings["intra_color"] = self.intra_btn.text()
        self._settings["layer_color"] = self.layer_btn.text()
        self._settings["bg_color"] = self.bg_btn.text()
        self._settings["text_color"] = self.text_btn.text()
        self._settings["show_layer_outline"] = self.layer_outline_chk.isChecked()
        self._settings["theme"] = self.theme_cb.currentText()
        # Details toggles
        self._settings["simple_view"] = self.simple_view_chk.isChecked()
        self._settings["detailed_layout"] = self.detailed_layout_cb.currentText().lower()
        self._settings["show_intra"] = self.intra_chk.isChecked()
        self._settings["show_desc"] = self.desc_chk.isChecked()
        self._settings["show_conn_type"] = self.conn_type_chk.isChecked()
        self._settings["show_node_ids"] = self.node_id_chk.isChecked()
        self._settings["show_layer_ids"] = self.layer_id_chk.isChecked()
        self._settings["show_model_ids"] = self.show_model_ids_chk.isChecked()
        self._settings["show_neuron_polarity"] = self.neuron_polarity_chk.isChecked()
        self._settings["show_connection_polarity"] = self.connection_polarity_chk.isChecked()
        # Gradient flow settings
        self._settings["grad_hdf5_path"] = self.grad_data_path.text()
        self._settings["grad_split"] = self.grad_split_cb.currentText()
        self._settings["grad_seq_len"] = self.grad_seq_len_spin.value()
        # Time mode settings
        is_dt_mode = "dt" in self.grad_time_mode_cb.currentText().lower()
        self._settings["grad_time_mode"] = "dt" if is_dt_mode else "total_time"
        try:
            dt_text = self.grad_dt_edit.text().strip()
            self._settings["grad_dt"] = float(dt_text) if dt_text else 37.0
        except ValueError:
            self._settings["grad_dt"] = 37.0
        try:
            total_text = self.grad_total_time_edit.text().strip()
            self._settings["grad_total_time_ns"] = float(total_text) if total_text else None
        except ValueError:
            self._settings["grad_total_time_ns"] = None
        # Parse feature scaling (empty = None)
        try:
            fmin_text = self.grad_feature_min.text().strip()
            self._settings["grad_feature_min"] = float(fmin_text) if fmin_text else None
        except ValueError:
            self._settings["grad_feature_min"] = None
        try:
            fmax_text = self.grad_feature_max.text().strip()
            self._settings["grad_feature_max"] = float(fmax_text) if fmax_text else None
        except ValueError:
            self._settings["grad_feature_max"] = None
        # Task type
        is_seq2seq = "seq2seq" in self.grad_task_type_cb.currentText().lower()
        self._settings["grad_task_type"] = "seq2seq" if is_seq2seq else "classification"
        # Sample selection
        try:
            self._settings["grad_class_id"] = int(self.grad_class_cb.currentText())
        except ValueError:
            self._settings["grad_class_id"] = 0
        self._settings["grad_sample_index"] = self.grad_sample_spin.value()
        # Gradient settings
        self._settings["grad_loss_fn"] = self.grad_loss_cb.currentText()
        self._settings["grad_log_scale"] = self.grad_log_scale_chk.isChecked()
        self._settings["grad_colormap"] = self.grad_colormap_cb.currentText()
        return self._settings

