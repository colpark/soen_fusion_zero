"""Dynamic Initialization tab for the model creation GUI.

Provides flux-matching weight initialization and other dynamic initialization methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING

from PyQt6.QtCore import QDir, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import torch

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.model_manager import ModelManager

from ...utils.ui_helpers import create_double_spinbox, show_error, show_info, show_warning

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Worker thread for flux matching (keeps UI responsive)
# -----------------------------------------------------------------------------


class FluxMatchingWorker(QThread):
    """Background worker for running flux matching iterations."""

    progress = pyqtSignal(int, str)  # (iteration, message)
    iteration_complete = pyqtSignal(dict)  # iteration stats
    finished_signal = pyqtSignal(object)  # FluxMatchingResult
    error = pyqtSignal(str)

    def __init__(
        self,
        model,
        data_path: str,
        config_dict: dict,
        data_config: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._data_path = data_path
        self._config_dict = config_dict
        self._data_config = data_config
        self._should_stop = False

    def stop(self) -> None:
        self._should_stop = True

    def run(self) -> None:
        try:
            from soen_toolkit.utils.flux_matching_init import (
                FluxMatchingConfig,
                flux_matching_from_hdf5,
            )

            # Custom log function that emits signals
            def log_fn(msg: str) -> None:
                # Parse iteration from message if present
                if msg.startswith("Iteration"):
                    try:
                        parts = msg.split("/")
                        current = int(parts[0].split()[-1])
                        total = int(parts[1].split(":")[0])
                        pct = int((current / total) * 100)
                        self.progress.emit(pct, msg)
                    except (ValueError, IndexError):
                        self.progress.emit(-1, msg)
                else:
                    self.progress.emit(-1, msg)

            # Get excluded connections
            exclude_connections = self._config_dict.get("exclude_connections")
            if exclude_connections:
                log_fn(f"Excluding {len(exclude_connections)} connections: {exclude_connections}")
            else:
                log_fn("No connections excluded - updating all")

            config = FluxMatchingConfig(
                phi_total_target=self._config_dict.get("phi_total_target", 0.5),
                phi_total_target_min=self._config_dict.get("phi_total_target_min"),
                phi_total_target_max=self._config_dict.get("phi_total_target_max"),
                num_iterations=self._config_dict.get("num_iterations", 5),
                batch_size=self._config_dict.get("batch_size", 32),
                num_batches=self._config_dict.get("num_batches"),
                min_state_clamp=self._config_dict.get("min_state_clamp", 0.01),
                alpha=self._config_dict.get("alpha", 1.0),
                weight_update_mode=self._config_dict.get("weight_update_mode", "connection_wise"),
                verbose=True,
                log_fn=log_fn,
                stop_check=lambda: self._should_stop,
                exclude_connections=exclude_connections,
            )

            # Run flux matching
            result = flux_matching_from_hdf5(
                self._model,
                self._data_path,
                split=self._config_dict.get("split", "train"),
                config=config,
                seq_len=self._data_config.get("seq_len"),
                feature_min=self._data_config.get("feature_min"),
                feature_max=self._data_config.get("feature_max"),
            )

            self.finished_signal.emit(result)

        except Exception as e:
            log.exception("Flux matching failed")
            self.error.emit(str(e))


class CriticalityTuningWorker(QThread):
    """Background worker for criticality-based weight tuning."""

    progress = pyqtSignal(int, str)  # (percent, message)
    finished_signal = pyqtSignal(object)  # CriticalityInitResult
    error = pyqtSignal(str)

    def __init__(
        self,
        model,
        data_path: str,
        config_dict: dict,
        data_config: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._data_path = data_path
        self._config_dict = config_dict
        self._data_config = data_config
        self._should_stop = False

    def stop(self) -> None:
        self._should_stop = True

    def run(self) -> None:
        try:
            from soen_toolkit.utils.criticality_init import (
                CriticalityInitConfig,
                criticality_init_from_hdf5,
            )

            num_iterations = self._config_dict.get("num_iterations", 100)

            # Custom log function that emits signals
            def log_fn(msg: str) -> None:
                # Parse iteration from message
                if msg.startswith("Iteration"):
                    try:
                        import re
                        match = re.search(r"Iteration (\d+)/(\d+)", msg)
                        if match:
                            curr, tot = int(match.group(1)), int(match.group(2))
                            pct = int((curr / tot) * 100)
                            self.progress.emit(pct, msg)
                            return
                    except (ValueError, IndexError):
                        pass
                self.progress.emit(-1, msg)

            config = CriticalityInitConfig(
                target=self._config_dict.get("target", 1.0),
                br_target=self._config_dict.get("target", 1.0),
                enable_br=self._config_dict.get("enable_br", True),
                br_weight=self._config_dict.get("br_weight", 1.0),
                enable_lyap=self._config_dict.get("enable_lyap", False),
                lyap_target=self._config_dict.get("target_lyap", 0.0),
                lyap_weight=self._config_dict.get("lyap_weight", 0.0),
                lyap_eps=self._config_dict.get("lyap_eps", 1e-4),
                lyap_time_horizon=self._config_dict.get("lyap_time_horizon"),
                range_penalty_weight=self._config_dict.get("range_penalty_weight", 0.0),
                range_clip=self._config_dict.get("range_clip"),
                log_lyap=True,
                max_nan_tol=self._config_dict.get("max_nan_tol", 0),
                num_iterations=num_iterations,
                learning_rate=self._config_dict.get("learning_rate", 0.01),
                batch_size=self._config_dict.get("batch_size", 32),
                num_batches=self._config_dict.get("num_batches", 5),
                tolerance=self._config_dict.get("tolerance", 0.01),
                patience=self._config_dict.get("patience", 10),
                grad_clip=self._config_dict.get("grad_clip", 1.0),
                verbose=True,
                log_fn=log_fn,
                stop_check=lambda: self._should_stop,
                log_every=self._config_dict.get("log_every", 5),
            )

            # Pass data config to the loader
            result = criticality_init_from_hdf5(
                self._model,
                self._data_path,
                split=self._config_dict.get("split", "train"),
                config=config,
                seq_len=self._data_config.get("seq_len"),
                feature_min=self._data_config.get("feature_min"),
                feature_max=self._data_config.get("feature_max"),
            )

            self.finished_signal.emit(result)

        except Exception as e:
            log.exception("Criticality tuning failed")
            self.error.emit(str(e))


# -----------------------------------------------------------------------------
# Flux Matching Settings Widget
# -----------------------------------------------------------------------------


class FluxMatchingSettingsWidget(QWidget):
    """Widget for configuring flux matching parameters including layer selection."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Use a scroll area for many settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        # --- Global Settings ---
        global_group = QGroupBox("Target Flux")
        global_layout = QFormLayout(global_group)

        # Target phi
        self.phi_target_spin = create_double_spinbox(min_val=-100.0, max_val=100.0, decimals=4, step=0.05, default=0.5)
        self.phi_target_spin.setMinimumWidth(100)
        self.phi_target_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.phi_target_spin.setToolTip(
            "Target total flux (phi_total). 0.5 is the optimal operating point for the source function."
        )
        global_layout.addRow("Target phi_total:", self.phi_target_spin)

        # Symmetry breaking range
        self.use_range_cb = QCheckBox("Use symmetry breaking range")
        self.use_range_cb.setToolTip(
            "Enable to use a range of targets across nodes within each layer, breaking symmetry."
        )
        self.use_range_cb.toggled.connect(self._on_range_toggled)
        global_layout.addRow(self.use_range_cb)

        range_row = QHBoxLayout()
        self.phi_min_spin = create_double_spinbox(min_val=-100.0, max_val=100.0, decimals=4, step=0.05, default=0.4)
        self.phi_min_spin.setMinimumWidth(100)
        self.phi_min_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.phi_min_spin.setEnabled(False)

        self.phi_max_spin = create_double_spinbox(min_val=-100.0, max_val=100.0, decimals=4, step=0.05, default=0.6)
        self.phi_max_spin.setMinimumWidth(100)
        self.phi_max_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.phi_max_spin.setEnabled(False)

        range_row.addWidget(QLabel("Min:"))
        range_row.addWidget(self.phi_min_spin)
        range_row.addWidget(QLabel("Max:"))
        range_row.addWidget(self.phi_max_spin)
        global_layout.addRow("Range:", range_row)

        content_layout.addWidget(global_group)

        # --- Algorithm Settings ---
        algo_group = QGroupBox("Algorithm")
        algo_layout = QFormLayout(algo_group)

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 100000)
        self.iterations_spin.setValue(50)
        self.iterations_spin.setMinimumWidth(100)
        self.iterations_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.iterations_spin.setToolTip("Number of flux-matching iterations")
        algo_layout.addRow("Iterations:", self.iterations_spin)

        self.alpha_spin = create_double_spinbox(min_val=0.0, max_val=1000.0, decimals=4, step=0.05, default=0.2)
        self.alpha_spin.setMinimumWidth(100)
        self.alpha_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.alpha_spin.setToolTip(
            "Step size for weight updates. 1.0 = full correction, 0.2 = gradual convergence."
        )
        algo_layout.addRow("Step size (alpha):", self.alpha_spin)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["connection_wise", "node_wise"])
        self.mode_combo.setMinimumWidth(140)
        self.mode_combo.setToolTip(
            "connection_wise: individual weight per connection (finer control)\n"
            "node_wise: uniform weight per destination node (faster)"
        )
        algo_layout.addRow("Update mode:", self.mode_combo)

        self.min_clamp_spin = create_double_spinbox(min_val=0.0, max_val=100.0, decimals=6, step=0.01, default=0.01)
        self.min_clamp_spin.setMinimumWidth(100)
        self.min_clamp_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.min_clamp_spin.setToolTip("Minimum state clamp to avoid division by zero")
        algo_layout.addRow("Min state clamp:", self.min_clamp_spin)

        content_layout.addWidget(algo_group)

        # --- Data Settings ---
        data_group = QGroupBox("Data")
        data_layout = QFormLayout(data_group)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setValue(32)
        self.batch_size_spin.setMinimumWidth(100)
        self.batch_size_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.batch_size_spin.setToolTip("Batch size for forward passes")
        data_layout.addRow("Batch size:", self.batch_size_spin)

        self.num_batches_spin = QSpinBox()
        self.num_batches_spin.setRange(0, 1000)
        self.num_batches_spin.setValue(10)
        self.num_batches_spin.setSpecialValueText("All")
        self.num_batches_spin.setMinimumWidth(100)
        self.num_batches_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.num_batches_spin.setToolTip("Number of batches to use (0 = all)")
        data_layout.addRow("Max batches:", self.num_batches_spin)

        self.split_combo = QComboBox()
        self.split_combo.addItems(["train", "val", "test", "all_data"])
        self.split_combo.setMinimumWidth(100)
        self.split_combo.setToolTip("Dataset split to use from HDF5 file")
        data_layout.addRow("Data split:", self.split_combo)

        content_layout.addWidget(data_group)

        # --- Connection Selection ---
        conn_group = QGroupBox("Connection Selection")
        conn_layout = QVBoxLayout(conn_group)

        conn_desc = QLabel(
            "Uncheck connections to exclude them from flux matching and post-steps (noise / inhibitory flip)."
        )
        conn_desc.setWordWrap(True)
        conn_desc.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        conn_layout.addWidget(conn_desc)

        # Connection list with checkboxes
        self.conn_list = QListWidget()
        self.conn_list.setMaximumHeight(140)
        self.conn_list.setToolTip(
            "Check = update this connection\n"
            "Uncheck = skip this connection\n"
            "Frozen connections are marked [frozen]."
        )
        conn_layout.addWidget(self.conn_list)

        # Quick selection buttons
        conn_btn_row = QHBoxLayout()
        self.conn_select_all_btn = QPushButton("Select All")
        self.conn_select_all_btn.clicked.connect(self._conn_select_all)
        conn_btn_row.addWidget(self.conn_select_all_btn)
        self.conn_select_none_btn = QPushButton("Select None")
        self.conn_select_none_btn.clicked.connect(self._conn_select_none)
        conn_btn_row.addWidget(self.conn_select_none_btn)
        conn_btn_row.addStretch()
        conn_layout.addLayout(conn_btn_row)

        # Info label
        self.conn_info_label = QLabel("Load a model to see connections")
        self.conn_info_label.setStyleSheet("color: #888; font-size: 11px;")
        conn_layout.addWidget(self.conn_info_label)

        content_layout.addWidget(conn_group)

        # --- Noise Injection ---
        noise_group = QGroupBox("Weight Noise Injection")
        noise_layout = QVBoxLayout(noise_group)

        noise_desc = QLabel(
            "Perturb selected connection weights with Gaussian noise.\n"
            "Uses the connection selection above. Small std (0.001) recommended."
        )
        noise_desc.setWordWrap(True)
        noise_desc.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        noise_layout.addWidget(noise_desc)

        # Noise std control
        noise_param_row = QHBoxLayout()
        noise_param_row.addWidget(QLabel("Noise std:"))
        self.noise_std_spin = create_double_spinbox(min_val=0.0, max_val=100.0, decimals=6, step=0.0001, default=0.001)
        self.noise_std_spin.setMinimumWidth(100)
        self.noise_std_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.noise_std_spin.setToolTip("Standard deviation of Gaussian noise to add")
        noise_param_row.addWidget(self.noise_std_spin)
        noise_param_row.addStretch()
        noise_layout.addLayout(noise_param_row)

        # Inject button
        self.inject_noise_btn = QPushButton("Inject Noise")
        self.inject_noise_btn.setToolTip(
            "Add Gaussian noise to connections selected above.\n"
            "If connection selection is collapsed, all learnable connections are used."
        )
        # Note: actual click handler is set externally by the parent tab
        noise_layout.addWidget(self.inject_noise_btn)

        # Status label
        self.noise_status_label = QLabel("")
        self.noise_status_label.setStyleSheet("color: #888; font-size: 11px;")
        noise_layout.addWidget(self.noise_status_label)

        content_layout.addWidget(noise_group)

        # --- Inhibitory Fraction (Preserve Flux) ---
        inhib_group = QGroupBox("Inhibitory Fraction (Preserve Target Flux)")
        inhib_layout = QVBoxLayout(inhib_group)

        inhib_desc = QLabel(
            "Flip a fraction of active synapses to inhibitory (negative) and rescale the rest so each\n"
            "destination node keeps its target external flux. Uses the connection selection above.\n"
            "Requires a completed flux-matching run (needs mean upstream states)."
        )
        inhib_desc.setWordWrap(True)
        inhib_desc.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        inhib_layout.addWidget(inhib_desc)

        inhib_param_row = QHBoxLayout()
        inhib_param_row.addWidget(QLabel("Inhibitory fraction:"))
        self.inhib_frac_spin = create_double_spinbox(
            min_val=0.0, max_val=0.99, decimals=3, step=0.01, default=0.2
        )
        self.inhib_frac_spin.setMinimumWidth(100)
        self.inhib_frac_spin.setToolTip("Fraction of active synapses to flip negative (per node, per edge)")
        inhib_param_row.addWidget(self.inhib_frac_spin)

        inhib_param_row.addSpacing(12)
        inhib_param_row.addWidget(QLabel("Seed:"))
        self.inhib_seed_spin = QSpinBox()
        self.inhib_seed_spin.setRange(-1, 2**31 - 1)
        self.inhib_seed_spin.setValue(0)
        self.inhib_seed_spin.setToolTip("Random seed (-1 = random)")
        inhib_param_row.addWidget(self.inhib_seed_spin)

        inhib_param_row.addStretch()
        inhib_layout.addLayout(inhib_param_row)

        self.apply_inhib_btn = QPushButton("Apply Inhibitory Flip")
        self.apply_inhib_btn.setToolTip(
            "Flip a fraction of synapses to inhibitory for connections selected above.\n"
            "Preserves each node's target external flux (fail-fast if impossible)."
        )
        # Note: click handler is set externally by the parent tab
        inhib_layout.addWidget(self.apply_inhib_btn)

        self.inhib_status_label = QLabel("")
        self.inhib_status_label.setStyleSheet("color: #888; font-size: 11px;")
        inhib_layout.addWidget(self.inhib_status_label)

        content_layout.addWidget(inhib_group)
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _on_range_toggled(self, checked: bool) -> None:
        self.phi_min_spin.setEnabled(checked)
        self.phi_max_spin.setEnabled(checked)
        self.phi_target_spin.setEnabled(not checked)

    # -------------------------------------------------------------------------
    # Connection selection helpers (shared for flux matching and noise)
    # -------------------------------------------------------------------------

    def _conn_select_all(self) -> None:
        """Check all connections in the list."""
        self.conn_list.blockSignals(True)
        for i in range(self.conn_list.count()):
            item = self.conn_list.item(i)
            item.setCheckState(Qt.CheckState.Checked)
        self.conn_list.blockSignals(False)

    def _conn_select_none(self) -> None:
        """Uncheck all connections in the list."""
        self.conn_list.blockSignals(True)
        for i in range(self.conn_list.count()):
            item = self.conn_list.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)
        self.conn_list.blockSignals(False)

    def set_connections(self, model) -> None:
        """Populate the connection list from the model.

        Args:
            model: The SOEN model with connections
        """
        self.conn_list.clear()
        if model is None or not hasattr(model, "connections"):
            self.conn_info_label.setText("No model loaded")
            return

        for name, param in model.connections.items():
            label = self._friendly_conn_label(name, param)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, name)  # Store connection key
            # Store frozen status
            is_frozen = hasattr(param, "requires_grad") and not param.requires_grad
            item.setData(Qt.ItemDataRole.UserRole + 1, is_frozen)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Default: check all (learnable checked, frozen also checked but marked)
            item.setCheckState(Qt.CheckState.Checked)
            self.conn_list.addItem(item)

        count = self.conn_list.count()
        self.conn_info_label.setText(f"{count} connections available")

    def _friendly_conn_label(self, key: str, param) -> str:
        """Create a human-readable label for a connection."""
        frozen = hasattr(param, "requires_grad") and not param.requires_grad
        suffix = " [frozen]" if frozen else ""

        # Parse external connections: J_X_to_Y
        m = re.match(r"^J_(\d+)_to_(\d+)$", key)
        if m:
            a, b = m.group(1), m.group(2)
            return f"{a} -> {b} (external){suffix}"

        # Parse internal connections: internal_X
        m2 = re.match(r"^internal_(\d+)$", key)
        if m2:
            a = m2.group(1)
            return f"{a} (internal){suffix}"

        return key + suffix

    def get_selected_connections(self) -> list[str]:
        """Get list of connection keys selected (checked) for updates."""
        selected = []
        for i in range(self.conn_list.count()):
            item = self.conn_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                key = item.data(Qt.ItemDataRole.UserRole)
                if key:
                    selected.append(str(key))
        return selected

    def get_excluded_connections(self) -> set[str]:
        """Get set of connection keys excluded (unchecked) from updates."""
        excluded = set()
        for i in range(self.conn_list.count()):
            item = self.conn_list.item(i)
            if item.checkState() != Qt.CheckState.Checked:
                key = item.data(Qt.ItemDataRole.UserRole)
                if key:
                    excluded.add(str(key))
        return excluded

    def get_noise_std(self) -> float:
        """Get the noise standard deviation value."""
        return self.noise_std_spin.value()

    def set_noise_status(self, msg: str) -> None:
        """Update the noise status label."""
        self.noise_status_label.setText(msg)

    def get_inhibitory_fraction(self) -> float:
        """Get inhibitory fraction (0..1)."""
        return float(self.inhib_frac_spin.value())

    def get_inhibitory_seed(self) -> int | None:
        """Get seed for inhibitory flip (-1 means random / None)."""
        v = int(self.inhib_seed_spin.value())
        return None if v < 0 else v

    def set_inhibitory_status(self, msg: str) -> None:
        """Update inhibitory status label."""
        self.inhib_status_label.setText(msg)

    def get_config(self) -> dict:
        """Get the current configuration as a dictionary."""
        config = {
            "num_iterations": self.iterations_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "min_state_clamp": self.min_clamp_spin.value(),
            "alpha": self.alpha_spin.value(),
            "weight_update_mode": self.mode_combo.currentText(),
            "split": self.split_combo.currentText(),
        }

        if self.use_range_cb.isChecked():
            config["phi_total_target_min"] = self.phi_min_spin.value()
            config["phi_total_target_max"] = self.phi_max_spin.value()
        else:
            config["phi_total_target"] = self.phi_target_spin.value()

        num_batches = self.num_batches_spin.value()
        if num_batches > 0:
            config["num_batches"] = num_batches

        # Add connection exclusion
        excluded_connections = self.get_excluded_connections()
        if excluded_connections:
            config["exclude_connections"] = excluded_connections

        return config

    def save_settings(self, settings: QSettings) -> None:
        """Save settings to QSettings."""
        settings.setValue("flux/phi_target", self.phi_target_spin.value())
        settings.setValue("flux/use_range", self.use_range_cb.isChecked())
        settings.setValue("flux/phi_min", self.phi_min_spin.value())
        settings.setValue("flux/phi_max", self.phi_max_spin.value())
        settings.setValue("flux/iterations", self.iterations_spin.value())
        settings.setValue("flux/alpha", self.alpha_spin.value())
        settings.setValue("flux/mode", self.mode_combo.currentIndex())
        settings.setValue("flux/min_clamp", self.min_clamp_spin.value())
        settings.setValue("flux/batch_size", self.batch_size_spin.value())
        settings.setValue("flux/num_batches", self.num_batches_spin.value())
        settings.setValue("flux/split", self.split_combo.currentIndex())

    def load_settings(self, settings: QSettings) -> None:
        """Load settings from QSettings."""
        self.phi_target_spin.setValue(float(settings.value("flux/phi_target", 0.5)))
        self.use_range_cb.setChecked(settings.value("flux/use_range", False, type=bool))
        self.phi_min_spin.setValue(float(settings.value("flux/phi_min", 0.4)))
        self.phi_max_spin.setValue(float(settings.value("flux/phi_max", 0.6)))
        self.iterations_spin.setValue(int(settings.value("flux/iterations", 50)))
        self.alpha_spin.setValue(float(settings.value("flux/alpha", 0.2)))
        self.mode_combo.setCurrentIndex(int(settings.value("flux/mode", 0)))
        self.min_clamp_spin.setValue(float(settings.value("flux/min_clamp", 0.01)))
        self.batch_size_spin.setValue(int(settings.value("flux/batch_size", 32)))
        self.num_batches_spin.setValue(int(settings.value("flux/num_batches", 10)))
        self.split_combo.setCurrentIndex(int(settings.value("flux/split", 0)))


# -----------------------------------------------------------------------------
# Criticality Tuning Settings Widget
# -----------------------------------------------------------------------------


class CriticalitySettingsWidget(QWidget):
    """Widget for configuring criticality-based weight tuning parameters."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        # --- Description ---
        desc_label = QLabel(
            "Criticality Tuning uses backpropagation to optimize\n"
            "connection weights towards targets such as branching ratio\n"
            "and (optionally) a Lyapunov-like stability objective.\n\n"
            "Branching Ratio = 1.0 keeps the network at the edge of chaos;\n"
            "Lyapunov λ≈0 keeps trajectories sensitive but stable."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; padding: 8px;")
        content_layout.addWidget(desc_label)

        # --- Target Settings ---
        target_group = QGroupBox("Target")
        target_layout = QFormLayout(target_group)

        self.target_spin = create_double_spinbox(min_val=0.0, max_val=100.0, decimals=4, step=0.05, default=1.0)
        self.target_spin.setMinimumWidth(100)
        self.target_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.target_spin.setToolTip(
            "Target branching ratio.\n"
            "1.0 = critical (edge of chaos)\n"
            "< 1.0 = subcritical (activity decays)\n"
            "> 1.0 = supercritical (activity explodes)"
        )
        target_layout.addRow("Target BR:", self.target_spin)

        self.tolerance_spin = create_double_spinbox(min_val=0.0, max_val=10.0, decimals=4, step=0.005, default=0.01)
        self.tolerance_spin.setMinimumWidth(100)
        self.tolerance_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.tolerance_spin.setToolTip("Stop early if |BR - target| < tolerance")
        target_layout.addRow("Tolerance:", self.tolerance_spin)

        content_layout.addWidget(target_group)

        # --- Objectives ---
        obj_group = QGroupBox("Objectives")
        obj_layout = QFormLayout(obj_group)

        self.enable_br_cb = QCheckBox("Optimize Branching Ratio")
        self.enable_br_cb.setChecked(True)
        self.br_weight_spin = create_double_spinbox(min_val=0.0, max_val=1000.0, decimals=4, step=0.1, default=1.0)
        self.br_weight_spin.setEnabled(True)
        self.enable_br_cb.toggled.connect(self.br_weight_spin.setEnabled)
        obj_layout.addRow(self.enable_br_cb, self.br_weight_spin)

        self.enable_lyap_cb = QCheckBox("Optimize Lyapunov (λ≈0)")
        self.enable_lyap_cb.setChecked(False)
        self.lyap_weight_spin = create_double_spinbox(min_val=0.0, max_val=1000.0, decimals=4, step=0.1, default=0.0)
        self.lyap_weight_spin.setEnabled(False)
        self.enable_lyap_cb.toggled.connect(self.lyap_weight_spin.setEnabled)

        obj_layout.addRow(self.enable_lyap_cb, self.lyap_weight_spin)

        self.lyap_target_spin = create_double_spinbox(min_val=-5.0, max_val=5.0, decimals=5, step=0.01, default=0.0)
        self.lyap_target_spin.setEnabled(False)
        self.enable_lyap_cb.toggled.connect(self.lyap_target_spin.setEnabled)
        obj_layout.addRow("Lyap target:", self.lyap_target_spin)

        self.lyap_eps_spin = create_double_spinbox(min_val=1e-8, max_val=1.0, decimals=8, step=0.0001, default=0.0001)
        self.lyap_eps_spin.setEnabled(False)
        self.enable_lyap_cb.toggled.connect(self.lyap_eps_spin.setEnabled)
        obj_layout.addRow("Lyap eps:", self.lyap_eps_spin)

        self.range_weight_spin = create_double_spinbox(min_val=0.0, max_val=1000.0, decimals=4, step=0.1, default=0.0)
        self.range_clip_spin = create_double_spinbox(min_val=0.0, max_val=1000.0, decimals=4, step=0.1, default=0.0)
        obj_layout.addRow("Range penalty w:", self.range_weight_spin)
        obj_layout.addRow("Range clip (|mean|):", self.range_clip_spin)

        content_layout.addWidget(obj_group)

        # --- Optimization Settings ---
        opt_group = QGroupBox("Optimization")
        opt_layout = QFormLayout(opt_group)

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 1000)
        self.iterations_spin.setValue(100)
        self.iterations_spin.setMinimumWidth(100)
        self.iterations_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.iterations_spin.setToolTip("Number of optimization iterations")
        opt_layout.addRow("Iterations:", self.iterations_spin)

        self.lr_spin = create_double_spinbox(min_val=0.0, max_val=100.0, decimals=6, step=0.005, default=0.01)
        self.lr_spin.setMinimumWidth(100)
        self.lr_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.lr_spin.setToolTip("Learning rate for weight updates")
        opt_layout.addRow("Learning rate:", self.lr_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100000)
        self.patience_spin.setValue(10)
        self.patience_spin.setMinimumWidth(100)
        self.patience_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.patience_spin.setToolTip("Stop if no improvement for N iterations")
        opt_layout.addRow("Patience:", self.patience_spin)

        self.grad_clip_spin = create_double_spinbox(min_val=0.0, max_val=1000.0, decimals=3, step=0.5, default=1.0)
        self.grad_clip_spin.setMinimumWidth(100)
        self.grad_clip_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.grad_clip_spin.setToolTip("Gradient clipping magnitude (0 = no clipping)")
        opt_layout.addRow("Grad clip:", self.grad_clip_spin)

        content_layout.addWidget(opt_group)

        # --- Data Settings ---
        data_group = QGroupBox("Data Settings")
        data_layout = QFormLayout(data_group)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 100000)
        self.batch_size_spin.setValue(32)
        self.batch_size_spin.setMinimumWidth(100)
        self.batch_size_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.batch_size_spin.setToolTip("Batch size for forward passes")
        data_layout.addRow("Batch size:", self.batch_size_spin)

        self.num_batches_spin = QSpinBox()
        self.num_batches_spin.setRange(1, 100000)
        self.num_batches_spin.setValue(5)
        self.num_batches_spin.setMinimumWidth(100)
        self.num_batches_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.num_batches_spin.setToolTip("Number of batches per iteration")
        data_layout.addRow("Batches/iter:", self.num_batches_spin)

        self.split_combo = QComboBox()
        self.split_combo.addItems(["train", "val", "test", "all_data"])
        self.split_combo.setMinimumWidth(100)
        self.split_combo.setToolTip("Dataset split to use from HDF5 file")
        data_layout.addRow("Data split:", self.split_combo)

        self.log_every_spin = QSpinBox()
        self.log_every_spin.setRange(1, 100000)
        self.log_every_spin.setValue(5)
        self.log_every_spin.setMinimumWidth(100)
        self.log_every_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.log_every_spin.setToolTip("Log progress every N iterations")
        data_layout.addRow("Log every:", self.log_every_spin)

        content_layout.addWidget(data_group)
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def get_config(self) -> dict:
        """Get the current configuration as a dictionary."""
        return {
            "target": self.target_spin.value(),
            "enable_br": self.enable_br_cb.isChecked(),
            "br_weight": self.br_weight_spin.value(),
            "enable_lyap": self.enable_lyap_cb.isChecked(),
            "lyap_weight": self.lyap_weight_spin.value(),
            "target_lyap": self.lyap_target_spin.value(),
            "lyap_eps": self.lyap_eps_spin.value(),
            "range_penalty_weight": self.range_weight_spin.value(),
            "range_clip": self.range_clip_spin.value(),
            "tolerance": self.tolerance_spin.value(),
            "num_iterations": self.iterations_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "patience": self.patience_spin.value(),
            "grad_clip": self.grad_clip_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "num_batches": self.num_batches_spin.value(),
            "split": self.split_combo.currentText(),
            "log_every": self.log_every_spin.value(),
        }

    def save_settings(self, settings: QSettings) -> None:
        """Save settings to QSettings."""
        settings.setValue("crit/target", self.target_spin.value())
        settings.setValue("crit/enable_br", self.enable_br_cb.isChecked())
        settings.setValue("crit/br_weight", self.br_weight_spin.value())
        settings.setValue("crit/enable_lyap", self.enable_lyap_cb.isChecked())
        settings.setValue("crit/lyap_weight", self.lyap_weight_spin.value())
        settings.setValue("crit/target_lyap", self.lyap_target_spin.value())
        settings.setValue("crit/lyap_eps", self.lyap_eps_spin.value())
        settings.setValue("crit/range_penalty_weight", self.range_weight_spin.value())
        settings.setValue("crit/range_clip", self.range_clip_spin.value())
        settings.setValue("crit/tolerance", self.tolerance_spin.value())
        settings.setValue("crit/iterations", self.iterations_spin.value())
        settings.setValue("crit/lr", self.lr_spin.value())
        settings.setValue("crit/patience", self.patience_spin.value())
        settings.setValue("crit/grad_clip", self.grad_clip_spin.value())
        settings.setValue("crit/batch_size", self.batch_size_spin.value())
        settings.setValue("crit/num_batches", self.num_batches_spin.value())
        settings.setValue("crit/split", self.split_combo.currentIndex())
        settings.setValue("crit/log_every", self.log_every_spin.value())

    def load_settings(self, settings: QSettings) -> None:
        """Load settings from QSettings."""
        self.target_spin.setValue(float(settings.value("crit/target", 1.0)))
        self.enable_br_cb.setChecked(settings.value("crit/enable_br", True, type=bool))
        self.br_weight_spin.setValue(float(settings.value("crit/br_weight", 1.0)))
        self.enable_lyap_cb.setChecked(settings.value("crit/enable_lyap", False, type=bool))
        self.lyap_weight_spin.setValue(float(settings.value("crit/lyap_weight", 0.0)))
        self.lyap_target_spin.setValue(float(settings.value("crit/target_lyap", 0.0)))
        self.lyap_eps_spin.setValue(float(settings.value("crit/lyap_eps", 0.0001)))
        self.range_weight_spin.setValue(float(settings.value("crit/range_penalty_weight", 0.0)))
        self.range_clip_spin.setValue(float(settings.value("crit/range_clip", 0.0)))
        self.tolerance_spin.setValue(float(settings.value("crit/tolerance", 0.01)))
        self.iterations_spin.setValue(int(settings.value("crit/iterations", 100)))
        self.lr_spin.setValue(float(settings.value("crit/lr", 0.01)))
        self.patience_spin.setValue(int(settings.value("crit/patience", 10)))
        self.grad_clip_spin.setValue(float(settings.value("crit/grad_clip", 1.0)))
        self.batch_size_spin.setValue(int(settings.value("crit/batch_size", 32)))
        self.num_batches_spin.setValue(int(settings.value("crit/num_batches", 5)))
        self.split_combo.setCurrentIndex(int(settings.value("crit/split", 0)))
        self.log_every_spin.setValue(int(settings.value("crit/log_every", 5)))


# -----------------------------------------------------------------------------
# Main Dynamic Initialization Tab
# -----------------------------------------------------------------------------


class DynamicInitTab(QWidget):
    """Main tab for dynamic initialization methods.

    The active method is determined by which settings tab is currently selected.
    Settings and data paths are persisted across GUI sessions.
    """

    SETTINGS_KEY = "soen_toolkit/dynamic_init"

    def __init__(self, manager: ModelManager) -> None:
        super().__init__()
        self._mgr = manager
        self._worker: FluxMatchingWorker | CriticalityTuningWorker | None = None
        self._local_model = None  # Model loaded separately from main GUI
        self._last_flux_result = None
        self._init_ui()
        self._load_settings()

        # Connect to model changes
        try:
            self._mgr.model_changed.connect(self._on_model_changed)
        except Exception:
            pass

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Create splitter for left/right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # =====================================================================
        # LEFT PANEL: Model & Data
        # =====================================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # --- Model Selection ---
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)

        # Use GUI model checkbox
        self.use_gui_model_cb = QCheckBox("Use model from Model Builder tab")
        self.use_gui_model_cb.setChecked(True)
        self.use_gui_model_cb.toggled.connect(self._on_use_gui_model_toggled)
        model_layout.addWidget(self.use_gui_model_cb)

        # File selector row
        file_row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select .soen or .yaml model file...")
        self.model_path_edit.setEnabled(False)
        file_row.addWidget(self.model_path_edit, 1)

        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.setEnabled(False)
        self.browse_model_btn.clicked.connect(self._browse_model)
        file_row.addWidget(self.browse_model_btn)

        self.load_model_btn = QPushButton("Load")
        self.load_model_btn.setEnabled(False)
        self.load_model_btn.clicked.connect(self._load_model)
        file_row.addWidget(self.load_model_btn)
        model_layout.addLayout(file_row)

        # Model info
        self.model_info_label = QLabel("No model loaded")
        self.model_info_label.setStyleSheet("color: #888;")
        model_layout.addWidget(self.model_info_label)

        left_layout.addWidget(model_group)

        # --- Data Selection ---
        data_group = QGroupBox("Data")
        data_layout = QVBoxLayout(data_group)

        # File path row
        data_row = QHBoxLayout()
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setPlaceholderText("Select HDF5 data file...")
        data_row.addWidget(self.data_path_edit, 1)

        self.browse_data_btn = QPushButton("Browse...")
        self.browse_data_btn.clicked.connect(self._browse_data)
        data_row.addWidget(self.browse_data_btn)
        data_layout.addLayout(data_row)

        # Sequence length row
        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("Seq length:"))
        self.seq_len_spin = QSpinBox()
        self.seq_len_spin.setRange(1, 100000)
        self.seq_len_spin.setValue(100)
        self.seq_len_spin.setMinimumWidth(80)
        self.seq_len_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.seq_len_spin.setToolTip("Sequence length (number of timesteps)")
        self.seq_len_spin.valueChanged.connect(self._update_time_info)
        seq_row.addWidget(self.seq_len_spin)
        seq_row.addStretch()
        data_layout.addLayout(seq_row)

        # Feature scaling row
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Feature min:"))
        self.feature_min_edit = QLineEdit()
        self.feature_min_edit.setPlaceholderText("auto")
        self.feature_min_edit.setMaximumWidth(80)
        self.feature_min_edit.setToolTip("Min value for input scaling (auto = no scaling)")
        scale_row.addWidget(self.feature_min_edit)
        scale_row.addWidget(QLabel("max:"))
        self.feature_max_edit = QLineEdit()
        self.feature_max_edit.setPlaceholderText("auto")
        self.feature_max_edit.setMaximumWidth(80)
        self.feature_max_edit.setToolTip("Max value for input scaling (auto = no scaling)")
        scale_row.addWidget(self.feature_max_edit)
        scale_row.addStretch()
        data_layout.addLayout(scale_row)

        # Time mode row
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Time:"))
        self.time_mode_cb = QComboBox()
        self.time_mode_cb.addItems(["Specify dt", "Specify total time (ns)"])
        self.time_mode_cb.setMinimumWidth(140)
        self.time_mode_cb.currentTextChanged.connect(self._on_time_mode_changed)
        time_row.addWidget(self.time_mode_cb)

        self.dt_label = QLabel("dt:")
        time_row.addWidget(self.dt_label)
        self.dt_edit = QLineEdit()
        self.dt_edit.setText("37")
        self.dt_edit.setMaximumWidth(80)
        self.dt_edit.setToolTip("Dimensionless timestep (default 37 ≈ 1ns)")
        self.dt_edit.textChanged.connect(self._update_time_info)
        time_row.addWidget(self.dt_edit)

        self.total_time_label = QLabel("Total (ns):")
        self.total_time_label.setVisible(False)
        time_row.addWidget(self.total_time_label)
        self.total_time_edit = QLineEdit()
        self.total_time_edit.setMaximumWidth(80)
        self.total_time_edit.setVisible(False)
        self.total_time_edit.setToolTip("Total simulation time in nanoseconds")
        self.total_time_edit.textChanged.connect(self._update_time_info)
        time_row.addWidget(self.total_time_edit)

        time_row.addStretch()
        data_layout.addLayout(time_row)

        # Time info label
        self.time_info_label = QLabel()
        self.time_info_label.setStyleSheet("color: #666; font-size: 10px;")
        data_layout.addWidget(self.time_info_label)
        self._update_time_info()

        left_layout.addWidget(data_group)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # =====================================================================
        # MIDDLE PANEL: Method Settings (tab selection = method)
        # =====================================================================
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)

        # Info label
        method_info = QLabel("Select a tab below to choose the initialization method:")
        method_info.setStyleSheet("color: #666; font-size: 11px; padding: 4px;")
        middle_layout.addWidget(method_info)

        # Settings tabs - tab selection determines the method to run
        self.settings_tabs = QTabWidget()
        self.flux_settings = FluxMatchingSettingsWidget()
        self.criticality_settings = CriticalitySettingsWidget()
        self.settings_tabs.addTab(self.flux_settings, "Flux Matching")
        self.settings_tabs.addTab(self.criticality_settings, "Criticality Tuning")
        middle_layout.addWidget(self.settings_tabs)

        # Connect noise injection button
        self.flux_settings.inject_noise_btn.clicked.connect(self._inject_noise)
        # Connect inhibitory flip button
        self.flux_settings.apply_inhib_btn.clicked.connect(self._apply_inhibitory_flip)

        splitter.addWidget(middle_widget)

        # =====================================================================
        # RIGHT PANEL: Output & Controls
        # =====================================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        self.log_output.setPlaceholderText("Initialization log will appear here...")
        progress_layout.addWidget(self.log_output)

        right_layout.addWidget(progress_group)

        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Initialization")
        self.run_btn.setStyleSheet("font-weight: bold;")
        self.run_btn.clicked.connect(self._run_initialization)
        btn_row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_initialization)
        btn_row.addWidget(self.stop_btn)
        action_layout.addLayout(btn_row)

        # Save row
        save_row = QHBoxLayout()
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setPlaceholderText("Output path for .soen file...")
        save_row.addWidget(self.save_path_edit, 1)

        self.browse_save_btn = QPushButton("Browse...")
        self.browse_save_btn.clicked.connect(self._browse_save)
        save_row.addWidget(self.browse_save_btn)

        self.save_btn = QPushButton("Save Model")
        self.save_btn.clicked.connect(self._save_model)
        save_row.addWidget(self.save_btn)
        action_layout.addLayout(save_row)

        # Apply to GUI button
        self.apply_to_gui_btn = QPushButton("Apply to Model Builder")
        self.apply_to_gui_btn.setToolTip("Update the Model Builder tab with the initialized model")
        self.apply_to_gui_btn.clicked.connect(self._apply_to_gui)
        action_layout.addWidget(self.apply_to_gui_btn)

        right_layout.addWidget(action_group)
        right_layout.addStretch()

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([300, 350, 350])

        main_layout.addWidget(splitter)

        # Initial state update
        self._update_model_info()

    # -------------------------------------------------------------------------
    # Model Handling
    # -------------------------------------------------------------------------

    def _on_use_gui_model_toggled(self, checked: bool) -> None:
        self.model_path_edit.setEnabled(not checked)
        self.browse_model_btn.setEnabled(not checked)
        self.load_model_btn.setEnabled(not checked)
        self._update_model_info()

    def _browse_model(self) -> None:
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            QDir.homePath(),
            "SOEN Models (*.soen *.pth *.yaml *.yml);;All files (*)",
        )
        if fname:
            self.model_path_edit.setText(fname)

    def _load_model(self) -> None:
        path = self.model_path_edit.text().strip()
        if not path:
            show_warning(self, "Load Model", "Please select a model file.")
            return

        try:
            from soen_toolkit.core import SOENModelCore

            p = Path(path)
            if p.suffix.lower() in (".yaml", ".yml"):
                self._local_model = SOENModelCore.build(str(p))
            else:
                self._local_model = SOENModelCore.load(str(p), show_logs=False)

            self._update_model_info()
            self._log(f"Loaded model from: {path}")
            show_info(self, "Model Loaded", f"Model loaded from:\n{path}")

        except Exception as e:
            log.exception("Failed to load model")
            show_error(self, "Load Failed", f"Failed to load model:\n{e}")

    def _get_active_model(self):
        """Get the model to use (GUI model or locally loaded)."""
        if self.use_gui_model_cb.isChecked():
            return self._mgr.model
        return self._local_model

    def _update_model_info(self) -> None:
        model = self._get_active_model()
        if model is None:
            self.model_info_label.setText("No model loaded")
            self.flux_settings.set_connections(None)
            return

        try:
            num_layers = len(model.layers_config)
            num_conns = len(model.connections_config)
            num_params = sum(p.numel() for p in model.parameters())

            self.model_info_label.setText(
                f"{num_layers} layers, {num_conns} connections, {num_params:,} params"
            )
            # Update connection list for flux matching and noise injection
            self.flux_settings.set_connections(model)

        except Exception as e:
            self.model_info_label.setText(f"Error: {e}")

    def _on_model_changed(self) -> None:
        """Handle model changes from the Model Builder tab."""
        if self.use_gui_model_cb.isChecked():
            self._update_model_info()

    # -------------------------------------------------------------------------
    # Data Handling
    # -------------------------------------------------------------------------

    def _browse_data(self) -> None:
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 Data File",
            QDir.homePath(),
            "HDF5 Files (*.h5 *.hdf5);;All files (*)",
        )
        if fname:
            self.data_path_edit.setText(fname)

    def _on_time_mode_changed(self, text: str) -> None:
        """Toggle between dt and total time mode."""
        use_total_time = "total" in text.lower()
        self.dt_label.setVisible(not use_total_time)
        self.dt_edit.setVisible(not use_total_time)
        self.total_time_label.setVisible(use_total_time)
        self.total_time_edit.setVisible(use_total_time)
        self._update_time_info()

    def _update_time_info(self) -> None:
        """Update the time info label based on current settings."""
        from soen_toolkit.physics.constants import get_omega_c

        omega_c = get_omega_c()
        seq_len = self.seq_len_spin.value()

        try:
            if self.time_mode_cb.currentText() == "Specify dt":
                dt_text = self.dt_edit.text().strip()
                if dt_text:
                    dt = float(dt_text)
                    total_dim = dt * seq_len
                    total_ns = (total_dim / omega_c) * 1e9
                    self.time_info_label.setText(
                        f"Total: {total_ns:.3f} ns ({total_dim:.0f} dimensionless)"
                    )
                else:
                    self.time_info_label.setText("")
            else:
                total_ns_text = self.total_time_edit.text().strip()
                if total_ns_text:
                    total_ns = float(total_ns_text)
                    total_dim = (total_ns / 1e9) * omega_c
                    dt = total_dim / seq_len if seq_len > 0 else 0
                    self.time_info_label.setText(
                        f"dt = {dt:.2f} (dimensionless)"
                    )
                else:
                    self.time_info_label.setText("")
        except ValueError:
            self.time_info_label.setText("")

    def _get_data_config(self) -> dict:
        """Get data configuration from UI controls."""
        from soen_toolkit.physics.constants import get_omega_c

        omega_c = get_omega_c()

        config = {
            "seq_len": self.seq_len_spin.value(),
        }

        # Parse feature scaling
        min_text = self.feature_min_edit.text().strip()
        max_text = self.feature_max_edit.text().strip()
        if min_text and min_text.lower() != "auto":
            try:
                config["feature_min"] = float(min_text)
            except ValueError:
                pass
        if max_text and max_text.lower() != "auto":
            try:
                config["feature_max"] = float(max_text)
            except ValueError:
                pass

        # Parse dt
        if self.time_mode_cb.currentText() == "Specify dt":
            dt_text = self.dt_edit.text().strip()
            if dt_text:
                try:
                    config["dt"] = float(dt_text)
                except ValueError:
                    config["dt"] = 37.0
            else:
                config["dt"] = 37.0
        else:
            # Convert total time to dt
            total_ns_text = self.total_time_edit.text().strip()
            if total_ns_text:
                try:
                    total_ns = float(total_ns_text)
                    seq_len = config["seq_len"]
                    total_dim = (total_ns / 1e9) * omega_c
                    config["dt"] = total_dim / seq_len if seq_len > 0 else 37.0
                except ValueError:
                    config["dt"] = 37.0
            else:
                config["dt"] = 37.0

        return config

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _run_initialization(self) -> None:
        model = self._get_active_model()
        if model is None:
            show_warning(self, "No Model", "Please load or build a model first.")
            return

        data_path = self.data_path_edit.text().strip()
        if not data_path:
            show_warning(self, "No Data", "Please select an HDF5 data file.")
            return

        if not Path(data_path).exists():
            show_warning(self, "File Not Found", f"Data file not found:\n{data_path}")
            return

        # Clear log and reset progress
        self.log_output.clear()
        self.progress_bar.setValue(0)

        # Disable controls
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Save settings before running
        self._save_settings()

        # Use the currently selected tab to determine method
        method_idx = self.settings_tabs.currentIndex()

        if method_idx == 0:
            # Flux Matching
            self._run_flux_matching(model, data_path)
        else:
            # Criticality Tuning
            self._run_criticality_tuning(model, data_path)

    def _run_flux_matching(self, model, data_path: str) -> None:
        """Run flux matching initialization."""
        # Config now includes layer selection from flux_settings widget
        config_dict = self.flux_settings.get_config()
        data_config = self._get_data_config()

        self._log("Starting flux matching initialization...")
        self._log(f"  Seq length: {data_config['seq_len']}, dt: {data_config['dt']:.2f}")

        # Start worker
        self._worker = FluxMatchingWorker(model, data_path, config_dict, data_config, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_flux_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _run_criticality_tuning(self, model, data_path: str) -> None:
        """Run criticality-based weight tuning."""
        config_dict = self.criticality_settings.get_config()
        data_config = self._get_data_config()

        self._log("Starting criticality tuning...")
        self._log(f"  Target BR: {config_dict['target']}")
        self._log(
            f"  Objectives -> BR(enabled={config_dict['enable_br']}, weight={config_dict['br_weight']}), "
            f"Lyap(enabled={config_dict['enable_lyap']}, weight={config_dict['lyap_weight']}, target={config_dict['target_lyap']})"
        )
        self._log(f"  Seq length: {data_config['seq_len']}, dt: {data_config['dt']:.2f}")

        # Start worker
        self._worker = CriticalityTuningWorker(model, data_path, config_dict, data_config, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_criticality_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop_initialization(self) -> None:
        if self._worker:
            self._worker.stop()
            self._log("Stopping...")

    def _on_progress(self, pct: int, msg: str) -> None:
        if pct >= 0:
            self.progress_bar.setValue(pct)
        self._log(msg)

    def _on_flux_finished(self, result) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        if result.converged:
            self._log("Flux matching CONVERGED!")
            show_info(
                self,
                "Initialization Complete",
                "Flux matching converged to target values.\n\n"
                "You can now save the model or apply it to the Model Builder.",
            )
        else:
            self._log("Flux matching completed (did not fully converge)")
            show_info(
                self,
                "Initialization Complete",
                "Flux matching completed but did not fully converge.\n"
                "Consider running more iterations or adjusting parameters.",
            )

        self._worker = None
        self._last_flux_result = result

    def _on_criticality_finished(self, result) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        self._log(f"Final branching ratio: {result.final_branching_ratio:.4f}")
        self._log(f"Best branching ratio: {result.best_branching_ratio:.4f} (iteration {result.best_iteration + 1})")
        if result.final_lyapunov is not None:
            self._log(f"Final Lyapunov: {result.final_lyapunov:.4f}")
        if result.best_lyapunov is not None:
            self._log(f"Best Lyapunov: {result.best_lyapunov:.4f}")

        if result.converged:
            self._log("Criticality tuning CONVERGED!")
            show_info(
                self,
                "Initialization Complete",
                f"Criticality tuning converged!\n\n"
                f"Final BR: {result.final_branching_ratio:.4f}\n"
                f"Target BR: {self.criticality_settings.target_spin.value()}\n"
                + (
                    f"Final Lyap: {result.final_lyapunov:.4f}\n"
                    if result.final_lyapunov is not None
                    else ""
                )
                + "You can now save the model or apply it to the Model Builder.",
            )
        else:
            self._log("Criticality tuning completed (did not fully converge)")
            show_info(
                self,
                "Initialization Complete",
                f"Criticality tuning completed.\n\n"
                f"Final BR: {result.final_branching_ratio:.4f}\n"
                f"Best BR: {result.best_branching_ratio:.4f}\n"
                + (
                    f"Final Lyap: {result.final_lyapunov:.4f}\n"
                    if result.final_lyapunov is not None
                    else ""
                )
                + (
                    f"Best Lyap: {result.best_lyapunov:.4f}\n\n"
                    if result.best_lyapunov is not None
                    else "\n"
                )
                + "Consider more iterations or adjusting learning rate.",
            )

        self._worker = None

    def _on_error(self, error_msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._log(f"ERROR: {error_msg}")
        show_error(self, "Initialization Failed", error_msg)
        self._worker = None

    # -------------------------------------------------------------------------
    # Saving
    # -------------------------------------------------------------------------

    def _browse_save(self) -> None:
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            QDir.homePath(),
            "SOEN Model (*.soen);;PyTorch Model (*.pth)",
        )
        if fname:
            p = Path(fname)
            if p.suffix.lower() not in (".soen", ".pth"):
                fname = str(p.with_suffix(".soen"))
            self.save_path_edit.setText(fname)

    def _save_model(self) -> None:
        model = self._get_active_model()
        if model is None:
            show_warning(self, "No Model", "No model to save.")
            return

        path = self.save_path_edit.text().strip()
        if not path:
            show_warning(self, "No Path", "Please specify a save path.")
            return

        try:
            p = Path(path)
            if p.suffix.lower() not in (".soen", ".pth"):
                p = p.with_suffix(".soen")

            model.save(str(p))
            self._log(f"Model saved to: {p}")
            show_info(self, "Saved", f"Model saved to:\n{p}")

        except Exception as e:
            log.exception("Failed to save model")
            show_error(self, "Save Failed", f"Failed to save model:\n{e}")

    def _apply_to_gui(self) -> None:
        """Apply the locally loaded/initialized model to the Model Builder."""
        model = self._get_active_model()
        if model is None:
            show_warning(self, "No Model", "No model to apply.")
            return

        # If using GUI model, it's already there
        if self.use_gui_model_cb.isChecked():
            show_info(
                self,
                "Already Applied",
                "The model from Model Builder is already being used.\n"
                "Changes have been applied in-place.",
            )
            # Emit signal to refresh other tabs
            self._mgr.model_changed.emit()
            return

        # Apply local model to manager
        try:
            self._mgr.model = self._local_model
            self._mgr.sim_config = self._local_model.sim_config
            self._mgr.layers = self._local_model.layers_config
            self._mgr.connections = self._local_model.connections_config
            self._mgr.model_changed.emit()

            self._log("Model applied to Model Builder")
            show_info(
                self,
                "Applied",
                "Model has been applied to the Model Builder tab.",
            )

        except Exception as e:
            log.exception("Failed to apply model")
            show_error(self, "Apply Failed", f"Failed to apply model:\n{e}")

    # -------------------------------------------------------------------------
    # Noise Injection
    # -------------------------------------------------------------------------

    def _inject_noise(self) -> None:
        """Inject Gaussian noise into selected connection weights."""
        model = self._get_active_model()
        if model is None:
            show_warning(self, "No Model", "Please load or build a model first.")
            return

        selected_keys = self.flux_settings.get_selected_connections()
        if not selected_keys:
            show_warning(
                self,
                "No Connections Selected",
                "Please select at least one connection to perturb.\n"
                "Expand the Connection Selection section and check some connections.",
            )
            return

        noise_std = self.flux_settings.get_noise_std()
        if noise_std <= 0:
            show_warning(
                self,
                "Invalid Noise Std",
                "Noise standard deviation must be positive.",
            )
            return

        # Perform noise injection
        perturbed_count = 0
        total_params_perturbed = 0

        with torch.no_grad():
            for key in selected_keys:
                if key not in model.connections:
                    continue

                param = model.connections[key]

                # Skip frozen weights (respect requires_grad=False)
                if not param.requires_grad:
                    self._log(f"  Skipping {key} (frozen)")
                    continue

                # Get mask if available to only perturb actual connections
                mask = model.connection_masks.get(key)

                # Generate noise
                noise = torch.randn_like(param) * noise_std

                # Apply mask: only add noise where connections exist
                if mask is not None:
                    noise = noise * mask

                # Add noise to weights
                param.add_(noise)

                # Count non-zero perturbations
                if mask is not None:
                    num_perturbed = int(mask.sum().item())
                else:
                    num_perturbed = param.numel()

                perturbed_count += 1
                total_params_perturbed += num_perturbed
                self._log(f"  Perturbed {key}: {num_perturbed} weights")

        # Update status
        status_msg = f"Injected noise (std={noise_std:.6f}) into {perturbed_count} connections, {total_params_perturbed:,} weights"
        self.flux_settings.set_noise_status(status_msg)
        self._log(status_msg)

        # Notify model changed
        self._mgr.model_changed.emit()

        show_info(
            self,
            "Noise Injected",
            f"Added Gaussian noise (std={noise_std:.6f}) to:\n"
            f"  {perturbed_count} connections\n"
            f"  {total_params_perturbed:,} total weights",
        )

    # -------------------------------------------------------------------------
    # Inhibitory Fraction (Preserve Flux)
    # -------------------------------------------------------------------------

    def _apply_inhibitory_flip(self) -> None:
        """Flip a fraction of synapses inhibitory while preserving target flux (post flux-matching)."""
        model = self._get_active_model()
        if model is None:
            show_warning(self, "No Model", "Please load or build a model first.")
            return

        if self._last_flux_result is None or getattr(self._last_flux_result, "final_mean_states", None) is None:
            show_warning(
                self,
                "Run Flux Matching First",
                "This step requires the mean upstream states produced by a flux-matching run.\n\n"
                "Run flux matching first, then apply inhibitory flip.",
            )
            return

        frac = self.flux_settings.get_inhibitory_fraction()
        seed = self.flux_settings.get_inhibitory_seed()

        if frac <= 0:
            show_warning(self, "Invalid Fraction", "Inhibitory fraction must be > 0.")
            return

        selected_keys = self.flux_settings.get_selected_connections()
        if not selected_keys:
            show_warning(self, "No Connections Selected", "Please select at least one connection.")
            return

        # Skip frozen weights (respect requires_grad=False)
        selected_keys = [
            k
            for k in selected_keys
            if k in model.connections and getattr(model.connections[k], "requires_grad", True)
        ]
        if not selected_keys:
            show_warning(self, "No Learnable Connections", "All selected connections are frozen or missing.")
            return

        try:
            from soen_toolkit.utils.flux_matching_init import (
                FluxMatchingConfig,
                apply_inhibitory_fraction_preserving_flux,
            )

            config_dict = self.flux_settings.get_config()
            config = FluxMatchingConfig(
                phi_total_target=config_dict.get("phi_total_target", 0.5),
                phi_total_target_min=config_dict.get("phi_total_target_min"),
                phi_total_target_max=config_dict.get("phi_total_target_max"),
                num_iterations=config_dict.get("num_iterations", 5),
                batch_size=config_dict.get("batch_size", 32),
                num_batches=config_dict.get("num_batches"),
                min_state_clamp=config_dict.get("min_state_clamp", 0.01),
                alpha=config_dict.get("alpha", 1.0),
                weight_update_mode=config_dict.get("weight_update_mode", "connection_wise"),
                exclude_connections=config_dict.get("exclude_connections"),
            )

            apply_inhibitory_fraction_preserving_flux(
                model,
                self._last_flux_result.final_mean_states,
                config=config,
                inhibitory_fraction=frac,
                include_connections=set(selected_keys),
                seed=seed,
            )

            status_msg = (
                f"Applied inhibitory flip (p={frac:.2f}) to {len(selected_keys)} connections "
                f"(seed={'random' if seed is None else seed})"
            )
            self.flux_settings.set_inhibitory_status(status_msg)
            self._log(status_msg)
            self._mgr.model_changed.emit()

            show_info(
                self,
                "Inhibitory Flip Applied",
                f"Applied inhibitory fraction p={frac:.2f} to:\n"
                f"  {len(selected_keys)} connections\n\n"
                "This preserves each node's target external flux (fail-fast if impossible).",
            )

        except Exception as e:
            log.exception("Failed to apply inhibitory flip")
            show_error(self, "Inhibitory Flip Failed", str(e))

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.log_output.append(msg)
        # Scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # -------------------------------------------------------------------------
    # Settings Persistence
    # -------------------------------------------------------------------------

    def _save_settings(self) -> None:
        """Save all settings to QSettings for persistence."""
        settings = QSettings("GreatSky", self.SETTINGS_KEY)

        # Data path
        settings.setValue("data_path", self.data_path_edit.text())

        # Data settings
        settings.setValue("seq_len", self.seq_len_spin.value())
        settings.setValue("feature_min", self.feature_min_edit.text())
        settings.setValue("feature_max", self.feature_max_edit.text())
        settings.setValue("time_mode", self.time_mode_cb.currentIndex())
        settings.setValue("dt", self.dt_edit.text())
        settings.setValue("total_time_ns", self.total_time_edit.text())

        # Model path
        settings.setValue("model_path", self.model_path_edit.text())
        settings.setValue("use_gui_model", self.use_gui_model_cb.isChecked())

        # Save path
        settings.setValue("save_path", self.save_path_edit.text())

        # Current tab
        settings.setValue("current_tab", self.settings_tabs.currentIndex())

        # Delegate to settings widgets
        self.flux_settings.save_settings(settings)
        self.criticality_settings.save_settings(settings)

    def _load_settings(self) -> None:
        """Load saved settings from QSettings."""
        settings = QSettings("GreatSky", self.SETTINGS_KEY)

        # Data path
        data_path = settings.value("data_path", "")
        if data_path:
            self.data_path_edit.setText(data_path)

        # Data settings
        self.seq_len_spin.setValue(int(settings.value("seq_len", 100)))
        self.feature_min_edit.setText(settings.value("feature_min", ""))
        self.feature_max_edit.setText(settings.value("feature_max", ""))
        self.time_mode_cb.setCurrentIndex(int(settings.value("time_mode", 0)))
        self.dt_edit.setText(settings.value("dt", "37"))
        self.total_time_edit.setText(settings.value("total_time_ns", ""))
        # Update visibility based on time mode
        self._on_time_mode_changed(self.time_mode_cb.currentText())

        # Model path
        model_path = settings.value("model_path", "")
        if model_path:
            self.model_path_edit.setText(model_path)
        use_gui = settings.value("use_gui_model", True, type=bool)
        self.use_gui_model_cb.setChecked(use_gui)

        # Save path
        save_path = settings.value("save_path", "")
        if save_path:
            self.save_path_edit.setText(save_path)

        # Current tab
        tab_idx = int(settings.value("current_tab", 0))
        self.settings_tabs.setCurrentIndex(tab_idx)

        # Delegate to settings widgets
        self.flux_settings.load_settings(settings)
        self.criticality_settings.load_settings(settings)

    def closeEvent(self, event) -> None:
        """Save settings when tab is closed."""
        self._save_settings()
        super().closeEvent(event)

