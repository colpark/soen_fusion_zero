# src/soen_toolkit/model_creation_gui/components/state_trajectory_dialog.py
from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path
import random
import time
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg
import torch

from soen_toolkit.physics.constants import (
    dt_nanoseconds_per_step,
    get_omega_c,
)
from soen_toolkit.utils.paths import GSC_PADDED_ZEROS_DATA_FILEPATH as DATA_FILEPATH
from soen_toolkit.utils.power_tracking import (
    convert_energy_to_physical,
    convert_power_to_physical,
)

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.model_manager import ModelManager

from ..utils.constants import FLOAT_MAX
from ..utils.ui_helpers import create_double_spinbox, create_float_validator, show_error, show_info, show_warning

# JAX import moved to methods that use it for lazy loading
_JAX_AVAILABLE = False


def _check_jax_available() -> bool:
    """Check if JAX and required dependencies are available without full initialization."""
    try:
        import jax  # noqa: F401

        from soen_toolkit.utils.port_to_jax import (  # noqa: F401
            convert_core_model_to_jax,
        )

        return True
    except Exception:
        return False


# Check JAX availability at module load time
if _check_jax_available():
    _JAX_AVAILABLE = True


def _format_sim_time(elapsed: float) -> str:
    """Convert elapsed seconds to a human-friendly string with appropriate units."""
    if elapsed >= 1.0:
        return f"{elapsed:.3f} s"
    ms = elapsed * 1e3
    if ms >= 1.0:
        return f"{ms:.3f} ms"
    us = elapsed * 1e6
    if us >= 1.0:
        return f"{us:.3f} µs"
    ns = elapsed * 1e9
    return f"{ns:.3f} ns"


# Define constants for float ranges and precision
DT_MIN = 0.0  # dt should be non-negative (specific to this dialog)

# Centralized omega_c access is provided by soen_toolkit.physics.constants


class StateTrajectoryDialog(QDialog):
    """Dialog for plotting state trajectories s(t) of all nodes per layer for a chosen sample.

    Controls:
      - Digit class selector
      - Sample index spinbox
      - "New Random Sample" button
      - Sequence length spinbox
      - "Plot Trajectory" button

    Displays a tab per layer, each plotting all neuron trajectories.
    """

    def __init__(self, parent=None, manager: ModelManager = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("State Trajectories")
        # Persistent settings store
        self._settings = QSettings("GreatSky", "SOEN-Toolkit")
        # Dataset path defaults to empty; will load from settings if present
        self.dataset_filepath = ""
        self._mgr = manager
        # Dataset management
        self.dataset = None
        self.available_groups = []
        self.current_group = None
        # Ensure labels/indices always exist to avoid attribute errors before dataset load
        self.labels = []
        self.indices = []
        # Lazy loading: track whether dataset has been loaded
        self._dataset_loaded = False
        # Saved UI state placeholders
        self._saved_group_name = None
        self._saved_metric = None
        self._saved_digit = None
        self._saved_sample = None
        self._saved_seq_len = None
        self._saved_input_src = None
        self._saved_mel_min = None
        self._saved_mel_max = None
        self._saved_encoding = None
        self._saved_vocab_size = None
        self._saved_dt = None
        self._saved_show_legend = None
        self._saved_show_total = None
        self._saved_include_s0 = None
        # New: append zeros settings
        self._saved_append_zeros_enabled = None
        self._saved_append_zeros_count = None
        self._saved_append_mode = None
        # New: prepend zeros settings
        self._saved_prepend_zeros_enabled = None
        self._saved_prepend_zeros_count = None
        # New: time-based zeros (ns)
        self._saved_append_zeros_ns = None
        self._saved_prepend_zeros_ns = None
        # Saved input generator params
        self._saved_input_noise_std = None
        self._saved_input_colored_beta = None
        self._saved_input_sine_freq_mhz = None
        self._saved_input_sine_amp = None
        self._saved_input_sine_phase_deg = None
        self._saved_input_sine_offset = None
        # Task type settings
        self._saved_task_type = None
        self._is_seq2seq = False
        # JAX conversion cache (per-session)
        self._jax_model = None
        self._jax_apply = None
        self._jax_source_model_id = None
        self._jax_warm_shapes = set()
        self._jax_last_shape_key = None

        # Load settings
        self._load_settings()
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ------------------------------------------------------------------
        #  Two-tab widget: "Plots" and "Settings"
        #  (Metric selector and Refresh button now live on the Plots tab)
        # ------------------------------------------------------------------
        main_tabs = QTabWidget()
        layout.addWidget(main_tabs, 1)

        # -------- PLOTS TAB --------
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)

        # ---- Top controls (Metric selector & Refresh) ----
        plot_ctrl = QHBoxLayout()
        plot_ctrl.addWidget(QLabel("Metric:"))
        # Create metric combo once and reuse everywhere
        self.metric_cb = QComboBox()
        self.metric_cb.addItems(
            [
                "State (s)",
                "Flux (phi)",
                "Non-Linearity (g)",
                "Power",
                "Energy",
            ]
        )
        plot_ctrl.addWidget(self.metric_cb)

        # Backend selection
        plot_ctrl.addWidget(QLabel("Backend:"))
        self.backend_cb = QComboBox()
        self.backend_cb.addItems(["Torch", "JAX"])
        if not _JAX_AVAILABLE:
            self.backend_cb.setCurrentText("Torch")
            self.backend_cb.setEnabled(False)
            self.backend_cb.setToolTip("JAX not available in this environment")
        plot_ctrl.addWidget(self.backend_cb)
        # Reset JAX cache when backend toggled
        with contextlib.suppress(Exception):
            self.backend_cb.currentTextChanged.connect(self._on_backend_changed)

        self.plot_btn = QPushButton("Refresh Plot")
        self.plot_btn.clicked.connect(self._on_plot)
        plot_ctrl.addWidget(self.plot_btn)
        plot_ctrl.addStretch(1)
        plot_layout.addLayout(plot_ctrl)

        # Layer tabs widget follows the controls
        self.tabs = pg.QtWidgets.QTabWidget()
        plot_layout.addWidget(self.tabs, 1)

        # -------- SETTINGS TAB --------
        settings_tab = QWidget()
        settings_vbox = QVBoxLayout(settings_tab)

        # --- Dataset file selector ---
        self.dataset_group = QGroupBox("Dataset")
        ds_vbox = QVBoxLayout()

        # File path row
        ds_layout = QHBoxLayout()
        ds_layout.addWidget(QLabel("Dataset File:"))
        self.dataset_edit = QLineEdit(self.dataset_filepath)
        self.dataset_edit.setToolTip("Path to HDF5 dataset file")
        ds_layout.addWidget(self.dataset_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dataset)
        ds_layout.addWidget(browse_btn)
        ds_vbox.addLayout(ds_layout)

        # Task type row
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Task Type:"))
        self.task_type_cb = QComboBox()
        self.task_type_cb.addItems(["Classification", "Seq2Seq (regression)"])
        self.task_type_cb.setToolTip("Choose how to treat labels: class IDs or time-series targets")
        self.task_type_cb.currentIndexChanged.connect(self._on_task_type_changed)
        task_layout.addWidget(self.task_type_cb)
        task_layout.addStretch()
        ds_vbox.addLayout(task_layout)

        # Group selector row (for HDF5 files with train/val/test groups)
        group_layout = QHBoxLayout()
        group_layout.addWidget(QLabel("Data Group:"))
        self.group_cb = QComboBox()
        self.group_cb.setToolTip("Select data group (for files with train/val/test splits)")
        self.group_cb.currentTextChanged.connect(self._on_group_changed)
        group_layout.addWidget(self.group_cb)
        group_layout.addStretch()
        ds_vbox.addLayout(group_layout)

        self.dataset_group.setLayout(ds_vbox)
        settings_vbox.addWidget(self.dataset_group)

        ctrl = QHBoxLayout()
        self.class_label = QLabel("Class ID:")
        self.digit_cb = QComboBox()
        self.digit_cb.setEditable(True)  # Allow manual entry for large datasets
        self.digit_cb.setToolTip("Select a class ID or type manually for large datasets")

        # Dataset sample index and randomizer (moved into dataset options panel)
        self.sample_spin = QSpinBox()
        self.new_btn = QPushButton("New Random Sample")
        self.new_btn.clicked.connect(self._random_sample)

        # Sequence length (moved near dt below)
        self.seq_spin = QSpinBox()
        # Allow longer sequences up to 100,000 steps
        self.seq_spin.setRange(1, 100000)
        self.seq_spin.setValue(100)
        # Connect selectors to update sample range when changed
        self.digit_cb.currentIndexChanged.connect(self._update_sample_range)
        self.seq_spin.valueChanged.connect(self._update_sample_range)
        # Input source selection (dataset sample or constant ones)
        ctrl.addWidget(QLabel("Input source:"))
        self.input_src_cb = QComboBox()
        self.input_src_cb.addItems(["Dataset sample", "Constant", "Gaussian noise", "Colored noise", "Sinusoid", "Square wave"])
        self.input_src_cb.currentTextChanged.connect(self._on_input_src_changed)
        ctrl.addWidget(self.input_src_cb)

        # Dataset-specific options (shown only when Input source = Dataset sample)
        self.dataset_opts_box = QWidget()
        ds_opts_v = QVBoxLayout(self.dataset_opts_box)
        ds_opts_v.setContentsMargins(0, 0, 0, 0)
        ds_opts_v.setSpacing(6)

        # Optional feature scaling (dataset-only)
        ds_scale_row = QHBoxLayout()
        ds_scale_row.setContentsMargins(0, 0, 0, 0)
        ds_scale_row.addWidget(QLabel("Feature min:"))
        self.mel_min_edit = QLineEdit()
        self.mel_min_edit.setValidator(create_float_validator(min_val=-FLOAT_MAX))
        self.mel_min_edit.setPlaceholderText("auto")
        self.mel_min_edit.setToolTip("Optional scaling lower bound. If both min/max are set, inputs are scaled to [min,max]. Use 'auto' to keep original scale.")
        ds_scale_row.addWidget(self.mel_min_edit)
        ds_scale_row.addWidget(QLabel("Feature max:"))
        self.mel_max_edit = QLineEdit()
        self.mel_max_edit.setValidator(create_float_validator(min_val=-FLOAT_MAX))
        self.mel_max_edit.setPlaceholderText("auto")
        self.mel_max_edit.setToolTip("Optional scaling upper bound. If both min/max are set, inputs are scaled to [min,max]. Use 'auto' to keep original scale.")
        ds_scale_row.addWidget(self.mel_max_edit)
        ds_scale_row.addStretch()
        ds_opts_v.addLayout(ds_scale_row)

        # Dataset sample selection row (Class ID, Sample Index, Random)
        ds_sel_row = QHBoxLayout()
        ds_sel_row.setContentsMargins(0, 0, 0, 0)
        ds_sel_row.addWidget(self.class_label)
        ds_sel_row.addWidget(self.digit_cb)
        ds_sel_row.addWidget(QLabel("Sample Index:"))
        ds_sel_row.addWidget(self.sample_spin)
        ds_sel_row.addWidget(self.new_btn)
        ds_sel_row.addStretch()
        ds_opts_v.insertLayout(0, ds_sel_row)

        # One-hot encoding controls for character data (dataset-only)
        encoding_row = QHBoxLayout()
        encoding_row.setContentsMargins(0, 0, 0, 0)
        encoding_row.addWidget(QLabel("Input Encoding:"))
        self.encoding_cb = QComboBox()
        self.encoding_cb.addItems(["Raw", "One-Hot"])
        self.encoding_cb.setToolTip("Choose input encoding: Raw (integers) or One-Hot (vectors)")
        encoding_row.addWidget(self.encoding_cb)
        encoding_row.addWidget(QLabel("Vocab Size:"))
        self.vocab_size_spin = QSpinBox()
        self.vocab_size_spin.setRange(1, 10000)
        self.vocab_size_spin.setValue(65)  # Default for Shakespeare
        self.vocab_size_spin.setToolTip("Number of unique characters/tokens (e.g., 65 for Shakespeare)")
        self.vocab_size_spin.setEnabled(False)  # Disabled by default
        encoding_row.addWidget(self.vocab_size_spin)
        # Auto-detect button
        self.auto_detect_btn = QPushButton("Auto-Detect")
        self.auto_detect_btn.setToolTip("Auto-detect encoding based on model input dimension")
        self.auto_detect_btn.clicked.connect(self._auto_detect_encoding)
        encoding_row.addWidget(self.auto_detect_btn)
        encoding_row.addStretch()
        ds_opts_v.addLayout(encoding_row)

        # Connect encoding selection to enable/disable vocab size
        self.encoding_cb.currentTextChanged.connect(self._on_encoding_changed)

        # Time settings row (mode: dt or total time)
        dt_ctrl = QHBoxLayout()
        self.time_mode_cb = QComboBox()
        self.time_mode_cb.addItems(["Specify dt", "Specify total time (ns)"])
        self.time_mode_label = QLabel("Time mode:")
        dt_ctrl.addWidget(self.time_mode_label)
        dt_ctrl.addWidget(self.time_mode_cb)

        # dt input
        self.dt_label = QLabel("dt (dimensionless):")
        dt_ctrl.addWidget(self.dt_label)
        self.dt_spin = QLineEdit()
        self.dt_spin.setValidator(create_float_validator(min_val=DT_MIN))
        # Initialize dt value from model if available
        if self._mgr and self._mgr.model:
            self.dt_spin.setText(str(self._mgr.model.dt))
        else:
            self.dt_spin.setText("37")
        dt_ctrl.addWidget(self.dt_spin)

        # total time input (ns)
        self.total_time_label = QLabel("Total time (ns):")
        dt_ctrl.addWidget(self.total_time_label)
        self.total_time_ns_spin = QLineEdit()
        self.total_time_ns_spin.setValidator(create_float_validator(min_val=0.0))
        self.total_time_ns_spin.setText("")
        dt_ctrl.addWidget(self.total_time_ns_spin)

        # live info labels
        self.label_dt_ns = QLabel("dt: — ns (dimless: —)")
        self.label_total_ns = QLabel("Total: — ns")
        try:
            self.label_dt_ns.setStyleSheet("color: #888;")
            self.label_total_ns.setStyleSheet("color: #888;")
        except Exception:
            pass
        dt_ctrl.addStretch()
        dt_ctrl.addWidget(self.label_dt_ns)
        dt_ctrl.addWidget(self.label_total_ns)

        # Append/prepend zeros control row
        zeros_ctrl = QHBoxLayout()
        self.append_zeros_chk = QCheckBox("Append zeros")
        self.append_zeros_chk.setChecked(False)
        self.append_zeros_chk.toggled.connect(self._on_append_zeros_toggled)
        zeros_ctrl.addWidget(self.append_zeros_chk)
        zeros_ctrl.addWidget(QLabel("Count:"))
        self.append_zeros_spin = QSpinBox()
        self.append_zeros_spin.setRange(0, 100000)
        self.append_zeros_spin.setValue(0)
        self.append_zeros_spin.setEnabled(False)
        zeros_ctrl.addWidget(self.append_zeros_spin)
        # Time (ns) input for total-time mode
        zeros_ctrl.addWidget(QLabel("Time (ns):"))
        self.append_zeros_time_ns = QLineEdit()
        self.append_zeros_time_ns.setValidator(create_float_validator(min_val=0.0))
        self.append_zeros_time_ns.setEnabled(False)
        zeros_ctrl.addWidget(self.append_zeros_time_ns)
        # Append mode (zeros or hold last)
        zeros_ctrl.addWidget(QLabel("Mode:"))
        self.append_mode_cb = QComboBox()
        self.append_mode_cb.addItem("Zeros", "zeros")
        self.append_mode_cb.addItem("Hold last", "hold_last")
        self.append_mode_cb.setEnabled(False)
        zeros_ctrl.addWidget(self.append_mode_cb)
        # Prepend zeros (zeros only)
        with contextlib.suppress(Exception):
            zeros_ctrl.addSpacing(16)
        self.prepend_zeros_chk = QCheckBox("Prepend zeros")
        self.prepend_zeros_chk.setChecked(False)
        self.prepend_zeros_chk.toggled.connect(self._on_prepend_zeros_toggled)
        zeros_ctrl.addWidget(self.prepend_zeros_chk)
        zeros_ctrl.addWidget(QLabel("Count:"))
        self.prepend_zeros_spin = QSpinBox()
        self.prepend_zeros_spin.setRange(0, 100000)
        self.prepend_zeros_spin.setValue(0)
        self.prepend_zeros_spin.setEnabled(False)
        zeros_ctrl.addWidget(self.prepend_zeros_spin)
        zeros_ctrl.addWidget(QLabel("Time (ns):"))
        self.prepend_zeros_time_ns = QLineEdit()
        self.prepend_zeros_time_ns.setValidator(create_float_validator(min_val=0.0))
        self.prepend_zeros_time_ns.setEnabled(False)
        zeros_ctrl.addWidget(self.prepend_zeros_time_ns)
        zeros_ctrl.addStretch()
        # Wrap parameter controls in a group box for clarity
        params_group = QGroupBox("Parameters & Simulation")
        params_vbox = QVBoxLayout()
        params_vbox.addLayout(ctrl)
        # Add dataset options below the main control row
        params_vbox.addWidget(self.dataset_opts_box)
        # Input generator parameter box (populated dynamically per input source)
        self.input_params_box = QGroupBox("Input Generator Parameters")
        # Make this row compact and avoid large blank fills
        self.input_params_box.setFlat(True)
        with contextlib.suppress(Exception):
            self.input_params_box.setStyleSheet(
                "QGroupBox { background-color: transparent; border: 1px solid rgba(255,255,255,30); margin-top: 8px; }\nQGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }",
            )
        self.input_params_layout = QHBoxLayout(self.input_params_box)
        try:
            self.input_params_layout.setContentsMargins(6, 6, 6, 6)
            self.input_params_layout.setSpacing(8)
            self.input_params_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        except Exception:
            pass
        # Keep height tight
        try:
            self.input_params_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.input_params_box.setMaximumHeight(72)
        except Exception:
            pass
        # Hidden by default; only shown for sources that expose parameters
        with contextlib.suppress(Exception):
            self.input_params_box.setVisible(False)
        params_vbox.addWidget(self.input_params_box)

        # Input help/description panel
        self.input_help_box = QGroupBox("Input Description")
        help_layout = QVBoxLayout()
        self.input_help_label = QLabel()
        self.input_help_label.setWordWrap(True)
        self.input_help_label.setText("Select an Input source to see parameters and examples.")
        help_layout.addWidget(self.input_help_label)
        self.input_help_box.setLayout(help_layout)
        with contextlib.suppress(Exception):
            self.input_help_box.setStyleSheet(
                "QGroupBox { background-color: transparent; border: 1px solid rgba(255,255,255,20); margin-top: 8px; }\nQGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }",
            )
        params_vbox.addWidget(self.input_help_box)

        # Place Seq Length directly above dt row
        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("Seq Length:"))
        seq_row.addWidget(self.seq_spin)
        seq_row.addStretch()
        params_vbox.addLayout(seq_row)

        params_vbox.addLayout(dt_ctrl)
        params_vbox.addLayout(zeros_ctrl)
        params_group.setLayout(params_vbox)
        settings_vbox.addWidget(params_group)

        # --- Display options (legend, total) ---
        display_group = QGroupBox("Display Options")
        disp_layout = QVBoxLayout()
        self.legend_chk = QCheckBox("Show Legend")
        self.legend_chk.setChecked(False)
        self.total_chk = QCheckBox("Show Total Line")
        self.total_chk.setChecked(False)
        # New: include initial state (t=0) option for State(s) plots
        self.include_s0_chk = QCheckBox("Include t=0 (initial state) for s(t)")
        self.include_s0_chk.setChecked(True)
        self.overlay_targets_chk = QCheckBox("Overlay targets on output layer (Seq2Seq)")
        self.overlay_targets_chk.setChecked(True)
        disp_layout.addWidget(self.legend_chk)
        disp_layout.addWidget(self.total_chk)
        disp_layout.addWidget(self.include_s0_chk)
        disp_layout.addWidget(self.overlay_targets_chk)
        display_group.setLayout(disp_layout)
        settings_vbox.addWidget(display_group)

        # Create export button (plot button now on Plots tab)
        self.export_btn = QPushButton("Export Sequences")
        self.export_btn.clicked.connect(self._export_sequences)

        # Action buttons group
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout()
        # Plot button moved to Plots tab; only export remains here
        actions_layout.addWidget(self.export_btn)
        actions_group.setLayout(actions_layout)
        settings_vbox.addWidget(actions_group)

        # Add stretch to push everything to the top of settings panel
        settings_vbox.addStretch(1)

        main_tabs.addTab(settings_tab, "Settings")

        # Finally add the plot tab SECOND so Settings appears first
        main_tabs.addTab(plot_tab, "Plots")

        # Apply saved UI settings
        self._apply_restored_settings()
        # Wire time mode change
        try:
            self.time_mode_cb.currentTextChanged.connect(self._on_time_mode_changed)
            self.dt_spin.textChanged.connect(self._update_time_labels)
            self.total_time_ns_spin.textChanged.connect(self._recompute_dt_from_total_time)
        except Exception:
            pass
        # Build initial input params UI
        with contextlib.suppress(Exception):
            self._on_input_src_changed()

        # Only auto-load dataset if a path is stored and exists
        try:
            p = Path(self.dataset_filepath) if self.dataset_filepath else None
            if p and p.exists():
                self.dataset_edit.setText(str(p))
                # Dataset will be loaded lazily when needed
        except Exception:
            pass

    def _apply_restored_settings(self) -> None:
        """Apply previously restored settings to the UI widgets."""
        # Metric
        if isinstance(self._saved_metric, str):
            idx = self.metric_cb.findText(self._saved_metric)
            if idx >= 0:
                self.metric_cb.setCurrentIndex(idx)
        # Backend
        try:
            saved_backend = self._settings.value("stateTraj/backend", "Torch", type=str)
            idxb = self.backend_cb.findText(saved_backend)
            if idxb >= 0:
                self.backend_cb.setCurrentIndex(idxb)
        except Exception:
            pass
        # Task type
        if isinstance(self._saved_task_type, str) and self._saved_task_type:
            idx = self.task_type_cb.findText(self._saved_task_type)
            if idx >= 0:
                self.task_type_cb.setCurrentIndex(idx)
                with contextlib.suppress(Exception):
                    self._on_task_type_changed()
        # Input source
        if isinstance(self._saved_input_src, str):
            idx = self.input_src_cb.findText(self._saved_input_src)
            if idx >= 0:
                self.input_src_cb.setCurrentIndex(idx)
        # Mel min/max
        if isinstance(self._saved_mel_min, str):
            self.mel_min_edit.setText(self._saved_mel_min)
        if isinstance(self._saved_mel_max, str):
            self.mel_max_edit.setText(self._saved_mel_max)
        # Encoding
        if isinstance(self._saved_encoding, str):
            idx = self.encoding_cb.findText(self._saved_encoding)
            if idx >= 0:
                self.encoding_cb.setCurrentText(self._saved_encoding)
        # Ensure enabled/disabled state updated
        with contextlib.suppress(Exception):
            self._on_encoding_changed()
        if isinstance(self._saved_vocab_size, int) and self._saved_vocab_size:
            self.vocab_size_spin.setValue(int(self._saved_vocab_size))
        # Sequence length
        if isinstance(self._saved_seq_len, int) and self._saved_seq_len:
            self.seq_spin.setValue(int(self._saved_seq_len))
        # Time mode & dt/total time
        try:
            saved_mode = self._settings.value("stateTraj/timeMode", "Specify dt", type=str)
            idxm = self.time_mode_cb.findText(saved_mode)
            if idxm >= 0:
                self.time_mode_cb.setCurrentIndex(idxm)
        except Exception:
            pass
        if isinstance(self._saved_dt, str) and self._saved_dt:
            self.dt_spin.setText(self._saved_dt)
        try:
            saved_tt = self._settings.value("stateTraj/totalTimeNs", "", type=str)
            if isinstance(saved_tt, str):
                self.total_time_ns_spin.setText(saved_tt)
        except Exception:
            pass
        try:
            self._on_time_mode_changed(self.time_mode_cb.currentText())
            self._update_time_labels()
        except Exception:
            pass
        # Legend/total
        if isinstance(self._saved_show_legend, bool):
            self.legend_chk.setChecked(self._saved_show_legend)
        if isinstance(self._saved_show_total, bool):
            self.total_chk.setChecked(self._saved_show_total)
        # Include s0
        if isinstance(self._saved_include_s0, bool):
            self.include_s0_chk.setChecked(self._saved_include_s0)
        # Append zeros
        if isinstance(self._saved_append_zeros_enabled, bool):
            self.append_zeros_chk.setChecked(self._saved_append_zeros_enabled)
            with contextlib.suppress(Exception):
                self._on_append_zeros_toggled(self._saved_append_zeros_enabled)
        if isinstance(self._saved_append_zeros_count, int) and self._saved_append_zeros_count is not None:
            self.append_zeros_spin.setValue(int(self._saved_append_zeros_count))
        if isinstance(self._saved_append_zeros_ns, str) and self._saved_append_zeros_ns is not None:
            self.append_zeros_time_ns.setText(self._saved_append_zeros_ns)
        # Append mode
        if hasattr(self, "append_mode_cb") and isinstance(self._saved_append_mode, str) and self._saved_append_mode:
            try:
                idx = self.append_mode_cb.findData(self._saved_append_mode)
                if idx >= 0:
                    self.append_mode_cb.setCurrentIndex(idx)
            except Exception:
                pass
        # Prepend zeros
        if isinstance(self._saved_prepend_zeros_enabled, bool):
            self.prepend_zeros_chk.setChecked(self._saved_prepend_zeros_enabled)
            with contextlib.suppress(Exception):
                self._on_prepend_zeros_toggled(self._saved_prepend_zeros_enabled)
        if isinstance(self._saved_prepend_zeros_count, int) and self._saved_prepend_zeros_count is not None:
            self.prepend_zeros_spin.setValue(int(self._saved_prepend_zeros_count))
        if isinstance(self._saved_prepend_zeros_ns, str) and self._saved_prepend_zeros_ns is not None:
            self.prepend_zeros_time_ns.setText(self._saved_prepend_zeros_ns)

    def _load_settings(self) -> None:
        """Restore settings from QSettings into instance variables."""
        try:
            self.dataset_filepath = self._settings.value("stateTraj/datasetPath", "", type=str)
        except Exception:
            self.dataset_filepath = ""
        self._saved_group_name = self._settings.value("stateTraj/group", None, type=str)
        self._saved_metric = self._settings.value("stateTraj/metric", None, type=str)
        self._saved_digit = self._settings.value("stateTraj/digit", None, type=str)
        self._saved_sample = self._settings.value("stateTraj/sampleIndex", None, type=int)
        self._saved_seq_len = self._settings.value("stateTraj/seqLen", None, type=int)
        self._saved_input_src = self._settings.value("stateTraj/inputSrc", None, type=str)
        self._saved_mel_min = self._settings.value("stateTraj/melMin", None, type=str)
        self._saved_mel_max = self._settings.value("stateTraj/melMax", None, type=str)
        self._saved_encoding = self._settings.value("stateTraj/encoding", None, type=str)
        self._saved_vocab_size = self._settings.value("stateTraj/vocabSize", None, type=int)
        self._saved_dt = self._settings.value("stateTraj/dt", None, type=str)
        self._saved_show_legend = self._settings.value("stateTraj/showLegend", False, type=bool)
        self._saved_show_total = self._settings.value("stateTraj/showTotal", False, type=bool)
        self._saved_include_s0 = self._settings.value("stateTraj/includeS0", True, type=bool)
        # Append zeros
        self._saved_append_zeros_enabled = self._settings.value("stateTraj/appendZerosEnabled", False, type=bool)
        self._saved_append_zeros_count = self._settings.value("stateTraj/appendZerosCount", 0, type=int)
        self._saved_append_mode = self._settings.value("stateTraj/appendMode", "zeros", type=str)
        # Prepend zeros
        self._saved_prepend_zeros_enabled = self._settings.value("stateTraj/prependZerosEnabled", False, type=bool)
        self._saved_prepend_zeros_count = self._settings.value("stateTraj/prependZerosCount", 0, type=int)
        # Task type
        self._saved_task_type = self._settings.value("stateTraj/taskType", None, type=str)
        # Input generator params
        try:
            self._saved_input_noise_std = self._settings.value("stateTraj/inputNoiseStd", 0.1, type=float)
        except Exception:
            self._saved_input_noise_std = 0.1
        try:
            self._saved_input_colored_beta = self._settings.value("stateTraj/inputColoredBeta", 2.0, type=float)
        except Exception:
            self._saved_input_colored_beta = 2.0
        try:
            self._saved_input_constant_val = self._settings.value("stateTraj/inputConstantVal", 1.0, type=float)
        except Exception:
            self._saved_input_constant_val = 1.0
        try:
            self._saved_input_sine_freq_mhz = self._settings.value("stateTraj/inputSineFreqMHz", 10.0, type=float)
        except Exception:
            self._saved_input_sine_freq_mhz = 10.0
        try:
            self._saved_input_sine_amp = self._settings.value("stateTraj/inputSineAmp", 1.0, type=float)
        except Exception:
            self._saved_input_sine_amp = 1.0
        try:
            self._saved_input_sine_phase_deg = self._settings.value("stateTraj/inputSinePhaseDeg", 0.0, type=float)
        except Exception:
            self._saved_input_sine_phase_deg = 0.0
        try:
            self._saved_input_sine_offset = self._settings.value("stateTraj/inputSineOffset", 0.0, type=float)
        except Exception:
            self._saved_input_sine_offset = 0.0

    def _save_settings(self) -> None:
        """Persist current UI state into QSettings."""
        with contextlib.suppress(Exception):
            self._settings.setValue("stateTraj/datasetPath", self.dataset_edit.text().strip())
        # Group
        grp = self.group_cb.currentData() if self.group_cb.count() > 0 else None
        self._settings.setValue("stateTraj/group", grp if grp is not None else "")
        # Task type
        with contextlib.suppress(Exception):
            self._settings.setValue("stateTraj/taskType", self.task_type_cb.currentText())
        # Metric, digit, sample
        self._settings.setValue("stateTraj/metric", self.metric_cb.currentText())
        self._settings.setValue("stateTraj/digit", self.digit_cb.currentText())
        self._settings.setValue("stateTraj/sampleIndex", int(self.sample_spin.value()))
        # Sequence length
        self._settings.setValue("stateTraj/seqLen", int(self.seq_spin.value()))
        # Backend
        with contextlib.suppress(Exception):
            self._settings.setValue("stateTraj/backend", self.backend_cb.currentText())
        # Input source and mel bounds
        self._settings.setValue("stateTraj/inputSrc", self.input_src_cb.currentText())
        self._settings.setValue("stateTraj/melMin", self.mel_min_edit.text().strip())
        self._settings.setValue("stateTraj/melMax", self.mel_max_edit.text().strip())
        # Encoding
        self._settings.setValue("stateTraj/encoding", self.encoding_cb.currentText())
        self._settings.setValue("stateTraj/vocabSize", int(self.vocab_size_spin.value()))
        # Time mode, dt, total time
        with contextlib.suppress(Exception):
            self._settings.setValue("stateTraj/timeMode", self.time_mode_cb.currentText())
        self._settings.setValue("stateTraj/dt", self.dt_spin.text().strip())
        self._settings.setValue("stateTraj/totalTimeNs", self.total_time_ns_spin.text().strip())
        # Display options
        self._settings.setValue("stateTraj/showLegend", bool(self.legend_chk.isChecked()))
        self._settings.setValue("stateTraj/showTotal", bool(self.total_chk.isChecked()))
        self._settings.setValue("stateTraj/includeS0", bool(self.include_s0_chk.isChecked()))
        # Append zeros
        try:
            self._settings.setValue("stateTraj/appendZerosEnabled", bool(self.append_zeros_chk.isChecked()))
            self._settings.setValue("stateTraj/appendZerosCount", int(self.append_zeros_spin.value()))
            self._settings.setValue("stateTraj/appendMode", self.append_mode_cb.currentData())
        except Exception:
            pass
        # Prepend zeros
        try:
            self._settings.setValue("stateTraj/prependZerosEnabled", bool(self.prepend_zeros_chk.isChecked()))
            self._settings.setValue("stateTraj/prependZerosCount", int(self.prepend_zeros_spin.value()))
        except Exception:
            pass
        # Input generator params
        try:
            if hasattr(self, "const_val_spin") and self.const_val_spin is not None:
                with contextlib.suppress(Exception):
                    self._settings.setValue("stateTraj/inputConstantVal", float(self.const_val_spin.value()))
            if hasattr(self, "noise_std_spin") and self.noise_std_spin is not None:
                with contextlib.suppress(Exception):
                    self._settings.setValue("stateTraj/inputNoiseStd", float(self.noise_std_spin.value()))
            if hasattr(self, "colored_beta_spin") and self.colored_beta_spin is not None:
                with contextlib.suppress(Exception):
                    self._settings.setValue("stateTraj/inputColoredBeta", float(self.colored_beta_spin.value()))
            if hasattr(self, "sine_freq_mhz_spin") and self.sine_freq_mhz_spin is not None:
                with contextlib.suppress(Exception):
                    self._settings.setValue("stateTraj/inputSineFreqMHz", float(self.sine_freq_mhz_spin.value()))
            if hasattr(self, "sine_amp_spin") and self.sine_amp_spin is not None:
                with contextlib.suppress(Exception):
                    self._settings.setValue("stateTraj/inputSineAmp", float(self.sine_amp_spin.value()))
            if hasattr(self, "sine_phase_deg_spin") and self.sine_phase_deg_spin is not None:
                with contextlib.suppress(Exception):
                    self._settings.setValue("stateTraj/inputSinePhaseDeg", float(self.sine_phase_deg_spin.value()))
            if hasattr(self, "sine_offset_spin") and self.sine_offset_spin is not None:
                with contextlib.suppress(Exception):
                    self._settings.setValue("stateTraj/inputSineOffset", float(self.sine_offset_spin.value()))
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        """Save settings when the dialog closes."""
        try:
            self._save_settings()
        finally:
            super().closeEvent(event)

    def _browse_dataset(self) -> None:
        """Open a file dialog to select a different HDF5 dataset."""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select dataset file",
            str(Path.home()),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if fname:
            self.dataset_edit.setText(fname)
            self._load_dataset_with_progress()
            self._save_settings()

    # ----- time helpers -----
    def _on_time_mode_changed(self, mode: str) -> None:
        specify_dt = "dt" in mode.lower()
        try:
            # dt vs total fields
            self.dt_label.setVisible(specify_dt)
            self.dt_spin.setVisible(specify_dt)
            self.total_time_label.setVisible(not specify_dt)
            self.total_time_ns_spin.setVisible(not specify_dt)
            # Toggle zeros controls: show only the relevant input types
            show_append = bool(self.append_zeros_chk.isChecked())
            show_prepend = bool(self.prepend_zeros_chk.isChecked())
            # Append
            self.append_zeros_spin.setVisible(specify_dt and show_append)
            if hasattr(self, "append_zeros_time_ns"):
                self.append_zeros_time_ns.setVisible((not specify_dt) and show_append)
            self.append_mode_cb.setVisible(show_append)
            # Prepend
            self.prepend_zeros_spin.setVisible(specify_dt and show_prepend)
            if hasattr(self, "prepend_zeros_time_ns"):
                self.prepend_zeros_time_ns.setVisible((not specify_dt) and show_prepend)
            # Ensure enabled state also matches
            if hasattr(self, "append_zeros_time_ns"):
                self.append_zeros_time_ns.setEnabled((not specify_dt) and show_append)
            self.append_zeros_spin.setEnabled(specify_dt and show_append)
            if hasattr(self, "prepend_zeros_time_ns"):
                self.prepend_zeros_time_ns.setEnabled((not specify_dt) and show_prepend)
            self.prepend_zeros_spin.setEnabled(specify_dt and show_prepend)
        except Exception:
            pass
        # When switching to total time mode, recompute dt if possible
        if not specify_dt:
            self._recompute_dt_from_total_time()
        self._update_time_labels()

    def _recompute_dt_from_total_time(self) -> None:
        try:
            tt_ns_txt = self.total_time_ns_spin.text().strip()
            if tt_ns_txt == "":
                return
            tt_ns = float(tt_ns_txt)
            steps = int(self.seq_spin.value())
            if steps <= 0:
                return
            # dt_dimless = (T_seconds * omega_c) / steps
            from soen_toolkit.physics.constants import get_omega_c

            dt_val = (tt_ns * 1e-9) * float(get_omega_c()) / float(steps)
            # Update dt field without recursion
            self.dt_spin.blockSignals(True)
            self.dt_spin.setText(f"{dt_val}")
            self.dt_spin.blockSignals(False)
        except Exception:
            pass
        self._update_time_labels()

    def _update_time_labels(self) -> None:
        try:
            # dt_ns = dt/omega_c * 1e9
            from soen_toolkit.physics.constants import dt_nanoseconds_per_step

            dt_txt = self.dt_spin.text().strip()
            dt_val = float(dt_txt) if dt_txt else 0.0
            dt_ns = dt_nanoseconds_per_step(dt_val)
            steps = int(self.seq_spin.value())
            total_ns = dt_ns * steps
            self.label_dt_ns.setText(f"dt: {dt_ns:.3g} ns (dimless: {dt_val:.3g})")
            self.label_total_ns.setText(f"Total: {total_ns:.3g} ns")
        except Exception:
            try:
                self.label_dt_ns.setText("dt: — ns (dimless: —)")
                self.label_total_ns.setText("Total: — ns")
            except Exception:
                pass

    def _load_dataset_with_progress(self) -> None:
        """Load dataset with progress feedback and group detection."""
        # Create and show progress dialog
        progress = QProgressDialog("Loading dataset...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Loading Dataset")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        QApplication.processEvents()

        try:
            ds_path = self.dataset_edit.text().strip() or DATA_FILEPATH
            if not Path(ds_path).exists():
                msg = f"Dataset file not found: {ds_path}"
                raise FileNotFoundError(msg)

            progress.setLabelText("Detecting HDF5 structure...")
            QApplication.processEvents()

            # Detect available groups
            self._detect_groups(ds_path)

            # Select default group (prefer 'val', then 'test', then 'train', then None)
            default_group = None
            if self.available_groups:
                for preferred in ["val", "test", "train"]:
                    if preferred in self.available_groups:
                        default_group = preferred
                        break
                if default_group is None:
                    default_group = self.available_groups[0]

            progress.setLabelText(f"Loading data from group: {default_group or 'root'}...")
            QApplication.processEvents()

            # Set the default group in combo box
            if default_group is not None:
                for i in range(self.group_cb.count()):
                    if self.group_cb.itemData(i) == default_group:
                        self.group_cb.setCurrentIndex(i)
                        break
            elif self.group_cb.count() > 0:
                self.group_cb.setCurrentIndex(0)  # Select first item (likely root level)

            # Load dataset with selected group
            selected_group = self.group_cb.currentData() if self.group_cb.count() > 0 else None
            self._load_dataset(ds_path, selected_group)

            progress.setLabelText("Processing labels...")
            QApplication.processEvents()

            # Check dataset size and inform user
            if len(self.dataset) > 50000:
                progress.setLabelText(f"Large dataset detected ({len(self.dataset)} samples). Using smart sampling...")
                QApplication.processEvents()

            # Update UI
            self._populate_digit_classes()
            # Mark dataset as successfully loaded for lazy loading cache
            self._dataset_loaded = True

        except Exception as e:
            self._dataset_loaded = False
            show_error(self, "Dataset Error", f"Failed to load dataset:\n{e!s}")
        finally:
            progress.close()

    def _detect_groups(self, ds_path: str) -> None:
        """Detect available groups in HDF5 file."""
        import h5py  # Lazy import: only when loading datasets

        self.available_groups = []

        with h5py.File(ds_path, "r") as f:
            # Check for predefined groups
            group_names = {k for k in f if isinstance(f[k], h5py.Group)}

            # Check if groups contain data and labels
            valid_groups = []
            for group_name in group_names:
                try:
                    grp = f[group_name]
                    if "data" in grp and "labels" in grp:
                        valid_groups.append(group_name)
                except Exception:
                    continue

            # Also check if root level has data and labels (no groups)
            has_root_data = "data" in f and "labels" in f

            self.available_groups = sorted(valid_groups)
            # Auto-detect task type from labels dtype/shape
            try:
                tgt = None
                if valid_groups:
                    preferred = "train" if "train" in valid_groups else valid_groups[0]
                    if "labels" in f[preferred]:
                        tgt = f[preferred]["labels"]
                elif has_root_data:
                    tgt = f["labels"]
                if tgt is not None:
                    is_int = np.issubdtype(tgt.dtype, np.integer)
                    is_1d = tgt.ndim == 1
                    self._is_seq2seq = not (is_int and is_1d)
            except Exception:
                pass

        # Update group combo box
        self.group_cb.clear()
        if has_root_data:
            self.group_cb.addItem("(root level)", None)

        for group in self.available_groups:
            self.group_cb.addItem(group, group)

        # Update tooltip with current file info
        if self.available_groups:
            groups_str = ", ".join(self.available_groups)
            tooltip = f"HDF5 file contains groups: {groups_str}. Select which group to use for plotting."
        elif has_root_data:
            tooltip = "HDF5 file contains data at root level (no groups)."
        else:
            tooltip = "No valid data groups found in HDF5 file."
        self.group_cb.setToolTip(tooltip)
        # Reflect auto-detected task type in UI
        try:
            self.task_type_cb.setCurrentText("Seq2Seq (regression)" if self._is_seq2seq else "Classification")
            self._on_task_type_changed()
        except Exception:
            pass

    def _load_dataset(self, ds_path: str, group: str | None = None) -> None:
        """Load dataset using GenericHDF5Dataset."""
        try:
            # Lazy import here: only load when needed
            from soen_toolkit.training.data.dataloaders import GenericHDF5Dataset

            # Use GenericHDF5Dataset which handles both grouped and non-grouped files
            # Note: For GUI visualization, we typically use raw encoding unless
            # the user specifically needs to test one-hot encoded inputs
            self.dataset = GenericHDF5Dataset(
                hdf5_path=ds_path,
                split=group,  # None for root level, group name for splits
                cache_in_memory=False,  # We only need labels for class selection
                input_encoding="raw",  # Default to raw for visualization
                vocab_size=None,
                one_hot_dtype="float32",
            )
            self.current_group = group

        except Exception as e:
            msg = f"Failed to create dataset: {e}"
            raise Exception(msg)

    def _populate_digit_classes(self) -> None:
        """Populate digit class combo box from dataset labels - efficiently!"""
        if self.dataset is None:
            return

        try:
            # Fast approach: Read labels directly from HDF5 instead of through dataset wrapper
            import h5py  # Lazy import: only when loading datasets

            ds_path = self.dataset_edit.text().strip() or DATA_FILEPATH

            with h5py.File(ds_path, "r") as f:
                # Get the right group
                if self.current_group:
                    labels_data = f[self.current_group]["labels"]
                else:
                    labels_data = f["labels"]

                # If labels are sequences or floats, switch to seq2seq mode and skip class discovery
                if labels_data.ndim > 1 or not np.issubdtype(labels_data.dtype, np.integer):
                    self._is_seq2seq = True
                    try:
                        self.task_type_cb.setCurrentText("Seq2Seq (regression)")
                        self._on_task_type_changed()
                    except Exception:
                        pass
                    # Store simple index list for sample selection
                    self.labels = list(range(labels_data.shape[0]))
                    # Populate a placeholder class value to avoid conversion errors
                    self.digit_cb.clear()
                    self.digit_cb.addItem("0")
                    self._update_sample_range()
                    return

                # For very large datasets, sample to find classes (much faster!)
                if labels_data.shape[0] > 10000:
                    # Sample 1000 random indices to find classes
                    import random

                    sample_size = min(1000, labels_data.shape[0])
                    sample_indices = sorted(random.sample(range(labels_data.shape[0]), sample_size))
                    sample_labels = labels_data[sample_indices]
                    classes = sorted({int(x) for x in sample_labels})
                else:
                    # For smaller datasets, read all labels directly (still much faster than dataset wrapper)
                    all_labels = labels_data[:]
                    classes = sorted({int(x) for x in all_labels})

                # Store labels for sample indexing (read efficiently)
                self.labels = labels_data[:].tolist()

            # Update class ID combo box
            self.digit_cb.clear()
            for c in classes:
                self.digit_cb.addItem(str(c))

            # Show helpful info in status (non-blocking)

            # Update window title to show current dataset info
            dataset_name = Path(self.dataset_edit.text()).name
            group_suffix = f" ({self.current_group})" if self.current_group else " (root)"
            self.setWindowTitle(f"State Trajectories - {dataset_name}{group_suffix}")

            self._update_sample_range()

        except Exception:
            # Fallback to a simple approach - just provide common classes
            self.digit_cb.clear()
            for c in range(10):  # Common for digit recognition
                self.digit_cb.addItem(str(c))
            # Create dummy labels list
            self.labels = list(range(len(self.dataset)))
            self._update_sample_range()

    def _on_group_changed(self) -> None:
        """Handle group selection change."""
        if not self.group_cb.currentData() and not self.group_cb.currentText():
            return  # Empty selection during population

        selected_group = self.group_cb.currentData()  # None for root, group name for splits
        ds_path = self.dataset_edit.text().strip()

        try:
            self._load_dataset(ds_path, selected_group)
            self._populate_digit_classes()
            self._save_settings()
        except Exception as e:
            show_error(self, "Dataset Error", f"Failed to load group data:\n{e!s}")

    def _update_sample_range(self) -> None:
        # If seq2seq/regression task, use all indices directly
        if getattr(self, "_is_seq2seq", False):
            total = len(self.dataset) if self.dataset is not None else len(getattr(self, "labels", []))
            self.indices = list(range(total))
            max_idx = max(0, total - 1)
            self.sample_spin.setRange(0, max_idx)
            # Keep current value within range
            cur = min(max(0, self.sample_spin.value()), max_idx)
            self.sample_spin.setValue(cur)
            return
        # Guard against empty or invalid selection (classification path)
        try:
            digit = int(self.digit_cb.currentText())
        except (ValueError, TypeError):
            if not getattr(self, "labels", []):
                self.indices = []
                self.sample_spin.setRange(0, 0)
                self.sample_spin.setValue(0)
            return
        if not getattr(self, "labels", []):
            self.indices = []
            self.sample_spin.setRange(0, 0)
            self.sample_spin.setValue(0)
            return
        self.indices = [i for i, lbl in enumerate(self.labels) if lbl == digit]
        if not self.indices:
            self.indices = list(range(len(self.labels)))
        max_idx = max(0, len(self.indices) - 1)
        self.sample_spin.setRange(0, max_idx)
        self.sample_spin.setValue(0)

    def _random_sample(self) -> None:
        if not getattr(self, "indices", []):
            return
        idx = random.choice(self.indices)
        pos = self.indices.index(idx)
        self.sample_spin.setValue(pos)

    def _downsample(self, spect: np.ndarray, target_len: int) -> np.ndarray:
        T, n_mels = spect.shape
        if target_len == T:
            return spect
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target_len)
        down = np.zeros((target_len, n_mels), dtype=np.float32)
        for i in range(n_mels):
            down[:, i] = np.interp(x_new, x_old, spect[:, i])
        return down

    def _load_sample(self, data_idx: int):
        """Load sample with encoding based on GUI settings."""
        # Load sample from current dataset
        if self.dataset is None:
            msg = "No dataset loaded"
            raise Exception(msg)

        # Get encoding settings from GUI
        encoding_settings = self._get_encoding_settings()

        # Get sample from dataset (it returns tensors)
        spect_tensor, _ = self.dataset[data_idx]
        spect = spect_tensor.numpy()  # Convert to numpy for processing

        target = int(self.seq_spin.value())
        mat = self._downsample(spect, target)

        # Apply one-hot encoding if requested
        if encoding_settings["encoding"] == "one_hot":
            vocab_size = encoding_settings["vocab_size"]

            # Convert to integers and apply one-hot encoding
            if mat.ndim == 1 or (mat.ndim == 2 and mat.shape[1] == 1):
                if mat.ndim == 2:
                    mat = mat.squeeze(1)

                mat_int = mat.astype(np.int64)
                seq_len = len(mat_int)

                # Clamp to valid range and warn if needed
                invalid_indices = (mat_int < 0) | (mat_int >= vocab_size)
                if np.any(invalid_indices):
                    mat_int = np.clip(mat_int, 0, vocab_size - 1)

                # Create one-hot encoding
                one_hot = np.zeros((seq_len, vocab_size), dtype=np.float32)
                one_hot[np.arange(seq_len), mat_int] = 1.0
                mat = one_hot

        return torch.tensor(mat, dtype=torch.float32)

    def _run_and_get_simulation_results(self):
        """Prepares inputs, runs the model simulation, and returns input features and histories
        for the selected metric (s, phi, or g).
        """
        if self._mgr.model is None:
            show_warning(self, "No model", "Build or load a model first.")
            return None, None, None, None

        # Lazy load dataset if needed
        src = self.input_src_cb.currentText()
        if src == "Dataset sample" and not self._dataset_loaded and self.dataset_filepath:
            try:
                p = Path(self.dataset_filepath)
                if p.exists() and not self.dataset:
                    self._load_dataset_with_progress()
                    self._dataset_loaded = True
            except Exception:
                show_warning(self, "Dataset Error", "Failed to load dataset before running simulation.")
                return None, None, None, None

        # ------------------------------------------------------------------
        # 1️⃣   Configure tracking based on selected metric
        # ------------------------------------------------------------------
        sel = self.metric_cb.currentText()
        # Save current tracking settings to restore later
        orig_phi = self._mgr.sim_config.track_phi
        orig_g = self._mgr.sim_config.track_g
        orig_s = self._mgr.sim_config.track_s
        orig_power = self._mgr.sim_config.track_power

        # Enable full tracking temporarily
        self._mgr.model.set_tracking(track_phi=True, track_g=True, track_s=True, track_power=True)

        # ------------------------------------------------------------------
        # 2️⃣   Validate dt input
        # ------------------------------------------------------------------
        try:
            new_dt = float(self.dt_spin.text() or "37")
            if new_dt <= 0:
                show_warning(self, "Invalid dt", "Time step (dt) must be positive.")
                return None, None, None, None
        except ValueError:
            show_warning(self, "Invalid dt", "Invalid numeric value for Time step (dt).")
            return None, None, None, None

        self._mgr.model.set_dt(new_dt)

        # ------------------------------------------------------------------
        # 3️⃣   Prepare input features (dataset sample or synthetic generators)
        # ------------------------------------------------------------------
        expected_dim = self._mgr.model.layers_config[0].params.get("dim")
        seq_len = self.seq_spin.value()

        src = self.input_src_cb.currentText()
        # Seed random generators for deterministic synthetic inputs
        if src in ("Gaussian noise", "Colored noise"):
            try:
                model_seed = getattr(self._mgr.model, "_creation_seed", None)
            except Exception:
                model_seed = None
            try:
                seed_val = int(model_seed) if model_seed is not None else 1337
            except Exception:
                seed_val = 1337
            with contextlib.suppress(Exception):
                random.seed(seed_val)
            with contextlib.suppress(Exception):
                np.random.seed(seed_val)
            with contextlib.suppress(Exception):
                torch.manual_seed(seed_val)
        if src == "Constant":
            # Prefer current widget value if present; fall back to saved setting
            try:
                if hasattr(self, "const_val_spin") and self.const_val_spin is not None:
                    val = float(self.const_val_spin.value())
                    # Keep in-memory and settings in sync
                    self._saved_input_constant_val = val
                else:
                    val = float(self._saved_input_constant_val if getattr(self, "_saved_input_constant_val", None) is not None else 1.0)
            except Exception:
                val = float(self._saved_input_constant_val if getattr(self, "_saved_input_constant_val", None) is not None else 1.0)
            features = torch.full((seq_len, expected_dim), fill_value=val, dtype=torch.float32)
        elif src == "Gaussian noise":
            # Prefer current widget value if present; fall back to saved setting
            try:
                if hasattr(self, "noise_std_spin") and self.noise_std_spin is not None:
                    std = float(self.noise_std_spin.value())
                    self._saved_input_noise_std = std
                else:
                    std = float(self._saved_input_noise_std if self._saved_input_noise_std is not None else 0.1)
            except Exception:
                std = float(self._saved_input_noise_std if self._saved_input_noise_std is not None else 0.1)
            features = torch.randn(seq_len, expected_dim, dtype=torch.float32) * std
        elif src == "Sinusoid":
            # Build timebase in seconds using dt and omega_c: dt/omega_c seconds per step
            try:
                dt_val = float(self.dt_spin.text())
            except Exception:
                dt_val = new_dt
            dt_s = dt_val / get_omega_c()
            t = torch.arange(seq_len, dtype=torch.float32) * dt_s
            # Frequency in MHz -> Hz
            try:
                if hasattr(self, "sine_freq_mhz_spin") and self.sine_freq_mhz_spin is not None:
                    f_mhz = float(self.sine_freq_mhz_spin.value())
                    self._saved_input_sine_freq_mhz = f_mhz
                else:
                    f_mhz = float(self._saved_input_sine_freq_mhz if self._saved_input_sine_freq_mhz is not None else 10.0)
            except Exception:
                f_mhz = float(self._saved_input_sine_freq_mhz if self._saved_input_sine_freq_mhz is not None else 10.0)
            f_hz = f_mhz * 1e6
            try:
                if hasattr(self, "sine_amp_spin") and self.sine_amp_spin is not None:
                    amp = float(self.sine_amp_spin.value())
                    self._saved_input_sine_amp = amp
                else:
                    amp = float(self._saved_input_sine_amp if self._saved_input_sine_amp is not None else 1.0)
            except Exception:
                amp = float(self._saved_input_sine_amp if self._saved_input_sine_amp is not None else 1.0)
            try:
                if hasattr(self, "sine_phase_deg_spin") and self.sine_phase_deg_spin is not None:
                    phase_deg = float(self.sine_phase_deg_spin.value())
                    self._saved_input_sine_phase_deg = phase_deg
                else:
                    phase_deg = float(self._saved_input_sine_phase_deg if self._saved_input_sine_phase_deg is not None else 0.0)
            except Exception:
                phase_deg = float(self._saved_input_sine_phase_deg if self._saved_input_sine_phase_deg is not None else 0.0)
            phase = math.radians(phase_deg)
            try:
                if hasattr(self, "sine_offset_spin") and self.sine_offset_spin is not None:
                    offset = float(self.sine_offset_spin.value())
                    self._saved_input_sine_offset = offset
                else:
                    offset = float(self._saved_input_sine_offset if self._saved_input_sine_offset is not None else 0.0)
            except Exception:
                offset = float(self._saved_input_sine_offset if self._saved_input_sine_offset is not None else 0.0)
            s = offset + amp * torch.sin(2 * math.pi * f_hz * t + phase)
            # Tile to expected_dim
            features = s.view(-1, 1).repeat(1, expected_dim)
        elif src == "Square wave":
            # Build timebase in seconds using dt and omega_c: dt/omega_c seconds per step
            try:
                dt_val = float(self.dt_spin.text())
            except Exception:
                dt_val = new_dt
            dt_s = dt_val / get_omega_c()
            t = torch.arange(seq_len, dtype=torch.float32) * dt_s
            # Frequency in MHz -> Hz
            try:
                if hasattr(self, "sine_freq_mhz_spin") and self.sine_freq_mhz_spin is not None:
                    f_mhz = float(self.sine_freq_mhz_spin.value())
                    self._saved_input_sine_freq_mhz = f_mhz
                else:
                    f_mhz = float(self._saved_input_sine_freq_mhz if self._saved_input_sine_freq_mhz is not None else 10.0)
            except Exception:
                f_mhz = float(self._saved_input_sine_freq_mhz if self._saved_input_sine_freq_mhz is not None else 10.0)
            f_hz = f_mhz * 1e6
            try:
                if hasattr(self, "sine_amp_spin") and self.sine_amp_spin is not None:
                    amp = float(self.sine_amp_spin.value())
                    self._saved_input_sine_amp = amp
                else:
                    amp = float(self._saved_input_sine_amp if self._saved_input_sine_amp is not None else 1.0)
            except Exception:
                amp = float(self._saved_input_sine_amp if self._saved_input_sine_amp is not None else 1.0)
            try:
                if hasattr(self, "sine_phase_deg_spin") and self.sine_phase_deg_spin is not None:
                    phase_deg = float(self.sine_phase_deg_spin.value())
                    self._saved_input_sine_phase_deg = phase_deg
                else:
                    phase_deg = float(self._saved_input_sine_phase_deg if self._saved_input_sine_phase_deg is not None else 0.0)
            except Exception:
                phase_deg = float(self._saved_input_sine_phase_deg if self._saved_input_sine_phase_deg is not None else 0.0)
            phase = math.radians(phase_deg)
            try:
                if hasattr(self, "sine_offset_spin") and self.sine_offset_spin is not None:
                    offset = float(self.sine_offset_spin.value())
                    self._saved_input_sine_offset = offset
                else:
                    offset = float(self._saved_input_sine_offset if self._saved_input_sine_offset is not None else 0.0)
            except Exception:
                offset = float(self._saved_input_sine_offset if self._saved_input_sine_offset is not None else 0.0)
            base = torch.sin(2 * math.pi * f_hz * t + phase)
            sq = torch.sign(base)
            s = offset + amp * sq
            # Tile to expected_dim
            features = s.view(-1, 1).repeat(1, expected_dim)
        elif src == "Colored noise":
            # Generate 1/f^beta colored noise along time using FFT shaping
            try:
                dt_val = float(self.dt_spin.text())
            except Exception:
                dt_val = new_dt
            dt_s = dt_val / get_omega_c()
            # Prefer current widget value if present; fall back to saved setting
            try:
                if hasattr(self, "colored_beta_spin") and self.colored_beta_spin is not None:
                    beta = float(self.colored_beta_spin.value())
                    self._saved_input_colored_beta = beta
                else:
                    beta = float(self._saved_input_colored_beta if self._saved_input_colored_beta is not None else 2.0)
            except Exception:
                beta = float(self._saved_input_colored_beta if self._saved_input_colored_beta is not None else 2.0)
            # Start from white noise [seq_len, dim]
            x = torch.randn(seq_len, expected_dim, dtype=torch.float32)
            # For extremely short sequences, shaping is ill-defined; fall back to white noise
            if seq_len < 2:
                features = x
            else:
                # FFT along time axis (dim=0); use rfft per dim column
                X = torch.fft.rfft(x, dim=0)
                # Frequency bins for rfft
                f = torch.fft.rfftfreq(seq_len, d=float(dt_s))
                f = f.view(-1, 1)  # [F, 1]
                # Gain ~ f^{-beta/2}, avoid f=0
                eps = 1e-12
                gain = (f.clamp_min(eps)) ** (-beta * 0.5)
                # Zero DC
                gain[0:1, :] = 0.0
                # Normalize gain energy roughly; guard against zero/NaN
                denom = torch.sqrt(torch.mean(gain.squeeze(1) ** 2))
                if torch.isfinite(denom) and denom.item() > 0:
                    gain = gain / denom
                # Apply gain broadcasting over dims
                Xc = X * gain
                xc = torch.fft.irfft(Xc, n=seq_len, dim=0)
                # Remove residual mean per column
                xc = xc - xc.mean(dim=0, keepdim=True)
                # Replace NaN/Inf if any due to numerical issues
                try:
                    xc = torch.nan_to_num(xc, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    xc[~torch.isfinite(xc)] = 0.0
                # If all zeros (e.g., seq_len==1 degenerate case), fall back to white noise
                if float(xc.abs().sum().item()) == 0.0:
                    features = x
                else:
                    features = xc.to(dtype=torch.float32)
        else:
            # Defensive: ensure indices exist and pos is in range
            if not getattr(self, "indices", []) or len(self.indices) == 0:
                show_warning(
                    self,
                    "No Samples",
                    "No valid sample indices available. Using constant ones input.",
                )
                features = torch.ones(seq_len, expected_dim, dtype=torch.float32)
            else:
                pos = int(self.sample_spin.value())
                if pos < 0 or pos >= len(self.indices):
                    pos = 0
                    with contextlib.suppress(Exception):
                        self.sample_spin.setValue(0)
                data_idx = self.indices[pos]
                try:
                    features = self._load_sample(data_idx)
                except Exception as e:
                    show_error(self, "Load Sample Error", str(e))
                    return None, None, None, None

            # Check dimension compatibility
            actual_dim = features.shape[1] if features.ndim > 1 else 1

            if actual_dim != expected_dim:
                if actual_dim < expected_dim:
                    # Input too small - might need one-hot encoding
                    encoding_settings = self._get_encoding_settings()
                    if encoding_settings["encoding"] == "raw":
                        show_warning(
                            self,
                            "Dimension Mismatch",
                            f"Input dimension mismatch!\n\n"
                            f"Model expects: {expected_dim}D\n"
                            f"Sample provides: {actual_dim}D\n\n"
                            f"For character data, try:\n"
                            f"1. Click 'Auto-Detect' button\n"
                            f"2. Or manually select 'One-Hot' encoding\n\n"
                            f"Defaulting to constant ones for now.",
                        )
                    else:
                        show_error(
                            self,
                            "Encoding Error",
                            f"One-hot encoding failed to produce correct dimensions.\n\nExpected: {expected_dim}D\nGot: {actual_dim}D\n\nCheck vocab_size setting.",
                        )
                elif actual_dim > expected_dim:
                    show_warning(
                        self,
                        "Input Mismatch",
                        f"Sample has input dim {actual_dim}; model expects {expected_dim}. Truncating to fit.",
                    )
                    # Truncate extra dimensions
                    features = features[:, :expected_dim]

                # If dimensions still don't match, fall back to ones
                if features.shape[1] != expected_dim:
                    features = torch.ones(seq_len, expected_dim, dtype=torch.float32)

        if src == "Dataset sample":
            min_txt = self.mel_min_edit.text().strip()
            max_txt = self.mel_max_edit.text().strip()
            if min_txt and max_txt:
                try:
                    mel_min = float(min_txt)
                    mel_max = float(max_txt)
                    if mel_max > mel_min:
                        fmin = features.min().item()
                        fmax = features.max().item()
                        if fmax != fmin:
                            features = ((features - fmin) / (fmax - fmin)) * (mel_max - mel_min) + mel_min
                except ValueError:
                    pass

        # Optionally prepend to extend the sequence at the front
        if bool(getattr(self, "prepend_zeros_chk", None) and self.prepend_zeros_chk.isChecked()):
            try:
                is_total = hasattr(self, "time_mode_cb") and "total" in self.time_mode_cb.currentText().lower()
                if is_total:
                    # Convert requested time (ns) to steps
                    tt_ns_txt = self.prepend_zeros_time_ns.text().strip() if hasattr(self, "prepend_zeros_time_ns") else ""
                    extra_ns = float(tt_ns_txt) if tt_ns_txt else 0.0
                    from soen_toolkit.physics.constants import dt_nanoseconds_per_step

                    step_ns = dt_nanoseconds_per_step(float(self.dt_spin.text() or 0.0))
                    zeros_to_prepend = round(extra_ns / step_ns) if step_ns > 0 else 0
                else:
                    zeros_to_prepend = int(self.prepend_zeros_spin.value())
            except Exception:
                zeros_to_prepend = 0
            if zeros_to_prepend > 0:
                if features.ndim == 1:
                    features = features.view(-1, 1)
                dim = features.shape[1]
                prepend_block = torch.zeros((zeros_to_prepend, dim), dtype=features.dtype, device=features.device)
                features = torch.cat([prepend_block, features], dim=0)
                seq_len = features.shape[0]

        # Optionally append to extend the sequence
        if bool(getattr(self, "append_zeros_chk", None) and self.append_zeros_chk.isChecked()):
            try:
                is_total = hasattr(self, "time_mode_cb") and "total" in self.time_mode_cb.currentText().lower()
                if is_total:
                    tt_ns_txt = self.append_zeros_time_ns.text().strip() if hasattr(self, "append_zeros_time_ns") else ""
                    extra_ns = float(tt_ns_txt) if tt_ns_txt else 0.0
                    from soen_toolkit.physics.constants import dt_nanoseconds_per_step

                    step_ns = dt_nanoseconds_per_step(float(self.dt_spin.text() or 0.0))
                    zeros_to_append = round(extra_ns / step_ns) if step_ns > 0 else 0
                else:
                    zeros_to_append = int(self.append_zeros_spin.value())
            except Exception:
                zeros_to_append = 0
            if zeros_to_append > 0:
                # Ensure features is [seq_len, dim]
                if features.ndim == 1:
                    features = features.view(-1, 1)
                dim = features.shape[1]
                mode = None
                try:
                    mode = self.append_mode_cb.currentData()
                except Exception:
                    mode = "zeros"
                if mode == "hold_last":
                    last_row = features[-1:].clone()
                    append_block = last_row.repeat(zeros_to_append, 1)
                else:
                    append_block = torch.zeros((zeros_to_append, dim), dtype=features.dtype, device=features.device)
                features = torch.cat([features, append_block], dim=0)
                # Also update seq_len locally for simulation input
                seq_len = features.shape[0]

        x = features.unsqueeze(0)  # Add batch dimension

        backend = self.backend_cb.currentText()
        try:
            # If JAX selected, only support State(s) metric for now
            if backend.startswith("JAX"):
                if not _JAX_AVAILABLE:
                    show_warning(self, "JAX backend unavailable", "JAX is not available in this environment. Falling back to Torch.")
                    backend = "Torch"
                elif not sel.startswith("State"):
                    show_info(self, "Metric not supported in JAX", "JAX backend currently supports only State(s) metric. Falling back to Torch.")
                    backend = "Torch"

            if backend.startswith("JAX"):
                # Ensure we have a cached JAX model/apply for this source Torch model
                ok = self._ensure_jax_model()
                if not ok:
                    backend = "Torch"
                else:
                    pass

            if backend.startswith("JAX"):
                # Run JAX forward using cached compiled apply
                import jax
                import jax.numpy as jnp

                with contextlib.suppress(Exception):
                    jax.config.update("jax_platforms", "cpu")
                x_np = x.detach().cpu().numpy()
                apply_fn = self._jax_apply
                arr = jnp.asarray(x_np)
                shape_key = tuple(arr.shape)
                # Invalidate cached JAX artifacts if timestep length changed
                if getattr(self, "_jax_last_shape_key", None) is not None and self._jax_last_shape_key != shape_key:
                    try:
                        self._jax_model = None
                        self._jax_apply = None
                        self._jax_source_model_id = None
                        self._jax_warm_shapes = set()
                    except Exception:
                        pass
                    ok2 = self._ensure_jax_model()
                    if not ok2:
                        msg = "Failed to rebuild JAX model after timestep change"
                        raise RuntimeError(msg)
                    apply_fn = self._jax_apply
                self._jax_last_shape_key = shape_key
                # Warm-up compile/execute without timing for first time this shape is seen
                if shape_key not in getattr(self, "_jax_warm_shapes", set()):
                    try:
                        wf_final, _ = apply_fn(arr)
                        with contextlib.suppress(Exception):
                            wf_final.block_until_ready()
                    finally:
                        try:
                            self._jax_warm_shapes.add(shape_key)
                        except Exception:
                            self._jax_warm_shapes = {shape_key}
                # Timed execution
                t0 = time.time()
                final_hist, all_hists = apply_fn(arr)
                with contextlib.suppress(Exception):
                    final_hist.block_until_ready()
                elapsed = time.time() - t0
                # Convert histories to Torch tensors
                raw_state_histories = [torch.tensor(np.array(h), dtype=torch.float32) for h in all_hists]
            else:
                # Torch path: run once and capture raw per-layer state histories (include initial state)
                start_time = time.time()
                # If total-time mode, recompute dt from total time before running
                try:
                    if hasattr(self, "time_mode_cb") and "total" in self.time_mode_cb.currentText().lower():
                        self._recompute_dt_from_total_time()
                except Exception:
                    pass
                _, raw_state_histories = self._mgr.model(x)
                elapsed = time.time() - start_time

            # ------------------------------------------------------------------
            # 4️⃣   Retrieve histories for selected metric
            # ------------------------------------------------------------------
            if sel.startswith("State"):
                # For state metric, use raw per-layer histories directly;
                # caller decides whether to include s0 or not.
                histories = raw_state_histories
            elif sel.startswith("Flux"):
                histories = self._mgr.model.get_phi_history()
            elif sel.startswith("Non-Linearity"):
                histories = self._mgr.model.get_g_history()
            elif sel.startswith("Power"):
                histories = []
                for lyr in self._mgr.model.layers:
                    dimless_bias = getattr(lyr, "power_bias_dimensionless", None)
                    dimless_diss = getattr(lyr, "power_diss_dimensionless", None)
                    if dimless_bias is None or dimless_diss is None:
                        histories.append(None)
                        continue
                    dimless_total = dimless_bias + dimless_diss
                    from soen_toolkit.physics.constants import DEFAULT_IC, DEFAULT_PHI0

                    phys_total = convert_power_to_physical(
                        dimless_total,
                        getattr(lyr, "Ic", DEFAULT_IC),
                        getattr(lyr, "Phi0", DEFAULT_PHI0),
                        getattr(lyr, "wc", float(get_omega_c())),
                    )
                    histories.append(phys_total)
            elif sel.startswith("Energy"):
                # Energy histories are not exposed via a dedicated method, so assemble manually
                histories = []
                for lyr in self._mgr.model.layers:
                    eb = getattr(lyr, "energy_bias_dimensionless", None)
                    ed = getattr(lyr, "energy_diss_dimensionless", None)
                    if eb is not None and ed is not None:
                        histories.append(eb + ed)
                    else:
                        histories.append(None)
            else:
                histories = []

            # Restore original tracking preferences
            self._mgr.model.set_tracking(
                track_phi=orig_phi,
                track_g=orig_g,
                track_s=orig_s,
                track_power=orig_power,
            )

            # Return batched input, metric-specific histories, raw state histories, and elapsed time
            return x, histories, raw_state_histories, elapsed
        except Exception as e:
            show_error(self, "Simulation Error", str(e))
            # Restore original tracking even on error
            self._mgr.model.set_tracking(
                track_phi=orig_phi,
                track_g=orig_g,
                track_s=orig_s,
                track_power=orig_power,
            )
            return None, None, None, None

    def _on_backend_changed(self, *_args) -> None:
        """Reset cached JAX artifacts when backend selection changes."""
        if self.backend_cb.currentText().startswith("JAX"):
            # Keep cache if same model; conversion happens lazily
            try:
                self._jax_warm_shapes.clear()
            except Exception:
                self._jax_warm_shapes = set()
        else:
            # Switching to Torch: keep cache but it won't be used
            pass

    def _ensure_jax_model(self) -> bool:
        """Convert and cache JAX model and compiled apply if needed; show progress dialog.

        Returns True if ready, False if failed and caller should fallback.
        """
        # Lazy-load JAX on first use
        global _JAX_AVAILABLE

        # Try to initialize JAX if not already done
        if not _JAX_AVAILABLE:
            try:
                os.environ.setdefault("JAX_PLATFORMS", "cpu")
                import jax

                with contextlib.suppress(Exception):
                    jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "cpu"))
                with contextlib.suppress(Exception):
                    jax.config.update("jax_enable_x64", False)
                from soen_toolkit.utils.port_to_jax import (
                    convert_core_model_to_jax as _convert_to_jax,
                )

                _JAX_AVAILABLE = True
            except Exception:
                return False
        else:
            # JAX is available, re-import for use in this method
            import jax

            from soen_toolkit.utils.port_to_jax import (
                convert_core_model_to_jax as _convert_to_jax,
            )

        # If model object changed, invalidate cache
        model_id = id(self._mgr.model)
        if (self._jax_model is not None) and (self._jax_source_model_id == model_id) and (self._jax_apply is not None):
            return True
        # Show conversion dialog
        progress = QProgressDialog("Converting model to JAX (CPU backend)...", None, 0, 0, self)
        progress.setWindowTitle("JAX Conversion")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()
        try:
            # Force CPU backend for converter
            try:
                os.environ["JAX_PLATFORMS"] = "cpu"
                jax.config.update("jax_platforms", "cpu")
            except Exception:
                pass
            jax_model = _convert_to_jax(self._mgr.model)
            with contextlib.suppress(Exception):
                jax_model.prepare()
            apply_jit = jax.jit(jax_model.__call__)
            # Cache
            self._jax_model = jax_model
            self._jax_apply = apply_jit
            self._jax_source_model_id = model_id
            try:
                self._jax_warm_shapes.clear()
            except Exception:
                self._jax_warm_shapes = set()
            self._jax_last_shape_key = None
            return True
        except NotImplementedError as nie:
            show_warning(self, "JAX Conversion Unsupported", f"This model cannot be converted to JAX:\n{nie!s}")
            return False
        except Exception as e:
            # Retry after forcing CPU already attempted; report and fail
            show_error(self, "JAX Conversion Error", f"Failed to convert model to JAX:\n{e!s}")
            return False
        finally:
            progress.close()

    def _on_plot(self) -> None:
        results = self._run_and_get_simulation_results()
        if not isinstance(results, tuple) or len(results) != 4:
            return
        input_features, histories, raw_state_histories, elapsed = results
        if input_features is None or histories is None:
            return  # Error already handled by _run_and_get_simulation_results

        # Persist current UI settings on successful run
        with contextlib.suppress(Exception):
            self._save_settings()

        # Use the histories from the run; raw_state_histories contains s0..sT

        # Store current tab index before clearing
        current_tab = self.tabs.currentIndex()

        self.tabs.clear()
        dt = self._mgr.model.dt  # grab current timestep

        # Print sim time to terminal (same as overlay on plot)
        try:
            backend_text = self.backend_cb.currentText()
        except Exception:
            backend_text = "Torch"
        try:
            # input_features has shape [B, T, D]
            _, T, D = input_features.shape
        except Exception:
            _T, _D = None, None
        with contextlib.suppress(Exception):
            pass

        # Optional: second run for Torch backend with tracking disabled to measure overhead
        try:
            if not str(backend_text).strip().upper().startswith("JAX"):
                # Save current tracking preferences
                o_phi = bool(getattr(self._mgr.sim_config, "track_phi", False))
                o_g = bool(getattr(self._mgr.sim_config, "track_g", False))
                o_s = bool(getattr(self._mgr.sim_config, "track_s", False))
                o_p = bool(getattr(self._mgr.sim_config, "track_power", False))
                # Disable all tracking and time a pure forward
                self._mgr.model.set_tracking(track_phi=False, track_g=False, track_s=False, track_power=False)
                try:
                    t0 = time.time()
                    with torch.no_grad():
                        _ = self._mgr.model(input_features)
                    time.time() - t0
                finally:
                    # Restore preferences
                    self._mgr.model.set_tracking(track_phi=o_phi, track_g=o_g, track_s=o_s, track_power=o_p)
                with contextlib.suppress(Exception):
                    pass
        except Exception:
            # Non-fatal if timing print fails for any reason
            pass

        # Convert to real time using centralized helper
        dt_ns = dt_nanoseconds_per_step(dt)

        sel_metric = self.metric_cb.currentText()
        y_label = {
            "State (s)": "s(t)",
            "Flux (phi)": "phi(t)",
            "Non-Linearity (g)": "g(t)",
            "Power": "P(t) [nW]",
            "Energy": "E(t) [nJ]",
        }.get(sel_metric, "Value")

        for idx, (cfg, hist) in enumerate(zip(self._mgr.layers, histories, strict=False)):
            layer_id = cfg.layer_id
            if hist is None:
                # Create placeholder widget
                w = pg.PlotWidget()
                w.setBackground("w")
                txt = pg.TextItem("No data for this layer", color=(200, 0, 0), anchor=(0.5, 0.5))
                txt.setFont(QFont("", 14))
                # Position text in center of view
                txt.setPos(0, 0)
                w.addItem(txt)
                self.tabs.addTab(w, f"Layer {layer_id}")
                continue
            # Choose source history:
            # - For State(s), use raw_state_histories (include s0 if enabled, else drop it)
            # - For other metrics, use provided histories as-is
            if sel_metric.startswith("State"):
                hist_source = raw_state_histories[idx] if self.include_s0_chk.isChecked() else raw_state_histories[idx][:, 1:, :]
            else:
                hist_source = hist

            # hist_source shape [batch, T, dim] or [batch, T+1, dim] for state (when include_s0)
            # Convert history tensor to CPU for processing
            hist_cpu = hist_source[0].detach().cpu()

            # Convert to physical units when required
            if sel_metric.startswith("Power"):
                lyr = self._mgr.model.layers[idx]
                from soen_toolkit.physics.constants import DEFAULT_IC, DEFAULT_PHI0

                Phi0 = getattr(lyr, "Phi0", getattr(lyr, "PHI0", DEFAULT_PHI0))
                Ic = getattr(lyr, "Ic", getattr(lyr, "IC", DEFAULT_IC))
                wc = getattr(lyr, "wc", getattr(lyr, "WC", float(get_omega_c())))
                hist_phys = convert_power_to_physical(hist_cpu, Ic, Phi0, wc) * 1e9  # nW
                arr = hist_phys.numpy()
            elif sel_metric.startswith("Energy"):
                lyr = self._mgr.model.layers[idx]
                from soen_toolkit.physics.constants import DEFAULT_IC, DEFAULT_PHI0

                Phi0 = getattr(lyr, "Phi0", getattr(lyr, "PHI0", DEFAULT_PHI0))
                Ic = getattr(lyr, "Ic", getattr(lyr, "IC", DEFAULT_IC))
                hist_phys = convert_energy_to_physical(hist_cpu, Ic, Phi0) * 1e9  # nJ
                arr = hist_phys.numpy()
            else:
                arr = hist_cpu.numpy()
            pw = pg.PlotWidget()
            pw.setBackground("w")
            pw.showGrid(x=True, y=True)

            # Improve axis label visibility
            label_style = {"color": "#000", "font-size": "12pt"}
            pw.setLabel("left", f"{y_label} - Layer {layer_id}", **label_style)
            # Compute time axis shift depending on metric semantics:
            # - State(s): if including s0, first point at 0; else first point at Δt
            # - Energy: first accumulated sample corresponds to time Δt
            # - Others (phi, g, power): first sample plotted at 0 by convention
            if sel_metric.startswith("State"):
                shift_steps = 0 if self.include_s0_chk.isChecked() else 1
            elif sel_metric.startswith("Energy"):
                shift_steps = 1
            else:
                shift_steps = 0

            # Build time axis
            T_plot = hist_cpu.shape[0]
            time_axis = [(i + shift_steps) * dt_ns for i in range(T_plot)]

            # Label shows Δt and end time for clarity
            try:
                t_end = time_axis[-1] if len(time_axis) > 0 else 0.0
            except Exception:
                t_end = 0.0
            pw.setLabel("bottom", f"Time (ns) [Δt={dt_ns:.3f} ns, end≈{t_end:.3f} ns]", **label_style)

            # Darker axis lines & larger tick fonts
            tick_font = QFont()
            tick_font.setPointSize(10)
            for ax_name in ("left", "bottom"):
                ax = pw.getAxis(ax_name)
                ax.setPen(pg.mkPen(color="k", width=1))
                ax.setTextPen("k")
                # setStyle is available in recent pyqtgraph
                with contextlib.suppress(Exception):
                    ax.setStyle(tickFont=tick_font)

            # Add legend for neuron traces and total
            show_legend = self.legend_chk.isChecked()
            show_total = self.total_chk.isChecked()

            # Always add legend so that total line is clearly labelled
            pw.addLegend()

            for neuron in range(arr.shape[1]):
                color = pg.intColor(neuron, arr.shape[1], alpha=200)
                name = f"N{neuron}" if show_legend else None
                pw.plot(
                    time_axis,
                    arr[:, neuron],
                    pen=pg.mkPen(color=color, width=2),
                    name=name,
                )

            # Plot total line if enabled
            if show_total:
                total_series = arr.sum(axis=1)
                pw.plot(
                    time_axis,
                    total_series,
                    pen=pg.mkPen(color="k", width=4),
                    name="Total",
                )

            # If seq2seq and output layer, overlay targets
            try:
                if (
                    idx == len(self._mgr.layers) - 1 and getattr(self, "_is_seq2seq", False) and self.overlay_targets_chk.isChecked() and self.input_src_cb.currentIndex() == 0  # "Dataset sample"
                ):
                    if self.dataset is not None and getattr(self, "indices", None):
                        plot_pos = int(self.sample_spin.value())
                        data_idx = self.indices[plot_pos] if self.indices else 0
                        try:
                            # Fetch label directly from loaded dataset to avoid path/group mismatches
                            _, tgt = self.dataset[data_idx]
                            y = tgt.detach().cpu().numpy() if hasattr(tgt, "detach") else np.array(tgt)
                            if y.ndim == 1:
                                y = y[:, None]
                            # Resample to match T_plot if needed
                            T_plot = arr.shape[0]
                            if y.shape[0] != T_plot and y.shape[0] > 0:
                                x_old = np.linspace(0.0, 1.0, y.shape[0])
                                x_new = np.linspace(0.0, 1.0, T_plot)
                                y = np.vstack([np.interp(x_new, x_old, y[:, j]) for j in range(y.shape[1])]).T.astype(np.float32)
                            for j in range(min(y.shape[1], arr.shape[1])):
                                pw.plot(
                                    time_axis,
                                    y[:, j],
                                    pen=pg.mkPen(color=(0, 0, 0, 180), width=2, style=Qt.PenStyle.DashLine),
                                    name=f"Target{j}",
                                )
                        except Exception:
                            pass
            except Exception:
                pass

            # overlay simulation time label unchanged
            try:
                vb = pw.getViewBox()
                rect = vb.viewRect()
                tlabel = "Sim took:\n" + _format_sim_time(elapsed)
                ti = pg.TextItem(tlabel, color=(0, 0, 0), anchor=(1, 1))
                ti.setFont(QFont("", 16))
                ti.setPos(rect.right(), rect.bottom())
                pw.addItem(ti)
            except Exception:
                pass
            self.tabs.addTab(pw, f"Layer {layer_id}")

        # Restore tab selection if possible
        if current_tab >= 0 and current_tab < self.tabs.count():
            self.tabs.setCurrentIndex(current_tab)

    def _on_task_type_changed(self) -> None:
        """Update UI visibility and internal flags when task type changes."""
        sel = self.task_type_cb.currentText().lower()
        self._is_seq2seq = sel.startswith("seq2seq")
        # Toggle class controls
        try:
            self.class_label.setVisible(not self._is_seq2seq)
            self.digit_cb.setVisible(not self._is_seq2seq)
        except Exception:
            pass
        # Recompute sample range
        with contextlib.suppress(Exception):
            self._update_sample_range()

    def _export_sequences(self) -> None:
        # For export we need raw state histories (include initial state)
        input_features_batched, _, raw_state_histories, _ = self._run_and_get_simulation_results()

        if input_features_batched is None or raw_state_histories is None:
            # Error message already shown by _run_and_get_simulation_results
            return

        # Get base filename from user
        default_filename = "exported_sequence"
        fname_base, _ = QFileDialog.getSaveFileName(
            self,
            "Save Sequences As",
            str(Path.home() / default_filename),  # Suggest a default path and base name
            "NumPy Array Files (*.npy)",  # This filter applies to the selection, not the extension logic below
        )

        if not fname_base:
            return  # User cancelled

        # Ensure the base path doesn't end with .npy yet, as we'll add suffixes.
        # QFileDialog might add it if the user types it and filter matches.
        fname_base = fname_base.removesuffix(".npy")

        input_filepath = Path(fname_base + "_input.npy")
        output_filepath = Path(fname_base + "_output.npy")
        weights_filepath = Path(fname_base + "_weights.pth")  # Path for model weights

        try:
            # Prepare input data: remove batch dim, convert to numpy
            # input_features_batched has shape [1, seq_len, input_dim]
            input_sequence = input_features_batched.squeeze(0).cpu().numpy()
            np.save(input_filepath, input_sequence)

            # Prepare output data: last layer, remove batch dim, skip initial state, convert to numpy
            # raw_state_histories is a list of tensors, last one is for the output layer
            # Each history tensor has shape [batch, seq_len+1, num_neurons]
            output_history_batched = raw_state_histories[-1].detach().cpu().numpy()
            # output_history_batched has shape [1, actual_seq_len_incl_initial_state, num_output_neurons]

            # Skip first time step (initial zero state); if only one step, use it
            if output_history_batched.shape[1] > 1:
                output_sequence = output_history_batched[0, 1:, :]  # Squeeze batch, skip initial state
            else:
                output_sequence = output_history_batched[0, :, :]  # Squeeze batch, use the single state

            np.save(output_filepath, output_sequence)

            # Save model weights
            if self._mgr.model is not None:
                torch.save(self._mgr.model.state_dict(), weights_filepath)
            else:
                # This case should ideally be caught by _run_and_get_simulation_results,
                # but as a safeguard:
                msg = "Model not available for saving weights."
                raise Exception(msg)

            show_info(
                self,
                "Export Successful",
                f"Input sequence saved to:\\n{input_filepath}\\n\\nOutput sequence saved to:\\n{output_filepath}\\n\\nModel weights saved to:\\n{weights_filepath}",
            )
        except Exception as e:
            show_error(self, "Export Error", f"Could not save sequences or weights:\\n{e!s}")

    def _on_encoding_changed(self) -> None:
        """Handle encoding selection change."""
        is_one_hot = self.encoding_cb.currentText() == "One-Hot"
        self.vocab_size_spin.setEnabled(is_one_hot)

    def _clear_input_params_layout(self) -> None:
        try:
            layout = self.input_params_layout
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    with contextlib.suppress(Exception):
                        w.setParent(None)
                    w.deleteLater()
            # Hide the box until new controls are added
            with contextlib.suppress(Exception):
                self.input_params_box.setVisible(False)
            # Process pending deletions so stale editors don't linger visually
            with contextlib.suppress(Exception):
                QApplication.processEvents()
        except Exception:
            pass

    def _compact(self, widget) -> None:
        """Set a compact size policy and width for inline parameter widgets."""
        try:
            widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            # Use a reasonable width for numeric inputs
            if isinstance(widget, (QDoubleSpinBox, QSpinBox, QLineEdit)):
                widget.setMinimumWidth(90)
                widget.setMaximumWidth(140)
        except Exception:
            pass

    def _on_input_src_changed(self) -> None:
        """Rebuild parameter controls based on selected input source."""
        self._clear_input_params_layout()
        src = self.input_src_cb.currentText()
        start_count = self.input_params_layout.count()
        # Toggle dataset-specific controls visibility (both header and options box)
        try:
            show_ds = src == "Dataset sample"
            self.dataset_opts_box.setVisible(show_ds)
            if hasattr(self, "dataset_group"):
                self.dataset_group.setVisible(show_ds)
            # Lazy load dataset when user selects "Dataset sample"
            if show_ds and not self._dataset_loaded and self.dataset_filepath:
                try:
                    p = Path(self.dataset_filepath)
                    if p.exists() and not self.dataset:
                        self._load_dataset_with_progress()
                        self._dataset_loaded = True
                except Exception:
                    pass
        except Exception:
            pass
        # Reset member refs
        self.const_val_spin = None
        self.noise_std_spin = None
        self.colored_beta_spin = None
        self.sine_freq_mhz_spin = None
        self.sine_amp_spin = None
        self.sine_phase_deg_spin = None
        self.sine_offset_spin = None
        if src == "Constant":
            self.input_params_layout.addWidget(QLabel("Value:"))
            val = float(getattr(self, "_saved_input_constant_val", 1.0) or 1.0)
            self.const_val_spin = create_double_spinbox(min_val=-1e9, max_val=1e9, decimals=6, step=0.1, default=val)
            self._compact(self.const_val_spin)
            self.const_val_spin.setToolTip("Constant input applied to all features.")
            self.input_params_layout.addWidget(self.const_val_spin)
        elif src == "Gaussian noise":
            self.input_params_layout.addWidget(QLabel("Std:"))
            val = float(self._saved_input_noise_std) if self._saved_input_noise_std is not None else 0.1
            self.noise_std_spin = create_double_spinbox(min_val=0.0, max_val=1e6, decimals=6, step=0.01, default=val)
            self._compact(self.noise_std_spin)
            self.noise_std_spin.setToolTip("Standard deviation of zero-mean Gaussian noise per feature.")
            self.input_params_layout.addWidget(self.noise_std_spin)
        elif src == "Colored noise":
            # 1/f^beta colored noise parameter: beta
            self.input_params_layout.addWidget(QLabel("Beta:"))
            val = float(self._saved_input_colored_beta) if self._saved_input_colored_beta is not None else 2.0
            self.colored_beta_spin = create_double_spinbox(min_val=0.0, max_val=10.0, decimals=3, step=0.1, default=val)
            self._compact(self.colored_beta_spin)
            self.colored_beta_spin.setToolTip("Exponent β in 1/f^β noise shaping. β=0 → white, β≈1 → pink, β≈2 → brown.")
            self.input_params_layout.addWidget(self.colored_beta_spin)
        elif src == "Sinusoid":
            # Frequency (MHz)
            self.input_params_layout.addWidget(QLabel("Freq (MHz):"))
            self.sine_freq_mhz_spin = create_double_spinbox(min_val=0.0, max_val=1e9, decimals=6, step=0.1, default=float(self._saved_input_sine_freq_mhz or 10.0))
            self._compact(self.sine_freq_mhz_spin)
            self.sine_freq_mhz_spin.setToolTip("Sine frequency in MHz. Sample rate is fs = ω_c / dt. Ensure frequency is below Nyquist (fs/2).")
            self.input_params_layout.addWidget(self.sine_freq_mhz_spin)
            # Amplitude
            self.input_params_layout.addWidget(QLabel("Amp:"))
            self.sine_amp_spin = create_double_spinbox(min_val=-1e6, max_val=1e6, decimals=6, step=0.1, default=float(self._saved_input_sine_amp or 1.0))
            self._compact(self.sine_amp_spin)
            self.sine_amp_spin.setToolTip("Sine amplitude.")
            self.input_params_layout.addWidget(self.sine_amp_spin)
            # Phase (deg)
            self.input_params_layout.addWidget(QLabel("Phase (deg):"))
            self.sine_phase_deg_spin = create_double_spinbox(min_val=-360.0, max_val=360.0, decimals=3, step=1.0, default=float(self._saved_input_sine_phase_deg or 0.0))
            self._compact(self.sine_phase_deg_spin)
            self.sine_phase_deg_spin.setToolTip("Sine phase in degrees.")
            self.input_params_layout.addWidget(self.sine_phase_deg_spin)
            # Offset
            self.input_params_layout.addWidget(QLabel("Offset:"))
            self.sine_offset_spin = create_double_spinbox(min_val=-1e6, max_val=1e6, decimals=6, step=0.1, default=float(self._saved_input_sine_offset or 0.0))
            self._compact(self.sine_offset_spin)
            self.sine_offset_spin.setToolTip("DC offset added to the sine wave.")
            self.input_params_layout.addWidget(self.sine_offset_spin)
        elif src == "Square wave":
            # Frequency (MHz)
            self.input_params_layout.addWidget(QLabel("Freq (MHz):"))
            self.sine_freq_mhz_spin = create_double_spinbox(min_val=0.0, max_val=1e9, decimals=6, step=0.1, default=float(self._saved_input_sine_freq_mhz or 10.0))
            self._compact(self.sine_freq_mhz_spin)
            self.sine_freq_mhz_spin.setToolTip("Square wave frequency in MHz. Sample rate is fs = ω_c / dt. Ensure frequency is below Nyquist (fs/2).")
            self.input_params_layout.addWidget(self.sine_freq_mhz_spin)
            # Amplitude
            self.input_params_layout.addWidget(QLabel("Amp:"))
            self.sine_amp_spin = create_double_spinbox(min_val=-1e6, max_val=1e6, decimals=6, step=0.1, default=float(self._saved_input_sine_amp or 1.0))
            self._compact(self.sine_amp_spin)
            self.sine_amp_spin.setToolTip("Square wave amplitude.")
            self.input_params_layout.addWidget(self.sine_amp_spin)
            # Phase (deg)
            self.input_params_layout.addWidget(QLabel("Phase (deg):"))
            self.sine_phase_deg_spin = create_double_spinbox(min_val=-360.0, max_val=360.0, decimals=3, step=1.0, default=float(self._saved_input_sine_phase_deg or 0.0))
            self._compact(self.sine_phase_deg_spin)
            self.sine_phase_deg_spin.setToolTip("Square wave phase in degrees.")
            self.input_params_layout.addWidget(self.sine_phase_deg_spin)
            # Offset
            self.input_params_layout.addWidget(QLabel("Offset:"))
            self.sine_offset_spin = create_double_spinbox(min_val=-1e6, max_val=1e6, decimals=6, step=0.1, default=float(self._saved_input_sine_offset or 0.0))
            self._compact(self.sine_offset_spin)
            self.sine_offset_spin.setToolTip("DC offset added to the square wave.")
            self.input_params_layout.addWidget(self.sine_offset_spin)
        else:
            # No extra params needed
            pass
        # Toggle visibility based on whether we added any parameter widgets
        try:
            has_params = self.input_params_layout.count() > start_count
            self.input_params_box.setVisible(bool(has_params))
        except Exception:
            pass

        # Update help text
        with contextlib.suppress(Exception):
            self._update_input_help(src)

    def _update_input_help(self, src: str) -> None:
        """Update the descriptive help text for the selected input source."""
        base_dt_hint = "Sampling: model dt sets sample interval Δt = dt / ω_c seconds. Nyquist frequency is fs/2 where fs = 1/Δt."
        if src == "Dataset sample":
            text = (
                "Use a sample from the loaded HDF5 dataset.\n"
                "- Feature min/max: optional scaling of inputs to [min,max] (leave 'auto' to keep original).\n"
                "- For seq2seq datasets, targets can be overlaid on the output layer."
            )
        elif src == "Constant":
            text = "Constant input applied to all features.\n- Value: amplitude of the constant signal."
        elif src == "Gaussian noise":
            text = f"Zero-mean Gaussian noise per feature.\n- Std: standard deviation of the noise.\n{base_dt_hint}"
        elif src == "Colored noise":
            text = f"Colored noise with power spectrum ∝ 1/f^β (generated via FFT shaping).\n- Beta: spectral slope (β=0 white, β≈1 pink, β≈2 brown).\n{base_dt_hint}"
        elif src == "Sinusoid":
            text = (
                "Deterministic sine wave replicated across features.\n"
                "- Freq (MHz): sine frequency in MHz.\n"
                "- Amp: amplitude.\n"
                "- Phase (deg): phase offset in degrees.\n"
                "- Offset: DC offset added to the wave.\n"
                f"{base_dt_hint}"
            )
        elif src == "Square wave":
            text = (
                "Deterministic square wave replicated across features.\n"
                "- Freq (MHz): square wave frequency in MHz.\n"
                "- Amp: amplitude (peak level).\n"
                "- Phase (deg): phase offset in degrees.\n"
                "- Offset: DC offset added to the wave.\n"
                f"{base_dt_hint}"
            )
        else:
            text = "Select an Input source to see parameters and examples."
        self.input_help_label.setText(text)

    def _on_append_zeros_toggled(self, checked: bool) -> None:
        try:
            if hasattr(self, "append_mode_cb"):
                self.append_mode_cb.setEnabled(bool(checked))
            # In total-time mode: enable time ns, disable count
            is_total = hasattr(self, "time_mode_cb") and "total" in self.time_mode_cb.currentText().lower()
            if hasattr(self, "append_zeros_time_ns"):
                self.append_zeros_time_ns.setEnabled(bool(checked and is_total))
                self.append_zeros_time_ns.setVisible(bool(checked and is_total))
            self.append_zeros_spin.setEnabled(bool(checked and not is_total))
            self.append_zeros_spin.setVisible(bool(checked and not is_total))
        except Exception:
            pass

    def _on_prepend_zeros_toggled(self, checked: bool) -> None:
        try:
            is_total = hasattr(self, "time_mode_cb") and "total" in self.time_mode_cb.currentText().lower()
            if hasattr(self, "prepend_zeros_time_ns"):
                self.prepend_zeros_time_ns.setEnabled(bool(checked and is_total))
                self.prepend_zeros_time_ns.setVisible(bool(checked and is_total))
            self.prepend_zeros_spin.setEnabled(bool(checked and not is_total))
            self.prepend_zeros_spin.setVisible(bool(checked and not is_total))
        except Exception:
            pass

    def _auto_detect_encoding(self) -> None:
        """Auto-detect encoding based on model input dimension and dataset characteristics."""
        if self._mgr.model is None:
            show_warning(self, "No Model", "Please load a model first to auto-detect encoding.")
            return

        try:
            # Get model's expected input dimension
            expected_dim = self._mgr.model.layers_config[0].params.get("dim")

            if expected_dim is None:
                show_warning(self, "Model Info", "Could not determine model input dimension.")
                return

            # Check dataset characteristics
            if self.dataset is None:
                show_warning(self, "No Dataset", "Please load a dataset first.")
                return

            # Sample a few data points to check dimensionality
            sample_data, _ = self.dataset[0]
            data_dim = sample_data.shape[-1] if sample_data.ndim > 1 else 1

            # Auto-detection logic
            if expected_dim > data_dim and expected_dim > 10:
                # Model expects high-dimensional input but data is low-dimensional
                # Likely needs one-hot encoding
                suggested_vocab = expected_dim

                show_info(
                    self,
                    "Auto-Detection Result",
                    f"Model expects {expected_dim}D input, but data is {data_dim}D.\nThis suggests one-hot encoding is needed.\n\nSetting: One-Hot encoding with vocab_size={suggested_vocab}",
                )

                self.encoding_cb.setCurrentText("One-Hot")
                self.vocab_size_spin.setValue(suggested_vocab)

            else:
                # Dimensions match or model expects low-dimensional input
                show_info(
                    self,
                    "Auto-Detection Result",
                    f"Model expects {expected_dim}D input, data is {data_dim}D.\nRaw encoding appears appropriate.",
                )

                self.encoding_cb.setCurrentText("Raw")

        except Exception as e:
            show_error(self, "Auto-Detection Error", f"Failed to auto-detect encoding:\n{e!s}")

    def _get_encoding_settings(self):
        """Get current encoding settings from GUI controls."""
        if self.encoding_cb.currentText() == "One-Hot":
            return {
                "encoding": "one_hot",
                "vocab_size": self.vocab_size_spin.value(),
            }
        return {
            "encoding": "raw",
            "vocab_size": None,
        }
