"""State Trajectory Dialog - Thin UI layer."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from ...components.collapsible_section import CollapsibleSection
from .controller import StateTrajectoryController
from .dataset_service import DatasetService
from .errors import DatasetServiceError, SimulationError
from .export import ExportService
from .jax_cache import JaxModelCache
from .model_adapter import ModelAdapter
from .plotting import PlotRenderer
from .settings import (
    Backend,
    ClassificationSettings,
    DisplaySettings,
    EncodingSettings,
    FFTSettings,
    InputKind,
    InputParams,
    Metric,
    QSettingsAdapter,
    ScalingBounds,
    StateTrajSettings,
    TaskType,
    TimeMode,
    ViewMode,
    ZeroPaddingSpec,
)
from .sim_backends import JaxRunner, TorchRunner

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.model_manager import ModelManager

from ...utils.constants import FLOAT_DECIMALS, FLOAT_MAX

# Check JAX availability
_JAX_AVAILABLE = False
try:
    import jax  # noqa: F401

    _JAX_AVAILABLE = True
except Exception:
    pass

# Constants for float validation
DT_MIN = 0.0  # dt should be non-negative (specific to this dialog)


class _DatasetLoadWorker(QObject):
    """Background worker to detect groups and load dataset.

    Emits:
      - progress(str): status message updates
      - finished(dict): result payload with groups, selection, counts
      - error(str): error message
    """

    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, dataset_service: DatasetService, path: Path) -> None:
        super().__init__()
        self._svc = dataset_service
        self._path = path

    def run(self) -> None:
        try:
            self.progress.emit("Detecting HDF5 structure...")
            groups, has_root, is_seq2seq = self._svc.detect_groups(self._path)

            # Choose default group preference
            default_group = None
            if groups:
                for preferred in ("val", "test", "train"):
                    if preferred in groups:
                        default_group = preferred
                        break
                if default_group is None:
                    default_group = groups[0]

            selected_group = default_group if default_group else (None if has_root else None)

            self.progress.emit(f"Loading data from group: {selected_group or 'root'}...")
            self._svc.load(self._path, selected_group)

            result = {
                "groups": groups,
                "has_root": has_root,
                "is_seq2seq": is_seq2seq,
                "selected_group": selected_group,
                "sample_count": self._svc.get_sample_count(),
            }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))



class StateTrajectoryDialog(QDialog):
    """State trajectory plotting dialog - thin UI layer.

    This dialog coordinates between UI widgets and the controller,
    delegating all business logic to services.
    """

    def __init__(self, parent=None, manager: ModelManager = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("State Trajectories")

        # Set a reasonable default size - wider aspect ratio, not as tall
        # This helps on platforms with different DPI/scaling (especially WSL/Linux)
        self.resize(1000, 650)

        # Ensure dialog doesn't exceed screen size
        screen = QApplication.primaryScreen()
        if screen:
            screen_geom = screen.availableGeometry()
            max_w = int(screen_geom.width() * 0.9)
            max_h = int(screen_geom.height() * 0.85)
            if self.width() > max_w:
                self.resize(max_w, self.height())
            if self.height() > max_h:
                self.resize(self.width(), max_h)

        # Services
        self._mgr = manager
        self._settings_adapter = QSettingsAdapter()
        # Use shared dataset service from manager if available (persists data across dialog opens)
        if hasattr(self._mgr, "dataset_service"):
            self._dataset_service = self._mgr.dataset_service
        else:
            self._dataset_service = DatasetService()
        self._model_adapter = ModelAdapter()
        self._plot_renderer = PlotRenderer()
        self._export_service = ExportService()

        # Backend runners (created lazily)
        self._backends = {}
        self._jax_cache = None

        # Current settings (loaded from persistence)
        self._settings = self._settings_adapter.load()

        # Controller (created after backends initialized)
        self._controller = None

        # UI initialization
        self._init_ui()
        self._apply_settings_to_ui()

        # Trigger initial input params population and visibility based on saved settings
        input_map = {
            InputKind.DATASET: "Dataset sample",
            InputKind.CONSTANT: "Constant",
            InputKind.GAUSSIAN: "Gaussian noise",
            InputKind.COLORED: "Colored noise",
            InputKind.SINE: "Sinusoid",
            InputKind.SQUARE: "Square wave",
        }
        current_src = input_map.get(self._settings.input_kind, "Dataset sample")
        self._populate_input_params(current_src)

        # Set initial visibility for dataset group and options
        show_dataset = current_src == "Dataset sample"
        self.dataset_section.setVisible(show_dataset)
        self.dataset_opts_box.setVisible(show_dataset)

        # Set initial visibility for classification group
        is_classification = self._settings.task_type == TaskType.CLASSIFICATION
        self.classification_section.setVisible(is_classification)

    def _init_ui(self) -> None:
        """Initialize UI widgets."""
        layout = QVBoxLayout(self)

        # Main tabs: Settings and Plots
        main_tabs = QTabWidget()
        layout.addWidget(main_tabs, 1)

        # Settings tab
        settings_tab = self._create_settings_tab()
        main_tabs.addTab(settings_tab, "Settings")

        # Plots tab
        plot_tab = self._create_plots_tab()
        main_tabs.addTab(plot_tab, "Plots")

    def _create_settings_tab(self) -> QWidget:
        """Create settings tab with all controls, wrapped in scroll area."""
        # Create scroll area container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        # Inner widget that holds the actual content
        content = QWidget()
        layout = QVBoxLayout(content)

        # Dataset section (store reference for visibility control)
        self.dataset_group = self._create_dataset_group()
        self.dataset_section = CollapsibleSection("Dataset")
        self.dataset_section.setContent(self.dataset_group)
        layout.addWidget(self.dataset_section)

        # Parameters section
        self.parameters_group = self._create_parameters_group()
        self.parameters_section = CollapsibleSection("Parameters & Simulation")
        self.parameters_section.setContent(self.parameters_group)
        layout.addWidget(self.parameters_section)

        # Display options
        self.display_group = self._create_display_group()
        self.display_section = CollapsibleSection("Display Options")
        self.display_section.setContent(self.display_group)
        layout.addWidget(self.display_section)

        # Classification visualization (only shown for classification tasks)
        self.classification_group = self._create_classification_group()
        self.classification_section = CollapsibleSection("Classification Visualization")
        self.classification_section.setContent(self.classification_group)
        layout.addWidget(self.classification_section)

        # Actions
        self.actions_group = self._create_actions_group()
        self.actions_section = CollapsibleSection("Actions", collapsed=True)
        self.actions_section.setContent(self.actions_group)
        layout.addWidget(self.actions_section)

        layout.addStretch(1)

        scroll.setWidget(content)
        return scroll

    def _create_plots_tab(self) -> QWidget:
        """Create plots tab with controls and plot area."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Top controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Metric:"))
        self.metric_cb = QComboBox()
        self.metric_cb.addItems(["State (s)", "Flux (phi)", "Non-Linearity (g)", "Power", "Energy"])
        ctrl.addWidget(self.metric_cb)

        ctrl.addWidget(QLabel("Backend:"))
        self.backend_cb = QComboBox()
        self.backend_cb.addItems(["Torch", "JAX"])
        if not _JAX_AVAILABLE:
            self.backend_cb.setCurrentText("Torch")
            self.backend_cb.setEnabled(False)
            self.backend_cb.setToolTip("JAX not available in this environment")
        ctrl.addWidget(self.backend_cb)

        # FFT View mode
        ctrl.addWidget(QLabel("View:"))
        self.view_mode_cb = QComboBox()
        self.view_mode_cb.addItems(["Time Only", "Frequency Only", "Time + Frequency", "Waterfall"])
        self.view_mode_cb.setToolTip("Select time domain, frequency domain (FFT), or combined view")
        self.view_mode_cb.currentTextChanged.connect(self._on_view_mode_changed)
        ctrl.addWidget(self.view_mode_cb)

        # FFT Settings button
        self.fft_settings_btn = QPushButton("FFT Settings...")
        self.fft_settings_btn.clicked.connect(self._open_fft_settings)
        ctrl.addWidget(self.fft_settings_btn)

        self.plot_btn = QPushButton("Refresh Plot")
        self.plot_btn.clicked.connect(self._on_plot)
        ctrl.addWidget(self.plot_btn)

        # FFT info label
        self.fft_info_label = QLabel("")
        self.fft_info_label.setStyleSheet("color: #666; font-size: 9pt;")
        ctrl.addWidget(self.fft_info_label)

        ctrl.addStretch(1)

        layout.addLayout(ctrl)

        # Plot tabs
        self.tabs = pg.QtWidgets.QTabWidget()
        self.tabs.setUsesScrollButtons(True)  # Enable horizontal scrolling for many layers
        # Compact tabs to fit more layers
        self.tabs.setStyleSheet("QTabBar::tab { min-width: 60px; padding: 4px; max-width: 150px; }")
        layout.addWidget(self.tabs, 1)

        return tab

    def _create_dataset_group(self) -> QWidget:
        """Create dataset configuration group."""
        group = QWidget()
        layout = QVBoxLayout(group)

        # File path row
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Dataset File:"))
        self.dataset_edit = QLineEdit()
        path_row.addWidget(self.dataset_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dataset)
        path_row.addWidget(browse_btn)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_dataset_with_progress)
        path_row.addWidget(load_btn)
        layout.addLayout(path_row)

        # Task type row
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task Type:"))
        self.task_type_cb = QComboBox()
        self.task_type_cb.addItems(["Classification", "Seq2Seq (regression)"])
        self.task_type_cb.currentTextChanged.connect(self._on_task_type_changed)
        task_row.addWidget(self.task_type_cb)
        task_row.addStretch()
        layout.addLayout(task_row)

        # Group selector
        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("Data Group:"))
        self.group_cb = QComboBox()
        self.group_cb.currentTextChanged.connect(self._on_group_changed)
        group_row.addWidget(self.group_cb)
        group_row.addStretch()
        layout.addLayout(group_row)

        # Sample selection row
        sample_row = QHBoxLayout()
        self.class_label = QLabel("Class ID:")
        sample_row.addWidget(self.class_label)
        self.digit_cb = QComboBox()
        self.digit_cb.setEditable(True)
        # Avoid triggering expensive label scans on every keystroke
        with contextlib.suppress(Exception):
            self.digit_cb.currentIndexChanged.connect(self._on_class_changed)
            if self.digit_cb.lineEdit() is not None:
                self.digit_cb.lineEdit().editingFinished.connect(self._on_class_changed)
        sample_row.addWidget(self.digit_cb)
        self.sample_spin_label = QLabel("Sample Index:")
        sample_row.addWidget(self.sample_spin_label)
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(0, 999999)  # Large default max
        sample_row.addWidget(self.sample_spin)
        self.new_btn = QPushButton("New Random Sample")
        self.new_btn.clicked.connect(self._random_sample)
        sample_row.addWidget(self.new_btn)
        sample_row.addStretch()
        layout.addLayout(sample_row)

        return group

    def _create_parameters_group(self) -> QWidget:
        """Create simulation parameters group."""
        group = QWidget()
        layout = QVBoxLayout(group)

        # Input source row
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Input source:"))
        self.input_src_cb = QComboBox()
        self.input_src_cb.addItems(["Dataset sample", "Constant", "Gaussian noise", "Colored noise", "Sinusoid", "Square wave"])
        self.input_src_cb.currentTextChanged.connect(self._on_input_src_changed)
        input_row.addWidget(self.input_src_cb)
        input_row.addStretch()
        layout.addLayout(input_row)

        # Dataset-specific options
        self.dataset_opts_box = self._create_dataset_options()
        layout.addWidget(self.dataset_opts_box)

        # Input generator params
        self.input_params_box = self._create_input_params_box()
        layout.addWidget(self.input_params_box)

        # Sequence length
        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("Seq Length:"))
        self.seq_spin = QSpinBox()
        self.seq_spin.setRange(1, 100000)
        self.seq_spin.setValue(100)
        self.seq_spin.valueChanged.connect(self._update_dt_info)
        seq_row.addWidget(self.seq_spin)
        seq_row.addStretch()
        layout.addLayout(seq_row)

        # Time settings
        layout.addLayout(self._create_time_controls())

        # Padding controls
        layout.addLayout(self._create_padding_controls())

        # Criticality Analysis
        self.criticality_chk = QCheckBox("Calculate Criticality Metrics (Branching Ratio, Susceptibility)")
        self.criticality_chk.setToolTip("Computes dynamical metrics to check if the network is at the 'Edge of Chaos'")
        layout.addWidget(self.criticality_chk)

        return group

    def _create_dataset_options(self) -> QWidget:
        """Create dataset-specific options widget."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Feature scaling row
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Feature min:"))
        self.mel_min_edit = QLineEdit()
        self.mel_min_edit.setPlaceholderText("auto")
        scale_row.addWidget(self.mel_min_edit)
        scale_row.addWidget(QLabel("Feature max:"))
        self.mel_max_edit = QLineEdit()
        self.mel_max_edit.setPlaceholderText("auto")
        scale_row.addWidget(self.mel_max_edit)
        scale_row.addStretch()
        layout.addLayout(scale_row)

        # Encoding row
        enc_row = QHBoxLayout()
        enc_row.addWidget(QLabel("Input Encoding:"))
        self.encoding_cb = QComboBox()
        self.encoding_cb.addItems(["Raw", "One-Hot"])
        enc_row.addWidget(self.encoding_cb)
        enc_row.addStretch()
        layout.addLayout(enc_row)

        widget.setLayout(layout)
        return widget

    def _create_input_params_box(self) -> QGroupBox:
        """Create input generator parameters box (populated dynamically)."""
        box = QGroupBox("Input Generator Parameters")
        box.setFlat(True)
        self.input_params_layout = QHBoxLayout(box)
        box.setVisible(False)
        return box

    def _create_time_controls(self) -> QHBoxLayout:
        """Create time mode and dt controls."""
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Time mode:"))
        self.time_mode_cb = QComboBox()
        self.time_mode_cb.addItems(["Specify dt", "Specify total time (ns)"])
        self.time_mode_cb.currentTextChanged.connect(self._on_time_mode_changed)
        layout.addWidget(self.time_mode_cb)

        self.dt_label = QLabel("dt (dimensionless):")
        layout.addWidget(self.dt_label)
        self.dt_spin = QLineEdit()
        validator_dt = QDoubleValidator(DT_MIN, FLOAT_MAX, FLOAT_DECIMALS)
        validator_dt.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.dt_spin.setValidator(validator_dt)
        self.dt_spin.setText("37")
        self.dt_spin.textChanged.connect(self._update_dt_info)
        layout.addWidget(self.dt_spin)

        self.total_time_label = QLabel("Total time (ns):")
        layout.addWidget(self.total_time_label)
        self.total_time_ns_spin = QLineEdit()
        validator_tt = QDoubleValidator(0.0, FLOAT_MAX, FLOAT_DECIMALS)
        validator_tt.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.total_time_ns_spin.setValidator(validator_tt)
        self.total_time_ns_spin.textChanged.connect(self._update_dt_info)
        layout.addWidget(self.total_time_ns_spin)

        # Info label for dt calculations (small print on the right)
        self.dt_info_label = QLabel()
        self.dt_info_label.setStyleSheet("color: #666; font-size: 10pt;")
        layout.addWidget(self.dt_info_label, 1)

        # Also track seq_len for calculations (will be set after seq_spin is created)
        self._seq_len = 100  # Default

        layout.addStretch(0)
        return layout

    def _create_padding_controls(self) -> QHBoxLayout:
        """Create padding controls."""
        layout = QHBoxLayout()

        # Prepend
        self.prepend_zeros_chk = QCheckBox("Prepend zeros")
        self.prepend_zeros_chk.toggled.connect(self._on_prepend_zeros_toggled)
        layout.addWidget(self.prepend_zeros_chk)
        layout.addWidget(QLabel("Count:"))
        self.prepend_zeros_spin = QSpinBox()
        self.prepend_zeros_spin.setRange(0, 100000)
        self.prepend_zeros_spin.setEnabled(False)
        layout.addWidget(self.prepend_zeros_spin)
        layout.addWidget(QLabel("Time (ns):"))
        self.prepend_zeros_time_ns = QLineEdit()
        validator_prepend = QDoubleValidator(0.0, FLOAT_MAX, FLOAT_DECIMALS)
        validator_prepend.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.prepend_zeros_time_ns.setValidator(validator_prepend)
        self.prepend_zeros_time_ns.setEnabled(False)
        self.prepend_zeros_time_ns.setVisible(False)
        layout.addWidget(self.prepend_zeros_time_ns)

        # Append
        layout.addSpacing(16)
        self.append_zeros_chk = QCheckBox("Append zeros")
        self.append_zeros_chk.toggled.connect(self._on_append_zeros_toggled)
        layout.addWidget(self.append_zeros_chk)
        layout.addWidget(QLabel("Count:"))
        self.append_zeros_spin = QSpinBox()
        self.append_zeros_spin.setRange(0, 100000)
        self.append_zeros_spin.setEnabled(False)
        layout.addWidget(self.append_zeros_spin)
        layout.addWidget(QLabel("Time (ns):"))
        self.append_zeros_time_ns = QLineEdit()
        validator_append = QDoubleValidator(0.0, FLOAT_MAX, FLOAT_DECIMALS)
        validator_append.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.append_zeros_time_ns.setValidator(validator_append)
        self.append_zeros_time_ns.setEnabled(False)
        self.append_zeros_time_ns.setVisible(False)
        layout.addWidget(self.append_zeros_time_ns)
        layout.addWidget(QLabel("Mode:"))
        self.append_mode_cb = QComboBox()
        self.append_mode_cb.addItem("Zeros", "zeros")
        self.append_mode_cb.addItem("Hold last", "hold_last")
        self.append_mode_cb.setEnabled(False)
        layout.addWidget(self.append_mode_cb)

        layout.addStretch()
        return layout

    def _create_display_group(self) -> QWidget:
        """Create display options group."""
        group = QWidget()
        layout = QVBoxLayout(group)

        self.legend_chk = QCheckBox("Show Legend")
        layout.addWidget(self.legend_chk)

        self.total_chk = QCheckBox("Show Total Line")
        layout.addWidget(self.total_chk)

        self.include_s0_chk = QCheckBox("Include t=0 (initial state) for s(t)")
        self.include_s0_chk.setChecked(True)
        layout.addWidget(self.include_s0_chk)

        self.overlay_targets_chk = QCheckBox("Show expected class/target on output layer")
        self.overlay_targets_chk.setChecked(True)
        self.overlay_targets_chk.setToolTip("For classification: shows expected class label\nFor seq2seq: overlays target sequence as dashed lines")
        layout.addWidget(self.overlay_targets_chk)

        return group

    def _create_classification_group(self) -> QWidget:
        """Create classification visualization settings group."""
        group = QWidget()
        layout = QVBoxLayout(group)

        # Enable classification mode checkbox
        self.classification_enabled_chk = QCheckBox("Enable multi-sample classification mode")
        self.classification_enabled_chk.setToolTip(
            "When enabled, concatenates multiple random samples from different classes\n"
            "with state carryover. Shows sample boundaries and accuracy markers."
        )
        self.classification_enabled_chk.stateChanged.connect(self._on_classification_toggled)
        layout.addWidget(self.classification_enabled_chk)

        # Number of samples
        samples_row = QHBoxLayout()
        samples_row.addWidget(QLabel("Number of samples:"))
        self.classification_num_samples_spin = QSpinBox()
        self.classification_num_samples_spin.setMinimum(1)
        self.classification_num_samples_spin.setMaximum(20)
        self.classification_num_samples_spin.setValue(3)
        self.classification_num_samples_spin.setEnabled(False)
        samples_row.addWidget(self.classification_num_samples_spin)
        samples_row.addStretch()
        layout.addLayout(samples_row)

        # Time pooling method
        pooling_row = QHBoxLayout()
        pooling_row.addWidget(QLabel("Time pooling:"))
        self.classification_pooling_cb = QComboBox()
        self.classification_pooling_cb.addItems(["Max over time", "Final timestep"])
        self.classification_pooling_cb.setEnabled(False)
        pooling_row.addWidget(self.classification_pooling_cb)
        pooling_row.addStretch()
        layout.addLayout(pooling_row)

        # Show accuracy checkbox
        self.classification_show_accuracy_chk = QCheckBox("Show accuracy markers (✓/✗)")
        self.classification_show_accuracy_chk.setChecked(True)
        self.classification_show_accuracy_chk.setEnabled(False)
        layout.addWidget(self.classification_show_accuracy_chk)

        return group

    def _create_actions_group(self) -> QWidget:
        """Create action buttons group."""
        group = QWidget()
        layout = QHBoxLayout(group)

        self.export_btn = QPushButton("Export Sequences")
        self.export_btn.clicked.connect(self._export_sequences)
        layout.addWidget(self.export_btn)

        return group

    def _apply_settings_to_ui(self) -> None:
        """Apply loaded settings to UI widgets."""
        s = self._settings

        # Dataset
        self.dataset_edit.setText(s.dataset_path)
        self.task_type_cb.setCurrentText("Seq2Seq (regression)" if s.task_type == TaskType.SEQ2SEQ else "Classification")

        # Input
        input_map = {
            InputKind.DATASET: "Dataset sample",
            InputKind.CONSTANT: "Constant",
            InputKind.GAUSSIAN: "Gaussian noise",
            InputKind.COLORED: "Colored noise",
            InputKind.SINE: "Sinusoid",
            InputKind.SQUARE: "Square wave",
        }
        self.input_src_cb.setCurrentText(input_map.get(s.input_kind, "Dataset sample"))

        # Metric & backend
        metric_map = {
            Metric.STATE: "State (s)",
            Metric.PHI: "Flux (phi)",
            Metric.G: "Non-Linearity (g)",
            Metric.POWER: "Power",
            Metric.ENERGY: "Energy",
        }
        self.metric_cb.setCurrentText(metric_map.get(s.metric, "State (s)"))
        self.backend_cb.setCurrentText("JAX" if s.backend == Backend.JAX else "Torch")

        # FFT view mode
        view_mode_map = {
            ViewMode.TIME_ONLY: "Time Only",
            ViewMode.FREQUENCY_ONLY: "Frequency Only",
            ViewMode.SPLIT: "Time + Frequency",
            ViewMode.WATERFALL: "Waterfall",
        }
        self.view_mode_cb.setCurrentText(view_mode_map.get(s.fft.view_mode, "Time Only"))

        # Sequence
        self.seq_spin.setValue(s.seq_len)

        # Time
        time_mode_map = {TimeMode.DT: "Specify dt", TimeMode.TOTAL_NS: "Specify total time (ns)"}
        self.time_mode_cb.setCurrentText(time_mode_map.get(s.time_mode, "Specify dt"))
        self.dt_spin.setText(str(s.dt))
        self.total_time_ns_spin.setText(str(s.total_time_ns) if s.total_time_ns > 0 else "")

        # Padding
        self.prepend_zeros_chk.setChecked(s.prepend.enabled)
        self.prepend_zeros_spin.setValue(s.prepend.count_steps)
        self.prepend_zeros_time_ns.setText(str(s.prepend.time_ns) if s.prepend.time_ns > 0 else "")
        self.append_zeros_chk.setChecked(s.append.enabled)
        self.append_zeros_spin.setValue(s.append.count_steps)
        self.append_zeros_time_ns.setText(str(s.append.time_ns) if s.append.time_ns > 0 else "")
        idx = self.append_mode_cb.findData(s.append.mode)
        if idx >= 0:
            self.append_mode_cb.setCurrentIndex(idx)
        # Update visibility/enabled state based on time mode
        self._update_padding_controls_visibility()

        # Encoding
        self.encoding_cb.setCurrentText("One-Hot" if s.encoding.mode == "one_hot" else "Raw")

        # Scaling
        if s.scaling.min_val is not None:
            self.mel_min_edit.setText(str(s.scaling.min_val))
        if s.scaling.max_val is not None:
            self.mel_max_edit.setText(str(s.scaling.max_val))

        # Display
        self.legend_chk.setChecked(s.display.show_legend)
        self.total_chk.setChecked(s.display.show_total)
        self.include_s0_chk.setChecked(s.display.include_s0)
        self.overlay_targets_chk.setChecked(s.display.overlay_targets)

        # Classification settings
        self.classification_enabled_chk.setChecked(s.classification.enabled)
        self.classification_num_samples_spin.setValue(s.classification.num_samples)
        pooling_map = {"max": "Max over time", "final": "Final timestep"}
        self.classification_pooling_cb.setCurrentText(pooling_map.get(s.classification.pooling_method, "Max over time"))
        self.classification_show_accuracy_chk.setChecked(s.classification.show_accuracy)
        self._on_classification_toggled()  # Update enabled state

        # Sample selection
        if s.class_id is not None:
            self.digit_cb.setCurrentText(str(s.class_id))
        self.sample_spin.setValue(s.sample_index)

        # Update dt info label
        self._update_dt_info()

        # Update FFT info label
        self._update_fft_info()

        # Criticality
        self.criticality_chk.setChecked(s.calculate_criticality)

    def _ui_to_settings(self) -> StateTrajSettings:
        """Build StateTrajSettings from current UI state."""
        # Parse metric
        metric_map = {
            "State (s)": Metric.STATE,
            "Flux (phi)": Metric.PHI,
            "Non-Linearity (g)": Metric.G,
            "Power": Metric.POWER,
            "Energy": Metric.ENERGY,
        }
        metric = metric_map.get(self.metric_cb.currentText(), Metric.STATE)

        # Parse backend
        backend = Backend.JAX if self.backend_cb.currentText() == "JAX" else Backend.TORCH

        # Parse input kind
        input_map = {
            "Dataset sample": InputKind.DATASET,
            "Constant": InputKind.CONSTANT,
            "Gaussian noise": InputKind.GAUSSIAN,
            "Colored noise": InputKind.COLORED,
            "Sinusoid": InputKind.SINE,
            "Square wave": InputKind.SQUARE,
        }
        input_kind = input_map.get(self.input_src_cb.currentText(), InputKind.DATASET)

        # Parse time mode
        time_mode = TimeMode.TOTAL_NS if "total" in self.time_mode_cb.currentText().lower() else TimeMode.DT

        # Parse task type
        task_type = TaskType.SEQ2SEQ if "seq2seq" in self.task_type_cb.currentText().lower() else TaskType.CLASSIFICATION

        # Get input params from widgets
        input_params = self._get_input_params_from_ui()

        # Parse dt and total_time_ns
        try:
            dt = float(self.dt_spin.text()) if self.dt_spin.text() else 37.0
        except ValueError:
            dt = 37.0

        try:
            total_time_ns = float(self.total_time_ns_spin.text()) if self.total_time_ns_spin.text() else 0.0
        except ValueError:
            total_time_ns = 0.0

        # Parse scaling bounds
        scaling_min = None
        scaling_max = None
        try:
            min_txt = self.mel_min_edit.text().strip()
            if min_txt and min_txt.lower() != "auto":
                scaling_min = float(min_txt)
        except ValueError:
            pass
        try:
            max_txt = self.mel_max_edit.text().strip()
            if max_txt and max_txt.lower() != "auto":
                scaling_max = float(max_txt)
        except ValueError:
            pass

        # Parse class ID
        try:
            class_id = int(self.digit_cb.currentText())
        except ValueError:
            class_id = 0

        return StateTrajSettings(
            dataset_path=self.dataset_edit.text().strip(),
            group=self.group_cb.currentData() if self.group_cb.count() > 0 else None,
            task_type=task_type,
            metric=metric,
            backend=backend,
            input_kind=input_kind,
            seq_len=self.seq_spin.value(),
            time_mode=time_mode,
            dt=dt,
            total_time_ns=total_time_ns,
            prepend=ZeroPaddingSpec(
                enabled=self.prepend_zeros_chk.isChecked(),
                count_steps=self.prepend_zeros_spin.value(),
                time_ns=self._get_prepend_time_ns(),
                mode="zeros",
            ),
            append=ZeroPaddingSpec(
                enabled=self.append_zeros_chk.isChecked(),
                count_steps=self.append_zeros_spin.value(),
                time_ns=self._get_append_time_ns(),
                mode=self.append_mode_cb.currentData(),
            ),
            encoding=EncodingSettings(
                mode="one_hot" if self.encoding_cb.currentText() == "One-Hot" else "raw",
                vocab_size=self._get_vocab_size_from_model(),
            ),
            scaling=ScalingBounds(
                min_val=scaling_min,
                max_val=scaling_max,
            ),
            display=DisplaySettings(
                show_legend=self.legend_chk.isChecked(),
                show_total=self.total_chk.isChecked(),
                include_s0=self.include_s0_chk.isChecked(),
                overlay_targets=self.overlay_targets_chk.isChecked(),
            ),
            classification=ClassificationSettings(
                enabled=self.classification_enabled_chk.isChecked(),
                num_samples=self.classification_num_samples_spin.value(),
                pooling_method="max" if self.classification_pooling_cb.currentText() == "Max over time" else "final",
                show_accuracy=self.classification_show_accuracy_chk.isChecked(),
            ),
            input_params=input_params,
            fft=self._get_fft_settings_from_ui(),
            class_id=class_id,
            sample_index=self.sample_spin.value(),
            calculate_criticality=self.criticality_chk.isChecked(),
        )

    def _get_input_params_from_ui(self) -> InputParams:
        """Extract input generator params from UI widgets."""
        # Start with current settings as base
        params = InputParams(
            constant_val=self._settings.input_params.constant_val,
            noise_std=self._settings.input_params.noise_std,
            colored_beta=self._settings.input_params.colored_beta,
            sine_freq_mhz=self._settings.input_params.sine_freq_mhz,
            sine_amp=self._settings.input_params.sine_amp,
            sine_phase_deg=self._settings.input_params.sine_phase_deg,
            sine_offset=self._settings.input_params.sine_offset,
        )

        # Read from dynamically created widgets if they exist
        if hasattr(self, "_input_param_widgets") and self._input_param_widgets:
            for key, widget in self._input_param_widgets.items():
                if isinstance(widget, QLineEdit):
                    try:
                        value = float(widget.text()) if widget.text() else getattr(params, key)
                        setattr(params, key, value)
                    except ValueError:
                        pass  # Keep default if parse fails

        return params

    def _get_fft_settings_from_ui(self) -> FFTSettings:
        """Extract FFT settings from UI (view mode from dropdown, rest from self._settings)."""
        # Parse view mode from dropdown
        view_mode_map = {
            "Time Only": ViewMode.TIME_ONLY,
            "Frequency Only": ViewMode.FREQUENCY_ONLY,
            "Time + Frequency": ViewMode.SPLIT,
            "Waterfall": ViewMode.WATERFALL,
        }
        view_mode = view_mode_map.get(self.view_mode_cb.currentText(), ViewMode.TIME_ONLY)

        # Use current FFT settings but update view mode from UI
        return FFTSettings(
            view_mode=view_mode,
            window_function=self._settings.fft.window_function,
            aggregation_mode=self._settings.fft.aggregation_mode,
            freq_min_mhz=self._settings.fft.freq_min_mhz,
            freq_max_mhz=self._settings.fft.freq_max_mhz,
            y_scale=self._settings.fft.y_scale,
            remove_dc=self._settings.fft.remove_dc,
            normalize=self._settings.fft.normalize,
            show_peaks=self._settings.fft.show_peaks,
            num_peaks=self._settings.fft.num_peaks,
        )

    def _get_vocab_size_from_model(self) -> int:
        """Auto-detect vocab size from model's input dimension."""
        if self._mgr.model is None:
            return 65  # Default fallback

        try:
            expected_dim = self._mgr.model.layers_config[0].params.get("dim")
            if expected_dim is not None and expected_dim > 0:
                return expected_dim
        except (AttributeError, IndexError, KeyError):
            pass

        return 65  # Default fallback

    def _get_backends(self):
        """Get or create backend runners dict."""
        if not self._backends:
            # Torch backend
            self._backends[Backend.TORCH] = TorchRunner(self._mgr.model, self._model_adapter)

            # JAX backend (if available)
            if _JAX_AVAILABLE and self._mgr.model:
                if not self._jax_cache:
                    self._jax_cache = JaxModelCache(self._mgr.model)
                self._backends[Backend.JAX] = JaxRunner(self._jax_cache, self._model_adapter)

        return self._backends

    def _get_controller(self) -> StateTrajectoryController:
        """Get or create controller."""
        if not self._controller:
            self._controller = StateTrajectoryController(
                self._mgr,
                self._dataset_service,
                self._get_backends(),
                self._model_adapter,
            )
        return self._controller

    def _on_plot(self) -> None:
        """Handle plot button click."""
        if self._mgr.model is None:
            QMessageBox.warning(self, "No model", "Build or load a model first.")
            return

        # Build settings from UI
        settings = self._ui_to_settings()

        # Fail fast if dataset input requested but dataset not loaded
        if settings.input_kind == InputKind.DATASET and not self._dataset_service.is_loaded():
            QMessageBox.warning(self, "No dataset loaded", "Please load a dataset before plotting with 'Dataset sample' input.")
            return

        # Save settings
        self._settings_adapter.save(settings)

        # Run simulation via controller
        try:
            controller = self._get_controller()

            # Check if classification mode is enabled
            if settings.classification.enabled and settings.task_type == TaskType.CLASSIFICATION:
                # Use classification mode
                tb, classification_result, elapsed = controller.run_classification(settings)
                # Convert ClassificationResult to format expected by plotting
                input_features = classification_result.input_features
                metric_h = classification_result.metric_histories
                raw_h = classification_result.raw_state_histories
                # Store classification result for plotting
                classification_data = classification_result
                # Compute FFT for classification mode too
                fft_data = None
                if settings.fft.view_mode != ViewMode.TIME_ONLY:
                    fft_data = controller._compute_fft_data(raw_h, tb.step_ns, settings.fft)
            else:
                # Standard single-sample mode
                tb, input_features, metric_h, raw_h, elapsed, fft_data, crit_metrics = controller.run(settings)
                classification_data = None

                if crit_metrics:
                    # Display metrics in a non-blocking way or a message box
                    msg = "Criticality Analysis Results\n\n"
                    msg += f"Branching Ratio (σ): {crit_metrics.branching_ratio:.4f}\n"
                    msg += f"Susceptibility (χ): {crit_metrics.susceptibility:.4f}\n\n"

                    if crit_metrics.local_expansion_rate is not None:
                        msg += f"Local Expansion Rate (Lyapunov): {crit_metrics.local_expansion_rate.item():.4f}\n"

                    msg += f"Avalanche Size Exp (α): {crit_metrics.avalanche_exponent_size}\n"
                    msg += f"Avalanche Dur Exp (β): {crit_metrics.avalanche_exponent_duration}\n"
                    msg += f"Avalanche Count: {crit_metrics.avalanche_count}\n\n"
                    msg += f"Is Critical? {'Yes' if crit_metrics.is_critical else 'No'}"

                    # Use information box (modal but standard for results)
                    QMessageBox.information(self, "Criticality Metrics", msg)

        except SimulationError as e:
            import logging
            import traceback

            logger = logging.getLogger(__name__)
            logger.error("Simulation error in dialog", exc_info=True)
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Simulation Error", str(e))
            return
        except Exception as e:
            import logging
            import traceback

            logger = logging.getLogger(__name__)
            logger.error("Unexpected error in dialog", exc_info=True)
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Unexpected Error", f"Simulation failed: {e}")
            return

        # Render plots
        try:
            is_seq2seq = settings.task_type == TaskType.SEQ2SEQ
            data_index = None
            dataset = None
            if settings.input_kind == InputKind.DATASET and self._dataset_service.is_loaded():
                indices = self._dataset_service.get_indices_for_class(settings.class_id, settings.task_type)
                if indices:
                    sample_idx = min(settings.sample_index, len(indices) - 1)
                    data_index = indices[sample_idx]
                    dataset = self._dataset_service.get_dataset()

            self._plot_renderer.render_layer_plots(
                metric_h,
                raw_h,
                settings.metric,
                tb,
                elapsed,
                self._mgr.layers,
                self.tabs,
                settings.display,
                fft_settings=settings.fft,
                fft_data=fft_data,
                dataset=dataset,
                data_index=data_index,
                is_seq2seq=is_seq2seq,
                prepend_spec=settings.prepend,
                append_spec=settings.append,
                time_mode=settings.time_mode,
                classification_result=classification_data,
                classification_settings=settings.classification if classification_data is not None else None,
            )
        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"Failed to render plots: {e}")

    def _export_sequences(self) -> None:
        """Handle export button click."""
        # Re-run simulation to get clean sequences
        settings = self._ui_to_settings()

        try:
            controller = self._get_controller()
            tb, input_features, metric_h, raw_h, elapsed, _ = controller.run(settings)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to run simulation: {e}")
            return

        # Get output sequence (last layer, skip initial state)
        output_seq = raw_h[-1][0, 1:, :].detach()  # [seq_len, dim]
        input_seq = input_features[0].detach()  # [seq_len, dim]

        # Get save path from user
        default_path = Path.home() / "exported_sequence"
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save Sequences As",
            str(default_path),
            "NumPy Archive (*.npz)",
        )

        if not fname:
            return

        base_path = Path(fname).with_suffix("")  # Remove extension

        try:
            paths = self._export_service.export_sequences(
                input_seq,
                output_seq,
                self._mgr.model,
                settings,
                tb,
                base_path,
            )

            msg = "Export successful!\n\n"
            msg += f"Sequences: {paths[0].name}\n"
            msg += f"Weights: {paths[1].name}\n"
            msg += f"Metadata: {paths[2].name}\n"
            msg += f"Info: {paths[3].name}"

            QMessageBox.information(self, "Export Successful", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    def _browse_dataset(self) -> None:
        """Handle browse button click."""
        # Prefer last-used directory or current text to avoid expensive home scans
        start_dir = Path(self.dataset_edit.text().strip())
        if not start_dir.exists():
            try:
                saved = Path(self._settings.dataset_path)
                start_dir = saved if saved.exists() else saved.parent
            except Exception:
                start_dir = Path.home()
        if start_dir.is_file():
            start_dir = start_dir.parent

        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select dataset file",
            str(start_dir if start_dir.exists() else Path.home()),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if fname:
            self.dataset_edit.setText(fname)

    def _load_dataset_with_progress(self) -> None:
        """Load dataset with progress dialog (non-blocking via QThread)."""
        path_str = self.dataset_edit.text().strip()
        if not path_str:
            return

        path = Path(path_str)
        if not path.exists():
            QMessageBox.warning(self, "File Not Found", f"Dataset file not found: {path}")
            return

        # Prevent concurrent loads
        try:
            if hasattr(self, "_loader_thread") and self._loader_thread is not None and self._loader_thread.isRunning():
                return
        except Exception:
            pass

        # Progress dialog (indeterminate)
        self._loader_progress = QProgressDialog("Loading dataset...", "Cancel", 0, 0, self)
        self._loader_progress.setWindowTitle("Loading Dataset")
        self._loader_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._loader_progress.show()
        QApplication.processEvents()

        # Disable UI controls during load
        with contextlib.suppress(Exception):
            self.dataset_section.setEnabled(False)
            self.dataset_group.setEnabled(False)
            self.plot_btn.setEnabled(False)

        # Worker in background thread
        self._loader_thread = QThread(self)
        self._loader_worker = _DatasetLoadWorker(self._dataset_service, path)
        self._loader_worker.moveToThread(self._loader_thread)

        # Wire signals
        self._loader_worker.progress.connect(self._on_loader_progress)
        self._loader_worker.finished.connect(self._on_loader_finished)
        self._loader_worker.error.connect(self._on_loader_error)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_progress.canceled.connect(self._cleanup_loader)

        self._loader_thread.start()

    def _on_loader_progress(self, message: str) -> None:
        if hasattr(self, "_loader_progress") and self._loader_progress is not None:
            self._loader_progress.setLabelText(message)
            QApplication.processEvents()

    def _on_loader_finished(self, result: dict) -> None:
        # Update group combo without triggering change handler repeatedly
        self.group_cb.blockSignals(True)
        try:
            self.group_cb.clear()
            if result.get("has_root"):
                self.group_cb.addItem("(root level)", None)
            for grp in result.get("groups", []):
                self.group_cb.addItem(grp, grp)

            # Select detected/default group
            selected_group = result.get("selected_group")
            idx = self.group_cb.findData(selected_group) if selected_group is not None else self.group_cb.findData(None)
            if idx >= 0:
                self.group_cb.setCurrentIndex(idx)
        finally:
            self.group_cb.blockSignals(False)

        # Update task type and classes
        is_seq2seq = bool(result.get("is_seq2seq", False))
        self.task_type_cb.setCurrentText("Seq2Seq (regression)" if is_seq2seq else "Classification")
        self._populate_digit_classes()

        QMessageBox.information(self, "Dataset Loaded", f"Loaded {int(result.get('sample_count', 0))} samples")
        self._cleanup_loader()

        # Re-enable controls
        with contextlib.suppress(Exception):
            self.dataset_section.setEnabled(True)
            self.dataset_group.setEnabled(True)
            self.plot_btn.setEnabled(True)

    def _on_loader_error(self, message: str) -> None:
        QMessageBox.critical(self, "Dataset Error", message)
        self._cleanup_loader()
        with contextlib.suppress(Exception):
            self.dataset_section.setEnabled(True)
            self.dataset_group.setEnabled(True)
            self.plot_btn.setEnabled(True)

    def _cleanup_loader(self) -> None:
        # Close progress and teardown thread/worker
        if hasattr(self, "_loader_progress") and self._loader_progress is not None:
            with contextlib.suppress(Exception):
                self._loader_progress.close()
            self._loader_progress = None
        if hasattr(self, "_loader_worker") and self._loader_worker is not None:
            with contextlib.suppress(Exception):
                self._loader_worker.deleteLater()
            self._loader_worker = None
        if hasattr(self, "_loader_thread") and self._loader_thread is not None:
            with contextlib.suppress(Exception):
                self._loader_thread.quit()
                self._loader_thread.wait()
                self._loader_thread.deleteLater()
            self._loader_thread = None

    def _populate_digit_classes(self) -> None:
        """Populate class selector from dataset."""
        classes = self._dataset_service.get_available_classes()
        self.digit_cb.clear()
        for c in classes:
            self.digit_cb.addItem(str(c))

        # Update sample range for first class
        if classes:
            self._update_sample_range()

    def _update_sample_range(self) -> None:
        """Update sample spinbox range based on available samples for selected class."""
        if not self._dataset_service.is_loaded():
            return

        settings = self._ui_to_settings()
        indices = self._dataset_service.get_indices_for_class(settings.class_id, settings.task_type)

        if indices:
            max_idx = len(indices) - 1
            self.sample_spin.setMaximum(max_idx)
            # Clamp current value if it exceeds new max
            if self.sample_spin.value() > max_idx:
                self.sample_spin.setValue(max_idx)

    def _random_sample(self) -> None:
        """Select a random sample."""
        import random

        if not self._dataset_service.is_loaded():
            QMessageBox.warning(self, "No Dataset", "Please load a dataset first.")
            return

        settings = self._ui_to_settings()
        indices = self._dataset_service.get_indices_for_class(settings.class_id, settings.task_type)

        if not indices:
            QMessageBox.warning(self, "No Samples", f"No samples found for class {settings.class_id}")
            return

        self.sample_spin.setValue(random.randint(0, len(indices) - 1))

    def _auto_load_dataset(self) -> None:
        """Auto-load dataset if path exists in settings."""
        if self._settings.dataset_path:
            path = Path(self._settings.dataset_path)
            if path.exists():
                with contextlib.suppress(Exception):
                    self._load_dataset_with_progress()

    # Signal handlers
    def _on_classification_toggled(self) -> None:
        """Handle classification mode toggle."""
        enabled = self.classification_enabled_chk.isChecked()
        self.classification_num_samples_spin.setEnabled(enabled)
        self.classification_pooling_cb.setEnabled(enabled)
        self.classification_show_accuracy_chk.setEnabled(enabled)

        # Also update visibility based on task type
        is_classification = "classification" in self.task_type_cb.currentText().lower()
        self.classification_section.setVisible(is_classification)

    def _on_task_type_changed(self) -> None:
        """Handle task type change."""
        is_classification = "classification" in self.task_type_cb.currentText().lower()
        # Show/hide classification-specific controls
        self.class_label.setVisible(is_classification)
        self.digit_cb.setVisible(is_classification)
        self.sample_spin_label.setVisible(is_classification)
        self.sample_spin.setVisible(is_classification)
        self.new_btn.setVisible(is_classification)

    def _on_group_changed(self) -> None:
        """Handle group selection change."""
        selected_group = self.group_cb.currentData()
        path = Path(self.dataset_edit.text().strip())
        if path.exists():
            # Skip reload if group/path unchanged
            if (
                self._dataset_service.get_current_path() == path
                and self._dataset_service.get_current_group() == selected_group
            ):
                return
            try:
                self._dataset_service.load(path, selected_group)
                self._populate_digit_classes()
            except DatasetServiceError as e:
                QMessageBox.critical(self, "Dataset Error", str(e))

    def _on_class_changed(self) -> None:
        """Handle class selection change."""
        self._update_sample_range()

    def _populate_input_params(self, source: str) -> None:
        """Populate input parameter widgets based on source type.

        Args:
            source: Input source name (e.g., "Constant", "Sinusoid", etc.)
        """
        # Clear existing widgets
        while self.input_params_layout.count():
            item = self.input_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Store widget references for later retrieval
        self._input_param_widgets = {}

        # Create widgets based on source
        if source == "Constant":
            self.input_params_box.setVisible(True)
            self.input_params_layout.addWidget(QLabel("Value:"))
            widget = QLineEdit()
            widget.setText(str(self._settings.input_params.constant_val))
            validator = QDoubleValidator(-FLOAT_MAX, FLOAT_MAX, FLOAT_DECIMALS)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            widget.setValidator(validator)
            self._input_param_widgets["constant_val"] = widget
            self.input_params_layout.addWidget(widget)

        elif source == "Gaussian noise":
            self.input_params_box.setVisible(True)
            self.input_params_layout.addWidget(QLabel("Std Dev:"))
            widget = QLineEdit()
            widget.setText(str(self._settings.input_params.noise_std))
            validator = QDoubleValidator(0.0, FLOAT_MAX, FLOAT_DECIMALS)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            widget.setValidator(validator)
            self._input_param_widgets["noise_std"] = widget
            self.input_params_layout.addWidget(widget)

        elif source == "Colored noise":
            self.input_params_box.setVisible(True)
            self.input_params_layout.addWidget(QLabel("Beta (exponent):"))
            widget = QLineEdit()
            widget.setText(str(self._settings.input_params.colored_beta))
            validator = QDoubleValidator(-FLOAT_MAX, FLOAT_MAX, FLOAT_DECIMALS)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            widget.setValidator(validator)
            self._input_param_widgets["colored_beta"] = widget
            self.input_params_layout.addWidget(widget)

        elif source in ("Sinusoid", "Square wave"):
            self.input_params_box.setVisible(True)

            # Frequency
            self.input_params_layout.addWidget(QLabel("Freq (MHz):"))
            freq_widget = QLineEdit()
            freq_widget.setText(str(self._settings.input_params.sine_freq_mhz))
            validator = QDoubleValidator(0.0, FLOAT_MAX, FLOAT_DECIMALS)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            freq_widget.setValidator(validator)
            self._input_param_widgets["sine_freq_mhz"] = freq_widget
            self.input_params_layout.addWidget(freq_widget)

            # Amplitude
            self.input_params_layout.addWidget(QLabel("Amplitude:"))
            amp_widget = QLineEdit()
            amp_widget.setText(str(self._settings.input_params.sine_amp))
            validator = QDoubleValidator(-FLOAT_MAX, FLOAT_MAX, FLOAT_DECIMALS)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            amp_widget.setValidator(validator)
            self._input_param_widgets["sine_amp"] = amp_widget
            self.input_params_layout.addWidget(amp_widget)

            # Phase
            self.input_params_layout.addWidget(QLabel("Phase (deg):"))
            phase_widget = QLineEdit()
            phase_widget.setText(str(self._settings.input_params.sine_phase_deg))
            validator = QDoubleValidator(-FLOAT_MAX, FLOAT_MAX, FLOAT_DECIMALS)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            phase_widget.setValidator(validator)
            self._input_param_widgets["sine_phase_deg"] = phase_widget
            self.input_params_layout.addWidget(phase_widget)

            # Offset
            self.input_params_layout.addWidget(QLabel("Offset:"))
            offset_widget = QLineEdit()
            offset_widget.setText(str(self._settings.input_params.sine_offset))
            validator = QDoubleValidator(-FLOAT_MAX, FLOAT_MAX, FLOAT_DECIMALS)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            offset_widget.setValidator(validator)
            self._input_param_widgets["sine_offset"] = offset_widget
            self.input_params_layout.addWidget(offset_widget)

        else:  # Dataset sample or unknown
            self.input_params_box.setVisible(False)

        # Add stretch to push widgets to the left
        self.input_params_layout.addStretch()

    def _on_input_src_changed(self) -> None:
        """Handle input source change."""
        src = self.input_src_cb.currentText()
        show_dataset = src == "Dataset sample"

        # Show/hide dataset group and dataset-specific options
        self.dataset_section.setVisible(show_dataset)
        self.dataset_opts_box.setVisible(show_dataset)

        # Populate input_params_box based on source
        self._populate_input_params(src)

    def _on_time_mode_changed(self) -> None:
        """Handle time mode change."""
        is_dt = "dt" in self.time_mode_cb.currentText().lower()
        self.dt_label.setVisible(is_dt)
        self.dt_spin.setVisible(is_dt)
        self.total_time_label.setVisible(not is_dt)
        self.total_time_ns_spin.setVisible(not is_dt)
        self._update_padding_controls_visibility()
        self._update_dt_info()

    def _update_dt_info(self) -> None:
        """Update the info label showing dt and physical time conversions.

        Displays info based on the current time mode:
        - If "Specify dt": shows dimensionless dt and physical time per step
        - If "Specify total time": shows computed dt and physical time per step from total time
        """
        try:
            seq_len = self.seq_spin.value()
            if seq_len <= 0:
                self.dt_info_label.setText("")
                return

            is_dt_mode = "dt" in self.time_mode_cb.currentText().lower()

            if is_dt_mode:
                # Mode: Specify dt (dimensionless)
                try:
                    dt_dimless = float(self.dt_spin.text()) if self.dt_spin.text() else 37.0
                except ValueError:
                    dt_dimless = 37.0

                # Import here to avoid circular imports
                from soen_toolkit.physics.constants import get_omega_c

                omega_c = get_omega_c()

                # Physical time per step: Δt_phys = dt_dimless / ω_c
                dt_phys_sec = dt_dimless / omega_c
                dt_phys_ns = dt_phys_sec * 1e9
                total_time_ns = seq_len * dt_phys_ns

                self.dt_info_label.setText(f"dt={dt_dimless:.3f} (dimless), Δt={dt_phys_ns:.4f} ns/step, total≈{total_time_ns:.3f} ns")
            else:
                # Mode: Specify total time (nanoseconds)
                try:
                    total_time_ns = float(self.total_time_ns_spin.text()) if self.total_time_ns_spin.text() else 0.0
                except ValueError:
                    total_time_ns = 0.0

                if total_time_ns > 0:
                    # Physical time per step
                    dt_phys_ns = total_time_ns / seq_len

                    # Import here to avoid circular imports
                    from soen_toolkit.physics.constants import get_omega_c

                    omega_c = get_omega_c()

                    # Dimensionless dt: dt = Δt_phys * ω_c
                    dt_dimless = dt_phys_ns * 1e-9 * omega_c

                    self.dt_info_label.setText(f"total={total_time_ns:.3f} ns, dt={dt_dimless:.3f} (dimless), Δt={dt_phys_ns:.4f} ns/step")
                else:
                    self.dt_info_label.setText("Enter total time > 0")
        except Exception:
            # Silently fail if calculations error
            pass

        # Also update FFT info since dt affects frequency axis
        try:
            self._update_fft_info()
        except Exception:
            pass

    def _on_prepend_zeros_toggled(self, checked: bool) -> None:
        """Handle prepend zeros toggle."""
        self._update_padding_controls_visibility()

    def _on_append_zeros_toggled(self, checked: bool) -> None:
        """Handle append zeros toggle."""
        self.append_mode_cb.setEnabled(checked)
        self._update_padding_controls_visibility()

    def _update_padding_controls_visibility(self) -> None:
        """Update visibility and enabled state of padding controls based on time mode."""
        is_dt = "dt" in self.time_mode_cb.currentText().lower()
        prepend_enabled = self.prepend_zeros_chk.isChecked()
        append_enabled = self.append_zeros_chk.isChecked()

        # Prepend controls
        self.prepend_zeros_spin.setVisible(is_dt and prepend_enabled)
        self.prepend_zeros_spin.setEnabled(is_dt and prepend_enabled)
        self.prepend_zeros_time_ns.setVisible(not is_dt and prepend_enabled)
        self.prepend_zeros_time_ns.setEnabled(not is_dt and prepend_enabled)

        # Append controls
        self.append_zeros_spin.setVisible(is_dt and append_enabled)
        self.append_zeros_spin.setEnabled(is_dt and append_enabled)
        self.append_zeros_time_ns.setVisible(not is_dt and append_enabled)
        self.append_zeros_time_ns.setEnabled(not is_dt and append_enabled)

    def _get_prepend_time_ns(self) -> float:
        """Get prepend time_ns value from UI."""
        try:
            txt = self.prepend_zeros_time_ns.text().strip()
            return float(txt) if txt else 0.0
        except ValueError:
            return 0.0

    def _get_append_time_ns(self) -> float:
        """Get append time_ns value from UI."""
        try:
            txt = self.append_zeros_time_ns.text().strip()
            return float(txt) if txt else 0.0
        except ValueError:
            return 0.0

    def _on_view_mode_changed(self, text: str) -> None:
        """Update FFT view mode when dropdown changes."""
        mode_map = {
            "Time Only": ViewMode.TIME_ONLY,
            "Frequency Only": ViewMode.FREQUENCY_ONLY,
            "Time + Frequency": ViewMode.SPLIT,
            "Waterfall": ViewMode.WATERFALL,
        }
        self._settings.fft.view_mode = mode_map.get(text, ViewMode.TIME_ONLY)
        self._update_fft_info()

    def _open_fft_settings(self) -> None:
        """Open FFT settings dialog."""
        from .fft_settings_dialog import FFTSettingsDialog

        dialog = FFTSettingsDialog(self, self._settings.fft)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._settings.fft = dialog.get_settings()
            self._settings_adapter.save(self._settings)
            self._update_fft_info()

    def _calculate_dt(self) -> float:
        """Calculate current dt in nanoseconds from UI state."""
        try:
            from soen_toolkit.physics.constants import get_omega_c
            omega_c = get_omega_c()

            is_dt_mode = "dt" in self.time_mode_cb.currentText().lower()

            if is_dt_mode:
                # dt specified directly (dimensionless)
                dt_dimless = float(self.dt_spin.text()) if self.dt_spin.text() else 37.0
                return dt_dimless / omega_c * 1e9  # Convert to ns
            else:
                # Total time specified
                total_time_ns = float(self.total_time_ns_spin.text()) if self.total_time_ns_spin.text() else 0.0
                seq_len = self.seq_spin.value()
                if seq_len > 0 and total_time_ns > 0:
                    return total_time_ns / seq_len
                return 0.0
        except Exception:
            return 0.0

    def _update_fft_info(self) -> None:
        """Update FFT info label with resolution and Nyquist frequency."""
        from .fft_analysis import FFTAnalysisService

        # Only show info if FFT view is active
        if self._settings.fft.view_mode == ViewMode.TIME_ONLY:
            self.fft_info_label.setText("")
            return

        fft_service = FFTAnalysisService()

        seq_len = self.seq_spin.value()
        dt_ns = self._calculate_dt()

        if dt_ns > 0:
            res_mhz = fft_service.get_frequency_resolution(seq_len, dt_ns)
            nyquist_ghz = fft_service.get_nyquist_frequency(dt_ns)
            self.fft_info_label.setText(
                f"FFT: Res={res_mhz:.3f} MHz | Nyquist={nyquist_ghz:.2f} GHz"
            )
        else:
            self.fft_info_label.setText("")

    def closeEvent(self, event) -> None:
        """Save settings on close."""
        try:
            settings = self._ui_to_settings()
            self._settings_adapter.save(settings)
        except Exception:
            pass
        super().closeEvent(event)
