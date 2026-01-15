# src/soen_toolkit/model_creation_gui/main_window.py
from __future__ import annotations

import contextlib
import logging
import pathlib

from PyQt6.QtCore import QDir, QSettings, Qt
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from soen_toolkit.core.model_yaml import dump_model_to_yaml
from soen_toolkit.model_creation_gui.features.analysis import AnalyseTab
from soen_toolkit.model_creation_gui.features.documentation import DocumentationTab
from soen_toolkit.model_creation_gui.features.dynamic_init import DynamicInitTab
from soen_toolkit.model_creation_gui.features.io import ModelIOTab
from soen_toolkit.model_creation_gui.features.model_builder import ModelBuildingTab
from soen_toolkit.model_creation_gui.features.unit_conversion import UnitConversionTab
from soen_toolkit.model_creation_gui.features.visualization import VisualisationTab
from soen_toolkit.model_creation_gui.features.web_viewer import WebViewerTab
from soen_toolkit.model_creation_gui.model_manager import ModelManager
from soen_toolkit.model_creation_gui.utils import theme
from soen_toolkit.model_creation_gui.utils.paths import icon

# +++ Import the new model tools +++
from soen_toolkit.utils.model_tools import export_model_to_json, model_from_json

# --- Import the new model tools ---
# +++ Import config classes for code generation +++
# --- Import config classes for code generation ---


log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, progress_callback=None) -> None:
        super().__init__()
        self.setWindowTitle("SOEN Model Builder")

        # Store callback for progress updates
        self._progress_callback = progress_callback or (lambda v, m="": None)

        # --- Theme setup (10%) ---
        self._progress_callback(10, "Setting up theme...")
        self._mgr = ModelManager()

        # Restore last loaded path from persistent settings
        self._settings = QSettings("GreatSky", "SOEN-Toolkit")
        last_path = self._settings.value("last_loaded_model_path", None)
        if last_path:
            self._mgr.last_loaded_path = pathlib.Path(last_path)

        # --- Model manager ready (15%) ---
        self._progress_callback(15, "Creating tabs...")

        # ---- Tabs (Load/Export first) ----
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self.io_tab = ModelIOTab(self._mgr)
        self._tabs.addTab(self.io_tab, "Load/Export")

        # Progress: tabs 1/6 (25%)
        self._progress_callback(25, "Loading Model Builder tab...")
        self.build_tab = ModelBuildingTab(self._mgr)

        # Progress: tabs 2/6 (35%)
        self._progress_callback(35, "Loading Visualisation tab...")
        self.visualise_tab = VisualisationTab(self._mgr)

        # Progress: tabs 3/6 (42%)
        self._progress_callback(42, "Loading Web Viewer tab...")
        self.web_tab = WebViewerTab(self._mgr)

        # Progress: tabs 4/6 (49%)
        self._progress_callback(49, "Loading Analysis tab...")
        self.analyse_tab = AnalyseTab(self._mgr)

        # Progress: tabs 5/7 (53%)
        self._progress_callback(53, "Loading Unit Conversion tab...")
        self.unit_conv_tab = UnitConversionTab(self._mgr)

        # Progress: tabs 6/7 (58%)
        self._progress_callback(58, "Loading Dynamic Initialization tab...")
        self.dynamic_init_tab = DynamicInitTab(self._mgr)

        # Progress: tabs 7/7 (63%)
        self._progress_callback(63, "Loading Documentation tab...")
        self.docs_tab = DocumentationTab(self._mgr)

        self._tabs.addTab(self.build_tab, "Model Builder")
        self._tabs.addTab(self.visualise_tab, "Visualise")
        # Add but hide the Web Viewer tab (unfinished feature)
        idx_web = self._tabs.addTab(self.web_tab, "Web Viewer")
        try:
            self._tabs.setTabVisible(idx_web, False)
        except Exception:
            # Fallback for Qt versions without setTabVisible: remove from display but keep reference
            self._tabs.removeTab(idx_web)
        self._tabs.addTab(self.analyse_tab, "Analyse")
        self._tabs.addTab(self.dynamic_init_tab, "Dynamic Init")
        self._tabs.addTab(self.unit_conv_tab, "Unit Conversion")
        self._tabs.addTab(self.docs_tab, "Documentation")

        # --- Tab layout (70%) ---
        self._progress_callback(70, "Setting up layout...")
        # Wrap tabs in a central container to allow gradient/opacity styling
        self._central_root = QWidget()
        self._central_root.setObjectName("central_root")
        _root_layout = QVBoxLayout(self._central_root)
        _root_layout.setContentsMargins(0, 0, 0, 0)
        _root_layout.addWidget(self._tabs)
        self.setCentralWidget(self._central_root)
        # Native macOS look for toolbar/title (no-op on other platforms)
        with contextlib.suppress(Exception):
            self.setUnifiedTitleAndToolBarOnMac(True)

        # ---- Status bar (75%) ---
        self._progress_callback(75, "Initializing status bar...")
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

        # --- Glass effect (80%) ---
        self._progress_callback(80, "Applying visual effects...")
        # Apply saved glass effect immediately on startup so user doesn't need to toggle
        with contextlib.suppress(Exception):
            self._apply_glass_effect(theme.get_glass_enabled())

        # --- Signal connections (85%) ---
        self._progress_callback(85, "Connecting signals...")
        # ---- Connect Model Changes ----
        self._mgr.model_changed.connect(self.build_tab._on_model_changed)
        self._mgr.model_changed.connect(self.visualise_tab._on_model_changed)
        # AnalyseTab internally wires its child tabs to the signal
        # but keep an explicit hook for manual refresh callers
        if hasattr(self.analyse_tab, "_on_model_changed"):
            self._mgr.model_changed.connect(self.analyse_tab._on_model_changed)
        # Keep IO tab lists updated
        self._mgr.model_changed.connect(self.io_tab._refresh_freeze_lists)

        # --- Initial refresh (90%) ---
        self._progress_callback(90, "Initializing model state...")
        try:
            self._mgr.model_changed.emit()
            # Kick initial refresh explicitly for tabs
            if hasattr(self.analyse_tab, "_on_model_changed"):
                self.analyse_tab._on_model_changed()
            self.visualise_tab._on_model_changed()
            self.build_tab._on_model_changed()
        except Exception:
            pass

        # --- Menu setup (95%) ---
        self._progress_callback(95, "Creating menu...")
        # Only a top menu bar; keep the window chrome clean (works cross‑platform)
        self._make_menu()

        # --- Complete (100%) ---
        self._progress_callback(100, "Ready!")

    def _make_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        # Core actions (previously in toolbar)
        act_new = QAction(QIcon(icon("plus")), "New Model", self)
        act_new.setShortcut("Ctrl+N")
        act_new.setToolTip("Clear the current configuration and unload any built model")
        act_new.setStatusTip("Start a fresh workspace")
        act_new.triggered.connect(self._new_model)

        act_open_io = QAction(QIcon(icon("open_model.svg")), "Model Load / Export", self)
        act_open_io.setShortcut("Ctrl+L")
        act_open_io.setToolTip("Jump to the Load/Export tab for loading, merging and saving models")
        act_open_io.setStatusTip("Open Load/Export tab")
        act_open_io.triggered.connect(lambda: self._tabs.setCurrentWidget(self.io_tab))

        act_export_yaml = QAction(QIcon(icon("code.svg")), "Export YAML (config only)...", self)
        act_export_yaml.setShortcut("Ctrl+Y")
        act_export_yaml.setToolTip("Export the current model architecture as a YAML file (no weights)")
        act_export_yaml.setStatusTip("Export YAML architecture")
        act_export_yaml.triggered.connect(self._save_model_yaml)

        file_menu.addAction(act_new)
        file_menu.addAction(act_open_io)
        file_menu.addAction(act_export_yaml)
        file_menu.addSeparator()

        # File actions
        act_open_model = QAction("Open Torch Model...", self)
        act_open_model.setShortcut("Ctrl+O")
        act_open_model.setToolTip("Load a saved Torch model (.soen/.pth). Replaces or merges via the IO tab controls.")
        act_open_model.triggered.connect(self._load_model_pth)

        act_open_json = QAction("Open Model JSON...", self)
        act_open_json.setToolTip("Load a full model (config + weights) saved as JSON")
        act_open_json.triggered.connect(self._open_model_json)

        act_save_model = QAction("Save Torch Model...", self)
        act_save_model.setShortcut("Ctrl+S")
        act_save_model.setToolTip("Save the built model to disk (.soen by default)")
        act_save_model.triggered.connect(self._save_model_pth)

        act_save_json = QAction("Save Model JSON...", self)
        act_save_json.setToolTip("Export the built model (config + weights) as a JSON file")
        act_save_json.triggered.connect(self._save_model_json)

        file_menu.addAction(act_open_model)
        file_menu.addAction(act_open_json)
        file_menu.addSeparator()
        file_menu.addAction(act_save_model)
        file_menu.addAction(act_save_json)

        view_menu = bar.addMenu("&View")
        # Theme submenu with multiple theme choices (persists via QSettings)
        theme_menu = view_menu.addMenu("Theme")
        from PyQt6.QtGui import QActionGroup

        group = QActionGroup(self)
        group.setExclusive(True)
        act_basic = QAction("Basic", self)
        act_basic.setCheckable(True)
        act_dark = QAction("Dark", self)
        act_dark.setCheckable(True)
        act_midnight = QAction("Midnight", self)
        act_midnight.setCheckable(True)
        act_arctic = QAction("Arctic", self)
        act_arctic.setCheckable(True)
        act_sunset = QAction("Sunset", self)
        act_sunset.setCheckable(True)
        act_minimal = QAction("Minimal", self)
        act_minimal.setCheckable(True)
        group.addAction(act_basic)
        group.addAction(act_dark)
        group.addAction(act_midnight)
        group.addAction(act_arctic)
        group.addAction(act_sunset)
        group.addAction(act_minimal)

        # Initialize checked state from saved theme
        try:
            current = theme.get_current_theme()
            if current == theme.DARK_THEME_NAME:
                act_dark.setChecked(True)
            elif current == theme.MIDNIGHT_THEME_NAME:
                act_midnight.setChecked(True)
            elif current == theme.ARCTIC_THEME_NAME:
                act_arctic.setChecked(True)
            elif current == theme.SUNSET_THEME_NAME:
                act_sunset.setChecked(True)
            elif current == theme.MINIMAL_THEME_NAME:
                act_minimal.setChecked(True)
            else:
                act_basic.setChecked(True)
        except Exception:
            act_basic.setChecked(True)

        act_basic.triggered.connect(lambda: self._set_theme(theme.BASIC_THEME_NAME))
        act_dark.triggered.connect(lambda: self._set_theme(theme.DARK_THEME_NAME))
        act_midnight.triggered.connect(lambda: self._set_theme(theme.MIDNIGHT_THEME_NAME))
        act_arctic.triggered.connect(lambda: self._set_theme(theme.ARCTIC_THEME_NAME))
        act_sunset.triggered.connect(lambda: self._set_theme(theme.SUNSET_THEME_NAME))
        act_minimal.triggered.connect(lambda: self._set_theme(theme.MINIMAL_THEME_NAME))
        theme_menu.addAction(act_basic)
        theme_menu.addAction(act_dark)
        theme_menu.addAction(act_midnight)
        theme_menu.addAction(act_arctic)
        theme_menu.addAction(act_sunset)
        theme_menu.addAction(act_minimal)

        # Font Size submenu - helps with DPI scaling issues on different platforms
        font_menu = view_menu.addMenu("Font Size")
        self._font_scale_group = QActionGroup(self)
        self._font_scale_group.setExclusive(True)

        # Preset scale options
        font_scales = [
            ("Tiny (50%)", 0.5),
            ("Small (70%)", 0.7),
            ("Compact (85%)", 0.85),
            ("Normal (100%)", 1.0),
            ("Large (115%)", 1.15),
            ("Larger (130%)", 1.3),
            ("Extra Large (150%)", 1.5),
        ]

        current_scale = theme.get_font_scale()
        self._font_scale_actions = {}

        for label, scale_value in font_scales:
            act = QAction(label, self)
            act.setCheckable(True)
            # Check if this is the current scale (with small tolerance)
            if abs(current_scale - scale_value) < 0.01:
                act.setChecked(True)
            act.triggered.connect(lambda checked, s=scale_value: self._set_font_scale(s))
            self._font_scale_group.addAction(act)
            font_menu.addAction(act)
            self._font_scale_actions[scale_value] = act

        font_menu.addSeparator()

        # Custom font scale option
        act_custom_font = QAction("Custom...", self)
        act_custom_font.triggered.connect(self._show_font_scale_dialog)
        font_menu.addAction(act_custom_font)

        # Effects submenu
        effects_menu = view_menu.addMenu("Effects")
        self._act_glass = QAction("Sleek Glass (semi-transparent)", self)
        self._act_glass.setCheckable(True)
        try:
            self._act_glass.setChecked(theme.get_glass_enabled())
        except Exception:
            self._act_glass.setChecked(False)
        self._act_glass.toggled.connect(self._apply_glass_effect)
        effects_menu.addAction(self._act_glass)

    def _update_last_path(self, path: str | pathlib.Path) -> None:
        """Update the last loaded path in both manager and persistent settings."""
        path_obj = pathlib.Path(path)
        self._mgr.last_loaded_path = path_obj
        self._settings.setValue("last_loaded_model_path", str(path_obj))

    def _new_model(self) -> None:
        # Confirmation before clearing
        reply = QMessageBox.question(
            self,
            "New Model",
            "This will clear the current configuration and unload the model. Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._mgr.layers.clear()
            self._mgr.connections.clear()
            self._mgr.sim_config = ModelManager().sim_config
            self._mgr.model = None
            self._mgr.model_changed.emit()
            QMessageBox.information(self, "New Model", "Configuration cleared.")
            # Also clear IO tab session history
            try:
                if hasattr(self.io_tab, "clear_history"):
                    self.io_tab.clear_history()
                # Proactively refresh dependent tabs so visuals clear immediately
                with contextlib.suppress(Exception):
                    self.visualise_tab._on_model_changed()
                try:
                    if hasattr(self.analyse_tab, "_on_model_changed"):
                        self.analyse_tab._on_model_changed()
                except Exception:
                    pass
            except Exception:
                pass

    def _open_model_json(self) -> None:
        """Loads a complete model state (config + weights) from a JSON file."""
        # Confirmation before overwriting
        reply = QMessageBox.question(
            self,
            "Open Model JSON",
            "Loading a model from JSON will replace the current configuration and model. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Cancel:
            return

        # Prefer last-used model directory; use non-native dialog to avoid slow Finder
        try:
            start = getattr(self._mgr, "last_loaded_path", None)
            start_dir = str(pathlib.Path(start).parent) if start else QDir.homePath()
        except Exception:
            start_dir = QDir.homePath()
        opts = QFileDialog.Options()
        try:
            opts |= QFileDialog.Option.DontUseNativeDialog
            opts |= QFileDialog.Option.DontResolveSymlinks
            opts |= QFileDialog.Option.ReadOnly
            with contextlib.suppress(Exception):
                opts |= QFileDialog.Option.DontUseCustomDirectoryIcons
        except Exception:
            pass
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Open SOEN Model JSON",
            start_dir,
            "JSON (*.json)",
            options=opts,
        )
        if fname:
            try:
                loaded_model = model_from_json(fname)
                self._mgr.model = loaded_model
                self._mgr.sim_config = self._mgr.model.sim_config
                self._mgr.layers = self._mgr.model.layers_config
                self._mgr.connections = self._mgr.model.connections_config
                self._mgr.model_changed.emit()
                self._update_last_path(fname)

                QMessageBox.information(self, "Success", f"Model loaded successfully from:\n{fname}")
                log.info("Model loaded from JSON: %s", fname)

            except Exception as e:
                log.exception("Failed to load model from JSON: %s", fname)
                QMessageBox.critical(self, "Load Failed", f"Error loading model from JSON:\n{e}")

    def _save_model_json(self) -> None:
        """Saves the complete model state (config + weights) to a JSON file."""
        # Check if the model is built
        if self._mgr.model is None:
            QMessageBox.warning(self, "No Model Built", "The model must be built before it can be saved to JSON. Please use the 'Build Model' button first.")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save SOEN Model JSON",
            QDir.homePath(),
            "JSON (*.json)",
        )
        if fname:
            # Ensure filename ends with .json
            if not fname.endswith(".json"):
                fname += ".json"

            try:
                # Use the new function to export the model to JSON
                export_model_to_json(self._mgr.model, fname)
                QMessageBox.information(self, "Success", f"Model saved successfully to:\n{fname}")
                log.info("Model saved to JSON: %s", fname)

            except Exception as e:
                log.exception("Failed to save model to JSON: %s", fname)
                QMessageBox.critical(self, "Save Failed", f"Error saving model to JSON:\n{e}")

    def _load_model_pth(self) -> None:
        """Loads model state dict and configuration from a PTH file."""
        # Confirmation before overwriting
        reply = QMessageBox.question(
            self,
            "Load Model",
            "Loading a model from PTH will replace the current configuration and model. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Cancel:
            return

        try:
            start = getattr(self._mgr, "last_loaded_path", None)
            start_dir = str(pathlib.Path(start).parent) if start else QDir.homePath()
        except Exception:
            start_dir = QDir.homePath()
        opts = QFileDialog.Options()
        try:
            opts |= QFileDialog.Option.DontUseNativeDialog
            opts |= QFileDialog.Option.DontResolveSymlinks
            opts |= QFileDialog.Option.ReadOnly
            with contextlib.suppress(Exception):
                opts |= QFileDialog.Option.DontUseCustomDirectoryIcons
        except Exception:
            pass
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Load Torch Model",
            start_dir,
            "Torch Model (*.soen *.pth)",
            options=opts,
        )
        if fname:
            try:
                # ModelManager's load_model_file already handles loading the model
                # and syncing the internal config lists.
                self._mgr.load_model_file(pathlib.Path(fname))
                self._update_last_path(fname)
                QMessageBox.information(self, "Success", f"Model loaded successfully from:\n{fname}")
                log.info("Model loaded from PTH: %s", fname)
            except Exception as e:
                log.exception("Failed to load model from PTH: %s", fname)
                QMessageBox.critical(self, "Load failed", str(e))

    def _save_model_pth(self) -> None:
        """Saves model state dict and configuration to a PTH file."""
        if self._mgr.model is None:
            QMessageBox.warning(self, "No model", "Build the model first.")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save Torch Model",
            QDir.homePath(),
            "Torch Model (*.soen *.pth)",
        )
        if fname:
            # Ensure filename ends with .soen or .pth (default to .soen)
            if not fname.endswith((".pth", ".soen")):
                fname += ".soen"
            try:
                self._mgr.save_model_file(pathlib.Path(fname))
                QMessageBox.information(self, "Success", f"Model successfully saved to:\n{fname}")
                log.info("Model saved to PTH: %s", fname)
            except Exception as e:
                log.exception("Failed to save model to PTH: %s", fname)
                QMessageBox.critical(self, "Save failed", str(e))

    def _save_model_yaml(self) -> None:
        """Saves the current model configuration (no weights) to a YAML file."""
        if self._mgr.model is None:
            QMessageBox.warning(self, "No Model Built", "Build or load a model before exporting YAML.")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model YAML (config only)",
            QDir.homePath(),
            "YAML (*.yaml *.yml)",
        )
        if fname:
            try:
                dump_model_to_yaml(self._mgr.model, fname)
                QMessageBox.information(self, "Success", f"Model YAML saved to:\n{fname}")
                log.info("Model YAML saved: %s", fname)
            except Exception as e:
                log.exception("Failed to save model YAML: %s", fname)
                QMessageBox.critical(self, "Save Failed", f"Error saving model YAML:\n{e}")

    def _load_additional_model(self) -> None:
        """Load a model and merge it into the current workspace (ID-shifted)."""
        try:
            start = getattr(self._mgr, "last_loaded_path", None)
            start_dir = str(pathlib.Path(start).parent) if start else QDir.homePath()
        except Exception:
            start_dir = QDir.homePath()
        opts = QFileDialog.Options()
        try:
            opts |= QFileDialog.Option.DontUseNativeDialog
            opts |= QFileDialog.Option.DontResolveSymlinks
            opts |= QFileDialog.Option.ReadOnly
            with contextlib.suppress(Exception):
                opts |= QFileDialog.Option.DontUseCustomDirectoryIcons
        except Exception:
            pass
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Load Additional SOEN Model",
            start_dir,
            "SOEN Model (*.pth *.soen);;JSON (*.json)",
            options=opts,
        )
        if not fname:
            return
        try:
            self._mgr.load_additional_model_file(pathlib.Path(fname))
            self._update_last_path(fname)
            QMessageBox.information(self, "Merged", f"Merged additional model from:\n{fname}")
        except Exception as e:
            QMessageBox.critical(self, "Merge failed", str(e))

    def _export_model_code(self) -> None:
        """Exports the current model configuration as a Python script."""
        if not self._mgr.layers:
            QMessageBox.warning(self, "No Layers", "Please define at least one layer before exporting code.")
            return

        # Suggest a filename based on the first layer's name or a default
        default_filename = "exported_model_script.py"
        # A more sophisticated way might involve getting a project name or similar if available

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Export Python Code",
            str(pathlib.Path(QDir.homePath()) / default_filename),
            "Python Files (*.py)",
        )

        if not fname:
            return  # User cancelled

        # Ensure filename ends with .py
        if not fname.endswith(".py"):
            fname += ".py"

        try:
            # Retrieve configurations from ModelManager
            sim_config = self._mgr.sim_config
            layers_config_data = self._mgr.layers
            connections_config_data = self._mgr.connections
            # Check if seeding is enabled in the ModelBuildingTab
            # This assumes build_tab is accessible and has seed_checkbox
            use_seed = self.build_tab.seed_checkbox.isChecked()

            # Start building the Python script string
            script_lines = []
            script_lines.append("# Generated SOEN Model Script")
            script_lines.append("import torch")
            script_lines.append("import numpy as np")
            script_lines.append("import random")
            script_lines.append("from pathlib import Path")
            script_lines.append("from soen_toolkit.core import SOENModelCore, SimulationConfig, LayerConfig, ConnectionConfig, NoiseConfig")
            script_lines.append("\n")

            # Seeding section
            if use_seed:
                script_lines.append("# --- Seeding ---")
                script_lines.append("seed = 42")
                script_lines.append("random.seed(seed)")
                script_lines.append("np.random.seed(seed)")
                script_lines.append("torch.manual_seed(seed)")
                script_lines.append("# torch.backends.cudnn.deterministic = True")
                script_lines.append("# torch.backends.cudnn.benchmark = False")
                script_lines.append("logging.info(f'Random seeds set to {seed}')")
                script_lines.append("\n")
            else:
                script_lines.append("# --- Seeding (disabled by user in GUI) ---")
                script_lines.append("# To enable deterministic model creation, uncomment and set a seed:")
                script_lines.append("# seed = 42")
                script_lines.append("# random.seed(seed)")
                script_lines.append("# np.random.seed(seed)")
                script_lines.append("# torch.manual_seed(seed)")
                script_lines.append("logging.info('Model created without fixed random seed.')")
                script_lines.append("\n")

            # SimulationConfig
            script_lines.append("# --- Simulation Configuration ---")
            script_lines.append(
                f"sim_config = SimulationConfig("
                f"dt={sim_config.dt}, "
                f"input_type='{sim_config.input_type}', "
                f"network_evaluation_method='{getattr(sim_config, 'network_evaluation_method', 'layerwise')}', "
                f"track_power={sim_config.track_power}, "
                f"track_phi={sim_config.track_phi}, "
                f"track_g={sim_config.track_g}, "
                f"track_s={sim_config.track_s}, "
                f"early_stopping_forward_pass={getattr(sim_config, 'early_stopping_forward_pass', False)}, "
                f"early_stopping_tolerance={getattr(sim_config, 'early_stopping_tolerance', 1e-6)}, "
                f"early_stopping_patience={getattr(sim_config, 'early_stopping_patience', 1)}, "
                f"early_stopping_min_steps={getattr(sim_config, 'early_stopping_min_steps', 1)}, "
                f"steady_window_min={getattr(sim_config, 'steady_window_min', 50)}, "
                f"steady_tol_abs={getattr(sim_config, 'steady_tol_abs', 1e-5)}, "
                f"steady_tol_rel={getattr(sim_config, 'steady_tol_rel', 1e-3)}"
                f")",
            )
            script_lines.append("\n")

            # LayersConfig
            script_lines.append("# --- Layers Configuration ---")
            script_lines.append("layers_config = [")
            for lc in layers_config_data:
                noise_conf_str = "None"
                if lc.noise:
                    noise_conf_str = f"NoiseConfig(**{lc.noise.__dict__})"
                perturb_conf_str = "None"
                if lc.perturb:
                    perturb_conf_str = f"PerturbationConfig(**{lc.perturb.__dict__})"
                # repr() is used for params to get a string representation of the dict
                script_lines.append(
                    f"    LayerConfig(layer_id={lc.layer_id}, "
                    f"layer_type='{lc.layer_type}', "
                    f"params={lc.params!r}, "  # Use repr for dictionary
                    f"model_id={getattr(lc, 'model_id', 0)}, "
                    f"description='{lc.description}', "
                    f"noise={noise_conf_str}, perturb={perturb_conf_str}),",
                )
            script_lines.append("]\n")

            # ConnectionsConfig
            script_lines.append("# --- Connections Configuration ---")
            script_lines.append("connections_config = [")
            for cc in connections_config_data:
                # repr() is used for params to get a string representation of the dict
                script_lines.append(
                    f"    ConnectionConfig(from_layer={cc.from_layer}, "
                    f"to_layer={cc.to_layer}, "
                    f"connection_type='{cc.connection_type}', "
                    f"params={cc.params!r}, "  # Use repr for dictionary
                    f"learnable={cc.learnable}),",
                )
            script_lines.append("]\n")

            # Model Instantiation
            script_lines.append("# --- Model Instantiation ---")
            script_lines.append("model = SOENModelCore(")
            script_lines.append("    sim_config=sim_config,")
            script_lines.append("    layers_config=layers_config,")
            script_lines.append("    connections_config=connections_config")
            script_lines.append(")")
            script_lines.append("logging.info(f'Model instantiated with {sum(p.numel() for p in model.parameters())} parameters.')")
            script_lines.append("\n")

            # Saving the model (optional)
            script_lines.append("# --- Save the Model (optional) ---")
            script_lines.append('output_model_filename = "exported_model.pth"')
            script_lines.append("try:")
            script_lines.append("    model.save(output_model_filename)")
            script_lines.append("    logging.info(f'Model saved to {output_model_filename}')")
            script_lines.append("except Exception as e:")
            script_lines.append("    logging.exception(f'Error saving model: {e}')")
            script_lines.append("\n")

            script_lines.append("# Example: To load this model later")
            script_lines.append("# loaded_model = SOENModelCore.load(output_model_filename)")
            script_lines.append("# print(f'Model loaded from {output_model_filename}')")

            # Write to file
            with open(fname, "w") as f:
                f.write("\n".join(script_lines))

            QMessageBox.information(self, "Success", f"Python script exported successfully to:\n{fname}")
            log.info("Model configuration exported to Python script: %s", fname)

        except Exception as e:
            log.exception("Failed to export model configuration to Python script: %s", fname)
            QMessageBox.critical(self, "Export Failed", f"Error exporting Python script:\n{e}")

    def _toggle_theme(self) -> None:
        theme.toggle_theme(QApplication.instance())

    # ----- theme/effects helpers -----
    def _set_theme(self, name: str) -> None:
        theme.apply_theme(QApplication.instance(), name)
        # Re-apply glass to keep styling consistent with new theme
        self._apply_glass_effect(self._act_glass.isChecked())

    def _apply_glass_effect(self, enabled: bool) -> None:
        """Apply a subtle translucent effect with a vertical gradient.

        Note: Full acrylic/vibrancy requires platform APIs. Here we emulate with
        a translucent central widget and gradient background. Users can toggle
        under View → Effects.
        """
        with contextlib.suppress(Exception):
            theme.set_glass_enabled(enabled)

        # Window opacity: leave titlebar solid, fade content slightly
        with contextlib.suppress(Exception):
            self.setWindowOpacity(0.96 if enabled else 1.0)

        # Gradient background on the central root
        try:
            if enabled:
                # Pick gradient based on current theme for best contrast
                try:
                    current_theme = theme.get_current_theme()
                except Exception:
                    current_theme = theme.BASIC_THEME_NAME
                if current_theme == theme.DARK_THEME_NAME:
                    grad = "#central_root {  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,     stop:0 rgba(255,255,255,8), stop:1 rgba(0,0,0,32));}"
                elif current_theme == theme.MIDNIGHT_THEME_NAME:
                    # Midnight theme: diagonal gradient with subtle teal shift
                    grad = "#central_root {  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,     stop:0 rgba(77,208,225,12), stop:0.5 rgba(58,63,79,8), stop:1 rgba(38,198,218,10));}"
                elif current_theme == theme.ARCTIC_THEME_NAME:
                    # Arctic: enhance the blue-lavender gradient
                    grad = "#central_root {  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,     stop:0 rgba(100,181,246,15), stop:0.5 rgba(255,255,255,20), stop:1 rgba(179,136,255,12));}"
                elif current_theme == theme.SUNSET_THEME_NAME:
                    # Sunset: enhance the peach-pink gradient
                    grad = "#central_root {  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,     stop:0 rgba(255,138,128,10), stop:0.5 rgba(255,255,255,18), stop:1 rgba(33,150,243,8));}"
                elif current_theme == theme.MINIMAL_THEME_NAME:
                    # Minimal: enhance the ice blue-mint gradient
                    grad = "#central_root {  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,     stop:0 rgba(33,150,243,12), stop:0.5 rgba(255,255,255,22), stop:1 rgba(129,199,132,10));}"
                else:
                    # Basic theme: avoid darkening, add a very soft white fade
                    grad = "#central_root {  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,     stop:0 rgba(255,255,255,64), stop:1 rgba(255,255,255,0));}"
                self._central_root.setStyleSheet(grad)
            else:
                self._central_root.setStyleSheet("")
        except Exception:
            pass

    def _set_font_scale(self, scale: float) -> None:
        """Apply a new font scale and update the menu checkmarks."""
        theme.apply_font_scale(QApplication.instance(), scale)
        # Re-apply glass effect to maintain consistency
        self._apply_glass_effect(self._act_glass.isChecked())

        # Update checkmarks - uncheck all first, then check the matching one
        for scale_value, action in self._font_scale_actions.items():
            action.setChecked(abs(scale - scale_value) < 0.01)

    def _show_font_scale_dialog(self) -> None:
        """Show a dialog to set a custom font scale value."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QSlider, QSpinBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Custom Font Size")
        dialog.setMinimumWidth(350)

        layout = QVBoxLayout(dialog)

        # Description
        desc = QLabel(
            "Adjust the font scale to fix display issues on different screens.\n"
            "A lower value makes text smaller, useful for high-DPI displays."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Current scale display
        current_scale = theme.get_font_scale()
        scale_label = QLabel(f"Scale: {int(current_scale * 100)}%")
        layout.addWidget(scale_label)

        # Slider + spinbox row
        row_layout = QHBoxLayout()

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(50)  # 50%
        slider.setMaximum(200)  # 200%
        slider.setValue(int(current_scale * 100))
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(25)

        spinbox = QSpinBox()
        spinbox.setMinimum(50)
        spinbox.setMaximum(200)
        spinbox.setValue(int(current_scale * 100))
        spinbox.setSuffix("%")

        # Sync slider and spinbox
        def on_slider_changed(val):
            spinbox.blockSignals(True)
            spinbox.setValue(val)
            spinbox.blockSignals(False)
            scale_label.setText(f"Scale: {val}%")

        def on_spinbox_changed(val):
            slider.blockSignals(True)
            slider.setValue(val)
            slider.blockSignals(False)
            scale_label.setText(f"Scale: {val}%")

        slider.valueChanged.connect(on_slider_changed)
        spinbox.valueChanged.connect(on_spinbox_changed)

        row_layout.addWidget(slider)
        row_layout.addWidget(spinbox)
        layout.addLayout(row_layout)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_scale = slider.value() / 100.0
            self._set_font_scale(new_scale)

    # --- Deprecated config methods ---
    # These methods in MainWindow are now replaced by _open_model_json and _save_model_json
    # def _open_config(self): ...
    # def _save_config(self): ...
