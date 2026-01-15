"""Visualization settings management.

Single source of truth for all visualization settings with automatic
QSettings persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from PyQt6.QtCore import QSettings


@dataclass
class VisualizationSettings:
    """All visualization settings with typed defaults.

    This is the single source of truth for settings - no more duplication
    across _default_settings, _load_settings(), and _save_settings().
    """

    # General layout
    orientation: str = "LR"  # 'LR' or 'TB'
    edge_routing: str = "true"  # Graphviz splines setting
    edge_thickness: float = 0.5
    arrow_size: float = 0.5
    layer_spacing: float = 1.0

    # Appearance
    theme: str = "default"  # 'default' or 'dark'
    inter_color: str = "#000000"
    intra_color: str = "#ff0000"
    layer_color: str = "#eae5be"
    bg_color: str = "#ffffff"
    text_color: str = "#000000"
    show_layer_outline: bool = False

    # View mode
    simple_view: bool = True
    detailed_layout: str = "linear"  # 'linear', 'circular', or 'grid'
    group_by_model: bool = False  # Group layers by model_id into visual clusters

    # Display toggles
    show_intra: bool = True
    show_desc: bool = True
    show_conn_type: bool = False
    show_layer_ids: bool = True
    show_node_ids: bool = False
    show_model_ids: bool = False
    show_neuron_polarity: bool = False
    show_connection_polarity: bool = False

    # Gradient flow - display toggle
    show_gradients: bool = False

    # Gradient flow - dataset settings
    grad_hdf5_path: str = ""
    grad_split: str = "train"
    grad_seq_len: int = 100
    grad_time_mode: str = "dt"  # 'dt' or 'total_time'
    grad_dt: float = 37.0
    grad_total_time_ns: float | None = None
    grad_feature_min: float | None = None
    grad_feature_max: float | None = None

    # Gradient flow - sample selection
    grad_task_type: str = "classification"  # 'classification' or 'seq2seq'
    grad_class_id: int = 0
    grad_sample_index: int = 0

    # Gradient flow - computation settings
    grad_loss_fn: str = "mse"  # 'mse', 'cross_entropy', 'sum_output'
    grad_log_scale: bool = False
    grad_colormap: str = "RdBu_r"

    # Settings key for QSettings
    SETTINGS_KEY: str = field(default="visualization", repr=False, compare=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like setting for backward compatibility."""
        if hasattr(self, key):
            setattr(self, key, value)

    def copy(self) -> dict[str, Any]:
        """Return a dict copy of settings for dialog compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "SETTINGS_KEY"}

    def update_from_dict(self, d: dict[str, Any]) -> None:
        """Update settings from a dictionary."""
        for key, value in d.items():
            if hasattr(self, key) and key != "SETTINGS_KEY":
                setattr(self, key, value)

    @classmethod
    def load(cls) -> VisualizationSettings:
        """Load settings from QSettings, using defaults for missing values."""
        settings = QSettings("GreatSky", "visualization")
        instance = cls()

        for f in fields(cls):
            if f.name == "SETTINGS_KEY":
                continue

            default = getattr(instance, f.name)

            # Handle different types appropriately
            if f.type == "str":
                value = _load_str(settings, f.name, default)
            elif f.type == "float":
                value = _load_float(settings, f.name, default)
            elif f.type == "int":
                value = _load_int(settings, f.name, default)
            elif f.type == "bool":
                value = _load_bool(settings, f.name, default)
            elif f.type == "float | None":
                value = _load_optional_float(settings, f.name)
            else:
                # Fallback - try to use type annotation
                value = settings.value(f.name, default)

            setattr(instance, f.name, value)

        return instance

    def save(self) -> None:
        """Save settings to QSettings."""
        settings = QSettings("GreatSky", self.SETTINGS_KEY)

        for f in fields(self):
            if f.name == "SETTINGS_KEY":
                continue

            value = getattr(self, f.name)

            # Handle None values for optional floats
            if value is None:
                settings.setValue(f.name, "")
            else:
                settings.setValue(f.name, value)


# Helper functions for safe QSettings loading

def _load_str(settings: QSettings, key: str, default: str) -> str:
    """Safely load a string setting."""
    try:
        val = settings.value(key, default, type=str)
        return str(val) if val else default
    except (ValueError, TypeError):
        return default


def _load_float(settings: QSettings, key: str, default: float) -> float:
    """Safely load a float setting."""
    try:
        val = settings.value(key, default, type=float)
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _load_int(settings: QSettings, key: str, default: int) -> int:
    """Safely load an int setting."""
    try:
        val = settings.value(key, default, type=int)
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _load_bool(settings: QSettings, key: str, default: bool) -> bool:
    """Safely load a bool setting."""
    try:
        val = settings.value(key, default, type=bool)
        return bool(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _load_optional_float(settings: QSettings, key: str) -> float | None:
    """Safely load an optional float setting."""
    try:
        val = settings.value(key, "", type=str)
        if not val or val.lower() in ("auto", "none", ""):
            return None
        return float(val)
    except (ValueError, TypeError):
        return None

