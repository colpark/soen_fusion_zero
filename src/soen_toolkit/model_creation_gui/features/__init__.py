"""GUI feature tabs for the model creation GUI.

This module intentionally uses lazy imports to avoid circular-import problems.
Several tabs depend on GUI components that also depend (indirectly) on the
ModelManager, which imports the dataset service under this package.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "AnalyseTab",
    "DocumentationTab",
    "DynamicInitTab",
    "HistogramsTab",
    "ModelBuildingTab",
    "ModelIOTab",
    "TextSummaryTab",
    "UnitConversionTab",
    "VisualisationTab",
    "WebViewerTab",
]

_TAB_IMPORTS: dict[str, str] = {
    "AnalyseTab": "soen_toolkit.model_creation_gui.features.analysis",
    "DocumentationTab": "soen_toolkit.model_creation_gui.features.documentation",
    "DynamicInitTab": "soen_toolkit.model_creation_gui.features.dynamic_init",
    "HistogramsTab": "soen_toolkit.model_creation_gui.features.histograms",
    "ModelIOTab": "soen_toolkit.model_creation_gui.features.io",
    "ModelBuildingTab": "soen_toolkit.model_creation_gui.features.model_builder",
    "TextSummaryTab": "soen_toolkit.model_creation_gui.features.summary",
    "UnitConversionTab": "soen_toolkit.model_creation_gui.features.unit_conversion",
    "VisualisationTab": "soen_toolkit.model_creation_gui.features.visualization",
    "WebViewerTab": "soen_toolkit.model_creation_gui.features.web_viewer",
}


def __getattr__(name: str) -> Any:
    module_path = _TAB_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(name)
    module = import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover
    from soen_toolkit.model_creation_gui.features.analysis import AnalyseTab
    from soen_toolkit.model_creation_gui.features.documentation import DocumentationTab
    from soen_toolkit.model_creation_gui.features.dynamic_init import DynamicInitTab
    from soen_toolkit.model_creation_gui.features.histograms import HistogramsTab
    from soen_toolkit.model_creation_gui.features.io import ModelIOTab
    from soen_toolkit.model_creation_gui.features.model_builder import ModelBuildingTab
    from soen_toolkit.model_creation_gui.features.summary import TextSummaryTab
    from soen_toolkit.model_creation_gui.features.unit_conversion import UnitConversionTab
    from soen_toolkit.model_creation_gui.features.visualization import VisualisationTab
    from soen_toolkit.model_creation_gui.features.web_viewer import WebViewerTab



