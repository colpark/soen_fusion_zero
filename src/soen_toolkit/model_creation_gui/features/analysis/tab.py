from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from soen_toolkit.model_creation_gui.features.histograms import HistogramsTab
from soen_toolkit.model_creation_gui.features.summary import TextSummaryTab

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.model_manager import ModelManager


class AnalyseTab(QWidget):
    def __init__(self, manager: ModelManager) -> None:
        super().__init__()
        self._mgr = manager
        self._init_ui()
        # Connect manager signal to both child tabs
        try:
            self._mgr.model_changed.connect(self.summary_tab._on_model_changed)
            self._mgr.model_changed.connect(self.histograms_tab._on_model_changed)
        except Exception:
            # If either child is missing a slot, ignore silently
            pass

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._subtabs = QTabWidget()
        # Child tabs reuse existing functionality
        self.summary_tab = TextSummaryTab(self._mgr)
        self.histograms_tab = HistogramsTab(self._mgr)
        self._subtabs.addTab(self.summary_tab, "Summary")
        self._subtabs.addTab(self.histograms_tab, "Histograms")
        layout.addWidget(self._subtabs)

    def _on_model_changed(self) -> None:
        # Propagate explicitly in case external callers invoke this method
        if hasattr(self.summary_tab, "_on_model_changed"):
            self.summary_tab._on_model_changed()
        if hasattr(self.histograms_tab, "_on_model_changed"):
            self.histograms_tab._on_model_changed()
