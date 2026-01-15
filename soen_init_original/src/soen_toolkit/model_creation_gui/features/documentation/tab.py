from __future__ import annotations

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QIcon
from PyQt6.QtWidgets import QMessageBox, QToolBar, QVBoxLayout, QWidget

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover
    QWebEngineView = None  # type: ignore

import contextlib
from typing import TYPE_CHECKING

from soen_toolkit.model_creation_gui.utils.paths import icon

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.model_manager import ModelManager


class DocumentationTab(QWidget):
    """Embeds the SOEN Toolkit documentation website inside the GUI.

    Simple web embed similar to Unit Conversion, but points to the docs site.
    """

    def __init__(self, manager: ModelManager, *, url: str = "https://greatsky-ai.github.io/soen-toolkit") -> None:
        super().__init__()
        self._mgr = manager
        self._url = url
        self._web: QWebEngineView | None = None

        layout = QVBoxLayout(self)

        tb = QToolBar("Documentation")

        act_open = QAction(QIcon(icon("open_model.svg")), "Open", self)
        act_open.setStatusTip("Open the documentation website")
        act_open.triggered.connect(self._open_page)
        tb.addAction(act_open)

        act_reload = QAction(QIcon(icon("refresh.svg")), "Reload", self)
        act_reload.triggered.connect(self._reload)
        tb.addAction(act_reload)

        act_open_ext = QAction("Open in Browser", self)
        act_open_ext.setStatusTip("Open the documentation in your default browser")
        act_open_ext.triggered.connect(self._open_in_browser)
        tb.addAction(act_open_ext)

        layout.addWidget(tb)

        if QWebEngineView is None:
            # Missing runtime dependency – give a friendly message
            from PyQt6.QtWidgets import QLabel

            msg = QLabel(
                "PyQt6-WebEngine is not installed. Install with: pip install PyQt6-WebEngine",
            )
            msg.setWordWrap(True)
            layout.addWidget(msg, 1)
            return

        self._web = QWebEngineView()
        layout.addWidget(self._web, 1)

    # ----- actions -----
    def _open_page(self) -> None:
        if not self._web:
            QMessageBox.warning(self, "WebEngine unavailable", "PyQt6-WebEngine is not installed.")
            return
        self._web.setUrl(self._to_qurl(self._url))

    def _reload(self) -> None:
        if self._web and self._web.url().isValid():
            with contextlib.suppress(Exception):
                self._web.reload()
        else:
            self._open_page()

    def _to_qurl(self, url: str) -> QUrl:
        try:
            return QUrl(url)
        except Exception:
            return QUrl.fromUserInput(url)

    def _open_in_browser(self) -> None:
        with contextlib.suppress(Exception):
            QDesktopServices.openUrl(self._to_qurl(self._url))


__all__ = ["DocumentationTab"]
