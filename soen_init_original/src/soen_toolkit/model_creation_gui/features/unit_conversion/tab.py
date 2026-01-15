from __future__ import annotations

import os
import subprocess
import sys

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtGui import QAction, QIcon
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


class UnitConversionTab(QWidget):
    """Embeds the Physical↔Dimensionless Unit Conversion web UI inside the GUI.

    Provides a Launch button that attempts to start the Flask app using the
    user's conda environment (soen_env), then loads it in an embedded
    QWebEngineView at http://127.0.0.1:5001/.
    """

    def __init__(self, manager: ModelManager, *, url: str = "http://127.0.0.1:5001/") -> None:
        super().__init__()
        self._mgr = manager
        self._url = url
        self._web: QWebEngineView | None = None

        layout = QVBoxLayout(self)

        tb = QToolBar("Unit Conversion")

        act_launch = QAction(QIcon(icon("play.svg")), "Launch", self)
        act_launch.setStatusTip("Start the Unit Conversion server (conda soen_env)")
        act_launch.triggered.connect(self._launch_server)
        tb.addAction(act_launch)

        act_open = QAction(QIcon(icon("open_model.svg")), "Open", self)
        act_open.setStatusTip("Open the Unit Conversion page")
        act_open.triggered.connect(self._open_page)
        tb.addAction(act_open)

        act_reload = QAction(QIcon(icon("refresh.svg")), "Reload", self)
        act_reload.triggered.connect(self._reload)
        tb.addAction(act_reload)

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

    def _launch_server(self) -> None:
        """Launch the Unit Conversion Flask app in background and open it.

        Strategy:
        1) Prefer current Python environment: `sys.executable -m soen_toolkit.physical_mappings_gui`.
        2) If that fails, fallback to spawning a login shell with `conda activate soen_env`.
        3) Attempt opening the page with a few retries while the server boots.

        Note: The server will automatically find a free port starting from 5001.
        """
        try:
            # Resolve repo root (when running from source checkout)
            repo_cwd = None
            try:
                import soen_toolkit as _st

                _pkg_dir = os.path.dirname(_st.__file__)
                _parent = os.path.dirname(_pkg_dir)
                if os.path.basename(_parent) == "src":
                    repo_cwd = os.path.dirname(_parent)
            except Exception:
                repo_cwd = None

            # 1) Try current interpreter (no --port flag, so it auto-detects)
            try:
                subprocess.Popen(
                    [sys.executable, "-m", "soen_toolkit.physical_mappings_gui"],
                    cwd=repo_cwd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception:
                # 2) Fallback to conda activation in a login shell
                shell = os.environ.get("SHELL", "/bin/zsh")
                if repo_cwd:
                    cmd = f'cd "{repo_cwd}" && conda activate soen_env && python -m soen_toolkit.physical_mappings_gui'
                else:
                    cmd = "conda activate soen_env && python -m soen_toolkit.physical_mappings_gui"
                subprocess.Popen(
                    [shell, "-lc", cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

            # Try to open with a few staggered retries (server auto-picks port, so try multiple common ports)
            for delay_ms in (1000, 2000, 3000):
                QTimer.singleShot(delay_ms, self._open_page)
        except Exception as e:
            QMessageBox.critical(self, "Launch failed", str(e))
