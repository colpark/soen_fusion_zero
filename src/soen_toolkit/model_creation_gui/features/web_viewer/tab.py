from __future__ import annotations

import json
import os

from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMessageBox, QToolBar, QVBoxLayout, QWidget

try:
    # Optional dependency – handled gracefully if not installed
    from PyQt6.QtWebEngineCore import QWebEngineDownloadRequest
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover
    QWebEngineView = None  # type: ignore
    QWebEngineDownloadRequest = None  # type: ignore

import contextlib
from typing import TYPE_CHECKING

from soen_toolkit.model_creation_gui.utils.paths import icon
from soen_toolkit.utils.model_tools import export_model_to_json

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.model_manager import ModelManager


class WebViewerTab(QWidget):
    """Embeds the external SOEN web viewer inside the GUI.

    Integration contract with the web app:
    - The page should listen for a CustomEvent named 'soen-json'.
      Example in the web app:
        window.addEventListener('soen-json', (e) => {
          const data = e.detail
          // JSON object
          // ... render graph here ...
        });

    - We will inject the current model JSON via that event when the user
      clicks "Send Model".

    - URL to load is taken from SOEN_VIEWER_URL env var, defaulting to
      http://localhost:5173/ (vite dev server). You can also point this to a
      file:// URL of a production build's index.html.
    """

    def __init__(self, mgr: ModelManager) -> None:
        super().__init__()
        self._mgr = mgr
        self._url = os.environ.get("SOEN_VIEWER_URL", "http://localhost:5173/")

        layout = QVBoxLayout(self)

        tb = QToolBar("Web Viewer")

        act_open = QAction(QIcon(icon("open_model.svg")), "Open Viewer", self)
        act_open.setStatusTip("Load the SOEN web viewer page")
        act_open.triggered.connect(self._open_viewer)
        tb.addAction(act_open)

        act_reload = QAction(QIcon(icon("refresh.svg")), "Reload", self)
        act_reload.setStatusTip("Reload the viewer page")
        act_reload.triggered.connect(self._reload)
        tb.addAction(act_reload)

        act_send = QAction(QIcon(icon("send.svg")), "Send Model", self)
        act_send.setStatusTip("Serialize current model to JSON and send to the viewer")
        act_send.triggered.connect(self._send_model)
        tb.addAction(act_send)

        act_export = QAction(QIcon(icon("save.svg")), "Export JSON", self)
        act_export.setStatusTip("Fetch the current JSON from the viewer and save to a file")
        act_export.triggered.connect(self._export_json)
        tb.addAction(act_export)

        layout.addWidget(tb)

        if QWebEngineView is None:
            # Missing runtime dependency – give a friendly message
            from PyQt6.QtWidgets import QLabel

            msg = QLabel(
                "PyQt6-WebEngine is not installed. Install with: pip install PyQt6-WebEngine",
            )
            msg.setWordWrap(True)
            layout.addWidget(msg)
            self._web = None
            return

        self._web = QWebEngineView()
        layout.addWidget(self._web, 1)

        # When a load fails (e.g., dev server not running), show helpful instructions
        with contextlib.suppress(Exception):
            self._web.loadFinished.connect(self._on_load_finished)

        # Hook downloads triggered from the embedded page (e.g., export button)
        try:
            prof = self._web.page().profile()
            prof.downloadRequested.connect(self._on_download_requested)
        except Exception:
            pass

        # Track whether the help page is currently shown
        self._showing_help = False

        # Auto-load on first show
        self._open_viewer()

    # ---- actions ----
    def _open_viewer(self) -> None:
        if not self._web:
            QMessageBox.warning(self, "WebEngine unavailable", "PyQt6-WebEngine is not installed.")
            return
        self._web.setUrl(self._to_qurl(self._url))

    def _reload(self) -> None:
        if not self._web:
            return
        # If we're showing the help page, navigate to the target URL instead of reloading the help
        if getattr(self, "_showing_help", False):
            self._open_viewer()
        else:
            self._web.reload()

    def _send_model(self) -> None:
        if not self._web:
            QMessageBox.warning(self, "WebEngine unavailable", "PyQt6-WebEngine is not installed.")
            return
        if self._mgr.model is None:
            QMessageBox.information(self, "No model", "Build or load a model first.")
            return
        try:
            # Export JSON and keep it so we can re-inject after a reload
            json_text = export_model_to_json(self._mgr.model, filename=None)
            self._last_json = json_text
            # Prefer a direct call if the page provided a helper; otherwise fire the event
            script = (
                "(function(){\n"
                "  const data = " + json_text + ";\n"
                "  if (typeof window.injectSoenJson === 'function') { window.injectSoenJson(data); }\n"
                "  else { window.dispatchEvent(new CustomEvent('soen-json', { detail: data })); }\n"
                "})();"
            )
            self._web.page().runJavaScript(script)
        except Exception as e:
            QMessageBox.critical(self, "Send failed", str(e))

    # ---- helpers ----
    def _to_qurl(self, s: str):
        from PyQt6.QtCore import QUrl

        # Heuristic: if it's an existing local path, use file://
        if os.path.exists(s) and not (s.startswith(("http://", "https://", "file://"))):
            return QUrl.fromLocalFile(os.path.abspath(s))
        return QUrl(s)

    # External signal hook (from ModelManager)
    def _on_model_changed(self) -> None:  # keep same signature as other tabs
        # No auto-send on change; user triggers explicitly to avoid clobbering their view
        pass

    def _export_json(self) -> None:
        if not self._web:
            QMessageBox.warning(self, "WebEngine unavailable", "PyQt6-WebEngine is not installed.")
            return
        try:

            def _got(text: str) -> None:
                try:
                    from PyQt6.QtWidgets import QFileDialog

                    path, _ = QFileDialog.getSaveFileName(self, "Save SOEN JSON", os.path.expanduser("~"), "JSON (*.json)")
                    if not path:
                        return
                    if not path.endswith(".json"):
                        path += ".json"
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(text or "{}")
                    QMessageBox.information(self, "Saved", f"JSON saved to:\n{path}")
                except Exception as e:
                    QMessageBox.critical(self, "Save failed", str(e))

            self._web.page().runJavaScript("(window.getSoenJson && window.getSoenJson()) || '{}'", _got)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _on_download_requested(self, req) -> None:  # QWebEngineDownloadRequest
        try:
            # Some Qt versions pass QWebEngineDownloadItem; unify by attribute access
            suggested = getattr(req, "downloadFileName", None)
            if callable(suggested):
                suggested = req.downloadFileName()
            elif isinstance(suggested, str):
                pass
            else:
                suggested = "soen_model.json"
            from PyQt6.QtWidgets import QFileDialog

            path, _ = QFileDialog.getSaveFileName(self, "Save File", os.path.join(os.path.expanduser("~"), suggested), "All Files (*)")
            if not path:
                return
            # QWebEngine 6 API
            if hasattr(req, "setDownloadDirectory") and hasattr(req, "setDownloadFileName") and hasattr(req, "accept"):
                req.setDownloadDirectory(os.path.dirname(path))
                req.setDownloadFileName(os.path.basename(path))
                req.accept()
                return
            # Fallback older API: try setting path attribute and accept
            if hasattr(req, "setPath"):
                req.setPath(path)
            if hasattr(req, "accept"):
                req.accept()
        except Exception:
            pass

    # ---- load error handling ----
    def _on_load_finished(self, ok: bool) -> None:
        # Clear help flag on success
        if ok:
            self._showing_help = False
            # Re-inject last JSON if available so manual reloads keep the current model
            try:
                if getattr(self, "_last_json", None):
                    script = (
                        "(function(){\n"
                        "  const data = " + self._last_json + ";\n"
                        "  if (typeof window.injectSoenJson === 'function') { window.injectSoenJson(data); }\n"
                        "  else { window.dispatchEvent(new CustomEvent('soen-json', { detail: data })); }\n"
                        "})();"
                    )
                    self._web.page().runJavaScript(script)
            except Exception:
                pass
            return
        # Replace default error page with a friendly guide
        viewer_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "web_view",
            "soen-graph",
        )
        cmd_block = f"cd {viewer_dir}\nnpm install\nnpm run dev -- --port 5173 --strictPort\n"
        # When user hits Reload, go to the intended URL (not refresh the help page)
        target_url = self._url if isinstance(self._url, str) else "http://localhost:5173/"
        # Make a compact help page with a reload button
        help_html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>SOEN Web Viewer – Not Running</title>
  <style>
    body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif
    margin: 32px
    color: #333
    }}
    code, pre {{ background: #f6f8fa
    border: 1px solid #e1e4e8
    border-radius: 6px
    padding: 8px 10px
    }}
    pre {{ white-space: pre-wrap
    }}
    .box {{ max-width: 820px
    }}
    .btn {{ display: inline-block
    margin-top: 12px
    padding: 8px 14px
    background: #2563eb
    color: white
    border-radius: 6px
    text-decoration: none
    }}
    .btn:active {{ transform: translateY(1px)
    }}
  </style>
  <script>
    function reloadPage() {{ window.location.href = {json.dumps(target_url)}
    }}
  </script>
  </head>
  <body>
    <div class=\"box\">
      <h2>SOEN Web Viewer isn’t running</h2>
      <p>The GUI tried to load <code>http://localhost:5173/</code> but it wasn’t reachable.</p>
      <p>Start the viewer dev server in a terminal, then click Reload:</p>
      <pre>{cmd_block}</pre>
      <p>If you use conda, ensure the right env is active first:</p>
      <pre>conda activate soen_env</pre>
      <a class=\"btn\" href=\"#\" onclick=\"reloadPage()\">Reload</a>
    </div>
  </body>
</html>
"""
        try:
            self._web.setHtml(help_html)
            self._showing_help = True
        except Exception:
            # As a fallback, show a dialog
            with contextlib.suppress(Exception):
                QMessageBox.information(
                    self,
                    "SOEN Web Viewer",
                    "Viewer not reachable at http://localhost:5173/.\n\nIn a terminal:\n\n" + cmd_block,
                )
