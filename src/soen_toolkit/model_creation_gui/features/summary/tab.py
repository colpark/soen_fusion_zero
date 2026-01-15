# src/soen_toolkit/model_creation_gui/tabs/tab_summary.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from soen_toolkit.model_creation_gui.features.state_trajectory.dialog import (
    StateTrajectoryDialog,
)

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.model_manager import ModelManager

log = logging.getLogger(__name__)


class TextSummaryTab(QWidget):
    """Displays styled HTML summary of layers, connections, and parameter counts."""

    def __init__(self, manager: ModelManager) -> None:
        super().__init__()
        self._mgr = manager
        self._init_ui()
        # Connection to manager.model_changed will be done in main_window.py

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Controls row
        controls_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh)
        controls_row.addWidget(self.btn_refresh)

        self.verbose_chk = QCheckBox("Verbose")
        self.verbose_chk.setChecked(True)
        controls_row.addWidget(self.verbose_chk)

        self.plot_btn = QPushButton("Plot Trajectory…")
        self.plot_btn.clicked.connect(self._show_trajectory_dialog)
        controls_row.addWidget(self.plot_btn)

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(self._export_csv)
        controls_row.addWidget(self.btn_export_csv)

        self.btn_export_json = QPushButton("Export JSON")
        self.btn_export_json.clicked.connect(self._export_json)
        controls_row.addWidget(self.btn_export_json)

        controls_row.addStretch()
        layout.addLayout(controls_row)

        # Split view: left KPIs/list, right HTML details
        self.splitter = QSplitter()

        self.kpi_list = QListWidget()
        self.kpi_list.setMinimumWidth(260)
        self.splitter.addWidget(self.kpi_list)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.splitter.addWidget(self.browser)
        self.splitter.setStretchFactor(1, 1)
        layout.addWidget(self.splitter, 1)

        # Initial refresh handled by _on_model_changed

    def _on_model_changed(self) -> None:
        """Slot called when the model manager signals a model change."""
        log.debug("TextSummaryTab reacting to model change.")
        # Clear the browser if the model is None (e.g., after an error)
        if self._mgr.model is None:
            self.browser.clear()
            self.plot_btn.setEnabled(False)  # Disable plot button if no model
        else:
            self.plot_btn.setEnabled(True)
            # Automatically refresh the summary text when the model changes.
            # This seems reasonable as summary generation should be fast.
            self._refresh()

    def _refresh(self) -> None:
        """Generates and displays the HTML summary."""
        import pandas as pd  # Lazy import: only when refresh is called

        if self._mgr.model is None:
            # This check might be redundant if _on_model_changed clears,
            # but good for direct calls to _refresh.
            self.browser.setHtml("<p><i>No model loaded or built.</i></p>")
            self.plot_btn.setEnabled(False)
            # QMessageBox.warning(self, "No model", "Build or load a model first.")
            return

        self.plot_btn.setEnabled(True)  # Ensure enabled if model exists
        try:
            # Compute unified summary
            info = self._mgr.model.compute_summary()
            # Still get layer_df for verbose tables if requested
            layer_df = self._mgr.model.summary(
                return_df=True,
                print_summary=False,
                verbose=self.verbose_chk.isChecked(),
            )
            conn_df = pd.DataFrame(info.get("connections", []))

            # Compute parameter counts
            model = self._mgr.model
            total_params = sum(p.numel() for p in model.parameters())
            learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_learnable_params = total_params - learnable_params

            # Calculate non-zero counts
            total_nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
            nonzero_learnable_params = sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)
            nonzero_non_learnable_params = sum((p != 0).sum().item() for p in model.parameters() if not p.requires_grad)

            # Create a more detailed and clearly labelled DataFrame
            kpis = info.get("kpis", {})
            mask_aware_trainable = kpis.get("trainable_parameters", learnable_params)
            counts_data = [
                {"Metric": "Total Parameters (incl. zeros)", "Count": f"{total_params:,}"},
                {"Metric": "Total Parameters (non-zero)", "Count": f"{total_nonzero_params:,}"},
                {"Metric": "Trainable Parameters (mask-aware)", "Count": f"{mask_aware_trainable:,}"},
                {"Metric": "Trainable Parameters (non-zero)", "Count": f"{nonzero_learnable_params:,}"},
                {"Metric": "Non-Trainable Parameters (total)", "Count": f"{non_learnable_params:,}"},
                {"Metric": "Non-Trainable Parameters (non-zero)", "Count": f"{nonzero_non_learnable_params:,}"},
            ]
            counts_df = pd.DataFrame(counts_data)

            # Build responsive HTML
            html_parts = [
                "<!DOCTYPE html>",
                '<html><head><meta charset="utf-8">',
                "<style>",
                "body { font-family: Arial, sans-serif; padding: 1em; }",
                "h2 { color: #2c3e50; }",
                "table { border-collapse: collapse; width: 100%; margin-bottom: 1.5em; }",
                "th, td { border: 1px solid #555; padding: 0.5em; text-align: left; }",
                "th { background-color: #34495e; color: #ffffff; }",
                "tr:nth-child(even) { background-color: rgba(255,255,255,0.05); }",
                "tr:nth-child(odd)  { background-color: rgba(0,0,0,0.05); }",
                "</style></head><body>",
                "<h1>SOEN Model Summary</h1>",
            ]
            # KPI Header
            k = info.get("kpis", {})
            kpi_rows = "".join(
                [
                    f"<tr><td><b>Layers</b></td><td>{k.get('layers', '')}</td></tr>",
                    f"<tr><td><b>Connections</b></td><td>{k.get('connections', '')}</td></tr>",
                    f"<tr><td><b>Parameters</b></td><td>{k.get('parameters', '')}</td></tr>",
                    f"<tr><td><b>Trainable</b></td><td>{k.get('trainable_parameters', '')}</td></tr>",
                    f"<tr><td><b>Non‑zero</b></td><td>{k.get('nonzero_parameters', '')}</td></tr>",
                ]
            )
            html_parts.extend(["<h2>Overview</h2>", f"<table>{kpi_rows}</table>"])

            # Simulation settings table
            from dataclasses import asdict

            sim_dict = asdict(self._mgr.sim_config)
            sim_df = pd.DataFrame([{"Setting": k, "Value": v} for k, v in sim_dict.items()])

            if not sim_df.empty:
                html_parts.extend(["<h2>Simulation Settings</h2>", sim_df.to_html(index=False, escape=False, classes="sim-table")])

            # Only add other tables if data exists
            if not layer_df.empty:
                html_parts.extend(["<h2>Layers</h2>", layer_df.to_html(index=False, escape=False, classes="layer-table")])
            if not conn_df.empty:
                html_parts.extend(["<h2>Connections</h2>", conn_df.to_html(index=False, escape=False, classes="conn-table")])
            if not counts_df.empty:
                html_parts.extend(["<h2>Parameter Counts</h2>", counts_df.to_html(index=False, escape=False, classes="param-counts-table")])

            html_parts.extend(["</body></html>"])
            full_html = "\n".join(html_parts)
            self.browser.setHtml(full_html)
            # Update KPI list panel too
            self._update_kpi_list(info)

        except Exception as e:
            log.exception("Error generating model summary HTML.")
            self.browser.setHtml(f"<p><b>Error generating summary:</b></p><pre>{e}</pre>")
            QMessageBox.critical(self, "Summary error", f"Could not generate summary: {e}")
            self.plot_btn.setEnabled(False)

    def _show_trajectory_dialog(self) -> None:
        """Opens the state trajectory plotting dialog."""
        if self._mgr.model is None:
            QMessageBox.warning(self, "No model", "Cannot plot trajectory: Build or load a model first.")
            return
        # Pass self (the parent widget) and the manager to the dialog
        dialog = StateTrajectoryDialog(self, self._mgr)
        dialog.exec()

    def _update_kpi_list(self, info: dict) -> None:
        self.kpi_list.clear()
        k = info.get("kpis", {})
        degrees = info.get("degrees", {})
        # KPI block
        for key, label in [
            ("layers", "Layers"),
            ("connections", "Connections"),
            ("parameters", "Parameters"),
            ("trainable_parameters", "Trainable"),
            ("nonzero_parameters", "Non‑zero"),
        ]:
            self.kpi_list.addItem(QListWidgetItem(f"{label}: {k.get(key, ''):,}" if isinstance(k.get(key), int) else f"{label}: {k.get(key, '')}`"))
        # Degrees per layer
        if degrees:
            self.kpi_list.addItem(QListWidgetItem(""))
            self.kpi_list.addItem(QListWidgetItem("Per‑layer Degrees:"))
            for lid in sorted(degrees):
                d = degrees[lid]
                self.kpi_list.addItem(QListWidgetItem(f"Layer {lid}: in={d.get('in', 0)} out={d.get('out', 0)}"))

    def _export_csv(self) -> None:
        if self._mgr.model is None:
            return
        try:
            from PyQt6.QtWidgets import QFileDialog

            fname, _ = QFileDialog.getSaveFileName(self, "Export summary CSV", "", "CSV (*.csv)")
            if not fname:
                return
            df = self._mgr.model.summary(return_df=True, print_summary=False, verbose=self.verbose_chk.isChecked())
            df.to_csv(fname, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Export", str(e))

    def _export_json(self) -> None:
        if self._mgr.model is None:
            return
        try:
            import json

            from PyQt6.QtWidgets import QFileDialog

            fname, _ = QFileDialog.getSaveFileName(self, "Export summary JSON", "", "JSON (*.json)")
            if not fname:
                return
            info = self._mgr.model.compute_summary()
            with open(fname, "w") as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Export", str(e))
