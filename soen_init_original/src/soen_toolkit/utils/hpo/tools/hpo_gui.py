#!/usr/bin/env python3
"""SOEN Criticality HPO GUI (PyQt6).

MVP features:
- Load model spec (YAML/JSON), optionally a base model for trial-params files.
- Load an existing HPO YAML to pre-populate settings.
- Configure run/simulation/input/objective/optuna/pruner settings.
- Choose target layers/connections, toggle enabled components.
- Save HPO YAML.
- Launch optimization (scripts/run_hpo.py) and stream logs with simple progress.
- Open/view resulting dashboard when complete (inline viewer if PyQt6-WebEngine available).

Future: rich per-parameter editors using enumerate_model_options schema.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableView,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Reuse robust visualisation stack from model_creation_gui
try:
    from soen_toolkit.model_creation_gui.model_manager import (
        ModelManager as _MCG_ModelManager,
    )
    from soen_toolkit.model_creation_gui.tabs.tab_visualisation import (
        VisualisationTab as _MCG_VisualisationTab,
    )
except Exception:
    _MCG_ModelManager = None
    _MCG_VisualisationTab = None
from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QProcess,
    QSortFilterProxyModel,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QPalette

# Optional web engine for modern plots
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except Exception:
    QWebEngineView = None
import signal

# HTML dashboard embedding removed for now.
import yaml

# Optional deps for in-app results
try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    msg = "pandas is required for the HPO GUI Results pane. Please install pandas."
    raise ImportError(msg) from e
try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None
try:
    import plotly.graph_objs as _go
    import plotly.io as _pio
except Exception:
    _go = None
    _pio = None
try:
    import plotly.express as px
except Exception:
    px = None

# study loader + processor for summarizing results (required)
import contextlib

from soen_toolkit.utils.hpo.tools.criticality_data_processor import (
    CriticalityDataProcessor,
    load_study_from_directory,
)


def _fmt_float(x: Any, default: str = "—") -> str:
    """Compact float formatting for summary text."""
    try:
        xv = float(x)
        if xv != xv:  # NaN check
            return default
        return f"{xv:.6g}"
    except Exception:
        return default


class _PandasModel(QAbstractTableModel):
    """Very small DataFrame → Qt model bridge for read-only tables."""

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df.copy() if df is not None else (pd.DataFrame() if pd is not None else None)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if pd is None or self.df is None:
            return 0
        return int(self.df.shape[0])

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if pd is None or self.df is None:
            return 0
        return int(self.df.shape[1])

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or pd is None or self.df is None:
            return None
        try:
            val = self.df.iat[index.row(), index.column()]
            if isinstance(val, float):
                return f"{val:.6g}"
            return str(val)
        except Exception:
            return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or pd is None or self.df is None:
            return None
        if orientation == Qt.Orientation.Horizontal:
            try:
                return str(self.df.columns[section])
            except Exception:
                return None
        else:
            return str(section)


class ResultsViewer(QWidget):
    """Embedded analysis dashboard split into sub-tabs.

    Tabs: Overview (KPIs + summary), Plots (performance, scatter, importance), Trials (table).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        vroot = QVBoxLayout(self)
        vroot.setContentsMargins(0, 0, 0, 0)
        vroot.setSpacing(8)

        self.tabs = QTabWidget()
        vroot.addWidget(self.tabs, 1)

        # ——— Overview tab ———
        pg_overview = QWidget()
        ov = QVBoxLayout(pg_overview)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.setSpacing(8)
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(12)
        ov.addLayout(kpi_row)

        def _kpi_box(title: str) -> tuple[QGroupBox, QLabel]:
            box = QGroupBox(title)
            # Provide enough room to prevent title chip overlap and value truncation
            box.setMinimumWidth(150)
            box.setMaximumHeight(72)
            box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            lay = QVBoxLayout(box)
            lay.setContentsMargins(10, 10, 10, 10)
            lbl = QLabel("—")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 16px; font-weight: 600;")
            lay.addWidget(lbl)
            return box, lbl

        self._kpi_best_box, self._kpi_best = _kpi_box("Best Value")
        self._kpi_trials_box, self._kpi_trials = _kpi_box("Trials")
        self._kpi_complete_box, self._kpi_complete = _kpi_box("Complete")
        self._kpi_pruned_box, self._kpi_pruned = _kpi_box("Pruned")
        self._kpi_failed_box, self._kpi_failed = _kpi_box("Failed")
        self._kpi_elapsed_box, self._kpi_elapsed = _kpi_box("Elapsed (s)")
        for b in (self._kpi_best_box, self._kpi_trials_box, self._kpi_complete_box, self._kpi_pruned_box, self._kpi_failed_box, self._kpi_elapsed_box):
            kpi_row.addWidget(b)
        kpi_row.addStretch(1)
        self.txt_summary = QPlainTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setPlainText("No study loaded.")
        self.txt_summary.setMaximumHeight(120)
        ov.addWidget(self.txt_summary)
        # Metrics summary box (Overview tab)
        metrics_box = QGroupBox("Metrics Summary")
        mlay = QVBoxLayout(metrics_box)
        self.txt_metrics_overview = QPlainTextEdit()
        self.txt_metrics_overview.setReadOnly(True)
        self.txt_metrics_overview.setMaximumHeight(140)
        mlay.addWidget(self.txt_metrics_overview)
        ov.addWidget(metrics_box)
        self.lbl_refresh = QLabel("")
        self.lbl_refresh.setStyleSheet("color: #666;")
        ov.addWidget(self.lbl_refresh)
        self.tabs.addTab(pg_overview, "Overview")

        # ——— Plots tab ———
        pg_plots = QWidget()
        pv = QVBoxLayout(pg_plots)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(8)
        controls = QHBoxLayout()
        pv.addLayout(controls)
        self.cmb_x = QComboBox()
        self.cmb_x.setMinimumWidth(200)
        self.cmb_x.setToolTip("X axis column (any trial/param/metric)")
        self.cmb_y = QComboBox()
        self.cmb_y.setMinimumWidth(180)
        self.cmb_y.setToolTip("Y axis column (numeric recommended)")
        # Back-compat aliases for existing methods
        self.cmb_param = self.cmb_x
        self.cmb_metric = self.cmb_y
        # Regression controls for scatter
        self.chk_regression = QCheckBox("Regression")
        self.chk_regression.setToolTip("Show regression fit on scatter (numeric X)")
        self.cmb_regression = QComboBox()
        self.cmb_regression.setToolTip("Regression method")
        self.cmb_regression.addItem("Linear")
        with contextlib.suppress(Exception):
            self.cmb_regression.addItem("LOWESS")
        self.sb_topn = QSpinBox()
        self.sb_topn.setRange(3, 50)
        self.sb_topn.setValue(15)
        self.sb_topn.setToolTip("Top-N parameters for importance")
        self.btn_export_current = QPushButton("Export Current Plot…")
        controls.addWidget(QLabel("X:"))
        controls.addWidget(self.cmb_x)
        controls.addWidget(QLabel("Y:"))
        controls.addWidget(self.cmb_y)
        controls.addWidget(self.chk_regression)
        controls.addWidget(self.cmb_regression)
        controls.addWidget(QLabel("Top-N:"))
        controls.addWidget(self.sb_topn)
        controls.addStretch(1)
        controls.addWidget(self.btn_export_current)

        self.plots_tabs = QTabWidget()
        pv.addWidget(self.plots_tabs, 1)

        # Subtabs with Plotly web views (fallbacks to PyQtGraph if needed)
        def _make_tab():
            w = QWidget()
            item = QVBoxLayout(w)
            item.setContentsMargins(0, 0, 0, 0)
            item.setSpacing(6)
            vw = None
            if (QWebEngineView is not None) and (_go is not None) and (_pio is not None):
                try:
                    vw = QWebEngineView()
                    item.addWidget(vw, 1)
                except Exception:
                    vw = None
            return w, item, vw

        scatter_w, scatter_l, self.web_scatter = _make_tab()
        imp_w, imp_l, self.web_importance = _make_tab()
        # Removed Parallel and Correlations per request

        # Fallbacks for PyQtGraph if web unavailable
        self.pg_scatter = None
        self.pg_importance = None
        if (self.web_scatter is None or self.web_importance is None) and pg is not None:
            try:
                if self.web_scatter is None:
                    self.pg_scatter = pg.PlotWidget(title="Metric vs Parameter")
                    scatter_l.addWidget(self.pg_scatter, 1)
                if self.web_importance is None:
                    self.pg_importance = pg.PlotWidget(title="Parameter Importance (|corr|)")
                    imp_l.addWidget(self.pg_importance, 1)
            except Exception:
                self.pg_scatter = self.pg_importance = None

        self.plots_tabs.addTab(scatter_w, "Scatter")
        self.plots_tabs.addTab(imp_w, "Importance")
        # Parallel and Correlations tabs removed

        # Add Plots tab
        self.tabs.addTab(pg_plots, "Plots")

        # ——— Trials tab ———
        pg_trials = QWidget()
        tv = QVBoxLayout(pg_trials)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(8)
        tbar = QHBoxLayout()
        tv.addLayout(tbar)
        self.le_filter = QLineEdit()
        self.le_filter.setPlaceholderText("Filter table (regex)…")
        self.btn_export_csv = QPushButton("Export Table CSV…")
        tbar.addWidget(QLabel("Filter:"))
        tbar.addWidget(self.le_filter, 1)
        tbar.addWidget(self.btn_export_csv)
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        tv.addWidget(self.table_view, 1)
        self.tabs.addTab(pg_trials, "Trials")

        # Hook control changes to re-render
        try:
            self.cmb_x.currentTextChanged.connect(lambda _=None: self._render_plots_from_df())
            self.cmb_y.currentTextChanged.connect(lambda _=None: self._render_plots_from_df())
            self.chk_regression.toggled.connect(lambda _=None: self._replot_scatter())
            self.cmb_regression.currentTextChanged.connect(lambda _=None: self._replot_scatter())
            self.sb_topn.valueChanged.connect(lambda _=None: self._replot_importance(self._last_importance or {}))
            self.btn_export_current.clicked.connect(self._export_current_plot)
            self.le_filter.textChanged.connect(self._apply_table_filter)
            self.btn_export_csv.clicked.connect(self._export_csv)
        except Exception:
            pass

    def _render_plots_from_df(self) -> None:
        try:
            df = getattr(self, "_df", None)
            if df is not None and not df.empty:
                self._populate_selectors(df)
            # Re-render subtabs
            self._replot_scatter()
            with contextlib.suppress(Exception):
                pass
        except Exception:
            pass

    def _export_current_plot(self) -> None:
        try:
            # Export currently selected Plotly tab to HTML using cached figures
            tab_name = self.plots_tabs.tabText(self.plots_tabs.currentIndex())
            view = None
            if tab_name == "Scatter":
                view = self.web_scatter
            elif tab_name == "Importance":
                view = self.web_importance
            if view is None or _pio is None:
                QMessageBox.information(self, "Export", "Export requires the web plot backend.")
                return
            p, _ = QFileDialog.getSaveFileName(self, "Export plot as…", str(Path.cwd() / f"hpo_{tab_name.lower()}.html"), "HTML (*.html)")
            if not p:
                return
            fig = (self._last_figs or {}).get(tab_name)
            if fig is None:
                # Render to refresh cache
                if tab_name == "Scatter":
                    self._replot_scatter()
                elif tab_name == "Importance":
                    self._replot_importance(self._last_importance or {})
                fig = (self._last_figs or {}).get(tab_name)
            if fig is None:
                QMessageBox.information(self, "Export", "Nothing to export.")
                return
            try:
                _pio.write_html(fig, file=p, include_plotlyjs="cdn", full_html=True)
                QMessageBox.information(self, "Export", f"Saved: {p}")
            except Exception as e:
                QMessageBox.warning(self, "Export", f"Failed to export: {e}")
        except Exception:
            pass

        # ——— Trials tab ———
        pg_trials = QWidget()
        tv = QVBoxLayout(pg_trials)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(8)
        tbar = QHBoxLayout()
        tv.addLayout(tbar)
        self.le_filter = QLineEdit()
        self.le_filter.setPlaceholderText("Filter table (regex)…")
        self.btn_export_csv = QPushButton("Export Table CSV…")
        tbar.addWidget(QLabel("Filter:"))
        tbar.addWidget(self.le_filter, 1)
        tbar.addWidget(self.btn_export_csv)
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        tv.addWidget(self.table_view, 1)
        self.tabs.addTab(pg_trials, "Trials")

        # Signals
        self.sb_topn.valueChanged.connect(self._replot_importance)
        self.cmb_y.currentTextChanged.connect(self._replot_scatter)
        self.cmb_x.currentTextChanged.connect(self._replot_scatter)
        self.chk_regression.toggled.connect(self._replot_scatter)
        self.cmb_regression.currentTextChanged.connect(self._replot_scatter)
        self.le_filter.textChanged.connect(self._apply_table_filter)
        self.btn_export_csv.clicked.connect(self._export_csv)

        # State
        self._last_dir: str | None = None
        self._df: pd.DataFrame | None = None
        self._table_proxy: QSortFilterProxyModel | None = None
        self._cat_maps: dict[str, list[str]] = {}
        self._table_model: _PandasModel | None = None
        self._scatter_perf = None
        self._scatter_param = None
        self._last_figs: dict[str, dict] = {}

    def _set_metrics_summary(self, text: str) -> None:
        """Set the metrics summary text in both locations, if present."""
        try:
            if hasattr(self, "txt_metrics_overview") and self.txt_metrics_overview is not None:
                self.txt_metrics_overview.setPlainText(text)
        except Exception:
            pass
        try:
            if hasattr(self, "txt_metrics_plots") and self.txt_metrics_plots is not None:
                self.txt_metrics_plots.setPlainText(text)
        except Exception:
            pass

    def load_study_dir(self, study_dir: str) -> None:
        """Load results from a study directory and render into widgets."""
        if not study_dir or not os.path.isdir(study_dir):
            self.txt_summary.setPlainText("Results: invalid study directory.")
            return
        # Required dependency: processor must import
        try:
            study, cfg = load_study_from_directory(study_dir)
            proc = CriticalityDataProcessor(study, cfg or {})
            data = proc.process_all()
        except Exception:
            self.txt_summary.setPlainText("Failed to load study. See log.")
            return

        # Summary + KPIs
        self._render_summary_text(data)
        self._update_kpis(data)

        # Trials table
        df = data.get("trials_df")
        if df is not None:
            # Construct a compact table: keep core columns and params/metrics up to a cap
            base_cols = [c for c in ["trial_number", "state", "value", "duration", "rank"] if c in df.columns]
            param_cols = [c for c in df.columns if c.startswith("param_")][:20]
            metric_cols = [c for c in df.columns if c.startswith("metric_")][:10]
            show = df[base_cols + param_cols + metric_cols].copy()
            self._df = show
            model = _PandasModel(show)
            proxy = QSortFilterProxyModel(self)
            proxy.setSourceModel(model)
            proxy.setDynamicSortFilter(True)
            proxy.setFilterKeyColumn(-1)  # search all columns
            self._table_proxy = proxy
            self._table_model = model
            self.table_view.setModel(proxy)
            self.table_view.resizeColumnsToContents()
            self._populate_selectors(show)

        # (History plot removed from Analysis; live plot is in Run tab)

        # Plots
        self._plot_all(data)

        # (Pairwise and Parallel plots removed)

        # Metrics summary text
        try:
            crit = data.get("criticality_metrics")
            if crit is not None:
                import numpy as np

                def _stats(vals) -> str:
                    a = np.array([v for v in (vals or []) if np.isfinite(v)], dtype=float)
                    if a.size == 0:
                        return "(no data)"
                    return f"mean={a.mean():.4g}, std={a.std():.4g}, min={a.min():.4g}, max={a.max():.4g}"

                # Show only metrics with non-zero weights; always include total_cost
                cfg = data.get("config") or {}
                w = cfg.get("weights") or cfg.get("objective", {}).get("weights") or cfg

                def _w(name: str) -> float:
                    try:
                        return float((w or {}).get(name, 0.0))
                    except Exception:
                        return 0.0

                lines = []
                if _w("w_branch") > 0.0:
                    lines.append(f"branching_sigma: {_stats(crit.branching_sigma)}")
                if _w("w_psd_t") > 0.0:
                    lines.append(f"psd_temporal_cost: {_stats(crit.psd_temporal_cost)}")
                if _w("w_psd_spatial") > 0.0:
                    lines.append(f"psd_spatial_cost: {_stats(crit.psd_spatial_cost)}")
                if _w("w_chi_inv") > 0.0:
                    lines.append(f"chi_inv_cost: {_stats(crit.chi_inv_cost)}")
                if _w("w_autocorr") > 0.0:
                    lines.append(f"autocorr_cost: {_stats(crit.autocorr_cost)}")
                if _w("w_jacobian") > 0.0:
                    # Prefer jacobian_cost if present; fall back to spectral radius summary
                    if hasattr(crit, "jacobian_cost"):
                        lines.append(f"jacobian_cost: {_stats(getattr(crit, 'jacobian_cost', []))}")
                if _w("w_lyapunov") > 0.0:
                    # Show per-step and per-second estimates; cost if available
                    if hasattr(crit, "lyapunov_cost"):
                        lines.append(f"lyapunov_cost: {_stats(getattr(crit, 'lyapunov_cost', []))}")
                    if hasattr(crit, "lyap_per_step"):
                        lines.append(f"lyap_per_step: {_stats(getattr(crit, 'lyap_per_step', []))}")
                    if hasattr(crit, "lyap_per_sec"):
                        lines.append(f"lyap_per_sec: {_stats(getattr(crit, 'lyap_per_sec', []))}")
                    if hasattr(crit, "jac_spectral_radius"):
                        lines.append(f"jac_spectral_radius: {_stats(getattr(crit, 'jac_spectral_radius', []))}")
                lines.append(f"total_cost: {_stats(crit.total_cost)}")
                self._set_metrics_summary("\n".join(lines))
        except Exception:
            pass

        # Update refresh label
        try:
            n_total = int(df.shape[0]) if df is not None else 0
            n_comp = int((df.get("is_complete", False)).sum()) if df is not None else 0
            from datetime import datetime as _dt

            ts = _dt.now().strftime("%H:%M:%S")
            self.lbl_refresh.setText(f"Last refresh {ts} • Trials: {n_total} • Complete: {n_comp} • Source: {study_dir}")
        except Exception:
            pass
        self._last_dir = study_dir

    def _render_summary_text(self, data: dict) -> None:
        try:
            ss = data.get("study_summary")
            cfg = data.get("config") or {}
            lines = []
            if ss is not None:
                lines.append(f"Study: {ss.study_name}")
                lines.append(f"Direction: {ss.direction}")
                lines.append(f"Trials: {ss.n_trials} • Complete: {ss.n_complete} • Pruned: {ss.n_pruned} • Failed: {ss.n_failed}")
                lines.append(f"Best: {ss.best_value:.6g} (Trial {ss.best_trial}) • Elapsed: {ss.elapsed_time:.1f}s")
            # Run config
            run = cfg.get("run") or {}
            if run:
                parts = []
                for k in ("n_trials", "n_jobs", "timeout", "seed", "study_name"):
                    if run.get(k) is not None:
                        parts.append(f"{k}={run.get(k)}")
                sampler = (cfg.get("optuna") or {}).get("sampler")
                if sampler:
                    parts.append(f"sampler={sampler}")
                if parts:
                    lines.append("Run: " + ", ".join(parts))
            # Objective weights
            obj = cfg.get("objective") or {}
            w = obj.get("weights") or {}
            if w:
                wtxt = ", ".join(f"{k}={float(v):.3g}" for k, v in w.items() if v is not None)
                lines.append("Weights: " + wtxt)
            # Paths
            paths = cfg.get("paths") or {}
            if paths:
                base = paths.get("base_model_spec")
                outd = paths.get("output_dir")
                if base or outd:
                    lines.append(f"Paths: base_model_spec={base} • output_dir={outd}")
            # Search-space summary (parameter counts)
            try:
                ss_sum = data.get("search_space_summary") or {}
                if ss_sum:
                    total = int(ss_sum.get("total_params") or 0)
                    by_type = ss_sum.get("by_type") or {}
                    cont = int(by_type.get("continuous") or 0)
                    cat = int(by_type.get("categorical") or 0)
                    mix_other = int(by_type.get("mixed") or 0) + int(by_type.get("other") or 0)
                    lines.append(f"Search space: params={total} • continuous={cont} • categorical={cat} • mixed/other={mix_other}")
            except Exception:
                pass
            # Best trial summary
            best = (data.get("best_trials_analysis") or {}).get("best_trials") or []
            if best:
                b0 = best[0]
                lines.append("")
                lines.append(f"Best Trial {b0.get('number')}: value={b0.get('value')}")
                params = b0.get("params") or {}
                if params:
                    # show top few params
                    showp = list(params.items())[:12]
                    lines.append("Params: " + ", ".join(f"{k}={v}" for k, v in showp))
                metrics = b0.get("metrics") or {}
                keym = {k: metrics[k] for k in ("branching_sigma", "total_cost", "psd_temporal_cost", "psd_spatial_cost") if k in metrics}
                if keym:
                    lines.append("Metrics: " + ", ".join(f"{k}={_fmt_float(v)}" for k, v in keym.items()))
            self.txt_summary.setPlainText("\n".join(lines) if lines else "Study loaded.")
        except Exception:
            # Fallback minimal
            ss = data.get("study_summary")
            if ss is not None:
                self.txt_summary.setPlainText(
                    f"Study: {ss.study_name}\nTrials: {ss.n_trials} (complete {ss.n_complete})\nBest: {ss.best_value:.6g}",
                )
            else:
                self.txt_summary.setPlainText("Study loaded.")

    def _update_kpis(self, data: dict) -> None:
        try:
            ss = data.get("study_summary")
            if ss is None:
                for lbl in (self._kpi_best, self._kpi_trials, self._kpi_complete, self._kpi_pruned, self._kpi_failed, self._kpi_elapsed):
                    lbl.setText("—")
                return
            self._kpi_best.setText(f"{getattr(ss, 'best_value', float('nan')):.6g}")
            self._kpi_trials.setText(str(getattr(ss, "n_trials", 0)))
            self._kpi_complete.setText(str(getattr(ss, "n_complete", 0)))
            self._kpi_pruned.setText(str(getattr(ss, "n_pruned", 0)))
            self._kpi_failed.setText(str(getattr(ss, "n_failed", 0)))
            self._kpi_elapsed.setText(f"{getattr(ss, 'elapsed_time', 0.0):.1f}")
        except Exception:
            pass

    def _populate_selectors(self, df: pd.DataFrame) -> None:
        try:
            cols = list(df.columns)
            # X can be any column
            x_candidates = cols
            # Y should be numeric-dominant; include 'value' and metric_* first
            import pandas as pd

            numeric_candidates = []
            for c in cols:
                try:
                    s = pd.to_numeric(df[c], errors="coerce")
                    if float(s.notna().mean()) >= 0.5:
                        numeric_candidates.append(c)
                except Exception:
                    continue
            preferred = [c for c in ["value"] if c in numeric_candidates]
            metric_cols = [c for c in cols if c.startswith("metric_") and c in numeric_candidates]
            y_candidates = list(dict.fromkeys(preferred + metric_cols + numeric_candidates))
            # Populate Y
            cur_y = self.cmb_y.currentText()
            self.cmb_y.blockSignals(True)
            self.cmb_y.clear()
            self.cmb_y.addItems(y_candidates or ["value"])
            if cur_y and cur_y in y_candidates:
                self.cmb_y.setCurrentText(cur_y)
            elif "value" in y_candidates:
                self.cmb_y.setCurrentText("value")
            self.cmb_y.blockSignals(False)
            # Populate X
            cur_x = self.cmb_x.currentText()
            self.cmb_x.blockSignals(True)
            self.cmb_x.clear()
            self.cmb_x.addItems(x_candidates)
            if cur_x and cur_x in x_candidates:
                self.cmb_x.setCurrentText(cur_x)
            elif "trial_number" in x_candidates:
                self.cmb_x.setCurrentText("trial_number")
            elif any(c.startswith("param_") for c in x_candidates):
                first_param = next(c for c in x_candidates if c.startswith("param_"))
                self.cmb_x.setCurrentText(first_param)
            self.cmb_x.blockSignals(False)
        except Exception:
            pass

    def _plot_all(self, data: dict) -> None:
        try:
            self._last_importance = data.get("parameter_importance") or {}
            self._replot_importance(self._last_importance)
        except Exception:
            pass
        self._render_plots_from_df()

    def _replot_importance(self, imp: dict | None = None) -> None:
        try:
            if imp is None:
                imp = {}
            topn = int(self.sb_topn.value())
            items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:topn]
            labels = [k.replace("param_", "") for k, _ in items]
            vals = [float(v) for _, v in items]
            if self.web_importance is not None and _go is not None and _pio is not None:
                try:
                    bar = _go.Bar(x=list(range(len(vals))), y=vals, marker={"color": "#76a5af"})
                    layout = {
                        "title": "Parameter Importance (|corr|)",
                        "xaxis": {"title": "param", "tickmode": "array", "tickvals": list(range(len(labels))), "ticktext": labels},
                        "yaxis": {"title": "|corr|"},
                        "template": "plotly_white",
                        "margin": {"l": 40, "r": 10, "t": 40, "b": 40},
                    }
                    fig = {"data": [bar], "layout": layout}
                    html = _pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
                    self.web_importance.setHtml(html)
                    with contextlib.suppress(Exception):
                        self._last_figs["Importance"] = fig
                    return
                except Exception:
                    pass
            if self.pg_importance is None:
                return
            self.pg_importance.clear()
            if not imp:
                return
            from pyqtgraph import BarGraphItem

            bg = BarGraphItem(x=list(range(len(vals))), height=vals, width=0.8, brush=pg.mkBrush("#76a5af"))
            self.pg_importance.addItem(bg)
            ax = self.pg_importance.getAxis("bottom")
            ax.setTicks([list(enumerate(labels))])
            self.pg_importance.setLabel("left", "Importance (|corr|)")
        except Exception:
            pass

    def _replot_perf(self) -> None:
        # Performance tab removed
        return

    def _replot_scatter(self) -> None:
        try:
            if self._df is None:
                return
            df = self._df
            x_col = (self.cmb_param.currentText() or "").strip()
            y_col = (self.cmb_metric.currentText() or "value").strip()
            if (not x_col) or (x_col not in df.columns) or (y_col not in df.columns):
                return
            import numpy as np
            import pandas as _pd

            s_x = df[x_col]
            s_y = _pd.to_numeric(df[y_col], errors="coerce")
            x_num = _pd.to_numeric(s_x, errors="coerce")
            x_is_categorical = (getattr(s_x, "dtype", None) is object) or (float(x_num.notna().mean()) < 0.6)
            if self.web_scatter is not None and _go is not None and _pio is not None:
                try:
                    traces = []
                    layout = {"title": f"{y_col} vs {x_col}", "template": "plotly_white", "margin": {"l": 40, "r": 10, "t": 40, "b": 40}}
                    if x_is_categorical:
                        cats = [str(c) for c in s_x.astype(str).fillna("NA").tolist()]
                        uniq = []
                        for c in cats:
                            if c not in uniq:
                                uniq.append(c)
                        mapping = {c: i for i, c in enumerate(uniq)}
                        x_codes = np.array([mapping.get(str(v), np.nan) for v in cats], dtype=float)
                        m = np.isfinite(x_codes) & s_y.notna().to_numpy()
                        xj = x_codes[m]
                        if xj.size:
                            jitter = (np.random.rand(xj.size) - 0.5) * 0.4
                            xj = xj + jitter
                        yy = s_y.to_numpy()[m]
                        traces.append(_go.Scatter(x=xj, y=yy, mode="markers", name="trials", marker={"size": 7, "color": "#0984e3"}))
                        layout.update(xaxis={"title": x_col.replace("param_", ""), "tickmode": "array", "tickvals": list(range(len(uniq))), "ticktext": uniq}, yaxis={"title": y_col})
                    else:
                        m = x_num.notna() & s_y.notna()
                        xx = x_num[m].to_numpy()
                        yy = s_y[m].to_numpy()
                        traces.append(_go.Scatter(x=xx, y=yy, mode="markers", name="trials", marker={"size": 7, "color": "#0984e3"}))
                        layout.update(xaxis={"title": x_col.replace("param_", "")}, yaxis={"title": y_col})
                        if self.chk_regression.isChecked() and xx.size >= 2:
                            method = (self.cmb_regression.currentText() or "Linear").lower()
                            if method.startswith("linear"):
                                try:
                                    coeffs = np.polyfit(xx, yy, 1)
                                    x_grid = np.linspace(np.nanmin(xx), np.nanmax(xx), 100)
                                    y_fit = coeffs[0] * x_grid + coeffs[1]
                                    ss_res = np.sum((yy - (coeffs[0] * xx + coeffs[1])) ** 2)
                                    ss_tot = np.sum((yy - np.mean(yy)) ** 2)
                                    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else np.nan)
                                    traces.append(_go.Scatter(x=x_grid, y=y_fit, mode="lines", name=f"fit (R²={r2:.3f})", line={"color": "#d35400", "width": 2}))
                                except Exception:
                                    pass
                            elif method.startswith("lowess"):
                                try:
                                    from statsmodels.nonparametric.smoothers_lowess import (
                                        lowess,
                                    )

                                    fitted = lowess(yy, xx, frac=0.66, return_sorted=True)
                                    traces.append(_go.Scatter(x=fitted[:, 0], y=fitted[:, 1], mode="lines", name="lowess", line={"color": "#d35400", "width": 2}))
                                except Exception:
                                    pass
                    fig = {"data": traces, "layout": layout}
                    html = _pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
                    self.web_scatter.setHtml(html)
                    with contextlib.suppress(Exception):
                        self._last_figs["Scatter"] = fig
                    return
                except Exception:
                    pass
            if (self.pg_scatter is None) or (pg is None):
                return
            plot = self.pg_scatter
            plot.clear()
            from pyqtgraph import ScatterPlotItem

            if x_is_categorical:
                cats = [str(c) for c in s_x.astype(str).fillna("NA").tolist()]
                if x_col not in getattr(self, "_cat_maps", {}):
                    self._cat_maps = getattr(self, "_cat_maps", {})
                    seen = []
                    for c in cats:
                        if c not in seen:
                            seen.append(c)
                    self._cat_maps[x_col] = seen
                mapping = {c: i for i, c in enumerate(self._cat_maps[x_col])}
                x_codes = np.array([mapping.get(str(v), np.nan) for v in cats], dtype=float)
                m = np.isfinite(x_codes) & s_y.notna().to_numpy()
                yy_full = s_y.to_numpy()
                row_pos = np.nonzero(m)[0]
                x_codes = x_codes[m]
                yy = yy_full[m]
                if x_codes.size:
                    jitter = (np.random.rand(x_codes.size) - 0.5) * 0.4
                    xj = x_codes + jitter
                    spots = [{"pos": (float(x), float(yv)), "data": {"row_pos": int(r)}} for x, yv, r in zip(xj, yy, row_pos, strict=False)]
                    self._scatter_param = ScatterPlotItem(size=7, pen=pg.mkPen("#0984e3"), brush=pg.mkBrush("#74b9ff"))
                    self._scatter_param.addPoints(spots)
                    self._scatter_param.sigClicked.connect(lambda plt, pts: self._on_points_clicked(pts))
                    plot.addItem(self._scatter_param, name="Trials")
                    try:
                        means = _pd.DataFrame({"x": x_codes, "y": yy}).groupby("x")["y"].mean()
                        plot.plot(means.index.to_numpy(), means.values, pen=None, symbol="t", symbolSize=12, symbolBrush=pg.mkBrush("#e17055"), symbolPen=pg.mkPen("#d35400"), name="Category mean")
                    except Exception:
                        pass
                    ax = plot.getAxis("bottom")
                    labels = self._cat_maps[x_col]
                    ax.setTicks([list(enumerate(labels))])
                plot.setLabel("bottom", f"{x_col.replace('param_', '')} (categorical)")
                plot.setLabel("left", y_col)
                plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            else:
                m = x_num.notna() & s_y.notna()
                xx = x_num[m].to_numpy()
                yy = s_y[m].to_numpy()
                row_pos = np.nonzero(m.to_numpy())[0]
                spots = [{"pos": (float(x), float(yv)), "data": {"row_pos": int(r)}} for x, yv, r in zip(xx, yy, row_pos, strict=False)]
                self._scatter_param = ScatterPlotItem(size=7, pen=pg.mkPen("#0984e3"), brush=pg.mkBrush("#74b9ff"))
                self._scatter_param.addPoints(spots)
                self._scatter_param.sigClicked.connect(lambda plt, pts: self._on_points_clicked(pts))
                plot.addItem(self._scatter_param, name="Trials")
                try:
                    if self.chk_regression.isChecked() and xx.size >= 2:
                        coeffs = np.polyfit(xx, yy, 1)
                        x_grid = np.linspace(np.nanmin(xx), np.nanmax(xx), 100)
                        y_fit = coeffs[0] * x_grid + coeffs[1]
                        plot.plot(x_grid, y_fit, pen=pg.mkPen("#d35400", width=2), name="fit")
                except Exception:
                    pass
                plot.setLabel("bottom", x_col.replace("param_", ""))
                plot.setLabel("left", y_col)
                plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        except Exception:
            pass

    def _on_points_clicked(self, points: list) -> None:
        try:
            if self._df is None:
                return
            pt = points[0] if points else None
            if pt is None:
                return
            d = pt.data() or {}
            row_pos = int(d.get("row_pos", -1))
            if row_pos < 0 or row_pos >= int(self._df.shape[0]):
                return
            # Select row in table
            try:
                if self._table_model is not None and self._table_proxy is not None:
                    src_idx = self._table_model.index(row_pos, 0)
                    view_idx = self._table_proxy.mapFromSource(src_idx)
                    if view_idx.isValid():
                        self.table_view.scrollTo(view_idx)
                        self.table_view.selectRow(view_idx.row())
            except Exception:
                pass
            # Show info dialog
            text = self._trial_info_text(row_pos)
            QMessageBox.information(self, "Trial Details", text)
        except Exception:
            pass

    def _trial_info_text(self, row_pos: int) -> str:
        try:
            if self._df is None:
                return ""
            row = self._df.iloc[row_pos]
            lines = []
            tn = row.get("trial_number", None)
            val = row.get("value", None)
            state = row.get("state", None)
            dur = row.get("duration", None)
            if tn is not None:
                lines.append(f"trial_number: {tn}")
            if val is not None:
                lines.append(f"value: {_fmt_float(val)}")
            if state is not None:
                lines.append(f"state: {state}")
            if dur is not None:
                lines.append(f"duration: {_fmt_float(dur)} s")
            # Metrics
            metric_items = [(k, row[k]) for k in self._df.columns if str(k).startswith("metric_") and k in row.index]
            if metric_items:
                show_m = metric_items[:8]
                lines.append("metrics: " + ", ".join(f"{k}={_fmt_float(v)}" for k, v in show_m))
            # Params
            param_items = [(k, row[k]) for k in self._df.columns if str(k).startswith("param_") and k in row.index]
            if param_items:
                show_p = param_items[:12]
                lines.append("params: " + ", ".join(f"{k}={v}" for k, v in show_p))
            return "\n".join(lines)
        except Exception:
            return ""

    def _replot_all(self) -> None:
        self._replot_scatter()

    def _apply_table_filter(self, text: str) -> None:
        try:
            if self._table_proxy is not None:
                # Accept a simple case-insensitive regex
                from PyQt6.QtCore import QRegularExpression

                rx = QRegularExpression(text, QRegularExpression.PatternOption.CaseInsensitiveOption)
                self._table_proxy.setFilterRegularExpression(rx)
        except Exception:
            pass

    def _export_csv(self) -> None:
        try:
            if self._df is None or self._df.empty:
                QMessageBox.information(self, "Export", "Nothing to export.")
                return
            p, _ = QFileDialog.getSaveFileName(self, "Export table as…", str(Path.cwd() / "hpo_results.csv"), "CSV (*.csv)")
            if not p:
                return
            self._df.to_csv(p, index=False)
            QMessageBox.information(self, "Export", f"Saved: {p}")
        except Exception as e:
            QMessageBox.warning(self, "Export", f"Failed to export: {e}")

    def apply_processed(self, study_dir: str, data: dict) -> None:
        try:
            # Summary
            self._render_summary_text(data)
            self._update_kpis(data)
            # Table
            df = data.get("trials_df")
            if df is not None:
                base_cols = [c for c in ["trial_number", "state", "value", "duration", "rank"] if c in df.columns]
                param_cols = [c for c in df.columns if c.startswith("param_")][:20]
                metric_cols = [c for c in df.columns if c.startswith("metric_")][:10]
                show = df[base_cols + param_cols + metric_cols].copy()
                # Keep full df for plotting and selectors
                self._df = df
                model = _PandasModel(show)
                proxy = QSortFilterProxyModel(self)
                proxy.setSourceModel(model)
                proxy.setDynamicSortFilter(True)
                proxy.setFilterKeyColumn(-1)
                self._table_proxy = proxy
                self._table_model = model
                self.table_view.setModel(proxy)
                self.table_view.resizeColumnsToContents()
                self._populate_selectors(self._df)
                # Render plots now that _df exists
                self._render_plots_from_df()
            # History plot removed from Analysis
            # Plots
            self._plot_all(data)
            # Pairwise & parallel removed
            # Metrics summary
            try:
                crit = data.get("criticality_metrics")
                if crit is not None:
                    import numpy as np

                    def _stats(vals) -> str:
                        a = np.array([v for v in (vals or []) if np.isfinite(v)], dtype=float)
                        if a.size == 0:
                            return "(no data)"
                        return f"mean={a.mean():.4g}, std={a.std():.4g}, min={a.min():.4g}, max={a.max():.4g}"

                    # Show only metrics with non-zero weights; always include total_cost
                    cfg = data.get("config") or {}
                    w = cfg.get("weights") or cfg.get("objective", {}).get("weights") or cfg

                    def _w(name: str) -> float:
                        try:
                            return float((w or {}).get(name, 0.0))
                        except Exception:
                            return 0.0

                    lines = []
                    if _w("w_branch") > 0.0:
                        lines.append(f"branching_sigma: {_stats(crit.branching_sigma)}")
                    if _w("w_psd_t") > 0.0:
                        lines.append(f"psd_temporal_cost: {_stats(crit.psd_temporal_cost)}")
                    if _w("w_psd_spatial") > 0.0:
                        lines.append(f"psd_spatial_cost: {_stats(crit.psd_spatial_cost)}")
                    if _w("w_chi_inv") > 0.0:
                        lines.append(f"chi_inv_cost: {_stats(crit.chi_inv_cost)}")
                    if _w("w_autocorr") > 0.0:
                        lines.append(f"autocorr_cost: {_stats(crit.autocorr_cost)}")
                    if _w("w_jacobian") > 0.0:
                        if hasattr(crit, "jacobian_cost"):
                            lines.append(f"jacobian_cost: {_stats(getattr(crit, 'jacobian_cost', []))}")
                    if _w("w_lyapunov") > 0.0:
                        if hasattr(crit, "lyapunov_cost"):
                            lines.append(f"lyapunov_cost: {_stats(getattr(crit, 'lyapunov_cost', []))}")
                        if hasattr(crit, "lyap_per_step"):
                            lines.append(f"lyap_per_step: {_stats(getattr(crit, 'lyap_per_step', []))}")
                        if hasattr(crit, "lyap_per_sec"):
                            lines.append(f"lyap_per_sec: {_stats(getattr(crit, 'lyap_per_sec', []))}")
                        if hasattr(crit, "jac_spectral_radius"):
                            lines.append(f"jac_spectral_radius: {_stats(getattr(crit, 'jac_spectral_radius', []))}")
                    lines.append(f"total_cost: {_stats(crit.total_cost)}")
                    self._set_metrics_summary("\n".join(lines))
            except Exception:
                pass
        finally:
            self._last_dir = study_dir
            # Update refresh label
            try:
                df = data.get("trials_df")
                n_total = int(df.shape[0]) if df is not None else 0
                n_comp = int((df.get("is_complete", False)).sum()) if df is not None else 0
                from datetime import datetime as _dt

                ts = _dt.now().strftime("%H:%M:%S")
                self.lbl_refresh.setText(f"Last refresh {ts} • Trials: {n_total} • Complete: {n_comp} • Source: {study_dir}")
            except Exception:
                pass


class _ResultsWorker(QThread):
    finished_data = pyqtSignal(str, dict)

    def __init__(self, study_dir: str) -> None:
        super().__init__()
        self.study_dir = study_dir

    def run(self) -> None:
        try:
            db_path = os.path.join(self.study_dir, "optuna_studies.db")
            if not os.path.isfile(db_path):
                # Fallback: standardized JSONL schema helper
                try:
                    from soen_toolkit.utils.hpo.data.schema import (
                        build_trials_df_from_jsonl as _from_jsonl,
                    )

                    jsonl_path = os.path.join(self.study_dir, "trials.jsonl")
                    df = _from_jsonl(jsonl_path)
                    if df is not None and not df.empty:
                        data = {"trials_df": df, "config": {}}
                        self.finished_data.emit(self.study_dir, data)
                        return
                    # If JSONL exists but yields no rows, warn visibly via empty result with flag
                    if os.path.isfile(jsonl_path):
                        data = {"trials_df": df, "config": {}, "warning": f"No trials read from {jsonl_path}"}
                        self.finished_data.emit(self.study_dir, data)
                        return
                except Exception as e:
                    # Emit error info so the UI can display it
                    self.finished_data.emit(self.study_dir, {"trials_df": None, "error": f"JSONL read failed: {e}"})
                return
            study, cfg = load_study_from_directory(self.study_dir)
            proc = CriticalityDataProcessor(study, cfg or {})
            data = proc.process_all()
            self.finished_data.emit(self.study_dir, data)
        except Exception:
            # Swallow errors during active runs; next tick will retry
            pass


def _repo_root() -> str:
    # Assume this file lives under Studies/Criticality/tools
    return str(Path(__file__).resolve().parents[3])


def _ensure_tools_importable() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_tools_importable()

# Import the enumerator functions and HPO config helpers
try:
    # Prefer package import
    from soen_toolkit.utils.hpo.tools.enumerate_model_options import (
        build_hpo_skeleton,
        build_option_schema,
    )
except Exception:
    # Fallback: relative import when running directly
    from .enumerate_model_options import build_hpo_skeleton, build_option_schema

try:
    from soen_toolkit.utils.hpo.config.hpo_config import (
        extract_spec_from_model,
        load_hpo_config as _load_hpo_cfg_util,
        save_hpo_config as _save_hpo_cfg_util,
    )
    from soen_toolkit.utils.hpo.config.hpo_project import HPOProject
except Exception:
    # When running directly without package import
    from ..config.hpo_config import (
        extract_spec_from_model,
        load_hpo_config as _load_hpo_cfg_util,
        save_hpo_config as _save_hpo_cfg_util,
    )
    from ..hpo_project import HPOProject


# ————————————————————————————————————————————————————————————
# Small UI helpers for ergonomic, no-text editing of search spaces
# ————————————————————————————————————————————————————————————
class MultiCheckList(QWidget):
    """Simple checklist using QListWidget with checkboxes."""

    def __init__(self, items: list[str] | None = None) -> None:
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setSelectionMode(self.list.SelectionMode.NoSelection)
        v.addWidget(self.list)
        if items:
            self.set_items(items)

    def set_items(self, items: list[str]) -> None:
        self.list.clear()
        for name in items:
            it = QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            self.list.addItem(it)

    def selected(self) -> list[str]:
        out: list[str] = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.text())
        return out


class NewProjectDialog(QDialog):
    """Simple dialog to create a new HPO project from a model.

    Just collects the model file path - output directory is auto-detected from model location.
    The HPO YAML will be saved later via the main Save button.
    """

    def __init__(self, parent: QWidget | None = None, *, start_dir: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New HPO Project From Model")
        self.setModal(True)
        self.resize(500, 200)

        v = QVBoxLayout(self)

        # Model path
        form = QFormLayout()
        self.le_model = QLineEdit()
        btn_model = QPushButton("Browse…")
        btn_model.clicked.connect(self._browse_model)
        row_model = QHBoxLayout()
        row_model.addWidget(self.le_model, 1)
        row_model.addWidget(btn_model)
        w_row_model = QWidget()
        w_row_model.setLayout(row_model)
        form.addRow("Model file:", w_row_model)
        v.addLayout(form)

        # Hint text
        hint = QLabel("Choose a model file (.yaml/.yml/.json/.soen/.pth). We'll extract a spec if needed and set up the HPO project. Use the Save button later to write the HPO config YAML.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-style: italic; margin: 10px 0;")
        v.addWidget(hint)

        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)

        # Initialize
        self._refresh_ok_state()

    def _refresh_ok_state(self) -> None:
        model_ok = bool(self.le_model.text().strip()) and os.path.exists(self.le_model.text().strip())
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(model_ok)

    def _browse_model(self) -> None:
        start_dir = str(Path.cwd())
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Choose model (spec or full model)",
            start_dir,
            "Model spec/Full model (*.yaml *.yml *.json *.soen *.pth);;All files (*)",
        )
        if p:
            self.le_model.setText(p)
        self._refresh_ok_state()

    def _on_accept(self) -> None:
        if not os.path.exists(self.le_model.text().strip()):
            QMessageBox.warning(self, "Missing model", "Please choose a valid model file.")
            return
        self.accept()

    def model_path(self) -> str:
        return self.le_model.text().strip()


class NumberBoundsEditor(QWidget):
    """Min/Max editor for float or int values."""

    def __init__(self, *, is_int: bool = False, default_min: float = 0.0, default_max: float = 1.0) -> None:
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        if is_int:
            self.min_sb = QSpinBox()
            self.min_sb.setRange(-(10**9), 10**9)
            self.max_sb = QSpinBox()
            self.max_sb.setRange(-(10**9), 10**9)
            self.min_sb.setValue(int(default_min))
            self.max_sb.setValue(int(default_max))
        else:
            self.min_sb = QDoubleSpinBox()
            self.min_sb.setRange(-1e12, 1e12)
            self.min_sb.setDecimals(9)
            self.max_sb = QDoubleSpinBox()
            self.max_sb.setRange(-1e12, 1e12)
            self.max_sb.setDecimals(9)
            self.min_sb.setValue(float(default_min))
            self.max_sb.setValue(float(default_max))
        h.addWidget(QLabel("min:"))
        h.addWidget(self.min_sb)
        h.addWidget(QLabel("max:"))
        h.addWidget(self.max_sb)

        self._is_int = is_int

    def get(self) -> dict:
        if isinstance(self.min_sb, QSpinBox):
            return {"min": int(self.min_sb.value()), "max": int(self.max_sb.value())}
        return {"min": float(self.min_sb.value()), "max": float(self.max_sb.value())}

    def set(self, bounds: dict) -> None:
        if bounds is None:
            return
        mn = bounds.get("min")
        mx = bounds.get("max")
        if mn is not None:
            self.min_sb.setValue(int(mn) if self._is_int else float(mn))
        if mx is not None:
            self.max_sb.setValue(int(mx) if self._is_int else float(mx))


class LayerParamEditor(QWidget):
    """Editor for a single layer parameter with distribution choices."""

    DIST_LIST = ["constant", "uniform", "normal", "loguniform", "lognormal"]

    def __init__(self, name: str, meta: dict) -> None:
        super().__init__()
        self.name = name
        self.meta = meta

        # Robust default bounds
        def _num(x, default=None):
            if isinstance(x, (int, float)):
                return float(x)
            try:
                return float(x)
            except Exception:
                return default

        dv = _num(meta.get("default_value"), None)
        mn_def = _num(meta.get("min_value_default"), None)
        mx_def = _num(meta.get("max_value_default"), None)
        if mn_def is None and mx_def is None:
            # no hints; set around default or [0,1]
            if dv is not None:
                lo = 0.0 if dv >= 0 else dv * 2.0
                hi = max(1.0, abs(dv) * 2.0)
            else:
                lo, hi = 0.0, 1.0
        else:
            lo = 0.0 if mn_def is None else mn_def
            hi = 1.0 if mx_def is None else mx_def
            if hi <= lo:
                hi = lo + max(1e-6, abs(lo) * 0.1 + 1.0)
        defaults = (lo, hi)
        log_param = bool(meta.get("is_log_param", False))

        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(8)
        head = QHBoxLayout()
        self.enabled_cb = QCheckBox(f"{name}")
        self.enabled_cb.setChecked(True)
        head.addWidget(self.enabled_cb)
        head.addStretch(1)
        self.log_scale_cb = QCheckBox("log-scale")
        self.log_scale_cb.setChecked(log_param)
        head.addWidget(self.log_scale_cb)
        v.addLayout(head)

        # Distribution checklist
        self.dist_checks: dict[str, QCheckBox] = {}
        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel("Distributions:"))
        for d in self.DIST_LIST:
            cb = QCheckBox(d)
            cb.setChecked(d in ("constant",))  # default to constant
            self.dist_checks[d] = cb
            dist_row.addWidget(cb)
        dist_row.addStretch(1)
        v.addLayout(dist_row)

        # Per-distribution bounds (hidden unless selected)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        r = 0
        # constant
        self.lbl_const = QLabel("constant: value_bounds")
        grid.addWidget(self.lbl_const, r, 0)
        self.const_bounds = NumberBoundsEditor(default_min=defaults[0], default_max=defaults[1])
        grid.addWidget(self.const_bounds, r, 1)
        r += 1
        # uniform
        self.lbl_unif_min = QLabel("uniform: min_bounds")
        grid.addWidget(self.lbl_unif_min, r, 0)
        self.unif_min = NumberBoundsEditor(default_min=defaults[0], default_max=defaults[1])
        grid.addWidget(self.unif_min, r, 1)
        r += 1
        self.lbl_unif_max = QLabel("uniform: max_bounds")
        grid.addWidget(self.lbl_unif_max, r, 0)
        self.unif_max = NumberBoundsEditor(default_min=defaults[0], default_max=defaults[1])
        grid.addWidget(self.unif_max, r, 1)
        r += 1
        # normal
        self.lbl_norm_mean = QLabel("normal: mean_bounds")
        grid.addWidget(self.lbl_norm_mean, r, 0)
        self.norm_mean = NumberBoundsEditor(default_min=defaults[0], default_max=defaults[1])
        grid.addWidget(self.norm_mean, r, 1)
        r += 1
        self.lbl_norm_std = QLabel("normal: std_bounds")
        grid.addWidget(self.lbl_norm_std, r, 0)
        self.norm_std = NumberBoundsEditor(default_min=0.0, default_max=max(1e-3, defaults[1] - defaults[0]))
        grid.addWidget(self.norm_std, r, 1)
        r += 1
        # loguniform
        self.lbl_logu = QLabel("loguniform: value_bounds")
        grid.addWidget(self.lbl_logu, r, 0)
        log_lo = defaults[0]
        log_hi = defaults[1]
        if log_lo <= 0:
            log_lo = 1e-9
        if log_hi <= log_lo:
            log_hi = log_lo * 10.0
        self.logu_bounds = NumberBoundsEditor(default_min=log_lo, default_max=log_hi)
        grid.addWidget(self.logu_bounds, r, 1)
        r += 1
        # lognormal
        self.lbl_logn_mean = QLabel("lognormal: log_mean_bounds")
        grid.addWidget(self.lbl_logn_mean, r, 0)
        self.ln_mean = NumberBoundsEditor(default_min=-4.0, default_max=2.0)
        grid.addWidget(self.ln_mean, r, 1)
        r += 1
        self.lbl_logn_std = QLabel("lognormal: log_std_bounds")
        grid.addWidget(self.lbl_logn_std, r, 0)
        self.ln_std = NumberBoundsEditor(default_min=0.01, default_max=1.0)
        grid.addWidget(self.ln_std, r, 1)
        r += 1
        v.addLayout(grid)

        # Wire up dynamic visibility
        for cb in self.dist_checks.values():
            cb.toggled.connect(self._refresh_dist_visibility)
        self._refresh_dist_visibility()

    def _refresh_dist_visibility(self) -> None:
        have = {name for name, cb in self.dist_checks.items() if cb.isChecked()}
        # constant
        show = "constant" in have
        self.lbl_const.setVisible(show)
        self.const_bounds.setVisible(show)
        # uniform
        show = "uniform" in have
        self.lbl_unif_min.setVisible(show)
        self.unif_min.setVisible(show)
        self.lbl_unif_max.setVisible(show)
        self.unif_max.setVisible(show)
        # normal
        show = "normal" in have
        self.lbl_norm_mean.setVisible(show)
        self.norm_mean.setVisible(show)
        self.lbl_norm_std.setVisible(show)
        self.norm_std.setVisible(show)
        # loguniform
        show = "loguniform" in have
        self.lbl_logu.setVisible(show)
        self.logu_bounds.setVisible(show)
        # lognormal
        show = "lognormal" in have
        self.lbl_logn_mean.setVisible(show)
        self.ln_mean.setVisible(show)
        self.lbl_logn_std.setVisible(show)
        self.ln_std.setVisible(show)

    def to_dict(self) -> dict:
        if not self.enabled_cb.isChecked():
            return {"enabled": False}
        dists = [name for name, cb in self.dist_checks.items() if cb.isChecked()]
        per: dict[str, dict] = {}
        if "constant" in dists:
            per["constant"] = {"value_bounds": self.const_bounds.get()}
        if "uniform" in dists:
            per["uniform"] = {"min_bounds": self.unif_min.get(), "max_bounds": self.unif_max.get()}
        if "normal" in dists:
            per["normal"] = {"mean_bounds": self.norm_mean.get(), "std_bounds": self.norm_std.get()}
        if "loguniform" in dists:
            per["loguniform"] = {"value_bounds": self.logu_bounds.get()}
        if "lognormal" in dists:
            per["lognormal"] = {"log_mean_bounds": self.ln_mean.get(), "log_std_bounds": self.ln_std.get()}
        out = {
            "enabled": True,
            "distributions": dists or ["constant"],
            "per_distribution": per,
        }
        if self.log_scale_cb.isChecked():
            out["log_scale"] = True
        return out

    def load_from(self, cfg: dict) -> None:
        if not isinstance(cfg, dict):
            return
        self.enabled_cb.setChecked(bool(cfg.get("enabled", True)))
        # dists
        want = set(cfg.get("distributions", []))
        for name, cb in self.dist_checks.items():
            cb.setChecked(name in want) if want else None
        self._refresh_dist_visibility()
        pd = cfg.get("per_distribution") or {}
        if "constant" in pd:
            self.const_bounds.set(pd["constant"].get("value_bounds", {}))
        if "uniform" in pd:
            self.unif_min.set(pd["uniform"].get("min_bounds", {}))
            self.unif_max.set(pd["uniform"].get("max_bounds", {}))
        if "normal" in pd:
            self.norm_mean.set(pd["normal"].get("mean_bounds", {}))
            self.norm_std.set(pd["normal"].get("std_bounds", {}))
        if "loguniform" in pd:
            self.logu_bounds.set(pd["loguniform"].get("value_bounds", {}))
        if "lognormal" in pd:
            self.ln_mean.set(pd["lognormal"].get("log_mean_bounds", {}))
            self.ln_std.set(pd["lognormal"].get("log_std_bounds", {}))


class ConnParamEditor(QWidget):
    """Editor for a connection parameter with optional applies_to types."""

    def __init__(self, name: str, meta_by_type: dict[str, dict], available_types: list[str]) -> None:
        super().__init__()
        self.name = name
        self.available_types = available_types
        # Determine overall type and default bounds/options from first meta
        any_meta = next(iter(meta_by_type.values())) if meta_by_type else {}
        kind = any_meta.get("type", "float") or "float"
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        # header
        self.enabled_cb = QCheckBox(name)
        self.enabled_cb.setChecked(False)
        v.addWidget(self.enabled_cb)
        # content container (hidden until enabled)
        self._content = QWidget()
        grid = QGridLayout(self._content)
        grid.setHorizontalSpacing(12)
        row = 0
        self.kind = kind
        if kind == "enum":
            self.lbl_choices = QLabel("choices:")
            grid.addWidget(self.lbl_choices, row, 0)
            opts = []
            # Merge unique options across types
            for m in meta_by_type.values():
                opts.extend(list(m.get("options", [])))
            opts = sorted(set(map(str, opts)))
            self.choices_list = MultiCheckList(opts)
            grid.addWidget(self.choices_list, row, 1)
            row += 1
        else:
            self.lbl_bounds = QLabel("bounds:")
            grid.addWidget(self.lbl_bounds, row, 0)
            is_int = kind == "int"

            def _num(x, default):
                if isinstance(x, (int, float)):
                    return float(x)
                try:
                    return float(x)
                except Exception:
                    return default

            mn = _num(any_meta.get("min"), 0.0)
            mx = _num(any_meta.get("max"), 1.0)
            if mx <= mn:
                mx = mn + (1.0 if not is_int else 1)
            self.bounds = NumberBoundsEditor(is_int=is_int, default_min=mn, default_max=mx)
            grid.addWidget(self.bounds, row, 1)
            row += 1
        # applies_to types
        self.lbl_applies = QLabel("applies_to types:")
        grid.addWidget(self.lbl_applies, row, 0)
        types_with_param = [t for t, m in meta_by_type.items() if m]
        self.applies_list = MultiCheckList(types_with_param or available_types)
        grid.addWidget(self.applies_list, row, 1)
        row += 1
        v.addWidget(self._content)
        self._content.setVisible(False)
        self.enabled_cb.toggled.connect(self._content.setVisible)

    def to_dict(self) -> dict:
        if not self.enabled_cb.isChecked():
            return {"enabled": False}
        out: dict = {"enabled": True}
        if self.kind == "enum":
            out["choices"] = self.choices_list.selected()
        else:
            out["type"] = self.kind
            out["bounds"] = self.bounds.get()
        sel_types = self.applies_list.selected()
        if sel_types:
            out["applies_to"] = sel_types
        return out

    def load_from(self, cfg: dict) -> None:
        if not isinstance(cfg, dict):
            return
        enabled = bool(cfg.get("enabled", False))
        self.enabled_cb.setChecked(enabled)
        self._content.setVisible(enabled)
        if self.kind == "enum":
            # Re-check items matching provided choices
            want = set(map(str, cfg.get("choices", [])))
            for i in range(self.choices_list.list.count()):
                it = self.choices_list.list.item(i)
                it.setCheckState(Qt.CheckState.Checked if it.text() in want else Qt.CheckState.Unchecked)
        else:
            b = cfg.get("bounds") or {}
            self.bounds.set(b)
        if "applies_to" in cfg:
            want = set(map(str, cfg.get("applies_to") or []))
            for i in range(self.applies_list.list.count()):
                it = self.applies_list.list.item(i)
                it.setCheckState(Qt.CheckState.Checked if it.text() in want else Qt.CheckState.Unchecked)


class ModelVisualizationWidget(QWidget):
    """Widget for displaying a visual representation of the loaded model."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()
        self._schema: dict[str, Any] | None = None

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Title
        title = QLabel("Model Structure")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Model info section
        self.info_group = QGroupBox("Model Information")
        info_layout = QFormLayout(self.info_group)

        self.model_path_label = QLabel("No model loaded")
        self.model_path_label.setWordWrap(True)
        info_layout.addRow("Model Path:", self.model_path_label)

        self.layer_count_label = QLabel("—")
        info_layout.addRow("Total Layers:", self.layer_count_label)

        self.connection_count_label = QLabel("—")
        info_layout.addRow("Total Connections:", self.connection_count_label)

        layout.addWidget(self.info_group)

        # Model visualization area
        viz_group = QGroupBox("Architecture Visualization")
        viz_layout = QVBoxLayout(viz_group)

        # Create a text area for the model diagram summary
        self.model_diagram = QTextEdit()
        self.model_diagram.setReadOnly(True)
        self.model_diagram.setMinimumHeight(200)
        self.model_diagram.setFont(self._get_monospace_font())
        viz_layout.addWidget(self.model_diagram)

        # Add actual model visualization plot using SVG
        self.model_plot_widget = QSvgWidget()
        self.model_plot_widget.setMinimumHeight(400)
        self.model_plot_widget.setStyleSheet("border: 1px solid #ccc; background-color: #2b2b2b;")
        viz_layout.addWidget(self.model_plot_widget)

        layout.addWidget(viz_group)

        # Layer details section
        self.layers_group = QGroupBox("Layer Details")
        self.layers_scroll = QScrollArea()
        self.layers_scroll.setWidgetResizable(True)
        self.layers_content = QWidget()
        self.layers_layout = QVBoxLayout(self.layers_content)
        self.layers_scroll.setWidget(self.layers_content)
        self.layers_group.setLayout(QVBoxLayout())
        self.layers_group.layout().addWidget(self.layers_scroll)
        layout.addWidget(self.layers_group)

        # Connection details section
        self.connections_group = QGroupBox("Connection Details")
        self.connections_scroll = QScrollArea()
        self.connections_scroll.setWidgetResizable(True)
        self.connections_content = QWidget()
        self.connections_layout = QVBoxLayout(self.connections_content)
        self.connections_scroll.setWidget(self.connections_content)
        self.connections_group.setLayout(QVBoxLayout())
        self.connections_group.layout().addWidget(self.connections_scroll)
        layout.addWidget(self.connections_group)

        layout.addStretch()

        # Initialize with placeholder
        self._create_placeholder_svg()

    def _get_monospace_font(self):
        """Get a monospace font for the diagram display."""
        from PyQt6.QtGui import QFont, QFontDatabase

        font = QFont("Consolas")  # Windows
        if not QFontDatabase.families().__contains__("Consolas"):
            font = QFont("Monaco")  # macOS
            if not QFontDatabase.families().__contains__("Monaco"):
                font = QFont("Courier New")  # Fallback
        font.setPointSize(10)
        return font

    def update_model(self, model_path: str, schema: dict[str, Any]) -> None:
        """Update the visualization with new model data."""
        self._schema = schema

        # Update model info
        self.model_path_label.setText(model_path)

        layers = schema.get("layers", {})
        connections = schema.get("connections", {})

        self.layer_count_label.setText(str(len(layers)))
        self.connection_count_label.setText(str(len(connections)))

        # Generate and display model diagram
        diagram = self._generate_model_diagram(schema)
        self.model_diagram.setHtml(diagram)

        # Generate actual model visualization plot
        self._generate_model_plot(model_path)

        # Update layer details
        self._update_layer_details(layers)

        # Update connection details
        self._update_connection_details(connections)

    def _generate_model_diagram(self, schema: dict[str, Any]) -> str:
        """Generate an HTML representation of the model architecture."""
        layers = schema.get("layers", {})
        connections = schema.get("connections", {})

        # Create a simple text-based diagram
        html = """
        <div style='font-family: monospace; background-color: #2b2b2b; color: #ffffff; padding: 15px; border-radius: 5px;'>
        <h3 style='color: #4CAF50; margin-top: 0;'>Model Architecture</h3>
        """

        if not layers:
            html += "<p style='color: #ff9800;'>No layers found in model</p>"
        else:
            # Sort layers by ID
            sorted_layers = sorted(layers.items(), key=lambda x: int(x[0]))

            html += "<div style='margin: 10px 0;'><strong>Layers:</strong></div>"
            for layer_id, layer_info in sorted_layers:
                layer_type = layer_info.get("layer_type", "Unknown")
                html += "<div style='margin: 5px 0; padding: 8px; background-color: #3c3c3c; border-left: 3px solid #4CAF50; border-radius: 3px;'>"
                html += f"<strong>Layer {layer_id}</strong> ({layer_type})</div>"

            if connections:
                html += "<div style='margin: 15px 0 10px 0;'><strong>Connections:</strong></div>"
                for conn_name, conn_info in connections.items():
                    conn_type = conn_info.get("current_type", "Unknown")
                    html += "<div style='margin: 5px 0; padding: 8px; background-color: #3c3c3c; border-left: 3px solid #2196F3; border-radius: 3px;'>"
                    html += f"<strong>{conn_name}</strong> ({conn_type})</div>"

        # Add simulation info if available
        sim_info = schema.get("simulation", {})
        if sim_info:
            html += "<div style='margin: 15px 0 10px 0;'><strong>Simulation Settings:</strong></div>"
            html += "<div style='margin: 5px 0; padding: 8px; background-color: #3c3c3c; border-left: 3px solid #FF9800; border-radius: 3px;'>"
            for key, value in sim_info.items():
                if isinstance(value, dict) and "default" in value:
                    html += f"<div><strong>{key}:</strong> {value['default']}</div>"
            html += "</div>"

        html += "</div>"
        return html

    def _generate_model_plot(self, model_path: str) -> None:
        """Generate the actual model visualization plot using model.visualize()."""
        try:
            # Load the model and generate visualization
            import os
            import tempfile

            from soen_toolkit.core import SOENModelCore
            from soen_toolkit.core.model_yaml import (
                build_model_from_yaml as _build_model_from_yaml,
            )

            # Load the model (support both full artifacts and YAML specs)
            ext = (os.path.splitext(model_path)[1] or "").lower()
            if ext in {".yaml", ".yml"}:
                model = _build_model_from_yaml(model_path)
            else:
                # Accept full model artifacts: .soen/.pth/.pt/.json
                model = SOENModelCore.load(
                    model_path,
                    device=None,
                    strict=False,
                    verbose=False,
                    show_logs=False,
                )

            # Create a temporary directory and base filename (no extension)
            temp_dir = tempfile.mkdtemp()
            temp_base = os.path.join(temp_dir, "model_viz")

            try:
                # Generate the SVG visualization with dark mode friendly settings
                viz_path = model.visualize(
                    save_path=temp_base,  # No extension - function adds it
                    file_format="svg",
                    dpi=150,
                    open_file=False,
                    bg_color="#2b2b2b",  # Dark background
                    edge_color="#ffffff",  # White edges
                    inter_color="#4CAF50",  # Green inter-layer connections
                    intra_color="#FF9800",  # Orange intra-layer connections
                    layer_color="#3c3c3c",  # Dark layer boxes
                    simple_view=True,
                    show_layer_ids=True,
                    show_conn_type=True,
                    orientation="LR",  # Left to right layout
                )

                # Load and display the SVG
                if os.path.exists(viz_path):
                    self.model_plot_widget.load(viz_path)
                else:
                    # If SVG doesn't exist, create a simple placeholder
                    self._create_placeholder_svg()

            finally:
                # Clean up temporary directory
                import shutil

                with contextlib.suppress(Exception):
                    shutil.rmtree(temp_dir)

        except Exception as e:
            # Create error placeholder SVG
            self._create_error_svg(str(e))

    def _create_placeholder_svg(self) -> None:
        """Create a placeholder SVG when no model is loaded."""
        svg_content = """<?xml version="1.0" encoding="UTF-8"?>
        <svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#2b2b2b"/>
            <text x="200" y="100" text-anchor="middle" fill="#888" font-family="Arial" font-size="16">
                No model visualization available
            </text>
        </svg>"""
        self.model_plot_widget.renderer().load(svg_content.encode())

    def _create_error_svg(self, error_msg: str) -> None:
        """Create an error SVG when visualization fails."""
        svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
        <svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#2b2b2b"/>
            <text x="200" y="90" text-anchor="middle" fill="#ff5555" font-family="Arial" font-size="14">
                Error generating visualization:
            </text>
            <text x="200" y="110" text-anchor="middle" fill="#ff5555" font-family="Arial" font-size="12">
                {error_msg[:50]}{"..." if len(error_msg) > 50 else ""}
            </text>
        </svg>"""
        self.model_plot_widget.renderer().load(svg_content.encode())

    def _update_layer_details(self, layers: dict[str, Any]) -> None:
        """Update the layer details section."""
        # Clear existing content
        for i in reversed(range(self.layers_layout.count())):
            child = self.layers_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()

        if not layers:
            no_layers = QLabel("No layers found")
            no_layers.setStyleSheet("color: #888; font-style: italic;")
            self.layers_layout.addWidget(no_layers)
            return

        # Add layer details
        for layer_id, layer_info in sorted(layers.items(), key=lambda x: int(x[0])):
            layer_widget = self._create_layer_detail_widget(layer_id, layer_info)
            self.layers_layout.addWidget(layer_widget)

        self.layers_layout.addStretch()

    def _create_layer_detail_widget(self, layer_id: str, layer_info: dict[str, Any]) -> QWidget:
        """Create a detailed widget for a single layer."""
        widget = QGroupBox(f"Layer {layer_id}")
        layout = QFormLayout(widget)

        # Layer type
        layer_type = layer_info.get("layer_type", "Unknown")
        type_label = QLabel(layer_type)
        type_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        layout.addRow("Type:", type_label)

        # Solver support
        if layer_info.get("supports_solver", False):
            solver_choices = layer_info.get("solver_choices", [])
            solver_label = QLabel(", ".join(solver_choices))
            layout.addRow("Supported Solvers:", solver_label)

        # Non-linearity support
        if layer_info.get("supports_source_func", False):
            source_funcs = layer_info.get("source_func_choices", [])
            if source_funcs:
                source_label = QLabel(f"{len(source_funcs)} available")
                layout.addRow("Non-Linearities:", source_label)

        # Parameter count
        params_schema = layer_info.get("params_schema", {})
        if params_schema:
            param_count = QLabel(str(len(params_schema)))
            layout.addRow("Parameters:", param_count)

        return widget

    def _update_connection_details(self, connections: dict[str, Any]) -> None:
        """Update the connection details section."""
        # Clear existing content
        for i in reversed(range(self.connections_layout.count())):
            child = self.connections_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()

        if not connections:
            no_connections = QLabel("No connections found")
            no_connections.setStyleSheet("color: #888; font-style: italic;")
            self.connections_layout.addWidget(no_connections)
            return

        # Add connection details
        for conn_name, conn_info in connections.items():
            conn_widget = self._create_connection_detail_widget(conn_name, conn_info)
            self.connections_layout.addWidget(conn_widget)

        self.connections_layout.addStretch()

    def _create_connection_detail_widget(self, conn_name: str, conn_info: dict[str, Any]) -> QWidget:
        """Create a detailed widget for a single connection."""
        widget = QGroupBox(conn_name)
        layout = QFormLayout(widget)

        # Current connection type
        current_type = conn_info.get("current_type", "Unknown")
        type_label = QLabel(current_type)
        type_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        layout.addRow("Current Type:", type_label)

        # Available types
        available_types = conn_info.get("available_types", [])
        if available_types:
            available_label = QLabel(f"{len(available_types)} types available")
            layout.addRow("Available Types:", available_label)

        return widget

    def clear_model(self) -> None:
        """Clear the current model visualization."""
        self._schema = None
        self.model_path_label.setText("No model loaded")
        self.layer_count_label.setText("—")
        self.connection_count_label.setText("—")
        self.model_diagram.clear()

        # Clear model plot
        self._create_placeholder_svg()

        # Clear layer details
        for i in reversed(range(self.layers_layout.count())):
            child = self.layers_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()

        # Clear connection details
        for i in reversed(range(self.connections_layout.count())):
            child = self.connections_layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()


class WeightEditor(QWidget):
    """Editor for weight init selection and per-method parameter bounds."""

    def __init__(self, weight_schema: dict) -> None:
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        # Init method choices
        v.addWidget(QLabel("Init method choices:"))
        self.methods = sorted(weight_schema.keys())
        self.method_list = MultiCheckList(self.methods)
        # Prefer normal and uniform by default
        for i in range(self.method_list.list.count()):
            it = self.method_list.list.item(i)
            if it.text() not in ("normal", "uniform"):
                it.setCheckState(Qt.CheckState.Unchecked)
        v.addWidget(self.method_list)

        # Method parameter editors
        self.method_param_boxes: dict[str, QGroupBox] = {}
        self.method_param_forms: dict[str, dict[str, NumberBoundsEditor | QComboBox | QCheckBox]] = {}
        for mname in self.methods:
            params = (weight_schema.get(mname) or {}).get("params", {})
            box = QGroupBox(f"{mname} parameters")
            form = QFormLayout(box)
            editors: dict[str, Any] = {}
            for pname, pmeta in params.items():
                # Only numeric parameters are supported ergonomically; strings get a small dropdown when known
                if str(pmeta.get("type", "")).lower() == "int":
                    ed = NumberBoundsEditor(is_int=True, default_min=-1000, default_max=1000)
                    editors[pname] = ed
                    form.addRow(f"{pname} (bounds):", ed)
                elif str(pmeta.get("type", "")).lower() in ("float", ""):
                    ed = NumberBoundsEditor(is_int=False, default_min=-1.0, default_max=1.0)
                    editors[pname] = ed
                    form.addRow(f"{pname} (bounds):", ed)
                elif pname == "nonlinearity":
                    cb = QComboBox()
                    cb.addItems(["relu", "leaky_relu"])  # common options
                    editors[pname] = cb
                    form.addRow("nonlinearity:", cb)
                else:
                    # Unsupported types -> skip from UI
                    continue
            self.method_param_forms[mname] = editors
            self.method_param_boxes[mname] = box
            box.setVisible(False)
            v.addWidget(box)

        # Update visibility when selection changes
        self.method_list.list.itemChanged.connect(self._update_method_boxes)
        self._update_method_boxes()

    def _update_method_boxes(self) -> None:
        selected = set()
        for i in range(self.method_list.list.count()):
            it = self.method_list.list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                selected.add(it.text())
        for mname, box in self.method_param_boxes.items():
            box.setVisible(mname in selected)

    def to_dict(self) -> dict:
        selected = set(self.method_list.selected())
        out: dict = {
            "init_method": {"enabled": True, "choices": list(selected or {"normal"})},
        }
        for mname, editors in self.method_param_forms.items():
            if selected and mname not in selected:
                continue
            if not editors:
                continue
            mblock: dict = {}
            for pname, ed in editors.items():
                if isinstance(ed, NumberBoundsEditor):
                    mblock[pname] = {"enabled": True, "bounds": ed.get()}
                elif isinstance(ed, QComboBox):
                    mblock[pname] = {"enabled": True, "choices": [ed.currentText()]}
            if mblock:
                out[mname] = mblock
        return out

    def load_from(self, cfg: dict) -> None:
        if not isinstance(cfg, dict):
            return
        init = cfg.get("init_method") or {}
        want = set(map(str, init.get("choices", [])))
        if want:
            for i in range(self.method_list.list.count()):
                it = self.method_list.list.item(i)
                it.setCheckState(Qt.CheckState.Checked if it.text() in want else Qt.CheckState.Unchecked)
        self._update_method_boxes()
        # Per-method params
        for mname, editors in self.method_param_forms.items():
            if mname not in cfg:
                continue
            block = cfg[mname]
            for pname, ed in editors.items():
                pdata = block.get(pname, {}) if isinstance(block, dict) else {}
                if isinstance(ed, NumberBoundsEditor):
                    ed.set(pdata.get("bounds", {}))
                elif isinstance(ed, QComboBox) and pdata.get("choices"):
                    opts = list(map(str, pdata.get("choices")))
                    if opts:
                        idx = ed.findText(opts[0])
                        if idx >= 0:
                            ed.setCurrentIndex(idx)


class HPOMinGui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SOEN HPO GUI")
        self._schema: dict[str, Any] | None = None
        self._project_mode: bool = False  # true when an HPO YAML project is loaded
        self._model_path: str | None = None  # model spec (or trial-params)
        self._base_model_path: str | None = None  # base model for structure
        self._output_dir: str | None = None
        self._hpo_yaml_path: str | None = None
        self._template_hpo_yaml_path: str | None = None
        self._running: bool = False
        self._proc: QProcess | None = None
        self._progress_target: int = 0
        self._progress_completed: int = 0
        self._last_dashboard_path: str | None = None
        self._last_study_name: str | None = None
        # Live results refresh
        self._results_timer = QTimer(self)
        self._results_timer.setInterval(3000)
        self._results_timer.timeout.connect(self._on_results_tick)
        self._results_worker: QThread | None = None

        # Dynamic space editors
        self.layer_param_editors: dict[str, LayerParamEditor] = {}
        self.conn_param_editors: dict[str, ConnParamEditor] = {}
        self.conn_type_list: MultiCheckList | None = None
        self.weight_editor: WeightEditor | None = None

        # Model visualization
        self.model_viz_widget: ModelVisualizationWidget | None = None

        self._init_ui()
        self._init_settings()
        self._apply_mode()
        # Cache for importance so Top-N re-renders without reprocessing
        self._last_importance: dict | None = None

    def _update_live_progress_plot(self, plot, df, study_summary=None) -> None:
        """Deprecated shim; use _refresh_live_plot instead."""
        with contextlib.suppress(Exception):
            self._refresh_live_plot(df, study_summary)

    def _refresh_live_plot(self, df, study_summary=None) -> None:
        try:
            c = df[(df.get("is_complete", False)) & (~df["value"].isna())]
            if c is None or c.empty:
                return
            import numpy as np

            xs = c["trial_number"].astype(float).to_numpy()
            ys = c["value"].astype(float).to_numpy()
            order = xs.argsort()
            xs = xs[order]
            ys = ys[order]
            # Plotly path
            if self.plot_live_web is not None and _go is not None and _pio is not None:
                try:
                    trials = _go.Scatter(x=xs, y=ys, mode="markers", name="trials", marker={"size": 6, "color": "#5865F2", "line": {"color": "#3f4ac8", "width": 0}})
                    mean = np.cumsum(ys) / (np.arange(len(ys)) + 1)
                    mean_tr = _go.Scatter(x=xs, y=mean, mode="lines", name="running mean", line={"color": "#00b894", "width": 2})
                    try:
                        if study_summary and isinstance(getattr(study_summary, "direction", None), str) and "MINIM" in study_summary.direction.upper():
                            best = np.minimum.accumulate(ys)
                        else:
                            best = np.maximum.accumulate(ys)
                    except Exception:
                        best = np.maximum.accumulate(ys)
                    best_tr = _go.Scatter(x=xs, y=best, mode="lines", name="running best", line={"color": "#e17055", "width": 2})
                    layout = {"title": "Optimization Progress", "xaxis": {"title": "trial"}, "yaxis": {"title": "value"}, "template": "plotly_white", "margin": {"l": 40, "r": 10, "t": 40, "b": 40}}
                    fig = {"data": [trials, mean_tr, best_tr], "layout": layout}
                    html = _pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
                    self.plot_live_web.setHtml(html)
                    return
                except Exception:
                    pass
            # PyQtGraph fallback
            if self.plot_live is not None and pg is not None:
                try:
                    self.plot_live.clear()
                    self.plot_live.addLegend(offset=(10, 10))
                    self.plot_live.plot(xs, ys, pen=None, symbol="o", symbolSize=6, symbolBrush=pg.mkBrush("#8899ff"), symbolPen=pg.mkPen("#5865F2"), name="trials")
                    mean = np.cumsum(ys) / (np.arange(len(ys)) + 1)
                    self.plot_live.plot(xs, mean, pen=pg.mkPen("#00b894", width=2), name="running mean")
                    try:
                        if study_summary and isinstance(getattr(study_summary, "direction", None), str) and "MINIM" in study_summary.direction.upper():
                            best = np.minimum.accumulate(ys)
                        else:
                            best = np.maximum.accumulate(ys)
                    except Exception:
                        best = np.maximum.accumulate(ys)
                    self.plot_live.plot(xs, best, pen=pg.mkPen("#e17055", width=2), name="running best")
                    self.plot_live.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
                except Exception:
                    pass
        except Exception:
            pass

    def _apply_mode(self) -> None:
        """Enable/disable widgets depending on whether a project (HPO YAML) is loaded."""
        # Allow editing/browsing even in project mode so users can tweak output dir
        # and point at an equivalent model path without being blocked.
        for w in (self.model_le, self.base_model_le, self.output_dir_le):
            w.setReadOnly(False)
        for b in (self.btn_load_schema, self.btn_browse_base, self.btn_browse_out):
            b.setEnabled(True)
        # Save button is always available; will prompt for a path if needed
        self.btn_export.setEnabled(True)

    def _init_ui(self) -> None:
        cw = QWidget(self)
        cw.setObjectName("MainBg")
        self.setCentralWidget(cw)
        v = QVBoxLayout(cw)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(10)

        # Top actions
        actions = QHBoxLayout()
        v.addLayout(actions)
        btn_load_yaml = QPushButton("Open HPO Config…")
        actions.addWidget(btn_load_yaml)
        btn_load_yaml.clicked.connect(self._on_load_hpo_yaml)
        btn_new_from_model = QPushButton("New From Model…")
        actions.addWidget(btn_new_from_model)
        btn_new_from_model.clicked.connect(self._on_new_from_model)
        self.btn_export = QPushButton("Save HPO YAML…")
        self.btn_export.setEnabled(False)
        actions.addWidget(self.btn_export)
        self.btn_export.clicked.connect(self._on_export)
        # Theme toggle
        self.cb_dark = QCheckBox("Dark mode")
        self.cb_dark.toggled.connect(self._apply_theme)
        # Default to dark theme to avoid poor contrast on systems using dark mode
        self.cb_dark.setChecked(True)
        actions.addWidget(self.cb_dark)
        actions.addStretch(1)

        # Tabs
        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Config tab (scrollable)
        self.tab_config = QScrollArea()
        self.tab_config.setWidgetResizable(True)
        config_page = QWidget()
        cv = QVBoxLayout(config_page)
        # Paths group
        paths_box = QGroupBox("Paths")
        cv.addWidget(paths_box)
        pf = QFormLayout(paths_box)
        # Model path (or trial params)
        pr1 = QHBoxLayout()
        self.model_le = QLineEdit()
        pr1.addWidget(self.model_le, 1)
        btn_browse_model = QPushButton("Browse…")
        btn_browse_model.clicked.connect(self._on_browse_model)
        self.btn_load_schema = QPushButton("Load Model")
        self.btn_load_schema.clicked.connect(self._on_load)
        pr1.addWidget(btn_browse_model)
        pr1.addWidget(self.btn_load_schema)
        w_pr1 = QWidget()
        w_pr1.setLayout(pr1)
        pf.addRow("Model file:", w_pr1)
        # Optional base model (when model is a trial params dump)
        pr2 = QHBoxLayout()
        self.base_model_le = QLineEdit()
        pr2.addWidget(self.base_model_le, 1)
        self.btn_browse_base = QPushButton("Browse…")
        self.btn_browse_base.clicked.connect(self._on_browse_base)
        pr2.addWidget(self.btn_browse_base)
        w_pr2 = QWidget()
        w_pr2.setLayout(pr2)
        pf.addRow("Base model (optional):", w_pr2)
        # Output dir
        pr3 = QHBoxLayout()
        self.output_dir_le = QLineEdit()
        pr3.addWidget(self.output_dir_le, 1)
        self.btn_browse_out = QPushButton("Browse…")
        self.btn_browse_out.clicked.connect(self._on_browse_outdir)
        pr3.addWidget(self.btn_browse_out)
        w_pr3 = QWidget()
        w_pr3.setLayout(pr3)
        pf.addRow("Output directory:", w_pr3)
        # (Removed) HPO YAML path field – saving now prompts for a path when needed

        # training config path
        pr4 = QHBoxLayout()
        self.train_config_le = QLineEdit()
        pr4.addWidget(self.train_config_le, 1)
        self.btn_browse_train_config = QPushButton("Browse…")
        self.btn_browse_train_config.clicked.connect(self._on_browse_train_config)
        pr4.addWidget(self.btn_browse_train_config)
        self.w_pr4 = QWidget()
        self.w_pr4.setLayout(pr4)
        pf.addRow("Training config (epoch mode):", self.w_pr4)

        # Run group
        run_box = QGroupBox("Run Settings")
        cv.addWidget(run_box)
        rf = QFormLayout(run_box)
        self.cmb_hpo_mode = QComboBox()
        self.cmb_hpo_mode.addItems(["forward", "epoch"])
        self.cmb_hpo_mode.currentTextChanged.connect(self._on_hpo_mode_changed)
        rf.addRow("HPO Mode:", self.cmb_hpo_mode)
        self.sb_trials = QSpinBox()
        self.sb_trials.setRange(1, 1000000)
        self.sb_trials.setValue(100)
        rf.addRow("n_trials:", self.sb_trials)
        self.sb_jobs = QSpinBox()
        self.sb_jobs.setRange(1, 512)
        self.sb_jobs.setValue(1)
        rf.addRow("n_jobs:", self.sb_jobs)
        self.sb_timeout = QSpinBox()
        self.sb_timeout.setRange(0, 7 * 24 * 3600)
        self.sb_timeout.setValue(0)
        rf.addRow("timeout (s, 0=none):", self.sb_timeout)
        self.sb_seed = QSpinBox()
        self.sb_seed.setRange(0, 10**9)
        self.sb_seed.setValue(42)
        rf.addRow("seed:", self.sb_seed)
        self.le_study_name = QLineEdit()
        rf.addRow("study_name:", self.le_study_name)
        self.cb_resume = QCheckBox("Resume existing study if found")
        rf.addRow(self.cb_resume)
        self.cb_profile = QCheckBox("Enable profiling")
        rf.addRow(self.cb_profile)

        # Simulation group
        sim_box = QGroupBox("Simulation")
        cv.addWidget(sim_box)
        sf = QFormLayout(sim_box)
        self.sb_seq_len = QSpinBox()
        self.sb_seq_len.setRange(1, 10**7)
        self.sb_seq_len.setValue(200)
        sf.addRow("seq_len:", self.sb_seq_len)
        self.sb_batch = QSpinBox()
        self.sb_batch.setRange(1, 4096)
        self.sb_batch.setValue(4)
        sf.addRow("batch_size:", self.sb_batch)
        self.dsb_dt = QDoubleSpinBox()
        self.dsb_dt.setRange(0.0, 1e9)
        self.dsb_dt.setDecimals(6)
        self.dsb_dt.setValue(0.0)
        self.dsb_dt.setSingleStep(0.1)
        sf.addRow("dt override (dt units, 0=use model):", self.dsb_dt)

        # Input group
        input_box = QGroupBox("Input")
        cv.addWidget(input_box)
        inf = QFormLayout(input_box)
        self.cmb_input_kind = QComboBox()
        self.cmb_input_kind.addItems(["white_noise", "colored_noise", "gp_rbf", "log_slope_noise", "hdf5_dataset"])
        self.cmb_input_kind.currentTextChanged.connect(self._update_input_fields_visibility)
        inf.addRow("kind:", self.cmb_input_kind)

        # Params per kind
        def _add_form_row(form, label, widget):
            lab = QLabel(label)
            form.addRow(lab, widget)
            return lab

        self.dsb_delta_n = QDoubleSpinBox()
        self.dsb_delta_n.setRange(0.0, 1e9)
        self.dsb_delta_n.setValue(1.0)
        self.lbl_delta_n = _add_form_row(inf, "white_noise: delta_n:", self.dsb_delta_n)
        self.dsb_beta = QDoubleSpinBox()
        self.dsb_beta.setRange(0.0, 10.0)
        self.dsb_beta.setValue(2.0)
        self.lbl_beta = _add_form_row(inf, "colored_noise: beta:", self.dsb_beta)
        self.dsb_sigma = QDoubleSpinBox()
        self.dsb_sigma.setRange(0.0, 1e9)
        self.dsb_sigma.setValue(1.0)
        self.lbl_sigma = _add_form_row(inf, "gp_rbf: sigma:", self.dsb_sigma)
        self.dsb_ell_ns = QDoubleSpinBox()
        self.dsb_ell_ns.setRange(0.0, 1e9)
        self.dsb_ell_ns.setDecimals(6)
        self.dsb_ell_ns.setValue(0.1)
        self.lbl_ell = _add_form_row(inf, "gp_rbf: ell_ns:", self.dsb_ell_ns)
        self.dsb_slope_db = QDoubleSpinBox()
        self.dsb_slope_db.setRange(-200.0, 0.0)
        self.dsb_slope_db.setValue(-20.0)
        self.lbl_slope = _add_form_row(inf, "log_slope_noise: slope_db_per_dec:", self.dsb_slope_db)
        self.dsb_fmin_frac = QDoubleSpinBox()
        self.dsb_fmin_frac.setRange(1e-6, 1.0)
        self.dsb_fmin_frac.setDecimals(6)
        self.dsb_fmin_frac.setValue(0.01)
        self.lbl_fmin = _add_form_row(inf, "log_slope_noise: fmin_frac:", self.dsb_fmin_frac)
        self.dsb_fmax_frac = QDoubleSpinBox()
        self.dsb_fmax_frac.setRange(1e-6, 1.0)
        self.dsb_fmax_frac.setDecimals(6)
        self.dsb_fmax_frac.setValue(0.5)
        self.lbl_fmax = _add_form_row(inf, "log_slope_noise: fmax_frac:", self.dsb_fmax_frac)

        # HDF5 dataset input fields
        self.le_h5_path = QLineEdit()
        btn_h5_browse = QPushButton("Browse…")
        btn_h5_browse.clicked.connect(self._on_browse_h5)
        h5_row = QHBoxLayout()
        h5_row.addWidget(self.le_h5_path, 1)
        h5_row.addWidget(btn_h5_browse)
        self.h5_path_row_widget = QWidget()
        self.h5_path_row_widget.setLayout(h5_row)
        self.lbl_h5_path = _add_form_row(inf, "hdf5_dataset: path:", self.h5_path_row_widget)
        self.le_h5_split = QLineEdit()
        self.le_h5_split.setPlaceholderText("optional: train/val/test")
        self.lbl_h5_split = _add_form_row(inf, "hdf5_dataset: split:", self.le_h5_split)
        self.le_h5_data_key = QLineEdit()
        self.le_h5_data_key.setText("data")
        self.lbl_h5_data_key = _add_form_row(inf, "hdf5_dataset: data_key:", self.le_h5_data_key)

        # Optional scaling controls
        self.cb_scale = QCheckBox("Scale inputs to [min, max]")
        inf.addRow(self.cb_scale)
        scale_row = QHBoxLayout()
        self.dsb_scale_min = QDoubleSpinBox()
        self.dsb_scale_min.setRange(-1e12, 1e12)
        self.dsb_scale_min.setDecimals(6)
        self.dsb_scale_min.setValue(0.0)
        self.dsb_scale_max = QDoubleSpinBox()
        self.dsb_scale_max.setRange(-1e12, 1e12)
        self.dsb_scale_max.setDecimals(6)
        self.dsb_scale_max.setValue(1.0)
        scale_row.addWidget(QLabel("min:"))
        scale_row.addWidget(self.dsb_scale_min)
        scale_row.addWidget(QLabel("max:"))
        scale_row.addWidget(self.dsb_scale_max)
        self.scale_row_widget = QWidget()
        self.scale_row_widget.setLayout(scale_row)
        self.lbl_scale = _add_form_row(inf, "Scaling:", self.scale_row_widget)
        # Preview button
        btn_preview = QPushButton("Preview input…")
        btn_preview.clicked.connect(self._on_preview_input)
        inf.addRow(btn_preview)
        # Set initial visibility
        self._update_input_fields_visibility()

        # Objective group
        obj_box = QGroupBox("Objective")
        cv.addWidget(obj_box)
        of = QFormLayout(obj_box)
        # Brief description for newcomers
        obj_desc = QLabel(
            "Combine one or more metrics into a single cost (lower is better).\n"
            "Set a weight to 0.0 to disable that metric. A simple starting point is\n"
            "w_branch = 1.0 and others = 0.0; add more terms as needed.",
        )
        obj_desc.setWordWrap(True)
        of.addRow(obj_desc)
        # Metric toggles + weights
        self.cb_w_branch = QCheckBox("Branching (w_branch)")
        self.dsb_w_branch = QDoubleSpinBox()
        self.dsb_w_branch.setRange(0.0, 1000.0)
        self.dsb_w_branch.setValue(1.0)
        self.dsb_w_branch.setToolTip(
            "Branching ratio cost. sigma≈1 at criticality.\nCost = (1 - sigma)^2; increase weight to favor sigma→1.",
        )
        self.cb_w_branch.toggled.connect(lambda on: self.dsb_w_branch.setEnabled(on))
        self.cb_w_branch.setChecked(True)
        of.addRow(self.cb_w_branch, self.dsb_w_branch)
        self.cb_w_psd_t = QCheckBox("Temporal PSD (w_psd_t)")
        self.dsb_w_psd_t = QDoubleSpinBox()
        self.dsb_w_psd_t.setRange(0.0, 1000.0)
        self.dsb_w_psd_t.setValue(0.0)
        self.dsb_w_psd_t.setToolTip(
            "Temporal power spectral density slope target (beta).\nCost = (beta_est - target_beta_t)^2; 0 disables.",
        )
        self.cb_w_psd_t.toggled.connect(lambda on: (self.dsb_w_psd_t.setEnabled(on), self.dsb_target_beta_t.setEnabled(on)))
        self.cb_w_psd_t.setChecked(False)
        of.addRow(self.cb_w_psd_t, self.dsb_w_psd_t)
        self.cb_w_psd_sp = QCheckBox("Spatial PSD (w_psd_spatial)")
        self.dsb_w_psd_sp = QDoubleSpinBox()
        self.dsb_w_psd_sp.setRange(0.0, 1000.0)
        self.dsb_w_psd_sp.setValue(0.0)
        self.dsb_w_psd_sp.setToolTip(
            "Spatial PSD slope target (beta) across neurons; experimental.\nCost = (beta_est - target_beta_spatial)^2; 0 disables.",
        )
        self.cb_w_psd_sp.toggled.connect(lambda on: (self.dsb_w_psd_sp.setEnabled(on), self.dsb_target_beta_sp.setEnabled(on)))
        self.cb_w_psd_sp.setChecked(False)
        of.addRow(self.cb_w_psd_sp, self.dsb_w_psd_sp)
        self.cb_w_chi_inv = QCheckBox("Inverse susceptibility (w_chi_inv)")
        self.dsb_w_chi_inv = QDoubleSpinBox()
        self.dsb_w_chi_inv.setRange(0.0, 1000.0)
        self.dsb_w_chi_inv.setValue(0.0)
        self.dsb_w_chi_inv.setToolTip(
            "Inverse susceptibility cost: encourages higher variance (susceptibility).\nCost ≈ 1/chi; use small weight to avoid dominance.",
        )
        self.cb_w_chi_inv.toggled.connect(lambda on: self.dsb_w_chi_inv.setEnabled(on))
        self.cb_w_chi_inv.setChecked(False)
        of.addRow(self.cb_w_chi_inv, self.dsb_w_chi_inv)
        self.cb_w_avalanche = QCheckBox("Avalanches (w_avalanche)")
        self.dsb_w_avalanche = QDoubleSpinBox()
        self.dsb_w_avalanche.setRange(0.0, 1000.0)
        self.dsb_w_avalanche.setValue(0.0)
        self.dsb_w_avalanche.setToolTip(
            "Avalanche size distribution cost (scale-free behavior).\nProxy via coefficient of variation; higher CV lowers cost.",
        )
        self.cb_w_avalanche.toggled.connect(lambda on: self.dsb_w_avalanche.setEnabled(on))
        self.cb_w_avalanche.setChecked(False)
        of.addRow(self.cb_w_avalanche, self.dsb_w_avalanche)
        self.cb_w_autocorr = QCheckBox("Autocorr (w_autocorr)")
        self.dsb_w_autocorr = QDoubleSpinBox()
        self.dsb_w_autocorr.setRange(0.0, 1000.0)
        self.dsb_w_autocorr.setValue(0.0)
        self.dsb_w_autocorr.setToolTip(
            "Autocorrelation decay cost; fits power-law exponent alpha (≈1 near critical).\nCost = |alpha - 1|; 0 disables.",
        )
        self.cb_w_autocorr.toggled.connect(lambda on: self.dsb_w_autocorr.setEnabled(on))
        self.cb_w_autocorr.setChecked(False)
        of.addRow(self.cb_w_autocorr, self.dsb_w_autocorr)
        self.cb_w_jacobian = QCheckBox("Jacobian ρ(J) (w_jacobian)")
        self.dsb_w_jacobian = QDoubleSpinBox()
        self.dsb_w_jacobian.setRange(0.0, 1000.0)
        self.dsb_w_jacobian.setValue(0.0)
        self.dsb_w_jacobian.setToolTip(
            "Jacobian spectral radius cost: encourages rho(J)≈1 (critical).\nCost = |rho(J) - 1| computed from state trajectories.",
        )
        self.cb_w_jacobian.toggled.connect(lambda on: self.dsb_w_jacobian.setEnabled(on))
        self.cb_w_jacobian.setChecked(False)
        of.addRow(self.cb_w_jacobian, self.dsb_w_jacobian)
        # Lyapunov exponent
        self.cb_w_lyapunov = QCheckBox("Lyapunov λ₁ (w_lyapunov)")
        self.dsb_w_lyapunov = QDoubleSpinBox()
        self.dsb_w_lyapunov.setRange(0.0, 1000.0)
        self.dsb_w_lyapunov.setValue(0.0)
        self.dsb_w_lyapunov.setToolTip(
            "Largest Lyapunov exponent (Benettin). Target λ≈0 at criticality.\nCost is near-quadratic around 0; linear tails.",
        )
        self.cb_w_lyapunov.toggled.connect(lambda on: self.dsb_w_lyapunov.setEnabled(on))
        self.cb_w_lyapunov.setChecked(False)
        of.addRow(self.cb_w_lyapunov, self.dsb_w_lyapunov)
        self.dsb_target_beta_t = QDoubleSpinBox()
        self.dsb_target_beta_t.setRange(0.0, 10.0)
        self.dsb_target_beta_t.setValue(2.0)
        self.dsb_target_beta_t.setToolTip(
            "Target temporal PSD slope beta (typical ~2.0).",
        )
        self.dsb_target_beta_t.setEnabled(False)
        of.addRow("target_beta_t:", self.dsb_target_beta_t)
        self.dsb_target_beta_sp = QDoubleSpinBox()
        self.dsb_target_beta_sp.setRange(0.0, 10.0)
        self.dsb_target_beta_sp.setValue(2.0)
        self.dsb_target_beta_sp.setToolTip(
            "Target spatial PSD slope beta (typical ~2.0).",
        )
        self.dsb_target_beta_sp.setEnabled(False)
        of.addRow("target_beta_spatial:", self.dsb_target_beta_sp)

        self.dsb_w_lyapunov = QDoubleSpinBox()
        self.dsb_w_lyapunov.setRange(0.0, 1000.0)
        self.dsb_w_lyapunov.setValue(0.0)
        self.cb_w_lyapunov = QCheckBox("w_lyapunov")
        self.cb_w_lyapunov.toggled.connect(lambda on: self.dsb_w_lyapunov.setEnabled(on))
        of.addRow(self.cb_w_lyapunov, self.dsb_w_lyapunov)

        self.dsb_w_train_loss = QDoubleSpinBox()
        self.dsb_w_train_loss.setRange(0.0, 1000.0)
        self.dsb_w_train_loss.setValue(0.0)
        self.cb_w_train_loss = QCheckBox("w_train_loss")
        self.cb_w_train_loss.toggled.connect(lambda on: self.dsb_w_train_loss.setEnabled(on))
        of.addRow(self.cb_w_train_loss, self.dsb_w_train_loss)

        # Targets
        targets_box = QGroupBox("Targets")
        cv.addWidget(targets_box)
        tf = QFormLayout(targets_box)
        tf.addRow("target_beta_t:", self.dsb_target_beta_t)
        tf.addRow("target_beta_spatial:", self.dsb_target_beta_sp)

        # Inline help button for fuller guidance
        btn_obj_help = QPushButton("Objective Help…")
        btn_obj_help.clicked.connect(self._show_objective_help)
        of.addRow(btn_obj_help)

        # Optuna/Pruner group
        opt_box = QGroupBox("Optuna")
        cv.addWidget(opt_box)
        of2 = QFormLayout(opt_box)
        self.cmb_sampler = QComboBox()
        self.cmb_sampler.addItems(["tpe", "cma", "random", "nsgaii", "botorch"])
        self.cmb_sampler.currentTextChanged.connect(self._on_sampler_changed)
        of2.addRow("sampler:", self.cmb_sampler)
        # Sampler-specific panels
        # TPE
        self.gb_tpe = QGroupBox("TPE options")
        tpef = QFormLayout(self.gb_tpe)
        self.sb_tpe_startup = QSpinBox()
        self.sb_tpe_startup.setRange(0, 100000)
        self.sb_tpe_startup.setValue(64)
        tpef.addRow("n_startup_trials:", self.sb_tpe_startup)
        self.sb_tpe_ei = QSpinBox()
        self.sb_tpe_ei.setRange(1, 100000)
        self.sb_tpe_ei.setValue(64)
        tpef.addRow("n_ei_candidates:", self.sb_tpe_ei)
        self.cb_tpe_multivariate = QCheckBox("multivariate")
        tpef.addRow(self.cb_tpe_multivariate)
        self.cb_tpe_group = QCheckBox("group")
        tpef.addRow(self.cb_tpe_group)
        self.cb_tpe_liar = QCheckBox("constant_liar")
        self.cb_tpe_liar.setChecked(True)
        tpef.addRow(self.cb_tpe_liar)
        of2.addRow(self.gb_tpe)
        # CMA-ES
        self.gb_cma = QGroupBox("CMA-ES options")
        cmaf = QFormLayout(self.gb_cma)
        self.sb_cma_pop = QSpinBox()
        self.sb_cma_pop.setRange(0, 100000)
        self.sb_cma_pop.setValue(24)
        cmaf.addRow("population_size:", self.sb_cma_pop)
        self.dsb_cma_sigma0 = QDoubleSpinBox()
        self.dsb_cma_sigma0.setRange(1e-9, 1e9)
        self.dsb_cma_sigma0.setDecimals(6)
        self.dsb_cma_sigma0.setValue(0.4)
        cmaf.addRow("sigma0:", self.dsb_cma_sigma0)
        self.cmb_cma_restart = QComboBox()
        self.cmb_cma_restart.addItems(["none", "ipop", "bipop"])
        cmaf.addRow("restart_strategy:", self.cmb_cma_restart)
        self.cb_cma_warn = QCheckBox("warn_independent_sampling")
        cmaf.addRow(self.cb_cma_warn)
        self.cb_cma_use_indep = QCheckBox("Use independent sampler")
        self.cb_cma_use_indep.toggled.connect(self._update_indep_visibility)
        cmaf.addRow(self.cb_cma_use_indep)
        # Independent sampler subpanel
        self.gb_indep = QGroupBox("Independent sampler")
        self.gb_indep.setVisible(False)
        inf2 = QFormLayout(self.gb_indep)
        self.cmb_indep_name = QComboBox()
        self.cmb_indep_name.addItems(["tpe", "random"])
        self.cmb_indep_name.currentTextChanged.connect(self._update_indep_visibility)
        inf2.addRow("name:", self.cmb_indep_name)
        # TPE suboptions
        self.indep_tpe_box = QGroupBox("TPE options")
        itf = QFormLayout(self.indep_tpe_box)
        self.sb_itpe_startup = QSpinBox()
        self.sb_itpe_startup.setRange(0, 100000)
        self.sb_itpe_startup.setValue(64)
        itf.addRow("n_startup_trials:", self.sb_itpe_startup)
        self.sb_itpe_ei = QSpinBox()
        self.sb_itpe_ei.setRange(1, 100000)
        self.sb_itpe_ei.setValue(64)
        itf.addRow("n_ei_candidates:", self.sb_itpe_ei)
        self.cb_itpe_multivariate = QCheckBox("multivariate")
        itf.addRow(self.cb_itpe_multivariate)
        self.cb_itpe_group = QCheckBox("group")
        itf.addRow(self.cb_itpe_group)
        self.cb_itpe_liar = QCheckBox("constant_liar")
        itf.addRow(self.cb_itpe_liar)
        inf2.addRow(self.indep_tpe_box)
        cmaf.addRow(self.gb_indep)
        of2.addRow(self.gb_cma)
        # NSGA-II
        self.gb_nsga = QGroupBox("NSGA-II options")
        nsf = QFormLayout(self.gb_nsga)
        self.sb_nsga_pop = QSpinBox()
        self.sb_nsga_pop.setRange(2, 100000)
        self.sb_nsga_pop.setValue(32)
        nsf.addRow("population_size:", self.sb_nsga_pop)
        of2.addRow(self.gb_nsga)
        # BoTorch
        self.gb_bo = QGroupBox("BoTorch options")
        bof = QFormLayout(self.gb_bo)
        self.sb_bo_startup = QSpinBox()
        self.sb_bo_startup.setRange(0, 100000)
        self.sb_bo_startup.setValue(20)
        bof.addRow("n_startup_trials:", self.sb_bo_startup)
        self.cmb_bo_candidates = QComboBox()
        self.cmb_bo_candidates.addItems(["qnei", "qei"])
        bof.addRow("candidates_func:", self.cmb_bo_candidates)
        of2.addRow(self.gb_bo)
        # Random has no options
        self._on_sampler_changed(self.cmb_sampler.currentText())

        pr_box = QGroupBox("Pruner")
        cv.addWidget(pr_box)
        prf = QFormLayout(pr_box)
        self.cb_prune_use = QCheckBox("Use MedianPruner")
        prf.addRow(self.cb_prune_use)
        self.sb_prune_startup = QSpinBox()
        self.sb_prune_startup.setRange(0, 100000)
        self.sb_prune_startup.setValue(20)
        prf.addRow("n_startup_trials:", self.sb_prune_startup)
        self.sb_prune_warmup = QSpinBox()
        self.sb_prune_warmup.setRange(0, 100000)
        self.sb_prune_warmup.setValue(5)
        prf.addRow("n_warmup_steps:", self.sb_prune_warmup)

        cv.addStretch(1)
        self.tab_config.setWidget(config_page)
        self.tabs.addTab(self.tab_config, "Config")

        # Model Visualization tab (reuse model_creation_gui VisualisationTab when available)
        self.tab_model_viz = QScrollArea()
        self.tab_model_viz.setWidgetResizable(True)
        if (_MCG_ModelManager is not None) and (_MCG_VisualisationTab is not None):
            try:
                self._mcg_mgr = _MCG_ModelManager()
                self.model_viz_widget = _MCG_VisualisationTab(self._mcg_mgr)
                self.tab_model_viz.setWidget(self.model_viz_widget)
            except Exception:
                self.model_viz_widget = ModelVisualizationWidget()
                self.tab_model_viz.setWidget(self.model_viz_widget)
        else:
            self.model_viz_widget = ModelVisualizationWidget()
            self.tab_model_viz.setWidget(self.model_viz_widget)
        self.tabs.addTab(self.tab_model_viz, "Model View")

        # Targets & Space tab (split view with wider editors)
        self.tab_targets = QScrollArea()
        self.tab_targets.setWidgetResizable(True)
        targets_page = QWidget()
        tv = QVBoxLayout(targets_page)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        tv.addWidget(splitter, 1)

        # Left panel: enabled components + targets
        left = QWidget()
        leftv = QVBoxLayout(left)
        comp_box = QGroupBox("Enabled Components")
        comp_form = QFormLayout(comp_box)
        self.cb_layers = QCheckBox("Layers")
        self.cb_layers.setChecked(True)
        self.cb_conns = QCheckBox("Connections")
        self.cb_conns.setChecked(True)
        self.cb_weights = QCheckBox("Weights")
        self.cb_weights.setChecked(True)
        comp_form.addRow(self.cb_layers)
        comp_form.addRow(self.cb_conns)
        comp_form.addRow(self.cb_weights)
        leftv.addWidget(comp_box)
        target_box = QGroupBox("Select Targets")
        th = QHBoxLayout(target_box)
        # layers list
        self.list_layers = QListWidget()
        self.list_layers.setSelectionMode(self.list_layers.SelectionMode.NoSelection)
        self.list_layers.setMinimumWidth(260)
        th.addWidget(self.list_layers, 1)
        # connections list
        self.list_conns = QListWidget()
        self.list_conns.setSelectionMode(self.list_conns.SelectionMode.NoSelection)
        self.list_conns.setMinimumWidth(260)
        th.addWidget(self.list_conns, 1)
        leftv.addWidget(target_box, 1)
        leftv.addStretch(1)

        # Right panel: space editors tabs
        right = QWidget()
        rightv = QVBoxLayout(right)
        self.space_tabs = QTabWidget()
        self.space_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Layers
        self.layer_space_scroll = QScrollArea()
        self.layer_space_scroll.setWidgetResizable(True)
        self.layer_space_page = QWidget()
        self.layer_space_layout = QVBoxLayout(self.layer_space_page)
        self.layer_space_scroll.setWidget(self.layer_space_page)
        self.space_tabs.addTab(self.layer_space_scroll, "Layers")
        # Connections
        self.conn_space_scroll = QScrollArea()
        self.conn_space_scroll.setWidgetResizable(True)
        self.conn_space_page = QWidget()
        self.conn_space_layout = QVBoxLayout(self.conn_space_page)
        self.conn_space_scroll.setWidget(self.conn_space_page)
        self.space_tabs.addTab(self.conn_space_scroll, "Connections")
        # Weights
        self.weight_space_scroll = QScrollArea()
        self.weight_space_scroll.setWidgetResizable(True)
        self.weight_space_page = QWidget()
        self.weight_space_layout = QVBoxLayout(self.weight_space_page)
        self.weight_space_scroll.setWidget(self.weight_space_page)
        self.space_tabs.addTab(self.weight_space_scroll, "Weights")
        rightv.addWidget(self.space_tabs, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        tv.addStretch(0)
        self.tab_targets.setWidget(targets_page)
        self.tabs.addTab(self.tab_targets, "Targets · Space")

        # Run tab (scrollable)
        self.tab_run = QScrollArea()
        self.tab_run.setWidgetResizable(True)
        run_page = QWidget()
        rv = QVBoxLayout(run_page)
        run_controls = QHBoxLayout()
        self.btn_start = QPushButton("Start Optimization")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_start.setEnabled(False)
        run_controls.addWidget(self.btn_start)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause)
        run_controls.addWidget(self.btn_pause)
        self.btn_resume = QPushButton("Resume")
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self._on_resume)
        run_controls.addWidget(self.btn_resume)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        run_controls.addWidget(self.btn_stop)
        run_controls.addStretch(1)
        rv.addLayout(run_controls)
        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        rv.addWidget(self.progress)
        # Live optimization plot (in Run tab)
        self.plot_live_web = None
        self.plot_live = None
        if (QWebEngineView is not None) and (_go is not None) and (_pio is not None):
            try:
                self.plot_live_web = QWebEngineView()
                rv.addWidget(self.plot_live_web, 1)
            except Exception:
                self.plot_live_web = None
        if self.plot_live_web is None and pg is not None:
            try:
                self.plot_live = pg.PlotWidget(title="Optimization Progress")
                self.plot_live.showGrid(x=True, y=True, alpha=0.3)
                self.plot_live.addLegend(offset=(10, 10))
                rv.addWidget(self.plot_live, 1)
            except Exception:
                self.plot_live = None
        # Log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        rv.addWidget(self.log, 1)
        # Save log button
        save_log_row = QHBoxLayout()
        self.btn_save_log = QPushButton("Save Log…")
        self.btn_save_log.clicked.connect(self._on_save_log)
        save_log_row.addWidget(self.btn_save_log)
        save_log_row.addStretch(1)
        rv.addLayout(save_log_row)
        self.tab_run.setWidget(run_page)
        self.tabs.addTab(self.tab_run, "Run")

        # Results tab (scrollable) with inner tabs
        self.tab_results = QScrollArea()
        self.tab_results.setWidgetResizable(True)
        results_page = QWidget()
        resv = QVBoxLayout(results_page)
        self.analysis_tabs = QTabWidget()
        resv.addWidget(self.analysis_tabs, 1)

        # Tools sub-tab
        tools_page = QWidget()
        tl = QVBoxLayout(tools_page)
        # Study directory path
        stud_box = QGroupBox("Study Directory")
        stf = QFormLayout(stud_box)
        self.le_study_dir = QLineEdit()
        self.le_study_dir.setPlaceholderText("Study directory will appear here after run/study name set")
        stf.addRow("Path:", self.le_study_dir)
        # Trial extraction
        extract_box = QGroupBox("Extract Trial Spec")
        exl = QFormLayout(extract_box)
        self.sb_extract_trial = QSpinBox()
        self.sb_extract_trial.setRange(0, 10**9)
        self.sb_extract_trial.setValue(0)
        exl.addRow("Trial number:", self.sb_extract_trial)
        out_row = QHBoxLayout()
        self.le_extract_output = QLineEdit()
        self.le_extract_output.setPlaceholderText("Optional: output .yaml path (else defaults to study dir)")
        btn_browse_out = QPushButton("Browse…")

        def _browse_out() -> None:
            p, _ = QFileDialog.getSaveFileName(self, "Save extracted spec as…", str(Path.cwd() / "trial_spec.yaml"), "YAML (*.yaml *.yml)")
            if p:
                self.le_extract_output.setText(p)

        btn_browse_out.clicked.connect(_browse_out)
        out_row.addWidget(self.le_extract_output)
        out_row.addWidget(btn_browse_out)
        w_out = QWidget()
        w_out.setLayout(out_row)
        exl.addRow("Output path:", w_out)
        self.btn_extract = QPushButton("Extract from Trial")
        self.btn_extract.clicked.connect(self._on_extract_trial)
        exl.addRow(self.btn_extract)
        # Tools toolbar
        results_toolbar = QHBoxLayout()
        self.btn_results_load = QPushButton("Load Results")
        self.btn_results_load.setToolTip("Load and display results from the study directory above")
        self.btn_results_load.clicked.connect(lambda: self._load_results_from_dir(self.le_study_dir.text().strip()))
        self.btn_results_refresh = QPushButton("Refresh")
        self.btn_results_refresh.setToolTip("Reload results from the same study directory")
        self.btn_results_refresh.clicked.connect(lambda: self._load_results_from_dir(self.le_study_dir.text().strip()))
        results_toolbar.addWidget(self.btn_results_load)
        results_toolbar.addWidget(self.btn_results_refresh)
        results_toolbar.addStretch(1)
        # Assemble tools page
        tl.addWidget(stud_box)
        tl.addWidget(extract_box)
        tl.addLayout(results_toolbar)
        tl.addStretch(1)
        self.analysis_tabs.addTab(tools_page, "Tools")

        # Results sub-tab (contains the overview/plots/trials triage)
        results_only_page = QWidget()
        rl = QVBoxLayout(results_only_page)
        self.results_view = ResultsViewer()
        rl.addWidget(self.results_view, 1)
        self.analysis_tabs.addTab(results_only_page, "Results")

        # (Removed embedded web view)
        self.web = None
        self.tab_results.setWidget(results_page)
        self.tabs.addTab(self.tab_results, "Analysis")

        # Status
        self.statusBar().showMessage("Ready. Load a model or HPO YAML.")

    def _load_results_from_dir(self, study_dir: str) -> None:
        if not study_dir:
            QMessageBox.information(self, "Results", "Enter or select a study directory first.")
            return
        try:
            self.results_view.load_study_dir(study_dir)
            self.statusBar().showMessage(f"Results loaded from: {study_dir}")
        except Exception as e:
            QMessageBox.warning(self, "Results", f"Failed to load results: {e}")

    # ————— Live results refresh —————
    def _current_study_dir_hint(self) -> str | None:
        txt = self.le_study_dir.text().strip()
        if txt and os.path.isdir(txt):
            return txt
        out_dir = self.output_dir_le.text().strip()
        sname = self.le_study_name.text().strip()
        if out_dir and sname:
            sd = os.path.join(out_dir, f"optuna_report_{sname}")
            return sd if os.path.isdir(sd) else None
        return None

    def _on_results_tick(self) -> None:
        # Avoid overlapping loads
        if self._results_worker is not None and self._results_worker.isRunning():
            return
        sd = self._current_study_dir_hint()
        if not sd:
            return
        try:
            worker = _ResultsWorker(sd)
            worker.finished_data.connect(self._on_results_ready)
            self._results_worker = worker
            worker.start()
        except Exception:
            pass

    def _on_results_ready(self, study_dir: str, data: dict) -> None:
        try:
            self.results_view.apply_processed(study_dir, data)
            # Update Run tab live plot as well
            try:
                df = data.get("trials_df")
                ss = data.get("study_summary")
                if df is not None:
                    # Route through unified refresh (Plotly or PyQtGraph)
                    self._refresh_live_plot(df, ss)
            except Exception:
                pass
        except Exception:
            pass

    def _on_tab_changed(self, idx: int) -> None:
        # Lazy-build space editors if schema loaded and area still empty
        try:
            if self.tabs.tabText(idx).startswith("Targets") and self._schema is not None:
                if (self.layer_space_layout.count() == 0) or (self.layer_space_layout.count() == 1):
                    self._build_space_editors(self._schema)
        except Exception:
            pass

    # ————— File pickers —————
    def _on_browse_model(self) -> None:
        # Allow changing model path even in project mode; we will validate spec equivalence at save time
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Choose model (spec or full model)",
            str(Path.cwd()),
            "Model spec/Full model (*.yaml *.yml *.json *.soen *.pth);;All files (*)",
        )
        if p:
            self.model_le.setText(p)

    def _on_browse_base(self) -> None:
        if self._project_mode:
            QMessageBox.information(self, "Project Loaded", "Use 'New From Model…' to create a new HPO config.")
            return
        p, _ = QFileDialog.getOpenFileName(self, "Choose base model spec", str(Path.cwd()), "YAML/JSON (*.yaml *.yml *.json);;All files (*)")
        if p:
            self.base_model_le.setText(p)

    def _on_browse_outdir(self) -> None:
        # Allow picking a new output directory in any mode
        d = QFileDialog.getExistingDirectory(self, "Choose output directory", str(Path.cwd()))
        if d:
            self.output_dir_le.setText(d)
            self._output_dir = d

    def _on_browse_train_config(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Select Training Config", self.train_config_le.text() or os.getcwd(), "YAML files (*.yaml *.yml)")
        if p:
            self.train_config_le.setText(p)

    def _on_hpo_mode_changed(self, *args) -> None:
        is_epoch = self.cmb_hpo_mode.currentText() == "epoch"
        self.w_pr4.setEnabled(is_epoch)

    def _on_browse_h5(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Choose HDF5 file", str(Path.cwd()), "HDF5 (*.h5 *.hdf5);;All files (*)")
        if p:
            self.le_h5_path.setText(p)
            # Auto-suggest split based on common layout (train/val/test)
            try:
                import h5py  # local import to keep GUI import light

                # Use consistent flags to avoid HDF5 file-locking mismatch on reopen
                with h5py.File(p, "r", swmr=True, libver="latest", locking=False) as f:
                    candidates = []
                    for name in ("train", "val", "test"):
                        if name in f and isinstance(f[name], h5py.Group):
                            g = f[name]
                            if "data" in g:
                                candidates.append(name)
                    if candidates:
                        # prefer 'train'
                        sel = "train" if "train" in candidates else candidates[0]
                        self.le_h5_split.setText(sel)
                    elif "data" in f:
                        # single-dataset layout; leave split empty
                        self.le_h5_split.setText("")
            except Exception:
                pass

    # ————— Dynamic UI reactions —————
    def _update_input_fields_visibility(self) -> None:
        kind = self.cmb_input_kind.currentText()

        def show_pair(lbl, w, show) -> None:
            lbl.setVisible(show)
            w.setVisible(show)

        show_pair(self.lbl_delta_n, self.dsb_delta_n, kind == "white_noise")
        show_pair(self.lbl_beta, self.dsb_beta, kind == "colored_noise")
        show_pair(self.lbl_sigma, self.dsb_sigma, kind == "gp_rbf")
        show_pair(self.lbl_ell, self.dsb_ell_ns, kind == "gp_rbf")
        show_pair(self.lbl_slope, self.dsb_slope_db, kind == "log_slope_noise")
        show_pair(self.lbl_fmin, self.dsb_fmin_frac, kind == "log_slope_noise")
        show_pair(self.lbl_fmax, self.dsb_fmax_frac, kind == "log_slope_noise")
        # HDF5 visibility
        show_pair(self.lbl_h5_path, self.h5_path_row_widget, kind == "hdf5_dataset")
        show_pair(self.lbl_h5_split, self.le_h5_split, kind == "hdf5_dataset")
        show_pair(self.lbl_h5_data_key, self.le_h5_data_key, kind == "hdf5_dataset")
        # Scaling controls are always visible; checkbox toggles whether they apply

    def _on_sampler_changed(self, name: str) -> None:
        name = (name or "").lower()
        self.gb_tpe.setVisible(name == "tpe")
        self.gb_cma.setVisible(name == "cma")
        self.gb_nsga.setVisible(name == "nsgaii")
        self.gb_bo.setVisible(name == "botorch")
        # nothing for random
        if name != "cma":
            self.gb_indep.setVisible(False)

    def _update_indep_visibility(self) -> None:
        show = self.cb_cma_use_indep.isChecked() and (self.cmb_indep_name.currentText() == "tpe")
        self.gb_indep.setVisible(self.cb_cma_use_indep.isChecked())
        self.indep_tpe_box.setVisible(show)

    # Preview input generator
    def _effective_dt_units(self) -> float:
        # Use override if >0 else try reading from model YAML
        if float(self.dsb_dt.value()) > 0:
            return float(self.dsb_dt.value())
        # Prefer loaded schema if available
        if getattr(self, "_schema", None):
            try:
                sim = self._schema.get("simulation") or {}
                dt = sim.get("dt")
                if dt is not None:
                    return float(dt)
            except Exception:
                pass
        # fallback: read model spec file
        try:
            import yaml as _y

            with open(self._model_path or "") as f:
                spec = _y.safe_load(f) or {}
            sim = spec.get("simulation") or spec.get("sim_config") or {}
            dt = sim.get("dt")
            return float(dt) if dt is not None else 1.0
        except Exception:
            return 1.0

    def _input_dim_from_model(self) -> int:
        # Prefer in-memory schema
        try:
            if getattr(self, "_schema", None):
                layers = self._schema.get("layers") or {}
                # layers is a mapping of ids to schemas
                if isinstance(layers, dict):
                    # layer 0 preferred
                    if "0" in layers or 0 in layers:
                        l0 = layers.get("0") or layers.get(0) or {}
                        dim = (l0.get("params_schema") or {}).get("dim", {})
                        d = dim.get("default_value")
                        if isinstance(d, int) and d > 0:
                            return d
                # Fall back to scan in schema
                for info in layers.values():
                    try:
                        dim = (info.get("params_schema") or {}).get("dim", {})
                        d = dim.get("default_value")
                        if isinstance(d, int) and d > 0:
                            return d
                    except Exception:
                        continue
        except Exception:
            pass
        # Try reading from a spec file on disk
        try:
            import yaml as _y

            with open(self._model_path or "") as f:
                spec = _y.safe_load(f) or {}
            for layer in spec.get("layers") or []:
                if int(layer.get("layer_id", -1)) == 0:
                    d = layer.get("params", {}).get("dim")
                    if isinstance(d, int) and d > 0:
                        return d
            for layer in spec.get("layers") or []:
                d = layer.get("params", {}).get("dim")
                if isinstance(d, int) and d > 0:
                    return d
        except Exception:
            pass
        # Try loading from full model file
        try:
            from soen_toolkit.core import SOENModelCore

            path = str(self._model_path or "")
            if path.lower().endswith((".soen", ".pth", ".json")):
                m = SOENModelCore.load(path, device=None, strict=False, verbose=False, show_logs=False)
                for cfg in getattr(m, "layers_config", []) or []:
                    try:
                        if int(getattr(cfg, "layer_id", -1)) == 0:
                            d = int(getattr(cfg, "params", {}).get("dim", 0))
                            if d > 0:
                                return d
                    except Exception:
                        continue
                # fallback: first
                for cfg in getattr(m, "layers_config", []) or []:
                    d = int(getattr(cfg, "params", {}).get("dim", 0))
                    if d > 0:
                        return d
        except Exception:
            pass
        return 1

    def _on_preview_input(self) -> None:
        try:
            import matplotlib as mpl
            import numpy as np

            mpl.use("QtAgg")
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            from matplotlib.figure import Figure
        except Exception as e:
            QMessageBox.warning(self, "Preview", f"Matplotlib is required for preview: {e}")
            return
        # Prepare params
        seq_len = min(int(self.sb_seq_len.value()), 500)
        dim = min(self._input_dim_from_model(), 5)
        dt_units = self._effective_dt_units()
        dt_ns = dt_units * 1.28e-3
        kind = self.cmb_input_kind.currentText()
        # Generate using optuna_runner if available
        x = None
        try:
            from ..optuna_runner import make_input_sequence

            mapped = {"white_noise": "white_noise", "colored_noise": "colored_noise", "gp_rbf": "gp_rbf", "log_slope_noise": "log_slope_noise"}
            if kind in mapped:
                x = make_input_sequence(
                    mapped[kind],
                    batch=1,
                    seq_len=seq_len,
                    dim=dim,
                    dt=float(dt_units) * 1.28e-12,
                    device="cpu",
                    delta_n=float(self.dsb_delta_n.value()),
                    beta=float(self.dsb_beta.value()),
                    sigma=float(self.dsb_sigma.value()),
                    ell_ns=float(self.dsb_ell_ns.value()),
                    slope_db_per_dec=float(self.dsb_slope_db.value()),
                    fmin_frac=float(self.dsb_fmin_frac.value()),
                    fmax_frac=float(self.dsb_fmax_frac.value()),
                )
                x = x.squeeze(0).cpu().numpy()  # [T,D]
        except Exception:
            pass
        if x is None:
            # Fallback: quick generators or HDF5 dataset
            t = np.arange(seq_len)
            if kind == "white_noise":
                x = np.random.randn(seq_len, dim) * np.sqrt(float(self.dsb_delta_n.value()))
            elif kind == "colored_noise":
                beta = float(self.dsb_beta.value())
                X = np.fft.rfft(np.random.randn(seq_len, dim), axis=0)
                # Use proper dt_s calculation: dt_units * 1.28e-12
                dt_s = dt_units * 1.28e-12
                f = np.fft.rfftfreq(seq_len, d=dt_s)
                f[0] = 1e-12
                gain = f ** (-beta * 0.5)
                # Limit gain more aggressively to keep amplitudes reasonable
                gain = np.clip(gain, None, 10.0)
                Xc = X * gain[:, None]
                x = np.fft.irfft(Xc, n=seq_len, axis=0)
            elif kind == "gp_rbf":
                sigma = float(self.dsb_sigma.value())
                ell_ns = float(self.dsb_ell_ns.value())
                t = np.arange(seq_len) * dt_ns
                tau = t[:, None] - t[None, :]
                K = (sigma**2) * np.exp(-0.5 * (tau / ell_ns) ** 2) + 1e-6 * np.eye(seq_len)
                L = np.linalg.cholesky(K)
                z = np.random.randn(seq_len, dim)
                x = L @ z
            elif kind == "log_slope_noise":
                slope_db_per_dec = float(self.dsb_slope_db.value())
                fmin_frac = float(self.dsb_fmin_frac.value())
                fmax_frac = float(self.dsb_fmax_frac.value())
                dt_s = dt_units * 1.28e-12

                # Generate white noise and transform
                X = np.fft.rfft(np.random.randn(seq_len, dim), axis=0)
                f = np.fft.rfftfreq(seq_len, d=dt_s)
                f[0] = 1e-12

                # Calculate frequency limits
                nyquist_freq = 1.0 / (2.0 * dt_s)
                fmin = fmin_frac * nyquist_freq
                fmax = fmax_frac * nyquist_freq

                # Create mask for frequency band
                mask = (f >= fmin) & (f <= fmax)

                # Calculate gain: convert dB/decade slope to power law exponent
                beta = slope_db_per_dec / 10.0
                gain = np.ones_like(f)
                gain[mask] = (f[mask] / fmin) ** (beta / 2.0)

                # Apply gain and convert back
                Xc = X * gain[:, None]
                x = np.fft.irfft(Xc, n=seq_len, axis=0)
            elif kind == "hdf5_dataset":
                # Lightweight preview loader using consistent HDF5 flags to avoid
                # file-locking mismatches between multiple opens within the GUI.
                try:
                    import h5py

                    h5_path = self.le_h5_path.text().strip()
                    if not h5_path:
                        msg = "No HDF5 path provided"
                        raise RuntimeError(msg)
                    split = self.le_h5_split.text().strip() or None
                    data_key = self.le_h5_data_key.text().strip() or "data"
                    # Auto-detect split if not provided
                    from soen_toolkit.training.data.dataloaders import open_hdf5_with_consistent_locking

                    if not split:
                        with open_hdf5_with_consistent_locking(h5_path) as f:
                            for name in ("train", "val", "test"):
                                if name in f and isinstance(f[name], h5py.Group) and data_key in f[name]:
                                    split = name
                                    break

                    with open_hdf5_with_consistent_locking(h5_path) as f:
                        grp = f[split] if split else f
                        if data_key not in grp:
                            msg = f"Dataset '{data_key}' not found in group '{grp.name}'"
                            raise RuntimeError(msg)
                        dset = grp[data_key]
                        if dset.shape[0] < 1:
                            msg = "HDF5 dataset has no samples"
                            raise RuntimeError(msg)
                        # Load first sample
                        sample = np.array(dset[0])
                        # Ensure 2D [T,D]
                        if sample.ndim == 1:
                            sample = sample[:, None]
                        T0, _D0 = sample.shape[0], (sample.shape[1] if sample.ndim > 1 else 1)
                        # Resample to seq_len if needed
                        if seq_len != T0:
                            x_old = np.linspace(0.0, 1.0, T0, dtype=np.float32)
                            x_new = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
                            res = np.vstack([np.interp(x_new, x_old, sample[:, i]) for i in range(sample.shape[1])]).T.astype(sample.dtype)
                            sample = res
                        x = sample
                        dim = x.shape[1] if x.ndim == 2 else 1
                except Exception as e:
                    QMessageBox.warning(self, "HDF5 Preview", f"Failed to load dataset: {e}")
                    return
            else:
                x = np.random.randn(seq_len, dim)

        # Optional scaling to [min,max]
        if self.cb_scale.isChecked() and x is not None:
            try:
                mn = float(self.dsb_scale_min.value())
                mx = float(self.dsb_scale_max.value())
                if mx == mn:
                    x = np.full_like(x, mn)
                else:
                    x_min = float(np.min(x))
                    x_max = float(np.max(x))
                    if x_max == x_min:
                        x = np.full_like(x, mn)
                    else:
                        x = mn + (mx - mn) * (x - x_min) / (x_max - x_min + 1e-12)
            except Exception:
                pass
        # Plot
        fig = Figure(figsize=(7, 3.5))
        ax = fig.add_subplot(111)
        t_ns = np.arange(seq_len) * dt_ns
        for d in range(min(dim, 3)):
            ax.plot(t_ns, x[:, d], label=f"dim {d}")
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("amplitude")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        canvas = FigureCanvas(fig)
        # Modal dialog that cleans up on close
        dlg = QDialog(self)
        dlg.setWindowTitle("Input preview")
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"kind={kind}, seq_len={seq_len}, dt={dt_units} dt_units ({dt_ns:.4f} ns)"))
        lay.addWidget(canvas)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)

        def _save_fig() -> None:
            p, _ = QFileDialog.getSaveFileName(self, "Save preview", str(Path.cwd() / "input_preview.png"), "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
            if p:
                try:
                    fig.savefig(p, bbox_inches="tight")
                except Exception as e:
                    QMessageBox.warning(self, "Save", f"Failed to save figure: {e}")

        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        btns.button(QDialogButtonBox.StandardButton.Save).clicked.connect(_save_fig)
        lay.addWidget(btns)
        dlg.resize(860, 500)
        dlg.exec()

    # ————— Load schema —————
    def _on_load(self) -> None:
        model = self.model_le.text().strip()
        base = self.base_model_le.text().strip() or None
        if not model:
            QMessageBox.warning(self, "Missing model", "Please choose a model spec file")
            return
        try:
            # Allow full model files (.soen/.pth/.json) by extracting a temporary spec
            path_l = model.lower()
            schema_model_path = model
            self._extracted_spec_path = None
            if path_l.endswith((".soen", ".pth", ".json")):
                try:
                    from dataclasses import asdict as _asdict

                    from soen_toolkit.core import SOENModelCore as _Core

                    _m = _Core.load(model, device=None, strict=False, verbose=False, show_logs=False)
                    spec_dict = {
                        "simulation": _asdict(_m.sim_config),
                        "layers": [_asdict(cfg) for cfg in _m.layers_config],
                        "connections": [_asdict(cfg) for cfg in _m.connections_config],
                    }
                    import tempfile as _tf

                    import yaml as _y

                    tmp = _tf.NamedTemporaryFile(prefix="soen_hpo_spec_", suffix=".yaml", delete=False, mode="w", encoding="utf-8")
                    _y.safe_dump(spec_dict, tmp, sort_keys=False)
                    tmp.flush()
                    tmp.close()
                    schema_model_path = tmp.name
                    self._extracted_spec_path = schema_model_path
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to extract spec from model: {e}")
                    return
            schema = build_option_schema(schema_model_path, base_model_path=base)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse schema: {e}")
            return
        self._schema = schema
        self._model_path = model
        self._base_model_path = base
        self._populate_from_schema(schema)

        # Update model visualization
        if self.model_viz_widget:
            # If using model_creation_gui visualiser, push the built/loaded model into its manager
            try:
                if hasattr(self, "_mcg_mgr") and (self._mcg_mgr is not None):
                    # Build/load the actual SOEN model and set it on the manager
                    import os as _os

                    from soen_toolkit.core import SOENModelCore as _Core
                    from soen_toolkit.core.model_yaml import (
                        build_model_from_yaml as _build_yaml,
                    )

                    ext = (_os.path.splitext(model)[1] or "").lower()
                    if ext in {".yaml", ".yml"}:
                        _m = _build_yaml(model)
                    else:
                        _m = _Core.load(model, device=None, strict=False, verbose=False, show_logs=False)
                    self._mcg_mgr.model = _m
                    self._mcg_mgr.sim_config = _m.sim_config
                    self._mcg_mgr.layers = _m.layers_config
                    self._mcg_mgr.connections = _m.connections_config
                    self._mcg_mgr.model_changed.emit()
                else:
                    # Fallback: internal lightweight visualiser
                    self.model_viz_widget.update_model(model, schema)
            except Exception:
                # Fallback gracefully to the internal light visualiser
                with contextlib.suppress(Exception):
                    self.model_viz_widget.update_model(model, schema)
            # Switch to the Model View tab to show the visualization
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "Model View":
                    self.tabs.setCurrentIndex(i)
                    break

        self.btn_export.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.statusBar().showMessage("Model schema loaded.")

    def _populate_from_schema(self, schema: dict[str, Any]) -> None:
        # Targets lists
        self.list_layers.clear()
        self.list_conns.clear()
        # layers
        layers_map: dict[str, Any] = schema.get("layers", {})
        for lid_str in sorted(layers_map, key=lambda s: int(s)):
            lid = int(lid_str)
            ltype = layers_map[lid_str].get("layer_type", "?")
            item = QListWidgetItem(f"Layer {lid} — {ltype}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Default: select non-linear layers (skip the pass-through input)
            checked = Qt.CheckState.Checked if ltype not in {"Input", "Linear"} else Qt.CheckState.Unchecked
            item.setCheckState(checked)
            self.list_layers.addItem(item)
        # connections
        conns_map: dict[str, Any] = schema.get("connections", {})
        for name in sorted(conns_map.keys()):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)  # default select all
            self.list_conns.addItem(item)
        # Default output dir suggestion
        if self._model_path and not self.output_dir_le.text().strip():
            self.output_dir_le.setText(str(Path(self._model_path).parent))

        # Build dynamic editors for search space
        self._build_space_editors(schema)
        self.statusBar().showMessage("Model schema loaded. Space editors ready.")

    def _build_space_editors(self, schema: dict[str, Any]) -> None:
        # Clear old editors
        def clear_layout(v: QVBoxLayout) -> None:
            while v.count():
                w = v.takeAt(0)
                if w and w.widget():
                    w.widget().deleteLater()

        clear_layout(self.layer_space_layout)
        clear_layout(self.conn_space_layout)
        clear_layout(self.weight_space_layout)
        self.layer_param_editors.clear()
        self.conn_param_editors.clear()
        self.conn_type_list = None
        self.weight_editor = None
        self.conn_param_boxes: dict[str, QGroupBox] = {}
        self.conn_param_applicability: dict[str, set[str]] = {}

        # Layers: union param names across used layers
        layers_map: dict[str, Any] = schema.get("layers", {})
        union_params: dict[str, dict] = {}
        for info in layers_map.values():
            pmap = info.get("params_schema") or {}
            for pname, pmeta in pmap.items():
                union_params.setdefault(pname, pmeta)
        if not union_params:
            self.layer_space_layout.addWidget(QLabel("No layer parameters discovered."))
        else:
            self.layer_space_layout.addWidget(QLabel("Layer parameters:"))
            for pname in sorted(union_params.keys()):
                editor = LayerParamEditor(pname, union_params[pname])
                box = QGroupBox()
                box.setTitle(pname)
                inner = QVBoxLayout(box)
                inner.addWidget(editor)
                self.layer_space_layout.addWidget(box)
                self.layer_param_editors[pname] = editor
        self.layer_space_layout.addStretch(1)

        # Connections
        conns_map: dict[str, Any] = schema.get("connections", {})
        any_conn = conns_map[next(iter(conns_map))] if conns_map else None
        if not any_conn:
            self.conn_space_layout.addWidget(QLabel("No connections in model."))
            self.weight_space_layout.addWidget(QLabel("No connections → no weights."))
            return
        types = list(map(str, any_conn.get("available_types", [])))
        # connection_type choices
        ct_box = QGroupBox("Connection type choices")
        ct_v = QVBoxLayout(ct_box)
        self.conn_type_list = MultiCheckList(types)
        ct_v.addWidget(self.conn_type_list)
        self.conn_space_layout.addWidget(ct_box)

        # Param meta by type
        ctype_map: dict[str, dict[str, Any]] = any_conn.get("type_params", {}) or {}
        # Build union: param -> {ctype: meta}
        param_union: dict[str, dict[str, dict]] = {}
        for tname, tinfo in ctype_map.items():
            pmap = (tinfo or {}).get("params", {}) or {}
            for pname, pmeta in pmap.items():
                param_union.setdefault(pname, {})[tname] = pmeta
            # Map selected common flags to enum choices where sensible
            common = (tinfo or {}).get("common", {}) or {}
            if "allow_self_connections" in common:
                param_union.setdefault("allow_self_connections", {})[tname] = {"type": "enum", "options": [True, False]}
        self.conn_space_layout.addWidget(QLabel("Connection parameters:"))

        # Sort parameters grouped by first applicable type to visually cluster per method
        def param_group_index(pname: str) -> int:
            applies = [t for t, m in (param_union.get(pname) or {}).items() if m]
            if not applies:
                return 10**6
            # pick the lowest index among available types
            idxs = [types.index(t) for t in applies if t in types]
            return min(idxs) if idxs else 10**6

        for pname in sorted(param_union.keys(), key=lambda n: (param_group_index(n), n)):
            ed = ConnParamEditor(pname, param_union[pname], types)
            # Title includes the connectivity methods this param applies to
            applies_to = [t for t, m in (param_union.get(pname) or {}).items() if m]
            applies_str = ", ".join(applies_to) if applies_to else "all"
            box = QGroupBox()
            box.setTitle(f"{pname} — for: {applies_str}")
            inner = QVBoxLayout(box)
            inner.addWidget(ed)
            self.conn_space_layout.addWidget(box)
            self.conn_param_editors[pname] = ed
            self.conn_param_boxes[pname] = box
            self.conn_param_applicability[pname] = set(applies_to)
        self.conn_space_layout.addStretch(1)

        # React to connection type selection changes: hide irrelevant params
        try:
            if self.conn_type_list is not None:
                self.conn_type_list.list.itemChanged.connect(self._refresh_conn_params_visibility)
                self._refresh_conn_params_visibility()
        except Exception:
            pass

        # Weights editor based on weight schema
        w_schema = any_conn.get("weight_init", {}) or {}
        self.weight_space_layout.addWidget(QLabel("Weight initialization:"))
        self.weight_editor = WeightEditor(w_schema)
        self.weight_space_layout.addWidget(self.weight_editor)
        self.weight_space_layout.addStretch(1)

    def _collect_layer_space(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for pname, ed in self.layer_param_editors.items():
            out[pname] = ed.to_dict()
        return out

    def _collect_conn_space(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        # connection_type
        if self.conn_type_list is not None:
            out["connection_type"] = {"enabled": True, "choices": self.conn_type_list.selected()}
        selected_types = set(self._selected_conn_types())
        for pname, ed in self.conn_param_editors.items():
            applies = self.conn_param_applicability.get(pname, set())
            if selected_types and not (applies & selected_types):
                # Skip params that don't apply to any selected connection type
                continue
            out[pname] = ed.to_dict()
        return out

    def _selected_conn_types(self) -> list[str]:
        if self.conn_type_list is None:
            return []
        return self.conn_type_list.selected()

    def _refresh_conn_params_visibility(self) -> None:
        selected = set(self._selected_conn_types())
        if not selected:
            # If nothing selected, show nothing but leave UI intact
            for box in self.conn_param_boxes.values():
                box.setVisible(False)
            return
        for pname, box in self.conn_param_boxes.items():
            applies = self.conn_param_applicability.get(pname, set())
            box.setVisible(bool(applies & selected))

    def _collect_weight_space(self) -> dict[str, Any]:
        if self.weight_editor is None:
            return {}
        return self.weight_editor.to_dict()

    def _show_objective_help(self) -> None:
        text = (
            "Objective combines weighted metrics into one cost (lower is better).\n\n"
            "Weights (set to 0.0 to disable):\n"
            "- w_branch: Branching ratio (sigma→1 at criticality).\n"
            "- w_psd_t: Temporal PSD slope vs target_beta_t (default 2.0).\n"
            "- w_psd_spatial: Spatial PSD slope vs target_beta_spatial.\n"
            "- w_chi_inv: Inverse susceptibility (larger variance ⇒ lower cost).\n"
            "- w_avalanche: Scale-free avalanches (proxy via high CV).\n"
            "- w_autocorr: Autocorrelation power-law decay alpha≈1.\n"
            "- w_jacobian: Spectral radius ρ(J)≈1 from states.\n"
            "- w_lyapunov: Largest Lyapunov λ₁≈0 (gold standard).\n\n"
            "Starter recipe: set w_branch=1.0 and others=0.0. If you want a single\n"
            "global criticality term, enable w_jacobian=1.0 as an alternative."
        )
        QMessageBox.information(self, "Objective Help", text)

    # ————— Export / Save YAML —————
    def _selected_layers(self) -> list[int]:
        out: list[int] = []
        for i in range(self.list_layers.count()):
            it = self.list_layers.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                try:
                    txt = it.text()
                    lid = int(txt.split()[1])
                    out.append(lid)
                except Exception:
                    continue
        return sorted(set(out))

    def _selected_conns(self) -> list[str]:
        out: list[str] = []
        for i in range(self.list_conns.count()):
            it = self.list_conns.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.text())
        return out

    def _collect_full_config(self) -> dict[str, Any]:
        """Collect the full HPO config dict from the current UI state.
        Assumes a project is loaded and paths are in the line edits.
        """
        # Paths
        paths_block = {
            "base_model_spec": self.model_le.text().strip(),
            "output_dir": self.output_dir_le.text().strip(),
            "train_config": self.train_config_le.text().strip() or None,
        }
        # Run
        run_block = {
            "hpo_mode": self.cmb_hpo_mode.currentText(),
            "n_trials": int(self.sb_trials.value()),
            "timeout": (int(self.sb_timeout.value()) or None),
            "n_jobs": int(self.sb_jobs.value()),
            "seed": int(self.sb_seed.value()),
            "study_name": (self.le_study_name.text().strip() or None),
            "profile": bool(self.cb_profile.isChecked()),
            "resume": bool(self.cb_resume.isChecked()),
        }
        # Simulation
        sim_block: dict[str, Any] = {
            "seq_len": int(self.sb_seq_len.value()),
            "batch_size": int(self.sb_batch.value()),
        }
        if float(self.dsb_dt.value()) > 0:
            sim_block["dt"] = float(self.dsb_dt.value())
        # Input
        params: dict[str, Any] = {}
        kind = self.cmb_input_kind.currentText()
        if kind == "white_noise":
            params["delta_n"] = float(self.dsb_delta_n.value())
        elif kind == "colored_noise":
            params["beta"] = float(self.dsb_beta.value())
        elif kind == "gp_rbf":
            params["sigma"] = float(self.dsb_sigma.value())
            params["ell_ns"] = float(self.dsb_ell_ns.value())
        elif kind == "log_slope_noise":
            params["slope_db_per_dec"] = float(self.dsb_slope_db.value())
            params["fmin_frac"] = float(self.dsb_fmin_frac.value())
            params["fmax_frac"] = float(self.dsb_fmax_frac.value())
        elif kind == "hdf5_dataset":
            params["path"] = self.le_h5_path.text().strip()
            if self.le_h5_split.text().strip():
                params["split"] = self.le_h5_split.text().strip()
            if self.le_h5_data_key.text().strip() and self.le_h5_data_key.text().strip() != "data":
                params["data_key"] = self.le_h5_data_key.text().strip()
        if self.cb_scale.isChecked():
            params["scale_min"] = float(self.dsb_scale_min.value())
            params["scale_max"] = float(self.dsb_scale_max.value())
        input_block = {"kind": kind, "params": params}
        # Objective
        objective_block = {
            "weights": {
                # Toggle semantics: weight==0 -> off; >0 -> on
                "w_branch": float(self.dsb_w_branch.value()) if self.cb_w_branch.isChecked() else 0.0,
                "w_psd_t": float(self.dsb_w_psd_t.value()) if self.cb_w_psd_t.isChecked() else 0.0,
                "w_psd_spatial": float(self.dsb_w_psd_sp.value()) if self.cb_w_psd_sp.isChecked() else 0.0,
                "w_chi_inv": float(self.dsb_w_chi_inv.value()) if self.cb_w_chi_inv.isChecked() else 0.0,
                "w_avalanche": float(self.dsb_w_avalanche.value()) if self.cb_w_avalanche.isChecked() else 0.0,
                "w_autocorr": float(self.dsb_w_autocorr.value()) if self.cb_w_autocorr.isChecked() else 0.0,
                "w_jacobian": float(self.dsb_w_jacobian.value()) if self.cb_w_jacobian.isChecked() else 0.0,
                "w_lyapunov": float(self.dsb_w_lyapunov.value()) if self.cb_w_lyapunov.isChecked() else 0.0,
                "w_train_loss": float(self.dsb_w_train_loss.value()) if self.cb_w_train_loss.isChecked() else 0.0,
            },
            "targets": {
                "target_beta_t": float(self.dsb_target_beta_t.value()),
                "target_beta_spatial": float(self.dsb_target_beta_sp.value()),
            },
        }
        # Optuna
        sampler_name = self.cmb_sampler.currentText()
        skw: dict[str, Any] = {}
        if sampler_name == "tpe":
            skw = {
                "n_startup_trials": int(self.sb_tpe_startup.value()),
                "n_ei_candidates": int(self.sb_tpe_ei.value()),
                "multivariate": bool(self.cb_tpe_multivariate.isChecked()),
                "group": bool(self.cb_tpe_group.isChecked()),
                "constant_liar": bool(self.cb_tpe_liar.isChecked()),
            }
        elif sampler_name == "cma":
            skw = {
                "population_size": int(self.sb_cma_pop.value()) or None,
                "sigma0": float(self.dsb_cma_sigma0.value()),
                "restart_strategy": (self.cmb_cma_restart.currentText() if self.cmb_cma_restart.currentText() != "none" else None),
                "warn_independent_sampling": bool(self.cb_cma_warn.isChecked()),
            }
            if self.cb_cma_use_indep.isChecked():
                indep_name = self.cmb_indep_name.currentText()
                if indep_name == "tpe":
                    skw["independent_sampler"] = {
                        "name": "tpe",
                        "kwargs": {
                            "n_startup_trials": int(self.sb_itpe_startup.value()),
                            "n_ei_candidates": int(self.sb_itpe_ei.value()),
                            "multivariate": bool(self.cb_itpe_multivariate.isChecked()),
                            "group": bool(self.cb_itpe_group.isChecked()),
                            "constant_liar": bool(self.cb_itpe_liar.isChecked()),
                        },
                    }
                else:
                    skw["independent_sampler"] = "random"
        elif sampler_name == "nsgaii":
            skw = {"population_size": int(self.sb_nsga_pop.value())}
        elif sampler_name == "botorch":
            skw = {
                "n_startup_trials": int(self.sb_bo_startup.value()),
                "candidates_func": self.cmb_bo_candidates.currentText(),
            }
        optuna_block = {"sampler": sampler_name, "sampler_kwargs": {k: v for k, v in skw.items() if v is not None}}
        # Pruner
        pruner_block = {
            "use": bool(self.cb_prune_use.isChecked()),
            "n_startup_trials": int(self.sb_prune_startup.value()),
            "n_warmup_steps": int(self.sb_prune_warmup.value()),
        }
        # Optimization config from editors
        opt_cfg = {
            "enabled_components": {
                "layers": bool(self.cb_layers.isChecked()),
                "connections": bool(self.cb_conns.isChecked()),
                "weights": bool(self.cb_weights.isChecked()),
            },
            "target_layers": self._selected_layers(),
            "target_connections": self._selected_conns(),
            "layer_parameters": self._collect_layer_space(),
            "connection_parameters": self._collect_conn_space(),
            "weight_parameters": self._collect_weight_space(),
        }
        return {
            "paths": paths_block,
            "run": run_block,
            "simulation": sim_block,
            "input": input_block,
            "objective": objective_block,
            "optuna": optuna_block,
            "pruner": pruner_block,
            "optimization_config": opt_cfg,
        }

    def _on_export(self) -> None:
        # Save HPO config; when in project mode, prompt for a destination instead of overwriting
        if self._project_mode and self._hpo_yaml_path:
            try:
                cfg_out = self._collect_full_config()
                # If model path changed relative to last loaded, ensure spec is equivalent before saving
                try:
                    import yaml as _y

                    old_model_path = None
                    try:
                        with open(self._hpo_yaml_path) as _f:
                            _cfg_prev = _y.safe_load(_f) or {}
                            old_model_path = (_cfg_prev.get("paths") or {}).get("base_model_spec")
                    except Exception:
                        old_model_path = None
                    new_model_path = (cfg_out.get("paths") or {}).get("base_model_spec")
                    if old_model_path and new_model_path and os.path.abspath(str(old_model_path)) != os.path.abspath(str(new_model_path)):
                        # Load both specs to compare content (normalize via hpo_config.resolve_base_model_spec logic)
                        try:

                            def _load_spec_or_model(pth: str) -> dict:
                                if str(pth).lower().endswith((".soen", ".pth", ".pt", ".json")) and not str(pth).lower().endswith(("_spec.yaml", "_spec_for_hpo.yaml")):
                                    from dataclasses import asdict as _asdict

                                    from soen_toolkit.core import SOENModelCore as _Core

                                    m = _Core.load(pth, device=None, strict=False, verbose=False, show_logs=False)
                                    return {
                                        "simulation": _asdict(m.sim_config),
                                        "layers": [_asdict(cfg) for cfg in m.layers_config],
                                        "connections": [_asdict(cfg) for cfg in m.connections_config],
                                    }
                                with open(pth) as _ff:
                                    return _y.safe_load(_ff) or {}

                            a = _load_spec_or_model(old_model_path)
                            b = _load_spec_or_model(new_model_path)

                            # Consider only structural parts for equality
                            def _struct_only(x: dict) -> dict:
                                return {
                                    "simulation": x.get("simulation") or x.get("sim_config") or {},
                                    "layers": x.get("layers") or [],
                                    "connections": x.get("connections") or [],
                                }

                            if _struct_only(a) != _struct_only(b):
                                msg = (
                                    "The selected model/spec appears different from the one this project was created with. "
                                    "Please create a new project or confirm you wish to proceed by saving to a new file."
                                )
                                QMessageBox.critical(self, "Model changed", msg)
                                return
                        except Exception:
                            # If comparison fails, be conservative and stop
                            QMessageBox.critical(self, "Model check failed", "Could not verify that the new model/spec matches the original. Save As to a new file to proceed.")
                            return
                except Exception:
                    pass
                # Suggest current path as default
                proposed = self._hpo_yaml_path
                save_path, _ = QFileDialog.getSaveFileName(self, "Save HPO config as…", proposed, "YAML (*.yaml *.yml)")
                if not save_path:
                    return
                # Confirm overwrite if file exists and differs
                if os.path.exists(save_path) and os.path.abspath(save_path) != os.path.abspath(self._hpo_yaml_path):
                    resp = QMessageBox.question(
                        self,
                        "Overwrite file?",
                        f"File already exists:\n{save_path}\n\nOverwrite?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if resp != QMessageBox.StandardButton.Yes:
                        return
                _save_hpo_cfg_util(cfg_out, save_path)
                self._hpo_yaml_path = save_path
                QMessageBox.information(self, "Saved", f"Wrote: {save_path}")
                self.statusBar().showMessage(f"HPO config saved: {save_path}")
                return
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save YAML: {e}")
                return
        # Legacy path: building from model to create a new HPO YAML
        if self._schema is None or self._model_path is None:
            QMessageBox.warning(self, "Nothing to save", "Load an HPO YAML or use 'New From Model…'")
            return
        # Prepare a model spec path for skeleton building (supports trained models)
        model_for_paths = self._base_model_path or self._model_path
        # If model is a trained artifact, extract a spec first
        try:
            if isinstance(model_for_paths, str) and model_for_paths.lower().endswith((".soen", ".pth", ".pt", ".json")):
                out_dir_hint = self.output_dir_le.text().strip() or str(Path.cwd())
                model_for_paths = extract_spec_from_model(model_for_paths, out_dir_hint)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare model spec: {e}")
            return

        # Build skeleton from the spec; then patch targets + enabled components
        try:
            opt_cfg = build_hpo_skeleton(model_for_paths, base_model_path=self._base_model_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to build skeleton: {e}")
            return

        # Patch enabled components
        opt_cfg.setdefault("enabled_components", {})
        opt_cfg["enabled_components"]["layers"] = bool(self.cb_layers.isChecked())
        opt_cfg["enabled_components"]["connections"] = bool(self.cb_conns.isChecked())
        opt_cfg["enabled_components"]["weights"] = bool(self.cb_weights.isChecked())

        # Patch targets
        opt_cfg["target_layers"] = self._selected_layers() or opt_cfg.get("target_layers", [])
        opt_cfg["target_connections"] = self._selected_conns() or opt_cfg.get("target_connections", [])

        # Replace skeleton spaces from visual editors
        opt_cfg["layer_parameters"] = self._collect_layer_space()
        opt_cfg["connection_parameters"] = self._collect_conn_space()
        opt_cfg["weight_parameters"] = self._collect_weight_space()

        # Paths block uses the prepared spec path
        paths_block = {
            "base_model_spec": model_for_paths,
            "output_dir": (self.output_dir_le.text().strip() or str(Path(model_for_paths).parent)),
        }

        run_block = {
            "n_trials": int(self.sb_trials.value()),
            "timeout": (int(self.sb_timeout.value()) or None),
            "n_jobs": int(self.sb_jobs.value()),
            "seed": int(self.sb_seed.value()),
            "study_name": (self.le_study_name.text().strip() or None),
            "profile": bool(self.cb_profile.isChecked()),
            "resume": bool(self.cb_resume.isChecked()),
        }

        sim_block = {
            "seq_len": int(self.sb_seq_len.value()),
            "batch_size": int(self.sb_batch.value()),
        }
        if float(self.dsb_dt.value()) > 0:
            sim_block["dt"] = float(self.dsb_dt.value())

        # Input block
        params: dict[str, Any] = {}
        kind = self.cmb_input_kind.currentText()
        if kind == "white_noise":
            params["delta_n"] = float(self.dsb_delta_n.value())
        elif kind == "colored_noise":
            params["beta"] = float(self.dsb_beta.value())
        elif kind == "gp_rbf":
            params["sigma"] = float(self.dsb_sigma.value())
            params["ell_ns"] = float(self.dsb_ell_ns.value())
        elif kind == "log_slope_noise":
            params["slope_db_per_dec"] = float(self.dsb_slope_db.value())
            params["fmin_frac"] = float(self.dsb_fmin_frac.value())
            params["fmax_frac"] = float(self.dsb_fmax_frac.value())
        elif kind == "hdf5_dataset":
            params["path"] = self.le_h5_path.text().strip()
            if self.le_h5_split.text().strip():
                params["split"] = self.le_h5_split.text().strip()
            if self.le_h5_data_key.text().strip() and self.le_h5_data_key.text().strip() != "data":
                params["data_key"] = self.le_h5_data_key.text().strip()
        # Optional scaling for any kind
        if self.cb_scale.isChecked():
            params["scale_min"] = float(self.dsb_scale_min.value())
            params["scale_max"] = float(self.dsb_scale_max.value())
        input_block = {"kind": kind, "params": params}

        # Objective
        objective_block = {
            "weights": {
                "w_branch": float(self.dsb_w_branch.value()) if self.cb_w_branch.isChecked() else 0.0,
                "w_psd_t": float(self.dsb_w_psd_t.value()) if self.cb_w_psd_t.isChecked() else 0.0,
                "w_psd_spatial": float(self.dsb_w_psd_sp.value()) if self.cb_w_psd_sp.isChecked() else 0.0,
                "w_chi_inv": float(self.dsb_w_chi_inv.value()) if self.cb_w_chi_inv.isChecked() else 0.0,
                "w_avalanche": float(self.dsb_w_avalanche.value()) if self.cb_w_avalanche.isChecked() else 0.0,
                "w_autocorr": float(self.dsb_w_autocorr.value()) if self.cb_w_autocorr.isChecked() else 0.0,
                "w_jacobian": float(self.dsb_w_jacobian.value()) if self.cb_w_jacobian.isChecked() else 0.0,
                "w_lyapunov": float(self.dsb_w_lyapunov.value()) if self.cb_w_lyapunov.isChecked() else 0.0,
                "w_train_loss": float(self.dsb_w_train_loss.value()) if self.cb_w_train_loss.isChecked() else 0.0,
            },
            "targets": {
                "target_beta_t": float(self.dsb_target_beta_t.value()),
                "target_beta_spatial": float(self.dsb_target_beta_sp.value()),
            },
        }

        # Optuna (structured)
        sampler_name = self.cmb_sampler.currentText()
        skw: dict[str, Any] = {}
        if sampler_name == "tpe":
            skw = {
                "n_startup_trials": int(self.sb_tpe_startup.value()),
                "n_ei_candidates": int(self.sb_tpe_ei.value()),
                "multivariate": bool(self.cb_tpe_multivariate.isChecked()),
                "group": bool(self.cb_tpe_group.isChecked()),
                "constant_liar": bool(self.cb_tpe_liar.isChecked()),
            }
        elif sampler_name == "cma":
            skw = {
                "population_size": int(self.sb_cma_pop.value()) or None,
                "sigma0": float(self.dsb_cma_sigma0.value()),
                "restart_strategy": (self.cmb_cma_restart.currentText() if self.cmb_cma_restart.currentText() != "none" else None),
                "warn_independent_sampling": bool(self.cb_cma_warn.isChecked()),
            }
            if self.cb_cma_use_indep.isChecked():
                indep_name = self.cmb_indep_name.currentText()
                if indep_name == "tpe":
                    skw["independent_sampler"] = {
                        "name": "tpe",
                        "kwargs": {
                            "n_startup_trials": int(self.sb_itpe_startup.value()),
                            "n_ei_candidates": int(self.sb_itpe_ei.value()),
                            "multivariate": bool(self.cb_itpe_multivariate.isChecked()),
                            "group": bool(self.cb_itpe_group.isChecked()),
                            "constant_liar": bool(self.cb_itpe_liar.isChecked()),
                        },
                    }
                else:
                    skw["independent_sampler"] = "random"
        elif sampler_name == "nsgaii":
            skw = {"population_size": int(self.sb_nsga_pop.value())}
        elif sampler_name == "botorch":
            skw = {
                "n_startup_trials": int(self.sb_bo_startup.value()),
                "candidates_func": self.cmb_bo_candidates.currentText(),
            }
        optuna_block = {"sampler": sampler_name, "sampler_kwargs": {k: v for k, v in skw.items() if v is not None}}

        pruner_block = {
            "use": bool(self.cb_prune_use.isChecked()),
            "n_startup_trials": int(self.sb_prune_startup.value()),
            "n_warmup_steps": int(self.sb_prune_warmup.value()),
        }

        cfg_out = {
            "paths": paths_block,
            "run": run_block,
            "simulation": sim_block,
            "input": input_block,
            "objective": objective_block,
            "optuna": optuna_block,
            "pruner": pruner_block,
            "optimization_config": opt_cfg,
        }

        # Determine path: always prompt user and suggest a safe default
        def _suggest_path() -> str:
            try:
                if self._template_hpo_yaml_path:
                    base_dir = os.path.dirname(self._template_hpo_yaml_path)
                    base_name = os.path.basename(self._template_hpo_yaml_path)
                    name, ext = os.path.splitext(base_name)
                    if not ext:
                        ext = ".yaml"
                    return os.path.join(base_dir, f"{name}_edited{ext}")
                if self._hpo_yaml_path:
                    return self._hpo_yaml_path
                base_dir = self.output_dir_le.text().strip() or str(Path.cwd())
                return os.path.join(base_dir, "HPO_config.yaml")
            except Exception:
                return str(Path.cwd() / "HPO_config.yaml")

        proposed = _suggest_path()
        save_path, _ = QFileDialog.getSaveFileName(self, "Save HPO config as…", proposed, "YAML (*.yaml *.yml)")
        if not save_path:
            return
        # Confirm overwrite if file exists
        if os.path.exists(save_path):
            resp = QMessageBox.question(
                self, "Overwrite file?", f"File already exists:\n{save_path}\n\nOverwrite?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
        try:
            _save_hpo_cfg_util(cfg_out, save_path)
            self._hpo_yaml_path = save_path
            self._project_mode = True
            self._apply_mode()
            QMessageBox.information(self, "Saved", f"Wrote: {save_path}")
            self.statusBar().showMessage(f"HPO config saved: {save_path}")
            # Enable start
            self.btn_start.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save YAML: {e}")

        # Pre-compute study dir hint
        study_name = self.le_study_name.text().strip() or None
        out_dir = paths_block["output_dir"]
        if study_name:
            sd = os.path.join(out_dir, f"optuna_report_{study_name}")
            self.le_study_dir.setText(sd)

    # ————— Run management —————
    def _on_start(self) -> None:
        if self._running:
            return
        if not self._project_mode or not self._hpo_yaml_path or not os.path.exists(self._hpo_yaml_path):
            QMessageBox.warning(self, "Missing HPO YAML", "Open an HPO config, or use 'New From Model…' then Save.")
            return
        # Reset run-derived paths/names
        self._last_dashboard_path = None
        self._last_study_name = None
        # Determine target progress count
        try:
            with open(self._hpo_yaml_path) as f:
                cfg = yaml.safe_load(f) or {}
            n_trials = int(((cfg.get("run") or {}).get("n_trials")) or 0)
        except Exception:
            n_trials = 0
        self._progress_target = n_trials
        self._progress_completed = 0
        if self._progress_target > 0:
            self.progress.setRange(0, self._progress_target)
            self.progress.setValue(0)
            self.progress.setFormat("%p% (%v/%m)")
        else:
            # Indeterminate/busy indicator
            self.progress.setRange(0, 0)
            self.progress.setFormat("Running…")
        # Start QProcess
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_proc_output)
        self._proc.finished.connect(self._on_proc_finished)
        python = sys.executable
        script = str(Path(__file__).resolve().parents[1] / "scripts/run_hpo.py")
        # Run python unbuffered so stdout/stderr flush promptly (progress updates)
        args = ["-m", "soen_toolkit.utils.hpo.scripts.run_hpo", "--hp-config", self._hpo_yaml_path]
        # Pass common overrides for convenience
        try:
            n_trials = int(self.sb_trials.value())
            if n_trials > 0:
                args += ["--trials", str(n_trials)]
        except Exception:
            pass
        try:
            n_jobs = int(self.sb_jobs.value())
            if n_jobs > 0:
                args += ["--n-jobs", str(n_jobs)]
        except Exception:
            pass
        try:
            timeout = int(self.sb_timeout.value())
            if timeout > 0:
                args += ["--timeout", str(timeout)]
        except Exception:
            pass
        sname = self.le_study_name.text().strip()
        if not sname:
            try:
                import time as _t

                sname = f"criticality_{int(_t.time())}"
                self.le_study_name.setText(sname)
            except Exception:
                sname = "criticality_run"
        if sname:
            args += ["--study-name", sname]
        # Pre-fill study dir hint for live loader
        try:
            out_dir_txt = self.output_dir_le.text().strip()
            if out_dir_txt:
                sd = os.path.join(out_dir_txt, f"optuna_report_{sname}")
                self.le_study_dir.setText(sd)
        except Exception:
            pass
        if self.cb_resume.isChecked():
            args += ["--resume"]
        self.log.clear()
        self._append_log(f"Launching: {python} {script} --hp-config {self._hpo_yaml_path}\n")
        self._proc.start(python, args)
        self._running = True
        self.btn_stop.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.statusBar().showMessage("Optimization running…")
        # Start live results refresh
        with contextlib.suppress(Exception):
            self._results_timer.start()

    # ————— Extract trial management —————
    def _on_extract_trial(self) -> None:
        if not self._hpo_yaml_path or not os.path.exists(self._hpo_yaml_path):
            QMessageBox.warning(self, "Missing HPO YAML", "Load or save an HPO YAML first")
            return
        if self._running:
            QMessageBox.information(self, "Busy", "Stop or wait for the current run before extracting a trial spec.")
            return
        trial_num = int(self.sb_extract_trial.value())
        if trial_num < 0:
            QMessageBox.warning(self, "Invalid trial", "Enter a non-negative trial number")
            return
        out_path = self.le_extract_output.text().strip() or None
        python = sys.executable
        script = str(Path(__file__).resolve().parents[1] / "scripts/run_hpo.py")
        args = ["-m", "soen_toolkit.utils.hpo.scripts.run_hpo", "--hp-config", self._hpo_yaml_path, "--extract-trial-num", str(trial_num)]
        sname = self.le_study_name.text().strip()
        if sname:
            args += ["--study-name", sname]
        if out_path:
            args += ["--extract-output", out_path]
        # Run extraction in a short-lived process
        self._proc_extract = QProcess(self)
        self._proc_extract.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc_extract.readyReadStandardOutput.connect(self._on_proc_extract_output)
        self._proc_extract.finished.connect(self._on_proc_extract_finished)
        self._append_log(f"Launching extract: {python} {script} --hp-config {self._hpo_yaml_path} --extract-trial-num {trial_num}\n")
        self._proc_extract.start(python, args)

    def _on_proc_extract_output(self) -> None:
        try:
            data = bytes(self._proc_extract.readAllStandardOutput()).decode("utf-8", errors="ignore")
        except Exception:
            return
        if data:
            self._append_log(data)

    def _on_proc_extract_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if code == 0:
            QMessageBox.information(self, "Extract", "Trial spec extraction completed.")
        else:
            QMessageBox.warning(self, "Extract", f"Extraction finished with exit code {code}")

    def _on_stop(self) -> None:
        if self._proc and self._running:
            self._proc.kill()
        self._running = False
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.statusBar().showMessage("Stopped.")
        with contextlib.suppress(Exception):
            self._results_timer.stop()

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text.rstrip("\n"))
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _on_proc_output(self) -> None:
        if not self._proc:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if not data:
            return
        self._append_log(data)
        # Parse simple progress lines like: "Trial 20: Best score = ..."
        for line in data.splitlines():
            if line.strip().startswith("Trial "):
                try:
                    n = int(line.strip().split()[1].strip(":"))
                    self._progress_completed = max(self._progress_completed, n)
                    if self._progress_target > 0:
                        self.progress.setRange(0, self._progress_target)
                        self.progress.setValue(self._progress_completed)
                except Exception:
                    pass
            # Catch study rename when duplicate exists
            s = line.strip()
            if "already exists; using '" in s:
                try:
                    start = s.index("using '") + len("using '")
                    end = s.index("'", start)
                    new_name = s[start:end]
                    if new_name:
                        self._last_study_name = new_name
                        self.le_study_name.setText(new_name)
                        # Update derived study dir hint
                        out_dir = self.output_dir_le.text().strip()
                        if out_dir:
                            self.le_study_dir.setText(os.path.join(out_dir, f"optuna_report_{new_name}"))
                except Exception:
                    pass
            # Capture generated dashboard path from runner output
            if "Criticality dashboard generated:" in s or "Dashboard saved to:" in s or "Dashboard generated successfully:" in s:
                try:
                    if ":" in s:
                        p = s.split(":", 1)[1].strip()
                        # Some lines may prefix with emojis; the path should be the tail token
                        p = p.strip()
                        if p:
                            self._last_dashboard_path = p
                            with contextlib.suppress(Exception):
                                self.le_study_dir.setText(str(Path(p).resolve().parent))
                            # Try to infer study name from directory
                            try:
                                parent = Path(p).resolve().parent.name
                                if parent.startswith("optuna_report_"):
                                    inferred = parent.replace("optuna_report_", "", 1)
                                    self._last_study_name = inferred
                                    self.le_study_name.setText(inferred)
                            except Exception:
                                pass
                except Exception:
                    pass

    def _on_proc_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self._running = False
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.statusBar().showMessage(f"Finished with code {code}")
        try:
            if self._progress_target > 0:
                self.progress.setRange(0, self._progress_target)
                self.progress.setValue(self._progress_target)
            else:
                self.progress.setRange(0, 100)
                self.progress.setValue(100)
                self.progress.setFormat("100%")
        except Exception:
            pass
        # Final structured results refresh
        try:
            sd = self._current_study_dir_hint()
            if sd and os.path.isdir(sd):
                from soen_toolkit.utils.hpo.tools.criticality_data_processor import (
                    load_study_from_directory as _load,
                )

                study, cfg = _load(sd)
                from soen_toolkit.utils.hpo.tools.criticality_data_processor import (
                    CriticalityDataProcessor as _Proc,
                )

                data = _Proc(study, cfg or {}).process_all()
                # Update analysis and live plot
                self.results_view.apply_processed(sd, data)
                try:
                    df = data.get("trials_df")
                    ss = data.get("study_summary")
                    if df is not None and hasattr(self, "plot_live") and self.plot_live is not None:
                        self._update_live_progress_plot(self.plot_live, df, ss)
                except Exception:
                    pass
        except Exception:
            pass
        with contextlib.suppress(Exception):
            self._results_timer.stop()

    # HTML dashboard actions removed

    # ————— HPO YAML load —————
    def _on_load_hpo_yaml(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Open HPO YAML", str(Path.cwd()), "YAML (*.yaml *.yml);;All files (*)")
        if not p:
            return
        # Use robust loader: resolves base_model_spec, extracts from trained models if needed,
        # and backfills optimization_config targets/spaces.
        try:
            cfg_all = _load_hpo_cfg_util(p, allow_extract=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read YAML: {e}")
            return

        self._hpo_yaml_path = p
        self._project_mode = True
        self._apply_mode()
        self._template_hpo_yaml_path = p  # remember original to avoid accidental overwrite
        # Paths (possibly resolved/updated by loader)
        paths = cfg_all.get("paths") or {}
        base_model = str(paths.get("base_model_spec", ""))
        out_dir = str(paths.get("output_dir", ""))
        train_cfg = str(paths.get("train_config", ""))
        self.model_le.setText(base_model)
        self.base_model_le.setText("")
        self.output_dir_le.setText(out_dir)
        self.train_config_le.setText(train_cfg)
        self._model_path = base_model
        self._base_model_path = None
        self._output_dir = out_dir

        # Run
        run = cfg_all.get("run") or {}
        if "hpo_mode" in run:
            self.cmb_hpo_mode.setCurrentText(str(run.get("hpo_mode")))
        if "n_trials" in run and run.get("n_trials") is not None:
            self.sb_trials.setValue(int(run.get("n_trials")))
        if "timeout" in run and run.get("timeout") is not None:
            self.sb_timeout.setValue(int(run.get("timeout") or 0))
        if "n_jobs" in run and run.get("n_jobs") is not None:
            self.sb_jobs.setValue(int(run.get("n_jobs")))
        if "seed" in run and run.get("seed") is not None:
            self.sb_seed.setValue(int(run.get("seed")))
        self.le_study_name.setText(str(run.get("study_name") or ""))
        self.cb_profile.setChecked(bool(run.get("profile", False)))
        self.cb_resume.setChecked(bool(run.get("resume", False)))
        # Simulation
        sim = cfg_all.get("simulation") or {}
        if "seq_len" in sim:
            self.sb_seq_len.setValue(int(sim.get("seq_len", self.sb_seq_len.value())))
        if "batch_size" in sim:
            self.sb_batch.setValue(int(sim.get("batch_size", self.sb_batch.value())))
        if "dt" in sim and sim.get("dt") is not None:
            with contextlib.suppress(Exception):
                self.dsb_dt.setValue(float(sim.get("dt")))
        else:
            self.dsb_dt.setValue(0.0)
        # Input
        inp = cfg_all.get("input") or {}
        kind = str(inp.get("kind", self.cmb_input_kind.currentText()))
        idx = self.cmb_input_kind.findText(kind)
        if idx >= 0:
            self.cmb_input_kind.setCurrentIndex(idx)
        params = inp.get("params") or {}
        if "delta_n" in params:
            self.dsb_delta_n.setValue(float(params.get("delta_n")))
        if "beta" in params:
            self.dsb_beta.setValue(float(params.get("beta")))
        if "sigma" in params:
            self.dsb_sigma.setValue(float(params.get("sigma")))
        if "ell_ns" in params:
            self.dsb_ell_ns.setValue(float(params.get("ell_ns")))
        if "slope_db_per_dec" in params:
            self.dsb_slope_db.setValue(float(params.get("slope_db_per_dec")))
        if "fmin_frac" in params:
            self.dsb_fmin_frac.setValue(float(params.get("fmin_frac")))
        if "fmax_frac" in params:
            self.dsb_fmax_frac.setValue(float(params.get("fmax_frac")))
        # HDF5 params
        if "path" in params:
            self.le_h5_path.setText(str(params.get("path")))
        if "split" in params:
            self.le_h5_split.setText(str(params.get("split")))
        if "data_key" in params:
            self.le_h5_data_key.setText(str(params.get("data_key")))
        # Scaling params
        if "scale_min" in params and "scale_max" in params:
            try:
                self.cb_scale.setChecked(True)
                self.dsb_scale_min.setValue(float(params.get("scale_min")))
                self.dsb_scale_max.setValue(float(params.get("scale_max")))
            except Exception:
                pass
        # Objective
        obj = cfg_all.get("objective") or {}
        w = obj.get("weights") or {}

        # Apply toggle semantics based on weight values
        def _apply_toggle(spin, checkbox, val) -> None:
            try:
                v = float(val)
            except Exception:
                v = 0.0
            spin.setValue(v)
            on = v > 0.0
            checkbox.setChecked(on)
            spin.setEnabled(on)

        if "w_branch" in w:
            _apply_toggle(self.dsb_w_branch, self.cb_w_branch, w.get("w_branch"))
        if "w_psd_t" in w:
            _apply_toggle(self.dsb_w_psd_t, self.cb_w_psd_t, w.get("w_psd_t"))
            self.dsb_target_beta_t.setEnabled(float(w.get("w_psd_t", 0.0)) > 0.0)
        if "w_psd_spatial" in w:
            _apply_toggle(self.dsb_w_psd_sp, self.cb_w_psd_sp, w.get("w_psd_spatial"))
            self.dsb_target_beta_sp.setEnabled(float(w.get("w_psd_spatial", 0.0)) > 0.0)
        if "w_chi_inv" in w:
            _apply_toggle(self.dsb_w_chi_inv, self.cb_w_chi_inv, w.get("w_chi_inv"))
        if "w_avalanche" in w:
            _apply_toggle(self.dsb_w_avalanche, self.cb_w_avalanche, w.get("w_avalanche"))
        if "w_autocorr" in w:
            _apply_toggle(self.dsb_w_autocorr, self.cb_w_autocorr, w.get("w_autocorr"))
        if "w_jacobian" in w:
            _apply_toggle(self.dsb_w_jacobian, self.cb_w_jacobian, w.get("w_jacobian"))
        if "w_lyapunov" in w:
            _apply_toggle(self.dsb_w_lyapunov, self.cb_w_lyapunov, w.get("w_lyapunov"))
        t = obj.get("targets") or {}
        if "target_beta_t" in t:
            self.dsb_target_beta_t.setValue(float(t.get("target_beta_t")))
        if "target_beta_spatial" in t:
            self.dsb_target_beta_sp.setValue(float(t.get("target_beta_spatial")))
        # Optuna
        opt = cfg_all.get("optuna") or {}
        sname = str(opt.get("sampler", self.cmb_sampler.currentText()))
        si = self.cmb_sampler.findText(sname)
        if si >= 0:
            self.cmb_sampler.setCurrentIndex(si)
        # Sampler kwargs → structured panels
        skw = opt.get("sampler_kwargs") or {}
        if sname == "tpe":
            self.sb_tpe_startup.setValue(int(skw.get("n_startup_trials", self.sb_tpe_startup.value())))
            self.sb_tpe_ei.setValue(int(skw.get("n_ei_candidates", self.sb_tpe_ei.value())))
            self.cb_tpe_multivariate.setChecked(bool(skw.get("multivariate", False)))
            self.cb_tpe_group.setChecked(bool(skw.get("group", False)))
            self.cb_tpe_liar.setChecked(bool(skw.get("constant_liar", True)))
        elif sname == "cma":
            if "population_size" in skw and skw.get("population_size") is not None:
                self.sb_cma_pop.setValue(int(skw.get("population_size")))
            if "sigma0" in skw:
                self.dsb_cma_sigma0.setValue(float(skw.get("sigma0")))
            rs = str(skw.get("restart_strategy", "none"))
            idx = self.cmb_cma_restart.findText(rs)
            if idx < 0:
                idx = self.cmb_cma_restart.findText("none")
            self.cmb_cma_restart.setCurrentIndex(idx)
            self.cb_cma_warn.setChecked(bool(skw.get("warn_independent_sampling", False)))
            indep = skw.get("independent_sampler")
            if indep is not None:
                self.cb_cma_use_indep.setChecked(True)
                self.gb_indep.setVisible(True)
                if isinstance(indep, str):
                    self.cmb_indep_name.setCurrentIndex(self.cmb_indep_name.findText(indep))
                elif isinstance(indep, dict):
                    self.cmb_indep_name.setCurrentIndex(self.cmb_indep_name.findText(str(indep.get("name", "tpe"))))
                    kw = indep.get("kwargs") or {}
                    self.sb_itpe_startup.setValue(int(kw.get("n_startup_trials", self.sb_itpe_startup.value())))
                    self.sb_itpe_ei.setValue(int(kw.get("n_ei_candidates", self.sb_itpe_ei.value())))
                    self.cb_itpe_multivariate.setChecked(bool(kw.get("multivariate", False)))
                    self.cb_itpe_group.setChecked(bool(kw.get("group", False)))
                    self.cb_itpe_liar.setChecked(bool(kw.get("constant_liar", False)))
        elif sname == "nsgaii":
            if "population_size" in skw:
                self.sb_nsga_pop.setValue(int(skw.get("population_size")))
        elif sname == "botorch":
            self.sb_bo_startup.setValue(int(skw.get("n_startup_trials", self.sb_bo_startup.value())))
            idx = self.cmb_bo_candidates.findText(str(skw.get("candidates_func", self.cmb_bo_candidates.currentText())))
            if idx >= 0:
                self.cmb_bo_candidates.setCurrentIndex(idx)
        # Pruner
        pr = cfg_all.get("pruner") or {}
        self.cb_prune_use.setChecked(bool(pr.get("use", False)))
        if "n_startup_trials" in pr:
            self.sb_prune_startup.setValue(int(pr.get("n_startup_trials")))
        if "n_warmup_steps" in pr:
            self.sb_prune_warmup.setValue(int(pr.get("n_warmup_steps")))
        # Optimization config → enabled components + targets
        oc = cfg_all.get("optimization_config") or {}
        en = oc.get("enabled_components") or {}
        self.cb_layers.setChecked(bool(en.get("layers", True)))
        self.cb_conns.setChecked(bool(en.get("connections", True)))
        self.cb_weights.setChecked(bool(en.get("weights", True)))
        # set targets — if schema could not be built, ensure lists at least reflect YAML targets
        tlayers = [int(x) for x in (oc.get("target_layers") or [])]
        tconns = [str(x) for x in (oc.get("target_connections") or [])]
        if self.list_layers.count() == 0 and tlayers:
            # Minimal fallback representation
            for lid in tlayers:
                it = QListWidgetItem(f"Layer {int(lid)}")
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked)
                self.list_layers.addItem(it)
        elif self.list_layers.count() > 0 and tlayers:
            for i in range(self.list_layers.count()):
                it = self.list_layers.item(i)
                try:
                    lid = int(it.text().split()[1])
                except Exception:
                    continue
                it.setCheckState(Qt.CheckState.Checked if lid in set(tlayers) else Qt.CheckState.Unchecked)
        if self.list_conns.count() == 0 and tconns:
            for name in tconns:
                it = QListWidgetItem(name)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked)
                self.list_conns.addItem(it)
        elif self.list_conns.count() > 0 and tconns:
            for i in range(self.list_conns.count()):
                it = self.list_conns.item(i)
                it.setCheckState(Qt.CheckState.Checked if it.text() in set(tconns) else Qt.CheckState.Unchecked)

        # Populate new visual editors when present
        # Build editors if schema exists and not yet built
        if self._schema is None and self._model_path and os.path.exists(self._model_path):
            try:
                self._schema = build_option_schema(self._model_path)
                self._build_space_editors(self._schema)
            except Exception:
                pass
        if self._schema is not None:
            # layer_parameters
            if hasattr(self, "layer_param_editors") and isinstance(oc.get("layer_parameters"), dict):
                for pname, pconf in (oc.get("layer_parameters") or {}).items():
                    ed = self.layer_param_editors.get(str(pname))
                    if ed:
                        ed.load_from(pconf)
            # connection_parameters
            cp = oc.get("connection_parameters") or {}
            if self.conn_type_list is not None and isinstance(cp.get("connection_type"), dict):
                want_types = set(map(str, (cp.get("connection_type") or {}).get("choices", [])))
                for i in range(self.conn_type_list.list.count()):
                    it = self.conn_type_list.list.item(i)
                    it.setCheckState(Qt.CheckState.Checked if it.text() in want_types else Qt.CheckState.Unchecked)
            for pname, pconf in cp.items():
                if pname == "connection_type":
                    continue
                ed = self.conn_param_editors.get(str(pname))
                if ed:
                    ed.load_from(pconf)
            # weight parameters
            if self.weight_editor is not None and isinstance(oc.get("weight_parameters"), dict):
                self.weight_editor.load_from(oc.get("weight_parameters"))

        self.btn_export.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.statusBar().showMessage("HPO YAML loaded.")

    # ————— Pause/Resume —————
    def _on_pause(self) -> None:
        if not self._proc or not self._running:
            return
        try:
            pid = int(self._proc.processId())
            if os.name == "posix" and pid > 0:
                os.kill(pid, signal.SIGSTOP)
                self.btn_pause.setEnabled(False)
                self.btn_resume.setEnabled(True)
                self.statusBar().showMessage("Paused.")
            else:
                QMessageBox.information(self, "Pause", "Pause is only supported on POSIX systems.")
        except Exception as e:
            QMessageBox.warning(self, "Pause failed", f"Could not pause process: {e}")

    def _on_resume(self) -> None:
        if not self._proc or not self._running:
            return
        try:
            pid = int(self._proc.processId())
            if os.name == "posix" and pid > 0:
                os.kill(pid, signal.SIGCONT)
                self.btn_pause.setEnabled(True)
                self.btn_resume.setEnabled(False)
                self.statusBar().showMessage("Resumed.")
            else:
                QMessageBox.information(self, "Resume", "Resume is only supported on POSIX systems.")
        except Exception as e:
            QMessageBox.warning(self, "Resume failed", f"Could not resume process: {e}")

    # ————— Settings & Theme —————
    def _init_settings(self) -> None:
        # No persistent settings; just apply current theme selection (default light)
        with contextlib.suppress(Exception):
            self._apply_theme(self.cb_dark.isChecked())

    def _apply_theme(self, dark: bool) -> None:
        app = QApplication.instance()
        if not app:
            return
        # Use Fusion style for consistency across platforms (helps macOS)
        with contextlib.suppress(Exception):
            app.setStyle("Fusion")

        if dark:
            # Dark palette and stylesheet
            pal = QPalette()
            pal.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
            pal.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            pal.setColor(QPalette.ColorRole.Base, QColor(38, 38, 38))
            pal.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
            pal.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
            pal.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
            pal.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            pal.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            pal.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            pal.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
            pal.setColor(QPalette.ColorRole.Highlight, QColor(102, 126, 234))
            pal.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
            app.setPalette(pal)
            style_dark = """
            QMainWindow, QWidget#MainBg { background-color: #2b2b2b
            }
            /* Card-like section containers */
            QGroupBox { background-color: #2f2f2f
            border: 1px solid #3f3f3f
            border-radius: 8px
            margin-top: 16px
            color: #eee
            }
            QGroupBox::title { subcontrol-origin: margin
            left: 12px
            padding: 6px 10px
            border-radius: 6px
                               background: rgba(102,126,234,0.15)
                               color: #e8eaf6
                               border: 1px solid #4a4f8a
                               font-weight: 600
                               }
            QTabWidget::pane { border: 1px solid #444
            background: #2f2f2f
            border-radius: 6px
            }
            QTabBar::tab { background: #3a3a3a
            color:#eee
            border: 1px solid #444
            border-bottom-color: transparent
            padding: 6px 12px
            margin-right: 2px
            border-top-left-radius: 6px
            border-top-right-radius: 6px
            }
            QTabBar::tab:selected { background: #444
            border-color: #667eea
            }
            QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #3a3a3a
            color:#eee
            border: 1px solid #555
            border-radius: 4px
            padding: 4px
            }
            QListWidget { background: #2f2f2f
            color: #eee
            border: 1px solid #444
            border-radius: 4px
            }
            /* Checkbox + list indicators (dark) */
            QCheckBox::indicator, QListView::indicator { width: 14px
            height: 14px
            }
            QCheckBox::indicator:unchecked, QListView::indicator:unchecked { border: 1px solid #777
            background: #2f2f2f
            }
            QCheckBox::indicator:checked, QListView::indicator:checked { border: 1px solid #8899ff
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #667eea, stop:1 #764ba2)
            }
            QPushButton { background: #3a3a3a
            color:#eee
            border: 1px solid #555
            border-radius: 6px
            padding: 6px 12px
            }
            QPushButton:hover { border-color: #667eea
            }
            QProgressBar { border: 1px solid #555
            border-radius: 4px
            text-align: center
            background: #3a3a3a
            color:#eee
            }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #667eea, stop:1 #764ba2)
            }
            """
            app.setStyleSheet(style_dark)
        else:
            # Reset default palette and apply light gradient stylesheet
            app.setPalette(QPalette())
            style_light = """
            QMainWindow, QWidget#MainBg {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #f6f9ff, stop:1 #eef2f9);
            }
            /* Card-like section containers */
            QGroupBox { background-color: rgba(255,255,255,0.96)
            border: 1px solid #e5e9f5
            border-radius: 10px
            margin-top: 16px
            }
            QGroupBox::title { subcontrol-origin: margin
            left: 12px
            padding: 6px 10px
            border-radius: 8px
                               background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(102,126,234,0.10), stop:1 rgba(118,75,162,0.10));
                               color: #24324a
                               border: 1px solid #d0d7ee
                               font-weight: 600
                               }
            QTabWidget::pane { border: 1px solid #dfe3ee
            background: rgba(255,255,255,0.95)
            border-radius: 6px
            }
            QTabBar::tab { background: #ffffff
            border: 1px solid #dfe3ee
            border-bottom-color: transparent
            padding: 6px 12px
            margin-right: 2px
            border-top-left-radius: 6px
            border-top-right-radius: 6px
            }
            QTabBar::tab:selected { background: #ffffff
            border-color: #b6c2e0
            }
            QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #ffffff
            border: 1px solid #c9d3e8
            border-radius: 4px
            padding: 4px
            }
            QListWidget { background: rgba(255,255,255,0.95)
            border: 1px solid #dfe3ee
            border-radius: 4px
            }
            /* Checkbox + list indicators (light) */
            QCheckBox::indicator, QListView::indicator { width: 14px
            height: 14px
            }
            QCheckBox::indicator:unchecked, QListView::indicator:unchecked { border: 1px solid #b6c2e0
            background: #ffffff
            }
            QCheckBox::indicator:checked, QListView::indicator:checked { border: 1px solid #667eea
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #667eea, stop:1 #764ba2)
            }
            QPushButton { background: #ffffff
            border: 1px solid #c9d3e8
            border-radius: 6px
            padding: 6px 12px
            }
            QPushButton:hover { border-color: #8899cc
            }
            QProgressBar { border: 1px solid #c9d3e8
            border-radius: 4px
            text-align: center
            background: #ffffff
            }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #667eea, stop:1 #764ba2)
            }
            """
            app.setStyleSheet(style_light)

    def closeEvent(self, event) -> None:
        # No persistence on close per user request
        super().closeEvent(event)

    def _on_save_log(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "Save log", str(Path.cwd() / "hpo_run.log"), "Text (*.txt);;All files (*)")
        if not p:
            return
        try:
            with open(p, "w") as f:
                f.write(self.log.toPlainText())
            QMessageBox.information(self, "Saved", f"Log saved to: {p}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save log: {e}")

    def _on_new_from_model(self) -> None:
        if self._running:
            QMessageBox.information(self, "Busy", "Stop the current run before creating a new project.")
            return
        dlg = NewProjectDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        model_path = dlg.model_path()
        # Auto-detect output directory from model location
        out_dir = str(Path(model_path).parent)
        try:
            project = HPOProject.from_model(model_path, out_dir)
        except Exception as e:
            QMessageBox.critical(self, "New From Model", f"Failed to build project from model: {e}")
            return
        # Don't auto-save - just set up the project for the user to configure
        self._project_mode = False  # Allow editing before saving
        self._hpo_yaml_path = None  # Will be set when user saves
        self._apply_mode()

        # Use the project config directly (no YAML file yet)
        cfg_all = project.cfg

        # Populate UI from the project config
        try:
            paths = cfg_all.get("paths") or {}
            base_model = str(paths.get("base_model_spec", ""))
            out_dir = str(paths.get("output_dir", ""))
            self.model_le.setText(base_model)
            self.base_model_le.setText("")
            self.output_dir_le.setText(out_dir)
            # Store current path for subsequent saves
            self._model_path = base_model
            self._base_model_path = None
            self._output_dir = out_dir
            self._schema = None
            if base_model and os.path.exists(base_model):
                self._schema = build_option_schema(base_model)
                self._populate_from_schema(self._schema)
            oc = cfg_all.get("optimization_config") or {}
            en = oc.get("enabled_components") or {}
            self.cb_layers.setChecked(bool(en.get("layers", True)))
            self.cb_conns.setChecked(bool(en.get("connections", True)))
            self.cb_weights.setChecked(bool(en.get("weights", True)))
            tlayers = [int(x) for x in (oc.get("target_layers") or [])]
            tconns = [str(x) for x in (oc.get("target_connections") or [])]
            if self.list_layers.count() == 0 and tlayers:
                for lid in tlayers:
                    it = QListWidgetItem(f"Layer {int(lid)}")
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(Qt.CheckState.Checked)
                    self.list_layers.addItem(it)
            elif self.list_layers.count() > 0 and tlayers:
                for i in range(self.list_layers.count()):
                    it = self.list_layers.item(i)
                    try:
                        lid = int(it.text().split()[1])
                    except Exception:
                        continue
                    it.setCheckState(Qt.CheckState.Checked if lid in set(tlayers) else Qt.CheckState.Unchecked)
            if self.list_conns.count() == 0 and tconns:
                for name in tconns:
                    it = QListWidgetItem(name)
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(Qt.CheckState.Checked)
                    self.list_conns.addItem(it)
            elif self.list_conns.count() > 0 and tconns:
                for i in range(self.list_conns.count()):
                    it = self.list_conns.item(i)
                    it.setCheckState(Qt.CheckState.Checked if it.text() in set(tconns) else Qt.CheckState.Unchecked)
            self.btn_export.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.statusBar().showMessage("New HPO project created and loaded.")
        except Exception as e:
            QMessageBox.warning(self, "New From Model", f"Project created but UI population had issues: {e}")


def main():
    """Launch the main application."""
    app = QApplication(sys.argv)
    # ... existing setup code ...
    ex = HPOMinGui()
    ex.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
