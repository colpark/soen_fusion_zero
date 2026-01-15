"""Main window for Cloud Management GUI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .tabs.credentials_tab import CredentialsTab
from .tabs.jobs_tab import JobsTab
from .tabs.pricing_tab import PricingTab
from .tabs.s3_transfer_tab import S3TransferTab
from .tabs.submit_tab import SubmitTab

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CloudMainWindow(QMainWindow):
    """Main window for cloud management."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SOEN Cloud Manager")
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_status_bar()

        # Auto-refresh jobs every 30 seconds
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._on_refresh_jobs)
        self._refresh_timer.start(30000)

        # Check credentials on startup
        QTimer.singleShot(100, self._check_credentials)

    def _setup_ui(self) -> None:
        """Set up the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # Tab widget
        self.tabs = QTabWidget()

        # Create tabs
        self.credentials_tab = CredentialsTab()
        self.submit_tab = SubmitTab()
        self.jobs_tab = JobsTab()
        self.s3_transfer_tab = S3TransferTab()
        self.pricing_tab = PricingTab()

        # Connect signals
        self.credentials_tab.credentials_changed.connect(self._on_credentials_changed)
        self.submit_tab.job_submitted.connect(self._on_job_submitted)

        # Add tabs
        self.tabs.addTab(self.credentials_tab, "🔐 Credentials")
        self.tabs.addTab(self.submit_tab, "📤 Submit Job")
        self.tabs.addTab(self.jobs_tab, "📋 Jobs")
        self.tabs.addTab(self.s3_transfer_tab, "📦 S3 Transfer")
        self.tabs.addTab(self.pricing_tab, "💰 Pricing")

        layout.addWidget(self.tabs)

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        if menubar is None:
            return

        # File menu
        file_menu = menubar.addMenu("&File")
        if file_menu is not None:
            load_config = QAction("Load Cloud Config...", self)
            load_config.triggered.connect(self._on_load_config)
            file_menu.addAction(load_config)

            save_config = QAction("Save Cloud Config...", self)
            save_config.triggered.connect(self._on_save_config)
            file_menu.addAction(save_config)

            file_menu.addSeparator()

            exit_action = QAction("E&xit", self)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")
        if view_menu is not None:
            refresh_action = QAction("Refresh Jobs", self)
            refresh_action.setShortcut("F5")
            refresh_action.triggered.connect(self._on_refresh_jobs)
            view_menu.addAction(refresh_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        if help_menu is not None:
            docs_action = QAction("Documentation", self)
            docs_action.triggered.connect(self._on_show_docs)
            help_menu.addAction(docs_action)

            about_action = QAction("About", self)
            about_action.triggered.connect(self._on_about)
            help_menu.addAction(about_action)

    def _setup_toolbar(self) -> None:
        """Set up the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        refresh_action = QAction("🔄 Refresh", self)
        refresh_action.triggered.connect(self._on_refresh_jobs)
        toolbar.addAction(refresh_action)

        toolbar.addSeparator()

        submit_action = QAction("📤 Submit Job", self)
        submit_action.triggered.connect(lambda: self.tabs.setCurrentWidget(self.submit_tab))
        toolbar.addAction(submit_action)

    def _setup_status_bar(self) -> None:
        """Set up the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _check_credentials(self) -> None:
        """Check if credentials are configured."""
        role = os.environ.get("SOEN_SM_ROLE", "")
        bucket = os.environ.get("SOEN_SM_BUCKET", "")

        if not role or not bucket:
            self.status_bar.showMessage(
                "⚠️ AWS credentials not configured. Go to Credentials tab to set up."
            )
            self.tabs.setCurrentWidget(self.credentials_tab)
        else:
            self.status_bar.showMessage(f"✓ Connected to bucket: {bucket}")

    def _on_credentials_changed(self) -> None:
        """Handle credentials change."""
        self._check_credentials()
        self.jobs_tab.refresh()

    def _on_job_submitted(self, job_name: str) -> None:
        """Handle job submission."""
        self.status_bar.showMessage(f"✓ Job submitted: {job_name}")
        self.tabs.setCurrentWidget(self.jobs_tab)
        self.jobs_tab.refresh()

    def _on_refresh_jobs(self) -> None:
        """Refresh jobs list."""
        self.jobs_tab.refresh()
        self.status_bar.showMessage("Jobs refreshed", 3000)

    def _on_load_config(self) -> None:
        """Load cloud configuration from file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Cloud Configuration",
            str(Path.home()),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self.credentials_tab.load_from_file(path)
            self.status_bar.showMessage(f"Loaded config from: {path}")

    def _on_save_config(self) -> None:
        """Save cloud configuration to file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Cloud Configuration",
            str(Path.home() / "cloud_config.yaml"),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self.credentials_tab.save_to_file(path)
            self.status_bar.showMessage(f"Saved config to: {path}")

    def _on_show_docs(self) -> None:
        """Show documentation."""
        import webbrowser

        docs_path = Path(__file__).parent.parent.parent.parent / "docs" / "08_Cloud_Training.md"
        if docs_path.exists():
            webbrowser.open(f"file://{docs_path}")
        else:
            QMessageBox.information(
                self,
                "Documentation",
                "See docs/08_Cloud_Training.md for cloud training documentation.",
            )

    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About SOEN Cloud Manager",
            "SOEN Cloud Manager\n\n"
            "A graphical interface for managing cloud training jobs.\n\n"
            "Features:\n"
            "• Configure AWS credentials\n"
            "• Submit training, inference, and processing jobs\n"
            "• Monitor job status\n"
            "• View cost estimates\n\n"
            "Part of SOEN Toolkit",
        )

