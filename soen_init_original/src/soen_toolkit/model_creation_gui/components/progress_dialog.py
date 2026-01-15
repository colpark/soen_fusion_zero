# src/soen_toolkit/model_creation_gui/components/progress_dialog.py
"""Modal progress dialog for long-running operations."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QDialog, QLabel, QProgressBar, QTextEdit, QVBoxLayout


class OperationProgressDialog(QDialog):
    """Modal dialog showing progress for long-running operations.

    Provides real-time updates about what step is currently executing.
    Prevents user interaction until the operation completes.
    """

    def __init__(self, parent=None, title: str = "Processing..."):
        """Initialize progress dialog.

        Args:
            parent: Parent widget
            title: Dialog title
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(250)

        # Prevent close button during operation
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title label
        self.title_label = QLabel()
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(self.title_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: #ecf0f1;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Status log (shows steps)
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(120)
        self.status_log.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: monospace;
                font-size: 9pt;
            }
        """)
        layout.addWidget(self.status_log)

        # Current step label
        self.step_label = QLabel()
        step_font = QFont()
        step_font.setPointSize(10)
        self.step_label.setFont(step_font)
        self.step_label.setStyleSheet("color: #34495e; font-weight: bold;")
        layout.addWidget(self.step_label)

        self.steps_history = []

    def set_title(self, title: str) -> None:
        """Set dialog title."""
        self.title_label.setText(title)

    def set_progress(self, value: int, step: str = "") -> None:
        """Update progress bar and step information.

        Args:
            value: Progress percentage (0-100)
            step: Current step description
        """
        self.progress_bar.setValue(max(0, min(100, value)))

        if step:
            self.step_label.setText(f"Current: {step}")
            # Add to history if different from last
            if not self.steps_history or self.steps_history[-1] != step:
                self.steps_history.append(step)
                # Show last 10 steps in log
                visible_steps = self.steps_history[-10:]
                log_text = "\n".join(f"• {s}" for s in visible_steps)
                self.status_log.setText(log_text)
                # Auto-scroll to bottom
                scrollbar = self.status_log.verticalScrollBar()
                if scrollbar is not None:
                    scrollbar.setValue(scrollbar.maximum())

    def add_step(self, step: str) -> None:
        """Add a step to the log without changing progress.

        Args:
            step: Step description to add
        """
        if not self.steps_history or self.steps_history[-1] != step:
            self.steps_history.append(step)
            visible_steps = self.steps_history[-10:]
            log_text = "\n".join(f"• {s}" for s in visible_steps)
            self.status_log.setText(log_text)
            scrollbar = self.status_log.verticalScrollBar()
            if scrollbar is not None:
                scrollbar.setValue(scrollbar.maximum())

    def finish(self) -> None:
        """Mark operation as complete and close dialog."""
        self.progress_bar.setValue(100)
        self.step_label.setText("Complete!")
        # Use timer to show completion briefly before closing
        QTimer.singleShot(300, self.accept)
