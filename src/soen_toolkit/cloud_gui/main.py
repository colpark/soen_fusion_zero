"""Main entry point for Cloud Management GUI."""

import logging
import sys

from PyQt6.QtWidgets import QApplication

from .main_window import CloudMainWindow

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Launch the Cloud Management GUI."""
    app = QApplication(sys.argv)
    app.setOrganizationName("SOEN-TOOLKIT")
    app.setApplicationName("CloudManagementGUI")

    # Apply a modern style
    app.setStyle("Fusion")

    window = CloudMainWindow()
    window.resize(1200, 800)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

