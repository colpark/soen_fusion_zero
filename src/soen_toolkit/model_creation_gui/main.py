# soen_toolkit/model_creation_gui/main.py
import logging
from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication

from soen_toolkit.model_creation_gui.components.splash_screen import SplashScreen
from soen_toolkit.model_creation_gui.main_window import MainWindow
from soen_toolkit.model_creation_gui.utils import theme

logging.basicConfig(level=logging.INFO)


def main() -> None:
    # Check if this is the first launch
    cache_dir = Path.home() / ".cache" / "soen-toolkit"
    flag_file = cache_dir / "gui_launched.flag"

    if not flag_file.exists():
        # Create cache directory and flag file
        cache_dir.mkdir(parents=True, exist_ok=True)
        flag_file.touch()

    app = QApplication(sys.argv)
    app.setOrganizationName("SOEN-TOOLKIT")
    app.setApplicationName("ModelCreationGUI")

    # Create and show splash screen IMMEDIATELY, before any heavy operations
    splash = SplashScreen()
    splash.set_progress(0, "Starting application...")
    splash.show()
    app.processEvents()  # Force initial render

    # Apply theme while splash is visible with progress tracking
    splash.set_progress(2, "Loading theme...")
    app.processEvents()
    theme.apply_initial_theme(app)
    app.processEvents()

    # Create main window with progress callback
    def on_progress(value: int, message: str = "") -> None:
        """Callback to update splash during main window initialization."""
        splash.set_progress(value, message)
        app.processEvents()

    try:
        win = MainWindow(progress_callback=on_progress)
        win.resize(1400, 860)

        # Fade out splash and show main window
        splash.fade_out(300)
        win.show()

    except Exception:
        logging.exception("Failed to create main window")
        splash.close()
        raise

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
