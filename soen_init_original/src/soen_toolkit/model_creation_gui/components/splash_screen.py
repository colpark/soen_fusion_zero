# src/soen_toolkit/model_creation_gui/components/splash_screen.py
"""Professional splash screen with progress bar for GUI startup."""

from PyQt6.QtCore import QPropertyAnimation, QRect, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen


class SplashScreen(QSplashScreen):
    """Custom splash screen with progress bar and status messages."""

    def __init__(self, parent=None):
        """Initialize splash screen with a simple colored background."""
        # Create a minimal pixmap for the splash (will be drawn on)
        pixmap = QPixmap(500, 300)
        pixmap.fill(Qt.GlobalColor.transparent)
        super().__init__(pixmap)

        self.progress = 0
        self.message_text = "Initializing..."
        self.is_dark_theme = self._detect_theme()

        # Center on screen
        self.move(QApplication.primaryScreen().geometry().center() - self.rect().center())

    def _detect_theme(self) -> bool:
        """Detect if dark theme is active based on application palette."""
        try:
            app = QApplication.instance()
            if not isinstance(app, QApplication):
                return False
            palette = app.palette()
            bg_color = palette.color(palette.ColorRole.Window)
            # Simple heuristic: dark if luminance < 128
            return (0.299 * bg_color.red() + 0.587 * bg_color.green() + 0.114 * bg_color.blue()) < 128
        except Exception:
            return False

    def set_progress(self, value: int, message: str = "") -> None:
        """Update progress bar and optional status message.

        Args:
            value: Progress percentage (0-100)
            message: Optional status message to display
        """
        self.progress = max(0, min(100, value))
        if message:
            self.message_text = message
        self.update()  # Trigger repaint
        # Force UI update immediately
        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents()

    def paintEvent(self, event):
        """Paint the splash screen with progress bar and text."""
        painter = QPainter(self)

        # Background color based on theme
        if self.is_dark_theme:
            bg_color = QColor(45, 45, 48)
            text_color = QColor(240, 240, 240)
            progress_bg = QColor(60, 60, 65)
            progress_fg = QColor(88, 166, 255)  # Nice blue accent
        else:
            bg_color = QColor(245, 245, 248)
            text_color = QColor(30, 30, 30)
            progress_bg = QColor(230, 230, 235)
            progress_fg = QColor(0, 102, 204)  # Professional blue

        # Fill background
        painter.fillRect(self.rect(), bg_color)

        # Draw title
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(text_color)
        painter.drawText(
            self.rect().adjusted(20, 40, -20, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "SOEN Model Builder",
        )

        # Draw status message
        msg_font = QFont()
        msg_font.setPointSize(10)
        painter.setFont(msg_font)
        painter.setPen(QColor(150, 150, 150) if not self.is_dark_theme else QColor(170, 170, 170))
        painter.drawText(
            self.rect().adjusted(20, 120, -20, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            self.message_text,
        )

        # Draw progress bar background
        progress_rect = QRect(60, 200, 380, 16)
        painter.fillRect(progress_rect, progress_bg)

        # Draw progress bar fill
        if self.progress > 0:
            fill_width = int((self.progress / 100.0) * 380)
            painter.fillRect(
                QRect(60, 200, fill_width, 16),
                progress_fg,
            )

        # Draw progress bar border
        pen = QPen(text_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(progress_rect)

        # Draw percentage text
        pct_font = QFont()
        pct_font.setPointSize(9)
        painter.setFont(pct_font)
        painter.setPen(text_color)
        painter.drawText(
            self.rect().adjusted(20, 220, -20, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            f"{self.progress}%",
        )

        painter.end()

    def fade_out(self, duration_ms: int = 300) -> None:
        """Smoothly fade out the splash screen.

        Args:
            duration_ms: Duration of fade animation in milliseconds
        """
        self.anim = QPropertyAnimation(self, b"windowOpacity")
        self.anim.setDuration(duration_ms)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.finished.connect(self.close)
        self.anim.start()
