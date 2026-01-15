"""Gradient colorbar widget for visualization."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPen
from PyQt6.QtWidgets import QWidget


class GradientColorBar(QWidget):
    """Simple colorbar widget showing gradient scale."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(40)
        self._colormap = "RdBu_r"
        self._vmin = -1.0
        self._vmax = 1.0
        self._log_scale = False
        self._label = "Gradient"

    def update_range(
        self,
        vmin: float,
        vmax: float,
        colormap: str = "RdBu_r",
        log_scale: bool = False,
        label: str = "Gradient",
    ) -> None:
        """Update the colorbar range and appearance."""
        self._vmin = vmin
        self._vmax = vmax
        self._colormap = colormap
        self._log_scale = log_scale
        self._label = label
        self.update()

    def paintEvent(self, event) -> None:
        """Draw the colorbar."""
        from PyQt6.QtGui import QLinearGradient, QPainter

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate bar dimensions
        margin = 60
        bar_height = 15
        bar_y = 5
        bar_width = self.width() - 2 * margin

        if bar_width < 50:
            return

        # Get colormap colors
        try:
            cmap = plt.get_cmap(self._colormap)
        except Exception:
            cmap = plt.get_cmap("RdBu_r")

        # Create gradient
        gradient = QLinearGradient(margin, 0, margin + bar_width, 0)
        n_stops = 20
        for i in range(n_stops + 1):
            t = i / n_stops
            rgba = cmap(t)
            r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            gradient.setColorAt(t, QColor(r, g, b))

        # Draw bar
        painter.fillRect(margin, bar_y, bar_width, bar_height, gradient)
        painter.setPen(QPen(QColor("#666666"), 1))
        painter.drawRect(margin, bar_y, bar_width, bar_height)

        # Draw labels
        painter.setPen(QColor("#333333"))
        font = QFont("Arial", 8)
        painter.setFont(font)

        # Format values with appropriate precision
        def fmt_val(v: float) -> str:
            if abs(v) < 0.001:
                return f"{v:.2e}"
            elif abs(v) < 1:
                return f"{v:.3f}"
            elif abs(v) < 100:
                return f"{v:.2f}"
            else:
                return f"{v:.1e}"

        vmin_str = fmt_val(self._vmin)
        vmax_str = fmt_val(self._vmax)
        label_suffix = " (log)" if self._log_scale else ""

        # Draw min label (left)
        painter.drawText(margin, bar_y + bar_height + 12, vmin_str)
        # Draw max label (right)
        fm = QFontMetrics(font)
        max_width = fm.horizontalAdvance(vmax_str)
        painter.drawText(margin + bar_width - max_width, bar_y + bar_height + 12, vmax_str)
        # Draw center label
        center_label = f"{self._label}{label_suffix}"
        center_width = fm.horizontalAdvance(center_label)
        painter.drawText(
            margin + (bar_width - center_width) // 2, bar_y + bar_height + 12, center_label
        )

        painter.end()

