import pathlib
import re

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

THEME_KEY = "theme"
BASIC_THEME_NAME = "basic"
DARK_THEME_NAME = "dark"
MIDNIGHT_THEME_NAME = "midnight"
ARCTIC_THEME_NAME = "arctic"
SUNSET_THEME_NAME = "sunset"
MINIMAL_THEME_NAME = "minimal"
GLASS_KEY = "glass_enabled"
FONT_SCALE_KEY = "font_scale"

# Default font scale (1.0 = 100%)
DEFAULT_FONT_SCALE = 1.0
# Reasonable bounds for font scaling
MIN_FONT_SCALE = 0.5
MAX_FONT_SCALE = 2.0


def _load_stylesheet(name: str) -> str:
    """Load a QSS file from the resources directory."""
    style_path = pathlib.Path(__file__).resolve().parent.parent / "resources" / f"{name}_theme.qss"
    if style_path.exists():
        return style_path.read_text()
    return ""


def _scale_font_sizes(css: str, scale: float) -> str:
    """Scale all font-size declarations in a stylesheet.

    Matches patterns like 'font-size: 14pt' or 'font-size:11pt' and scales the numeric value.
    """
    if scale == 1.0:
        return css

    def replace_size(match: re.Match) -> str:
        prefix = match.group(1)  # 'font-size:' or 'font-size: '
        size = float(match.group(2))
        unit = match.group(3)  # 'pt' or 'px'
        scaled_size = round(size * scale, 1)
        # Use integer if whole number for cleaner output
        if scaled_size == int(scaled_size):
            scaled_size = int(scaled_size)
        return f"{prefix}{scaled_size}{unit}"

    # Match font-size declarations with pt or px units
    pattern = r"(font-size:\s*)(\d+(?:\.\d+)?)(pt|px)"
    return re.sub(pattern, replace_size, css)


STYLES = {
    BASIC_THEME_NAME: _load_stylesheet(BASIC_THEME_NAME),
    DARK_THEME_NAME: _load_stylesheet(DARK_THEME_NAME),
    MIDNIGHT_THEME_NAME: _load_stylesheet(MIDNIGHT_THEME_NAME),
    ARCTIC_THEME_NAME: _load_stylesheet(ARCTIC_THEME_NAME),
    SUNSET_THEME_NAME: _load_stylesheet(SUNSET_THEME_NAME),
    MINIMAL_THEME_NAME: _load_stylesheet(MINIMAL_THEME_NAME),
}


def get_current_theme() -> str:
    """Get the saved theme from QSettings, defaulting to basic."""
    return QSettings().value(THEME_KEY, BASIC_THEME_NAME)


def set_current_theme(name: str) -> None:
    """Save the selected theme to QSettings."""
    QSettings().setValue(THEME_KEY, name)


def get_font_scale() -> float:
    """Get the saved font scale factor from QSettings."""
    value = QSettings().value(FONT_SCALE_KEY, DEFAULT_FONT_SCALE)
    try:
        scale = float(value)
        return max(MIN_FONT_SCALE, min(MAX_FONT_SCALE, scale))
    except (TypeError, ValueError):
        return DEFAULT_FONT_SCALE


def set_font_scale(scale: float) -> None:
    """Save the font scale factor to QSettings."""
    scale = max(MIN_FONT_SCALE, min(MAX_FONT_SCALE, scale))
    QSettings().setValue(FONT_SCALE_KEY, scale)


def apply_theme(app: QApplication, name: str) -> None:
    """Apply a theme to the application and persist the choice.

    This overrides system appearance. Use View → Theme in the app to switch.
    The font scale setting is automatically applied.
    """
    # Fallback to basic if stylesheet missing
    css = STYLES.get(name) or STYLES.get(BASIC_THEME_NAME, "")

    # Apply font scaling
    scale = get_font_scale()
    css = _scale_font_sizes(css, scale)

    app.setStyleSheet(css)
    # Force widget polish/unpolish to flush cached palette on some platforms
    try:
        for w in app.topLevelWidgets():
            w.setStyleSheet(w.styleSheet())
            w.update()
    except Exception:
        pass
    set_current_theme(name if name in STYLES else BASIC_THEME_NAME)


def apply_font_scale(app: QApplication, scale: float) -> None:
    """Apply a new font scale and persist the choice.

    Re-applies the current theme with the new scale.
    """
    set_font_scale(scale)
    apply_theme(app, get_current_theme())


def get_glass_enabled() -> bool:
    v = QSettings().value(GLASS_KEY, False)
    # QSettings may return string; normalize
    return bool(v or str(v).lower() in ("1", "true", "yes", "on"))


def set_glass_enabled(enabled: bool) -> None:
    QSettings().setValue(GLASS_KEY, bool(enabled))


def toggle_theme(app: QApplication) -> None:
    """Toggle between basic and dark themes."""
    current = get_current_theme()
    next_theme = DARK_THEME_NAME if current == BASIC_THEME_NAME else BASIC_THEME_NAME
    apply_theme(app, next_theme)


def apply_initial_theme(app: QApplication) -> None:
    """Apply the saved theme on application startup."""
    initial_theme = get_current_theme()
    apply_theme(app, initial_theme)
