"""Unified interaction handling for visualization views.

Consolidates mouse event handling across different view types.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtWidgets import QGraphicsView

if TYPE_CHECKING:
    from PyQt6.QtCore import QObject
    from PyQt6.QtGui import QMouseEvent


class GraphicsViewInteractionHandler:
    """Handles mouse interactions for QGraphicsView-based views.

    Provides unified press/release/click handling for both hybrid and native views.
    """

    # Minimum movement threshold for distinguishing clicks from drags
    CLICK_THRESHOLD = 3

    def __init__(
        self,
        view: QGraphicsView,
        click_handler: Callable[[QPointF], None],
        interaction_mode_getter: Callable[[], str],
    ) -> None:
        """Initialize the interaction handler.

        Args:
            view: The QGraphicsView to handle interactions for
            click_handler: Function to call with scene coordinates on click
            interaction_mode_getter: Function that returns current mode ('select' or 'pan')
        """
        self._view = view
        self._click_handler = click_handler
        self._get_mode = interaction_mode_getter
        self._press_pos: QPointF | None = None

    def handle_press(self, event: QMouseEvent) -> bool:
        """Handle mouse button press.

        Args:
            event: The mouse press event

        Returns:
            True if event was handled, False otherwise
        """
        if event.button() != Qt.MouseButton.LeftButton:
            return False

        self._press_pos = event.position()

        if self._get_mode() == "pan":
            self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        return False  # Allow default handling

    def handle_release(self, event: QMouseEvent) -> bool:
        """Handle mouse button release.

        Args:
            event: The mouse release event

        Returns:
            True if event was handled, False otherwise
        """
        if event.button() != Qt.MouseButton.LeftButton:
            return False

        if self._get_mode() == "select":
            # Check if this was a click (minimal movement)
            if self._press_pos:
                delta = event.position() - self._press_pos
                if delta.manhattanLength() <= self.CLICK_THRESHOLD:
                    # Map to scene and handle click
                    scene_pt = self._view.mapToScene(
                        int(event.position().x()),
                        int(event.position().y()),
                    )
                    self._click_handler(scene_pt)
        else:
            # Pan mode - reset drag mode
            self._view.setDragMode(QGraphicsView.DragMode.NoDrag)

        return False  # Allow default handling

    def filter_event(self, source: QObject, event: QEvent) -> bool:
        """Process an event for this handler's view.

        Args:
            source: The event source
            event: The event to process

        Returns:
            True if event was consumed, False to pass through
        """
        if source is not self._view.viewport():
            return False

        if event.type() == QEvent.Type.MouseButtonPress:
            return self.handle_press(event)
        elif event.type() == QEvent.Type.MouseButtonRelease:
            return self.handle_release(event)

        return False


def is_click(
    press_pos: QPointF | None,
    release_pos: QPointF,
    threshold: int = 3,
) -> bool:
    """Determine if press/release constitutes a click.

    Args:
        press_pos: The position of the initial press
        release_pos: The position of the release
        threshold: Maximum movement in pixels

    Returns:
        True if this is a click (minimal movement)
    """
    if press_pos is None:
        return False
    delta = release_pos - press_pos
    return delta.manhattanLength() <= threshold

