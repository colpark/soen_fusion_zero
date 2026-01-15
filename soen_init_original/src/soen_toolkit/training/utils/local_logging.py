"""Training logger shim.

Delegates configuration to ``soen_toolkit.utils.logger.configure_logging``
to ensure a single, consistent logging setup across the project.
"""

import logging
from pathlib import Path

from soen_toolkit.utils.logger import configure_logging


def setup_logger(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    console: bool = True,
    console_level: int | None = None,
    file_mode: str = "w",
) -> logging.Logger:
    return configure_logging(
        log_file=log_file,
        level=level,
        console=console,
        console_level=console_level,
        file_mode=file_mode,
    )
