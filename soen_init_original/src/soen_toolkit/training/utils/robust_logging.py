"""Robust logging setup that always writes to files and stdout.

This module provides a simple, foolproof logging system that:
1. Always writes logs to a file (in the repeat log directory)
2. Always prints to stdout/stderr for visibility
3. Flushes immediately after every log message
4. Works regardless of how the code is called
"""

import logging
from pathlib import Path
import sys


class AlwaysFlushFileHandler(logging.FileHandler):
    """File handler that flushes immediately after every emit."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()
        # Also force OS-level flush
        if hasattr(self.stream, "fileno"):
            try:
                import os

                os.fsync(self.stream.fileno())
            except (OSError, ValueError):
                pass  # Not all streams support fsync


class AlwaysFlushStreamHandler(logging.StreamHandler):  # type: ignore
    """Stream handler that flushes immediately after every emit."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_robust_logging(
    log_file: Path,
    log_level: str = "INFO",
    console: bool = True,
    console_level: str | None = None,
) -> logging.Logger:
    """Set up robust logging that always works.

    Args:
        log_file: Path to the log file (will be created if needed)
        log_level: Logging level for file (default: INFO)
        console: Whether to also log to console (default: True)
        console_level: Logging level for console (default: same as log_level)

    Returns:
        The root logger instance
    """
    # Ensure log directory exists
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # If we can't create directory, at least log to stderr
        print(f"ERROR: Could not create log directory {log_file.parent}: {e}", file=sys.stderr)  # noqa: T201
        sys.stderr.flush()

    # Get numeric log levels
    file_level = getattr(logging, log_level.upper(), logging.INFO)
    console_level_str = console_level or log_level
    console_level_num = getattr(logging, console_level_str.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Always accept all messages, handlers filter

    # Clear existing handlers to prevent duplication/conflict (crucial for empty logs fix)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File handler - always created (only if we can write to the file)
    file_handler = None
    try:
        file_handler = AlwaysFlushFileHandler(log_file, mode="a")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # If we can't create file handler, at least log to stderr
        print(f"ERROR: Could not create log file {log_file}: {e}", file=sys.stderr)  # noqa: T201
        sys.stderr.flush()

    # Console handler - optional
    if console:
        try:
            console_handler = AlwaysFlushStreamHandler(sys.stdout)
            console_handler.setLevel(console_level_num)
            console_handler.setFormatter(
                logging.Formatter(
                    "%(levelname)s - %(message)s"  # Simpler format for console
                )
            )
            root_logger.addHandler(console_handler)
        except Exception as e:
            print(f"ERROR: Could not create console handler: {e}", file=sys.stderr)  # noqa: T201
            sys.stderr.flush()

    # Also set up the soen_toolkit logger specifically
    pkg_logger = logging.getLogger("soen_toolkit")
    pkg_logger.setLevel(logging.DEBUG)
    # Handlers propagate from root, so we don't need to add them here

    # Print confirmation to both file and console
    try:
        root_logger.info(f"Logging initialized: file={log_file}, level={log_level}")
        if console:
            root_logger.info(f"Console logging: level={console_level_str}")
    except Exception:
        # If logging fails, at least print to stderr
        print(f"Logging initialized: file={log_file}, level={log_level}", file=sys.stderr)  # noqa: T201
        sys.stderr.flush()

    return root_logger


def print_and_log(message: str, level: str = "INFO", logger: logging.Logger | None = None) -> None:
    """Print message to both stdout and log file.

    This is a fallback that ensures messages are always visible.

    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, etc.)
        logger: Logger to use (default: root logger)
    """
    if logger is None:
        logger = logging.getLogger()

    # Always print to stdout/stderr
    print(message, file=sys.stdout if level.upper() in ("INFO", "DEBUG") else sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()

    # Also log via logger
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)

    # Force flush file handlers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            if hasattr(handler.stream, "fileno"):
                try:
                    import os

                    os.fsync(handler.stream.fileno())
                except (OSError, ValueError):
                    pass
