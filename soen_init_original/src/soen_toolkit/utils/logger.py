# FILEPATH: soen/utils/logger.py


from datetime import datetime
import logging
import os
from pathlib import Path


class FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after each emit to prevent log corruption."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record and flush immediately."""
        super().emit(record)
        self.flush()


def setup_logger(name: str, log_level=logging.WARNING):
    """Set up and configure a logger instance.

    Parameters
    ----------
    - name (str): Name of the logger (typically __name__ from the calling module)
    - log_level: Logging level (default: logging.WARNING) The options are: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL (in that order)

    Returns
    -------
    - logging.Logger: Configured logger instance

    """
    # Create logger
    logger = logging.getLogger(name)

    # Set log level on each call so the caller's desired level is respected
    logger.setLevel(log_level)

    # Only add handlers if the logger doesn't already have them
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # Create handlers
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # File handler - create new log file for each day
        log_filename = os.path.join(logs_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        file_handler = FlushingFileHandler(log_filename)
        file_handler.setLevel(log_level)

        # Create formatters and add them to the handlers
        # Detailed format for file logs
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        )
        # Simpler format for console output
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
        )

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def configure_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    console: bool = True,
    console_level: int | None = None,
    file_mode: str = "w",
) -> logging.Logger:
    """Configure and return the root logger.

    This mirrors the training logger setup so all components share one
    consistent logging configuration.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pkg_logger = logging.getLogger("soen_toolkit")
    pkg_logger.setLevel(level)

    # Only remove managed handlers if we're replacing them
    # If log_file is None, keep existing file handlers
    if log_file is not None:
        root_managed_handlers = [h for h in logger.handlers if getattr(h, "_soen_managed", False) and isinstance(h, logging.FileHandler)]
        for handler in root_managed_handlers:
            logger.removeHandler(handler)
            handler.close()
    if console:
        root_managed_console = [h for h in logger.handlers if getattr(h, "_soen_managed", False) and isinstance(h, logging.StreamHandler)]
        for handler in root_managed_console:
            logger.removeHandler(handler)
            handler.close()

    new_file_handler: logging.Handler | None = None
    if log_file is not None:
        if isinstance(log_file, str):
            log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        new_file_handler = FlushingFileHandler(log_file, mode=file_mode)
        new_file_handler.setLevel(level)
        new_file_handler.setFormatter(formatter)
        new_file_handler._soen_managed = True
        logger.addHandler(new_file_handler)
        pkg_logger.addHandler(new_file_handler)

    new_console_handler: logging.Handler | None = None
    if console:
        new_console_handler = logging.StreamHandler()
        new_console_handler.setLevel(console_level if console_level is not None else level)
        new_console_handler.setFormatter(formatter)
        new_console_handler._soen_managed = True
        logger.addHandler(new_console_handler)
        pkg_logger.addHandler(new_console_handler)

    # Ensure child loggers propagate to root logger so all messages are captured
    # Don't disable propagation - we want all messages to reach handlers
    pkg_logger.propagate = True
    for logger_name, logger_obj in logging.Logger.manager.loggerDict.items():
        if isinstance(logger_obj, logging.Logger) and logger_name.startswith("soen_toolkit."):
            logger_obj.propagate = True
            logger_obj.setLevel(logging.NOTSET)  # Let parent handle level filtering

    return logger


def add_file_handler(log_file: Path | str, level: int = logging.INFO, file_mode: str = "w") -> None:
    """Add a file handler to root logger without removing existing handlers.

    Use this to add per-repeat experiment logs while keeping application logs active.
    """
    if isinstance(log_file, str):
        log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = FlushingFileHandler(log_file, mode=file_mode)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler._soen_managed = True
    root_logger = logging.getLogger()
    # Ensure root logger level is low enough to let messages through
    # CRITICAL=50, ERROR=40, WARNING=30, INFO=20, DEBUG=10
    if root_logger.level > level:
        root_logger.setLevel(level)
    root_logger.addHandler(handler)
    # Also ensure handler is actually attached and root level is correct
    root_logger.setLevel(logging.DEBUG)  # Force to lowest level so handler can filter

    # Ensure all soen_toolkit child loggers propagate and don't filter
    for logger_name, logger_obj in logging.Logger.manager.loggerDict.items():
        if isinstance(logger_obj, logging.Logger) and logger_name.startswith("soen_toolkit."):
            logger_obj.propagate = True
            logger_obj.setLevel(logging.NOTSET)  # Let parent handle level filtering

    # DEBUG: Test handler immediately
    root_logger.critical("HELLO FROM add_file_handler - HANDLER CREATED AND ATTACHED")
    handler.flush()
