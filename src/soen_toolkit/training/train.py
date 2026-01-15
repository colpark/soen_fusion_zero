# FILEPATH: src/soen_toolkit/training/train.py

"""Simplified training interface for SOEN models.

This module provides a clean, simple interface for training SOEN models
without requiring complex path manipulation or environment setup.
"""

import logging
import os
from pathlib import Path
import sys


def setup_environment():
    """Set up the environment for training."""
    # Find project root by looking for pyproject.toml or .git
    current = Path(__file__).resolve()
    for _ in range(len(current.parts)):
        cand = current if current.is_dir() else current.parent
        if (cand / ".git").is_dir() or (cand / "pyproject.toml").is_file():
            os.environ["SOEN_PROJECT_ROOT"] = str(cand)
            return cand
        if cand.parent == cand:
            break
        current = cand.parent

    # Fallback: assume we're in src/soen_toolkit/training/train.py
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    os.environ["SOEN_PROJECT_ROOT"] = str(project_root)
    return project_root


def train_from_config(
    config_path: str | Path,
    log_level: str = "WARNING",
    log_dir: str | Path | None = None,
) -> None:
    """Train a SOEN model from a configuration file.

    Args:
        config_path: Path to the YAML configuration file
        log_level: Console logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (defaults to project_root/logs)

    Example:
        >>> from soen_toolkit.training.train import train_from_config
        >>> train_from_config("path/to/config.yaml", log_level="INFO")

    """
    # Set up environment
    project_root = setup_environment()

    # Import after environment setup
    from soen_toolkit.training.trainers.experiment import run_from_config

    # Validate config file
    config_file = Path(config_path)
    if not config_file.is_file():
        msg = f"Configuration file not found: {config_file}"
        raise FileNotFoundError(msg)

    # Set up logging - simple approach
    log_dir = project_root / "logs" if log_dir is None else Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    # Simple: add handlers to root logger
    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    file_handler = FlushingFileHandler(log_dir / "training.log", mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    console_handler = logging.StreamHandler()
    numeric_level = getattr(logging, log_level.upper(), logging.WARNING)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Using configuration file: {config_file}")
    logger.info(f"Project root: {project_root}")

    # Run training
    run_from_config(config_file, script_dir=project_root)


def main() -> None:
    """Command-line interface for train.py."""
    import argparse

    parser = argparse.ArgumentParser(description="Train a SOEN model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Console logging level (default: WARNING)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory for log files (default: project_root/logs)",
    )

    args = parser.parse_args()

    try:
        train_from_config(
            config_path=args.config,
            log_level=args.log_level,
            log_dir=args.log_dir,
        )
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
