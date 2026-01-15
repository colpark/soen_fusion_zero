# FILEPATH: src/soen_toolkit/training/__main__.py

"""Main entry point for the SOEN training module.

This allows running training via:
    python -m soen_toolkit.training --config path/to/config.yaml

Or using the train subcommand:
    python -m soen_toolkit.training train --config path/to/config.yaml
"""

# Suppress SyntaxWarnings from AWS packages before they're imported.
# sagemaker_core and smdebug_rulesconfig have invalid escape sequences
# that AWS hasn't fixed for Python 3.12+ compatibility.
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"sagemaker_core\..*")
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"smdebug_rulesconfig\..*")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

import yaml  # noqa: E402


def setup_project_root():
    """Set up the project root environment variable."""
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

    # Fallback: assume we're in src/soen_toolkit/training/__main__.py
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    os.environ["SOEN_PROJECT_ROOT"] = str(project_root)
    return project_root


def _print_env_hint(prefix: str = "") -> None:
    """Print a concise, actionable environment activation hint."""
    with contextlib.suppress(Exception):
        os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")


def _warn_on_suspicious_env() -> None:
    """Warn when the active environment name looks wrong for this project."""
    try:
        env_name = os.environ.get("CONDA_DEFAULT_ENV")
        if env_name and env_name != "soen_env":
            pass
    except Exception:
        pass


def main() -> int | None:
    """Main entry point for training."""
    # Set up project root before any imports
    project_root = setup_project_root()

    # Quick environment sanity warning (non-fatal)
    _warn_on_suspicious_env()

    # Import training components
    from soen_toolkit.training.trainers.experiment import run_from_config

    parser = argparse.ArgumentParser(
        description="SOEN Toolkit Training",
        prog="soen_toolkit.training",
    )

    # Support both:
    # 1. train path/to/config.yml (simple positional - preferred)
    # 2. train --config path/to/config.yml (backward compatible)

    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--config",
        type=str,
        dest="config_flag",
        help="Path to the YAML configuration file (alternative flag form)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Console logging level (default: WARNING)",
    )

    args = parser.parse_args()

    # Determine config file - prefer positional, fallback to flag
    config_path = args.config or args.config_flag
    log_level = getattr(args, "log_level", "WARNING")

    if not config_path:
        parser.print_help()
        sys.exit(1)

    # Validate config file exists
    config_file = Path(config_path)
    if not config_file.is_file():
        sys.exit(1)

    # Set up logging - simple approach
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Simple: use basicConfig, but preserve existing handlers if any
    # Create file handler that flushes
    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    file_handler = FlushingFileHandler(log_dir / "training.log", mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Using configuration file: {config_file}")
    logger.info(f"Project root: {project_root}")

    # Cloud integration: if cloud.active: true (or legacy training.use_cloud) set in the YAML, run on SageMaker
    try:
        from soen_toolkit.training.configs.config_classes import (
            ExperimentConfig as _ExpCfg,
        )
        from soen_toolkit.training.utils.sagemaker_training_integration import (
            maybe_launch_on_sagemaker,
        )

        # Load raw YAML without validation to avoid local path checks when using S3
        with open(config_file) as _f:
            _raw = yaml.safe_load(_f) or {}
        _cloud_active = False
        try:
            _cloud_active = bool(((_raw.get("cloud") or {}).get("active")) or ((_raw.get("training") or {}).get("use_cloud")))
        except Exception:
            _cloud_active = False
        if _cloud_active:
            _cfg_obj = _ExpCfg.from_dict(_raw)
            handled = maybe_launch_on_sagemaker(config_file, _cfg_obj, script_dir=project_root)
            if handled:
                return 0
    except SystemExit as _e:
        raise

    # Run training locally
    # Use project_root as script_dir since we're not in examples anymore
    script_dir = project_root
    run_from_config(config_file, script_dir=script_dir)
    return None


if __name__ == "__main__":
    main()
