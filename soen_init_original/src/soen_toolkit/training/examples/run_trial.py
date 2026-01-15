# run_trial.py
import argparse
import logging
import os
from pathlib import Path
import sys

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Define the source root directory containing the 'soen_toolkit' package
SRC_ROOT = SCRIPT_DIR.parent.parent / "src"  # Go up two levels and into src
sys.path.insert(0, str(SRC_ROOT))  # Prepend src root to ensure it's found first

# === ROBUST PROJECT ROOT SETUP ===
# Set environment variable for paths.py BEFORE any imports
os.environ["SOEN_PROJECT_ROOT"] = str(SRC_ROOT.parent)

# NOW we can safely import from soen_toolkit
from soen_toolkit.training.trainers.experiment import run_from_config  # noqa: E402
from soen_toolkit.training.utils.local_logging import setup_logger  # noqa: E402

if __name__ == "__main__":
    # Set up logging relative to script dir
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger(log_file=log_dir / "experiment.log", console_level=logging.WARNING)

    # --- Argument Parsing for config file ---
    parser = argparse.ArgumentParser(description="Run an experiment from a YAML configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "template.yaml"),  # Default to template.yaml in script's dir
        help="Path to the YAML configuration file. Defaults to 'template.yaml' in the script directory.",
    )
    args = parser.parse_args()
    # --- End Argument Parsing ---

    # Config path from argparse
    config_path = Path(args.config)

    # Check if the specified config file exists
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Using configuration file: {config_path}")
    logger.info(f"Project root: {os.environ['SOEN_PROJECT_ROOT']}")

    # Run experiment from config, passing the script directory
    run_from_config(config_path, script_dir=SCRIPT_DIR)
