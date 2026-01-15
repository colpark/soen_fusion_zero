#!/usr/bin/env python3
"""Configuration validation tool for soen_toolkit training.
Validates training configurations and provides helpful suggestions.
"""

import argparse
import logging
from pathlib import Path
import sys

from soen_toolkit.training.configs.config_validation import (
    auto_detect_task_type,
    validate_config,
)
from soen_toolkit.training.configs.experiment_config import load_config


def setup_logging(verbose: bool = False) -> None:
    """Set up logging for the validation tool."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> int | None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate soen_toolkit training configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a config file
  python -m soen_toolkit.training.tools.validate_config config.yaml

  # Auto-detect task type from data
  python -m soen_toolkit.training.tools.validate_config config.yaml --auto-detect

  # Show detailed suggestions
  python -m soen_toolkit.training.tools.validate_config config.yaml --verbose

  # Only show errors (no warnings)
  python -m soen_toolkit.training.tools.validate_config config.yaml --errors-only
        """,
    )

    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--auto-detect",
        "-a",
        action="store_true",
        help="Auto-detect task type from data and show suggestions",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed validation information",
    )

    parser.add_argument(
        "--errors-only",
        "-e",
        action="store_true",
        help="Only show errors, suppress warnings",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation, only auto-detect",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    config_path = Path(args.config_path)
    if not config_path.exists():
        return 1

    try:
        # Load config without validation first
        config = load_config(config_path, validate=False, auto_detect=False)

        # Auto-detection if requested
        if args.auto_detect and hasattr(config.data, "data_path") and config.data.data_path:
            suggestions = auto_detect_task_type(config.data.data_path)

            if suggestions["confidence"] == "high" or suggestions["confidence"] == "medium":
                pass
            else:
                pass

            current_paradigm = getattr(config.training, "paradigm", "supervised")
            current_mapping = getattr(config.training, "mapping", "seq2static")

            if suggestions["paradigm"]:
                "[OK]" if suggestions["paradigm"] == current_paradigm else "[WARNING]"

            if suggestions["mapping"]:
                "[OK]" if suggestions["mapping"] == current_mapping else "[WARNING]"

            if suggestions["losses"]:
                current_losses = [loss.name for loss in config.training.loss.losses] if hasattr(config.training, "loss") and config.training.loss else []
                suggested_losses = [loss["name"] for loss in suggestions["losses"]]
                "[OK]" if set(suggested_losses).intersection(current_losses) else "[WARNING]"
                if current_losses:
                    pass

            if suggestions["num_classes"]:
                current_classes = getattr(config.data, "num_classes", None)
                "[OK]" if current_classes == suggestions["num_classes"] else "[WARNING]"

        # Validation if requested
        if not args.no_validate:
            warnings, errors = validate_config(config, raise_on_error=False)

            # Display results
            if errors:
                for _i, _error in enumerate(errors, 1):
                    pass

                if warnings and not args.errors_only:
                    for _i, _warning in enumerate(warnings, 1):
                        pass

                return 1

            if warnings and not args.errors_only:
                for _i, _warning in enumerate(warnings, 1):
                    pass

                return 0

            return 0

        return 0

    except Exception:
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
