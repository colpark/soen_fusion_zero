"""Alias runner for the training CLI.

Usage:
    python -m soen_toolkit.train --config path/to/training_config.yaml

This forwards to `soen_toolkit.training.__main__.main`.
"""

from __future__ import annotations


def main() -> int:
    from soen_toolkit.training.__main__ import main as _main

    result = _main()
    return result if result is not None else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
