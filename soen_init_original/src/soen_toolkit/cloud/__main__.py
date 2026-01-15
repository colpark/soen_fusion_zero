"""CLI entry point for soen_toolkit.cloud.

Allows running the cloud CLI via:
    python -m soen_toolkit.cloud train --config experiment.yaml
"""

from .cli import main

if __name__ == "__main__":
    main()

