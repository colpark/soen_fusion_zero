"""Launch the HPO GUI.

Usage:
    python -m soen_toolkit.hpo_gui

Thin launcher delegating to ``soen_toolkit.utils.hpo.tools.hpo_gui``.
Seeds a couple of legacy aliases used by the GUI internals.
"""

from __future__ import annotations

from soen_toolkit.utils.hpo.tools.hpo_gui import main as _main


def main() -> int:  # pragma: no cover - thin wrapper
    return _main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
