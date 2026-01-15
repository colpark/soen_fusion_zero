# FILEPATH: src/soen_toolkit/model_creation_gui/utils/paths.py
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent / "resources" / "icons"


def icon(name: str) -> str:
    p = _BASE / name
    return str(p) if p.exists() else ""
