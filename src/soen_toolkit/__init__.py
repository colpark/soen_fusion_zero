"""Top-level package for soen_toolkit.

This module exposes a few conveniences and lazily forwards attribute access
to common subpackages so code like ``soen_toolkit.core`` works after importing
``soen_toolkit``.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution, version as _pkg_version
from importlib.resources import files as _pkg_files
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

try:  # Best-effort version string
    __version__ = _pkg_version("soen-toolkit")
except PackageNotFoundError:  # During editable installs/tests
    __version__ = "0.0.0"

# Provide a simple, printable alias like numpy/torch patterns
version = __version__


# Remove eager guarded import; expose via __getattr__ lazily to fail fast on use


def __getattr__(name: str):
    """Lazy import selected subpackages on attribute access.

    Enables ``import soen_toolkit
    soen_toolkit.core`` style usage.
    """
    lazy_submodules = {
        "core",
        "layers",
        "training",
        "utils",
        "model_creation_gui",
        # "criticality",  # deprecated; use utils.hpo instead
        "robustness_tool",
    }
    if name in lazy_submodules:
        return import_module(f"{__name__}.{name}")
    if name == "SOENModelCore":
        from .core import SOENModelCore as _SOENModelCore

        return _SOENModelCore
    # Expose runtime helpers even if someone imported a stale module object
    if name in {"print_structure", "find_readmes", "read_readme", "read_main_readme"}:
        return globals()[name]
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = (
    "SOENModelCore",
    "__version__",
    "find_readmes",
    "print_structure",
    "read_main_readme",
    "read_readme",
    "version",
)


def _iter_submodules(package_name: str) -> Iterable[tuple[str, bool]]:
    pkg = import_module(package_name)
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        yield mod.name, mod.ispkg


def print_structure(max_depth: int = 2) -> None:
    """Pretty-print the package tree up to max_depth.

    Example:
        >>> import soen_toolkit as st
        >>> st.print_structure()

    """
    root = __name__
    parts_to_depth: dict[str, int] = {}
    for fullname, _ispkg in _iter_submodules(root):
        depth = fullname.count(".") - root.count(".")
        if depth <= max_depth:
            parts_to_depth[fullname] = depth
    for name in sorted(parts_to_depth, key=lambda n: (parts_to_depth[n], n)):
        depth = parts_to_depth[name]
        "  " * depth


def find_readmes() -> list[tuple[str, str]]:
    """Return a list of (module, relative_path) for README-like files shipped in the wheel."""
    candidates = []
    modules = [
        "soen_toolkit",
        "soen_toolkit.training",
        "soen_toolkit.robustness_tool",
        "soen_toolkit.model_creation_gui",
    ]
    names = ["README.md", "README.txt", "SOENSimFramework.md"]
    for mod in modules:
        try:
            base = _pkg_files(mod)
        except Exception:
            continue
        for nm in names:
            try:
                p = base / nm
                if p.is_file():
                    candidates.append((mod, nm))
            except Exception:
                pass
    return candidates


def read_readme(module: str, filename: str | None = None) -> str:
    """Read and return the README content from a submodule packaged file.

    Args:
        module: e.g. "soen_toolkit.training"
        filename: override file name. Defaults to README.md.

    """
    fname = filename or "README.md"
    path = _pkg_files(module) / fname
    return path.read_text(encoding="utf-8")


def read_main_readme() -> str:
    """Read the package-level README if bundled.

    Tries `soen_toolkit/README.md`. Falls back to the training README.
    """
    try:
        return read_readme("soen_toolkit", "README.md")
    except Exception:
        # Try to extract the long_description from installed METADATA
        try:
            dist = distribution("soen-toolkit")
            meta_text = dist.read_text("METADATA") or ""
            # METADATA header is RFC822; long description follows the first blank line
            parts = meta_text.split("\n\n", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1]
        except Exception:
            pass
        # Last resort: training README in package
        return read_readme("soen_toolkit.training", "README.md")
