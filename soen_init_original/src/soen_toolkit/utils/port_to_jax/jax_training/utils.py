from __future__ import annotations

from typing import Any

from soen_toolkit.core.soen_model_core import SOENModelCore
from soen_toolkit.utils.port_to_jax.convert import convert_core_model_to_jax


def load_soen_and_convert(path: str):
    core = SOENModelCore.load(path)
    core.eval()
    return convert_core_model_to_jax(core)


def tree_flat_params(params: dict[str, Any]):
    # Placeholder if we later want to flatten named params; not used yet
    return params


def tree_unflat_params(flat) -> dict[str, Any]:
    return flat
