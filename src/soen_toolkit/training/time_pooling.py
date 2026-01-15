"""Declarative specs for time pooling options used by the training GUI.

This module does not change runtime behavior
it only exposes registries and
parameter metadata so UIs can render forms without hardcoding values.
"""

from __future__ import annotations

from typing import Any

# Available methods
TIME_POOLING_REGISTRY = [
    "max",
    "mean",
    "rms",
    "final",
    "mean_last_n",
    "mean_range",
    "ewa",
]

# Parameter specs per method. Minimal, but enough for UI.
# Schema for each param: {type, default, min?, max?, step?, options? , description?}
TIME_POOLING_PARAM_TYPES: dict[str, dict[str, dict[str, Any]]] = {
    "max": {
        "scale": {"type": "float", "default": 1.0, "min": 0.0, "step": 0.1, "description": "Multiply pooled feature by this factor."},
    },
    "mean": {
        "scale": {"type": "float", "default": 1.0, "min": 0.0, "step": 0.1},
    },
    "rms": {
        "scale": {"type": "float", "default": 1.0, "min": 0.0, "step": 0.1},
    },
    "final": {
        "scale": {"type": "float", "default": 1.0, "min": 0.0, "step": 0.1},
    },
    "mean_last_n": {
        "n": {"type": "int", "default": 10, "min": 1, "description": "Average over last N timesteps."},
        "scale": {"type": "float", "default": 1.0, "min": 0.0, "step": 0.1},
    },
    "mean_range": {
        "range_start": {"type": "int", "default": 0, "min": 0, "description": "Start timestep (inclusive)."},
        "range_end": {"type": "int", "default": 100, "min": 1, "description": "End timestep (exclusive)."},
        "scale": {"type": "float", "default": 1.0, "min": 0.0, "step": 0.1},
    },
    "ewa": {
        "min_weight": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "description": "Relative weight of the oldest timestep."},
        "normalize_weights": {"type": "bool", "default": True},
        "scale": {"type": "float", "default": 1.0, "min": 0.0, "step": 0.1},
    },
}

TIME_POOLING_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "max": {"description": "Maximum over time."},
    "mean": {"description": "Mean over time."},
    "rms": {"description": "Root-mean-square over time."},
    "final": {"description": "Final timestep only."},
    "mean_last_n": {"description": "Mean of the last N timesteps."},
    "mean_range": {"description": "Mean over a start:end range of timesteps."},
    "ewa": {"description": "Exponentially weighted average over time."},
}
