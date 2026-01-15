#!/usr/bin/env python3
"""Metric interface and simple context type for objective composition."""

from __future__ import annotations

from typing import Any, Protocol


class Metric(Protocol):
    def compute(self, context: dict[str, Any], model: Any, s_histories: Any) -> dict[str, float]: ...
