from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import torch


@dataclass
class GradientStatsStep:
    """Statistics captured for a single parameter at a specific optimization step."""

    step: int
    mean: float
    std: float
    min: float
    max: float
    l2_norm: float
    abs_mean: float
    zero_fraction: float
    non_finite: int
    total_elements: int


@dataclass
class _RunningSummary:
    """Internal structure to accumulate summary statistics."""

    steps: int = 0
    mean_sum: float = 0.0
    std_sum: float = 0.0
    abs_mean_sum: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    max_l2_norm: float = 0.0
    zero_elements: int = 0
    total_elements: int = 0
    non_finite: int = 0


@dataclass
class GradientStatsCollector:
    """Collect per-parameter gradient statistics across training steps."""

    track_per_step: bool = True
    max_steps_per_param: int | None = 200
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._compiled_include = [re.compile(p) for p in self.include_patterns or []]
        self._compiled_exclude = [re.compile(p) for p in self.exclude_patterns or []]
        self._per_param: dict[str, list[GradientStatsStep]] = {}
        self._summaries: dict[str, _RunningSummary] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record(self, param_name: str, grad: torch.Tensor, step: int) -> None:
        """Record gradient statistics for ``param_name`` at ``step``."""
        if not self._should_track(param_name) or grad is None:
            return

        stats = self._compute_stats(grad, step)
        if stats is None:
            return

        if self.track_per_step:
            bucket = self._per_param.setdefault(param_name, [])
            if self.max_steps_per_param is None or len(bucket) < self.max_steps_per_param:
                bucket.append(stats)
        self._update_summary(param_name, stats)

    def to_dict(self) -> dict[str, Any]:
        """Return the collected statistics as a pure-Python dict."""
        payload: dict[str, Any] = {}
        for name, summary in self._summaries.items():
            summary_dict = self._finalize_summary(summary)
            payload[name] = {"summary": summary_dict}
            if self.track_per_step:
                steps = [asdict(step) for step in self._per_param.get(name, [])]
                payload[name]["steps"] = steps
        return {"parameters": payload}

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize collected stats as JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | Path, *, indent: int = 2) -> Path:
        """Persist collected stats to ``path``."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(indent=indent))
        return destination

    def clear(self) -> None:
        """Reset all collected statistics."""
        self._per_param.clear()
        self._summaries.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _should_track(self, name: str) -> bool:
        included = True
        if self._compiled_include:
            included = any(pattern.search(name) for pattern in self._compiled_include)
        if not included:
            return False
        if self._compiled_exclude and any(pattern.search(name) for pattern in self._compiled_exclude):
            return False
        return True

    def _compute_stats(self, grad: torch.Tensor, step: int) -> GradientStatsStep | None:
        tensor = grad.detach()
        if tensor.is_sparse:
            tensor = tensor.coalesce().values()
        tensor = tensor.float().reshape(-1)
        total = tensor.numel()
        if total == 0:
            return None

        finite_mask = torch.isfinite(tensor)
        finite = tensor[finite_mask]
        non_finite = int(total - finite.numel())
        if finite.numel() == 0:
            # Treat as zero gradients to avoid NaNs propagating downstream.
            return GradientStatsStep(
                step=step,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                l2_norm=0.0,
                abs_mean=0.0,
                zero_fraction=1.0,
                non_finite=non_finite,
                total_elements=total,
            )

        mean = float(finite.mean().item())
        std = float(finite.std(unbiased=False).item())
        min_val = float(finite.min().item())
        max_val = float(finite.max().item())
        l2_norm = float(torch.linalg.vector_norm(finite).item())
        abs_mean = float(finite.abs().mean().item())
        zero_fraction = float((finite == 0).sum().item()) / float(total)

        return GradientStatsStep(
            step=step,
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            l2_norm=l2_norm,
            abs_mean=abs_mean,
            zero_fraction=zero_fraction,
            non_finite=non_finite,
            total_elements=total,
        )

    def _update_summary(self, param_name: str, stats: GradientStatsStep) -> None:
        summary = self._summaries.setdefault(param_name, _RunningSummary())
        summary.steps += 1
        summary.mean_sum += stats.mean
        summary.std_sum += stats.std
        summary.abs_mean_sum += stats.abs_mean
        summary.min_value = min(summary.min_value, stats.min)
        summary.max_value = max(summary.max_value, stats.max)
        summary.max_l2_norm = max(summary.max_l2_norm, stats.l2_norm)
        summary.zero_elements += int(stats.zero_fraction * stats.total_elements)
        summary.total_elements += stats.total_elements
        summary.non_finite += stats.non_finite

    @staticmethod
    def _finalize_summary(summary: _RunningSummary) -> dict[str, float | int]:
        if summary.steps == 0:
            return {}
        zero_fraction = (
            float(summary.zero_elements) / float(summary.total_elements) if summary.total_elements else 0.0
        )
        return {
            "steps": summary.steps,
            "mean": summary.mean_sum / summary.steps,
            "std": summary.std_sum / summary.steps,
            "abs_mean": summary.abs_mean_sum / summary.steps,
            "min": summary.min_value,
            "max": summary.max_value,
            "max_l2_norm": summary.max_l2_norm,
            "zero_fraction": zero_fraction,
            "non_finite": summary.non_finite,
        }

