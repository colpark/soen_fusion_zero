"""Gradient analysis mixin for SOENModelCore."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from torch import nn

    from soen_toolkit.analysis import GradientStatsCollector


class GradientAnalysisMixin:
    """Mixin providing gradient flow analysis utilities."""

    if TYPE_CHECKING:
        from torch import nn
        from soen_toolkit.analysis import GradientStatsCollector

        # Attributes expected from the composed class
        def reset_stateful_components(self) -> None: ...
        def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def analyze_gradient_flow(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | str | None = None,
        batch_size: int = 8,
        max_batches: int | None = None,
        device: torch.device | str | None = None,
        output_reduction: str = "final",
        reset_state_each_batch: bool = True,
        gradient_stats_collector: GradientStatsCollector | None = None,
    ) -> dict[str, Any]:
        """Collect gradient statistics for a batch of inputs without a trainer loop.

        Args:
            inputs: Input tensor [N, seq_len, input_dim]
            targets: Target tensor [N] or [N, output_dim]
            loss_fn: Loss function (callable) or name ('mse', 'cross_entropy')
            batch_size: Batch size for gradient computation
            max_batches: Maximum number of batches to process
            device: Device to run on (defaults to model's current device)
            output_reduction: How to reduce outputs ('final', 'mean', 'none')
            reset_state_each_batch: Whether to reset stateful components between batches
            gradient_stats_collector: Optional external collector

        Returns:
            Dictionary with 'loss', 'batches_processed', and 'gradient_stats'
        """
        from soen_toolkit.analysis import GradientStatsCollector

        original_device = next(cast(Any, self).parameters()).device
        target_device = torch.device(device) if device else original_device

        if target_device != original_device:
            cast(Any, self).to(target_device)

        resolved_loss, loss_kind = self._resolve_gradient_loss(loss_fn, targets)

        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        owns_collector = gradient_stats_collector is None
        collector = gradient_stats_collector or GradientStatsCollector(
            track_per_step=True,
            max_steps_per_param=200,
        )

        self.reset_stateful_components()
        cast(Any, self).eval()
        total_loss = 0.0
        processed = 0

        try:
            for batch_idx, (batch_inputs, batch_targets) in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                batch_inputs = batch_inputs.to(target_device)
                batch_targets = batch_targets.to(target_device)

                if reset_state_each_batch:
                    self.reset_stateful_components()
                cast(Any, self).zero_grad(set_to_none=True)

                outputs, _ = cast(Any, self)(batch_inputs)
                outputs_for_loss = self._reduce_outputs_for_analysis(outputs, mode=output_reduction)
                targets_for_loss = self._prepare_targets_for_loss(batch_targets, loss_kind)

                loss = resolved_loss(outputs_for_loss, targets_for_loss)
                loss.backward()

                processed += 1
                total_loss += float(loss.detach().cpu().item())

                if collector is not None:
                    for name, param in cast(Any, self).named_parameters():
                        if param.requires_grad and param.grad is not None:
                            collector.record(name, param.grad, step=batch_idx)

                            # Debug: log suspiciously exact gradients
                            grad_max = param.grad.abs().max().item()
                            if abs(grad_max - 1.0) < 1e-6:
                                pass  # DEBUG: grad_max close to 1.0

            avg_loss = total_loss / processed if processed else float("nan")
            result = {
                "loss": avg_loss,
                "batches_processed": processed,
                "gradient_stats": collector.to_dict() if collector is not None else {},
            }
        finally:
            cast(Any, self).zero_grad(set_to_none=True)
            if target_device != original_device:
                cast(Any, self).to(original_device)
            if owns_collector and collector is not None:
                collector.clear()

        return result

    @staticmethod
    def _resolve_gradient_loss(
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | str | None,
        targets: torch.Tensor,
    ) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], str]:
        """Resolve loss function from string name or callable.

        Args:
            loss_fn: Loss function or string name
            targets: Target tensor to infer loss type if needed

        Returns:
            Tuple of (loss_function, loss_kind_string)
        """
        import torch.nn.functional as F

        if callable(loss_fn):
            return loss_fn, "custom"

        loss_name = loss_fn or ("mse" if torch.is_floating_point(targets) else "cross_entropy")

        if loss_name == "mse":
            return F.mse_loss, "mse"
        if loss_name == "cross_entropy":
            return F.cross_entropy, "cross_entropy"

        msg = f"Unsupported loss name '{loss_name}'. Use 'mse', 'cross_entropy', or provide a callable."
        raise ValueError(msg)

    @staticmethod
    def _reduce_outputs_for_analysis(outputs: torch.Tensor, *, mode: str) -> torch.Tensor:
        """Reduce model outputs for loss computation.

        Args:
            outputs: Model outputs [batch, seq_len, output_dim] or [batch, output_dim]
            mode: Reduction mode ('final', 'mean', 'none')

        Returns:
            Reduced outputs suitable for loss computation
        """
        if outputs.dim() == 2:
            return outputs

        if mode == "final":
            return outputs[:, -1, :]
        if mode == "mean":
            return outputs.mean(dim=1)
        if mode == "none":
            return outputs

        msg = f"Unsupported output_reduction mode '{mode}'. Use 'final', 'mean', or 'none'."
        raise ValueError(msg)

    @staticmethod
    def _prepare_targets_for_loss(targets: torch.Tensor, loss_kind: str) -> torch.Tensor:
        """Prepare targets for loss computation.

        Args:
            targets: Target tensor
            loss_kind: Loss type ('mse', 'cross_entropy', 'custom')

        Returns:
            Targets in appropriate format for loss function
        """
        if loss_kind == "cross_entropy":
            if targets.dim() > 1:
                return targets.argmax(dim=-1)
            return targets.long()

        return targets.float()

