"""Wrapper to cut gradients at specific timesteps in SOEN models."""

from typing import Any

import torch
from torch import nn


class GradientCutoffWrapper(nn.Module):
    """Wrapper that cuts gradients at a specific timestep during forward pass.

    This is more efficient than doing gradient cutoff in the loss function
    because it prevents unnecessary gradient computation for early timesteps.
    """

    def __init__(self, soen_model: nn.Module, cutoff_timestep: int = -1, keep_last_n: int = 1) -> None:
        """Args:
        soen_model: The SOEN model to wrap
        cutoff_timestep: Timestep before which to cut gradients
                       (-1 means cut all but the final timestep)
        keep_last_n: Number of final timesteps to keep gradients for
                    (only used if cutoff_timestep == -1).

        """
        super().__init__()
        self.soen_model = soen_model
        self.cutoff_timestep = cutoff_timestep
        self.keep_last_n = keep_last_n

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass with gradient cutoff.

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Tuple of (final_state_with_cutoff, all_states_with_cutoff)

        """
        # Get full model output
        final_state, all_states = self.soen_model(x)

        # Apply gradient cutoff to final state
        final_state_cutoff = self._apply_gradient_cutoff(final_state)

        # Apply gradient cutoff to all states
        all_states_cutoff = [self._apply_gradient_cutoff(state) for state in all_states]

        return final_state_cutoff, all_states_cutoff

    def _apply_gradient_cutoff(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Apply gradient cutoff to a state tensor."""
        if state_tensor.dim() != 3:
            return state_tensor  # Not a sequence, no cutoff needed

        _batch_size, seq_len_plus_one, _output_dim = state_tensor.shape

        if self.cutoff_timestep >= 0:
            # Cut gradients at specific timestep
            if self.cutoff_timestep >= seq_len_plus_one:
                return state_tensor  # Cutoff point is beyond sequence, no cutoff

            cutoff_state = torch.zeros_like(state_tensor)

            # Detach early timesteps
            if self.cutoff_timestep > 0:
                cutoff_state[:, : self.cutoff_timestep, :] = state_tensor[:, : self.cutoff_timestep, :].detach()

            # Keep gradients for later timesteps
            cutoff_state[:, self.cutoff_timestep :, :] = state_tensor[:, self.cutoff_timestep :, :]

            return cutoff_state
        # Keep gradients only for the last N timesteps
        cutoff_point = max(0, seq_len_plus_one - self.keep_last_n)

        cutoff_state = torch.zeros_like(state_tensor)

        # Detach early timesteps
        if cutoff_point > 0:
            cutoff_state[:, :cutoff_point, :] = state_tensor[:, :cutoff_point, :].detach()

        # Keep gradients for final timesteps
        cutoff_state[:, cutoff_point:, :] = state_tensor[:, cutoff_point:, :]

        return cutoff_state

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.soen_model, name)
