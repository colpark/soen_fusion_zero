"""Spike-Timing-Dependent Plasticity (STDP) for continuous activities.

This module implements STDP rules that operate on layer trajectories.
"""

import torch

from soen_toolkit.training.local_learning.rules.base import AbstractLocalRule


class STDP(AbstractLocalRule):
    """Correlation-based STDP for continuous activity trajectories.

    This rule approximates STDP by computing the interaction between 
    pre-synaptic and post-synaptic activity traces over time.

    Formula:
        ΔW = η · Σ_t [ (pre(t) · post_trace(t)) - (pre_trace(t) · post(t)) ]
        where traces are exponentially filtered versions of the activity.

    Positive updates (LTP) occur when pre-synaptic activity precedes post-synaptic.
    Negative updates (LTD) occur when post-synaptic precedes pre-synaptic.
    """

    def __init__(
        self,
        lr: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 1.0,
        a_minus: float = 1.0
    ):
        """Initialize STDP rule.

        Args:
            lr: Learning rate
            tau_plus: Time constant for LTP trace (in timesteps)
            tau_minus: Time constant for LTD trace (in timesteps)
            a_plus: Magnitude of LTP
            a_minus: Magnitude of LTD
        """
        super().__init__(lr)
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

    @property
    def requires_trajectory(self) -> bool:
        return True

    def compute_update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        modulator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute STDP weight update.

        Args:
            pre_activity: [batch, seq_len, pre_dim]
            post_activity: [batch, seq_len, post_dim]
            weights: [post_dim, pre_dim]
            modulator: Optional modulation signal [batch]

        Returns:
            weight_update: [post_dim, pre_dim]
        """
        # Detach to prevent gradients
        pre = pre_activity.detach()
        post = post_activity.detach()

        batch_size, seq_len, _ = pre.shape

        # 1. Compute traces (exponentially filtered activity)
        # pre_trace(t) = sum_{k=0}^t pre(k) * exp(-(t-k)/tau_minus)
        # post_trace(t) = sum_{k=0}^t post(k) * exp(-(t-k)/tau_plus)
        pre_trace = self._compute_trace(pre, self.tau_minus)  # [batch, seq, pre_dim]
        post_trace = self._compute_trace(post, self.tau_plus) # [batch, seq, post_dim]

        # 2. Compute LTP (pre occurs before post)
        # LTP = A_plus * sum_t (post(t) * pre_trace(t-1))
        # LTD = A_minus * sum_t (pre(t) * post_trace(t-1))

        # LTP term: post[t] * pre_trace[t]
        ltp = torch.einsum('btj,bti->ji', post, pre_trace) * self.a_plus

        # LTD term: pre[t] * post_trace[t]
        ltd = torch.einsum('bti,btj->ji', pre, post_trace) * self.a_minus

        # 3. Apply modulation if present
        update = (ltp - ltd) / (batch_size * seq_len)

        if modulator is not None:
            # Simple scalar or per-sample modulation
            # Note: For trajectory rules, modulation could also be temporal,
            # but here we keep it simple for now.
            if modulator.dim() == 0:
                update = update * modulator
            elif modulator.dim() == 1:
                # Modulator [batch] -> weight it into the ltp/ltd computation
                # This would require re-doing the einsum with modulation
                ltp_mod = torch.einsum('b,btj,bti->ji', modulator, post, pre_trace) * self.a_plus
                ltd_mod = torch.einsum('b,bti,btj->ji', modulator, pre, post_trace) * self.a_minus
                update = (ltp_mod - ltd_mod) / (batch_size * seq_len)

        return self.lr * update

    def _compute_trace(self, activity: torch.Tensor, tau: float) -> torch.Tensor:
        """Compute exponential trace of activity series.
        
        trace[t] = alpha * trace[t-1] + activity[t]
        where alpha = exp(-1/tau)
        """
        alpha = torch.exp(torch.tensor(-1.0 / tau, device=activity.device))

        trace = torch.zeros_like(activity)
        current_trace = torch.zeros(activity.size(0), activity.size(2), device=activity.device)

        for t in range(activity.size(1)):
            current_trace = alpha * current_trace + activity[:, t, :]
            trace[:, t, :] = current_trace

        return trace
