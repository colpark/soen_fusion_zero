"""LeakyGRU virtual layer.

This module provides:
- `LeakyRU`: a GRU-shaped recurrence that behaves like an array of leaky integrators
  with per-unit trainable time-constant (via a trainable update bias), while using
  PyTorch's fused GRU kernel under the hood for speed.
- `LeakyGRULayer`: a `SequentialLayerBase` wrapper so LeakyRU plugs into the toolkit's
  connectivity/noise/tracking plumbing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
import torch.nn.functional as F

from soen_toolkit.core.layers.common import InitializerSpec, ParameterDef

from .recurrent import SequentialLayerBase

try:
    # Optional cuDNN weight flattener to avoid per-call compaction warnings.
    # This symbol isn't available in all torch versions.
    from torch.nn.utils.rnn import _cudnn_rnn_flatten_weight  # type: ignore
except ImportError:  # pragma: no cover
    _cudnn_rnn_flatten_weight = None


class LeakyRU(nn.Module):
    """GRU-like recurrence specialized to leaky integrators by masking gate edges.

    Equations (GRU convention):
        r = sigmoid(b_r)                 (fixed ~ 1)
        z = sigmoid(b_z)                 (optionally trainable; no edges)
        n_t = tanh(W_in x_t + b_n + r * (W_hn h_{t-1}))
        h_t = z * h_{t-1} + (1 - z) * n_t

    Time-constant parameterization (steps):
        z = exp(-dt / tau)
        leak_rate (mix-in) = 1 - z
    """

    mask_ih: torch.Tensor
    mask_hh: torch.Tensor
    bias_r_fixed: torch.Tensor
    eyeH: torch.Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        batch_first: bool = False,
        dt: float = 1.0,
        reset_bias: float = 10.0,
        tau_init: float | tuple[float, float] | torch.Tensor = 20.0,
        tau_spacing: str = "geometric",
        tau_eps: float = 1e-3,
        candidate_diag: str = "zero",
        train_alpha: bool = True,
        # Allow injecting parameters (e.g. from ParameterRegistry)
        bias_z: nn.Parameter | None = None,
        bias_n: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_first = bool(batch_first)
        self.dt = float(dt)

        H, input_dim = self.hidden_size, self.input_size

        # Raw GRU-style blocks (3H x I) and (3H x H) ordered (r, z, n).
        self.weight_ih = nn.Parameter(torch.empty(3 * H, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(3 * H, H))

        # Fixed reset bias; update bias and candidate bias trainable (unless frozen).
        self.register_buffer("bias_r_fixed", torch.full((H,), float(reset_bias)))

        # Use injected parameters if provided, otherwise create new ones
        if bias_z is not None:
            self.bias_z = bias_z
        else:
            self.bias_z = nn.Parameter(torch.empty(H), requires_grad=True)

        if bias_n is not None:
            self.bias_n = bias_n
        else:
            self.bias_n = nn.Parameter(torch.empty(H), requires_grad=True)

        # Masks: wipe ALL edges feeding r and z from BOTH x and h. Keep candidate edges.
        mask_ih = torch.zeros(3 * H, input_dim)
        mask_hh = torch.zeros(3 * H, H)
        mask_ih[2 * H : 3 * H, :] = 1.0  # candidate (n) input edges
        mask_hh[2 * H : 3 * H, :] = 1.0  # candidate (n) hidden edges
        self.register_buffer("mask_ih", mask_ih)
        self.register_buffer("mask_hh", mask_hh)

        if candidate_diag not in {"zero", "one", "free"}:
            raise ValueError("candidate_diag must be one of: 'zero', 'one', 'free'")
        self.candidate_diag = str(candidate_diag)
        self.register_buffer("eyeH", torch.eye(H))

        self.reset_parameters()

        # Only initialize tau if bias_z was NOT injected (or force init?)
        # Logic: If injected, we assume the caller handles initialization OR we construct it here.
        # LeakyGRULayer pattern: Create params (default const), THEN init them physically.
        # So we should run init regardless, as long as it's safe.
        self.init_update_bias_from_tau(tau_init, spacing=tau_spacing, tau_eps=tau_eps)

        if not train_alpha:
            self.freeze_alpha()

    def freeze_alpha(self) -> None:
        """Freeze alpha by freezing bias_z (since alpha = 1 - sigmoid(bias_z))."""

        self.bias_z.requires_grad_(False)

    def unfreeze_alpha(self) -> None:
        """Unfreeze alpha by allowing bias_z to receive gradients."""

        self.bias_z.requires_grad_(True)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        with torch.no_grad():
            self.bias_n.zero_()

        # Gate weights don't matter (masked), but keep tidy.
        with torch.no_grad():
            H = self.hidden_size
            self.weight_ih[: 2 * H].zero_()
            self.weight_hh[: 2 * H].zero_()

    @staticmethod
    def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def init_update_bias_from_tau(
        self,
        tau_init: float | tuple[float, float] | torch.Tensor,
        *,
        spacing: str = "geometric",
        tau_eps: float = 1e-3,
    ) -> None:
        """Initialize bias_z so that z = exp(-dt/tau)."""

        H = self.hidden_size
        dt = self.dt

        if isinstance(tau_init, torch.Tensor):
            tau = tau_init.to(dtype=torch.float32).flatten()
            if tau.numel() != H:
                raise ValueError(f"tau_init tensor must have {H} elements, got {tau.numel()}.")
        elif isinstance(tau_init, (tuple, list)) and len(tau_init) == 2:
            tau_min, tau_max = float(tau_init[0]), float(tau_init[1])
            tau_min = max(tau_min, tau_eps)
            tau_max = max(tau_max, tau_eps)

            if spacing == "linear":
                tau = torch.linspace(tau_min, tau_max, H)
            elif spacing == "geometric":
                tau = torch.exp(torch.linspace(math.log(tau_min), math.log(tau_max), H))
            else:
                raise ValueError("For (tau_min,tau_max), spacing must be 'linear' or 'geometric'.")
        else:
            tau_scalar = max(float(tau_init), tau_eps)
            tau = torch.full((H,), tau_scalar)

        z = torch.exp(-dt / tau)
        b = self._logit(z)

        with torch.no_grad():
            self.bias_z.copy_(b.to(self.bias_z.device, self.bias_z.dtype))

    @property
    def z(self) -> torch.Tensor:
        return torch.sigmoid(self.bias_z)

    @property
    def leak_rate(self) -> torch.Tensor:
        return 1.0 - self.z

    @property
    def tau(self) -> torch.Tensor:
        z = self.z.clamp(1e-8, 1.0 - 1e-8)
        return -self.dt / torch.log(z)

    def _effective_candidate_hh(self) -> torch.Tensor:
        H = self.hidden_size
        W_hh_eff = self.weight_hh * self.mask_hh
        W_hn = cast(torch.Tensor, W_hh_eff[2 * H : 3 * H])  # (H, H)

        if self.candidate_diag == "free":
            return W_hn

        fixed = 0.0 if self.candidate_diag == "zero" else 1.0
        eye = self.eyeH.to(W_hn.device, W_hn.dtype)
        return W_hn * (1.0 - eye) + (fixed * eye)

    def _masked_weights_and_biases(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return masked weights/biases in standard GRU layout for fused kernels."""

        H = self.hidden_size

        W_ih_eff = self.weight_ih * self.mask_ih
        W_hh_eff = self.weight_hh * self.mask_hh

        W_hn = self._effective_candidate_hh()
        W_hh_eff = W_hh_eff.clone()
        W_hh_eff[2 * H : 3 * H] = W_hn

        device = W_ih_eff.device
        dtype = W_ih_eff.dtype
        bias_r = self.bias_r_fixed.to(device=device, dtype=dtype)
        bias_ih = torch.cat([bias_r, self.bias_z, self.bias_n], dim=0)
        bias_hh = torch.zeros_like(bias_ih)

        return W_ih_eff, W_hh_eff, bias_ih, bias_hh

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Single recurrent step (used by the toolkit stepwise path)."""

        H = self.hidden_size

        W_ih_eff = self.weight_ih * self.mask_ih
        W_in = W_ih_eff[2 * H : 3 * H]  # (H, I)
        W_hn = self._effective_candidate_hh()  # (H, H)

        r = torch.sigmoid(self.bias_r_fixed).unsqueeze(0)  # (1, H)
        z = torch.sigmoid(self.bias_z).unsqueeze(0)  # (1, H)

        n_preact = F.linear(x_t, W_in, self.bias_n) + r * F.linear(h, W_hn, None)
        n = torch.tanh(n_preact)

        return z * h + (1.0 - z) * n

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused sequence forward using the internal GRU kernel."""

        if x.dim() != 3:
            raise ValueError("Input x must be 3D (T,B,I) or (B,T,I).")

        if not hasattr(torch, "_VF") or not hasattr(torch._VF, "gru"):  # pragma: no cover
            raise RuntimeError(
                "This torch build does not expose torch._VF.gru. "
                "LeakyRU requires a torch version that provides the internal GRU kernel."
            )

        W_ih, W_hh, b_ih, b_hh = self._masked_weights_and_biases()

        if h0 is None:
            batch_dim = x.shape[0] if self.batch_first else x.shape[1]
            h0 = torch.zeros(1, batch_dim, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            if h0.dim() == 2:
                h0 = h0.unsqueeze(0)
            if h0.dim() != 3:
                raise ValueError("h0 must be shape (B,H) or (1,B,H).")

        if x.is_cuda and _cudnn_rnn_flatten_weight is not None:  # pragma: no cover
            _cudnn_rnn_flatten_weight([W_ih, W_hh, b_ih, b_hh], 1, False)

        y, h_n = torch._VF.gru(
            x,
            h0,
            (W_ih, W_hh, b_ih, b_hh),
            True,  # has_biases
            1,  # num_layers
            0.0,  # dropout
            self.training,
            False,  # bidirectional
            self.batch_first,
        )

        return y, h_n.squeeze(0)


class LeakyGRULayer(SequentialLayerBase):
    """Toolkit virtual layer wrapper for `LeakyRU`.

    - Fast path: no internal connectivity + no noise → fused sequence forward.
    - General path: stepwise loop (supports internal connectivity and noise).
    """

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        solver: str | None = None,
        # LeakyRU configuration (intended to come from cfg.params)
        reset_bias: float = 10.0,
        tau_init: float | tuple[float, float] | torch.Tensor = 20.0,
        tau_spacing: str = "geometric",
        tau_eps: float = 1e-3,
        candidate_diag: str = "zero",
        train_alpha: bool = True,
        # Internal connectivity (toolkit pattern)
        connectivity: torch.Tensor | None = None,
        connectivity_spec: dict[str, object] | None = None,
        connectivity_constraints: dict[str, float] | None = None,
        learnable_connectivity: bool = True,
        **kwargs,
    ) -> None:
        # Define parameters for registry
        parameters = [
            ParameterDef(
                name="decay_bias",
                default=4.0,  # Maps to bias_z
                learnable=True,
                initializer=InitializerSpec(method="constant", params={"value": 4.0}),
            ),
            ParameterDef(
                name="candidate_bias",
                default=0.0,  # Maps to bias_n
                learnable=True,
                initializer=InitializerSpec(method="constant", params={"value": 0.0}),
            ),
        ]

        super().__init__(
            dt=dt,
            dim=dim,
            parameters=parameters,
            connectivity=connectivity,
            connectivity_spec=connectivity_spec,
            connectivity_constraints=connectivity_constraints,
            learnable_connectivity=learnable_connectivity,
        )

        dt_value = float(dt) if not isinstance(dt, torch.Tensor) else float(dt.item())

        # At this point, self.decay_bias and self.candidate_bias exist as Parameters (or properties if transformed)
        # We pass them to LeakyRU.

        self.core = LeakyRU(
            input_size=dim,
            hidden_size=dim,
            batch_first=True,
            dt=dt_value,
            reset_bias=reset_bias,
            tau_init=tau_init,
            tau_spacing=tau_spacing,
            tau_eps=tau_eps,
            candidate_diag=candidate_diag,
            train_alpha=train_alpha,
            bias_z=cast(nn.Parameter, self.decay_bias),
            bias_n=cast(nn.Parameter, self.candidate_bias),
        )
        self.add_module("core", self.core)

    # ------------------------------------------------------------------
    # Convenience passthroughs
    # ------------------------------------------------------------------
    def freeze_alpha(self) -> None:
        self.core.freeze_alpha()

    def unfreeze_alpha(self) -> None:
        self.core.unfreeze_alpha()

    @property
    def z(self) -> torch.Tensor:
        return self.core.z

    @property
    def leak_rate(self) -> torch.Tensor:
        return self.core.leak_rate

    @property
    def tau(self) -> torch.Tensor:
        return self.core.tau

    # ------------------------------------------------------------------
    # SequentialLayerBase hooks
    # ------------------------------------------------------------------
    def _init_hidden(self, initial_state: torch.Tensor) -> torch.Tensor:
        return initial_state

    def _step(self, phi_t: torch.Tensor, hidden: Any) -> tuple[torch.Tensor, torch.Tensor]:
        next_state = self.core.step(phi_t, cast(torch.Tensor, hidden))
        return next_state, next_state

    def forward(  # type: ignore[override]
        self,
        phi: torch.Tensor,
        *,
        noise_config: dict[str, object] | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Fast path: no internal connectivity + no noise → fused sequence forward.
        if self.connectivity is None and noise_config is None:
            batch, steps, dim = self.validate_input(phi)
            init_state = self.prepare_initial_state(phi, initial_state)

            self._clear_histories()
            self.feature_stack.set_noise_config(noise_config)
            self._on_noise_config_updated(noise_config)

            y, _hT = self.core(phi, h0=init_state)

            history = torch.empty(batch, steps + 1, dim, device=phi.device, dtype=phi.dtype)
            history[:, 0, :] = init_state
            history[:, 1:, :] = y

            # Tracking sequences (match stepwise semantics: no init state in history lists).
            self._set_phi_history_sequence(phi)
            self._set_state_history_sequence(y)
            return history

        # General path: stepwise loop supports connectivity and noise.
        return super().forward(phi, noise_config=noise_config, initial_state=initial_state)


__all__ = ["LeakyGRULayer", "LeakyRU"]


