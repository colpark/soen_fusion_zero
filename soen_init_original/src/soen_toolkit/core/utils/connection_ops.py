"""Connection operators for external edges: fixed matrix and multiplier-based.

This module provides vectorized helpers to compute upstream phi contributions for
connections either as a static matrix multiply or as a dynamic multiplier circuit
that integrates a per-edge state using forward Euler steps.
"""

from __future__ import annotations

from typing import Any

import torch

# Import at module level for reliability
from soen_toolkit.core.source_functions import SOURCE_FUNCTIONS

# Smoothing parameter for soft_abs: controls the "sharpness" of the abs approximation
_SOFT_ABS_EPS = 0.01


def soft_abs(x: torch.Tensor, eps: float = _SOFT_ABS_EPS) -> torch.Tensor:
    """Smooth approximation of abs(x) using sqrt(x² + ε).

    Unlike torch.abs(), this function has a continuous derivative everywhere,
    which is important for numerical stability in Forward Euler integration.

    Args:
        x: Input tensor
        eps: Smoothing parameter (default: 0.01)

    Returns:
        Smooth approximation of abs(x)
    """
    return torch.sqrt(x * x + eps)


def build_edge_index(mask: torch.Tensor | None, J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build edge index arrays (src, dst) for a connection.

    Args:
        mask: Optional binary mask tensor of shape [to_dim, from_dim]. If None, treat as dense.
        J: Weight matrix of shape [to_dim, from_dim] (used for shape/device inference).

    Returns:
        (edge_src_idx, edge_dst_idx): 1-D LongTensors of length E giving source and destination indices per edge.

    """
    device = J.device
    to_dim, from_dim = J.shape
    if mask is None:
        # Dense: generate full grid indices
        dst = torch.arange(to_dim, device=device, dtype=torch.long)
        src = torch.arange(from_dim, device=device, dtype=torch.long)
        grid_dst = dst.view(-1, 1).expand(to_dim, from_dim).reshape(-1)
        grid_src = src.view(1, -1).expand(to_dim, from_dim).reshape(-1)
        return grid_src, grid_dst
    idx = (mask > 0).nonzero(as_tuple=False)
    if idx.numel() == 0:
        # No edges allowed: return empty indices on correct device
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    edge_dst_idx = idx[:, 0].to(dtype=torch.long)
    edge_src_idx = idx[:, 1].to(dtype=torch.long)
    return edge_src_idx, edge_dst_idx


class StaticMatrixOp:
    """Static matrix connection: phi = s @ J^T."""

    @staticmethod
    def layerwise(s_hist: torch.Tensor, J_eff: torch.Tensor) -> torch.Tensor:
        """Compute [B,T,D] from s_hist [B,T,F] and J_eff [D,F]."""
        return torch.matmul(s_hist, J_eff.t())

    @staticmethod
    def step(s_src: torch.Tensor, J_eff: torch.Tensor) -> torch.Tensor:
        """Compute [B,D] from s_src [B,F] and J_eff [D,F]."""
        return torch.matmul(s_src, J_eff.t())


_SF_CACHE: dict[str, Any] = {}


def _get_source_function(key: str):
    k = str(key or "RateArray")
    fn = _SF_CACHE.get(k)
    if fn is not None:
        return fn
    builder = SOURCE_FUNCTIONS.get(k)
    if builder is None:
        # Fallback to RateArray if available
        builder = SOURCE_FUNCTIONS.get("RateArray")
    if builder is None:
        msg = f"Unknown source function type '{key}'. Available: {list(SOURCE_FUNCTIONS.keys())}"
        raise ValueError(msg)
    inst = builder()
    _SF_CACHE[k] = inst
    return inst


def _expand_param_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar or 1D tensor to match ``target`` shape.

    Supports target with rank 2 ([B,E]) or 3 ([B,T,E]) and higher. If ``x`` is:
    - scalar: returns a tensor of ones broadcast to target.
    - 1D of length E: aligns to last dimension and broadcasts leading dims as 1.
    - already broadcastable: attempts expand_as.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.to(dtype=target.dtype, device=target.device)
    if x.dim() == 0:
        # Shape of all ones with same rank as target
        shape = [1] * target.dim()
        return x.view(*shape).expand_as(target)
    if x.dim() == 1:
        if x.shape[0] == 1:
            shape = [1] * target.dim()
            return x.view(*shape).expand_as(target)
        if x.shape[0] == target.shape[-1]:
            shape = [1] * (target.dim() - 1) + [target.shape[-1]]
            return x.view(*shape).expand_as(target)
        # Fallback to broadcast; may raise a clear error if incompatible
        return x.view(1, *([1] * (target.dim() - 1))).expand_as(target)
    # If shapes already match or broadcastable, expand
    try:
        return x.expand_as(target)
    except Exception:
        return x


class MultiplierOp:
    """Dynamic multiplier connection that integrates per-edge states.

    Uses the same steady-state dynamics as the `MultiplierLayer`:
        ds/dt = gamma_plus * (g(phi + phi_y) - g(phi - phi_y))

    where `phi` is the upstream state (per edge), and `phi_y` is the programmed
    weight per edge derived from the connection matrix entries.
    """

    @staticmethod
    def layerwise(
        s_hist: torch.Tensor,
        J_eff: torch.Tensor,
        edge_src_idx: torch.Tensor,
        edge_dst_idx: torch.Tensor,
        *,
        dt: torch.Tensor | float,
        gamma_plus: torch.Tensor | float,
        gamma_minus: torch.Tensor | float,
        bias_current: torch.Tensor | float,
        source_func_key: str = "RateArray",
        phi_y_add: float = 0.0,
        j_in: torch.Tensor | float = 1.0,
        j_out: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Integrate multiplier edges over full sequence and scatter to [B,T,D].

        Args:
            s_hist: [B, T, F] upstream states per source layer.
            J_eff: [D, F] effective phi_y per destination-source.
            edge_src_idx: [E] source indices for edges.
            edge_dst_idx: [E] destination indices for edges.
            dt: scalar time step.
            gamma_plus: scalar or per-edge drive gain.
            gamma_minus: scalar or per-edge leak/damping term.
            bias_current: scalar or per-edge bias current.
            source_func_key: key into SOURCE_FUNCTIONS.
            j_in: Input coupling gain - scales upstream state (default: 1.0)
            j_out: Output coupling gain - scales edge state output (default: 1.0)

        Returns:
            [B, T, D] phi contribution for the destination layer.

        """
        B, T, F = s_hist.shape
        D, FJ = J_eff.shape
        if F != FJ:
            msg = f"Source dim mismatch: s_hist has F={F}, J has F={FJ}."
            raise ValueError(msg)
        device = s_hist.device
        dtype = s_hist.dtype
        # Gather per-edge inputs
        if edge_src_idx.numel() == 0:
            return torch.zeros(B, T, D, device=device, dtype=dtype)
        x_e = s_hist[..., edge_src_idx]  # [B,T,E]
        # Scale input by j_in
        if not isinstance(j_in, torch.Tensor):
            j_in = torch.as_tensor(float(j_in), dtype=dtype, device=device)
        else:
            j_in = j_in.to(device=device, dtype=dtype)
        x_e = x_e * j_in
        phi_y_e = J_eff[edge_dst_idx, edge_src_idx]  # [E]
        phi_y_e = phi_y_e.to(device=device, dtype=dtype).view(1, 1, -1).expand_as(x_e)
        if phi_y_add:
            phi_y_e = phi_y_e + torch.as_tensor(float(phi_y_add), dtype=dtype, device=device).view(1, 1, 1).expand_as(phi_y_e)

        # Use the exact same dynamics as MultiplierLayer (lazy import to avoid circular dependency)
        from soen_toolkit.core.layers.physical.dynamics import MultiplierDynamics

        sf = _get_source_function(source_func_key)
        dynamics = MultiplierDynamics(source_function=sf)

        # Forward Euler integration over time
        if not isinstance(dt, torch.Tensor):
            dt = torch.as_tensor(float(dt), dtype=dtype, device=device)
        else:
            dt = dt.to(device=device, dtype=dtype)

        # Track edge state per timestep
        s_e = torch.zeros(B, x_e.shape[-1], device=device, dtype=dtype)  # [B,E]
        out_e = torch.empty(B, T, x_e.shape[-1], device=device, dtype=dtype)

        for t in range(T):
            # Build params dict for this timestep
            params = {
                "phi_y": phi_y_e[:, t, :],  # [B,E]
                "gamma_plus": _expand_param_like(torch.as_tensor(gamma_plus), x_e[:, t, :]),
                "gamma_minus": _expand_param_like(torch.as_tensor(gamma_minus), x_e[:, t, :]),
                "bias_current": _expand_param_like(torch.as_tensor(bias_current), x_e[:, t, :]),
            }
            # Call MultiplierDynamics directly
            ds_dt = dynamics(s_e, x_e[:, t, :], params)  # [B,E]
            s_e = s_e + dt * ds_dt
            out_e[:, t, :] = s_e

        # Scatter-add edges into destination dimension
        phi_dst = torch.zeros(B, T, D, device=device, dtype=dtype)
        phi_dst.index_add_(dim=2, index=edge_dst_idx, source=out_e)
        # Scale output by j_out
        if not isinstance(j_out, torch.Tensor):
            j_out = torch.as_tensor(float(j_out), dtype=dtype, device=device)
        else:
            j_out = j_out.to(device=device, dtype=dtype)
        return phi_dst * j_out

    @staticmethod
    def step(
        s_src: torch.Tensor,
        J_eff: torch.Tensor,
        edge_src_idx: torch.Tensor,
        edge_dst_idx: torch.Tensor,
        edge_state: torch.Tensor,
        *,
        dt: torch.Tensor | float,
        gamma_plus: torch.Tensor | float,
        gamma_minus: torch.Tensor | float,
        bias_current: torch.Tensor | float,
        source_func_key: str = "RateArray",
        phi_y_add: float = 0.0,
        j_in: torch.Tensor | float = 1.0,
        j_out: torch.Tensor | float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance multiplier edges by one step and scatter to [B,D].

        Args:
            s_src: [B, F]
            J_eff: [D, F]
            edge_src_idx, edge_dst_idx: [E]
            edge_state: [B, E] current per-edge states
            dt: time step
            gamma_plus: drive gain
            gamma_minus: leak/damping term
            bias_current: bias current
            source_func_key: key into SOURCE_FUNCTIONS
            j_in: Input coupling gain - scales upstream state (default: 1.0)
            j_out: Output coupling gain - scales edge state output (default: 1.0)
        Returns:
            (phi_t: [B, D], new_edge_state: [B, E])

        """
        B, F = s_src.shape
        D, FJ = J_eff.shape
        if F != FJ:
            msg = f"Source dim mismatch: s_src has F={F}, J has F={FJ}."
            raise ValueError(msg)
        device = s_src.device
        dtype = s_src.dtype

        if edge_src_idx.numel() == 0:
            return (
                torch.zeros(B, D, device=device, dtype=dtype),
                edge_state,
            )

        # Gather inputs per edge
        x_e = s_src[:, edge_src_idx]  # [B,E]
        # Scale input by j_in
        if not isinstance(j_in, torch.Tensor):
            j_in = torch.as_tensor(float(j_in), dtype=dtype, device=device)
        else:
            j_in = j_in.to(device=device, dtype=dtype)
        x_e = x_e * j_in
        phi_y_e = J_eff[edge_dst_idx, edge_src_idx].to(device=device, dtype=dtype)  # [E]
        phi_y_e = phi_y_e.view(1, -1).expand_as(x_e)
        if phi_y_add:
            phi_y_e = phi_y_e + torch.as_tensor(float(phi_y_add), dtype=dtype, device=device).view(1, 1).expand_as(x_e)

        # Use the exact same dynamics as MultiplierLayer (lazy import to avoid circular dependency)
        from soen_toolkit.core.layers.physical.dynamics import MultiplierDynamics

        sf = _get_source_function(source_func_key)
        dynamics = MultiplierDynamics(source_function=sf)

        # Build params dict
        params = {
            "phi_y": phi_y_e,  # [B,E]
            "gamma_plus": _expand_param_like(torch.as_tensor(gamma_plus), x_e),
            "gamma_minus": _expand_param_like(torch.as_tensor(gamma_minus), x_e),
            "bias_current": _expand_param_like(torch.as_tensor(bias_current), x_e),
        }
        # Call MultiplierDynamics directly
        ds_dt = dynamics(edge_state, x_e, params)  # [B,E]

        if not isinstance(dt, torch.Tensor):
            dt = torch.as_tensor(float(dt), dtype=dtype, device=device)
        else:
            dt = dt.to(device=device, dtype=dtype)
        s_next = edge_state + dt * ds_dt  # [B,E]

        # Scatter to destination dimension
        phi_t = torch.zeros(B, D, device=device, dtype=dtype)
        phi_t.index_add_(dim=1, index=edge_dst_idx, source=s_next)
        # Scale output by j_out
        if not isinstance(j_out, torch.Tensor):
            j_out = torch.as_tensor(float(j_out), dtype=dtype, device=device)
        else:
            j_out = j_out.to(device=device, dtype=dtype)
        return phi_t * j_out, s_next


class MultiplierNOCCOp:
    """Dynamic multiplier v2 connection with dual SQUID states and aggregated output.

    Uses the new flux collection mechanism with:
    - Two SQUID states per edge: s1_ij and s2_ij
    - One aggregated output state per destination node: m_i

    ODEs:
        (beta + 2*N*beta_out) * dot_m_i = sum_j(g1_ij + g2_ij) - alpha * m_i
        beta * dot_s1_ij = g1_ij - beta_out * dot_m_i - alpha * s1_ij
        beta * dot_s2_ij = g2_ij - beta_out * dot_m_i - alpha * s2_ij
    """

    @staticmethod
    def layerwise(
        s_hist: torch.Tensor,
        J_eff: torch.Tensor,
        edge_src_idx: torch.Tensor,
        edge_dst_idx: torch.Tensor,
        *,
        dt: torch.Tensor | float,
        alpha: torch.Tensor | float,
        beta: torch.Tensor | float,
        beta_out: torch.Tensor | float,
        bias_current: torch.Tensor | float,
        j_in: torch.Tensor | float = 1.0,
        j_out: torch.Tensor | float = 1.0,
        source_func_key: str = "RateArray",
        phi_y_add: float = 0.0,
    ) -> torch.Tensor:
        """Integrate multiplier v2 edges over full sequence and scatter to [B,T,D].

        Args:
            s_hist: [B, T, F] upstream states per source layer
            J_eff: [D, F] effective phi_y per destination-source
            edge_src_idx: [E] source indices for edges
            edge_dst_idx: [E] destination indices for edges
            dt: scalar time step
            alpha: Dimensionless resistance
            beta: Inductance of incoming branches
            beta_out: Inductance of output branch
            bias_current: Bias current
            j_in: Input coupling gain - scales upstream state (default: 1.0)
            j_out: Output coupling gain - scales m state output (default: 1.0)
            source_func_key: key into SOURCE_FUNCTIONS

        Returns:
            [B, T, D] phi contribution for the destination layer (m state scaled by j_out)
        """
        B, T, F = s_hist.shape
        D, FJ = J_eff.shape
        if F != FJ:
            msg = f"Source dim mismatch: s_hist has F={F}, J has F={FJ}."
            raise ValueError(msg)
        device = s_hist.device
        dtype = s_hist.dtype

        # Gather per-edge inputs
        if edge_src_idx.numel() == 0:
            return torch.zeros(B, T, D, device=device, dtype=dtype)

        E = edge_src_idx.shape[0]
        x_e = s_hist[..., edge_src_idx]  # [B, T, E]
        # Scale input by j_in
        if not isinstance(j_in, torch.Tensor):
            j_in = torch.as_tensor(float(j_in), dtype=dtype, device=device)
        else:
            j_in = j_in.to(device=device, dtype=dtype)
        x_e = x_e * j_in
        phi_y_e = J_eff[edge_dst_idx, edge_src_idx]  # [E]
        phi_y_e = phi_y_e.to(device=device, dtype=dtype).view(1, 1, -1).expand_as(x_e)
        if phi_y_add:
            phi_y_e = phi_y_e + torch.as_tensor(float(phi_y_add), dtype=dtype, device=device).view(1, 1, 1).expand_as(x_e)

        # Use MultiplierNOCCDynamics
        sf = _get_source_function(source_func_key)

        # Convert parameters to tensors
        if not isinstance(dt, torch.Tensor):
            dt = torch.as_tensor(float(dt), dtype=dtype, device=device)
        else:
            dt = dt.to(device=device, dtype=dtype)

        # Parameters are scalar or per-destination, not per-edge
        # Convert to tensors and ensure correct shape for broadcasting
        alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
        beta_t = torch.as_tensor(beta, dtype=dtype, device=device)
        beta_out_t = torch.as_tensor(beta_out, dtype=dtype, device=device)
        bc_t = torch.as_tensor(bias_current, dtype=dtype, device=device)

        # Compute fan-in per destination node
        fan_in = torch.zeros(D, device=device, dtype=torch.long)
        fan_in.index_add_(0, edge_dst_idx, torch.ones_like(edge_dst_idx))
        fan_in = fan_in.float()  # [D]

        # Track edge states and destination states
        s1_e = torch.zeros(B, E, device=device, dtype=dtype)  # [B, E]
        s2_e = torch.zeros(B, E, device=device, dtype=dtype)  # [B, E]
        m_d = torch.zeros(B, D, device=device, dtype=dtype)  # [B, D]
        out_m = torch.empty(B, T, D, device=device, dtype=dtype)

        for t in range(T):
            # Compute phi_a and phi_b for edges
            phi_x_e = x_e[:, t, :]  # [B, E]
            phi_y_et = phi_y_e[:, t, :]  # [B, E]
            phi_a = phi_x_e + phi_y_et
            phi_b = phi_x_e - phi_y_et

            # Compute source functions
            if getattr(sf, "uses_squid_current", False):
                squid_current_1 = bc_t - s1_e
                squid_current_2 = -bc_t + s2_e
                # Use soft_abs() (smooth approximation of abs) to handle negative squid currents.
                # RateArray only supports positive currents (0.95 to 2.5). soft_abs provides
                # a continuous derivative which is important for numerical stability in
                # Forward Euler integration at different dt values.
                g1 = sf.g(phi_a, squid_current=soft_abs(squid_current_1))
                g2 = sf.g(phi_b, squid_current=soft_abs(squid_current_2))
            else:
                g1 = sf.g(phi_a, squid_current=bc_t)
                g2 = sf.g(phi_b, squid_current=-bc_t)

            # Aggregate g1 - g2 per destination node (opposite orientation)
            g_sum = g1 - g2  # [B, E]
            g_agg = torch.zeros(B, D, device=device, dtype=dtype)
            g_agg.index_add_(1, edge_dst_idx, g_sum)  # [B, D]

            # Compute dot_m per destination node
            # (beta + 2*N*beta_out) * dot_m_i = g_agg - alpha * m_d
            fan_in_bc = fan_in.view(1, -1)  # [1, D]
            beta_eff_d = beta_t + 2 * fan_in_bc * beta_out_t  # scalar/[D] + [1,D] * scalar/[D] -> broadcasts to [1,D] or [B,D]
            dot_m = (g_agg - alpha_t * m_d) / beta_eff_d  # [B, D]

            # Gather dot_m for each edge
            dot_m_e = dot_m[:, edge_dst_idx]  # [B, E]

            # Compute dot_s1 and dot_s2
            # beta * dot_s1 = g1 - beta_out * dot_m_e - alpha * s1
            # beta * dot_s2 = g2 - beta_out * dot_m_e - alpha * s2
            # Parameters are scalar and broadcast to [B, E]
            dot_s1 = (g1 - beta_out_t * dot_m_e - alpha_t * s1_e) / beta_t  # [B, E]
            dot_s2 = (g2 - beta_out_t * dot_m_e - alpha_t * s2_e) / beta_t  # [B, E]

            # Update states with Forward Euler
            s1_e = s1_e + dt * dot_s1
            s2_e = s2_e + dt * dot_s2
            m_d = m_d + dt * dot_m

            out_m[:, t, :] = m_d

        # Output is m state scaled by j_out, which for NOCC is fixed, and not fan-in dependent. Unlike WICC.
        if not isinstance(j_out, torch.Tensor):
            j_out = torch.as_tensor(float(j_out), dtype=dtype, device=device)
        else:
            j_out = j_out.to(device=device, dtype=dtype)

        return out_m * j_out

    @staticmethod
    def step(
        s_src: torch.Tensor,
        J_eff: torch.Tensor,
        edge_src_idx: torch.Tensor,
        edge_dst_idx: torch.Tensor,
        edge_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        dt: torch.Tensor | float,
        alpha: torch.Tensor | float,
        beta: torch.Tensor | float,
        beta_out: torch.Tensor | float,
        bias_current: torch.Tensor | float,
        j_in: torch.Tensor | float = 1.0,
        j_out: torch.Tensor | float = 1.0,
        source_func_key: str = "RateArray",
        phi_y_add: float = 0.0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Advance multiplier v2 edges by one step and scatter to [B,D].

        Args:
            s_src: [B, F]
            J_eff: [D, F]
            edge_src_idx, edge_dst_idx: [E]
            edge_state: Tuple of (s1_e [B,E], s2_e [B,E], m_d [B,D])
            dt, alpha, beta, beta_out, bias_current: parameters
            j_in: Input coupling gain - scales upstream state (default: 1.0)
            j_out: Output coupling gain - scales m state output (default: 1.0)
            source_func_key: key into SOURCE_FUNCTIONS

        Returns:
            (phi_t: [B, D] scaled by j_out, new_edge_state: tuple of [B,E], [B,E], [B,D])
        """
        B, F = s_src.shape
        D, FJ = J_eff.shape
        if F != FJ:
            msg = f"Source dim mismatch: s_src has F={F}, J has F={FJ}."
            raise ValueError(msg)
        device = s_src.device
        dtype = s_src.dtype

        if edge_src_idx.numel() == 0:
            return (
                torch.zeros(B, D, device=device, dtype=dtype),
                edge_state,
            )

        s1_e, s2_e, m_d = edge_state

        # Gather inputs per edge
        x_e = s_src[:, edge_src_idx]  # [B, E]
        # Scale input by j_in
        if not isinstance(j_in, torch.Tensor):
            j_in = torch.as_tensor(float(j_in), dtype=dtype, device=device)
        else:
            j_in = j_in.to(device=device, dtype=dtype)
        x_e = x_e * j_in
        phi_y_e = J_eff[edge_dst_idx, edge_src_idx].to(device=device, dtype=dtype)  # [E]
        phi_y_e = phi_y_e.view(1, -1).expand_as(x_e)
        if phi_y_add:
            phi_y_e = phi_y_e + torch.as_tensor(float(phi_y_add), dtype=dtype, device=device).view(1, 1).expand_as(x_e)

        # Use MultiplierNOCCDynamics
        sf = _get_source_function(source_func_key)

        # Convert parameters to tensors (scalar or per-destination)
        alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
        beta_t = torch.as_tensor(beta, dtype=dtype, device=device)
        beta_out_t = torch.as_tensor(beta_out, dtype=dtype, device=device)
        bc_t = torch.as_tensor(bias_current, dtype=dtype, device=device)

        # Compute fan-in per destination node
        fan_in = torch.zeros(D, device=device, dtype=torch.long)
        fan_in.index_add_(0, edge_dst_idx, torch.ones_like(edge_dst_idx))
        fan_in = fan_in.float()  # [D]

        # Compute phi_a and phi_b
        phi_a = x_e + phi_y_e
        phi_b = x_e - phi_y_e

        # Compute source functions
        if getattr(sf, "uses_squid_current", False):
            squid_current_1 = bc_t - s1_e
            squid_current_2 = -bc_t + s2_e  # Fixed: must match layerwise and layer dynamics
            # Use soft_abs() for smooth derivative (RateArray is symmetric around I=0)
            g1 = sf.g(phi_a, squid_current=soft_abs(squid_current_1))
            g2 = sf.g(phi_b, squid_current=soft_abs(squid_current_2))
        else:
            g1 = sf.g(phi_a, squid_current=bc_t)
            g2 = sf.g(phi_b, squid_current=-bc_t)

        # Aggregate g1 - g2 per destination node (opposite orientation)
        g_sum = g1 - g2  # [B, E]
        g_agg = torch.zeros(B, D, device=device, dtype=dtype)
        g_agg.index_add_(1, edge_dst_idx, g_sum)  # [B, D]

        # Compute dot_m per destination node
        fan_in_bc = fan_in.view(1, -1)  # [1, D]
        beta_eff_d = beta_t + 2 * fan_in_bc * beta_out_t  # scalar/[D] broadcasts to [1,D]
        dot_m = (g_agg - alpha_t * m_d) / beta_eff_d  # [B, D]

        # Gather dot_m for each edge
        dot_m_e = dot_m[:, edge_dst_idx]  # [B, E]

        # Compute dot_s1 and dot_s2
        # Parameters are scalar and broadcast to [B, E]
        dot_s1 = (g1 - beta_out_t * dot_m_e - alpha_t * s1_e) / beta_t  # [B, E]
        dot_s2 = (g2 - beta_out_t * dot_m_e - alpha_t * s2_e) / beta_t  # [B, E]

        # Update states
        if not isinstance(dt, torch.Tensor):
            dt = torch.as_tensor(float(dt), dtype=dtype, device=device)
        else:
            dt = dt.to(device=device, dtype=dtype)

        s1_next = s1_e + dt * dot_s1
        s2_next = s2_e + dt * dot_s2
        m_next = m_d + dt * dot_m

        # Output is m state scaled by j_out
        if not isinstance(j_out, torch.Tensor):
            j_out_t = torch.as_tensor(float(j_out), dtype=dtype, device=device)
        else:
            j_out_t = j_out.to(device=device, dtype=dtype)

        phi_out = m_next * j_out_t
        return phi_out, (s1_next, s2_next, m_next)


def parse_connection_config(raw_params: dict) -> tuple[str, dict]:
    """Extract connection mode and parameters from connection config dict.

    Single source of truth for mode-first design with simplified API.
    Supports new 'connection_params' dict and legacy 'dynamic' block for backward compatibility.

    Args:
        raw_params: Raw connection parameters dict (may contain 'mode', 'connection_params', etc.)

    Returns:
        (mode, params) where mode is "fixed", "WICC", or "NOCC"
        and params contains all relevant parameters with defaults applied.

    Modes:
        - "fixed": Static matrix multiplication (default)
        - "WICC": With Collection Coil (v1 multiplier, single edge state)
        - "NOCC": No Collection Coil (v2 multiplier, dual SQUID states)

    Example:
        >>> mode, params = parse_connection_config({"mode": "NOCC", "connection_params": {"alpha": 1.5}})
        >>> mode
        'NOCC'
        >>> params['alpha']
        1.5
        >>> params['beta']  # default
        303.85
    """
    import warnings

    # Extract mode - accept both canonical and legacy names
    raw_mode_value = raw_params.get("mode", raw_params.get("connection_mode", None))
    raw_mode = str(raw_mode_value if raw_mode_value is not None else "fixed")
    mode_was_provided = ("mode" in raw_params) or ("connection_mode" in raw_params)

    # Map legacy mode names to canonical names for backwards compatibility
    legacy_map = {
        "dynamic": None,  # Ambiguous - will infer from parameters
        "multiplier": None,  # Ambiguous - will infer from parameters
        "programmable": "WICC",
        "dynamic_v1": "WICC",
        "v1": "WICC",
        "dynamic_v2": "NOCC",
        "multiplier_v2": "NOCC",
        "v2": "NOCC",
    }

    # Check if it's a canonical name or needs mapping
    conn_mode = raw_mode
    was_ambiguous_legacy = False
    if conn_mode not in {"fixed", "WICC", "NOCC"}:
        # Try to map from legacy name
        mapped_mode = legacy_map.get(raw_mode.lower())
        if mapped_mode is not None:
            conn_mode = mapped_mode
            # Note: We silently convert for backwards compatibility
        elif raw_mode.lower() in legacy_map:
            # Ambiguous legacy name (like "dynamic") - will infer from parameters
            conn_mode = "fixed"  # Temporary placeholder for inference
            was_ambiguous_legacy = True
        else:
            # Unknown mode - raise error with suggestions
            msg = (
                f"Invalid connection mode '{raw_mode}'. "
                f"Valid modes: 'fixed', 'WICC' (With Collection Coil), 'NOCC' (No Collection Coil). "
                f"Legacy names like 'dynamic' are automatically converted based on parameters."
            )
            raise ValueError(msg)

    # Extract parameter block - prefer new 'connection_params', fall back to legacy 'dynamic'
    param_block = raw_params.get("connection_params")
    legacy_block = raw_params.get("dynamic") or raw_params.get("multiplier")

    if param_block is not None:
        # New API: connection_params
        param_cfg = dict(param_block)
    elif legacy_block is not None:
        # Legacy API: dynamic or multiplier
        param_cfg = dict(legacy_block)
        # Emit deprecation warning for legacy API
        warnings.warn(
            "Using 'dynamic' or 'multiplier' parameter blocks is deprecated. Use 'connection_params' instead. Example: mode='NOCC', connection_params={'alpha': 1.5, 'beta': 303.85}",
            DeprecationWarning,
            stacklevel=3,
        )
    else:
        param_cfg = {}

    # Infer mode from parameters when:
    # 1. Mode was not provided and a parameter block exists, OR
    # 2. An ambiguous legacy name like "dynamic" was used
    should_infer = ((not mode_was_provided) and conn_mode == "fixed" and param_cfg) or was_ambiguous_legacy

    if should_infer and param_cfg:
        v2_keys = {"alpha", "beta", "beta_out", "bias_current"}
        v1_keys = {"gamma_plus", "bias_current"}
        if any(k in param_cfg for k in v2_keys):
            conn_mode = "NOCC"  # v2
        elif any(k in param_cfg for k in v1_keys):
            conn_mode = "WICC"  # v1
        else:
            # Empty or unrecognized: default to NOCC (v2) for backwards compatibility
            conn_mode = "NOCC"

    # Build parameter dict with defaults based on mode
    if conn_mode == "WICC":
        params = {
            "source_func": param_cfg.get("source_func") or param_cfg.get("source_func_type") or "RateArray",
            "gamma_plus": param_cfg.get("gamma_plus", 0.001),
            "gamma_minus": param_cfg.get("gamma_minus", 0.001),
            "bias_current": param_cfg.get("bias_current", 2.0),
            "j_in": param_cfg.get("j_in", 0.38),
            "j_out": param_cfg.get("j_out", 0.38),  # although this should actually be fan-in computed!
            "half_flux_offset": bool(param_cfg.get("half_flux_offset", False)),
        }
    elif conn_mode == "NOCC":
        params = {
            "source_func": param_cfg.get("source_func") or param_cfg.get("source_func_type") or "RateArray",
            "alpha": param_cfg.get("alpha", 1.64053),
            "beta": param_cfg.get("beta", 303.85),
            "beta_out": param_cfg.get("beta_out", 91.156),
            "bias_current": param_cfg.get("bias_current", 2.1),
            "j_in": param_cfg.get("j_in", 0.38),
            "j_out": param_cfg.get("j_out", 0.38),
            "half_flux_offset": bool(param_cfg.get("half_flux_offset", False)),
        }
    else:
        params = {}

    return conn_mode, params


class ConnectionState:
    """Lightweight state holder for dynamic connection edge states.

    For WICC (v1): edge_state is a Tensor [B, E]
    For NOCC (v2): edge_state is a tuple (s1_e, s2_e, m_d)
    For fixed: edge_state is None
    """

    def __init__(self):
        self.edge_state = None


def apply_connection_step(
    s_src: torch.Tensor,
    weight: torch.Tensor,
    mode: str,
    params: dict,
    edge_indices: tuple[torch.Tensor, torch.Tensor],
    state: ConnectionState,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    """Apply connection for single timestep with automatic mode dispatch.

    Args:
        s_src: Source state [B, F]
        weight: Connection weight matrix [D, F]
        mode: "fixed", "WICC", or "NOCC"
        params: Parameter dict with defaults (from parse_connection_config)
        edge_indices: (src_idx, dst_idx) edge index tensors
        state: ConnectionState holding edge state (managed internally)
        dt: Time step

    Returns:
        phi_out: Output contribution [B, D]
    """
    src_idx, dst_idx = edge_indices

    if mode == "fixed" or src_idx is None or dst_idx is None:
        return StaticMatrixOp.step(s_src, weight)

    if mode == "WICC":
        # WICC (v1): With Collection Coil - single edge state
        E = int(src_idx.numel())
        B = s_src.shape[0]
        if state.edge_state is None or state.edge_state.shape != (B, E):
            state.edge_state = torch.zeros(B, E, device=s_src.device, dtype=s_src.dtype)

        phi_y_add = 0.5 if params.get("half_flux_offset", False) else 0.0
        phi_out, state.edge_state = MultiplierOp.step(
            s_src,
            weight,
            src_idx,
            dst_idx,
            state.edge_state,
            dt=dt,
            gamma_plus=params.get("gamma_plus", 0.001),
            gamma_minus=params.get("gamma_minus", 0.001),
            bias_current=params.get("bias_current", 2.0),
            j_in=params.get("j_in", 0.38),
            j_out=params.get("j_out", 0.38),
            source_func_key=params.get("source_func", "RateArray"),
            phi_y_add=phi_y_add,
        )
        return phi_out

    if mode == "NOCC":
        # NOCC (v2): No Collection Coil - dual SQUID states
        E = int(src_idx.numel())
        B = s_src.shape[0]
        D = weight.shape[0]
        if state.edge_state is None or not isinstance(state.edge_state, tuple) or state.edge_state[0].shape != (B, E) or state.edge_state[2].shape != (B, D):
            state.edge_state = (
                torch.zeros(B, E, device=s_src.device, dtype=s_src.dtype),
                torch.zeros(B, E, device=s_src.device, dtype=s_src.dtype),
                torch.zeros(B, D, device=s_src.device, dtype=s_src.dtype),
            )

        phi_y_add = 0.5 if params.get("half_flux_offset", False) else 0.0
        phi_out, state.edge_state = MultiplierNOCCOp.step(
            s_src,
            weight,
            src_idx,
            dst_idx,
            state.edge_state,
            dt=dt,
            alpha=params.get("alpha", 1.64053),
            beta=params.get("beta", 303.85),
            beta_out=params.get("beta_out", 91.156),
            bias_current=params.get("bias_current", 2.1),
            j_in=params.get("j_in", 0.38),
            j_out=params.get("j_out", 0.38),
            source_func_key=params.get("source_func", "RateArray"),
            phi_y_add=phi_y_add,
        )
        return phi_out

    # Fallback to fixed if unknown mode
    return StaticMatrixOp.step(s_src, weight)


def apply_connection_layerwise(
    s_hist: torch.Tensor,
    weight: torch.Tensor,
    mode: str,
    params: dict,
    edge_indices: tuple[torch.Tensor, torch.Tensor],
    dt: float | torch.Tensor,
) -> torch.Tensor:
    """Apply connection over full sequence with automatic mode dispatch.

    Args:
        s_hist: Source state history [B, T, F]
        weight: Connection weight matrix [D, F]
        mode: "fixed", "WICC", or "NOCC"
        params: Parameter dict with defaults (from parse_connection_config)
        edge_indices: (src_idx, dst_idx) edge index tensors
        dt: Time step

    Returns:
        phi_out: Output contribution [B, T, D]
    """
    src_idx, dst_idx = edge_indices

    if mode == "fixed" or src_idx is None or dst_idx is None:
        return StaticMatrixOp.layerwise(s_hist, weight)

    if mode == "WICC":
        phi_y_add = 0.5 if params.get("half_flux_offset", False) else 0.0
        return MultiplierOp.layerwise(
            s_hist,
            weight,
            src_idx,
            dst_idx,
            dt=dt,
            gamma_plus=params.get("gamma_plus", 0.001),
            gamma_minus=params.get("gamma_minus", 0.001),
            bias_current=params.get("bias_current", 2.0),
            j_in=params.get("j_in", 0.38),
            j_out=params.get("j_out", 0.38),
            source_func_key=params.get("source_func", "RateArray"),
            phi_y_add=phi_y_add,
        )

    if mode == "NOCC":
        phi_y_add = 0.5 if params.get("half_flux_offset", False) else 0.0
        return MultiplierNOCCOp.layerwise(
            s_hist,
            weight,
            src_idx,
            dst_idx,
            dt=dt,
            alpha=params.get("alpha", 1.64053),
            beta=params.get("beta", 303.85),
            beta_out=params.get("beta_out", 91.156),
            bias_current=params.get("bias_current", 2.1),
            j_in=params.get("j_in", 0.38),
            j_out=params.get("j_out", 0.38),
            source_func_key=params.get("source_func", "RateArray"),
            phi_y_add=phi_y_add,
        )

    # Fallback to fixed if unknown mode
    return StaticMatrixOp.layerwise(s_hist, weight)


__all__ = [
    "MultiplierOp",
    "MultiplierNOCCOp",
    "StaticMatrixOp",
    "build_edge_index",
    "parse_connection_config",
    "ConnectionState",
    "apply_connection_step",
    "apply_connection_layerwise",
]
