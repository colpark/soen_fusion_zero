from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .noise_jax import NoiseConfig
    from .topology_arrays import TopologyArrays

from soen_toolkit.core.layer_registry import is_multiplier_nocc
from soen_toolkit.core.layers.common.metadata import LAYER_PARAM_CONFIGS

from .forward_trace import ForwardTrace
from .layers_jax import (
    GRULayerJAX,
    GRUParamsJAX,
    LeakyGRULayerJAX,
    LeakyGRUParamsJAX,
    LinearLayerJAX,
    LSTMLayerJAX,
    LSTMParamsJAX,
    MinGRULayerJAX,
    MinGRUParamsJAX,
    MultiplierLayerJAX,
    MultiplierNOCCLayerJAX,
    MultiplierNOCCParamsJAX,
    MultiplierParamsJAX,
    NonLinearLayerJAX,
    NonLinearParamsJAX,
    ScalingLayerJAX,
    ScalingParamsJAX,
    SingleDendriteLayerJAX,
    SingleDendriteParamsJAX,
    SoftmaxLayerJAX,
    SoftmaxParamsJAX,
    SomaLayerJAX,
    SynapseLayerJAX,
    soft_abs_jax,
)


@dataclass(eq=False)
class LayerSpec:
    layer_id: int
    kind: str  # only "Multiplier" supported initially
    dim: int
    params: dict[str, jax.Array]
    internal_J: jax.Array | None
    source_key: str | None = None
    internal_mode: str = "fixed"  # "fixed" | "dynamic" | "dynamic_v2" # is it even called this anmore??
    internal_dynamic_params: dict[str, float] | None = None
    learnable_params: dict[str, bool] | None = None  # Which params are trainable
    internal_J_learnable: bool = True  # Whether internal_J connection is learnable
    internal_mask: jax.Array | None = None  # Structural mask for internal connections [D_to, D_from]
    # Optional: non-smooth layer configuration (e.g., surrogate gradients for spiking)
    surrogate_kind: str | None = None
    surrogate_params: dict[str, float] | None = None


@dataclass(eq=False)
class ConnectionSpec:
    from_layer: int
    to_layer: int
    J: jax.Array
    mask: jax.Array | None = None
    index: int = 0  # position in connections list
    mode: str = "fixed"  # "fixed" | "dynamic" | "dynamic_v2"
    source_key: str | None = None  # e.g., "RateArray" or Heaviside
    # V1 dynamic parameters
    gamma_plus: float = 1e-3
    gamma_minus: float = 1e-3
    bias_current: float = 2.0
    j_in: float = 0.38
    j_out: float | jax.Array = 0.38
    # V2 dynamic parameters
    alpha: float | None = None
    beta: float | None = None
    beta_out: float | None = None
    # Note: For V2 dynamic connections (NOCC), bias_current has different defaults than WICC.
    # The bias_current field above (default 2.0) is used for WICC. For NOCC, check for None
    # and use 2.1 as the default in the step functions.
    # Optional Torch parity flag: add +0.5 to phi_y (per-edge) when enabled
    half_flux_offset: bool = False
    learnable: bool = True  # Whether this connection is learnable


@dataclass(eq=False)
class JAXModel:
    dt: float
    layers: list[LayerSpec]
    connections: list[ConnectionSpec]
    connection_constraints: dict[str, dict[str, float]] | None = None
    connection_constraint_min_matrices: dict[str, jax.Array] | None = None
    connection_constraint_max_matrices: dict[str, jax.Array] | None = None
    network_evaluation_method: str = "layerwise"
    # New: propagate input semantics from SimulationConfig ("flux" | "state")
    input_type: str = "flux"
    # Connection noise settings: dict mapping "J_{from}_to_{to}" -> NoiseConfig
    # Applied to connection weights J during forward pass
    connection_noise_settings: dict[str, NoiseConfig] | None = None
    # Caches to reduce Python work per call (populated by prepare(), before JIT)
    _cached_impls: list | None = None
    _cached_inbound: dict[int, list[ConnectionSpec]] | None = None
    _multiplier_indices: list[int] | None = None
    _connidx_to_multidx: dict[int, int] | None = None
    # Fast-path caches
    _topology_arrays: Any | None = None
    _use_fast_layerwise: bool = False
    _use_fast_gs: bool = False
    _use_fast_jacobi: bool = False
    # Optional connection override array for training: [E,D_max,F_max]
    _conn_override: jax.Array | None = None
    # Optional internal connection overrides for training: dict[layer_id -> internal_J]
    _internal_conn_override: dict[int, jax.Array] | None = None
    # Optional layer parameter override arrays for training updates
    _layer_param_override: tuple[jax.Array, ...] | None = None

    def _build_layer(self, spec: LayerSpec):
        # Build JAX layer implementation based on kind
        from .source_functions_jax import (
            HeavisideStateDepJAX,
            RateArrayJAX,
            ReLUJAX,
            SimpleGELUJAX,
            TanhGauss1p7IBFitJAX,
            TanhJAX,
            TeLUJAX,
        )

        kind = spec.kind.lower()
        # Select source function implementation
        source_key = (spec.source_key or "Heaviside_state_dep").lower()
        source_impl: Any
        if source_key == "ratearray":
            source_impl = RateArrayJAX()
        elif source_key in ("heaviside_state_dep", "heaviside_fit_state_dep", "heaviside"):
            source_impl = HeavisideStateDepJAX()
        elif source_key == "tanh":
            source_impl = TanhJAX()
        elif source_key == "telu":
            source_impl = TeLUJAX()
        elif source_key in ("simplegelu", "simple_gelu", "gelu"):
            source_impl = SimpleGELUJAX()
        elif source_key == "relu":
            source_impl = ReLUJAX()
        elif source_key == "tanhgauss1p7ibfit":
            source_impl = TanhGauss1p7IBFitJAX()
        else:
            # Fail fast for unknown source functions instead of silent fallback
            msg = f"Unknown source function key '{spec.source_key}' for JAX. Supported: RateArray, Heaviside_state_dep, Tanh, TeLU, SimpleGELU, ReLU, TanhGauss1p7IBFit"
            raise ValueError(msg)

        if kind == "multiplier":
            return MultiplierLayerJAX(
                dim=spec.dim, dt=self.dt, source=source_impl, internal_mode=spec.internal_mode, internal_dynamic_params=spec.internal_dynamic_params, internal_mask=getattr(spec, "internal_mask", None)
            )
        if kind in ("multiplierv2", "multiplier_v2", "multipliernocc", "nocc"):
            return MultiplierNOCCLayerJAX(
                dim=spec.dim, dt=self.dt, source=source_impl, internal_mode=spec.internal_mode, internal_dynamic_params=spec.internal_dynamic_params, internal_mask=getattr(spec, "internal_mask", None)
            )
        if kind in ("singledendrite", "single_dendrite", "dendrite"):
            # For SingleDendrite, the source function is typically Heaviside or RateArray
            # But we need to ensure we're passing the correct implementation
            # SingleDendriteLayerJAX expects HeavisideStateDepJAX interface for g()
            # RateArrayJAX also implements g(), so it should be compatible
            return SingleDendriteLayerJAX(
                dim=spec.dim, dt=self.dt, source=source_impl, internal_mode=spec.internal_mode, internal_dynamic_params=spec.internal_dynamic_params, internal_mask=getattr(spec, "internal_mask", None)
            )
        if kind == "soma":
            # Soma: reuse SingleDendrite-like dynamics + spike head (surrogate gradients)
            return SomaLayerJAX(
                dim=spec.dim,
                dt=self.dt,
                source=source_impl,
                surrogate_kind=spec.surrogate_kind or "triangle",
                surrogate_params=spec.surrogate_params or {},
                internal_mode=spec.internal_mode,
                internal_dynamic_params=spec.internal_dynamic_params,
                internal_mask=getattr(spec, "internal_mask", None),
            )
        if kind == "synapse":
            return SynapseLayerJAX(dim=spec.dim, dt=self.dt)
        if kind == "mingru":
            return MinGRULayerJAX(dim=spec.dim, dt=self.dt, internal_mask=getattr(spec, "internal_mask", None))

        if kind == "gru":
            return GRULayerJAX(dim=spec.dim, dt=self.dt, internal_mask=getattr(spec, "internal_mask", None))
        if kind == "leakygru":
            return LeakyGRULayerJAX(dim=spec.dim, dt=self.dt, internal_mask=getattr(spec, "internal_mask", None))

        if kind == "lstm":
            return LSTMLayerJAX(dim=spec.dim, dt=self.dt, internal_mask=getattr(spec, "internal_mask", None))
        if kind == "linear":
            return LinearLayerJAX(dim=spec.dim, dt=self.dt)
        if kind == "nonlinear":
            # use source_key if provided; default to Tanh
            return NonLinearLayerJAX(dim=spec.dim, dt=self.dt, source_key=(spec.source_key or "Tanh"))
        if kind in ("scalinglayer", "scaling"):
            return ScalingLayerJAX(dim=spec.dim, dt=self.dt)
        if kind == "softmax":
            return SoftmaxLayerJAX(dim=spec.dim, dt=self.dt)
        msg = f"Unsupported layer kind for JAXModel: {spec.kind}"
        raise NotImplementedError(msg)

    def _ensure_cache(self) -> None:
        if self._cached_impls is None:
            sorted_layers = sorted(self.layers, key=lambda layer: layer.layer_id)
            self._cached_impls = [self._build_layer(s) for s in sorted_layers]
            self._layer_id_to_pos = {spec.layer_id: idx for idx, spec in enumerate(sorted_layers)}
        if self._cached_inbound is None:
            inbound: dict[int, list[ConnectionSpec]] = {layer.layer_id: [] for layer in self.layers}
            for c in self.connections:
                inbound[c.to_layer].append(c)
            self._cached_inbound = inbound
        # Map each connection to its position in self.connections for edge-state indexing
        if getattr(self, "_conn_pos_map", None) is None:
            pos_map: dict[tuple[int, int, int], int] = {}
            for pos, c in enumerate(self.connections):
                pos_map[(c.from_layer, c.to_layer, getattr(c, "index", pos))] = pos
            self._conn_pos_map = pos_map
        if self._multiplier_indices is None or self._connidx_to_multidx is None:
            mult_idx_list: list[int] = []
            connidx_to_m: dict[int, int] = {}
            for i, c in enumerate(self.connections):
                cat = self._mode_category(c.mode)
                if cat == "wicc":
                    connidx_to_m[i] = len(mult_idx_list)
                    mult_idx_list.append(i)
            self._multiplier_indices = mult_idx_list
            self._connidx_to_multidx = connidx_to_m
    # flag
    def prepare(self) -> JAXModel:
        """Pre-build layer implementations and inbound maps on the host.

        Call this once after constructing the JAXModel and before any jitted
        calls. This avoids constructing source-function tables (e.g., RateArray)
        under JIT tracing, which can lead to tracer leaks.
        """
        self._ensure_cache()
        # Build topology arrays for future fast kernels (no behavior change)
        try:
            from .topology_arrays import build_topology_arrays

            self._topology_arrays = build_topology_arrays(self)
        except ValueError as e:
            # Specific errors like array-valued j_out mean fast path not supported
            import logging
            logger = logging.getLogger(__name__)
            if "array-valued j_out" in str(e) or "array-valued j_in" in str(e):
                logger.info(f"[JAX] Fast path not available: {e}")
                logger.info("[JAX] Model will use slow path (per-connection iteration). This is correct but slower.")
            else:
                logger.warning(f"[JAX] Failed to build topology arrays: {e}")
            self._topology_arrays = None
        except Exception as e:
            # Log warning but keep robust if optional module fails
            import logging
            import traceback
            logger = logging.getLogger(__name__)
            logger.warning(f"[JAX] Failed to build topology arrays for fast path. Training will be slower. Error: {e}")
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            self._topology_arrays = None
        return self

    def _layer_params(
        self,
        spec: LayerSpec,
        batch: int,
        internal_conn_override: dict[int, jax.Array] | None = None,
        layer_param_override: tuple[jax.Array, ...] | None = None,
    ):
        """Build layer parameters for forward pass.

        Args:
            spec: Layer specification
            batch: Batch size
            internal_conn_override: Optional internal connection override dict (passed functionally to avoid tracer leaks)
            layer_param_override: Optional layer parameter override tuple (passed functionally to avoid tracer leaks)
        """
        # Broadcast/cast params to [B,D]; at convert-time we ensured 1D [D]
        # flag
        def bcast_1d_to_bd(x):
            if x.ndim == 1 and x.shape[0] == spec.dim:
                x = x[jnp.newaxis, :]
            elif x.ndim == 2:
                pass
            else:
                x = jnp.reshape(x, (1, spec.dim))
            return jnp.broadcast_to(x, (batch, spec.dim))

        kind = spec.kind.lower()
        override_params = None
        # Use passed override (functional style) or fall back to self (legacy)
        effective_layer_param_override = layer_param_override if layer_param_override is not None else self._layer_param_override
        if effective_layer_param_override is not None and hasattr(self, "_layer_id_to_pos"):
            pos = self._layer_id_to_pos.get(spec.layer_id)
            if pos is not None and pos < len(effective_layer_param_override):
                arr = effective_layer_param_override[pos]
                names = self._layer_param_names_for_kind(spec.kind)
                if names and arr.shape[0] == len(names):
                    override_params = {name: arr[idx_row] for idx_row, name in enumerate(names)}

        if override_params is not None:
            params = override_params
            # Override params are in log-space (from optimizer during training)
            params_in_log_space = True
        else:
            params_data = spec.params if spec.params is not None else {}
            # spec.params are in real-space (from PyTorch conversion)
            clamped = self._clamp_layer_params(spec.kind, params_data, params_in_log_space=False)
            params = clamped if clamped is not None else {}
            # Don't mutate spec.params here - it causes tracer leaks in JIT
            # The clamped params are returned and used correctly without mutation
            params_in_log_space = False

        # CRITICAL FIX: Convert log-space params to real-space for forward pass
        # During training: override params are in log-space and need exp()
        # During inference: spec.params are already in real-space, no conversion needed
        if params_in_log_space:
            canonical = self._canonical_layer_kind(spec.kind)
            configs = LAYER_PARAM_CONFIGS.get(canonical, [])
            config_map = {cfg.name: cfg for cfg in configs}
            params_real = dict(params)
            for name, arr in params.items():
                cfg = config_map.get(name)
                if cfg and getattr(cfg, "is_log_param", False):
                    # Convert from log-space to real-space via exp()
                    params_real[name] = jnp.exp(arr)
            params = params_real

        # Use internal connection override if available, otherwise use spec.internal_J
        # Internal connections are weights that should be updated during training, just like external connections
        # Check passed parameter first (functional style), then fall back to self (legacy)
        effective_internal_override = internal_conn_override if internal_conn_override is not None else self._internal_conn_override
        internal_J = spec.internal_J
        if effective_internal_override is not None and spec.layer_id in effective_internal_override:
            internal_J = effective_internal_override[spec.layer_id]
            # Verify override is being used (for debugging)
            # The override contains the updated weights from training
        if kind == "multiplier":
            phi_y = bcast_1d_to_bd(params["phi_y"])  # guaranteed by converter
            bias_current = bcast_1d_to_bd(params["bias_current"])  # guaranteed
            gamma_plus = bcast_1d_to_bd(params["gamma_plus"])  # now in real-space (exp'd)
            gamma_minus = bcast_1d_to_bd(params["gamma_minus"])  # now in real-space (exp'd)
            return MultiplierParamsJAX(
                phi_y=phi_y,
                bias_current=bias_current,
                gamma_plus=gamma_plus,
                gamma_minus=gamma_minus,
                internal_J=internal_J,
            )
        if kind in ("multiplierv2", "multiplier_v2", "multipliernocc", "nocc"):
            phi_y = bcast_1d_to_bd(params["phi_y"])  # guaranteed by converter
            alpha = bcast_1d_to_bd(params["alpha"])  # dimensionless resistance
            beta = bcast_1d_to_bd(params["beta"])  # inductance of incoming branches
            beta_out = bcast_1d_to_bd(params["beta_out"])  # inductance of output branch
            bias_current = bcast_1d_to_bd(params["bias_current"])  # bias current
            return MultiplierNOCCParamsJAX(
                phi_y=phi_y,
                alpha=alpha,
                beta=beta,
                beta_out=beta_out,
                bias_current=bias_current,
                internal_J=internal_J,
            )
        if kind in ("singledendrite", "single_dendrite", "dendrite"):
            phi_offset = bcast_1d_to_bd(params["phi_offset"])  # guaranteed by converter
            bias_current = bcast_1d_to_bd(params["bias_current"])  # guaranteed
            gamma_plus = bcast_1d_to_bd(params["gamma_plus"])  # now in real-space (exp'd)
            gamma_minus = bcast_1d_to_bd(params["gamma_minus"])  # now in real-space (exp'd)
            return SingleDendriteParamsJAX(
                phi_offset=phi_offset,
                bias_current=bias_current,
                gamma_plus=gamma_plus,
                gamma_minus=gamma_minus,
                internal_J=internal_J,
            )
        if kind == "mingru":
            # Expect PyTorch MinGRU has weight matrices: gate_proj (D,D) and hidden_proj (D,D)
            W_gate = params["W_gate"]
            W_hidden = params["W_hidden"]
            # Broadcast isn't necessary (matrices), but ensure correct dtype
            return MinGRUParamsJAX(
                W_hidden=jnp.asarray(W_hidden),
                W_gate=jnp.asarray(W_gate),
                internal_J=internal_J,
                force_sequential=(internal_J is not None),
            )
        if kind == "linear":
            # no params; build history directly from phi
            return None
        if kind == "nonlinear":
            phi_offset = params.get("phi_offset")
            bias_current = params.get("bias_current")
            # Broadcast if provided
            po = bcast_1d_to_bd(phi_offset) if phi_offset is not None else None
            bc = bcast_1d_to_bd(bias_current) if bias_current is not None else None
            return NonLinearParamsJAX(phi_offset=po, bias_current=bc)
        if kind in ("scalinglayer", "scaling"):
            scale_factor = params.get("scale_factor")
            # Broadcast [D] -> [B,D]
            sf = bcast_1d_to_bd(scale_factor) if scale_factor is not None else jnp.ones((batch, spec.dim), dtype=jnp.float32)
            return ScalingParamsJAX(scale_factor=sf)
        if kind == "softmax":
            beta = params.get("beta")
            # Broadcast [D] -> [B,D]
            bt = bcast_1d_to_bd(beta) if beta is not None else jnp.ones((batch, spec.dim), dtype=jnp.float32)
            return SoftmaxParamsJAX(beta=bt)
        if kind == "gru":
            # Expect weight_ih, weight_hh, bias_ih, bias_hh shaped like PyTorch GRU
            W_ih = params["weight_ih"]  # [3*D, D] or [3,D,D]
            W_hh = params["weight_hh"]
            b_ih = params.get("bias_ih")
            b_hh = params.get("bias_hh")

            def to_blocks(W):
                W = jnp.asarray(W)
                if W.ndim == 2 and W.shape[0] == 3 * spec.dim:
                    return jnp.stack([W[0 : spec.dim, :], W[spec.dim : 2 * spec.dim, :], W[2 * spec.dim : 3 * spec.dim, :]], axis=0)
                if W.ndim == 3 and W.shape[0] == 3:
                    return W
                # Fallback
                return jnp.stack([W, W, W], axis=0)

            # flag
            def to_bias(b):
                if b is None:
                    return jnp.zeros((3, spec.dim), dtype=jnp.float32)
                b = jnp.asarray(b)
                if b.ndim == 1 and b.shape[0] == 3 * spec.dim:
                    return jnp.stack([b[0 : spec.dim], b[spec.dim : 2 * spec.dim], b[2 * spec.dim : 3 * spec.dim]], axis=0)
                if b.ndim == 2 and b.shape[0] == 3:
                    return b
                return jnp.zeros((3, spec.dim), dtype=b.dtype)


            return GRUParamsJAX(
                weight_ih=to_blocks(W_ih),
                weight_hh=to_blocks(W_hh),
                bias_ih=to_bias(b_ih),
                bias_hh=to_bias(b_hh),
                internal_J=internal_J,
                force_sequential=True,
            )
        if kind == "leakygru":
            W_in = params["W_in"]
            W_hn = params["W_hn"]
            bias_z = params["bias_z"]
            bias_n = params["bias_n"]
            bias_r = params["bias_r"]
            return LeakyGRUParamsJAX(
                W_in=jnp.asarray(W_in),
                W_hn=jnp.asarray(W_hn),
                bias_z=jnp.asarray(bias_z),
                bias_n=jnp.asarray(bias_n),
                bias_r=jnp.asarray(bias_r),
                internal_J=internal_J,
                force_sequential=True,
            )
        if kind == "lstm":

            # Expect weight_ih, weight_hh, bias_ih, bias_hh shaped like PyTorch LSTM
            # LSTM has 4 gates: input, forget, cell, output (i, f, g, o)
            W_ih = params["weight_ih"]  # [4*D, D] or [4,D,D]
            W_hh = params["weight_hh"]
            b_ih = params.get("bias_ih")
            b_hh = params.get("bias_hh")

            def to_blocks(W):
                W = jnp.asarray(W)
                if W.ndim == 2 and W.shape[0] == 4 * spec.dim:
                    # Split [4*D, D] into 4 blocks of [D, D]
                    return jnp.stack([
                        W[0 : spec.dim, :],
                        W[spec.dim : 2 * spec.dim, :],
                        W[2 * spec.dim : 3 * spec.dim, :],
                        W[3 * spec.dim : 4 * spec.dim, :],
                    ], axis=0)
                if W.ndim == 3 and W.shape[0] == 4:
                    return W
                # Fallback
                return jnp.stack([W, W, W, W], axis=0)

            def to_bias(b):
                if b is None:
                    return jnp.zeros((4, spec.dim), dtype=jnp.float32)
                b = jnp.asarray(b)
                if b.ndim == 1 and b.shape[0] == 4 * spec.dim:
                    # Split [4*D] into 4 blocks of [D]
                    return jnp.stack([
                        b[0 : spec.dim],
                        b[spec.dim : 2 * spec.dim],
                        b[2 * spec.dim : 3 * spec.dim],
                        b[3 * spec.dim : 4 * spec.dim],
                    ], axis=0)
                if b.ndim == 2 and b.shape[0] == 4:
                    return b
                return jnp.zeros((4, spec.dim), dtype=b.dtype)

            return LSTMParamsJAX(
                weight_ih=to_blocks(W_ih),
                weight_hh=to_blocks(W_hh),
                bias_ih=to_bias(b_ih),
                bias_hh=to_bias(b_hh),
                internal_J=internal_J,
                force_sequential=True,
            )
        msg = f"Unsupported layer kind for params mapping: {spec.kind}"
        raise NotImplementedError(msg)

    # ------------------------------------------------------------------
    # Multiplier connection helpers (external connections)
    # ------------------------------------------------------------------
    # flag
    def _select_source_impl(self, key: str | None):
        from .source_functions_jax import (
            HeavisideStateDepJAX,
            RateArrayJAX,
            ReLUJAX,
            SimpleGELUJAX,
            TanhJAX,
            TeLUJAX,
        )

        k = (key or "Heaviside_state_dep").lower()
        if k == "ratearray":
            # Fail fast: RateArray initialization errors should not silently change
            # model semantics by swapping source functions.
            return RateArrayJAX()
        if k in ("heaviside_state_dep", "heaviside_fit_state_dep", "heaviside"):
            return HeavisideStateDepJAX()
        if k == "tanh":
            return TanhJAX()
        if k == "telu":
            return TeLUJAX()
        if k in ("simplegelu", "simple_gelu", "gelu"):
            return SimpleGELUJAX()
        if k == "relu":
            return ReLUJAX()
        raise ValueError(
            f"Unknown source function key '{key}' for JAX connections. "
            "Supported: RateArray, Heaviside_state_dep, Heaviside_fit_state_dep, Heaviside, Tanh, TeLU, SimpleGELU, ReLU."
        )

    def _mode_category(self, mode_str: str | None) -> str:
        """Normalize connection modes/aliases to one of: 'wicc', 'nocc', 'fixed'."""
        m = str(mode_str or "fixed").lower()
        if m in {"wicc", "dynamic", "multiplier", "programmable", "dynamic_v1", "v1"}:
            return "wicc"
        if m in {"nocc", "dynamic_v2", "multiplier_v2", "v2"}:
            return "nocc"
        return "fixed"

    def _get_conn_override(self, conn: ConnectionSpec, layer_idx: int | jax.Array, conn_offset: int = 0) -> jax.Array | None:
        """Extract connection override from _conn_override if available (legacy method).

        Args:
            conn: Connection specification
            layer_idx: Index of destination layer in sorted layers (can be traced array)
            conn_offset: Offset of this connection within the layer's inbound edges (0 for first, 1 for second, etc.)

        Returns:
            Override J matrix [D,F] or None if not available
        """
        return self._get_conn_override_func(conn, layer_idx, conn_offset, self._conn_override)

    def _get_conn_override_func(self, conn: ConnectionSpec, layer_idx: int | jax.Array, conn_offset: int, conn_override: jax.Array | None) -> jax.Array | None:
        """Extract connection override from passed array (functional style to avoid tracer leaks).

        Args:
            conn: Connection specification
            layer_idx: Index of destination layer in sorted layers (can be traced array)
            conn_offset: Offset of this connection within the layer's inbound edges
            conn_override: Override array passed functionally [E, D_max, F_max]

        Returns:
            Override J matrix [D,F] or None if not available
        """
        if conn_override is None or self._topology_arrays is None:
            return None
        topo = cast(Any, self._topology_arrays)
        # Use JAX operations that work with traced values
        start_i_val = jax.lax.index_in_dim(topo.inbound_starts, int(layer_idx), axis=0, keepdims=False)
        edge_idx = start_i_val + conn_offset
        # Use jnp.take which works with traced indices
        J_override = jnp.take(conn_override, edge_idx, axis=0)
        # Slice to actual connection dimensions
        J_override = J_override[: conn.J.shape[0], : conn.J.shape[1]]
        return J_override

    def _clamp_connections(self, conn: ConnectionSpec) -> jax.Array:
        """Apply connection-level constraints (min/max) to weight matrix."""
        constraints = None
        if self.connection_constraints:
            constraints = self.connection_constraints.get(f"J_{conn.from_layer}_to_{conn.to_layer}")
        if constraints is None:
            # Fallback to topology constraints if available
            try:
                topo = getattr(self, "_topology_arrays", None)
                if topo is not None:
                    constraints = topo.connection_constraints.get(f"J_{conn.from_layer}_to_{conn.to_layer}")
            except Exception:
                constraints = None
        if not constraints:
            return conn.J
        min_val = constraints.get("min")
        max_val = constraints.get("max")
        J = conn.J
        if min_val is not None:
            J = jnp.maximum(J, min_val)
        if max_val is not None:
            J = jnp.minimum(J, max_val)
        return J

    @staticmethod
    def _canonical_layer_kind(kind: str) -> str:
        mapping = {
            "singledendrite": "SingleDendrite",
            "single_dendrite": "SingleDendrite",
            "dendrite": "SingleDendrite",
            "multiplier": "Multiplier",
            "wicc": "Multiplier",
            "multiplierv2": "MultiplierNOCC",
            "multiplier_v2": "MultiplierNOCC",
            "multipliernocc": "MultiplierNOCC",
            "nocc": "MultiplierNOCC",
            "readout": "DendriteReadout",
            "dendritereadout": "DendriteReadout",
            "softmax": "Softmax",
        }
        return mapping.get(kind.lower(), kind)

    @staticmethod
    def _layer_param_names_for_kind(kind: str) -> tuple[str, ...]:
        k = kind.lower()
        if k in {"multiplier", "wicc"}:
            return ("phi_y", "bias_current", "gamma_plus", "gamma_minus")
        if k in {"multiplierv2", "multiplier_v2", "multipliernocc", "nocc"}:
            return ("phi_y", "bias_current", "alpha", "beta", "beta_out")
        if k in {"singledendrite", "single_dendrite", "dendrite"}:
            return ("phi_offset", "bias_current", "gamma_plus", "gamma_minus")
        if k in {"nonlinear"}:
            return ("phi_offset", "bias_current")
        if k in {"scalinglayer", "scaling"}:
            return ("scale_factor",)
        if k == "softmax":
            return ("beta",)
        return ()

    def _clamp_layer_params(self, kind: str, params: dict[str, jax.Array] | None, *, params_in_log_space: bool = True) -> dict[str, jax.Array] | None:
        """Clamp layer parameters to their configured bounds.

        Args:
            kind: Layer type (e.g., "SingleDendrite", "Multiplier")
            params: Dictionary of parameter arrays
            params_in_log_space: If True, log-space params (gamma_plus, gamma_minus) are already
                                in log-space and should be clamped directly. If False, they're in
                                real-space and need log->clamp->exp conversion.

        Returns:
            Dictionary of clamped parameters (in same space as input)
        """
        if not params:
            return params
        canonical = self._canonical_layer_kind(kind)
        configs = LAYER_PARAM_CONFIGS.get(canonical)
        if not configs:
            return params
        config_map = {cfg.name: cfg for cfg in configs}
        clamp_map = {}
        for name, cfg in config_map.items():
            arr = params.get(name)
            if arr is None:
                continue
            arr = jnp.asarray(arr)
            is_log = getattr(cfg, "is_log_param", False)

            # CRITICAL FIX: Handle params that are already in log-space (from optimizer)
            # PyTorch stores log params in log-space and clamps them there
            if is_log and params_in_log_space:
                # Params are already in log-space, clamp directly without conversion
                arr_clamped = arr
                if cfg.min_value is not None and cfg.min_value > 0:
                    arr_clamped = jnp.maximum(arr_clamped, jnp.log(cfg.min_value))
                if cfg.max_value is not None and cfg.max_value > 0 and cfg.max_value < float("inf"):
                    arr_clamped = jnp.minimum(arr_clamped, jnp.log(cfg.max_value))
                # Return in log-space (don't exp)
                clamp_map[name] = arr_clamped
            elif is_log and not params_in_log_space:
                # Legacy path: params in real-space, convert to log, clamp, convert back
                arr = jnp.log(jnp.maximum(arr, 1e-10))
                arr_clamped = arr
                if cfg.min_value is not None:
                    val = jnp.log(cfg.min_value) if cfg.min_value > 0 else -float("inf")
                    arr_clamped = jnp.maximum(arr_clamped, val)
                if cfg.max_value is not None and cfg.max_value < float("inf"):
                    val = jnp.log(cfg.max_value) if cfg.max_value > 0 else float("inf")
                    arr_clamped = jnp.minimum(arr_clamped, val)
                clamp_map[name] = jnp.exp(arr_clamped)
            else:
                # Non-log params: clamp directly in real-space
                arr_clamped = arr
                if cfg.min_value is not None:
                    arr_clamped = jnp.maximum(arr_clamped, cfg.min_value)
                if cfg.max_value is not None:
                    arr_clamped = jnp.minimum(arr_clamped, cfg.max_value)
                clamp_map[name] = arr_clamped
        if not clamp_map:
            return params
        new_params = dict(params)
        new_params.update(clamp_map)
        return new_params

    def clamp_and_apply_layer_param_arrays(
        self,
        layer_param_arrays: tuple | list | None,
        *,
        apply_to_specs: bool = True,
    ) -> tuple | list | None:
        if not layer_param_arrays:
            return layer_param_arrays
        layers_sorted = sorted(self.layers, key=lambda s: s.layer_id)
        updated_arrays: list = []
        for idx, spec in enumerate(layers_sorted):
            if idx >= len(layer_param_arrays):
                break
            arr = layer_param_arrays[idx]
            if isinstance(arr, tuple):
                updated_arrays.append(arr)
                continue
            names = self._layer_param_names_for_kind(spec.kind)
            if not names or arr.shape[0] != len(names):
                updated_arrays.append(arr)
                continue
            param_dict = {name: arr[row_idx] for row_idx, name in enumerate(names)}
            # CRITICAL: Params from optimizer are in log-space for log params
            clamped_dict = self._clamp_layer_params(spec.kind, param_dict, params_in_log_space=True) or param_dict
            if apply_to_specs:
                if spec.params is None:
                    spec.params = {}
                # Convert log-space params back to real-space before storing in spec
                # spec.params should always contain real-space values for compatibility
                canonical = self._canonical_layer_kind(spec.kind)
                configs = LAYER_PARAM_CONFIGS.get(canonical, [])
                config_map = {cfg.name: cfg for cfg in configs}
                for name in names:
                    value = clamped_dict[name]
                    cfg = config_map.get(name)
                    # If this is a log param, exp() it back to real-space for spec storage
                    if cfg and getattr(cfg, "is_log_param", False):
                        spec.params[name] = jnp.exp(value)
                    else:
                        spec.params[name] = value
            stacked = jnp.stack([clamped_dict[name] for name in names], axis=0)
            updated_arrays.append(stacked)
        if len(layer_param_arrays) > len(updated_arrays):
            updated_arrays.extend(layer_param_arrays[len(updated_arrays) :])
        return tuple(updated_arrays)

    def _multiplier_phi_layerwise(self, s_hist: jax.Array, conn: ConnectionSpec, dtype, J_override: jax.Array | None = None) -> jax.Array:
        """Compute phi contribution over time for one multiplier connection.

        s_hist: [B,T,F], conn.J: [D,F] -> returns [B,T,D]
        """
        B, _T, _F = s_hist.shape
        # Use override if provided, otherwise use conn.J
        J_base = J_override if J_override is not None else self._clamp_connections(conn)
        D, FJ = J_base.shape
        J_eff = J_base if (conn.mask is None) else (J_base * conn.mask)
        src = self._select_source_impl(conn.source_key)
        gp = jnp.asarray(conn.gamma_plus, dtype=dtype)
        gm = jnp.asarray(conn.gamma_minus, dtype=dtype)
        bc = jnp.asarray(conn.bias_current, dtype=dtype)
        dt = jnp.asarray(self.dt, dtype=dtype)

        j_in = jnp.asarray(conn.j_in, dtype=dtype)
        j_out = jnp.asarray(conn.j_out, dtype=dtype)

        def step_fn(S_prev, x_t):
            # x_t: [B, F]
            # Broadcast to [B,D,F]
            x_e = x_t[:, None, :] * j_in
            phi_y_e = jnp.broadcast_to(J_eff[None, :, :], (B, D, FJ))
            # Optional half-flux offset parity: add +0.5 to φ_y
            if bool(getattr(conn, "half_flux_offset", False)):
                phi_y_e = phi_y_e + 0.5
            phi_a = x_e + phi_y_e
            phi_b = x_e - phi_y_e
            # Broadcast gp, gm, bias_current
            gp_e = jnp.broadcast_to(gp.reshape(1, 1), (B, D))[:, :, None]
            gm_e = jnp.broadcast_to(gm.reshape(1, 1), (B, D))[:, :, None]
            bc_e = jnp.broadcast_to(bc.reshape(1, 1), (B, D))[:, :, None]
            # Source nonlinearities per edge
            # Mirror torch MultiplierDynamics squid current dependency: +/- state
            squid_current_a = bc_e - S_prev
            squid_current_b = bc_e + S_prev
            g_a = src.g(phi_a, squid_current=squid_current_a)
            g_b = src.g(phi_b, squid_current=squid_current_b)
            ds_dt = gp_e * (g_a - g_b) - gm_e * S_prev
            S_next = S_prev + dt * ds_dt
            phi_t = jnp.sum(S_next, axis=-1) * j_out  # sum over F -> [B,D]
            return S_next, phi_t

        S0 = jnp.zeros((B, D, FJ), dtype=dtype)
        # Scan over time dimension
        _, phi_seq = jax.lax.scan(step_fn, S0, s_hist.swapaxes(0, 1))  # [T,B,D]
        return phi_seq.swapaxes(0, 1)

    def _multiplier_step(
        self,
        s_src: jax.Array,
        edge_state: jax.Array,
        conn: ConnectionSpec,
        dtype,
        J_override: jax.Array | None = None,
        rng_key: jax.Array | None = None,
        conn_noise_cfg=None,
        perturb_offset: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array | None]:
        """Advance multiplier edges one step: s_src [B,F], edge_state [B,D,F].

        Args:
            s_src: Source states [B, F]
            edge_state: Current edge states [B, D, F]
            conn: Connection specification
            dtype: Data type
            J_override: Optional override for J matrix
            rng_key: JAX random key for noise generation (optional)
            conn_noise_cfg: Noise configuration for this connection (optional)
            perturb_offset: Precomputed perturbation offset (optional)

        Returns:
            Tuple of (phi_t, S_next, updated_rng_key)
        """
        B, _F = s_src.shape
        D, FJ = conn.J.shape
        # Use override if provided, otherwise use conn.J
        J_base = J_override if J_override is not None else self._clamp_connections(conn)
        J_eff = J_base if (conn.mask is None) else (J_base * conn.mask)

        # Apply noise if configured
        if conn_noise_cfg is not None and rng_key is not None:
            if conn_noise_cfg.noise is not None and conn_noise_cfg.noise.std > 0.0:
                from .noise_jax import apply_connection_noise
                rng_key, j_noise_key = jax.random.split(rng_key)
                J_eff = apply_connection_noise(j_noise_key, J_eff, conn_noise_cfg, perturb_offset)
            elif perturb_offset is not None:
                J_eff = J_eff + perturb_offset
        elif perturb_offset is not None:
            J_eff = J_eff + perturb_offset

        src = self._select_source_impl(conn.source_key)
        gp = jnp.asarray(conn.gamma_plus, dtype=dtype)
        gm = jnp.asarray(conn.gamma_minus, dtype=dtype)
        bc = jnp.asarray(conn.bias_current, dtype=dtype)
        dt = jnp.asarray(self.dt, dtype=dtype)
        j_in = jnp.asarray(conn.j_in, dtype=dtype)
        j_out = jnp.asarray(conn.j_out, dtype=dtype)

        x_e = s_src[:, None, :] * j_in
        phi_y_e = jnp.broadcast_to(J_eff[None, :, :], (B, D, FJ))
        if bool(getattr(conn, "half_flux_offset", False)):
            phi_y_e = phi_y_e + 0.5
        phi_a = x_e + phi_y_e
        phi_b = x_e - phi_y_e
        gp_e = jnp.broadcast_to(gp.reshape(1, 1), (B, D))[:, :, None]
        gm_e = jnp.broadcast_to(gm.reshape(1, 1), (B, D))[:, :, None]
        bc_e = jnp.broadcast_to(bc.reshape(1, 1), (B, D))[:, :, None]
        # Use squid current +/- edge_state per-edge
        squid_current_a = bc_e - edge_state
        squid_current_b = bc_e + edge_state
        g_a = src.g(phi_a, squid_current=squid_current_a)
        g_b = src.g(phi_b, squid_current=squid_current_b)
        ds_dt = gp_e * (g_a - g_b) - gm_e * edge_state
        S_next = edge_state + dt * ds_dt
        phi_t = jnp.sum(S_next, axis=-1) * j_out
        return phi_t, S_next, rng_key

    def _multiplier_v2_phi_layerwise(self, s_hist: jax.Array, conn: ConnectionSpec, dtype, J_override: jax.Array | None = None) -> jax.Array:
        """Compute phi contribution over time for one multiplier v2 connection.

        Uses dual SQUID states per edge and aggregated output per destination node.
        s_hist: [B,T,F], conn.J: [D,F] -> returns [B,T,D]
        """
        B, _T, _F = s_hist.shape
        # Use override if provided, otherwise use conn.J
        J_base = J_override if J_override is not None else self._clamp_connections(conn)
        D, F = J_base.shape
        J_eff = J_base if (conn.mask is None) else (J_base * conn.mask)
        src = self._select_source_impl(conn.source_key)

        # V2 parameters with defaults
        alpha = jnp.asarray(conn.alpha if conn.alpha is not None else 1.64053, dtype=dtype)
        beta = jnp.asarray(conn.beta if conn.beta is not None else 303.85, dtype=dtype)
        beta_out = jnp.asarray(conn.beta_out if conn.beta_out is not None else 91.156, dtype=dtype)
        # For V2/NOCC connections, use bias_current field (default 2.1)
        bc_val = conn.bias_current if conn.bias_current is not None else 2.1
        bias_current_arr = jnp.asarray(bc_val, dtype=dtype)
        dt = jnp.asarray(self.dt, dtype=dtype)

        # Build mask gate and masked fan-in per destination
        M = jnp.ones_like(J_eff) if conn.mask is None else (conn.mask != 0).astype(J_eff.dtype)
        fan_in = jnp.sum(M, axis=1).astype(dtype)  # [D]
        j_in = jnp.asarray(conn.j_in, dtype=dtype)
        j_out = jnp.asarray(conn.j_out, dtype=dtype)

        def step_fn(carry, x_t):
            # carry: (s1_edges [B,D,F], s2_edges [B,D,F], m_dest [B,D])
            # x_t: [B, F] source states at time t
            s1_e, s2_e, m_d = carry

            # Broadcast source states and weights to [B,D,F]
            x_e = x_t[:, None, :] * j_in  # [B,1,F]
            phi_y_e = jnp.broadcast_to(J_eff[None, :, :], (B, D, F))  # [B,D,F]
            if bool(getattr(conn, "half_flux_offset", False)):
                phi_y_e = phi_y_e + 0.5

            # Compute phi inputs for each edge
            phi_a = x_e + phi_y_e  # [B,D,F]
            phi_b = x_e - phi_y_e  # [B,D,F]

            # Broadcast parameters
            bc_bc = jnp.broadcast_to(bias_current_arr.reshape(1, 1, 1), (B, D, F))

            # Compute SQUID currents for each edge
            squid_current_1 = bc_bc - s1_e
            squid_current_2 = -bc_bc + s2_e

            # Compute source functions (use soft_abs for smooth derivative)
            g1 = src.g(phi_a, squid_current=soft_abs_jax(squid_current_1))  # [B,D,F]
            g2 = src.g(phi_b, squid_current=soft_abs_jax(squid_current_2))  # [B,D,F]

            # Aggregate g1 - g2 per destination node (opposite orientation), mask-gated
            g_sum = jnp.sum((g1 - g2) * M[None, :, :], axis=-1)  # [B,D]

            # Compute effective inductance per destination
            fan_in_bc = fan_in[None, :]  # [1,D]
            # Ensure scalar parameters broadcast correctly with [1,D]
            beta_eff = beta + 2 * fan_in_bc * beta_out  # scalar + [1,D] * scalar = [1,D]

            # Compute dot_m
            dot_m = (g_sum - alpha * m_d) / beta_eff  # [B,D]

            # Broadcast dot_m for edge updates
            dot_m_e = dot_m[:, :, None]  # [B,D,1]

            # Broadcast other parameters
            alpha_bc = jnp.broadcast_to(alpha.reshape(1, 1, 1), (B, D, F))
            beta_bc = jnp.broadcast_to(beta.reshape(1, 1, 1), (B, D, F))
            beta_out_bc = jnp.broadcast_to(beta_out.reshape(1, 1, 1), (B, D, F))

            # Compute dot_s1 and dot_s2; freeze masked edges
            dot_s1 = ((g1 - beta_out_bc * dot_m_e - alpha_bc * s1_e) / beta_bc) * M[None, :, :]
            dot_s2 = ((g2 - beta_out_bc * dot_m_e - alpha_bc * s2_e) / beta_bc) * M[None, :, :]

            # Update states
            s1_next = s1_e + dt * dot_s1
            s2_next = s2_e + dt * dot_s2
            m_next = m_d + dt * dot_m

            return (s1_next, s2_next, m_next), m_next

        # Initialize states
        s1_0 = jnp.zeros((B, D, F), dtype=dtype)
        s2_0 = jnp.zeros((B, D, F), dtype=dtype)
        m_0 = jnp.zeros((B, D), dtype=dtype)

        # Scan over time
        _, phi_seq = jax.lax.scan(step_fn, (s1_0, s2_0, m_0), s_hist.swapaxes(0, 1))  # [T,B,D]
        return phi_seq.swapaxes(0, 1) * j_out  # [B,T,D]

    def _multiplier_v2_step(
        self,
        s_src: jax.Array,
        edge_state: tuple,
        conn: ConnectionSpec,
        dtype,
        J_override: jax.Array | None = None,
        rng_key: jax.Array | None = None,
        conn_noise_cfg=None,
        perturb_offset: jax.Array | None = None,
    ) -> tuple[jax.Array, tuple, jax.Array | None]:
        """Advance multiplier v2 edges one step.

        Args:
            s_src: [B,F] source states
            edge_state: tuple of (s1_e [B,D,F], s2_e [B,D,F], m_d [B,D])
            conn: ConnectionSpec with v2 parameters
            dtype: data type
            J_override: Optional connection weight override [D,F] for training
            rng_key: JAX random key for noise generation (optional)
            conn_noise_cfg: Noise configuration for this connection (optional)
            perturb_offset: Precomputed perturbation offset (optional)

        Returns:
            phi_t: [B,D] output flux (m state)
            next_state: tuple of updated states
            updated_rng_key: Updated random key
        """
        B, F = s_src.shape
        D, FJ = conn.J.shape
        # Use override if provided, otherwise use conn.J
        J_base = J_override if J_override is not None else self._clamp_connections(conn)
        J_eff = J_base if (conn.mask is None) else (J_base * conn.mask)

        # Apply noise if configured
        if conn_noise_cfg is not None and rng_key is not None:
            if conn_noise_cfg.noise is not None and conn_noise_cfg.noise.std > 0.0:
                from .noise_jax import apply_connection_noise
                rng_key, j_noise_key = jax.random.split(rng_key)
                J_eff = apply_connection_noise(j_noise_key, J_eff, conn_noise_cfg, perturb_offset)
            elif perturb_offset is not None:
                J_eff = J_eff + perturb_offset
        elif perturb_offset is not None:
            J_eff = J_eff + perturb_offset

        src = self._select_source_impl(conn.source_key)

        # V2 parameters with defaults
        alpha = jnp.asarray(conn.alpha if conn.alpha is not None else 1.64053, dtype=dtype)
        beta = jnp.asarray(conn.beta if conn.beta is not None else 303.85, dtype=dtype)
        beta_out = jnp.asarray(conn.beta_out if conn.beta_out is not None else 91.156, dtype=dtype)
        # For V2/NOCC connections, use bias_current field (default 2.1)
        bc_val = conn.bias_current if conn.bias_current is not None else 2.1
        bias_current_arr = jnp.asarray(bc_val, dtype=dtype)
        dt = jnp.asarray(self.dt, dtype=dtype)

        s1_e, s2_e, m_d = edge_state

        # Mask and fan-in
        M = jnp.ones_like(J_eff) if conn.mask is None else (conn.mask != 0).astype(J_eff.dtype)
        fan_in = jnp.sum(M, axis=1).astype(dtype)
        j_in = jnp.asarray(conn.j_in, dtype=dtype)
        j_out = jnp.asarray(conn.j_out, dtype=dtype)

        # Broadcast source states and weights to [B,D,F]
        x_e = s_src[:, None, :] * j_in  # [B,1,F]
        phi_y_e = jnp.broadcast_to(J_eff[None, :, :], (B, D, FJ))
        if bool(getattr(conn, "half_flux_offset", False)):
            phi_y_e = phi_y_e + 0.5

        # Compute phi inputs
        phi_a = x_e + phi_y_e
        phi_b = x_e - phi_y_e

        # Broadcast parameters
        bc_bc = jnp.broadcast_to(bias_current_arr.reshape(1, 1, 1), (B, D, FJ))

        # SQUID currents (must match PyTorch MultiplierNOCCLayer and connection_ops):
        # Right branch has opposite orientation: -bias_current + s2
        squid_current_1 = bc_bc - s1_e
        squid_current_2 = -bc_bc + s2_e  # Fixed: was +bc_bc, must be -bc_bc for circuit parity

        # Source functions (use soft_abs for smooth derivative)
        g1 = src.g(phi_a, squid_current=soft_abs_jax(squid_current_1))
        g2 = src.g(phi_b, squid_current=soft_abs_jax(squid_current_2))

        # Aggregate (opposite branch orientation), mask-gated
        g_sum = jnp.sum((g1 - g2) * M[None, :, :], axis=-1)

        # Effective inductance per destination
        fan_in_bc = fan_in[None, :]  # [1, D]
        # Scalar parameters broadcast correctly with [1, D]
        beta_eff = beta + 2 * fan_in_bc * beta_out  # scalar + [1,D] * scalar = [1,D]

        # dot_m
        dot_m = (g_sum - alpha * m_d) / beta_eff

        # Broadcast for edge updates
        dot_m_e = dot_m[:, :, None]
        alpha_bc = jnp.broadcast_to(alpha.reshape(1, 1, 1), (B, D, FJ))
        beta_bc = jnp.broadcast_to(beta.reshape(1, 1, 1), (B, D, FJ))
        beta_out_bc = jnp.broadcast_to(beta_out.reshape(1, 1, 1), (B, D, FJ))

        # dot_s1, dot_s2
        dot_s1 = (g1 - beta_out_bc * dot_m_e - alpha_bc * s1_e) / beta_bc
        dot_s2 = (g2 - beta_out_bc * dot_m_e - alpha_bc * s2_e) / beta_bc

        # Update
        s1_next = s1_e + dt * dot_s1
        s2_next = s2_e + dt * dot_s2
        m_next = m_d + dt * dot_m

        return m_next * j_out, (s1_next, s2_next, m_next), rng_key


    def __call__(
        self,
        external_phi: jax.Array,
        initial_states: dict[int, jax.Array] | None = None,
        s1_inits: dict[int, jax.Array] | None = None,
        s2_inits: dict[int, jax.Array] | None = None,
        conn_override: jax.Array | None = None,
        internal_conn_override: dict[int, jax.Array] | None = None,
        layer_param_override: tuple[jax.Array, ...] | None = None,
        rng_key: jax.Array | None = None,
        *,
        return_trace: bool = False,
        track_phi: bool = False,
        track_g: bool = False,
    ) -> tuple[jax.Array, list[jax.Array]] | tuple[jax.Array, list[jax.Array], ForwardTrace]:
        """Main forward pass - delegates to unified forward module.

        Supports all solvers: layerwise, stepwise_jacobi, stepwise_gauss_seidel.
        For training, pass conn_override/internal_conn_override/layer_param_override.
        For noise injection (layer or connection noise), pass rng_key.

        Args:
            external_phi: [B, T, D_in] input sequence.
            initial_states: Optional dict mapping layer_id to initial main state [B, D].
            s1_inits: Optional dict mapping layer_id to initial s1 state [B, D] for MultiplierNOCC.
            s2_inits: Optional dict mapping layer_id to initial s2 state [B, D] for MultiplierNOCC.
            conn_override: Optional connection parameter overrides [E, D_max, F_max].
            internal_conn_override: Optional internal connection overrides dict.
            layer_param_override: Optional layer parameter overrides tuple.
            rng_key: Optional JAX RNG key for noise injection.

        Returns:
            final_history, histories_per_layer
        """
        from .unified_forward import forward
        return forward(
            self, external_phi,
            initial_states=initial_states,
            s1_inits=s1_inits,
            s2_inits=s2_inits,
            conn_override=conn_override,
            internal_conn_override=internal_conn_override,
            layer_param_override=layer_param_override,
            rng_key=rng_key,
            return_trace=return_trace,
            track_phi=track_phi,
            track_g=track_g,
        )

    # -------------------------------------------------------------------------
    # DEPRECATED METHODS REMOVED
    # The following methods were removed as part of the unified forward refactoring:
    # - apply() - redundant wrapper around __call__
    # - _forward_layerwise() - replaced by unified_forward._forward_layerwise_common
    # - apply_with_conn_params() - replaced by unified_forward.forward()
    # - apply_with_param_tree() - unused
    # - _forward_stepwise_jacobi() - replaced by unified_stepwise._forward_stepwise_common
    # - _forward_stepwise_gauss_seidel() - replaced by unified_stepwise._forward_stepwise_common
    # Use unified_forward.forward() or JAXModel.__call__() instead.
    # -------------------------------------------------------------------------

    def extract_final_states(
        self,
        histories: list[jax.Array],
        *,
        s1_final_by_layer: tuple[jax.Array | None, ...] | None = None,
        s2_final_by_layer: tuple[jax.Array | None, ...] | None = None,
    ) -> tuple[dict[int, jax.Array], dict[int, jax.Array], dict[int, jax.Array]]:
        """Extract final states from histories and MultiplierNOCC layers.

        Args:
            histories: List of state histories [B, T+1, D] for each layer (ordered by layer_id).

        Returns:
            Tuple of (main_states, s1_states, s2_states) where each is a dict mapping
            layer_id to final state [B, D]. s1_states and s2_states only contain
            MultiplierNOCC layers.
        """
        layers_sorted = sorted(self.layers, key=lambda layer: layer.layer_id)
        main_states = {}
        s1_states = {}
        s2_states = {}

        for idx, (spec, history) in enumerate(zip(layers_sorted, histories, strict=False)):
            # Extract final main state from history
            main_states[spec.layer_id] = history[:, -1, :]

            # Check if this is a MultiplierNOCC layer
            kind_l = str(spec.kind).lower()
            if is_multiplier_nocc(kind_l):
                # Auxiliary NOCC states must be provided explicitly (pure style).
                if s1_final_by_layer is not None and idx < len(s1_final_by_layer):
                    s1 = s1_final_by_layer[idx]
                    if s1 is not None:
                        s1_states[spec.layer_id] = s1
                if s2_final_by_layer is not None and idx < len(s2_final_by_layer):
                    s2 = s2_final_by_layer[idx]
                    if s2 is not None:
                        s2_states[spec.layer_id] = s2

        return main_states, s1_states, s2_states


__all__ = ["ConnectionSpec", "JAXModel", "LayerSpec"]
