from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import torch

from soen_toolkit.core.configs import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
)

# Lazy imports only where needed to avoid circulars during type checking
from soen_toolkit.core.soen_model_core import SOENModelCore
from soen_toolkit.core.utils.connection_ops import build_edge_index

from .jax_model import ConnectionSpec, JAXModel, LayerSpec
from .parameter_specs import validate_layer_params


def _to_python_value(value):
    """Convert Torch/JAX/numpy values to plain Python or numpy objects."""
    if isinstance(value, torch.Tensor):
        value_cpu = value.detach().cpu()
        if value_cpu.ndim == 0:
            return float(value_cpu.item())
        return value_cpu.numpy()
    # Handle JAX arrays or numpy-like objects
    if hasattr(value, "__array__"):
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr.item())
        return arr
    return value


def _coerce_scalar_value(value, default):
    """Return float scalar, accepting tensors/arrays."""
    if value is None:
        return float(default)
    coerced = _to_python_value(value)
    if isinstance(coerced, (list, tuple)):
        coerced = np.asarray(coerced)
    if isinstance(coerced, np.ndarray):
        return float(np.asarray(coerced).item())
    return float(coerced)


def _coerce_array_or_scalar_value(value, default):
    """Return float or numpy array preserving vector values."""
    if value is None:
        return default
    coerced = _to_python_value(value)
    if isinstance(coerced, (list, tuple)):
        coerced = np.asarray(coerced)
    if isinstance(coerced, np.ndarray):
        if coerced.shape == ():
            return float(coerced.item())
        return coerced
    return float(coerced)


def _get_param_learnability(layer, param_name: str) -> bool:
    """Extract learnability for a single parameter from a PyTorch layer.

    Uses the ParameterRegistry if available, otherwise falls back to direct attribute access.
    Fails fast if the parameter cannot be found (no silent defaults).

    Args:
        layer: PyTorch layer module
        param_name: Name of the parameter (e.g., "bias_current", "gamma_plus")

    Returns:
        True if the parameter is learnable (requires_grad=True), False otherwise.

    Raises:
        ValueError: If the parameter cannot be found on the layer.
    """
    registry = getattr(layer, "_param_registry", None)

    if registry is not None:
        # Use ParameterRegistry for reliable attribute name lookup
        attr_names = getattr(registry, "_attr_names", {})
        attr_name = attr_names.get(param_name)

        if attr_name is not None:
            param_obj = getattr(layer, attr_name, None)
            if param_obj is not None and isinstance(param_obj, (torch.nn.Parameter, torch.Tensor)):
                return bool(param_obj.requires_grad)
            # Parameter is a buffer (non-learnable by definition)
            return False

    # Fallback: direct attribute access (for layers without ParameterRegistry)
    # Try both the direct name and log_ variant
    for attr_name in (param_name, f"log_{param_name}"):
        param_obj = getattr(layer, attr_name, None)
        if param_obj is not None and isinstance(param_obj, (torch.nn.Parameter, torch.Tensor)):
            return bool(param_obj.requires_grad)

    # If we reach here, the parameter was not found - this is a bug
    msg = f"Could not find parameter '{param_name}' on layer {type(layer).__name__}. "
    msg += f"Available attributes: {[k for k in dir(layer) if not k.startswith('_')]}"
    raise ValueError(msg)


def _extract_internal_connectivity(layer) -> jnp.ndarray | None:
    """Extract internal connectivity matrix from a PyTorch layer.

    Returns:
        JAX array of internal connectivity weights, or None if not present.
    """
    # Try ConnectivityModule first
    conn = getattr(layer, "connectivity", None)
    if conn is not None and hasattr(conn, "weight"):
        return jnp.asarray(conn.weight.detach().cpu().numpy())

    # Fallback to direct internal_J attribute
    internal_J = getattr(layer, "internal_J", None)
    if internal_J is not None:
        return jnp.asarray(internal_J.detach().cpu().numpy())

    return None


def _extract_layer_params(
    layer,
    layer_type: str,
    required_params: tuple[str, ...],
) -> tuple[dict[str, jnp.ndarray], dict[str, bool]]:
    """Extract parameters and learnability from a PyTorch layer.

    This is a unified extraction function that handles all layer types consistently.
    It fails fast on missing parameters and uses the ParameterRegistry for reliable
    learnability detection.

    Args:
        layer: PyTorch layer module
        layer_type: Human-readable layer type for error messages
        required_params: Tuple of required parameter names

    Returns:
        Tuple of (params_dict, learnable_dict)

    Raises:
        ValueError: If any required parameter is missing or learnability cannot be determined.
    """
    p = layer.parameter_values()
    out: dict[str, jnp.ndarray] = {}
    learnable: dict[str, bool] = {}

    for param_name in required_params:
        # Get parameter value
        value = p.get(param_name)
        if value is None:
            msg = f"{layer_type} layer missing required parameter '{param_name}'. "
            msg += f"Available params: {list(p.keys())}"
            raise ValueError(msg)

        # Convert to JAX array
        out[param_name] = jnp.asarray(value.detach().cpu().numpy())

        # Get learnability (fail-fast)
        learnable[param_name] = _get_param_learnability(layer, param_name)

    # Extract internal connectivity if present
    internal_J = _extract_internal_connectivity(layer)
    if internal_J is not None:
        out["internal_J"] = internal_J

    return out, learnable


def _extract_multiplier_params(layer) -> tuple[dict[str, jnp.ndarray], dict[str, bool]]:
    """Extract Multiplier (WICC) parameters from PyTorch layer."""
    return _extract_layer_params(
        layer,
        layer_type="Multiplier",
        required_params=("phi_y", "bias_current", "gamma_plus", "gamma_minus"),
    )


def _extract_single_dendrite_params(layer) -> tuple[dict[str, jnp.ndarray], dict[str, bool]]:
    """Extract SingleDendrite parameters from PyTorch layer."""
    return _extract_layer_params(
        layer,
        layer_type="SingleDendrite",
        required_params=("phi_offset", "bias_current", "gamma_plus", "gamma_minus"),
    )


def _extract_multiplier_v2_params(layer) -> tuple[dict[str, jnp.ndarray], dict[str, bool]]:
    """Extract MultiplierNOCC parameters from PyTorch layer."""
    return _extract_layer_params(
        layer,
        layer_type="MultiplierNOCC",
        required_params=("phi_y", "bias_current", "alpha", "beta", "beta_out"),
    )


def _extract_scaling_params(layer) -> dict[str, np.ndarray]:
    p = layer.parameter_values()
    out: dict[str, np.ndarray] = {}
    v = p.get("scale_factor")
    if v is not None:
        # Convert to numpy array, will be converted to jax later during model build
        out["scale_factor"] = v.detach().cpu().numpy()
    return out


def _extract_softmax_params(layer) -> dict[str, np.ndarray]:
    p = layer.parameter_values()
    out: dict[str, np.ndarray] = {}
    v = p.get("beta")
    if v is not None:
        out["beta"] = v.detach().cpu().numpy()
    return out


def convert_core_model_to_jax(core_model) -> JAXModel:
    """Convert a SOENModelCore to a JAXModel.

    Feedback connections are supported only for stepwise global solvers (Jacobi/Gauss–Seidel).
    They are not supported for layerwise.
    """
    # Determine global solver early and validate feedback only for layerwise
    try:
        _gs = str(getattr(core_model.sim_config, "network_evaluation_method", "layerwise")).lower()
    except Exception:
        _gs = "layerwise"
    if _gs == "layerwise":
        # Block feedback edges only in layerwise mode
        for conn_cfg in core_model.connections_config:
            if conn_cfg.from_layer > conn_cfg.to_layer:
                msg = "Feedback connections are not supported in JAXModel with network_evaluation_method='layerwise'. Use 'stepwise_jacobi' or 'stepwise_gauss_seidel'."
                raise NotImplementedError(
                    msg,
                )

    dt = float(core_model.sim_config.dt)

    # Build LayerSpec list
    layers: list[LayerSpec] = []
    for cfg, layer in zip(core_model.layers_config, core_model.layers, strict=False):
        kind = str(cfg.layer_type)
        dim = int(layer.dim)
        kl = kind.lower()
        if kl == "multiplier":
            params, learnable = _extract_multiplier_params(layer)
            internal_J = params.pop("internal_J", None)
            # Validate: all required params must exist (fail-fast, no silent defaults)
            validate_layer_params("WICC", params)
            source_key = cfg.params.get("source_func") or "Heaviside_state_dep"
            # Extract internal connection mode and params directly from layer
            internal_mode = getattr(layer, "_internal_conn_mode", "fixed")
            internal_dynamic_params = getattr(layer, "_internal_conn_params", None)
            # NOTE: internal_J will be overridden later from connections_config if present
            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="Multiplier",
                    dim=dim,
                    params=params,
                    internal_J=internal_J,
                    source_key=str(source_key),
                    internal_mode=internal_mode,
                    internal_dynamic_params=internal_dynamic_params if internal_mode != "fixed" else None,
                    learnable_params=learnable,
                ),
            )
        elif kl in ("singledendrite", "single_dendrite", "dendrite"):
            params, learnable = _extract_single_dendrite_params(layer)
            internal_J = params.pop("internal_J", None)
            # Validate: all required params must exist (fail-fast, no silent defaults)
            validate_layer_params("SingleDendrite", params)
            source_key = cfg.params.get("source_func") or "RateArray"
            # Extract internal connection mode and params directly from layer
            internal_mode = getattr(layer, "_internal_conn_mode", "fixed")
            internal_dynamic_params = getattr(layer, "_internal_conn_params", None)
            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="SingleDendrite",
                    dim=dim,
                    params=params,
                    internal_J=internal_J,
                    source_key=str(source_key),
                    internal_mode=internal_mode,
                    internal_dynamic_params=internal_dynamic_params if internal_mode != "fixed" else None,
                    learnable_params=learnable,
                ),
            )
        elif kl in ("multiplierv2", "multiplier_v2", "multipliernocc", "nocc"):
            params, learnable = _extract_multiplier_v2_params(layer)
            internal_J = params.pop("internal_J", None)
            # Validate: all required params must exist (fail-fast, no silent defaults)
            validate_layer_params("NOCC", params)
            source_key = cfg.params.get("source_func") or "RateArray"
            # Extract internal connection mode and params directly from layer
            internal_mode = getattr(layer, "_internal_conn_mode", "fixed")
            internal_dynamic_params = getattr(layer, "_internal_conn_params", None)
            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="MultiplierNOCC",
                    dim=dim,
                    params=params,
                    internal_J=internal_J,
                    source_key=str(source_key),
                    internal_mode=internal_mode,
                    internal_dynamic_params=internal_dynamic_params if internal_mode != "fixed" else None,
                    learnable_params=learnable,
                ),
            )
        elif kl == "soma":
            # Soma: spiking head + surrogate gradients (Torch defines behavior)
            try:
                p = layer.parameter_values()
            except Exception as e:
                raise RuntimeError(f"Failed to extract Soma parameters: {e}") from e
            params = {}
            for k in ("phi_offset", "bias_current", "gamma_plus", "gamma_minus", "threshold"):
                if k not in p:
                    raise ValueError(f"Soma layer missing required parameter '{k}'")
                params[k] = jnp.asarray(p[k].detach().cpu().numpy())
            validate_layer_params("Soma", params)

            surrogate_kind = getattr(getattr(layer, "surrogate", None), "kind", "triangle")
            surrogate_params = {}
            surrogate_obj = getattr(layer, "surrogate", None)
            if surrogate_obj is not None:
                # width/scale/clip are optional; store only if present
                for key in ("width", "scale", "clip"):
                    if hasattr(surrogate_obj, key):
                        v = getattr(surrogate_obj, key)
                        if v is not None:
                            surrogate_params[key] = float(v)
            internal_mode = getattr(layer, "_internal_conn_mode", "fixed")
            internal_dynamic_params = getattr(layer, "_internal_conn_params", None)
            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="Soma",
                    dim=dim,
                    params=params,
                    internal_J=None,
                    source_key=str(cfg.params.get("source_func") or "RateArray"),
                    internal_mode=internal_mode,
                    internal_dynamic_params=internal_dynamic_params if internal_mode != "fixed" else None,
                    surrogate_kind=str(surrogate_kind),
                    surrogate_params=surrogate_params,
                ),
            )
        elif kl == "synapse":
            try:
                p = layer.parameter_values()
            except Exception as e:
                raise RuntimeError(f"Failed to extract Synapse parameters: {e}") from e
            if "alpha" not in p:
                raise ValueError("Synapse layer missing required parameter 'alpha'")
            params = {"alpha": jnp.asarray(p["alpha"].detach().cpu().numpy())}
            validate_layer_params("Synapse", params)
            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="Synapse",
                    dim=dim,
                    params=params,
                    internal_J=None,
                    source_key=None,
                ),
            )
        elif kl == "mingru":
            # Extract weight matrices from the PyTorch MinGRU layer
            W_hidden = None
            W_gate = None
            try:
                W_hidden = layer.hidden_proj.weight.detach().cpu().numpy()
                W_gate = layer.gate_proj.weight.detach().cpu().numpy()
            except Exception as e:
                msg = f"Failed to extract MinGRU weights: {e}"
                raise RuntimeError(msg)
            params = {
                "W_hidden": jnp.asarray(W_hidden),
                "W_gate": jnp.asarray(W_gate),
            }
            internal_J = None
            conn = getattr(layer, "connectivity", None)
            if conn is not None and hasattr(conn, "weight"):
                internal_J = jnp.asarray(conn.weight.detach().cpu().numpy())
            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="MinGRU",
                    dim=dim,
                    params=params,
                    internal_J=internal_J,
                    source_key=None,
                ),
            )
        elif kl == "gru":
            # Extract PyTorch nn.GRU weights (single-layer, input_size=dim, hidden_size=dim)
            try:
                W_ih = layer.core.weight_ih_l0.detach().cpu().numpy()  # [3D, D]
                W_hh = layer.core.weight_hh_l0.detach().cpu().numpy()  # [3D, D]
                b_ih = None
                b_hh = None
                try:
                    b_ih = layer.core.bias_ih_l0.detach().cpu().numpy()
                    b_hh = layer.core.bias_hh_l0.detach().cpu().numpy()
                except Exception:
                    pass
            except Exception as e:
                msg = f"Failed to extract GRU weights: {e}"
                raise RuntimeError(msg)
            params = {
                "weight_ih": jnp.asarray(W_ih),
                "weight_hh": jnp.asarray(W_hh),
            }
            if b_ih is not None:
                params["bias_ih"] = jnp.asarray(b_ih)
            if b_hh is not None:
                params["bias_hh"] = jnp.asarray(b_hh)
            internal_J = None
            conn = getattr(layer, "connectivity", None)
            if conn is not None and hasattr(conn, "weight"):
                internal_J = jnp.asarray(conn.weight.detach().cpu().numpy())

            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="GRU",
                    dim=dim,
                    params=params,
                    internal_J=internal_J,
                    source_key=None,
                ),
            )
        elif kl == "leakygru":
            # Extract LeakyRU parameters
            try:
                # Need access to internal core or access via properties/attributes
                # LeakyGRULayer exposes .core which is LeakyRU
                core = layer.core
                W_in = core.weight_ih[2 * dim : 3 * dim].detach().cpu().numpy()  # (H, I)
                W_hn = core._effective_candidate_hh().detach().cpu().numpy()  # (H, H)
                bias_z = core.bias_z.detach().cpu().numpy()
                bias_n = core.bias_n.detach().cpu().numpy()
                bias_r = core.bias_r_fixed.detach().cpu().numpy()
            except Exception as e:
                msg = f"Failed to extract LeakyGRU weights: {e}"
                raise RuntimeError(msg)

            params = {
                "W_in": jnp.asarray(W_in),
                "W_hn": jnp.asarray(W_hn),
                "bias_z": jnp.asarray(bias_z),
                "bias_n": jnp.asarray(bias_n),
                "bias_r": jnp.asarray(bias_r),
            }
            # Handle additional toolkit connectivity if any
            internal_J = None
            conn = getattr(layer, "connectivity", None)
            if conn is not None and hasattr(conn, "weight"):
                internal_J = jnp.asarray(conn.weight.detach().cpu().numpy())

            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="LeakyGRU",
                    dim=dim,
                    params=params,
                    internal_J=internal_J,
                    source_key=None,
                ),
            )
        elif kl == "lstm":

            # Extract PyTorch nn.LSTM weights (single-layer, input_size=dim, hidden_size=dim)
            try:
                # LSTM weights stored as [4*D, D] for 4 gates: input, forget, cell, output
                W_ih = layer.module.weight_ih_l0.detach().cpu().numpy()  # [4D, D]
                W_hh = layer.module.weight_hh_l0.detach().cpu().numpy()  # [4D, D]
                b_ih = None
                b_hh = None
                try:
                    b_ih = layer.module.bias_ih_l0.detach().cpu().numpy()  # [4D]
                    b_hh = layer.module.bias_hh_l0.detach().cpu().numpy()  # [4D]
                except Exception:
                    pass
            except Exception as e:
                msg = f"Failed to extract LSTM weights: {e}"
                raise RuntimeError(msg)
            params = {
                "weight_ih": jnp.asarray(W_ih),
                "weight_hh": jnp.asarray(W_hh),
            }
            if b_ih is not None:
                params["bias_ih"] = jnp.asarray(b_ih)
            if b_hh is not None:
                params["bias_hh"] = jnp.asarray(b_hh)
            internal_J = None
            conn = getattr(layer, "connectivity", None)
            if conn is not None and hasattr(conn, "weight"):
                internal_J = jnp.asarray(conn.weight.detach().cpu().numpy())
            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="LSTM",
                    dim=dim,
                    params=params,
                    internal_J=internal_J,
                    source_key=None,
                ),
            )
        elif kl == "linear":
            # No parameters; just mark kind
            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="Linear",
                    dim=dim,
                    params={},
                    internal_J=None,
                    source_key=None,
                ),
            )
        elif kl == "nonlinear":
            # Extract optional phi_offset and bias_current; validate supported source
            params = {}
            try:
                p = layer.parameter_values()
                for k in ("phi_offset", "bias_current"):
                    v = p.get(k)
                    if v is not None:
                        params[k] = jnp.asarray(v.detach().cpu().numpy())
            except Exception:
                pass
            source_key = cfg.params.get("source_func") or cfg.params.get("source_func_type") or "Tanh"
            sk_low = str(source_key).lower()
            supported = {"tanh", "simplegelu", "simple_gelu", "gelu_simple", "telu"}
            if sk_low not in supported:
                msg = f"Unsupported NonLinear source function '{source_key}' for JAX conversion. Supported: Tanh, SimpleGELU, TeLU"
                raise ValueError(
                    msg,
                )
            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="NonLinear",
                    dim=dim,
                    params=params,
                    internal_J=None,
                    source_key=str(source_key),
                ),
            )
        elif kl in ("scalinglayer", "scaling"):
            # Extract scale_factor parameter
            params_raw = _extract_scaling_params(layer)
            params = {}
            if "scale_factor" in params_raw:
                params["scale_factor"] = jnp.asarray(params_raw["scale_factor"])
            # Validate: all required params must exist (fail-fast, no silent defaults)
            validate_layer_params("ScalingLayer", params)
            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="ScalingLayer",
                    dim=dim,
                    params=params,
                    internal_J=None,
                    source_key=None,
                ),
            )
        elif kl == "softmax":
            # Extract beta parameter
            params_raw = _extract_softmax_params(layer)
            params = {}
            if "beta" in params_raw:
                params["beta"] = jnp.asarray(params_raw["beta"])
            # Validate: all required params must exist (fail-fast, no silent defaults)
            validate_layer_params("Softmax", params)
            layers.append(
                LayerSpec(
                    layer_id=cfg.layer_id,
                    kind="Softmax",
                    dim=dim,
                    params=params,
                    internal_J=None,
                    source_key=None,
                ),
            )
        else:
            msg = f"Unsupported layer type in JAX port: {kind}"
            raise NotImplementedError(msg)

    # First pass: extract internal connections from connections_config
    layer_internal_conns: dict[int, tuple] = {}  # layer_id -> (J, mode, params, learnable, mask)
    for conn_cfg in core_model.connections_config:
        if conn_cfg.from_layer == conn_cfg.to_layer:
            key = f"J_{conn_cfg.from_layer}_to_{conn_cfg.to_layer}"
            J = core_model.connections.get(key, None)
            if J is not None:
                # Extract mode and params using unified parser
                from soen_toolkit.core.utils.connection_ops import parse_connection_config

                conn_mode, conn_params = parse_connection_config(conn_cfg.params or {})
                learnable = bool(conn_cfg.learnable)
                # Extract mask if present
                mask = None
                try:
                    mask = core_model.connection_masks.get(key, None)
                except Exception:
                    mask = None
                layer_internal_conns[conn_cfg.from_layer] = (J, conn_mode, conn_params, learnable, mask)

    # Inject internal connections into layers
    for layer_spec in layers:
        if layer_spec.layer_id in layer_internal_conns:
            J_tensor, conn_mode, conn_params, learnable, mask_tensor = layer_internal_conns[layer_spec.layer_id]
            layer_spec.internal_J = jnp.asarray(J_tensor.detach().cpu().numpy())
            layer_spec.internal_mode = conn_mode
            layer_spec.internal_dynamic_params = conn_params if conn_mode != "fixed" else None
            layer_spec.internal_J_learnable = learnable
            # Validate mask shape matches J if present
            if mask_tensor is not None:
                mask_jax = jnp.asarray(mask_tensor.detach().cpu().numpy())
                if mask_jax.shape != layer_spec.internal_J.shape:
                    msg = f"Internal connection mask shape mismatch for layer {layer_spec.layer_id}: mask {mask_jax.shape} != J {layer_spec.internal_J.shape}"
                    raise ValueError(msg)
                layer_spec.internal_mask = mask_jax
            else:
                layer_spec.internal_mask = None

    # Build ConnectionSpec list (only inter-layer J_*), include masks
    connections: list[ConnectionSpec] = []
    conn_mode_map = getattr(core_model, "_connection_modes", {}) or {}
    conn_params_map = getattr(core_model, "_connection_params", {}) or {}

    for idx, conn_cfg in enumerate(core_model.connections_config):
        # Skip internal connections (already processed above)
        if conn_cfg.from_layer == conn_cfg.to_layer:
            continue

        # Handle external connections
        key = f"J_{conn_cfg.from_layer}_to_{conn_cfg.to_layer}"
        J = core_model.connections.get(key, None)
        if J is None:
            continue

        mask = None
        try:
            mask = core_model.connection_masks.get(key, None)
        except Exception:
            mask = None
        mode = "fixed"
        source_key = None
        gamma_plus = 1e-3
        gamma_minus = 1e-3
        bias_current = 2.0
        j_in = 0.38
        j_out = 0.38
        # Initialize v2 scalars to defaults so JAX side always has concrete values
        alpha = 1.64053
        beta = 303.85
        beta_out = 91.156
        # Note: For NOCC, bias_current default is 2.1, for WICC it's 2.0
        # The bias_current variable above (initialized to 2.0) is used for WICC
        half_flux_offset = False
        try:
            params = dict(conn_cfg.params or {})
            raw_mode = str(params.get("mode", params.get("connection_mode", "fixed"))).lower()
            # Normalize all aliases to canonical mode names
            if raw_mode in {"dynamic", "multiplier", "programmable", "wicc", "dynamic_v1", "v1"}:
                normalized_mode = "WICC"
            elif raw_mode in {"dynamic_v2", "multiplier_v2", "nocc", "v2"}:
                normalized_mode = "NOCC"
            else:
                normalized_mode = "fixed"

            # Dynamic params block may be under 'connection_params' (preferred), 'dynamic', or legacy 'multiplier'
            dyn_block = params.get("connection_params") or params.get("dynamic") or params.get("multiplier")
            dyn = dyn_block or {}
            source_key = dyn.get("source_func") or dyn.get("source_func_type")

            # Infer mode ONLY when a dynamic block is explicitly present
            # Note: bias_current is used by both WICC and NOCC, so we check for v2-specific keys
            v2_keys = {"alpha", "beta", "beta_out"}
            v1_keys = {"gamma_plus", "gamma_minus"}
            mode = normalized_mode
            if dyn_block is not None:
                if any(k in dyn for k in v2_keys):
                    mode = "NOCC"
                elif any(k in dyn for k in v1_keys):
                    mode = "WICC"

            # Prefer builder-provided metadata (includes fan-in-based j_out overrides)
            builder_mode = conn_mode_map.get(key)
            if isinstance(builder_mode, str) and builder_mode:
                mode = builder_mode
            builder_dyn = conn_params_map.get(key)
            if builder_dyn:
                dyn = {k: _to_python_value(v) for k, v in dict(builder_dyn).items()}
            elif dyn_block:
                dyn = {k: _to_python_value(v) for k, v in dict(dyn_block).items()}
            else:
                dyn = {}

            if dyn:
                source_key = dyn.get("source_func") or dyn.get("source_func_type") or source_key

            if mode == "WICC":
                # WICC (v1) parameters
                if dyn.get("gamma_plus") is not None:
                    gamma_plus = _coerce_scalar_value(dyn.get("gamma_plus"), gamma_plus)
                if dyn.get("gamma_minus") is not None:
                    gamma_minus = _coerce_scalar_value(dyn.get("gamma_minus"), gamma_minus)
                if dyn.get("bias_current") is not None:
                    bias_current = _coerce_scalar_value(dyn.get("bias_current"), bias_current)
                if dyn.get("j_in") is not None:
                    j_in = _coerce_scalar_value(dyn.get("j_in"), j_in)
                if dyn.get("j_out") is not None:
                    j_out = _coerce_array_or_scalar_value(dyn.get("j_out"), j_out)
            elif mode == "NOCC":
                # V2 (NOCC) parameters - use bias_current instead of ib
                # Default bias_current for NOCC is 2.1
                if dyn.get("bias_current") is not None:
                    bias_current = _coerce_scalar_value(dyn.get("bias_current"), 2.1)
                else:
                    bias_current = 2.1  # NOCC default
                if dyn.get("alpha") is not None:
                    alpha = _coerce_scalar_value(dyn.get("alpha"), alpha)
                if dyn.get("beta") is not None:
                    beta = _coerce_scalar_value(dyn.get("beta"), beta)
                if dyn.get("beta_out") is not None:
                    beta_out = _coerce_scalar_value(dyn.get("beta_out"), beta_out)
                if dyn.get("j_in") is not None:
                    j_in = _coerce_scalar_value(dyn.get("j_in"), j_in)
                if dyn.get("j_out") is not None:
                    j_out = _coerce_array_or_scalar_value(dyn.get("j_out"), j_out)
            half_flux_offset = bool(dyn.get("half_flux_offset", half_flux_offset))
        except Exception:
            pass
        connections.append(
            ConnectionSpec(
                from_layer=conn_cfg.from_layer,
                to_layer=conn_cfg.to_layer,
                J=jnp.asarray(J.detach().cpu().numpy()),
                mask=(jnp.asarray(mask.detach().cpu().numpy()) if mask is not None else None),
                index=idx,
                mode=mode,
                source_key=(str(source_key) if source_key is not None else None),
                gamma_plus=gamma_plus,
                gamma_minus=gamma_minus,
                bias_current=bias_current,
                j_in=j_in,
                j_out=j_out,
                alpha=alpha,
                beta=beta,
                beta_out=beta_out,
                # Note: ib field removed - using bias_current for both WICC and NOCC
                half_flux_offset=half_flux_offset,
                learnable=bool(conn_cfg.learnable),
            ),
        )

    # Propagate global solver (layerwise or stepwise_jacobi)
    try:
        gs = str(getattr(core_model.sim_config, "network_evaluation_method", "layerwise")).lower()
    except Exception:
        gs = "layerwise"
    # Only support layerwise, stepwise_jacobi, and stepwise_gauss_seidel in JAX path
    if gs not in {"layerwise", "stepwise_jacobi", "stepwise_gauss_seidel"}:
        gs = "layerwise"
    # Propagate input_type semantics ("flux" or "state") to ensure consistent first-layer handling
    try:
        it = str(getattr(core_model.sim_config, "input_type", "flux")).lower()
    except Exception:
        it = "flux"
    if it not in {"flux", "state"}:
        it = "flux"
    connection_constraints = getattr(core_model, "connection_constraints", None)
    if connection_constraints:
        connection_constraints = {key: dict(value or {}) for key, value in connection_constraints.items()}
    else:
        connection_constraints = None

    # Extract per-element constraint matrices for polarity enforcement
    constraint_min_mats = getattr(core_model, "connection_constraint_min_matrices", None)
    constraint_max_mats = getattr(core_model, "connection_constraint_max_matrices", None)
    connection_constraint_min_matrices = None
    connection_constraint_max_matrices = None
    if constraint_min_mats:
        connection_constraint_min_matrices = {
            key: jnp.asarray(mat.detach().cpu().numpy())
            for key, mat in constraint_min_mats.items()
        }
    if constraint_max_mats:
        connection_constraint_max_matrices = {
            key: jnp.asarray(mat.detach().cpu().numpy())
            for key, mat in constraint_max_mats.items()
        }

    # Convert connection noise settings from PyTorch to JAX format
    # Must distinguish between GaussianNoise (per-timestep) and GaussianPerturbation (fixed per forward)
    connection_noise_settings = None
    torch_conn_noise = getattr(core_model, "connection_noise_settings", None)
    if torch_conn_noise:
        from soen_toolkit.core.noise import GaussianNoise, GaussianPerturbation

        from .noise_jax import GaussianNoiseConfig, NoiseConfig, PerturbationConfig

        connection_noise_settings = {}
        for key, settings in torch_conn_noise.items():
            # Extract j noise strategy from PyTorch NoiseSettings
            j_noise = getattr(settings, "j", None)
            if j_noise is None:
                continue

            noise_cfg = None
            perturb_cfg = None

            # Check actual type to distinguish Noise (per-step) from Perturbation (fixed)
            if isinstance(j_noise, GaussianNoise):
                std = float(j_noise.std)
                relative = bool(j_noise.relative)
                if std != 0.0:
                    noise_cfg = GaussianNoiseConfig(std=std, relative=relative)
            elif isinstance(j_noise, GaussianPerturbation):
                mean = float(j_noise.mean)
                std = float(j_noise.std)
                if mean != 0.0 or std != 0.0:
                    perturb_cfg = PerturbationConfig(mean=mean, std=std)
            else:
                # Unknown type - try to extract std for backward compatibility
                std = getattr(j_noise, "std", 0.0)
                if std != 0.0:
                    noise_cfg = GaussianNoiseConfig(std=float(std), relative=False)

            if noise_cfg is not None or perturb_cfg is not None:
                connection_noise_settings[key] = NoiseConfig(
                    noise=noise_cfg,
                    perturbation=perturb_cfg,
                )

    return JAXModel(
        dt=dt,
        layers=layers,
        connections=connections,
        network_evaluation_method=gs,
        input_type=it,
        connection_constraints=connection_constraints,
        connection_constraint_min_matrices=connection_constraint_min_matrices,
        connection_constraint_max_matrices=connection_constraint_max_matrices,
        connection_noise_settings=connection_noise_settings,
    )


def convert_file_to_jax(path: str) -> JAXModel:
    """Load a .yaml/.json/.soen via the core builder and convert to JAXModel."""
    from soen_toolkit.core.soen_model_core import SOENModelCore

    core = SOENModelCore.build(path)
    return convert_core_model_to_jax(core)


def _torch_from_jax_array(x: jax.Array | np.ndarray | list | tuple) -> torch.Tensor:
    try:
        return torch.tensor(np.asarray(x), dtype=torch.float32)
    except Exception:
        # Best-effort fallback
        return torch.as_tensor(x, dtype=torch.float32)


def _layer_type_from_kind(kind: str) -> str:
    k = kind.strip()
    # JAXModel kinds already match Torch layer_type strings for supported layers
    # Keep passthrough; normalize common aliases
    low = k.lower()
    if low in ("singledendrite", "single_dendrite", "dendrite"):
        return "SingleDendrite"
    if low == "mingru":
        return "MinGRU"
    if low == "nonlinear":
        return "NonLinear"
    if low == "linear":
        return "Linear"
    if low == "multiplier":
        return "Multiplier"
    if low == "gru":
        return "GRU"
    if low == "lstm":
        return "LSTM"
    if low in ("scalinglayer", "scaling"):
        return "ScalingLayer"
    if low == "softmax":
        return "Softmax"
    return k


def convert_jax_to_core_model(jax_model: JAXModel, base_core: SOENModelCore | None = None) -> SOENModelCore:
    """Rebuild a SOENModelCore from a JAXModel, or update an existing core in place.

    Notes:
    - Noise/perturbation settings are not represented in JAXModel and will use defaults.
    - Learnability flags are not represented in JAXModel and default to True for new cores.
    - Internal layer connectivity is restored via LayerConfig.params["internal_J"].
    - Dynamic external connections ("mode" == "dynamic") are restored via ConnectionConfig params.

    """
    def _scalar_param(val, default):
        if val is None:
            return float(default)
        if isinstance(val, (jax.Array, np.ndarray)):
            return float(np.asarray(val).item())
        if isinstance(val, (list, tuple)):
            return float(np.asarray(val).item())
        return float(val)

    def _j_out_param(val, default):
        if val is None:
            return float(default)
        if isinstance(val, (jax.Array, np.ndarray, list, tuple)):
            tensor_val = _torch_from_jax_array(val)
            if tensor_val.ndim == 0:
                return float(tensor_val.item())
            return tensor_val
        return float(val)

    # Update in place when a compatible base_core is provided
    if base_core is not None:
        # Update sim semantics when possible
        with contextlib.suppress(Exception):
            base_core.sim_config.dt = float(jax_model.dt)
        try:
            if hasattr(jax_model, "input_type"):
                base_core.sim_config.input_type = str(getattr(jax_model, "input_type", "flux"))
            if hasattr(jax_model, "network_evaluation_method"):
                base_core.sim_config.network_evaluation_method = str(getattr(jax_model, "network_evaluation_method", "layerwise"))
        except Exception:
            pass

        # Map layer_id -> (index, layer module)
        layer_by_id: dict[int, tuple[int, torch.nn.Module]] = {cfg.layer_id: (i, layer) for i, (cfg, layer) in enumerate(zip(base_core.layers_config, base_core.layers, strict=False))}

        for spec in jax_model.layers:
            entry = layer_by_id.get(int(spec.layer_id))
            if entry is None:
                continue
            _idx, layer = entry
            kind_l = str(spec.kind).lower()
            # Push parameter vectors via ParameterRegistry where available
            reg = getattr(layer, "_param_registry", None)
            if reg is not None:
                params = dict(spec.params or {})
                for name in ("phi_y", "bias_current", "gamma_plus", "gamma_minus", "phi_offset", "scale_factor", "beta"):
                    if name in params and params[name] is not None:
                        with contextlib.suppress(Exception):
                            reg.override_parameter(name, value=_torch_from_jax_array(params[name]))
            # Restore internal connectivity when provided
            if getattr(spec, "internal_J", None) is not None:
                try:
                    layer.internal_J = _torch_from_jax_array(spec.internal_J)
                except Exception:
                    # Fallback: some layers expose connectivity.weight directly
                    try:
                        if getattr(layer, "connectivity", None) is not None:
                            with torch.no_grad():
                                layer.connectivity.weight.copy_(_torch_from_jax_array(spec.internal_J))
                    except Exception:
                        pass
                # Also update the unified self-connection parameter in base_core.connections (J_i_to_i)
                try:
                    self_key = f"J_{int(spec.layer_id)}_to_{int(spec.layer_id)}"
                    if self_key in base_core.connections:
                        with torch.no_grad():
                            base_core.connections[self_key].data.copy_(_torch_from_jax_array(spec.internal_J))
                        # Recompute edge indices for this self connection
                        try:
                            mask_t = base_core.connection_masks.get(self_key)
                            w_t = base_core.connections.get(self_key)
                            if w_t is not None:
                                src_idx, dst_idx = build_edge_index(mask_t, w_t)
                                base_core._connection_edge_maps[self_key] = (src_idx, dst_idx)
                        except Exception:
                            pass
                except Exception:
                    # Non-fatal: continue if unified key is absent
                    pass
            # MinGRU: copy weight matrices
            if kind_l == "mingru":
                try:
                    p = dict(spec.params or {})
                    if "W_hidden" in p:
                        layer.hidden_proj.weight.data.copy_(_torch_from_jax_array(p["W_hidden"]))
                    if "W_gate" in p:
                        layer.gate_proj.weight.data.copy_(_torch_from_jax_array(p["W_gate"]))
                except Exception:
                    pass

        # External connections: weights, masks, and dynamic params
        for c in jax_model.connections:
            key = f"J_{int(c.from_layer)}_to_{int(c.to_layer)}"
            if key not in base_core.connections:
                # Skip unknown mappings when updating in place
                continue
            with contextlib.suppress(Exception):
                base_core.connections[key].data.copy_(_torch_from_jax_array(c.J))
            if getattr(c, "mask", None) is not None:
                with contextlib.suppress(Exception):
                    base_core.connection_masks[key] = _torch_from_jax_array(c.mask)
            try:
                mode_l = str(getattr(c, "mode", "fixed")).lower()
                if mode_l in ("dynamic", "multiplier", "wicc", "dynamic_v1", "v1"):
                    base_core._connection_modes[key] = "WICC"
                    base_core._connection_params[key] = {
                        "source_func": (c.source_key or "RateArray"),
                        "gamma_plus": _scalar_param(getattr(c, "gamma_plus", None), 1e-3),
                        "bias_current": _scalar_param(getattr(c, "bias_current", None), 2.0),
                        "j_in": _scalar_param(getattr(c, "j_in", None), 0.38),
                        "j_out": _j_out_param(getattr(c, "j_out", None), 0.38),
                        "half_flux_offset": bool(getattr(c, "half_flux_offset", False)),
                    }
                elif mode_l in ("dynamic_v2", "multiplier_v2", "nocc", "v2"):
                    base_core._connection_modes[key] = "NOCC"
                    base_core._connection_params[key] = {
                        "source_func": (c.source_key or "RateArray"),
                        "alpha": _scalar_param(getattr(c, "alpha", None), 1.64053),
                        "beta": _scalar_param(getattr(c, "beta", None), 303.85),
                        "beta_out": _scalar_param(getattr(c, "beta_out", None), 91.156),
                        "bias_current": _scalar_param(getattr(c, "bias_current", None), 2.1),
                        "j_in": _scalar_param(getattr(c, "j_in", None), 0.38),
                        "j_out": _j_out_param(getattr(c, "j_out", None), 0.38),
                        "half_flux_offset": bool(getattr(c, "half_flux_offset", False)),
                    }
                else:
                    base_core._connection_modes[key] = "fixed"
            except Exception:
                pass
            # Recompute edge indices to match updated mask/weights
            try:
                mask_t = base_core.connection_masks.get(key)
                w_t = base_core.connections.get(key)
                if w_t is not None:
                    src_idx, dst_idx = build_edge_index(mask_t, w_t)
                    base_core._connection_edge_maps[key] = (src_idx, dst_idx)
            except Exception:
                pass

        return base_core

    # Otherwise, construct a fresh SOENModelCore
    sim = SimulationConfig(
        dt=float(jax_model.dt),
        input_type=str(getattr(jax_model, "input_type", "flux")),
        network_evaluation_method=str(getattr(jax_model, "network_evaluation_method", "layerwise")),
    )

    layers_cfg: list[LayerConfig] = []
    for spec in sorted(jax_model.layers, key=lambda spec: spec.layer_id):
        layer_type = _layer_type_from_kind(spec.kind)
        params = {"dim": int(spec.dim)}
        # Propagate source function identity when present
        if getattr(spec, "source_key", None) is not None:
            params["source_func"] = str(spec.source_key)
        # Internal connectivity (if supplied)
        if getattr(spec, "internal_J", None) is not None:
            params["internal_J"] = _torch_from_jax_array(spec.internal_J)
        # Do NOT embed vector-valued parameters via initializers here; assign after build via registry

        layers_cfg.append(
            LayerConfig(layer_id=int(spec.layer_id), layer_type=layer_type, params=params),
        )

    connections_cfg: list[ConnectionConfig] = []
    for c in jax_model.connections:
        params_c: dict[str, object] = {"structure": {"type": "dense"}}
        mode_l = str(getattr(c, "mode", "fixed")).lower()
        if mode_l in ("dynamic", "multiplier", "wicc", "dynamic_v1", "v1"):
            dyn = {
                "source_func": (c.source_key or "RateArray"),
                "gamma_plus": _scalar_param(getattr(c, "gamma_plus", None), 1e-3),
                "bias_current": _scalar_param(getattr(c, "bias_current", None), 2.0),
                "j_in": _scalar_param(getattr(c, "j_in", None), 0.38),
                "j_out": _j_out_param(getattr(c, "j_out", None), 0.38),
                "half_flux_offset": bool(getattr(c, "half_flux_offset", False)),
            }
            params_c["mode"] = "WICC"
            params_c["connection_params"] = dyn
        elif mode_l in ("dynamic_v2", "multiplier_v2", "nocc", "v2"):
            dyn = {
                "source_func": (c.source_key or "RateArray"),
                "alpha": _scalar_param(getattr(c, "alpha", None), 1.64053),
                "beta": _scalar_param(getattr(c, "beta", None), 303.85),
                "beta_out": _scalar_param(getattr(c, "beta_out", None), 91.156),
                "bias_current": _scalar_param(getattr(c, "bias_current", None), 2.1),
                "j_in": _scalar_param(getattr(c, "j_in", None), 0.38),
                "j_out": _j_out_param(getattr(c, "j_out", None), 0.38),
                "half_flux_offset": bool(getattr(c, "half_flux_offset", False)),
            }
            params_c["mode"] = "NOCC"
            params_c["connection_params"] = dyn
        connections_cfg.append(
            ConnectionConfig(
                from_layer=int(c.from_layer),
                to_layer=int(c.to_layer),
                connection_type="dense",
                params=params_c,
                learnable=bool(getattr(c, "learnable", True)),
            ),
        )

    core = SOENModelCore(sim_config=sim, layers_config=layers_cfg, connections_config=connections_cfg)

    # Copy external connection weights and masks verbatim
    for c in jax_model.connections:
        key = f"J_{int(c.from_layer)}_to_{int(c.to_layer)}"
        if key in core.connections:
            with contextlib.suppress(Exception):
                core.connections[key].data.copy_(_torch_from_jax_array(c.J))
            if getattr(c, "mask", None) is not None:
                with contextlib.suppress(Exception):
                    core.connection_masks[key] = _torch_from_jax_array(c.mask)
            # Preserve learnability flag
            if key in core.connections:
                core.connections[key].requires_grad = bool(getattr(c, "learnable", True))
            # Recompute edge indices based on actual mask/weights
            try:
                mask_t = core.connection_masks.get(key)
                w_t = core.connections.get(key)
                if w_t is not None:
                    src_idx, dst_idx = build_edge_index(mask_t, w_t)
                    core._connection_edge_maps[key] = (src_idx, dst_idx)
            except Exception:
                pass

    # Populate connection metadata (mode + params) so Torch runtime sees the same fan-in j_out vector
    for c in jax_model.connections:
        key = f"J_{int(c.from_layer)}_to_{int(c.to_layer)}"
        if key not in core.connections:
            continue
        mode_l = str(getattr(c, "mode", "fixed")).lower()
        if mode_l in ("dynamic", "multiplier", "wicc", "dynamic_v1", "v1"):
            core._connection_modes[key] = "WICC"
            core._connection_params[key] = {
                "source_func": (c.source_key or "RateArray"),
                "gamma_plus": _scalar_param(getattr(c, "gamma_plus", None), 1e-3),
                "bias_current": _scalar_param(getattr(c, "bias_current", None), 2.0),
                "j_in": _scalar_param(getattr(c, "j_in", None), 0.38),
                "j_out": _j_out_param(getattr(c, "j_out", None), 0.38),
                "half_flux_offset": bool(getattr(c, "half_flux_offset", False)),
            }
        elif mode_l in ("dynamic_v2", "multiplier_v2", "nocc", "v2"):
            core._connection_modes[key] = "NOCC"
            core._connection_params[key] = {
                "source_func": (c.source_key or "RateArray"),
                "alpha": _scalar_param(getattr(c, "alpha", None), 1.64053),
                "beta": _scalar_param(getattr(c, "beta", None), 303.85),
                "beta_out": _scalar_param(getattr(c, "beta_out", None), 91.156),
                "bias_current": _scalar_param(getattr(c, "bias_current", None), 2.1),
                "j_in": _scalar_param(getattr(c, "j_in", None), 0.38),
                "j_out": _j_out_param(getattr(c, "j_out", None), 0.38),
                "half_flux_offset": bool(getattr(c, "half_flux_offset", False)),
            }
        else:
            core._connection_modes[key] = "fixed"

    # Handle internal connections: preserve learnability and restore weights
    for spec in jax_model.layers:
        if getattr(spec, "internal_J", None) is not None:
            layer_id = int(spec.layer_id)
            internal_key = f"J_{layer_id}_to_{layer_id}"
            internal_learnable = bool(getattr(spec, "internal_J_learnable", True))

            # Update learnability if connection exists in core
            if internal_key in core.connections:
                core.connections[internal_key].requires_grad = internal_learnable
                # Also update the ConnectionConfig learnability flag
                for conn_cfg in core.connections_config:
                    if conn_cfg.from_layer == layer_id and conn_cfg.to_layer == layer_id:
                        conn_cfg.learnable = internal_learnable
                        break
            # If connection doesn't exist yet, it will be created with correct learnability during build
            # But we should also ensure the weights are copied if the connection exists
            if internal_key in core.connections:
                try:
                    core.connections[internal_key].data.copy_(_torch_from_jax_array(spec.internal_J))
                except Exception:
                    pass

    # Assign per-layer parameter vectors using ParameterRegistry
    layer_by_id = {cfg.layer_id: (i, layer) for i, (cfg, layer) in enumerate(zip(core.layers_config, core.layers, strict=False))}
    for spec in jax_model.layers:
        entry = layer_by_id.get(int(spec.layer_id))
        if entry is None:
            continue
        _, layer = entry
        reg = getattr(layer, "_param_registry", None)
        if reg is None:
            # Layer has no parameter registry - skip (e.g., Linear layers)
            continue
        p = dict(spec.params or {})
        for name in ("phi_y", "bias_current", "gamma_plus", "gamma_minus", "phi_offset", "scale_factor", "beta"):
            if name in p and p[name] is not None:
                # FAIL-FAST: Don't suppress exceptions during parameter transfer
                try:
                    reg.override_parameter(name, value=_torch_from_jax_array(p[name]))
                except Exception as e:
                    msg = f"Failed to transfer JAX parameter '{name}' to Torch layer {spec.layer_id} ({spec.kind}). This indicates a bug in the JAX→Torch conversion. Error: {e}"
                    raise RuntimeError(msg) from e

    # Assign MinGRU weights if present (overrides any initializer)
    idx_by_id: dict[int, int] = {cfg.layer_id: i for i, cfg in enumerate(core.layers_config)}
    for spec in jax_model.layers:
        if _layer_type_from_kind(spec.kind) != "MinGRU":
            continue
        i = idx_by_id.get(int(spec.layer_id))
        if i is None:
            continue
        layer = core.layers[i]
        p = dict(spec.params or {})
        try:
            if "W_hidden" in p:
                layer.hidden_proj.weight.data.copy_(_torch_from_jax_array(p["W_hidden"]))
            if "W_gate" in p:
                layer.gate_proj.weight.data.copy_(_torch_from_jax_array(p["W_gate"]))
        except Exception:
            pass

    return core


__all__ = [
    "convert_core_model_to_jax",
    "convert_file_to_jax",
    "convert_jax_to_core_model",
]
