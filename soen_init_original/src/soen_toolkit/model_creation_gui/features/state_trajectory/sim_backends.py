"""Simulation backend runners for Torch and JAX."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import logging
import random
import time
import traceback

import numpy as np
import torch

from .errors import SimulationError

logger = logging.getLogger(__name__)


class BackendRunner(ABC):
    """Abstract base class for simulation backends."""

    @abstractmethod
    def run(
        self,
        x: torch.Tensor,
        metric: str,
        include_s0: bool,
        initial_states: dict[int, torch.Tensor] | None = None,
        s1_states: dict[int, torch.Tensor] | None = None,
        s2_states: dict[int, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
        """Run simulation and return histories.

        Args:
            x: Input tensor [batch, seq_len, dim]
            metric: Metric to collect (state, phi, g, power, energy)
            include_s0: Whether to include initial state in results
            initial_states: Optional dict mapping layer_idx to initial state [1, 1, D]
                          for state carryover between samples
            s1_states: Optional dict mapping layer_idx to s1 auxiliary state [1, 1, D]
            s2_states: Optional dict mapping layer_idx to s2 auxiliary state [1, 1, D]

        Returns:
            Tuple of (metric_histories, raw_state_histories, elapsed_seconds)
        """
        ...


class TorchRunner(BackendRunner):
    """Torch backend with deterministic execution context."""

    def __init__(self, model, model_adapter):
        """Initialize Torch runner.

        Args:
            model: SOENModelCore instance
            model_adapter: ModelAdapter for operations
        """
        self.model = model
        self.adapter = model_adapter
        # Cache for final states from last run (for state carryover)
        self._last_raw_histories: list[torch.Tensor] | None = None

    @contextmanager
    def _deterministic_context(self, seed: int):
        """Set up deterministic execution context.

        This context manager:
        - Seeds all RNGs (random, numpy, torch)
        - Sets model to eval mode
        - Wraps forward in torch.no_grad()
        - Resets model state/histories
        - Restores training mode after execution

        Args:
            seed: Random seed for reproducibility
        """
        # Seed all RNGs
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Enable deterministic algorithms if available
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # Not available in all PyTorch versions

        # Save training state and switch to eval
        was_training = bool(self.model.training)
        self.model.eval()

        # Reset model state
        try:
            self.adapter.reset_state(self.model)
        except Exception:
            pass  # Non-fatal if reset not available

        try:
            with torch.no_grad():
                yield
        finally:
            # Restore training mode if it was on
            if was_training:
                try:
                    self.model.train()
                except Exception:
                    pass

            # Restore non-deterministic behavior
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass

    def run(
        self,
        x: torch.Tensor,
        metric: str,
        include_s0: bool,
        initial_states: dict[int, torch.Tensor] | None = None,
        s1_states: dict[int, torch.Tensor] | None = None,
        s2_states: dict[int, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
        """Run Torch simulation with deterministic context.

        Args:
            x: Input tensor [batch, seq_len, dim]
            metric: Metric to collect
            include_s0: Whether to include initial state
            initial_states: Optional dict mapping layer_idx to initial state [1, 1, D]
            s1_states: Optional dict mapping layer_idx to s1 auxiliary state [1, 1, D]
            s2_states: Optional dict mapping layer_idx to s2 auxiliary state [1, 1, D]

        Returns:
            Tuple of (metric_histories, raw_state_histories, elapsed_seconds)

        Raises:
            SimulationError: If simulation fails
        """
        # Get seed from model or use default
        try:
            seed = int(getattr(self.model, "_creation_seed", None) or 1337)
        except (ValueError, TypeError):
            seed = 1337

        try:
            with self._deterministic_context(seed):
                t0 = time.time()
                kwargs = {}
                if initial_states:
                    kwargs["initial_states"] = {k: v.squeeze(1) for k, v in initial_states.items()}
                if s1_states:
                    kwargs["s1_inits"] = {k: v.squeeze(1) for k, v in s1_states.items()}
                if s2_states:
                    kwargs["s2_inits"] = {k: v.squeeze(1) for k, v in s2_states.items()}

                try:
                    if kwargs:
                        _, raw_state_histories = self.model(x, **kwargs)
                    else:
                        _, raw_state_histories = self.model(x)
                except TypeError:
                    # Model doesn't support passing initial states; ignore kwargs
                    _, raw_state_histories = self.model(x)
                elapsed = time.time() - t0

            # Collect metric histories
            metric_histories = self.adapter.collect_metric_histories(self.model, metric, include_s0, raw_state_histories)

            # Cache raw histories for get_last_states
            self._last_raw_histories = raw_state_histories

            return metric_histories, raw_state_histories, elapsed

        except Exception as e:
            logger.error("Torch simulation failed", exc_info=True)
            msg = f"Torch simulation failed: {e}"
            raise SimulationError(msg) from e

    def get_last_states(self) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Extract final states from last run with correct layer_id mapping.

        Returns:
            Tuple of (main_states, s1_states, s2_states) dicts mapping layer_id -> tensor.
            s1/s2 states are empty for now (Torch doesn't expose them yet).
        """
        if self._last_raw_histories is None:
            return {}, {}, {}

        # Map list index to layer_id using layers_config
        main_states: dict[int, torch.Tensor] = {}
        for idx, hist in enumerate(self._last_raw_histories):
            if hist is not None and idx < len(self.model.layers_config):
                layer_id = self.model.layers_config[idx].layer_id
                # Extract final timestep: [B, T+1, D] -> [B, 1, D]
                main_states[layer_id] = hist[:, -1:, :].detach().clone()

        return main_states, {}, {}


class JaxRunner(BackendRunner):
    """JAX backend with JIT compilation and caching."""

    def __init__(self, jax_cache, model_adapter):
        """Initialize JAX runner.

        Args:
            jax_cache: JaxModelCache instance for conversion/caching
            model_adapter: ModelAdapter for operations
        """
        self.cache = jax_cache
        self.adapter = model_adapter
        self._last_main_states: dict[int, torch.Tensor] | None = None
        self._last_s1_states: dict[int, torch.Tensor] | None = None
        self._last_s2_states: dict[int, torch.Tensor] | None = None

    def run(
        self,
        x: torch.Tensor,
        metric: str,
        include_s0: bool,
        initial_states: dict[int, torch.Tensor] | None = None,
        s1_states: dict[int, torch.Tensor] | None = None,
        s2_states: dict[int, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
        """Run JAX simulation with JIT compilation.

        Args:
            x: Input tensor [batch, seq_len, dim]
            metric: Metric to collect (currently only 'state' supported)
            include_s0: Whether to include initial state
            initial_states: Optional initial states (currently not supported in JAX backend)

        Returns:
            Tuple of (metric_histories, raw_state_histories, elapsed_seconds)

        Raises:
            SimulationError: If JAX not available or simulation fails
        """
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as e:
            msg = "JAX is not available. Install JAX to use the JAX backend."
            raise SimulationError(msg) from e

        try:
            # Ensure CPU backend
            try:
                jax.config.update("jax_platforms", "cpu")
            except Exception:
                pass

            # Get or build compiled apply function
            apply_fn = self.cache.get_apply()

            # Convert input to JAX array
            x_np = x.detach().cpu().numpy()
            arr = jnp.asarray(x_np)

            # Prepare initial states (main state only for now)
            init_states_jax = None
            if initial_states:
                init_states_jax = {}
                for layer_id, tensor in initial_states.items():
                    try:
                        state_np = tensor.detach().cpu().numpy()
                    except Exception:
                        state_np = np.array(tensor, dtype=np.float32)
                    # Ensure shape [B, D]
                    if state_np.ndim == 3 and state_np.shape[1] == 1:
                        state_np = state_np[:, 0, :]
                    elif state_np.ndim == 1:
                        state_np = state_np[None, :]
                    init_states_jax[int(layer_id)] = jnp.asarray(state_np, dtype=arr.dtype)
                if not init_states_jax:
                    init_states_jax = None

            s1_states_jax = None
            if s1_states:
                s1_states_jax = {}
                for layer_id, tensor in s1_states.items():
                    try:
                        state_np = tensor.detach().cpu().numpy()
                    except Exception:
                        state_np = np.array(tensor, dtype=np.float32)
                    if state_np.ndim == 3 and state_np.shape[1] == 1:
                        state_np = state_np[:, 0, :]
                    elif state_np.ndim == 1:
                        state_np = state_np[None, :]
                    s1_states_jax[int(layer_id)] = jnp.asarray(state_np, dtype=arr.dtype)
                if not s1_states_jax:
                    s1_states_jax = None

            s2_states_jax = None
            if s2_states:
                s2_states_jax = {}
                for layer_id, tensor in s2_states.items():
                    try:
                        state_np = tensor.detach().cpu().numpy()
                    except Exception:
                        state_np = np.array(tensor, dtype=np.float32)
                    if state_np.ndim == 3 and state_np.shape[1] == 1:
                        state_np = state_np[:, 0, :]
                    elif state_np.ndim == 1:
                        state_np = state_np[None, :]
                    s2_states_jax[int(layer_id)] = jnp.asarray(state_np, dtype=arr.dtype)
                if not s2_states_jax:
                    s2_states_jax = None

            # Check if model has connection noise configured
            jax_model = self.cache._jax_model
            has_conn_noise = (
                jax_model.connection_noise_settings is not None
                and len(jax_model.connection_noise_settings) > 0
            )

            # Generate RNG key for noise if needed
            rng_key = None
            if has_conn_noise:
                # Generate a fresh RNG key for each forward pass
                import time as time_mod
                seed = int(time_mod.time() * 1e6) % (2**31)
                rng_key = jax.random.PRNGKey(seed)

            # Warmup pass (to avoid measuring JIT compilation overhead)
            try:
                metric_lower = metric.lower()
                wants_phi = metric_lower.startswith("phi") or metric_lower.startswith("flux")
                warmup_kwargs = {}
                if init_states_jax is not None:
                    warmup_kwargs["initial_states"] = init_states_jax
                if s1_states_jax is not None:
                    warmup_kwargs["s1_inits"] = s1_states_jax
                if s2_states_jax is not None:
                    warmup_kwargs["s2_inits"] = s2_states_jax
                if rng_key is not None:
                    # Use a different key for warmup
                    warmup_kwargs["rng_key"] = jax.random.PRNGKey(0)

                # Always request trace so we can carry auxiliary states (e.g. NOCC s1/s2)
                # without relying on mutating layer objects.
                if wants_phi:
                    _, _, _ = apply_fn(arr, **warmup_kwargs, return_trace=True, track_phi=True)
                else:
                    _, _, _ = apply_fn(arr, **warmup_kwargs, return_trace=True, track_phi=False)
            except Exception:
                pass  # Non-fatal if warmup fails

            # Actual measurement pass (after JIT compilation)
            t0 = time.time()
            run_kwargs = {}
            if init_states_jax is not None:
                run_kwargs["initial_states"] = init_states_jax
            if s1_states_jax is not None:
                run_kwargs["s1_inits"] = s1_states_jax
            if s2_states_jax is not None:
                run_kwargs["s2_inits"] = s2_states_jax
            if rng_key is not None:
                run_kwargs["rng_key"] = rng_key

            # For φ tracking (and future g tracking) we need an opt-in trace.
            metric_lower = metric.lower()
            wants_phi = metric_lower.startswith("phi") or metric_lower.startswith("flux")
            wants_g = metric_lower.startswith("g") or metric_lower.startswith("non")

            if wants_g:
                # Fail fast: we don't yet expose g traces in JAX.
                raise SimulationError(
                    "JAX backend does not currently support g-history tracing. "
                    "Use Torch backend for g-based metrics, or choose metric='state'/'phi'."
                )

            # Always request trace so we can cache final auxiliary states (s1/s2) for carryover.
            if wants_phi:
                final_hist, all_hists, trace = apply_fn(arr, **run_kwargs, return_trace=True, track_phi=True)
            else:
                final_hist, all_hists, trace = apply_fn(arr, **run_kwargs, return_trace=True, track_phi=False)

            # Block until computation completes (important for timing)
            try:
                final_hist.block_until_ready()
            except Exception:
                pass

            elapsed = time.time() - t0

            # Convert JAX arrays back to torch tensors
            raw_state_histories = [torch.tensor(np.array(h), dtype=torch.float32) for h in all_hists]

            # Cache final states for state carryover (main + auxiliary)
            try:
                main_states_jax, s1_states_jax_out, s2_states_jax_out = self.cache._jax_model.extract_final_states(
                    all_hists,
                    s1_final_by_layer=getattr(trace, "s1_final_by_layer", None),
                    s2_final_by_layer=getattr(trace, "s2_final_by_layer", None),
                )

                def _convert_state_dict(jax_dict):
                    out: dict[int, torch.Tensor] = {}
                    for lid, state in jax_dict.items():
                        np_state = np.array(state)
                        if np_state.ndim == 2:
                            np_state = np_state[:, None, :]
                        elif np_state.ndim == 1:
                            np_state = np_state[None, None, :]
                        out[int(lid)] = torch.tensor(np_state, dtype=torch.float32)
                    return out

                self._last_main_states = _convert_state_dict(main_states_jax)
                self._last_s1_states = _convert_state_dict(s1_states_jax_out)
                self._last_s2_states = _convert_state_dict(s2_states_jax_out)
            except Exception:
                self._last_main_states = None
                self._last_s1_states = None
                self._last_s2_states = None

            if wants_phi:
                # trace.phi_by_layer is [B, T, D] per layer
                phi_hists = trace.phi_by_layer
                if phi_hists is None:
                    raise SimulationError(
                        "JAX backend returned no phi trace despite track_phi=True. This is a bug."
                    )
                metric_histories = [torch.tensor(np.array(h), dtype=torch.float32) for h in phi_hists]
            elif metric_lower.startswith("state") or metric_lower.startswith("s"):
                if include_s0:
                    metric_histories = raw_state_histories
                else:
                    metric_histories = [h[:, 1:, :] for h in raw_state_histories]
            else:
                # Unknown metric: be explicit and return raw states (legacy behavior)
                metric_histories = raw_state_histories

            return metric_histories, raw_state_histories, elapsed

        except Exception as e:
            logger.error("JAX simulation failed", exc_info=True)
            # Log additional context about the error
            logger.error(f"JAX simulation error type: {type(e).__name__}")
            logger.error(f"JAX simulation error message: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            msg = f"JAX simulation failed: {e}"
            raise SimulationError(msg) from e

    def get_last_states(self) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Return cached final states from last run."""
        return (
            self._last_main_states or {},
            self._last_s1_states or {},
            self._last_s2_states or {},
        )
