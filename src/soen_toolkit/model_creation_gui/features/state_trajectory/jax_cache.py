"""JAX model conversion and caching with parameter fingerprinting."""

from __future__ import annotations

import hashlib
import logging
import traceback

from .errors import SimulationError

logger = logging.getLogger(__name__)


class JaxModelCache:
    """Cache for JAX models with param/shape fingerprinting.

    This cache:
    - Converts Torch model to JAX on first use
    - JIT-compiles the JAX apply function
    - Tracks parameter fingerprint to detect changes
    - Invalidates cache when model parameters change
    - Tracks warm shapes to skip redundant warmup
    """

    def __init__(self, torch_model):
        """Initialize JAX cache.

        Args:
            torch_model: Source Torch model to convert
        """
        self.torch_model = torch_model
        self._jax_model = None
        self._jax_apply = None
        self._param_fingerprint: str | None = None
        self._warm_shapes = set()

    def _compute_fingerprint(self) -> str:
        """Compute SHA256 fingerprint of model parameters.

        This fingerprint changes whenever weights/biases change, triggering
        cache invalidation and reconversion to JAX.

        Returns:
            Hex digest of parameter fingerprint
        """
        state = self.torch_model.state_dict()
        h = hashlib.sha256()

        # Hash parameters in sorted order for consistency
        for key in sorted(state.keys()):
            h.update(key.encode("utf-8"))
            tensor = state[key]
            h.update(tensor.cpu().numpy().tobytes())

        return h.hexdigest()

    def invalidate(self) -> None:
        """Invalidate cache, forcing rebuild on next use."""
        self._jax_model = None
        self._jax_apply = None
        self._param_fingerprint = None
        self._warm_shapes.clear()

    def get_apply(self, progress_callback=None):
        """Get or build JAX apply function.

        Returns the JIT-compiled JAX apply function, building/rebuilding as needed
        when parameters change.

        Args:
            progress_callback: Optional callback(message: str) for progress updates

        Returns:
            JIT-compiled JAX apply function

        Raises:
            SimulationError: If JAX not available or conversion fails
        """
        # Check if cache is valid
        current_fp = self._compute_fingerprint()
        cache_valid = self._jax_apply is not None and self._param_fingerprint == current_fp

        if cache_valid:
            return self._jax_apply

        # Cache invalid, need to rebuild
        if progress_callback:
            progress_callback("Converting model to JAX...")

        try:
            # Import JAX (lazy)
            import jax
        except ImportError as e:
            msg = "JAX is not available. Install JAX to use the JAX backend."
            raise SimulationError(msg) from e

        try:
            # Force CPU backend
            import os

            os.environ.setdefault("JAX_PLATFORMS", "cpu")
            try:
                jax.config.update("jax_platforms", "cpu")
                jax.config.update("jax_enable_x64", False)
            except Exception:
                pass

            # Convert using model's built-in port_to_jax method
            if progress_callback:
                progress_callback("Running port_to_jax()...")

            # Use the model's port_to_jax() method which handles conversion
            self._jax_model = self.torch_model.port_to_jax(prepare=True)

            if progress_callback:
                progress_callback("JIT-compiling apply function...")

            # JIT compile the apply function
            # Important: forward output structure changes when `return_trace=True`.
            # Mark these flags static so JAX compiles separate variants safely.
            self._jax_apply = jax.jit(
                self._jax_model.__call__,
                static_argnames=("return_trace", "track_phi", "track_g"),
            )

            # Update cache metadata
            self._param_fingerprint = current_fp
            self._warm_shapes.clear()

            if progress_callback:
                progress_callback("JAX model ready")

            return self._jax_apply

        except NotImplementedError as e:
            logger.error("Model cannot be converted to JAX", exc_info=True)
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            msg = f"Model cannot be converted to JAX: {e}"
            raise SimulationError(msg) from e
        except Exception as e:
            logger.error("JAX conversion failed", exc_info=True)
            logger.error(f"Conversion error type: {type(e).__name__}")
            logger.error(f"Conversion error message: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            msg = f"JAX conversion failed: {e}"
            raise SimulationError(msg) from e

    def warmup_shape(self, shape_key: tuple) -> bool:
        """Check if shape has been warmed up and mark it.

        Args:
            shape_key: Tuple representing input shape (e.g., (batch, seq_len, dim))

        Returns:
            True if shape was already warm, False if this is first time
        """
        if shape_key in self._warm_shapes:
            return True
        self._warm_shapes.add(shape_key)
        return False
