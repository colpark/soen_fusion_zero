from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax

from .unified_forward import forward

if TYPE_CHECKING:
    from .jax_model import JAXModel


class SoenEqxModel(eqx.Module):
    """Equinox wrapper around the existing JAX topology.

    This is Stage-1 Equinox integration:
    - Topology and solver behavior remain in `JAXModel` + `unified_forward`.
    - We introduce an `eqx.Module` entrypoint that can later evolve into a fully
      Equinox-native model (where trainable parameters live as module leaves).
    """

    topology: JAXModel = eqx.field(static=True)
    # Trainable leaves (Stage-2): these are the actual optimizer parameters.
    layer_params: tuple[Any, ...]
    connection_params: jax.Array
    internal_connections: dict[int, jax.Array]

    def __call__(
        self,
        x: jax.Array,
        *,
        initial_states: dict[int, jax.Array] | None = None,
        s1_inits: dict[int, jax.Array] | None = None,
        s2_inits: dict[int, jax.Array] | None = None,
        rng_key: jax.Array | None = None,
        return_trace: bool = False,
        track_phi: bool = False,
        track_g: bool = False,
    ):
        return forward(
            self.topology,
            x,
            initial_states=initial_states,
            s1_inits=s1_inits,
            s2_inits=s2_inits,
            conn_override=self.connection_params,
            internal_conn_override=self.internal_connections,
            layer_param_override=self.layer_params,
            rng_key=rng_key,
            return_trace=return_trace,
            track_phi=track_phi,
            track_g=track_g,
        )

    def as_params_dict(self) -> dict[str, Any]:
        """Return a backward-compatible params dict (for checkpointing/converters)."""
        return {
            "layer_params": self.layer_params,
            "connection_params": self.connection_params,
            "internal_connections": self.internal_connections,
        }


__all__ = ["SoenEqxModel"]

