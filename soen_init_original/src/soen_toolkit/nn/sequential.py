"""Sequential: Convenience wrapper for feedforward SOEN networks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import init, structure
from .graph import Graph

if TYPE_CHECKING:
    from .layers import LayerFactory


class Sequential(Graph):
    """Sequential feedforward network builder.

    Convenience wrapper around Graph that automatically creates dense forward
    connections between consecutive layers. Ideal for simple feedforward models.

    Example:
        >>> from soen_toolkit.nn import Sequential, layers
        >>>
        >>> net = Sequential([
        ...     layers.Linear(dim=10),
        ...     layers.SingleDendrite(dim=50, solver="FE", source_func_type="RateArray",
        ...                          bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3),
        ...     layers.DendriteReadout(dim=5, source_func_type="RateArray", bias_current=1.7)
        ... ])
        >>>
        >>> output = net(input_tensor)

    """

    def __init__(
        self,
        layers: list[LayerFactory] | None = None,
        auto_connect: bool = True,
        connection_init: init.InitSpec | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Sequential network.

        Args:
            layers: List of layer specifications (layer IDs auto-assigned 0, 1, 2, ...)
            auto_connect: Whether to automatically add dense connections between consecutive layers
            connection_init: Initialization for auto-generated connections (default: normal)
            **kwargs: Additional arguments passed to Graph constructor

        """
        # Force layerwise solver for feedforward
        if "network_evaluation_method" not in kwargs:
            kwargs["network_evaluation_method"] = "layerwise"

        super().__init__(**kwargs)

        self._auto_connect = auto_connect
        self._connection_init = connection_init or init.normal()

        if layers is not None:
            for i, layer in enumerate(layers):
                self.add_layer(i, layer)

            # Auto-connect consecutive layers if enabled
            if auto_connect and len(layers) > 1:
                for i in range(len(layers) - 1):
                    self.connect(
                        i,
                        i + 1,
                        structure=structure.dense(),
                        init=self._connection_init,
                    )

    def append(self, layer: LayerFactory) -> Sequential:
        """Append a layer to the end of the sequence.

        Automatically connects it to the previous layer if auto_connect is enabled.

        Args:
            layer: Layer specification to append

        Returns:
            self for method chaining

        """
        new_id = len(self._layer_specs)
        self.add_layer(new_id, layer)

        if self._auto_connect and new_id > 0:
            self.connect(
                new_id - 1,
                new_id,
                structure=structure.dense(),
                init=self._connection_init,
            )

        return self

    def __repr__(self) -> str:
        """String representation."""
        num_layers = len(self._layer_specs)
        compiled = "compiled" if self._compiled_core is not None else "not compiled"
        return f"Sequential(layers={num_layers}, {compiled})"
