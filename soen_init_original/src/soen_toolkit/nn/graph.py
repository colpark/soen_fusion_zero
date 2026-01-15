"""Graph: PyTorch-style imperative model builder for SOEN networks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

import torch
from torch import nn

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)

from .layers import LayerFactory
from .param_specs import ParamSpec
from .specs import ConnectionSpec, DynamicSpec, DynamicV2Spec, InitSpec, LayerSpec, StructureSpec

if TYPE_CHECKING:
    from collections.abc import Iterator


class Graph(nn.Module):
    """PyTorch-style graph container for SOEN models.

    Provides an imperative API for building SOEN networks while maintaining full
    compatibility with the existing spec-based system. Under the hood, Graph compiles
    to SimulationConfig, LayerConfig, and ConnectionConfig objects.

    Example:
        >>> from soen_toolkit.nn import Graph, layers, init, structure
        >>>
        >>> g = Graph(dt=37, network_evaluation_method="layerwise")
        >>> g.add_layer(0, layers.Linear(dim=10))
        >>> g.add_layer(1, layers.SingleDendrite(
        ...     dim=5, solver="FE", source_func_type="RateArray",
        ...     bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3
        ... ))
        >>> g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())
        >>>
        >>> output = g(input_tensor)

    """

    def __init__(
        self,
        dt: float = 37.0,
        dt_learnable: bool = False,
        network_evaluation_method: str = "layerwise",
        input_type: str = "state",
        track_phi: bool = False,
        track_g: bool = False,
        track_s: bool = False,
        track_power: bool = False,
        early_stopping_forward_pass: bool = False,
        early_stopping_tolerance: float = 1e-6,
        early_stopping_patience: int = 1,
        steady_window_min: int = 50,
        steady_tol_abs: float = 1e-5,
        steady_tol_rel: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        """Initialize Graph container.

        Args:
            dt: Time step size (dimensionless)
            dt_learnable: Whether dt is trainable
            network_evaluation_method: Network evaluation method ("layerwise", "stepwise_gauss_seidel", "stepwise_jacobi")
            input_type: How to interpret layer 0 ("state" or "flux")
            track_phi: Record input flux history
            track_g: Record source function values
            track_s: Record state trajectories
            track_power: Track power consumption (SingleDendrite only)
            early_stopping_forward_pass: Enable early stopping
            early_stopping_tolerance: Tolerance for early stopping
            early_stopping_patience: Patience for early stopping
            steady_window_min: Window size for steady-state detection
            steady_tol_abs: Absolute tolerance for steady-state
            steady_tol_rel: Relative tolerance for steady-state
            **kwargs: Additional SimulationConfig parameters

        """
        super().__init__()

        # Store simulation config parameters
        self._sim_params = {
            "dt": dt,
            "dt_learnable": dt_learnable,
            "network_evaluation_method": network_evaluation_method,
            "input_type": input_type,
            "track_phi": track_phi,
            "track_g": track_g,
            "track_s": track_s,
            "track_power": track_power,
            "early_stopping_forward_pass": early_stopping_forward_pass,
            "early_stopping_tolerance": early_stopping_tolerance,
            "early_stopping_patience": early_stopping_patience,
            "steady_window_min": steady_window_min,
            "steady_tol_abs": steady_tol_abs,
            "steady_tol_rel": steady_tol_rel,
            **kwargs,
        }

        # Layer and connection specs (not compiled yet)
        self._layer_specs: dict[int, LayerSpec] = {}
        self._connection_specs: list[ConnectionSpec] = []

        # Compiled core model (created on first forward or explicit compile())
        self._compiled_core: SOENModelCore | None = None

        # History attributes (populated after forward if tracking enabled)
        self.phi_history: list[torch.Tensor] | None = None
        self.s_history: list[torch.Tensor] | None = None
        self.g_history: list[torch.Tensor] | None = None
        self.power_history: list[torch.Tensor] | None = None

        # Device/dtype tracking for lazy compilation
        self._target_device: torch.device | None = None
        self._target_dtype: torch.dtype | None = None

    def add_layer(
        self,
        layer_id: int,
        layer: LayerFactory,
        learnable_params: dict[str, bool] | None = None,
        description: str = "",
    ) -> Graph:
        """Add a layer to the graph.

        Args:
            layer_id: Unique integer ID for this layer (layer 0 is always input)
            layer: Layer specification from soen_toolkit.nn.layers
            learnable_params: Dict mapping parameter names to learnable flags
            description: Optional description

        Returns:
            self for method chaining

        Example:
            >>> g.add_layer(0, layers.Linear(dim=10))
            >>> g.add_layer(1, layers.SingleDendrite(dim=5, ...))

        """
        if layer_id in self._layer_specs:
            msg = f"Layer ID {layer_id} already exists"
            raise ValueError(msg)

        if not isinstance(layer, LayerFactory):
            msg = f"layer must be a LayerFactory from soen_toolkit.nn.layers, got {type(layer)}"
            raise TypeError(msg)

        self._layer_specs[layer_id] = LayerSpec(
            layer_id=layer_id,
            module=None,
            layer_type=layer.layer_type,
            params=layer.params.copy(),
            learnable_params=learnable_params,
            description=description,
        )

        # Invalidate compiled core since topology changed
        self._compiled_core = None

        return self

    def connect(
        self,
        from_layer: int,
        to_layer: int,
        structure: StructureSpec,
        init: InitSpec,
        mode: str = "fixed",
        connection_params: dict[str, Any] | None = None,
        constraints: dict[str, float] | None = None,
        learnable: bool = True,
        allow_self_connections: bool = True,
        visualization_metadata: dict | None = None,
        dynamic: DynamicSpec | DynamicV2Spec | None = None,  # Deprecated - kept for backward compatibility
    ) -> Graph:
        """Add a connection between layers.

        Args:
            from_layer: Source layer ID
            to_layer: Target layer ID
            structure: Connectivity structure (from soen_toolkit.nn.structure)
            init: Weight initialization (from soen_toolkit.nn.init)
            mode: Connection mode - "fixed" (default), "WICC" (v1), or "NOCC" (v2)
            connection_params: Optional dict of connection parameters (e.g., {"alpha": 1.5, "beta": 303.85})
            constraints: Weight constraints {"min": ..., "max": ...}
            learnable: Whether connection weights are trainable
            allow_self_connections: Allow self-connections for internal connections
            visualization_metadata: Optional metadata dict for visualization purposes.
                Can also be specified in the structure spec. If provided in both places,
                this parameter takes precedence.
            dynamic: [DEPRECATED] Use mode + connection_params instead

        Returns:
            self for method chaining

        Examples:
            Fixed connection (default):
            >>> g.connect(
            ...     0, 1,
            ...     structure=structure.dense(),
            ...     init=init.xavier_uniform()
            ... )

            WICC with defaults:
            >>> g.connect(
            ...     0, 1,
            ...     structure=structure.dense(),
            ...     init=init.xavier_uniform(),
            ...     mode="WICC"
            ... )

            NOCC with custom parameters:
            >>> g.connect(
            ...     0, 1,
            ...     structure=structure.dense(),
            ...     init=init.uniform(-0.15, 0.15),
            ...     mode="NOCC",
            ...     connection_params={"alpha": 1.5, "beta": 303.85}
            ... )

        """
        if from_layer not in self._layer_specs:
            msg = f"Source layer {from_layer} does not exist"
            raise ValueError(msg)
        if to_layer not in self._layer_specs:
            msg = f"Target layer {to_layer} does not exist"
            raise ValueError(msg)

        # Handle backward compatibility with deprecated 'dynamic' parameter
        if dynamic is not None:
            warnings.warn(
                "The 'dynamic' parameter is deprecated. Use 'mode' with optional 'connection_params' instead. Example: mode='NOCC', connection_params={'alpha': 1.5, 'beta': 303.85}",
                DeprecationWarning,
                stacklevel=2,
            )
            # Convert dynamic spec to new format
            if isinstance(dynamic, DynamicV2Spec):
                if mode == "fixed":  # Only override if mode wasn't explicitly set
                    mode = "NOCC"
                if connection_params is None:
                    connection_params = dynamic.to_dict()
            elif isinstance(dynamic, DynamicSpec):
                if mode == "fixed":  # Only override if mode wasn't explicitly set
                    mode = "WICC"
                if connection_params is None:
                    connection_params = dynamic.to_dict()

        # Accept legacy mode names for backwards compatibility
        mode_aliases = {
            "dynamic": "WICC",
            "dynamic_v1": "WICC",
            "multiplier": "WICC",
            "v1": "WICC",
            "dynamic_v2": "NOCC",
            "multiplier_v2": "NOCC",
            "v2": "NOCC",
        }
        if mode and mode in mode_aliases:
            mode = mode_aliases[mode]

        # Validate mode
        if mode not in {"fixed", "WICC", "NOCC"}:
            msg = f"Invalid connection mode '{mode}'. Valid modes: 'fixed', 'WICC' (With Collection Coil), 'NOCC' (No Collection Coil)."
            raise ValueError(msg)

        # Merge visualization_metadata: parameter takes precedence over structure spec
        final_viz_metadata = structure.visualization_metadata or {}
        if visualization_metadata:
            final_viz_metadata = {**final_viz_metadata, **visualization_metadata}

        self._connection_specs.append(
            ConnectionSpec(
                from_layer=from_layer,
                to_layer=to_layer,
                structure=structure,
                init=init,
                mode=mode,
                dynamic=None,  # Store in new format
                connection_params=connection_params,
                constraints=constraints,
                learnable=learnable,
                allow_self_connections=allow_self_connections,
                visualization_metadata=final_viz_metadata if final_viz_metadata else None,
            )
        )

        # Invalidate compiled core
        self._compiled_core = None

        return self

    def compile(self) -> SOENModelCore:
        """Compile specs to SOENModelCore.

        Converts all layer and connection specifications to the standard config
        format and builds the core model.

        Returns:
            The compiled SOENModelCore

        """
        if len(self._layer_specs) == 0:
            msg = "Cannot compile empty graph - add layers first"
            raise RuntimeError(msg)

        # Build SimulationConfig
        sim_config = SimulationConfig(**self._sim_params)

        # Build LayerConfigs
        layer_configs = []
        for layer_id in sorted(self._layer_specs.keys()):
            spec = self._layer_specs[layer_id]

            # Convert ParamSpec objects to dicts and collect learnable flags
            params = {}
            learnable_from_specs = {}

            for key, value in spec.params.items():
                if isinstance(value, ParamSpec):
                    params[key] = value.to_dict()
                    # Extract learnable flag if present
                    if value.learnable is not None:
                        learnable_from_specs[key] = value.learnable
                else:
                    params[key] = value

            # Merge learnable_params: spec.learnable_params takes precedence
            final_learnable = learnable_from_specs.copy()
            if spec.learnable_params is not None:
                final_learnable.update(spec.learnable_params)

            if final_learnable:
                params["learnable_params"] = final_learnable

            layer_configs.append(
                LayerConfig(
                    layer_id=spec.layer_id,
                    layer_type=spec.layer_type,
                    params=params,
                    description=spec.description,
                )
            )

        # Build ConnectionConfigs
        connection_configs = []
        for conn_spec in self._connection_specs:
            # Build connection params dict
            conn_params: dict[str, Any] = {}

            # Add structure
            conn_params["structure"] = conn_spec.structure.to_dict()

            # Add init
            conn_params["init"] = conn_spec.init.to_dict()

            # Add allow_self_connections
            conn_params["allow_self_connections"] = conn_spec.allow_self_connections

            # Add mode and connection_params if applicable
            if conn_spec.mode in ("WICC", "NOCC"):
                conn_params["mode"] = conn_spec.mode
                # Use new connection_params format, or fall back to legacy dynamic
                if conn_spec.connection_params:
                    conn_params["connection_params"] = conn_spec.connection_params
                elif conn_spec.dynamic:  # Legacy backward compatibility
                    conn_params["dynamic"] = conn_spec.dynamic.to_dict()

            # Add constraints if provided
            if conn_spec.constraints:
                conn_params["constraints"] = conn_spec.constraints

            # Determine connection_type from structure
            connection_type = conn_spec.structure.type

            connection_configs.append(
                ConnectionConfig(
                    from_layer=conn_spec.from_layer,
                    to_layer=conn_spec.to_layer,
                    connection_type=connection_type,
                    params=conn_params,
                    learnable=conn_spec.learnable,
                    visualization_metadata=conn_spec.visualization_metadata,
                )
            )

        # Build core model
        core = SOENModelCore(
            sim_config=sim_config,
            layers_config=layer_configs,
            connections_config=connection_configs,
        )

        # Move to target device/dtype if set
        if self._target_device is not None:
            core = core.to(self._target_device)
        if self._target_dtype is not None:
            core = core.to(self._target_dtype)

        self._compiled_core = core
        return core

    def forward(
        self,
        x: torch.Tensor,
        initial_states: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor (batch, time, input_dim)
            initial_states: Optional initial states per layer

        Returns:
            Output tensor from the last layer

        """
        # Compile if not already done
        if self._compiled_core is None:
            self.compile()

        assert self._compiled_core is not None

        # Run forward pass
        output, histories = self._compiled_core(x, initial_states=initial_states)

        # Populate history attributes if tracking enabled
        if self._sim_params.get("track_phi", False):
            self.phi_history = self._compiled_core.get_phi_history()

        if self._sim_params.get("track_g", False):
            self.g_history = self._compiled_core.get_g_history()

        if self._sim_params.get("track_s", False):
            self.s_history = histories

        if self._sim_params.get("track_power", False):
            self.power_history = self._compiled_core.get_power_history()

        return output

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return an iterator over module parameters.

        Args:
            recurse: Whether to recurse into submodules

        Yields:
            Parameters from the compiled core

        """
        if self._compiled_core is not None:
            yield from self._compiled_core.parameters(recurse=recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Return an iterator over module parameters, yielding name and parameter.

        Args:
            prefix: Prefix to prepend to parameter names
            recurse: Whether to recurse into submodules
            remove_duplicate: Whether to remove duplicate parameters

        Yields:
            (name, parameter) tuples

        """
        if self._compiled_core is not None:
            yield from self._compiled_core.named_parameters(
                prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
            )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Return state dict from compiled core.

        Returns:
            State dictionary

        """
        if self._compiled_core is None:
            return {}
        return self._compiled_core.state_dict(*args, **kwargs)

    def load_state_dict(  # type: ignore[override]
        self, state_dict: dict[str, Any], strict: bool = True
    ) -> Any:
        """Load state dict into compiled core.

        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly enforce key matching

        Returns:
            Result from the underlying load_state_dict call

        """
        if self._compiled_core is None:
            self.compile()
        assert self._compiled_core is not None
        return self._compiled_core.load_state_dict(state_dict, strict=strict)

    def to(self, *args: Any, **kwargs: Any) -> Graph:
        """Move graph to device/dtype.

        Returns:
            self

        """
        # Parse device/dtype from args/kwargs
        device = None
        dtype = None

        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, str):
                device = torch.device(arg)

        if "device" in kwargs:
            device = kwargs["device"]
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]

        # Store for lazy compilation
        if device is not None:
            self._target_device = device
        if dtype is not None:
            self._target_dtype = dtype

        # Move compiled core if it exists
        if self._compiled_core is not None:
            self._compiled_core = self._compiled_core.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def save(self, path: str, **kwargs: Any) -> None:
        """Save model to file.

        Args:
            path: Path to save to (.soen, .pth, or .json)
            **kwargs: Additional arguments passed to core.save()

        Example:
            >>> g.save("my_model.soen")
            >>> g.save("my_model.json")

        """
        if self._compiled_core is None:
            self.compile()
        assert self._compiled_core is not None
        self._compiled_core.save(path, **kwargs)

    def visualize(self, **kwargs: Any) -> str:
        """Visualize the network architecture.

        Args:
            save_path: Base path (no extension) for saving the diagram
            file_format: "png" (default), "svg", "jpg", or "pdf"
            dpi: DPI for raster outputs (default: 300)
            open_file: If True, opens the saved file (default: False)
            orientation: "LR" (left-right) or "TB" (top-bottom)
            simple_view: If True, compact layer boxes
            if False, detailed view
            show_descriptions: Show layer descriptions
            theme: "default", "dark", or custom theme name
            **kwargs: Additional visualization options

        Returns:
            str: Full file path of the saved diagram

        Example:
            >>> g.visualize(save_path="my_network", file_format="svg")
            >>> g.visualize(orientation="TB", simple_view=False)

        """
        if self._compiled_core is None:
            self.compile()
        assert self._compiled_core is not None
        return self._compiled_core.visualize(**kwargs)

    def visualize_grid_of_grids(self, **kwargs: Any) -> str | None:
        """Visualize network as a grid-of-grids (each layer as a module).

        Displays each network layer as a module (colored box) arranged in a grid.
        Nodes are shown as small squares within each module, with connections drawn between them.
        Optionally supports hierarchical block structure visualization for networks created with
        create_hierarchical_mask().

        Args:
            save_path (str, optional): Base path (without extension) for saving visualization.
                If None, only displays inline. Default: None
            file_format (str): Output format - "png" (default), "svg", "jpg", or "pdf".
                Note: PDFs display inline. Default: "png"
            dpi (int): Dots per inch for rasterized output. Default: 200
            show (bool): If True (default), display inline in notebooks. If False, only save to file.
                Default: True

        Module Styling:
            bg_color (str): Background color of canvas. Default: "white"
            module_bg_color (str): Background color for middle-tier modules. Default: "#B8D4D8"
            module_bg_color_input (str): Background color for input (first) layer. Default: "#D6D9DD"
            module_bg_color_output (str): Background color for output (last) layer. Default: "#CFE3FF"
            module_border_color (str): Color of module borders. Default: "#1a3a52" (dark blue)
            module_border_linewidth (float): Width of module border lines. Default: 2.0
            module_border_alpha (float): Transparency of borders, range 0-1. Default: 0.85

        Node Styling:
            node_fill_color (str): Fill color for node squares. Default: "#C89FA3"
            node_edge_color (str): Color of node borders. Default: "white"
            node_edge_linewidth (float): Width of node border lines. Default: 0.8
            node_square_size (float): Size of node squares in plot units. Default: 0.36
                Automatically increased to 0.8 for hierarchical block-structured layers.

        Connection Styling:
            intra_edge_color (str): Color for within-layer connections. Default: "black" (dark theme: "#7aa2f7")
            intra_edge_width (float): Line width for within-layer connections. Default: 1.0
            intra_edge_alpha (float): Transparency for within-layer edges, range 0-1. Default: 0.6
            inter_edge_color (str): Color for between-layer connections. Default: "black" (dark theme: "#e5e5e5")
            inter_edge_width (float): Line width for between-layer connections. Default: 1.8
            inter_edge_alpha (float): Transparency for between-layer edges, range 0-1. Default: 0.5

        Edge Routing:
            edge_routing (str): Style for drawing edges - "straight" (default), "curved", or "orthogonal".
                • "straight": Direct lines from source to destination (default, clearest)
                • "curved": Smooth arc connections
                • "orthogonal": Right-angle paths (subway-map style)
                Default: "straight"
            edge_routing_rad_scale (float): Curvature scale for "curved" routing (default: 0.2).
                Higher values = more curved edges. Only used when edge_routing="curved".

        Theme and Labels:
            theme (str): Visual theme - "default" or "dark". Default: "default"
            font_name (str, optional): Font family for text (e.g., "DejaVu Sans", "Helvetica").
                If None, uses system default. Default: None
            title_font_size (float): Font size for layer labels. Default: 14.0
            show_layer_ids (bool): If True, display layer ID labels (e.g., "L0"). Default: False
            show_layer_types (bool): If True, display layer type (e.g., "SingleDendrite"). Default: False
            show_dimensions (bool): If True, display layer dimensions (e.g., "Dim: 10"). Default: False
            show_descriptions (bool): If True, display layer descriptions. Default: True

            Note: Label elements are stacked vertically below each module in the order: IDs, types,
            dimensions, descriptions. Only enabled elements are shown.

        Hierarchical Block Structure:
            show_block_structure (bool): If True, show hierarchical block organization with nested
                rectangles and reorganize nodes within their blocks. Default: False
            levels (int, optional): Optional global fallback for hierarchical tiers (e.g., 3 for base_size^3 nodes).
                Used for layers without metadata. Per-layer metadata (from structure.custom()) takes precedence.
                Default: None
            base_size (int, optional): Optional global fallback base block size (e.g., 4). Total nodes must equal
                base_size^levels. Used for layers without metadata. Per-layer metadata takes precedence. Default: None
            block_linewidth (float): Width of block boundary lines. Default: 1.0
            block_alpha (float): Base transparency for block boundaries, adjusted per tier, range 0-1.
                Larger tiers get lower alpha. Default: 0.4
            block_colors (list[str], optional): List of hex colors for each tier [tier0, tier1, ...].
                If None, uses gray palette ["#606060", "#808080", "#a0a0a0", "#c0c0c0"].
                Example: ["#FF0000", "#00FF00", "#0000FF"]. Default: None

            Note: Hierarchical structure can be specified per-layer via visualization_metadata:
                g.connect(0, 0, structure=structure.custom("mask.npz",
                    visualization_metadata={"hierarchical": {"levels": 3, "base_size": 4}}))
            This allows different layers to have different hierarchical structures and is fully
            extensible for future visualization features.

        Returns:
            str: Full file path of the saved visualization (if save_path provided), or None if only
                displayed inline.

        Notes:
            - Each layer is treated as a single module (ignores model_id/module_id)
            - Module background color indicates role: input (first), output (last), or middle layer
            - Within-layer edges are drawn from the intra-layer connection matrix
            - Between-layer edges are drawn from inter-layer connection blocks
            - Hierarchical positioning (when show_block_structure=True):
                * Nodes are physically organized within their hierarchical blocks
                * Each block attempts square aspect ratio for visual clarity
                * Nested rectangles show structure of each tier level
                * Useful for visualizing networks from create_hierarchical_mask()

        Example:
            >>> # Basic visualization
            >>> g.visualize_grid_of_grids()

            >>> # Save to file with custom styling
            >>> path = g.visualize_grid_of_grids(
            ...     save_path="my_network",
            ...     module_border_linewidth=3.0,
            ...     inter_edge_width=2.0
            ... )

            >>> # With hierarchical block structure
            >>> g.visualize_grid_of_grids(
            ...     show_block_structure=True,
            ...     levels=3,
            ...     base_size=4,
            ...     block_colors=["#FF6B6B", "#4ECDC4", "#95E1D3"],
            ...     save_path="hierarchical_network"
            ... )

            >>> # Dark theme with custom colors
            >>> g.visualize_grid_of_grids(
            ...     theme="dark",
            ...     module_bg_color="#2a2e35",
            ...     inter_edge_color="#e5e5e5"
            ... )

        """
        from soen_toolkit.utils.visualization import visualize_grid_of_grids

        if self._compiled_core is None:
            self.compile()
        assert self._compiled_core is not None
        return visualize_grid_of_grids(self._compiled_core, **kwargs)

    def summary(
        self,
        return_df: bool = False,
        print_summary: bool = True,
        create_histograms: bool = False,
        verbose: bool = False,
        dpi: int = 300,
        notebook_view: bool = False,
    ) -> Any:
        """Print or return a summary of the model.

        Args:
            return_df: If True, returns a pandas DataFrame
            print_summary: If True, prints the summary to console
            create_histograms: If True, creates parameter histograms
            verbose: If True, includes more detailed information
            dpi: DPI for histogram plots
            notebook_view: If True, renders nicely formatted markdown in Jupyter notebooks

        Returns:
            Optional[pd.DataFrame]: Summary dataframe if return_df=True

        Example:
            >>> g.summary()
            >>> g.summary(notebook_view=True)  # Nice tables in Jupyter
            >>> df = g.summary(return_df=True, print_summary=False)

        """
        if self._compiled_core is None:
            self.compile()
        assert self._compiled_core is not None
        return self._compiled_core.summary(
            return_df=return_df,
            print_summary=print_summary,
            create_histograms=create_histograms,
            verbose=verbose,
            dpi=dpi,
            notebook_view=notebook_view,
        )

    def compute_summary(self) -> dict[str, Any]:
        """Compute and return summary statistics as a dictionary.

        Returns:
            Dict[str, Any]: Summary statistics including layer info, parameters, etc.

        Example:
            >>> stats = g.compute_summary()
            >>> print(f"Total parameters: {stats['total_parameters']}")

        """
        if self._compiled_core is None:
            self.compile()
        assert self._compiled_core is not None
        return self._compiled_core.compute_summary()

    @classmethod
    def load(cls, path: str) -> Graph:
        """Load model from file.

        Args:
            path: Path to load from

        Returns:
            Graph instance with loaded core

        """
        core = SOENModelCore.load(path)
        return cls.from_core(core)

    @classmethod
    def from_core(cls, core: SOENModelCore) -> Graph:
        """Create Graph from an existing SOENModelCore.

        Args:
            core: Existing SOENModelCore instance

        Returns:
            Graph wrapping the core

        """
        # Extract sim config params
        sim_config = core.sim_config
        sim_params: dict[str, Any] = {
            "dt": core.dt if isinstance(core.dt, float) else core.dt.item(),
            "dt_learnable": getattr(core, "dt_learnable", False),
            "network_evaluation_method": sim_config.network_evaluation_method,
            "input_type": sim_config.input_type,
            "track_phi": sim_config.track_phi,
            "track_g": sim_config.track_g,
            "track_s": sim_config.track_s,
            "track_power": sim_config.track_power,
            "early_stopping_forward_pass": sim_config.early_stopping_forward_pass,
            "early_stopping_tolerance": sim_config.early_stopping_tolerance,
            "early_stopping_patience": sim_config.early_stopping_patience,
            "steady_window_min": sim_config.steady_window_min,
            "steady_tol_abs": sim_config.steady_tol_abs,
            "steady_tol_rel": sim_config.steady_tol_rel,
        }

        graph = cls(**sim_params)
        graph._compiled_core = core

        # Note: We don't reconstruct layer_specs and connection_specs from the core
        # The graph is in "compiled-only" mode

        return graph

    @classmethod
    def from_yaml(cls, path: str) -> Graph:
        """Build Graph from YAML spec file.

        Args:
            path: Path to YAML file

        Returns:
            Graph instance

        """
        core = SOENModelCore.build(path)
        return cls.from_core(core)

    def reset_parameters(self) -> None:
        """Reset all parameters using stored initialization specs.

        This rebuilds the connections with their configured initializers.
        """
        if self._compiled_core is not None:
            # For now, just recompile - this will reinit everything
            # Could be optimized to only reinit without rebuilding topology
            self._compiled_core = None
            self.compile()
        else:
            warnings.warn("No compiled core to reset - compile first", stacklevel=2)

    def __repr__(self) -> str:
        """String representation."""
        num_layers = len(self._layer_specs)
        num_connections = len(self._connection_specs)
        compiled = "compiled" if self._compiled_core is not None else "not compiled"
        return f"Graph(layers={num_layers}, connections={num_connections}, {compiled})"
