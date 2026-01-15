# FILEPATH: src/soen_toolkit/core/configs.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .noise import NoiseSettings


@dataclass(frozen=True)
class NoiseConfig:
    """Stochastic noise levels applied each timestep."""

    phi: float = 0.0
    g: float = 0.0
    s: float = 0.0
    bias_current: float = 0.0
    j: float = 0.0
    relative: bool = False
    extras: dict[str, float] = field(default_factory=dict)

    def to_settings(self, perturb: PerturbationConfig | None = None) -> NoiseSettings:
        from .noise import build_noise_strategies

        return build_noise_strategies(self, perturb)


@dataclass(frozen=True)
class PerturbationConfig:
    """Deterministic offsets with optional Gaussian spread."""

    phi_mean: float = 0.0
    phi_std: float = 0.0
    g_mean: float = 0.0
    g_std: float = 0.0
    s_mean: float = 0.0
    s_std: float = 0.0
    bias_current_mean: float = 0.0
    bias_current_std: float = 0.0
    j_mean: float = 0.0
    j_std: float = 0.0
    extras_mean: dict[str, float] = field(default_factory=dict)
    extras_std: dict[str, float] = field(default_factory=dict)

    def to_settings(self, noise: NoiseConfig | None = None) -> NoiseSettings:
        from .noise import build_noise_strategies

        return build_noise_strategies(noise, self)


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters.

    Attributes:
        dt: Time step for integration
        input_type: Type of input ("flux" or "state")
        track_power: Whether to track power consumption
        track_phi: Whether to track phi input to each layer

    """

    dt: float = 37
    dt_learnable: bool = False

    input_type: str = "state"
    track_power: bool = False
    track_phi: bool = False
    track_g: bool = False
    track_s: bool = False

    # Global network update strategy:
    # - "layerwise": run each layer over the full sequence in ascending layer_id order
    # - "stepwise_gauss_seidel": advance one time step using freshest values per layer (enables feedback, 1×Δt on backward edges)
    # - "stepwise_jacobi": advance one time step using previous states snapshot (parallelizable across layers)
    network_evaluation_method: str = "layerwise"

    # Early stopping for stepwise forward pass
    # If True, stepwise solvers may terminate the forward pass early once
    # the maximum absolute state change across all layers drops below
    # `early_stopping_tolerance` for `early_stopping_patience` consecutive steps.
    # Not supported for the "layerwise" global solver.
    early_stopping_forward_pass: bool = False
    early_stopping_tolerance: float = 1e-6
    early_stopping_patience: int = 1
    early_stopping_min_steps: int = 1
    # Windowed steady-state detection (preferred when >0)
    # If steady_window_min > 0, stepwise solvers consider the max |Δs|
    # over the trailing window and compare against (steady_tol_abs + steady_tol_rel*|s|_window_max).
    steady_window_min: int = 50
    steady_tol_abs: float = 1e-5
    steady_tol_rel: float = 1e-3

    def __post_init__(self):
        # Normalize network_evaluation_method strictly to {'layerwise','stepwise_gauss_seidel','stepwise_jacobi'}
        try:
            if isinstance(self.network_evaluation_method, str):
                gs = self.network_evaluation_method.strip().lower()
                if gs in {"layerwise", "layer", "layer_sequence", "layer-sequence"}:
                    self.network_evaluation_method = "layerwise"
                elif gs in {"stepwise_gauss_seidel", "stepwise (gauss–seidel)", "stepwise (gauss-seidel)", "stepwise_gs"}:
                    self.network_evaluation_method = "stepwise_gauss_seidel"
                elif gs in {"stepwise_jacobi", "stepwise (jacobi)"}:
                    self.network_evaluation_method = "stepwise_jacobi"
                else:
                    self.network_evaluation_method = "layerwise"
        except Exception:
            self.network_evaluation_method = "layerwise"

        # Clamp/validate early stopping parameters
        try:
            tol = float(self.early_stopping_tolerance)
            self.early_stopping_tolerance = max(tol, 0.0)
        except Exception:
            self.early_stopping_tolerance = 1e-6
        try:
            pat = int(self.early_stopping_patience)
            self.early_stopping_patience = max(pat, 1)
        except Exception:
            self.early_stopping_patience = 1
        try:
            ms = int(self.early_stopping_min_steps)
            self.early_stopping_min_steps = max(ms, 0)
        except Exception:
            self.early_stopping_min_steps = 1

        # Clamp/validate steady-state window params
        try:
            w = int(self.steady_window_min)
            self.steady_window_min = max(w, 0)
        except Exception:
            self.steady_window_min = 50
        try:
            a = float(self.steady_tol_abs)
            self.steady_tol_abs = max(a, 0.0)
        except Exception:
            self.steady_tol_abs = 1e-5
        try:
            r = float(self.steady_tol_rel)
            self.steady_tol_rel = max(r, 0.0)
        except Exception:
            self.steady_tol_rel = 1e-3


@dataclass
class LayerConfig:
    """Configuration for a layer in a SOEN model.

    Attributes:
        layer_id: Unique identifier for the layer
        model_id: Identifier for the sub‑module/group this layer belongs to (for visualisation/provenance)
        layer_type: Type of layer (e.g., "SingleDendrite", "MultiplierWICC", "MultiplierNOCC", "MinGRU")
        params: Dictionary of parameters for the layer
        description: Optional description of the layer
        noise: Optional noise configuration for this specific layer

    """

    layer_id: int
    layer_type: str
    params: dict
    model_id: int = 0
    description: str = ""
    noise: NoiseConfig | None = None
    perturb: PerturbationConfig | None = None

    def __post_init__(self):
        # Initialize noise config if not provided
        if self.noise is None:
            self.noise = NoiseConfig()
        if self.perturb is None:
            self.perturb = PerturbationConfig()


@dataclass
class ConnectionConfig:
    """Configuration for a connection between layers in a SOEN model.

    Attributes:
        from_layer: Source layer ID
        to_layer: Target layer ID
        connection_type: Type of connectivity pattern
        params: Optional dictionary of parameters for the connection
        learnable: Whether the connection weights can be updated during training
        visualization_metadata: Optional metadata for visualization purposes (e.g., hierarchical structure info)

    """

    from_layer: int
    to_layer: int
    connection_type: str
    params: dict | None = None
    learnable: bool = True
    noise: NoiseConfig | None = None
    perturb: PerturbationConfig | None = None
    visualization_metadata: dict | None = None

    def __post_init__(self):
        if self.noise is None:
            self.noise = NoiseConfig()
        if self.perturb is None:
            self.perturb = PerturbationConfig()
