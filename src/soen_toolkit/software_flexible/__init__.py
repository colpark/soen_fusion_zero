"""Software-Flexible SOEN Components - CAN MODIFY FREELY.

This package re-exports all algorithmic components that represent
design choices rather than physical constraints. You can freely modify
these to explore different training strategies, architectures, and
optimization approaches.

Components included:
    - Surrogate gradients (for BPTT through spikes)
    - Loss functions
    - Learning rules (STDP, three-factor, etc.)
    - ODE solvers (numerical methods)
    - Network architecture utilities
    - Initialization strategies
    - Virtual (non-physical) layers

These choices affect HOW you train, not WHAT the trained network does.
Trained weights will still transfer correctly to physical hardware.

See: reports/hardware_software_split_architecture.md for full documentation.
"""

from __future__ import annotations

# =============================================================================
# SURROGATE GRADIENTS
# These only affect the backward pass during training
# Hardware doesn't have gradients - these are purely for optimization
# =============================================================================
from soen_toolkit.ops.surrogates import (
    SurrogateSpec,       # Configuration for surrogate shape
    SurrogateRegistry,   # Registry of available surrogates
    SURROGATES,          # Global registry instance
)

# =============================================================================
# LOSS FUNCTIONS
# Any differentiable loss can be used - hardware doesn't compute loss
# =============================================================================
from soen_toolkit.training.losses.loss_functions import (
    cross_entropy,
    mse,
    gap_loss,
    top_gap_loss,
    reg_J_loss,
    autoregressive_loss,
    local_expansion_loss,
)

# =============================================================================
# LEARNING RULES
# Different approaches to finding good weights
# =============================================================================
from soen_toolkit.training.local_learning.rules import (
    AbstractLocalRule,
    HebbianRule,
    OjaRule,
    BCMRule,
    RewardModulatedHebbianRule,
    RewardModulatedOjaRule,
)

from soen_toolkit.training.local_learning.modulators import (
    AbstractModulator,
    MSEErrorModulator,
    CrossEntropyErrorModulator,
)

# =============================================================================
# ODE SOLVERS
# Different numerical methods for solving the same underlying equations
# =============================================================================
from soen_toolkit.core.layers.common.solvers import (
    ForwardEulerSolver,  # Simple and fast
    ParaRNNSolver,       # GPU-optimized parallel scan
)

# =============================================================================
# VIRTUAL LAYERS
# Non-physical computational layers
# =============================================================================
from soen_toolkit.core.layers.virtual import (
    InputLayer,
    LeakyGRU,
    MinGRU,
)

# =============================================================================
# NETWORK BUILDING UTILITIES
# Architecture design is a free choice
# =============================================================================
from soen_toolkit.nn import (
    SOENGraph,
    SOENSequential,
)

# =============================================================================
# INITIALIZATION STRATEGIES
# How you initialize weights is an algorithmic choice
# =============================================================================
from soen_toolkit.core.layers.common.parameters.config import (
    ParameterSpec,
    INITIALISERS,
)

# =============================================================================
# TRAINING INFRASTRUCTURE
# Complete freedom in training setup
# =============================================================================
from soen_toolkit.training.trainers.experiment import run_from_config
from soen_toolkit.training.models.lightning_wrapper import SOENLightningModule

# =============================================================================
# CONFIGURATION
# Network configuration is flexible (within physical layer constraints)
# =============================================================================
from soen_toolkit.core.configs import (
    SimulationConfig,
    LayerConfig,
    ConnectionConfig,
    NoiseConfig,
)

# =============================================================================
# MODULE METADATA
# =============================================================================

__all__ = [
    # Surrogate gradients
    "SurrogateSpec",
    "SurrogateRegistry",
    "SURROGATES",
    # Loss functions
    "cross_entropy",
    "mse",
    "gap_loss",
    "top_gap_loss",
    "reg_J_loss",
    "autoregressive_loss",
    "local_expansion_loss",
    # Learning rules
    "AbstractLocalRule",
    "HebbianRule",
    "OjaRule",
    "BCMRule",
    "RewardModulatedHebbianRule",
    "RewardModulatedOjaRule",
    # Modulators
    "AbstractModulator",
    "MSEErrorModulator",
    "CrossEntropyErrorModulator",
    # Solvers
    "ForwardEulerSolver",
    "ParaRNNSolver",
    # Virtual layers
    "InputLayer",
    "LeakyGRU",
    "MinGRU",
    # Network building
    "SOENGraph",
    "SOENSequential",
    # Initialization
    "ParameterSpec",
    "INITIALISERS",
    # Training
    "run_from_config",
    "SOENLightningModule",
    # Configuration
    "SimulationConfig",
    "LayerConfig",
    "ConnectionConfig",
    "NoiseConfig",
]

# Classification metadata
CLASSIFICATION = "SOFTWARE_FLEXIBLE"
MODIFICATION_RISK = "LOW"
RATIONALE = """
These components represent algorithmic choices that don't affect physical accuracy:

1. SURROGATE GRADIENTS: Only used during backpropagation (training).
   Hardware doesn't have gradients - spikes are discrete events.

2. LOSS FUNCTIONS: Define what "good" means during training.
   Hardware just runs inference - it doesn't compute loss.

3. LEARNING RULES: Alternative ways to find weights (STDP, Hebbian).
   The final weights are what matter, not how you found them.

4. ODE SOLVERS: Numerical methods for the same underlying equations.
   All converge to the same physics with sufficient precision.

5. ARCHITECTURE: Network topology is a design choice.
   Hardware can implement any topology you design.

6. INITIALIZATION: Starting point for optimization.
   Doesn't affect final trained behavior.

You can freely experiment with these to improve training performance.
"""
