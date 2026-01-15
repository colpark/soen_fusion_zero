"""Local learning module for SOEN models.

This module provides local learning rules (2-factor and 3-factor) that update
weights based on local activity rather than global error backpropagation.

Architecture Overview:
---------------------
The module is organized into clean, testable components:

1. **Learning Rules** (rules/):
   - AbstractLocalRule: Base class for all rules
   - HebbianRule, OjaRule, BCMRule: 2-factor unsupervised rules
   - RewardModulatedHebbianRule, RewardModulatedOjaRule: 3-factor supervised rules
   - AdaptiveLRWrapper, LRScheduleWrapper: Adaptive learning rate mechanisms

2. **Modulators** (modulators/):
   - AbstractModulator: Base class for modulators
   - MSEErrorModulator, CrossEntropyErrorModulator: Error-based signals
   - AccuracyBasedModulator: Binary reward signal

3. **Core Components** (new in v2):
   - ConnectionResolver: Determines which connections to train
   - StateCollector: Extracts layer states from model
   - LossStrategy: Pluggable loss computation
   - WeightUpdater: Applies weight updates with validation

4. **Trainers**:
   - LocalTrainer: Main training orchestrator (now uses v2 components)
   - LocalTrainerLegacy: Original implementation (for backward compatibility)

Example Usage:
-------------
```python
from soen_toolkit.training.local_learning import (
    LocalTrainer,
    HebbianRule,
    AdaptiveLRWrapper,
)

# Simple Hebbian learning
trainer = LocalTrainer(
    model=my_model,
    rule=HebbianRule(lr=0.01),
    layers=[1, 2],
    loss="mse"
)

# With adaptive learning rate
base_rule = HebbianRule(lr=0.1)
adaptive_rule = AdaptiveLRWrapper(base_rule, method='weight_norm', target_weight_norm=2.0)
trainer = LocalTrainer(model, rule=adaptive_rule, layers=[1, 2])

# Training loop
for inputs, targets in dataloader:
    metrics = trainer.step(inputs, targets)
    print(f"Loss: {metrics['loss']:.4f}")
```
"""

# Learning rules
# Core components (v2)
from soen_toolkit.training.local_learning.connection_resolver import (
    ConnectionInfo,
    ConnectionKey,
    ConnectionNotFoundError,
    ConnectionResolver,
    InvalidLayerError,
)
from soen_toolkit.training.local_learning.loss_strategy import (
    CrossEntropyLoss,
    HuberLoss,
    LossFactory,
    LossStrategy,
    MAELoss,
    MSELoss,
    NoLoss,
)

# Modulators
from soen_toolkit.training.local_learning.modulators.base import AbstractModulator
from soen_toolkit.training.local_learning.modulators.error_based import (
    AccuracyBasedModulator,
    CrossEntropyErrorModulator,
    MSEErrorModulator,
)
from soen_toolkit.training.local_learning.rules.adaptive import (
    AdaptiveLRWrapper,
    LRScheduleWrapper,
)
from soen_toolkit.training.local_learning.rules.base import AbstractLocalRule
from soen_toolkit.training.local_learning.rules.three_factor import (
    RewardModulatedHebbianRule,
    RewardModulatedOjaRule,
)
from soen_toolkit.training.local_learning.rules.two_factor import (
    BCMRule,
    HebbianRule,
    OjaRule,
)
from soen_toolkit.training.local_learning.state_collector import (
    ForwardResult,
    StateCollectionError,
    StateCollector,
)
from soen_toolkit.training.local_learning.trainer import LocalTrainer as LocalTrainerLegacy

# Trainers
from soen_toolkit.training.local_learning.trainer_v2 import LocalTrainer

# Utilities
from soen_toolkit.training.local_learning.utils import get_dt_for_time_constant
from soen_toolkit.training.local_learning.weight_updater import (
    DimensionMismatchError,
    UpdateMetrics,
    WeightUpdater,
)

__all__ = [
    # Learning rules
    "AbstractLocalRule",
    "HebbianRule",
    "OjaRule",
    "BCMRule",
    "RewardModulatedHebbianRule",
    "RewardModulatedOjaRule",
    "AdaptiveLRWrapper",
    "LRScheduleWrapper",
    # Modulators
    "AbstractModulator",
    "MSEErrorModulator",
    "CrossEntropyErrorModulator",
    "AccuracyBasedModulator",
    # Core components
    "ConnectionResolver",
    "ConnectionKey",
    "ConnectionInfo",
    "ConnectionNotFoundError",
    "InvalidLayerError",
    "StateCollector",
    "ForwardResult",
    "StateCollectionError",
    "LossStrategy",
    "MSELoss",
    "CrossEntropyLoss",
    "MAELoss",
    "HuberLoss",
    "NoLoss",
    "LossFactory",
    "WeightUpdater",
    "UpdateMetrics",
    "DimensionMismatchError",
    # Trainers
    "LocalTrainer",
    "LocalTrainerLegacy",
    # Utilities
    "get_dt_for_time_constant",
]
