# Training Module 🔧 SOFTWARE-FLEXIBLE

## Classification: CAN MODIFY FREELY

This module contains all training-related code. Everything here represents
algorithmic choices for finding good weights, not physical constraints.
You can freely modify these to explore different training strategies.

---

## Contents

| Subfolder | Purpose | Modification Freedom |
|-----------|---------|---------------------|
| `losses/` | Loss function definitions | ✅ FULL |
| `local_learning/` | STDP, Hebbian rules | ✅ FULL |
| `callbacks/` | Training callbacks | ✅ FULL |
| `trainers/` | Training orchestration | ✅ FULL |
| `configs/` | Training configuration | ✅ FULL |
| `models/` | Lightning wrapper | ✅ FULL |
| `data/` | Data loading | ✅ FULL |
| `distillation/` | Knowledge distillation | ✅ FULL |

---

## Why Training Code Is Flexible

### The Hardware Doesn't Train

SOEN hardware only performs **forward inference**:
- Input photons arrive
- Dendrites integrate current
- Neurons spike when threshold is exceeded
- Output photons are produced

There is no "loss function" or "gradient" in the hardware. These are
**simulation constructs** for finding good weights.

### Trained Weights Transfer

What matters for hardware is the **final weight values**, not how you found them:

```
Simulation Training → Learned Weights → Hardware Deployment
                      ▲
                      │
         This is what transfers to hardware
         (not the training algorithm)
```

---

## Safe Modifications

### Loss Functions (`losses/`)

```python
# Use any differentiable loss
def custom_loss(output, target):
    # Your custom loss computation
    return loss_value

# Combine multiple losses
total_loss = cross_entropy_loss + 0.1 * regularization_loss
```

### Learning Rules (`local_learning/`)

```python
# Experiment with different local rules
from soen_toolkit.training.local_learning.rules import (
    HebbianRule,
    OjaRule,
    BCMRule,
    RewardModulatedHebbianRule,
)

# Try different modulation strategies
from soen_toolkit.training.local_learning.modulators import (
    MSEErrorModulator,
    CrossEntropyErrorModulator,
)
```

### Optimizers and Schedulers (`callbacks/`, `configs/`)

```yaml
# training_config.yaml
training:
  optimizer:
    name: "adam"  # or "sgd", "adamw", "lion", etc.
    lr: 0.001

  callbacks:
    lr_scheduler:
      type: "cosine"
      max_lr: 0.01
      min_lr: 1e-6
```

---

## Hardware-Compatible Training Options

While all training code is flexible, some approaches are more
"hardware-compatible" in that they could potentially run on-chip:

| Approach | Hardware Compatible? | Notes |
|----------|---------------------|-------|
| Backpropagation + SGD | ❌ No | Requires global gradients |
| STDP | ✅ Yes | Local, spike-timing based |
| Three-factor rules | ✅ Yes | Local with reward modulation |
| Reward-modulated Hebbian | ✅ Yes | RL-style with local updates |

---

## Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Data ──► Model Forward ──► Loss ──► Backward ──► Optimizer ──► Weights    │
│           (physics-based)    (flexible)  (surrogate)   (flexible)           │
│                                                                             │
│  Hardware-fixed ─┘           └─────── Software-flexible ──────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Examples

See `examples/training_configs/` for configuration templates:
- `minimal_training_config.yaml` - Basic setup
- `comprehensive_training_config.yaml` - All options documented

---

## Imported By

- `soen_toolkit.software_flexible` (re-exports)
- Tutorial notebooks
- Training scripts

---

## See Also

- `reports/hardware_software_split_architecture.md` - Classification rationale
- `tutorial_notebooks/02_train_a_model.ipynb` - Training tutorial
- `ops/surrogates.py` - Surrogate gradient definitions
