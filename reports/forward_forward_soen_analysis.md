# Forward-Forward Algorithm for SOEN: Analysis and Implementation Strategy

## Executive Summary

The Forward-Forward (FF) algorithm (Hinton, 2022) is **remarkably well-suited** for SOEN hardware. Unlike backpropagation, FF:
- Requires only forward passes (no backward pass)
- Uses local objectives per layer (no global error propagation)
- Was explicitly designed for analog hardware with unknown non-linearities
- Can work with "black boxes" in the computational pipeline

This makes FF a strong candidate for **on-chip learning** in SOEN systems.

---

## 0. Hardware-Fixed vs Software-Flexible Classification

**CRITICAL**: The Forward-Forward algorithm is entirely a **SOFTWARE-FLEXIBLE** training method. It does NOT modify any hardware-fixed components.

### What Remains HARDWARE-FIXED (Do Not Modify)

| Component | File Location | Reason |
|-----------|---------------|--------|
| Dendritic ODE: `ds/dt = γ⁺g(φ) - γ⁻s` | `core/layers/physical/dynamics/` | Physics of superconducting loop |
| Source function `g(φ)` | `core/source_functions/` | Measured SQUID response curve |
| Physical constants (Φ₀, h, e) | `physics/constants.py` | Laws of nature |
| Spike mechanism | `ops/spike.py` | Threshold physics |
| Forward pass dynamics | `core/soen_model_core.py` | How hardware computes |

### What FF Adds/Modifies (SOFTWARE-FLEXIBLE)

| Component | Classification | Rationale |
|-----------|---------------|-----------|
| Goodness function `Σsⱼ²` | 🟢 SOFTWARE | Training objective, not physics |
| Layer normalization | 🟢 SOFTWARE | Training technique for FF |
| Positive/negative data creation | 🟢 SOFTWARE | Data preparation |
| Per-layer loss computation | 🟢 SOFTWARE | Replaces backprop loss |
| Weight update rule | 🟢 SOFTWARE | How to find good weights |

### The Key Insight

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FF RESPECTS THE HARDWARE/SOFTWARE SPLIT                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   HARDWARE-FIXED (unchanged)        SOFTWARE-FLEXIBLE (FF changes this)    │
│   ─────────────────────────         ───────────────────────────────────    │
│                                                                             │
│   ds/dt = γ⁺g(φ) - γ⁻s             Loss function: goodness-based           │
│   g(φ) lookup table                 Training: two forward passes            │
│   Spike threshold                   No backward pass needed                 │
│   Physical constants                No surrogate gradients needed           │
│                                                                             │
│   The PHYSICS stays the same.       Only HOW WE TRAIN changes.             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Where FF Code Should Live

```
soen_toolkit/
├── hardware_fixed/          # DO NOT MODIFY - physics
│   └── (unchanged)
│
└── software_flexible/       # FF implementation goes here
    └── training/
        └── ff/              # NEW: Forward-Forward module
            ├── __init__.py
            ├── goodness.py      # Goodness functions (Σsⱼ²)
            ├── layer_norm.py    # FF layer normalization
            ├── ff_layer.py      # FF-wrapped SOEN layers
            ├── trainer.py       # FF training loop
            └── README.md        # Documentation
```

### Important Implementation Note

When implementing FF for SOEN, the forward pass **MUST** use the existing hardware-fixed dynamics:

```python
# CORRECT: Use hardware-fixed dynamics, add software-flexible goodness
class FFSOENLayer:
    def forward(self, x):
        # Hardware-fixed: Use existing SOEN dynamics (DO NOT MODIFY)
        states = self.soen_dendrite.forward(x)  # Calls ds/dt = γ⁺g(φ) - γ⁻s

        # Software-flexible: Compute goodness for FF training
        goodness = self.compute_goodness(states)  # NEW: Σsⱼ²

        # Software-flexible: Normalize for next layer
        normalized = self.layer_norm(states[:, -1, :])  # NEW

        return normalized, goodness

# WRONG: Do not modify the dynamics equation!
# states = custom_dynamics(x)  # ❌ This would change hardware behavior
```

---

## 1. Forward-Forward Algorithm Overview

### 1.1 Core Concept

FF replaces backpropagation's forward+backward passes with **two forward passes**:

| Pass | Data | Objective |
|------|------|-----------|
| **Positive** | Real data | Maximize goodness |
| **Negative** | Corrupted/wrong data | Minimize goodness |

### 1.2 The Goodness Function

```
p(positive) = σ(Σⱼ yⱼ² - θ)
```

Where:
- `yⱼ` = activity of hidden unit j (before layer normalization)
- `θ` = threshold
- `σ` = logistic function

**Goodness** = `Σⱼ yⱼ²` (sum of squared activities)

### 1.3 Layer Normalization (Critical)

Between layers, FF normalizes the hidden vector. This:
- Removes length information (used for goodness)
- Passes only **orientation** to the next layer
- Prevents trivial solutions (can't just check vector length)

```
h_normalized = h / ||h||
```

### 1.4 Why FF Works Without Backprop

Each layer learns independently:
- Layer 1 learns to distinguish positive/negative based on input
- Layer 2 sees normalized output of Layer 1, learns its own distinction
- No gradient flow between layers needed

---

## 2. SOEN-FF Compatibility Analysis

### 2.1 Why FF is Perfect for SOEN

| FF Requirement | SOEN Capability | Match |
|----------------|-----------------|-------|
| Forward passes only | Hardware does forward inference | ✅ Perfect |
| Local per-layer objectives | Dendrites compute locally | ✅ Perfect |
| Works with unknown non-linearities | g(φ) is complex, measured curve | ✅ Perfect |
| Designed for analog hardware | SOEN is analog superconducting | ✅ Perfect |
| Can have "black boxes" | Dendritic dynamics are complex | ✅ Perfect |
| No stored activities needed | Enables real-time learning | ✅ Perfect |

### 2.2 Key Insight from Hinton

From Section 8 of the FF paper:

> "An energy efficient way to multiply an activity vector by a weight matrix is to implement activities as voltages and weights as conductances... The use of two forward passes instead of a forward and a backward pass should make these A-to-D converters unnecessary."

**This describes exactly the SOEN approach** - weights as synaptic strengths, activities as currents/photon rates.

### 2.3 Mapping FF Concepts to SOEN

| FF Concept | SOEN Implementation |
|------------|---------------------|
| Activity `yⱼ` | Dendritic current `sⱼ` (stored in superconducting loop) |
| Goodness `Σyⱼ²` | Sum of squared currents `Σsⱼ²` |
| Layer normalization | Normalize current vector before next layer |
| Positive data | Input with correct label |
| Negative data | Input with wrong label |
| Weight update | Adjust synaptic strengths (optical attenuations) |

---

## 3. SOEN-FF Architecture for Classification

### 3.1 Network Structure

For MNIST classification (following Hinton's approach):

```
Input Layer: 784 pixels + 10 label neurons = 794 neurons
    │
    ▼ (weights: optical synaptic connections)
Hidden Layer 1: 500 SOEN neurons
    │ (layer normalization)
    ▼ (weights)
Hidden Layer 2: 500 SOEN neurons
    │ (layer normalization)
    ▼
Output: Goodness computed from each hidden layer
```

### 3.2 Label Embedding (Supervised FF)

Following Hinton's supervised approach, **embed the label in the input**:

```python
# Replace first 10 pixels with one-hot label encoding
def create_ff_input(image, label, is_positive=True):
    """
    image: 28x28 = 784 pixels (flattened)
    label: integer 0-9
    is_positive: True for positive pass, False for negative
    """
    # Create one-hot label vector
    label_vec = torch.zeros(10)

    if is_positive:
        label_vec[label] = 1.0  # Correct label
    else:
        # Random wrong label
        wrong_labels = [i for i in range(10) if i != label]
        wrong_label = random.choice(wrong_labels)
        label_vec[wrong_label] = 1.0

    # Concatenate: [label_vec (10), image (784)] = 794 inputs
    return torch.cat([label_vec, image])
```

### 3.3 SOEN-Specific Goodness Function

```python
class SOENLayerGoodness(nn.Module):
    """Compute goodness from SOEN dendritic currents."""

    def __init__(self, threshold: float = 2.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, dendrite_currents: torch.Tensor) -> torch.Tensor:
        """
        dendrite_currents: [batch, time, neurons] - s values from SOEN dynamics

        Returns goodness per sample
        """
        # Use final time step or average over time window
        final_state = dendrite_currents[:, -1, :]  # [batch, neurons]

        # Goodness = sum of squared currents
        goodness = torch.sum(final_state ** 2, dim=-1)  # [batch]

        return goodness

    def loss(self, goodness: torch.Tensor, is_positive: torch.Tensor) -> torch.Tensor:
        """
        Binary cross-entropy loss for positive/negative classification.

        goodness: [batch] - computed goodness values
        is_positive: [batch] - 1.0 for positive samples, 0.0 for negative
        """
        # p(positive) = sigmoid(goodness - threshold)
        logits = goodness - self.threshold
        loss = F.binary_cross_entropy_with_logits(logits, is_positive)
        return loss
```

### 3.4 SOEN Layer Normalization

```python
class SOENLayerNorm(nn.Module):
    """
    Layer normalization for SOEN outputs.

    Normalizes the dendritic current vector to unit length,
    removing goodness information while preserving orientation.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, currents: torch.Tensor) -> torch.Tensor:
        """
        currents: [batch, neurons] - dendritic currents s

        Returns normalized currents (unit length vectors)
        """
        # Compute L2 norm
        norm = torch.sqrt(torch.sum(currents ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize to unit length
        normalized = currents / norm

        return normalized
```

---

## 4. Complete FF-SOEN Layer Implementation

### 4.1 Key Principle: Wrap, Don't Reimplement

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ❌ WRONG: Reimplement dynamics        ✅ CORRECT: Wrap existing dynamics   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  class FFLayer:                        class FFLayer:                       │
│      def dynamics(self, x):                def __init__(self):              │
│          # Reimplementing ODE ❌            # Use EXISTING hardware module  │
│          s = alpha*s + beta*g              from hardware_fixed import (    │
│          ...                                   SingleDendrite              │
│                                            )                                │
│                                            self.dendrite = SingleDendrite() │
│                                                                             │
│                                        def forward(self, x):                │
│                                            # Call hardware-fixed dynamics   │
│                                            states = self.dendrite(x) ✅     │
│                                            # Add software-flexible goodness │
│                                            goodness = sum(states**2)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Correct Implementation

```python
# FILE: soen_toolkit/training/ff/ff_layer.py
# LOCATION: software_flexible (training module)

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# IMPORT FROM HARDWARE-FIXED (do not modify these!)
from soen_toolkit.hardware_fixed import (
    SingleDendrite,           # Hardware-fixed dendritic dynamics
    SingleDendriteDynamics,   # The ODE kernel
)

# IMPORT FROM SOFTWARE-FLEXIBLE (can modify these)
from soen_toolkit.training.ff.goodness import SOENLayerGoodness
from soen_toolkit.training.ff.layer_norm import SOENLayerNorm


class FFSOENLayer(nn.Module):
    """
    Forward-Forward wrapper around HARDWARE-FIXED SOEN dendrite.

    This class:
    - USES existing SingleDendrite (hardware-fixed, unchanged)
    - ADDS goodness computation (software-flexible, new for FF)
    - ADDS layer normalization (software-flexible, new for FF)

    The dendritic dynamics (ds/dt = γ⁺g(φ) - γ⁻s) are NOT reimplemented here.
    They come from the hardware_fixed module.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dendrite_config: dict,      # Pass to hardware-fixed SingleDendrite
        goodness_threshold: float = 2.0,
    ):
        super().__init__()

        # ═══════════════════════════════════════════════════════════════════
        # HARDWARE-FIXED: Use existing dendrite implementation
        # DO NOT modify SingleDendrite - it implements the physics
        # ═══════════════════════════════════════════════════════════════════
        self.dendrite = SingleDendrite(
            input_size=input_size,
            output_size=output_size,
            **dendrite_config  # gamma_plus, gamma_minus, dt, etc.
        )

        # ═══════════════════════════════════════════════════════════════════
        # SOFTWARE-FLEXIBLE: FF-specific components (new, can modify)
        # ═══════════════════════════════════════════════════════════════════
        self.goodness = SOENLayerGoodness(threshold=goodness_threshold)
        self.layer_norm = SOENLayerNorm()

    def forward(
        self,
        x: torch.Tensor,
        return_goodness: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through FF-SOEN layer.

        x: [batch, input_size] - input (raw or normalized from previous layer)

        Returns:
            normalized output for next layer
            optionally: goodness value for FF loss
        """
        # ═══════════════════════════════════════════════════════════════════
        # HARDWARE-FIXED: Run existing SOEN dynamics (do not modify!)
        # This calls ds/dt = γ⁺g(φ) - γ⁻s internally
        # ═══════════════════════════════════════════════════════════════════
        states = self.dendrite(x)  # [batch, time, neurons]

        # ═══════════════════════════════════════════════════════════════════
        # SOFTWARE-FLEXIBLE: Compute goodness for FF training
        # ═══════════════════════════════════════════════════════════════════
        goodness = self.goodness(states)  # [batch] - sum of squared currents

        # ═══════════════════════════════════════════════════════════════════
        # SOFTWARE-FLEXIBLE: Normalize for next layer (FF requirement)
        # ═══════════════════════════════════════════════════════════════════
        final_state = states[:, -1, :]  # [batch, neurons]
        normalized_output = self.layer_norm(final_state)  # [batch, neurons]

        if return_goodness:
            return normalized_output, goodness
        return normalized_output
```

---

## 5. FF-SOEN Training Procedure

### 5.1 Training Loop

```python
class FFSOENTrainer:
    """Forward-Forward trainer for SOEN networks."""

    def __init__(
        self,
        layers: List[FFSOENLayer],
        learning_rate: float = 0.001,
    ):
        self.layers = layers
        self.optimizers = [
            torch.optim.Adam(layer.parameters(), lr=learning_rate)
            for layer in layers
        ]

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        One training step with positive and negative passes.

        images: [batch, 784] - MNIST images
        labels: [batch] - integer labels 0-9
        """
        batch_size = images.shape[0]
        device = images.device

        losses = {}

        # === POSITIVE PASS ===
        # Create positive data: image + correct label
        pos_labels_onehot = F.one_hot(labels, num_classes=10).float()
        pos_input = torch.cat([pos_labels_onehot, images], dim=1)  # [batch, 794]

        # === NEGATIVE PASS ===
        # Create negative data: image + wrong label
        wrong_labels = (labels + torch.randint(1, 10, (batch_size,), device=device)) % 10
        neg_labels_onehot = F.one_hot(wrong_labels, num_classes=10).float()
        neg_input = torch.cat([neg_labels_onehot, images], dim=1)  # [batch, 794]

        # Process each layer independently (greedy layer-wise training)
        pos_x = pos_input
        neg_x = neg_input

        for layer_idx, (layer, optimizer) in enumerate(zip(self.layers, self.optimizers)):
            optimizer.zero_grad()

            # Forward pass for positive and negative
            pos_out, pos_goodness = layer(pos_x, return_goodness=True)
            neg_out, neg_goodness = layer(neg_x, return_goodness=True)

            # Compute FF loss
            is_positive = torch.cat([
                torch.ones(batch_size, device=device),
                torch.zeros(batch_size, device=device)
            ])
            all_goodness = torch.cat([pos_goodness, neg_goodness])

            loss = layer.goodness.loss(all_goodness, is_positive)

            # Backward (only through this layer's goodness)
            loss.backward()
            optimizer.step()

            losses[f'layer_{layer_idx}_loss'] = loss.item()

            # Prepare input for next layer (detached, no gradient flow between layers!)
            pos_x = pos_out.detach()
            neg_x = neg_out.detach()

        return losses
```

### 5.2 Inference Procedure

```python
def ff_classify(
    layers: List[FFSOENLayer],
    image: torch.Tensor,
    num_classes: int = 10,
) -> int:
    """
    Classify an image using trained FF-SOEN network.

    Strategy: Run network with each possible label, pick label with highest goodness.
    """
    device = image.device
    image = image.unsqueeze(0) if image.dim() == 1 else image  # [1, 784]

    goodness_per_label = []

    for label in range(num_classes):
        # Create input with this label
        label_onehot = torch.zeros(1, num_classes, device=device)
        label_onehot[0, label] = 1.0
        x = torch.cat([label_onehot, image], dim=1)  # [1, 794]

        # Accumulate goodness from all layers (except first, following Hinton)
        total_goodness = 0.0

        for layer_idx, layer in enumerate(layers):
            x, goodness = layer(x, return_goodness=True)

            if layer_idx > 0:  # Skip first layer (following paper)
                total_goodness += goodness.item()

        goodness_per_label.append(total_goodness)

    # Return label with highest goodness
    return int(torch.tensor(goodness_per_label).argmax())
```

---

## 6. Configuration for SOEN Hardware

### 6.1 YAML Configuration

```yaml
# ff_soen_mnist.yaml
model:
  type: "forward_forward_soen"

  architecture:
    input_size: 794  # 784 pixels + 10 label neurons
    hidden_layers:
      - size: 500
        goodness_threshold: 2.0
      - size: 500
        goodness_threshold: 2.0

  # SOEN dynamics (hardware-fixed)
  soen_dynamics:
    num_timesteps: 50
    dt: 0.1
    gamma_plus: 1.0
    gamma_minus: 0.1

  # Layer normalization
  layer_norm:
    type: "l2"  # Normalize to unit length
    eps: 1e-8

training:
  algorithm: "forward_forward"

  # FF-specific settings
  ff_config:
    goodness_type: "sum_squared"  # Σsⱼ²
    negative_data: "wrong_label"  # Use wrong labels as negative
    threshold: 2.0

  # Optimizer (per-layer, independent)
  optimizer:
    type: "adam"
    lr: 0.001

  # Training schedule
  epochs: 60
  batch_size: 64

data:
  dataset: "mnist"
  label_embedding: "first_10_neurons"  # Embed label in first 10 inputs
```

### 6.2 Hardware Parameter Constraints

| Parameter | Value | Source |
|-----------|-------|--------|
| `gamma_plus` | 1.0 | Device-dependent, measured |
| `gamma_minus` | 0.1 | Device-dependent, measured |
| `num_timesteps` | 50 | Simulation resolution |
| `dt` | 0.1 | Discretization step |
| Source function | g(φ) lookup | Measured from device |

---

## 7. Advantages of FF for SOEN

### 7.1 Enables On-Chip Learning

With FF, SOEN hardware could potentially learn **on-chip** without external simulation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ON-CHIP FF LEARNING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Present positive data (image + correct label)               │
│     → Run forward pass → Compute goodness → Increase weights    │
│                                                                 │
│  2. Present negative data (image + wrong label)                 │
│     → Run forward pass → Compute goodness → Decrease weights    │
│                                                                 │
│  No backpropagation needed! No stored activities!               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Works with Unknown Non-linearities

FF doesn't need to know the exact form of g(φ). It treats the dendritic dynamics as a "black box" and learns to use whatever non-linearity is present.

### 7.3 Local Learning Rules

Each layer has its own objective. This maps naturally to SOEN's dendritic structure where local computations are preferred.

### 7.4 Real-time Learning

FF can learn while pipelining data through the network. No need to store activities for a backward pass.

---

## 8. Comparison: FF vs Current SOEN Training

| Aspect | Current (Backprop) | Forward-Forward |
|--------|-------------------|-----------------|
| Training location | Simulation only | Potentially on-chip |
| Backward pass | Required | Not needed |
| Gradient through spike | Requires surrogate | Not needed |
| Activity storage | Required for BPTT | Not needed |
| Per-layer objectives | No (global loss) | Yes (local goodness) |
| Hardware compatibility | Low | High |
| Training speed | Faster | Slightly slower |
| Accuracy (MNIST) | ~98.6% | ~98.5-99.4% |

---

## 9. Implementation Roadmap

### Critical Rule: Maintain Hardware/Software Split

Throughout all phases, **DO NOT** modify any hardware-fixed code:
- `core/layers/physical/dynamics/` - ODE kernels
- `core/source_functions/` - g(φ) lookup
- `physics/constants.py` - Physical constants
- `ops/spike.py` - Spike mechanism

All FF code goes in `training/ff/` (software-flexible).

### Phase 1: Simulation Validation
1. Create `training/ff/` module structure
2. Implement `FFSOENLayer` that **wraps** existing `SingleDendrite` (don't reimplement!)
3. Add `goodness.py` and `layer_norm.py` (software-flexible components)
4. Validate on MNIST, compare to backprop baseline

```python
# CORRECT: Wrap existing hardware-fixed dynamics
from soen_toolkit.hardware_fixed import SingleDendrite
self.dendrite = SingleDendrite(...)  # Use existing implementation

# WRONG: Reimplement dynamics
# s = alpha * s + beta * g  # ❌ Don't do this
```

### Phase 2: Full SOEN Dynamics Integration
1. Ensure FF wrapper correctly uses full g(φ) source function (via SingleDendrite)
2. Test with realistic dendritic dynamics (hardware-fixed, unchanged)
3. Tune FF hyperparameters (goodness threshold, layer sizes) - software-flexible

### Phase 3: Hardware Considerations
1. Design goodness computation circuit (new hardware, not modifying existing)
2. Design layer normalization circuit (new hardware)
3. Design weight update mechanism (new hardware)

### Phase 4: On-Chip Learning
1. Implement weight updates in hardware
2. Test positive/negative data switching
3. Validate learning performance

### File Organization

```
soen_toolkit/
├── hardware_fixed/          # ❌ DO NOT MODIFY
│   ├── __init__.py
│   └── (exports existing physics)
│
├── software_flexible/       # ✅ FF goes here
│   └── training/
│       └── ff/              # NEW MODULE
│           ├── __init__.py
│           ├── goodness.py      # SOENLayerGoodness
│           ├── layer_norm.py    # SOENLayerNorm
│           ├── ff_layer.py      # FFSOENLayer (wraps SingleDendrite)
│           ├── trainer.py       # FFSOENTrainer
│           └── README.md
│
└── core/                    # Existing - layers use hardware_fixed imports
```

---

## 10. Open Questions

1. **Goodness function choice**: Should we use `Σsⱼ²` or `-Σsⱼ²` (minimize for positive)?
   - Hinton notes minimizing works "slightly better"

2. **Temporal integration**: How to compute goodness over SOEN's temporal dynamics?
   - Final timestep? Average over window?

3. **Layer normalization in hardware**: Can this be implemented in superconducting circuits?

4. **Weight update mechanism**: How to adjust optical synaptic weights based on goodness?

5. **Negative data generation**: On-chip, how to generate wrong labels?

---

## 11. Conclusion

The Forward-Forward algorithm is an excellent match for SOEN hardware because:

1. **No backpropagation** - Only forward passes needed
2. **Local objectives** - Each layer learns independently
3. **Designed for analog hardware** - Explicitly mentioned by Hinton
4. **Works with black boxes** - Doesn't need to know exact g(φ)
5. **Potential for on-chip learning** - Could enable hardware-based training

FF represents a significant opportunity to enable **learning directly on SOEN hardware**, which would be a major advancement over the current approach of training in simulation and deploying weights.

---

## References

- Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations. arXiv:2212.13345
- SOEN papers in `papers/` folder
- Current SOEN training implementation in `soen_toolkit/training/`
