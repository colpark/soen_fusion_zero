# Software-Flexible Components 🔧

## ✅ THESE MODULES CAN BE FREELY MODIFIED

This package contains components that represent **algorithmic choices**
rather than physical constraints. You can freely modify these to explore
different training strategies, architectures, and optimization approaches.

---

## What's in Here

| Module | Purpose | Why Flexible |
|--------|---------|--------------|
| **Surrogate Gradients** | BPTT through spikes | Training-only, hardware has no gradients |
| **Loss Functions** | Define "good" performance | Training-only, hardware doesn't compute loss |
| **Learning Rules** | STDP, Hebbian, etc. | Different paths to same weights |
| **ODE Solvers** | Forward Euler, ParaRNN | Numerical methods for same equations |
| **Virtual Layers** | GRU, MinGRU, etc. | Non-physical computational units |
| **Architecture Utils** | Network building | Design choice, not physics |
| **Initialization** | Weight starting points | Optimization strategy |

---

## Safe Modifications

### Surrogate Gradients
```python
# Try different surrogate shapes
from soen_toolkit.software_flexible import SurrogateSpec

# Default triangle
surrogate = SurrogateSpec(kind="triangle", width=1.0)

# Experiment with different widths
surrogate = SurrogateSpec(kind="triangle", width=0.5, scale=2.0)
```

### Loss Functions
```python
# Use any differentiable loss
from soen_toolkit.software_flexible import cross_entropy, mse

# Or define your own
def custom_loss(output, target):
    return your_loss_computation(output, target)
```

### Network Architecture
```python
# Change layer counts and sizes freely
config = {
    "layers": [
        {"layer_id": 0, "layer_type": "Input", "params": {"dim": 10}},
        {"layer_id": 1, "layer_type": "SingleDendrite", "params": {"dim": 100}},
        {"layer_id": 2, "layer_type": "SingleDendrite", "params": {"dim": 50}},
        {"layer_id": 3, "layer_type": "Input", "params": {"dim": 5}},  # Output
    ]
}
```

### Initialization Strategies
```python
# Experiment with different initialization
from soen_toolkit.software_flexible import ParameterSpec

# Normal distribution
init = ParameterSpec(distribution="normal", params={"mean": 0.0, "std": 0.1})

# Uniform distribution
init = ParameterSpec(distribution="uniform", params={"min": -0.1, "max": 0.1})
```

---

## Why These Are Safe to Modify

| Component | Reason |
|-----------|--------|
| Surrogate gradients | Only used in backward pass; forward pass unchanged |
| Loss functions | Hardware doesn't compute loss |
| Learning rules | Final weights matter, not how you found them |
| ODE solvers | All approximate same physics |
| Architecture | Hardware can implement any topology |
| Initialization | Just starting point for optimization |

---

## Things to Keep in Mind

1. **Use physical layer types**: When building architectures, use `SingleDendrite`,
   `Multiplier`, etc. for layers that should map to hardware.

2. **Virtual layers**: `LeakyGRU`, `MinGRU` are software-only and won't
   map directly to SOEN hardware.

3. **Operating point**: Even with flexible initialization, aim for φ ≈ 0.5
   for optimal hardware sensitivity.

4. **Numerical stability**: When changing ODE solvers or dt, ensure
   dt × γ⁻ < 1 for stability.

---

## Usage

```python
# Import software-flexible components
from soen_toolkit.software_flexible import (
    SurrogateSpec,              # Surrogate gradient config
    cross_entropy,              # Loss function
    ForwardEulerSolver,         # ODE solver
    SimulationConfig,           # Configuration
    run_from_config,            # Training launcher
)

# Freely experiment with these
config = SimulationConfig(dt=100, dt_learnable=True)
surrogate = SurrogateSpec(kind="triangle", width=0.8)
```

---

## See Also

- `reports/hardware_software_split_architecture.md` - Full classification
- `training/examples/` - Training configuration examples
- `tutorial_notebooks/02_train_a_model.ipynb` - Training tutorial
