# ParaRNN Solver

The ParaRNN solver enables **O(log T) parallel training** of nonlinear RNNs, based on the algorithm from:

> Danieli et al. (2025) "ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models"
> https://arxiv.org/abs/2505.14825

## Overview

Traditional RNN training is inherently sequential: computing $h_t$ requires $h_{t-1}$, giving $O(T)$ depth. ParaRNN reformulates this as a system of nonlinear equations solved via Newton's method, where each iteration's linear system is solved in $O(\log T)$ using parallel prefix scan.

## Algorithm

1. **Initialize**: $h^0_t = f(0, x_t)$ for all timesteps
2. **Newton iteration** (typically 3-5 iterations):
   - Compute Jacobian $A_t = \frac{\partial f}{\partial h}$ at current guess
   - Compute residual $b_t = f(h^k_{t-1}, x_t) - A_t \cdot h^k_{t-1}$
   - Solve linear recurrence $h^{k+1}_t = A_t \cdot h^{k+1}_{t-1} + b_t$ via parallel scan
3. **Converge** when $\lVert h^{k+1} - h^k \rVert < \text{tolerance}$

## Jacobian Structure Requirement

The algorithm mathematically works for **any** differentiable recurrence. However, the computational cost depends on Jacobian structure:

| Jacobian Type | Cost per Multiply | Practical? |
|---------------|-------------------|------------|
| Dense ($d \times d$) | $O(d^3)$ | ❌ Too slow for large $d$ |
| Diagonal | $O(d)$ | ✔️ Very fast |
| Block-diagonal | $O(k^3 \cdot d/k)$ | ⚠️ Needs custom kernels |

For SOEN SingleDendrite layers, the Jacobian is:

$$J = \frac{\partial s_{\text{next}}}{\partial s_{\text{prev}}} = \alpha \cdot I + \beta \cdot \text{diag}(g') \odot J_{\text{recurrent}}^T$$

This is **diagonal** if and only if $J_{\text{recurrent}}$ (internal connectivity) is diagonal.

## Recurrent Weight Support

ParaRNN **does support recurrent weights**, but they must be structured appropriately:

| Recurrent Weight Type | Jacobian | Supported? |
|-----------------------|----------|------------|
| None | Diagonal | ✔️ |
| Diagonal (element-wise) | Diagonal | ✔️ |
| Dense matrix | Dense | ❌ |

**Key insight**: The diagonal constraint means each neuron's state update depends only on its own previous state during the Newton linearization step. Feature mixing across neurons should be done via **inter-layer connections**, similar to how Mamba and Transformer architectures operate.

## Usage

### PyTorch

```python
from soen_toolkit.core.layers.physical import SingleDendriteLayer
import torch

# Without recurrent weights - always works
layer = SingleDendriteLayer(dim=64, dt=37.0, solver="PARARNN")

# With DIAGONAL recurrent weights - works
diag_weights = torch.diag(torch.randn(64) * 0.1)
layer_diag = SingleDendriteLayer(
    dim=64, dt=37.0, solver="PARARNN",
    connectivity=diag_weights,
    connectivity_mode="fixed"
)

# With DENSE recurrent weights - raises RuntimeError
dense_weights = torch.randn(64, 64) * 0.1
layer_dense = SingleDendriteLayer(
    dim=64, dt=37.0, solver="PARARNN",
    connectivity=dense_weights,  # Will raise error
    connectivity_mode="fixed"
)
```

### JAX

```python
from soen_toolkit.utils.port_to_jax.layers_jax import SingleDendriteLayerJAX
import jax.numpy as jnp

layer = SingleDendriteLayerJAX(dim=64, dt=37.0, source=TanhJAX())

# Use ParaRNN solver
history = layer.forward(phi, params, solver="pararnn")
```

## Performance Characteristics

| Source Function | Typical Iterations | Notes |
|-----------------|-------------------|-------|
| Tanh | 3-5 | Smooth, fast convergence |
| SimpleGELU | 3-5 | Smooth, fast convergence |
| ReLU | 5-8 | Piecewise linear |
| Heaviside/RateArray | 10-15 | Discontinuous, slower convergence |

## Compatibility

### Layer-level Solvers

| Solver | Complexity | Recurrent Weights | Source Functions |
|--------|-----------|-------------------|------------------|
| FE (Forward Euler) | O(T) | Any | Any |
| PS (Parallel Scan) | O(log T) | None | Linear only |
| PARARNN | O(log T) | Diagonal only | Any |

### Network-level Solvers

ParaRNN (layer-level) is compatible with all network-level solvers:

- `layerwise`: Each layer processes full sequence → **best parallelization**
- `stepwise_jacobi`: Layers parallelized per timestep → works but no time parallelization
- `stepwise_gauss_seidel`: Sequential layers per timestep → works but no time parallelization

## References

- [ParaRNN Paper](https://arxiv.org/abs/2505.14825) - Full algorithm and analysis
- Equation 3.3: Diagonal Jacobian structure A_* = diag(a_*)
- Appendix D.4: Block-diagonal extensions with custom CUDA kernels

