# Local Learning Algorithms for SOEN: Summary Report

## Executive Summary

We explored three local learning algorithms for SOEN hardware:
1. **Forward-Forward (FF)** - Works well for flat input
2. **Equilibrium Propagation (EP)** - Works well for flat input
3. **Temporal Processing** - Struggles regardless of algorithm

**Key Finding**: The temporal credit assignment problem is fundamental and algorithm-independent. Local learning rules (FF, EP) solve *spatial* credit assignment but not *temporal* credit assignment.

---

## 1. What We Explored

### 1.1 Forward-Forward Algorithm

**Notebook**: `classification_forward_forward_mnist_recurrent.ipynb`

| Configuration | Accuracy | Notes |
|--------------|----------|-------|
| Flat (784 pixels) | ~80% | Works with 24 hidden neurons |
| Temporal (28 rows) | ~17% | Barely above random (10%) |

FF trains by comparing "goodness" (sum of squared activations) between positive and negative samples. Each layer learns independently - no backpropagation needed.

### 1.2 Equilibrium Propagation

**Notebook**: `classification_equilibrium_propagation_mnist.ipynb`

| Configuration | Accuracy | Notes |
|--------------|----------|-------|
| Standard EP (flat) | ~70-80% | Energy-based settling |
| SOEN-EP (flat) | ~70-80% | Uses SOEN dynamics |
| Temporal EP (broken) | ~10-15% | Original implementation had bug |
| Temporal EP (fixed) | ~15-20% | Still struggles even with fix |

EP trains by comparing free-phase (no target) vs clamped-phase (nudged toward target) correlations. Mathematically equivalent to backprop as β→0.

### 1.3 BPTT Baseline (RNN)

| Configuration | Accuracy | Notes |
|--------------|----------|-------|
| Leaky RNN with BPTT | ~60-80% | Same architecture as temporal SOEN |

This proves the **architecture can learn** - the issue is the training algorithm.

---

## 2. Why Temporal Processing Fails

### 2.1 The Core Problem

Both FF and EP solve the **spatial credit assignment problem** (how to assign credit across layers) but NOT the **temporal credit assignment problem** (how to assign credit across timesteps).

```
SPATIAL (Works):                    TEMPORAL (Fails):

Input → Layer1 → Layer2 → Output    t=0 → t=1 → ... → t=27 → Output
  ↑        ↑        ↑       ↑         ↑                        ↑
  └────────┴────────┴───────┘         └── How does credit ─────┘
        Credit assignment                 flow back in time?
        via local correlations            Local rules can't do this!
```

### 2.2 The Math

With SOEN dynamics: `s[t+1] = α × s[t] + (1-α) × g(W × x[t])`

Where `α = 1 - dt × γ⁻` is the retention factor.

For α = 0.97 (γ⁻ = 0.03), after 28 timesteps:
- Row 0 contribution: 0.97²⁷ ≈ **44%**
- Row 14 contribution: 0.97¹³ ≈ **67%**
- Row 27 contribution: **100%**

The learning signal for early rows is inherently weaker because:
1. Their information has decayed in the hidden state
2. The target/clamping signal can't propagate backward in time

### 2.3 This Is the Same Problem RNNs Have

SOEN temporal dynamics ARE a leaky RNN:

```
SOEN:     s[t+1] = (1 - dt×γ⁻) × s[t] + dt×γ⁺ × g(W×x[t])
Leaky RNN: h[t+1] =      α      × h[t] +  (1-α) × g(W×x[t])
```

Standard RNN training uses BPTT (backprop through time), which explicitly computes gradients backward through time. Local learning rules (FF, EP) don't have this mechanism.

---

## 3. What We Tried and Learned

### 3.1 Bug Fix: Accumulated Correlations

**Problem**: Original temporal EP only computed correlation with the last row.

**Fix**: Accumulate correlations at each timestep.

**Result**: Small improvement, but still doesn't solve the fundamental issue.

### 3.2 Why Even "Fixed" Temporal EP Struggles

Even with accumulated correlations:
- The **difference** between clamped and free correlations is what drives learning
- For early rows, `corr_clamped(t) ≈ corr_free(t)` because the clamping signal hasn't propagated
- This gives weak learning signal for early timesteps

### 3.3 dt (Time Delta) Is Essentially Unitless

In simulation, only the **product** `dt × γ⁻` matters:
- `dt=1, γ⁻=0.03` is equivalent to `dt=0.1, γ⁻=0.3`
- What matters is the retention factor α, not individual values

---

## 4. Algorithm Comparison

| Algorithm | Spatial Credit | Temporal Credit | Hardware Compatible |
|-----------|---------------|-----------------|---------------------|
| Backprop | ✓ Exact | ✗ N/A (flat only) | ✗ No |
| Backprop + BPTT | ✓ Exact | ✓ Exact | ✗ No |
| Forward-Forward | ✓ Approximate | ✗ Fails | ✓ Yes |
| Equilibrium Propagation | ✓ Exact (β→0) | ✗ Fails | ✓ Yes |
| RTRL | ✓ Exact | ✓ Exact | ✗ O(n⁴) expensive |
| Eligibility Traces | ~ Approximate | ~ Approximate | ✓ Yes |

**No free lunch**: Every hardware-compatible local learning algorithm struggles with temporal credit assignment.

---

## 5. Practical Recommendations

### 5.1 For Accuracy-Critical Tasks

Use **flat input** where FF and EP work well:
- Present all 784 pixels simultaneously
- 70-80% accuracy achievable with 24 hidden neurons
- Hardware-compatible local learning works

### 5.2 For Temporal/Streaming Applications

**Option A: Train Offline, Deploy Online**
- Train with BPTT in simulation
- Deploy learned weights to SOEN hardware
- Hardware does inference only (no on-chip learning)
- This is the standard approach for neural accelerators

**Option B: Accept Reduced Accuracy**
- Use temporal processing with FF/EP
- Accept ~15-20% accuracy on MNIST (still above random)
- May be sufficient for some streaming applications

**Option C: Explore Eligibility Traces (Future Work)**
- Maintains local "memory" of what caused what
- Could bridge the gap between local learning and temporal credit
- More complex to implement

### 5.3 Hardware/Software Split

The key insight from this exploration:

| Component | Classification | Rationale |
|-----------|---------------|-----------|
| SOEN dynamics (ds/dt = γ⁺g(φ) - γ⁻s) | **Hardware-Fixed** | Physics of the device |
| FF/EP goodness functions | **Software-Flexible** | Training objective |
| BPTT for temporal training | **Software-Flexible** | Simulation only |
| Deployed weights | Transfer from software to hardware | Result of training |

---

## 6. Files and Notebooks

### Reports
- `reports/forward_forward_soen_analysis.md` - Original FF analysis
- `reports/local_learning_algorithms_summary.md` - This summary

### Analysis Documents
- `claudedocs/temporal_ff_analysis.md` - Why temporal FF fails
- `claudedocs/temporal_ff_principled_fix.md` - Gradient compensation approach

### Notebooks
- `classification_forward_forward_mnist_recurrent.ipynb` - FF experiments
- `classification_equilibrium_propagation_mnist.ipynb` - EP experiments (includes temporal EP and RNN baseline)

---

## 7. Conclusions

1. **FF and EP work well for flat input** - both achieve ~70-80% on MNIST with 24 neurons

2. **Temporal processing struggles regardless of algorithm** - this is a fundamental limitation of local learning rules, not a bug

3. **SOEN temporal dynamics = Leaky RNN** - the architecture can learn (BPTT proves it), but local learning rules can't train it

4. **Practical path forward**: Train with BPTT in simulation, deploy to hardware for inference

5. **Future research**: Eligibility traces may offer a hardware-compatible solution to temporal credit assignment, but require additional investigation

---

## References

- Hinton, G. (2022). The Forward-Forward Algorithm. arXiv:2212.13345
- Scellier & Bengio (2017). Equilibrium Propagation. Frontiers in Computational Neuroscience
- Williams & Zipser (1989). RTRL for Recurrent Networks
- Bellec et al. (2020). e-prop: Eligibility Traces for RNNs
