# Systematic Analysis: Why Temporal Forward-Forward Fails

## Executive Summary

The flat Forward-Forward model achieves ~80% accuracy on 10-class MNIST with 24 neurons.
The temporal (row-by-row) Forward-Forward model achieves only ~17% accuracy (barely above random 10%).

This document provides a systematic investigation of the root causes.

---

## 1. Side-by-Side Comparison

| Aspect | Flat (Works) | Temporal (Fails) |
|--------|--------------|------------------|
| Input dimensions | 794 (784 pixels + 10 label) | 38 (28 pixels + 10 label) |
| Timesteps | 1 | 28 |
| Total input features | 794 × 1 = 794 | 38 × 28 = 1064 |
| gamma_minus | **1e-6** (no decay) | **0.05** (significant decay) |
| Parameters | 794 × 24 = 19,056 | 38 × 24 = 912 |
| Goodness source | Only timestep (t=0) | Final timestep (t=27) |
| Training | Single optimizer, accumulated loss | Layer-wise optimizers, retain_graph |

---

## 2. Root Cause Analysis

### 2.1 Information Decay Through Temporal Dynamics

The SOEN ODE is:
```
ds/dt = γ⁺ × g(φ) - γ⁻ × s
```

With Forward Euler discretization:
```
s[t+1] = (1 - dt × γ⁻) × s[t] + dt × γ⁺ × g(φ)
       = α × s[t] + β × g(φ)
```

Where `α = 1 - dt × γ⁻` is the **retention factor**.

**Flat model**: γ⁻ = 1e-6 → α = 0.999999 → **~100% retention**
**Temporal model**: γ⁻ = 0.05 → α = 0.95 → **95% retention per step**

After 28 timesteps in temporal model:
```
Retention = 0.95^28 ≈ 0.23 = 23%
```

**77% of information from the first row is LOST before goodness is computed!**

### 2.2 Vanishing Gradients

The gradient flow through time follows the same dynamics.

For goodness computed at t=27, the gradient w.r.t. state at t=0:
```
∂L/∂s[0] = ∂L/∂s[27] × ∏(t=0 to 26) ∂s[t+1]/∂s[t]
         = ∂L/∂s[27] × α^27
         = ∂L/∂s[27] × 0.95^27
         ≈ ∂L/∂s[27] × 0.25
```

**Gradients for early timesteps are only 25% of final timestep gradients!**

This is the classic **vanishing gradient problem** in RNNs.

### 2.3 Gradient Flow Diagram

```
FLAT MODEL (Works):
═══════════════════
Input ──► Layer ──► Goodness ──► Loss
  │          │          │          │
  └──────────┴──────────┴──────────┘
        Direct gradient flow (1 timestep)


TEMPORAL MODEL (Fails):
═══════════════════════
t=0   t=1   t=2   ...  t=26  t=27
 │     │     │           │     │
 ▼     ▼     ▼           ▼     ▼
[s]──►[s]──►[s]──► ... ──►[s]──►[s]──► Goodness ──► Loss
 │                                         │
 └────────────── 0.95^27 ≈ 25% ───────────┘
        Vanishing gradient over 28 steps
```

### 2.4 Why This Matters for Forward-Forward

Forward-Forward learns by:
1. Positive sample → increase goodness
2. Negative sample → decrease goodness

With vanishing gradients:
- Weights connected to early rows receive weak learning signals
- The network cannot learn what makes early rows "good" or "bad"
- Only the last few rows contribute meaningful gradient
- **Top rows of MNIST digits are largely ignored during training!**

This is catastrophic for digit recognition since:
- Digit '1' vs '7': Distinguished by top horizontal stroke
- Digit '3' vs '8': Distinguished by gaps in upper half
- Digit '4' vs '9': Distinguished by upper loop structure

---

## 3. Evidence Gathering

### 3.1 Mathematical Proof of Information Loss

Let's trace a signal from row 0 through to final state:

```python
# Initial state from row 0 input
s[0] = γ⁺ × g(φ[0])

# After row 1
s[1] = 0.95 × s[0] + γ⁺ × g(φ[1])
     = 0.95 × γ⁺ × g(φ[0]) + γ⁺ × g(φ[1])

# After row 27 (final)
s[27] = γ⁺ × (0.95^27 × g(φ[0]) + 0.95^26 × g(φ[1]) + ... + g(φ[27]))
```

Contribution weights by row:
- Row 0 (top): 0.95^27 ≈ 0.25 (25%)
- Row 7 (quarter): 0.95^20 ≈ 0.36 (36%)
- Row 14 (half): 0.95^13 ≈ 0.51 (51%)
- Row 21 (three-quarter): 0.95^6 ≈ 0.74 (74%)
- Row 27 (bottom): 0.95^0 = 1.00 (100%)

**The bottom of the image dominates the final state by 4:1 ratio!**

### 3.2 Comparison with Working Flat Model

In the flat model with γ⁻ = 1e-6:
- α = 0.999999
- After 1 timestep: retention = 99.9999%
- All 784 pixels contribute equally to state
- Gradient flows directly without attenuation

---

## 4. Why Previous Fixes Were "Trivial"

### 4.1 Fix: Change gamma_minus to 1e-6

**Why it's trivial**: This just makes temporal = flat. The state becomes a pure accumulator with no temporal dynamics. We lose the "scanning" behavior entirely.

**Why it doesn't address the real problem**: The network isn't learning temporal patterns; it's just accumulating inputs like the flat version.

### 4.2 Fix: Compute goodness at every timestep

**Why it's trivial**: This provides gradient signal at each step but:
1. It's NOT how Forward-Forward should work (goodness should reflect full representation)
2. It doesn't leverage temporal dependencies
3. Each timestep sees only partial image information

### 4.3 Fix: Sum goodness over all timesteps

**Why it's trivial**: This still requires BPTT through all timesteps, so vanishing gradients remain a problem for early timesteps.

---

## 5. Fundamental Issue: Forward-Forward + Temporal Dynamics

### 5.1 The Core Conflict

Forward-Forward algorithm:
- Requires comparing "complete" representations
- Goodness measures "how good is this representation"
- Positive/negative distinction must be clear in goodness

Temporal SOEN dynamics:
- Information decays over time
- Early inputs have diminished representation
- State at final timestep is weighted toward recent inputs

**These two concepts conflict!**

For Forward-Forward to work temporally, the final state must faithfully represent the ENTIRE input sequence, not just recent timesteps.

### 5.2 What Would Work?

Option A: **Pure accumulator (γ⁻ ≈ 0)**
- Defeats the purpose of temporal processing
- State just sums all inputs
- No temporal selectivity

Option B: **Reset mechanism**
- Reset state at sequence start
- Accumulate during sequence
- But this requires external control

Option C: **Different goodness computation**
- Don't use final timestep only
- Use temporal integration of goodness
- But this changes the algorithm fundamentally

Option D: **Hardware-aware local learning**
- Each timestep computes its own local loss
- Weights update based on immediate feedback
- No BPTT needed
- **This is biologically plausible and hardware-compatible**

---

## 6. Conclusion: Root Cause

The temporal Forward-Forward model fails because:

1. **Structural**: γ⁻ = 0.05 causes 77% information loss over 28 timesteps
2. **Gradient**: Vanishing gradients prevent learning for early timesteps
3. **Algorithmic**: Forward-Forward requires complete representation in goodness, but temporal dynamics favor recent inputs
4. **Capacity**: 912 parameters (vs 19,056) may be insufficient

**The fundamental issue is that standard Forward-Forward is incompatible with leaky integrator temporal dynamics when goodness is computed only at the final timestep.**

---

## 7. Recommended Approach

Rather than trivial parameter tweaks, a principled fix requires rethinking the algorithm:

### Approach: Temporal Forward-Forward with Local Learning

Key insight: Hardware-compatible learning should be **local in both space AND time**.

Each timestep should provide learning signal:
- At timestep t, compute local goodness from current state
- Compare positive vs negative for that timestep
- Update weights based on local error signal
- No need for gradients to flow through time

This is:
- ✓ Hardware compatible (no BPTT)
- ✓ Biologically plausible (local learning)
- ✓ Solves vanishing gradient (gradients are immediate)
- ✓ Makes temporal dynamics meaningful (each row contributes)

The key question is: **What should the "target" be at each timestep?**

Options:
1. Same label signal at each timestep (simplest)
2. Progressive label revelation
3. No label until final timestep, then backpropagate credit

This requires more careful design, not just parameter adjustment.
