# Principled Fix: Temporal Forward-Forward with Gradient Compensation

## The Core Problem (Recap)

Standard Forward-Forward + Temporal dynamics fails because:
1. **Information decay**: γ⁻ = 0.05 → 0.95²⁸ ≈ 0.23 (77% loss over 28 steps)
2. **Vanishing gradients**: Gradients decay at same rate through BPTT
3. **Early rows ignored**: With final-timestep-only loss, top rows get ~25% of gradient signal

---

## The Implemented Solution: Gradient Weight Compensation

### Key Insight

Since we're training in simulation to find weights for hardware deployment, we can compensate
for vanishing gradients by weighting the loss at each timestep inversely proportional to the
gradient decay.

### The Math

With SOEN dynamics: `s[t] = α × s[t-1] + β × g(φ[t])` where `α = 0.95`

Gradient decay from timestep t to final timestep T-1:
```
∂s[T-1]/∂s[t] = α^(T-1-t)
```

For t=0 (first row), gradient is α²⁷ ≈ 0.25 of final timestep.

**Compensation**: Weight loss at timestep t by `(1/α)^(T-1-t)`

This makes the **effective gradient contribution** uniform:
```
effective[t] = weight[t] × decay[t]
             = (1/α)^(T-1-t) × α^(T-1-t)
             = 1.0  (constant for all t!)
```

### Numerical Verification

```
Gradient weight compensation for vanishing gradients:
============================================================
t= 0: loss_weight=  1.8369, grad_decay=0.2503, effective=0.4598
t= 7: loss_weight=  1.2827, grad_decay=0.3585, effective=0.4598
t=14: loss_weight=  0.8958, grad_decay=0.5133, effective=0.4598
t=21: loss_weight=  0.6256, grad_decay=0.7351, effective=0.4598
t=27: loss_weight=  0.4598, grad_decay=1.0000, effective=0.4598

Without compensation (uniform loss weights):
t= 0: effective_grad=0.2503
t= 7: effective_grad=0.3585
t=14: effective_grad=0.5133
t=21: effective_grad=0.7351
t=27: effective_grad=1.0000
```

---

## Training Modes Comparison

| Mode | Loss Computation | Gradient Flow | Expected Performance |
|------|------------------|---------------|---------------------|
| `goodness_mode='final'` | Final timestep only | Standard BPTT | Poor (vanishing) |
| `goodness_mode='all'` | All timesteps, equal weight | Standard BPTT | Better but still biased |
| `local_in_time=True` | All timesteps, compensated weight | Compensated BPTT | Best (uniform effective gradients) |

---

## Why This Is Principled (Not Trivial)

### Previous "Trivial" Fixes (Rejected)

1. **γ⁻ = 1e-6**: Eliminates temporal dynamics entirely; not temporal anymore
2. **Loss at every timestep (equal weight)**: Still has vanishing gradients
3. **Sum goodness over time**: Same problem

### Current Fix (Gradient Compensation)

1. **Preserves temporal dynamics**: γ⁻ = 0.05 remains, real decay behavior
2. **Addresses root cause**: Compensates for vanishing gradients mathematically
3. **Principled derivation**: Weight = 1/decay ensures uniform effective gradients
4. **Hardware-relevant**: Finds weights suitable for temporal hardware inference

---

## Implementation

```python
# In training loop:
for t in range(seq_len):
    g_pos = compute_goodness(layer_states_pos[layer_idx][:, t, :])
    g_neg = compute_goodness(layer_states_neg[layer_idx][:, t, :])

    timestep_loss = forward_forward_loss(g_pos, g_neg, margin)

    # Gradient compensation: early timesteps get higher weight
    weight = (1.0 / alpha) ** (seq_len - 1 - t)
    weight = weight / norm_factor * seq_len  # Normalize total weight

    total_loss = total_loss + weight * timestep_loss

total_loss.backward()
```

---

## Limitations and Notes

1. **Still uses BPTT**: This is simulation-based training, not true hardware-local learning
2. **Compensation is approximate**: Assumes linear gradient decay (ignores nonlinearity of g())
3. **Hardware deploys trained weights**: The learning algorithm isn't hardware-compatible,
   but the resulting weights are deployed for hardware inference
4. **True local-in-time would require**: Model modifications to support per-timestep
   forward passes with gradient detachment

---

## Expected Results

With gradient compensation enabled:
- Early rows (t=0-7) should contribute equally to learning
- Digit features in upper portion of images should be learned
- Overall accuracy should improve compared to final-only or uniform-weight modes

This approach maintains the temporal scanning paradigm while fixing the fundamental
gradient flow problem that prevented the original temporal model from learning.
