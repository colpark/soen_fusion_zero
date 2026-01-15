# Criticality Metrics in SOEN Toolkit

This guide explains the metrics used to quantify criticality in Superconducting Optoelectronic Networks (SOENs). Operating at the "Edge of Chaos" (criticality) is theorized to maximize information processing capability, sensitivity, and dynamic range.

## 1. Branching Ratio ($\sigma$)

The branching ratio measures the propagation of activity through the network.

### A. Flux-Based Branching Ratio (`branching_ratio`)
**Type:** Information-Theoretic / Structural
**Differentiable:** Yes

Calculated as the ratio of the total output flux to the total input flux for the recurrent core of the network.

$$ \sigma_{\text{flux}} = \frac{\Phi_{\text{out}}}{\Phi_{\text{in}}} \approx 1.0 $$

- **$\sigma < 1$ (Subcritical):** Activity dies out. The network is stable but may have limited memory.
- **$\sigma \approx 1$ (Critical):** Activity is sustained without exploding. Optimal for computation.
- **$\sigma > 1$ (Supercritical):** Activity grows uncontrollably (runaway excitation).

### B. Local Expansion Rate (`local_expansion_rate`)
**Type:** Dynamical / Lyapunov
**Differentiable:** No (typically used for diagnostics)

Also known as the **dynamical branching ratio**, this metric measures how a small perturbation spreads through the network over time. It serves as a local, discrete-time approximation of the largest Lyapunov exponent.

$$ \sigma_{\text{dyn}} = \left\langle \frac{\lVert h'_t - h_t \rVert_2}{\lVert h'_{t-1} - h_{t-1} \rVert_2} \right\rangle_{t, \text{trials}} $$

- **$h_t$**: State vector of the original trajectory.
- **$h'_t$**: State vector of a perturbed trajectory (started with a tiny difference at $t=0$).
- **$\lVert \dots \rVert_2$**: Euclidean distance.

**Interpretation:**
- **$\sigma < 1$:** Perturbations shrink; the system is insensitive to small changes (stable).
- **$\sigma > 1$:** Perturbations grow exponentially; chaos (unstable/sensitive).
- **$\sigma \approx 1$:** "Edge of Chaos" – sensitivity is maximized but controlled.

---

## 2. Susceptibility ($\chi$)

**Type:** Information-Theoretic
**Differentiable:** Yes

Susceptibility measures the variance of the network's mean activity. In thermodynamic systems, susceptibility diverges (becomes very large) at a critical phase transition.

$$ \chi = \text{Var}(\langle s(t) \rangle_{\text{neurons}}) $$

- **High $\chi$:** Indicates the network is capable of visiting a wide range of states and is highly responsive to inputs (characteristic of criticality).
- **Low $\chi$:** Indicates the network is stuck in a fixed state or a simple limit cycle.

---

## 3. Avalanche Analysis

**Type:** Dynamical (The "Gold Standard")
**Differentiable:** No

Based on **Self-Organized Criticality (SOC)** theory, this analysis looks for scale-free behavior in bursts of network activity ("avalanches").

1.  **Avalanche:** A contiguous period where network activity (number of active nodes) exceeds a threshold.
2.  **Size ($S$):** Total integrated activity during the avalanche.
3.  **Duration ($T$):** Time steps the avalanche lasts.

If critical, the distributions of $S$ and $T$ follow power laws:

$$ P(S) \propto S^{-\alpha}, \quad P(T) \propto T^{-\beta} $$

**Typical Critical Exponents for Neural Systems:**
- **$\alpha \approx 1.5$**
- **$\beta \approx 2.0$**

The toolkit fits these distributions and returns the exponents ($\alpha, \beta$) and $R^2$ values to quantify how well the network exhibits scale-free dynamics.

---

## Usage in Code

```python
from soen_toolkit.utils.metrics import quantify_criticality

# Compute metrics (requires inputs for dynamical measures)
metrics = quantify_criticality(model, inputs=data)

print(f"Flux Branching Ratio: {metrics.branching_ratio}")
print(f"Local Expansion Rate: {metrics.local_expansion_rate}")  # New name for Lyapunov ratio
print(f"Susceptibility: {metrics.susceptibility}")
print(f"Avalanche Exponents: alpha={metrics.avalanche_exponent_size}, beta={metrics.avalanche_exponent_duration}")
```

### Initialization knobs (criticality_init)

- `br_weight` / `enable_br`: optimize branching ratio toward `br_target` (default on).
- `lyap_weight` / `enable_lyap`: optimize Lyapunov-like objective toward `lyap_target` (default off).
- `lyap_eps`: perturbation scale for the twin-trajectory Lyapunov estimate.
- `range_penalty_weight` + `range_clip`: optional guard against saturated activity.
- Any objective weight can be set to 0 to disable that term (e.g., Lyap-only or BR-only).

