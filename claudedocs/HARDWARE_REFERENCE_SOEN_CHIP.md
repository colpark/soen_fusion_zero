# SOEN Hardware Reference: ASCR Year-1 Chip

**Source**: ASCR_NC_Concept.pdf (Year-1 Program Deliverable)
**Last Updated**: 2026-01-26

---

## Chip Specifications Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Neurons** | 28 | Non-spiking, continuous-valued |
| **Recurrent synapses** | 784 (28×28) | Fully connected crossbar |
| **External inputs** | 8 | Up to 12 GHz each |
| **Input synapses** | 224 (8×28) | Each input connects to all neurons |
| **Total synapses** | 1,008 | 784 + 224 |
| **Weight precision** | ≥4-bit | Dynamically reconfigurable |
| **State range** | 0 ≤ s ≤ 1 | Normalized inductor current |
| **Die size** | 1.2 cm × 1.2 cm | |
| **Process** | 365 nm i-line | 1 µm minimum feature |
| **Metal layers** | 5 | |
| **Josephson junctions** | 4,144 | |

---

## Neuron Model

### State Representation
- Neuron state `s` is an **inductor current** (normalized, dimensionless)
- Range: `0 ≤ s ≤ 1`
- Continuous-valued (NOT spiking)

### Dynamics
```
phi_i = sum_j(J_ij * s_j)      # Flux from weighted inputs
s_i[t+1] = f(s_i[t], phi_i)    # State update via source function
```

Where:
- `J_ij` = weight matrix (reconfigurable, ≥4-bit precision)
- `phi_i` = total flux into neuron i
- `f()` = SOEN source function (state-dependent, flux-quantized)

---

## Connectivity Architecture

### Crossbar Structure
```
        Neuron 1  Neuron 2  ...  Neuron 28
           ↓         ↓              ↓
Input 1 → [W11]    [W12]    ...  [W1,28]
Input 2 → [W21]    [W22]    ...  [W2,28]
  ...
Input 8 → [W81]    [W82]    ...  [W8,28]
           ↓         ↓              ↓
Neuron 1→ [J11]    [J12]    ...  [J1,28]
Neuron 2→ [J21]    [J22]    ...  [J2,28]
  ...
Neuron28→ [J28,1]  [J28,2]  ...  [J28,28]
```

### Connection Types
1. **W_ih (Input-to-Hidden)**: 8 inputs × 28 neurons = 224 weights
2. **W_hh (Hidden-to-Hidden)**: 28 × 28 = 784 weights (includes self-connections)
3. **Total**: 1,008 programmable weights

---

## Implications for MNIST Classification

### Constraints

| MNIST Requirement | Chip Capability | Gap |
|-------------------|-----------------|-----|
| 784 input pixels | 8 external inputs | **98× mismatch** |
| ~50-100 neurons (typical) | 28 neurons | May be sufficient |
| Real-valued weights | 4-bit weights | Quantization needed |

### Strategies for MNIST on This Chip

#### Option 1: Temporal Multiplexing (Current Approach)
- Feed image row-by-row (28 rows × 28 pixels)
- **Problem**: 28 pixels/row but only 8 inputs
- **Solution**: Further subdivide rows OR downsample

```
Original: 28×28 image → 28 timesteps × 28 pixels
Hardware: 8 inputs → need 28/8 ≈ 4 sub-timesteps per row
Total: 28 rows × 4 sub-timesteps = 112 timesteps
```

#### Option 2: Downsampled Input
- Downsample 28×28 → 8×8 = 64 pixels
- Feed as 8 rows × 8 pixels = 8 timesteps
- **Tradeoff**: Loss of spatial resolution

#### Option 3: Feature Extraction Preprocessing
- Use analog front-end to extract 8 features per row
- E.g., edge detectors, Gabor filters
- **Tradeoff**: Requires additional hardware

#### Option 4: Multi-Chip Cascade
- Chip 1: Process first 8 input channels
- Chip 2: Process next 8 input channels
- Aggregate results
- **Tradeoff**: Complexity, synchronization

---

## Alignment with Current Notebook Work

### Our Temporal FF Implementation

| Parameter | Notebook | Hardware | Match? |
|-----------|----------|----------|--------|
| Neurons | 24 | 28 | ✓ Close |
| W_hh | 24×24=576 | 28×28=784 | ✓ Yes |
| W_ih | 38×24=912 | 8×28=224 | ✗ Need adaptation |
| Input/timestep | 38 (28px + 10 label) | 8 | ✗ Need adaptation |

### Recommended Modifications

1. **Reduce input dimension**:
   - Current: 28 pixels + 10 label = 38 per timestep
   - Hardware: 8 inputs max
   - Options: Downsample, multiplex, or preprocess

2. **Use exactly 28 neurons**:
   - Matches hardware exactly
   - W_hh = 28×28 = 784 weights

3. **Weight quantization**:
   - Train with full precision
   - Quantize to 4-bit for deployment
   - May need quantization-aware training

---

## Performance Characteristics

### Speed (from prototype measurements)
- Prototype achieved: **10 million frames/second** for small image classification
- Year-1 chip expected: Similar or better throughput

### Energy (from SOEN physics)
- Superconducting: Near-zero static power
- Switching energy: ~10^-19 J per operation (estimated)
- **Orders of magnitude below CMOS**

### Latency
- Signal propagation: Speed of light in superconductor
- Settling time: ~100 ps typical (from source function dynamics)
- **Sub-nanosecond inference possible**

---

## Key References from Document

### Figures
- **Fig. 8**: Existing prototype chip (throughput demo)
- **Fig. 9**: Architectural layout (N, W, M cells)
- **Fig. 10**: Draft layout of Year-1 chip

### Tables
- **Table 2**: Chip specifications summary

### Not Specified in Document
- Per-neuron cell dimensions/pitch
- Detailed neuron block geometry
- Prototype chip neuron/synapse counts

---

## Action Items for Hardware Alignment

- [ ] Create 8-input variant of temporal MNIST notebook
- [ ] Implement 4-bit weight quantization
- [ ] Test with exactly 28 neurons
- [ ] Design input preprocessing for 28→8 pixel mapping
- [ ] Validate SoftSOEN → real SOEN transfer

---

## Version History

| Date | Change |
|------|--------|
| 2026-01-26 | Initial creation from ASCR_NC_Concept.pdf |
