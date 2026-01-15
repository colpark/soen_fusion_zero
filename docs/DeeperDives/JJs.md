---
layout: default
title: Josephson Junctions and SOEN Physics
---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="../Introduction.md" style="margin-right: 2em;">&#8592; Previous: Introduction</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="../Building_Models.md" style="margin-left: 2em;">Next: Building Models &#8594;</a>
</div>

# Josephson Junctions and SOEN Physics

This document provides technical background on the physics underlying SOEN circuits. Understanding this material is not required to use SOEN-Toolkit, but it provides valuable context for why the models behave the way they do.

---

## What are Josephson Junctions?

Josephson Junctions (JJs) are electronic devices made from two superconducting materials separated by a non-superconducting barrier. They are the fundamental building block of SOEN circuits and provide several characteristics that make them suited for neuromorphic systems:

### 1. High-Speed Operation

JJs can be used to construct rapid single flux quantum (RSFQ) devices, capable of operating at frequencies in the GHz range. Modern computer transistors typically operate between 1-5 GHz, whereas RSFQ devices can exceed 100 GHz.

Currently, the synapse recovery rate from a spike input is approximately 30 MHz. However, there is not a one-to-one mapping between these rates because much of a neural network's computation is performed in-between spike events. On the order of 1000 computations within the dendritic tree are performed for each spike produced. SOENs leverage JJs for their high speed, low energy operation for these internal dendritic tree computations. For detailed analysis of fluxonic processing in photonic synapse events, see [Shainline (2019)](https://arxiv.org/abs/1904.02807).

### 2. Energy Efficiency

JJs operate at very low voltages, typically in the range of tens of millivolts, contributing to the overall power efficiency of the system. The most energy intensive part of the process is the spike itself, which occurs less frequently to conserve energy.

### 3. Threshold and Saturating Non-linearities

JJs introduce two important non-linearities into SOENs:

- **Thresholding behavior**: When the critical current ($I_c$) in the receiving loop is reached, the JJs exhibit a threshold response analogous to the firing threshold in biological neurons.
- **Saturating nonlinearity**: The JJ circuits used in SOENs have a saturating nonlinearity that prevents runaway activity.

These non-linearities are important for running many machine-learning tasks.

---

## Circuit Dynamics

The operation of a superconducting neuron as envisioned by GreatSky can be described through the interaction of several key components:

![SOEN neuron components](../Figures/DeeperDives/soen_components.png)
*Simplified illustration of components involved in SOENs. The receiving loop (black) contains Josephson junctions and is inductively coupled to the synapse. The integration loop (red) stores current with a characteristic decay time.*

### 1. Synaptic Input Flux

The computational process begins with flux produced by a synapse after a photon detection event by a single photon detector (SPD). The synapse circuit is inductively coupled to a superconducting loop known as the receiving loop. In networks of dendrites without optical components, there are no SPDs or synapses in this sense. However, input flux still arrives—it is directly coupled from upstream dendrites via a collection coil rather than through photon detection events.

### 2. Receiving Loop (SQUID)

This loop contains JJs and remains superconducting until the critical current ($I_c$) is reached. The loop is referred to as a SQUID (Superconducting Quantum Interference Device) and functions as a flux-to-voltage converter.

When a photon is detected at the SPD and flux is induced in the receiving loop, the current increases in an adjacent loop called the integration loop. Once the total current—which is the current induced by the flux added to the existing bias current—reaches the critical current $I_c$, the JJs in the loop begin emitting pulses of voltage, rapidly switching between resistive and superconducting states.

### 3. Integration Loop

The pulses from the JJs add current to the adjacent integration loop. This loop, being inductor-resistor based, slowly releases its stored current with a characteristic time constant, $\tau$. This creates the temporal integration and memory behavior of the dendrite.

### 4. Transmitter

Both dendrites and somas share the same common base circuit. Somas are realized by building a thresholding unit on top of the dendrite. Once the current in the integration loop reaches the threshold, the transmitter emits a burst of photons. Simultaneously, an inhibitory dendrite feeds back into the soma temporarily reducing activity. These photons fan out via a network of optical waveguides to the synapses (SPDs) of downstream neurons.

---

## Lagrangian Formulation

The dynamics of SOENs can be derived from fundamental physical principles using a Lagrangian formulation. This method applies the principle of least action, leading to the Euler-Lagrange equation that governs the system's equations of motion.

In describing the temporal evolution of a SOEN, appropriate state variables must be selected. While SOENs are complex circuits with many possible variables to consider, two were chosen:

- The **current stored in the integration loop**
- The **flux into the receiving loop**

To simplify calculations, dimensionless versions of these variables are used, denoted as $s$ and $\phi$ respectively.

The Lagrangian formulation establishes SOENs as energy-based models, evolving along paths of least action, and provides potential routes for deriving energy functions and more rigorous mathematical analysis of their dynamics.

---

## Phenomenological Model

To abstract away the complex physics of the system, a phenomenological model is employed. A phenomenological model captures the high-level dynamics without trying to model all the fundamental physics.

The equation of motion in vector notation:

$$\frac{ds}{dt} = \gamma^{+} \odot g(\phi, s) - \gamma^{-} s$$

where:
- $s$ is the state (current in integration loop)
- $\gamma^{+}$ is an integration loop rate filling factor (inverse dimensionless inductance)
- $g(\phi, s;i_b)$ is the source function (see below)
- $\gamma^{-}$ is the decay rate constant of the integration loop (inverse of the time constant of the integration loop)

The input flux $\phi$ is given by:

$$\phi(t) = Js(t) + \phi_{ext}(t)$$

where $J$ is the static coupling matrix and $\phi_{ext}$ is the external flux.

### Discrete to Continuous Transition

When the phenomenological model was first proposed in 2023, it was only a partial step towards a fully continuous model. The discrete spikes were still dealt with manually using active checks for threshold crossings in the state variable s.

To maximize model efficiency whilst maintaining high agreement with the partial phenomenological model, this was changed to implicitly deal with spikes within the source functions themselves. This change required the use of two source functions:
- $g_d$: describes the response from flux incident from a dendrite
- $g_n$: describes the response from flux incident from a neuron

The reason this transition from discrete to continuous modeling is justified is that downstream neurons perform low-pass filtering. The high-frequency, picosecond-scale changes in voltage caused by individual fluxons are largely inconsequential. Focusing instead on the longer, nanosecond-scale dynamics, the full phenomenological model describes the dynamics of the time-averaged voltage. Note that our current hardware is dendrite-only in that we have not yet built optical chips and therefore have no need of $g_n$.

---

## Source Functions

A key component of this model is the **source function**. The source function represents a dimensionless rate of fluxon production and encapsulates the unique physics of the flux-to-current conversion in any given dendrite or soma circuit.

### Terminology Note

Within the broader context of machine learning literature, the SOEN-based 'source function' plays the role of an **activation function**. To clarify terminology:

- **Source function**: The SOEN-specific activation function that models the circuit behavior and encompasses the physics of the SOEN hardware
- **Activation function**: The general ML term for the nonlinear transformation applied to neuron outputs

The two terms are not precisely equivalent but are often used interchangeably in SOEN literature.

### Implementation

As of this writing, at GreatSky, the source function has been approximated using lookup tables created by solving JJ circuit equations in simulation. However, various methods of modeling source functions can be employed for increased speed such as the `HeavisideFit` (see the source functions catalog in the main documentation).

---

## Periodicity in Source Functions

A distinctive feature of SOEN source functions that sets them apart from typical neural network activation functions is their **inherent periodicity**. This periodicity stems from the quantum mechanical nature of Josephson junctions.

### Why Are Source Functions Periodic?

1. **Flux Quantization**: In SOENs, the total flux through the receiving loop must be quantized due to the quantum-scale energy levels involved. Specifically, it must be an integer multiple of the flux quantum:

   $$\Phi_0 = \frac{h}{2e}$$

   where $h$ is Planck's constant and $e$ is the elementary charge. This quantization underlies the fluxon production by JJs and gives rise to the periodic behavior of the source function.

2. **Quantum Interference**: The periodicity is rooted in the quantum interference of Cooper pair wavefunctions across the JJ. This interference is governed by the phase difference of these wavefunctions, which is directly related to the magnetic flux threading the superconducting loop containing the JJ.

3. **Phase-Current Relationship**: The relationship between the supercurrent through the JJ and the phase difference (and thus the flux) is intrinsically periodic, with a period of one flux quantum.

4. **Peak Location**: The peak of the source function occurs at half a flux quantum (where $\phi$ is normalized to $\Phi_0$). This is because, at this point, the flux through the loop is maximally distant from being a whole number of flux quanta, resulting in the strongest response from the circuit.

![Periodic behavior of the dendrite source function](../Figures/DeeperDives/periodic_source_function.png)
*Example showing the periodic nature of a typical dendrite source function, demonstrating the quantum mechanical periodicity inherent in Josephson junction circuits.*


### Implications for SOEN-Toolkit

This periodicity is why you see source functions in the toolkit with periodic behavior. Some source function implementations explicitly enforce this periodicity through transformations. Understanding this helps explain why SOEN networks behave differently from traditional ReLU or sigmoid-based networks.

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="../Introduction.md" style="margin-right: 2em;">&#8592; Previous: Introduction</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="../Building_Models.md" style="margin-left: 2em;">Next: Building Models &#8594;</a>
</div>
