---
layout: default
title: Introduction to SOEN-Toolkit
---

# Introduction to SOEN-Toolkit

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="index.md" style="margin-right: 2em;">&#8592; Previous: Documentation Index</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Getting_Started.md" style="margin-left: 2em;">Next: Getting Started &#8594;</a>
</div>

---

## 1.1 What is SOEN-Toolkit?

SOEN-Toolkit is a Python package for simulating and training Superconducting Optoelectronic networks (SOENs). It provides custom PyTorch modules that implement phenomenological mathematical models of GreatSky's superconducting circuits, enabling researchers to:

- **Design** network architectures using biologically-inspired superconducting computational primitives (e.g dendrites, synapses)
- **Simulate** network dynamics with hardware-realistic parameters and constraints
- **Train** models using standard gradient-based optimization
- **Analyze** performance before committing to physical fabrication

Originally developed as a simple way to set up and simulate SOEN models, the toolkit has grown into a collection of internal research tools. This is in active development, so you will encounter bugs and inconsistencies as features are often added as-needed rather than following a grand design. However, it provides a solid foundation for exploring SOEN architectures, implementing learning algorithms, and bridging the gap between simulation and hardware.

---

## 1.2 The Context: From AI to SOEN

### Artificial Intelligence → Machine Learning → Deep Learning → RNNs

This section places SOEN technology in context. If you're already familiar with these concepts, feel free to skip to [1.3](#13-how-greatskys-hardware-fits-in).

#### Artificial Intelligence

The term "artificial intelligence" has become increasingly overloaded in recent years, often reduced to chatbots and virtual assistants—which are just narrow applications of a much broader field. Artificial intelligence is more of an idea than a single technology—the pursuit of understanding and reproducing intelligence itself.

Defining intelligence is hard. You will hear a different definition depending on who you ask. Psychologist David Wechsler defined it as "the ability to act purposefully, think rationally, and deal effectively with the environment." Computer scientist Marvin Minsky kept it simpler: intelligence is "the ability to solve hard problems." Regardless of definition, at GreatSky we believe the study and application of intelligence is worth pursuing.

#### Machine Learning

Machine learning sits at the intersection of mathematics and computer science. It is the study of how to build systems that can automatically improve their performance on tasks through experience, without being explicitly programmed for every possible scenario. Rather than hand-coding solutions, we provide algorithms with data and let them discover patterns and relationships. This approach has proven remarkably effective for problems where traditional programming falls short—like recognizing images, understanding language, or making predictions from complex datasets.

#### Deep Learning

Deep learning represents a particularly powerful subset of machine learning, inspired directly by the structure of biological neural networks. These systems use layers of interconnected nodes (artificial neurons) to process information hierarchically—much like how our brains build understanding from simple features to complex concepts. What makes deep learning "deep" is this layered architecture, where each layer learns increasingly sophisticated representations of the input data.

#### Recurrent Neural Networks

Most deep learning systems process information in a single forward pass, but biological brains are fundamentally dynamic—neurons continuously interact, creating loops of activity that persist over time. Recurrent Neural Networks (RNNs) attempt to capture this temporal aspect by allowing information to flow not just forward through layers, but also backward and sideways, creating memory and enabling the network to maintain state over time. This makes them particularly well-suited for processing sequences like speech, text, or time-series data.

![The AI/ML/DL hierarchy showing where SOEN networks sit](Figures/Introduction/ai_ml_hierarchy.png)
*Broadly speaking, the diagram above provides a sense of relationship between our hardware and existing fields.*

---

## 1.3 How GreatSky's Hardware Fits In

GreatSky looks to biology and asks: **What can evolution teach us about information processing?**

Rather than **simulating** neural dynamics in software running on conventional hardware, we're **building them directly** into hardware using superconducting circuits. The research team at GreatSky has already designed and built the core superconducting counterparts to biological neural networks: neurons, synapses, and dendrites. It is worth noting that although we greatly admire and take inspiration from biological brains, we are in no way constrained to imitate them perfectly. 

### The Key Difference

In SOEN networks, the graph structure is the physical circuit. Connections are actual wires you can point to. Memory and computation happen in the same place. This is fundamentally different from traditional neural networks, where the model is an abstraction that gets mapped onto GPU hardware.

**Traditional deep learning**:
> Software simulation of neurons → runs on silicon transistors → fundamentally digital

**SOEN approach**:
> Physical superconducting circuits → naturally exhibit neural dynamics → fundamentally analog


This difference matters. Superconducting circuits can:
- Operate at much higher speeds limited only by JJ physics (picosecond timescales in theory)
- Integrate and process signals with extremely low energy dissipation
- Naturally implement asynchronous temporal dynamics

### Current Research Directions

Moving forward, we must think about how to best utilize this hardware. Many questions naturally arise:

- How can we explore the network design space efficiently?
- What ML paradigms are best suited for SOEN architectures?
- How do we balance influence from neuroscience versus machine learning?
- How can we scale up in-simulation training?
- How would we implement on-chip, in-situ learning algorithms for ultra-fast and scalable training?
- Countless others...

SOEN-Toolkit was developed to help answer these research questions.

---

## 1.4 The SOEN Design Pipeline

![The SOEN design pipeline from circuits to trained models](Figures/Introduction/design_pipeline.png)
*The design process from low-level circuits to deployed models.*

The workflow proceeds in stages:

1. **Circuit Design**: Design superconducting circuits with specific computational properties
2. **Mathematical Modeling**: Derive phenomenological models (ODEs) that capture circuit behavior
3. **Software Implementation**: Models are implemented in SOEN-Toolkit as custom PyTorch modules
4. **Simulation & Training**: Researchers design network architectures and train them using standard gradient-based methods
5. **Hardware Instantiation**: Trained networks are mapped to physical chips or interfaced through tools like Imprint [whitepaper pending]

**This toolkit focuses on stages 3-4**, enabling rapid iteration on network designs before committing to fabrication.

---

## 1.5 What You Can Do With This Toolkit

SOEN-Toolkit enables:

**Network Design**
- Construct networks from software models of superconducting components. The toolkit provides building blocks that compute nonlinear functions (dendrites), scale signals (multipliers), and convert outputs to readable voltages (readouts). These models capture the essential dynamics without requiring you to understand the underlying circuit physics.
- Define custom connectivity patterns (dense, sparse, spatially-organized via power-law connectivity patterns)

**Simulation**
- The toolkit handles all the differential equation solving automatically—you define the network structure, and it computes how voltages and currents evolve over time. You interact with the model like any PyTorch module.
- Supports converting models to JAX for increased evaluation speed

**Training**
- Use standard PyTorch optimization (Adam, SGD, etc.)
- Implement custom loss functions and learning schedules
- Options to apply quantization-aware training for hardware limitations
- Leverage PyTorch Lightning for distributed training and logging
- MLFlow integration for experiment management and metric visualizations

**Analysis**
- Visualize network architectures and connectivity patterns
- Generate power/energy estimates
- Analyze robustness to noise and parameter variations with `soen_toolkit.robustness_tool`
- Export trained models to multiple formats (PyTorch, JSON, YAML)


## 1.6 What You Cannot Do With This Toolkit

You cannot currently interface directly with hardware. This is simulation-only software that models the underlying superconducting circuits. Models can be trained and evaluated here, then validated and deployed to hardware using separate tools.

---

## 1.7 What You Need to Know

You don't need to deeply understand superconducting circuit physics to use this toolkit effectively. The phenomenological models abstract away most fine details of the true circuit dynamics while remaining highly accurate models. You can approach this toolkit from either a hardware/physics background or a machine learning background.

---

## 1.8 Academic Papers


### Key References

To understand the physics and theory underlying SOEN-Toolkit, these papers are invaluable:

1. **[Superconducting Optoelectronic Circuits for Neuromorphic Computing](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.7.034013)** (PRApplied 2017)
   - The foundational paper introducing SOEN technology
   - Describes the core dendrite circuit and its dynamics

2. **[Phenomenological Model of Superconducting Optoelectronic Loop Neurons](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.013164)** (PRResearch 2023)
   - Derives the mathematical models implemented in this toolkit
   - Explains the source functions and their physical origins

3. **[Relating Superconducting Optoelectronic Networks to Classical Neurodynamics](https://arxiv.org/abs/2409.18016)** (arXiv 2024)
   - Connects SOEN dynamics to established neuroscience models

The tutorials in `src/soen_toolkit/tutorial_notebooks/` assume you've at least skimmed these papers.

### Optional Deep Dives

- [Josephson junction physics](DeeperDives/JJs.md) (for circuit-level understanding)


---

## 1.9 What's Next?

Continue to **[Getting_Started](Getting_Started.md)** for installation instructions, or jump to **[Building_Models](Building_Models.md)** for deeper conceptual understanding.

---

## Getting Help

- **Bug reports**: Open an issue on GitHub with a minimal reproducible example
- **Feature requests**: Describe your use case and why existing functionality doesn't work
- **Internal team**: Slack

Remember: This is research software. We prioritize rapid experimentation over polish. Your feedback helps improve it for everyone.

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="index.md" style="margin-right: 2em;">&#8592; Previous: Documentation Index</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Getting_Started.md" style="margin-left: 2em;">Next: Getting Started &#8594;</a>
</div>