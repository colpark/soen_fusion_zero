---
layout: default
title: GUI Tools
---
# GUI Tools

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="DATASETS.md" style="margin-right: 2em;">&#8592; Previous: Datasets</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Advanced_Features.md" style="margin-left: 2em;">Next: Advanced Features &#8594;</a>
</div>

The SOEN Toolkit includes a graphical interface for building, visualizing, and analyzing SOEN models. This page walks through creating a simple model from scratch using the GUI.

**Prerequisites**: Read through [Building Models](Building_Models.md) to understand the underlying concepts.

---

## Launching the GUI

Once you have activated the virtual environment as instructed in [Getting_Started](Getting_Started.md) you can launch the model creation graphical user interface from the command line by running:

```bash
(soen-toolkit) 
path/to/dir/ main ✔ 
▶  gui
```

The interface takes a while to load the first time you open it, but should be faster the second time. You'll see the main window with several tabs.

![Main Page](Figures/GUI_Tools/screenshot_mainpage.png)

---

## Walkthrough: Creating a Simple Model

We'll create a small feedforward model with 3 inputs, a hidden dendrite layer with 10 units, and 3 outputs.

### Step 1: Set Global Simulation Settings

Navigate to the **Model Builder** tab. The top bar contains global settings:

- **dt**: Dimensionless timestep. Set to `37` (≈0.1ns depending on $\omega_c$)
- **Global Solver**: Layerwise (default)
- **Input type**:
    - `State` (default): Input layer states are clamped to the input values (bypasses layer dynamics)
    - `Flux`: Input values are processed through the first layer's dynamics
- **Dynamic tracking**: Enable if you want to track internal states during simulation

![Building Page](Figures/GUI_Tools/screenshot_building_page.png)

---

### Step 2: Create the Input Layer

Click **Add Layer** to create the first layer. This will be a placeholder for input sequences. Note that the first layer is always treated as the input layer.

**Configuration:**
- **Layer type**: `Linear`
- **Dimensions**: `3` (this has to match the number of input features in your dataset when training)

![Input Layer Creation](Figures/GUI_Tools/temp_dump/image_4.png)

**Note**: In the SOEN toolkit, layers represent physical sets of circuits/nodes, and connections between them are specified separately. This differs from typical nn.Sequential PyTorch networks for instance where:
- **PyTorch**: Each layer has `in_dim` and `out_dim`, implicitly defining connectivity to the previous layer
- **SOEN**: Layers define only their own dimensionality (number of circuits/nodes). Connectivity is explicitly specified as separate connection objects between layer pairs. The reason for this is it allows arbitrary connectivity structures in our networks.

This design reflects the physical reality: you fabricate collections of superconducting circuits (layers), then wire them together (connections). The input layer is a placeholder for current sources that feeds data into the first physical layer.

---

### Step 3: Add a Dendrite Layer

Click **Add Layer** again to create a hidden layer.

**Configuration:**
- **Layer type**: `Dendrite`
- **Dimensions**: `10` (10 dendrites)
- **Solver**: Forward Euler (default)
- **Source function**: Fitted (nonlinearity approximates the rate array lookup table; see [Josephson Junction Physics](DeeperDives/JJs.md))

![Dendrite Layer](Figures/GUI_Tools/temp_dump/image_5.png)


---

### Step 4: Inspect Layer Parameters

Navigate to the **Parameters** tab to see default parameter initializations for the dendrite layer.

![Parameters Tab](Figures/GUI_Tools/temp_dump/image_6.png)

**Notes:**
- **Learnable toggles**: If you enable "learnable" for a parameter (e.g., `phi_offset`), each node gets its own trainable value that can be optimized. You can set limits in the **Constraints** section.
- **Constraints**: Applied after initialization to clip sampled values (e.g., from normal distributions) to valid ranges. Also applied during training after every optimizer step.
- **Descriptions**: You can add layer descriptions (optional) that appear during visualization.

---

### Step 5: Create an Output Layer

Add one more layer:

**Configuration:**
- **Layer type**: `Linear`
- **Dimensions**: `3` (matching number of output targets)

You should now have 3 layers total: `input_layer`, `dendrite_layer`, `output_layer`.

---

### Step 6: Connect the Layers

#### Auto-Connect (Quick Method)

Click the **Auto Connect** button to create feedforward connections automatically.

![Auto Connect](Figures/GUI_Tools/temp_dump/image_7.png)


This creates:
- Connection from layer 0 → layer 1
- Connection from layer 1 → layer 2

#### Manual Connection Editing

Double-click any connection to open the connection dialog:

![Auto Connect Dialog](Figures/GUI_Tools/temp_dump/image_8.png)
![Connection Creation](Figures/GUI_Tools/connection_creation.png)

**Connection settings:**
- **From Layer**: Source layer ID
- **To Layer**: Destination layer ID
- **Connectivity**: `Dense` (all-to-all)
- **Weights initialization**: Normal distribution (mean=0, std=1)
- **Learnable**: Toggle to freeze/unfreeze weights during training

**Tip**: To add intra-layer (recurrent) connections, set both "From" and "To" to the same layer ID.


---

### Step 7: Build the Model

Click **Build from Model Specification**.

![Connection Dialog Details](Figures/GUI_Tools/temp_dump/image_9.png)

**Build options:**
- **Seed**: Set for reproducible initialization (weights are sampled from distributions)
- **Preserve on rebuild**: For model composition workflows (can ignore for now)

You'll see a popup confirming "Model successfully built".

---

### Step 8: Visualize the Network

Navigate to the **Visualize** tab. Multiple visualization modes are available.

#### Simple View

Click **Render** to generate an SVG showing layer-level connectivity.

![Simple View](Figures/GUI_Tools/temp_dump/image_10.png)


#### Detailed View

In the **Settings** panel, turn off "Simple View" to see individual nodes and connections.

![Detailed View Example](Figures/GUI_Tools/temp_dump/image_11.png)

**Interactive features:**
- Click nodes to see layer/unit information
- Adjust settings to show/hide intra-layer connections
- Export as SVG or PNG

---

### Step 9: Save the Model

Go to **File → Save Model** (or use the save button).

![Save Dialog](Figures/GUI_Tools/temp_dump/image_12.png)

**Export formats:**

![Export Options](Figures/GUI_Tools/temp_dump/image_13.png)

- **`.soen`**: Full model with weights (recommended)
- **`.json`**: Human-readable version of `.soen`
- **`_spec.yaml`**: Model specification only (architecture, no weights)

---

## Testing Your Model

Before training, it's good practice to run a forward pass and inspect state trajectories.

### Step 10: Analyze State Trajectories

Navigate to **Analyze → Summary → Plot Trajectory**.

![State Trajectory Tool Settings](Figures/GUI_Tools/state_traj_settings.png)

**Features:**
- Pass various input signals through the model
- Visualize each layer's state evolution over time
- Measure energy usage (if power tracking is enabled and a supported layer type is used)

**Energy tracking (optional):**
- Enable "Show total line" in display options
- Select metric: `Energy (cumulative power)`
- Typical small networks: ~1e-15 J per inference (1 fJ)
- Note: Total energy includes ~300× multiplier for cryogenic cooling overhead

![Input Signals Example](Figures/GUI_Tools/state_traj_input.png)
*Example input signals passed to the model (square wave at 100 MHz).*

![Output Layer State Trajectory](Figures/GUI_Tools/state_traj_output.png)
*State trajectory of the output layer in response to the input signals.*

---



## Additional Features

### Unit Converter

Access physical↔dimensionless conversions in the **Unit Conversion** tab. Click **Launch** to embed the converter webapp.

![Unit Conversion](Figures/GUI_Tools/screenshot_unit_conversion_mainpage.png)

See the [Unit Converter guide](Unit_Converter.md) for details.

### Model Composition

Model composition allows you to combine multiple separate `.soen` models into a single workspace. This is useful when you want to connect pre-trained models together or build complex architectures from modular components.

**How it works:**

1. **Load the first model** normally using File → Load Model
2. **Load additional models** using File → Load Model, but select the **"Merge as additional (ID-Shift)"** action in the load dialog
3. The framework automatically:
   - Shifts all layer IDs in the new model to avoid conflicts (e.g., if your first model has layers 0-3, the second model's layers are shifted by 4)
   - Updates all connection indices to match the new layer IDs
   - Increments the Model ID attribute for each layer (used for visualization and bookkeeping)
4. **Connect the models** by creating new connections between layers from different models
5. **Optional:** Enable "Preserve on rebuild" for models whose weights you want to keep
6. **Rebuild** to create a single unified model

**Example workflow:**
```
First model: Input → Hidden → Output (layers 0, 1, 2)
Second model: Input → Hidden → Output (layers 0, 1, 2)

After ID-Shift:
First model: layers 0, 1, 2
Second model: layers 3, 4, 5  ← automatically shifted

Now you can add connections like:
- layer 2 → layer 3 (connect output of first model to input of second)
```

**Tip:** The Model ID attribute is visible in the visualizer, making it easy to see which layers came from which original model.

### Layer Merging

Layer merging is a different feature that collapses multiple layers *within a single model* into a functionally equivalent "super-layer". This is useful for simplifying architectures, reducing the number of solver steps, or preparing models for specific hardware constraints.

**Access:** Model Builder tab → **Merge Layers** button

![Automerge Workflow](Figures/GUI_Tools/automerge_workflow.png)

**What it does:**

Takes a contiguous group of same-type layers and replaces them with a single larger layer by:
- Concatenating node-wise parameters (like `gamma_plus`, `phi_offset`)
- Assembling connection weight matrices into the appropriate block structure
- Preserving the exact numeric behavior of the original layer group

**Example:**
```
Before:
Layer 1 (Dendrite, dim=10) → Layer 2 (Dendrite, dim=15) → Layer 3 (Dendrite, dim=10)

After merging layers 1, 2, 3:
Merged Layer (Dendrite, dim=35)  ← functionally identical to the original group
```

**Key requirements:**
- All selected layers must be the same type (e.g., all Dendrite, all Linear)
- The layers should typically be contiguous in the network topology
- The merged layer is assigned a new ID (defaults to the minimum ID of the group)


**Important distinction:**
- **Model Composition** = combining separate models into one workspace
- **Layer Merging** = collapsing layers within a model into a super-layer

---

## Tips

- **Layer descriptions**: Add Markdown descriptions to document each layer's purpose
- **Seed everything**: Use consistent seeds for reproducible model creation
- **Visualize early**: Check connectivity before training to catch mistakes
- **Test trajectories**: Always run a forward pass to verify signal propagation
- **Model IDs for composition**: When composing models, pay attention to the Model ID attribute in the visualizer to track which layers came from which original model
- **Layer merging validation**: The GUI will warn you if your layer selection creates feedback loops or affects input/output layers—these merges are allowed but may require careful solver configuration


---

## Summary

The Model Creation GUI provides:
- Visual layer and connection design
- Parameter inspection and constraint setting
- Multiple visualization modes (simple, detailed, matrix, grid)
- State trajectory analysis with energy tracking
- Model export in multiple formats
- Unit conversion tools
- Model composition workflows

For programmatic model creation, see the [Building Models](Building_Models.md) guide.

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="DATASETS.md" style="margin-right: 2em;">&#8592; Previous: Datasets</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Advanced_Features.md" style="margin-left: 2em;">Next: Advanced Features &#8594;</a>
</div>
