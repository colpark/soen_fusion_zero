# Criticality HPO — README

> **TL;DR**  
>
> - **Uses CMA-ES with restarts (BIPOP)** and **TPE (or Random) independent sampler** for continuous and conditional/categorical params respectively.  
> - Stores runs under `criticality_runs/optuna_report_<study_name>`, with a JSON summary and a YAML of the best spec.
> Currently the best way to use the module is to run the GUI via `python -m soen_toolkit.utils.hpo`. Load in a HPO config modeled after one of the example YAML's, and running the optimization should be straightforward.

---

## What's here

This repo/script is an Optuna-driven hyperparameter search to push a SOEN model toward **criticality** using several metrics. The system now supports **multiple spatial connectivity patterns** (power_law, exponential, constant) and **improved spatial PSD analysis** that works with any layer size:

- **Branching ratio** (σ, with a robust regression estimator)
  - For layers with outgoing connections: solves φ_out ≈ σ · φ_in where φ_out uses the sum of outgoing weights per source neuron.
  - For terminal layers (no outgoing connections): φ_out is taken as the layer's states themselves (φ_out = s), so terminal layers are included in σ.
- **Temporal PSD slope** (βₜ)
- **Spatial PSD slope** (β_spatial; radial 2D FFT ring-averaged, now supports all layer sizes)
- **Susceptibility** proxy (variance of total activity)
- **Avalanche** size proxy (burst statistics)
- **Autocorrelation** decay exponent (α)
- **Jacobian Spectral Radius**: Pushes the spectral radius (largest eigenvalue magnitude) of the system's Jacobian toward 1.0, another indicator of a system at the "edge of chaos."
- **Largest Lyapunov Exponent**: Aims for a largest Lyapunov exponent of zero, which characterizes the transition between stable (negative exponent) and chaotic (positive exponent) dynamics.

The objective is a **weighted sum of costs** (minimize) that’s turned into a score (maximize `-total_cost` for Optuna).

---

## Layout

This directory is organized into several submodules, each handling a specific part of the HPO process.

-   **`scripts/`**: Main entry point for running HPO studies.
    -   `run_hpo.py`: The primary CLI script for launching an optimization run.
    -   **Usage**: `python -m soen_toolkit.utils.hpo.scripts.run_hpo --hp-config path/to/your_hpo_config.yaml`

-   **`config/`**: Manages loading, validation, and processing of HPO configuration files.
    -   `hpo_config.py`: Contains `load_hpo_config`, which reads the main YAML file, resolves paths, and can automatically generate a default `optimization_config` from a base model spec.

-   **`study/`**: Core logic for the Optuna study, including sampling, objectives, and adapters.
    -   `sampling.py`: Defines `ConfigurableHyperparameterSampler`, which interprets the `optimization_config` in your YAML to define the search space for Optuna.
    -   `optuna_adapter.py`: A thin wrapper around Optuna to create samplers, pruners, and study objects based on the configuration.
    -   `objective/`: Implements the modular objective function. `core.py` calculates the final score based on weighted metrics defined in `metrics_builtin.py`.

-   **`io/`**: Handles all file input/output, such as saving results and managing paths.
    -   `exporters.py`: Writes artifacts like `optuna_summary.json`, `best_spec.yaml`, and a live-updating `trials.jsonl` log.

-   **`inputs/`**: Manages the generation or loading of input data for model evaluation.
    -   `builtin.py`: Provides various input generators, including white/colored noise and data from HDF5 files.

-   **`tools/`**: Contains helper utilities for scaffolding configurations and analyzing results.
    -   `enumerate_model_options.py`: A CLI tool to generate a skeleton `optimization_config` from a model spec, making it easier to set up a new HPO study.
    -   `hpo_gui.py`: The entry point for the HPO GUI.

-   **`templates/`**: Includes example `HPO_config.yaml` and model spec files to use as a starting point.

-   **`extract_trial_spec.py`**: A standalone script for extracting the model spec of a specific trial from a completed study.

---

## Using the GUI (HPO-first workflow)

The GUI is organized around a single source of truth: the HPO YAML.

- Open HPO Config…: load an existing HPO YAML. The app:
  - Normalizes paths, resolves/extracts `paths.base_model_spec` if needed,
  - Backfills `optimization_config` targets/spaces when missing,
  - Populates all tabs from the YAML immediately (no need to load a model separately).

- New From Model…: create a new HPO YAML from a model file:
  - Accepts a model spec (`.yaml/.yml`) or a trained model (`.soen/.pth/.json`).
  - If trained, it extracts `<output_dir>/<stem>_spec_for_hpo.yaml` and uses that.
  - Builds a starter `optimization_config` from the spec, then prompts you to save the HPO YAML, and opens it.

Notes:
- While a project (HPO YAML) is loaded, ad‑hoc model picking is disabled to avoid conflicting sources. Use “New From Model…” to start a separate project.
- The Results tab includes an “Extract Trial Spec” tool: enter a trial number and generate the corresponding YAML spec using the current HPO config and study.

Troubleshooting:
- If the base model spec or dataset path is missing, the loader will warn you. Use the file pickers to locate the correct path and save the YAML again.

---

## Results & Plots 

- Scatter (custom X/Y): choose any X (e.g., `trial_number`, `param_*`, `metric_*`) and any numeric Y (e.g., `value`, `metric_*`).
  - Categorical X: plotted with light jitter; shows category means.
  - Numeric X: optional regression overlay.
    - Linear fit available by default (shows R²).
    - LOWESS fit available when `statsmodels` is installed.
  - Point click selects the corresponding trial in the table and opens a details dialog.
- Importance: top‑N parameter importance (|corr|) with adjustable N.
- Export: “Export Current Plot…” saves the current Plotly figure to a standalone HTML file (includes Plotly via CDN).
- Performance: the “Performance” sub‑tab has been removed from the Plots tab to avoid duplication; use the live performance chart in the Run tab.

Optional dependencies for richer plots:
- `PyQt6-WebEngine` (recommended) to render interactive Plotly in‑app; falls back to PyQtGraph when unavailable.
- `statsmodels` for LOWESS regression in the Scatter plot.

## Spatial Connectivity Patterns

The HPO system supports **three spatial connectivity patterns** for critical network design:

- **power_law**: `p(w_ij) = p_0 * d_ij^(-α)` - Distance-biased with power-law decay
- **exponential**: `p(w_ij) = p_0 * exp(-d_ij/d_0)` - Exponential distance decay  
- **constant**: `p(d) = p_0` - Uniform connectivity probability

All patterns use deterministic 2D grid coordinates (`_grid_coords`) and support the same parameters:
- `expected_fan_out`: Average number of connections per neuron
- `allow_self_connections`: Whether to allow recurrent connections

Additional pattern-specific parameters:
- **power_law**: `alpha` (decay exponent, default 2.0)
- **exponential**: `d_0` (characteristic length scale, default 2.0)

## Configuration (YAML)

### HPO Modes

The HPO toolkit now supports two modes, configured via the `hpo_mode` key in your `HPO_config.yaml`:

1.  **`forward` (Default)**: This is the original simulation-only mode. Each trial generates a model specification, which is then run through a simulation to evaluate criticality metrics. This mode is ideal for optimizing network structure and parameters based on intrinsic dynamics.

2.  **`epoch`**: This new mode integrates model training into the HPO loop. For each trial, the generated model is trained for a single epoch. The final training loss is included as a component in the objective function, allowing optimization for both criticality and task performance.

### Configuration for `'epoch'` Mode

To use the `'epoch'` mode, you must provide a path to a training configuration file:

```yaml
run:
  hpo_mode: 'epoch'
  # ... other run settings

paths:
  train_config: 'path/to/your/train_config.yaml'
  # ... other paths
```

-   `hpo_mode`: Set to `'epoch'`.
-   `train_config`: A path (absolute or relative to the HPO config) to a valid training configuration YAML. This file defines the dataset, optimizer, and other training-related settings.

When `hpo_mode` is `'forward'`, the `train_config` key is ignored.

See `hpo/templates/ExampleHPOSpecs/HPO_config.yaml` for an example with extensive comments.

