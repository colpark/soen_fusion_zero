# State Trajectory Module - Refactored Architecture

## Overview

This module provides state trajectory plotting and simulation for SOEN models. 

## Module Structure

```
state_trajectory/
├─ __init__.py               # Public API exports
├─ errors.py                 # Local exceptions
├─ settings.py               # Dataclasses + QSettings adapter
├─ timebase.py              # Time conversions (dt, fs, ns)
├─ padding.py               # Zero padding operations
├─ inputs.py                # Input source strategies
├─ sim_backends.py          # Torch/JAX backend runners
├─ jax_cache.py             # JAX model caching with fingerprinting
├─ model_adapter.py         # Model operations façade
├─ dataset_service.py       # Dataset loading and management
├─ plotting.py              # Plot rendering
├─ export.py                # Atomic export with metadata
├─ controller.py            # Orchestrates simulation runs
├─ dialog.py                # Thin Qt UI layer
└─ tests/                   # Comprehensive unit tests
   ├─ test_timebase.py
   ├─ test_padding.py
   ├─ test_inputs.py
   ├─ test_model_adapter.py
   └─ ...
```

## Key Design Principles

### 1. Single Responsibility
Each module has one clear purpose:
- `timebase.py`: Time conversions only
- `padding.py`: Padding operations only
- `inputs.py`: Input generation strategies only
- etc.

### 2. Encapsulation
Implementation details hidden behind clean interfaces:
- `QSettingsAdapter` hides Qt persistence
- `JaxModelCache` hides JAX conversion complexity
- `ModelAdapter` isolates model quirks

### 3. Loose Coupling
Modules communicate through well-defined interfaces:
- Controller uses `InputSource` protocol, not concrete classes
- Backends implement `BackendRunner` interface
- Dialog only knows about Controller, not simulation details

### 4. Testability
All non-UI logic is testable without Qt:
- `Timebase`: Pure math, no dependencies
- `Padding`: Pure tensor operations
- `Inputs`: Strategy pattern with dependency injection
- `Controller`: Orchestration with mockable dependencies

### 5. Determinism
Fixed multiple sources of non-determinism:
- `TorchRunner._deterministic_context`: Seeds all RNGs
- Single `dt` application point in Controller
- Parameter fingerprinting for JAX cache invalidation

## Usage

### Basic Usage

```python
from soen_toolkit.model_creation_gui.state_trajectory import StateTrajectoryDialog

# In your GUI
dialog = StateTrajectoryDialog(parent=self, manager=model_manager)
dialog.show()
```

### Programmatic Usage (No GUI)

```python
from soen_toolkit.model_creation_gui.state_trajectory import (
    StateTrajectoryController,
    StateTrajSettings,
    Backend,
    Metric,
)
from soen_toolkit.model_creation_gui.state_trajectory.model_adapter import ModelAdapter
from soen_toolkit.model_creation_gui.state_trajectory.dataset_service import DatasetService
from soen_toolkit.model_creation_gui.state_trajectory.sim_backends import TorchRunner

# Set up services
dataset_service = DatasetService()
model_adapter = ModelAdapter()
backends = {
    Backend.TORCH: TorchRunner(model, model_adapter)
}

# Create controller
controller = StateTrajectoryController(
    model_manager,
    dataset_service,
    backends,
    model_adapter
)

# Configure settings
settings = StateTrajSettings(
    metric=Metric.STATE,
    backend=Backend.TORCH,
    seq_len=100,
    dt=37.0,
    # ... other settings
)

# Run simulation
timebase, input_features, metric_histories, raw_histories, elapsed = controller.run(settings)
```

## Key Bug Fixes

### 1. Determinism
**Problem**: Runs with same settings produced different results.
**Solution**: `TorchRunner._deterministic_context` seeds all RNGs (random, numpy, torch), sets eval mode, and resets state.

### 2. dt Application
**Problem**: dt applied multiple times or inconsistently.
**Solution**: Controller applies dt once before input generation, using `Timebase` as single source of truth.

### 3. JAX Cache Invalidation
**Problem**: JAX model not rebuilt when parameters changed.
**Solution**: `JaxModelCache` computes SHA256 fingerprint of `state_dict()`, invalidates cache on mismatch.

### 4. Legend Performance
**Problem**: Creating legend with hundreds of traces slowed rendering.
**Solution**: Only add legend when `show_legend` or `show_total` enabled.

### 5. Duplicate Forward Pass
**Problem**: `_on_plot` ran simulation twice (once timed, once for display).
**Solution**: Single forward pass in backends, results reused for plotting.

## Testing

Run all tests:
```bash
pytest src/soen_toolkit/model_creation_gui/state_trajectory/tests/
```

Run specific module tests:
```bash
pytest src/soen_toolkit/model_creation_gui/state_trajectory/tests/test_timebase.py
pytest src/soen_toolkit/model_creation_gui/state_trajectory/tests/test_padding.py
pytest src/soen_toolkit/model_creation_gui/state_trajectory/tests/test_inputs.py
```

## Migration Notes

### Breaking Changes
None - the public API (`StateTrajectoryDialog`) remains unchanged.

### Import Updates
Old:
```python
from soen_toolkit.model_creation_gui.components.state_trajectory_dialog import StateTrajectoryDialog
```

New:
```python
from soen_toolkit.model_creation_gui.state_trajectory import StateTrajectoryDialog
```

### Settings Persistence
Settings are now managed through `QSettingsAdapter` and stored as structured dataclasses. Old settings will be automatically migrated on first load.

## Performance Improvements

1. **Legend rendering**: Only added when needed (not for every plot)
2. **JAX caching**: Avoids redundant recompilation
3. **Lazy dataset loading**: Dataset only loaded when needed
4. **Single forward pass**: Removed duplicate simulation run

## Future Enhancements

Potential improvements (not currently implemented):

1. **Level of Detail (LOD)**: Decimate dense traces for faster rendering
2. **Async simulation**: Run simulation in background thread
3. **Batch export**: Export multiple configurations at once
4. **Custom input sources**: Plugin architecture for user-defined generators
5. **Streaming plots**: Update plots during long simulations

## Dependencies

- PyTorch: Simulation and tensor operations
- NumPy: Input generation and array operations
- PyQt6: GUI framework
- pyqtgraph: Plotting widgets
- JAX (optional): JAX backend support
- h5py: HDF5 dataset support

## License

Same as parent project.

