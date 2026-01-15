"""Test parity YAML models for forward determinism and Torch/JAX equivalence.

This test suite loads all YAML models from parity_test_models/ and verifies:
1. Forward pass determinism (same input produces same output)
2. PyTorch vs JAX parity (JAX produces same output as PyTorch)
3. All global solvers (layerwise, stepwise_gauss_seidel, stepwise_jacobi)
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest
import torch

from soen_toolkit.core.model_yaml import build_model_from_yaml


def _get_parity_test_models_dir() -> Path:
    """Get the directory containing parity test YAML models."""
    return Path(__file__).parent.parent.parent / "tests" / "utils" / "parity_test_models"


def _discover_yaml_models() -> list[Path]:
    """Discover all YAML model files in parity_test_models directory."""
    parity_dir = _get_parity_test_models_dir()
    if not parity_dir.exists():
        pytest.skip(f"Parity test models directory not found: {parity_dir}")
    yaml_files = sorted(parity_dir.glob("*.yaml"))
    if not yaml_files:
        pytest.skip(f"No YAML models found in {parity_dir}")
    return yaml_files


def _load_test_data() -> torch.Tensor | None:
    """Load test data if available."""
    parity_dir = _get_parity_test_models_dir()
    test_data_path = parity_dir / "test_data_8ch_166seq.npy"
    if test_data_path.exists():
        import numpy as np

        data = np.load(test_data_path)
        # Convert to [B, T, D] format
        if data.ndim == 2:
            data = data[np.newaxis, :, :]  # Add batch dimension
        return torch.from_numpy(data.astype("float32"))
    return None


def _create_test_input(model, batch: int = 2, seq_len: int = 100) -> torch.Tensor:
    """Create test input matching model's input dimension."""
    if len(model.layers_config) == 0:
        raise RuntimeError("Model has no layers")
    input_dim = int(model.layers_config[0].params.get("dim", 8))
    return torch.randn(batch, seq_len, input_dim, dtype=torch.float32)


@pytest.fixture(scope="module")
def test_input():
    """Create a deterministic test input."""
    torch.manual_seed(42)
    return torch.randn(2, 166, 8, dtype=torch.float32)


@pytest.mark.parametrize("yaml_path", _discover_yaml_models())
def test_yaml_model_forward_determinism(yaml_path: Path, test_input: torch.Tensor) -> None:
    """Test that forward pass is deterministic (same input -> same output)."""
    # Load model
    model = build_model_from_yaml(yaml_path)
    model.eval()

    # Adjust input if needed
    expected_input_dim = int(model.layers_config[0].params.get("dim", 8))
    if test_input.shape[2] != expected_input_dim:
        test_input = _create_test_input(model, batch=test_input.shape[0], seq_len=test_input.shape[1])

    # Run forward pass multiple times with same seed
    outputs = []
    for _ in range(3):
        torch.manual_seed(42)
        with torch.no_grad():
            y, _ = model(test_input)
        outputs.append(y.clone())

    # All outputs should be identical
    max_diff_12 = (outputs[0] - outputs[1]).abs().max().item()
    max_diff_13 = (outputs[0] - outputs[2]).abs().max().item()

    assert max_diff_12 == 0.0, f"Forward pass not deterministic: max diff run1-2 = {max_diff_12:.2e}"
    assert max_diff_13 == 0.0, f"Forward pass not deterministic: max diff run1-3 = {max_diff_13:.2e}"


@pytest.mark.parametrize("yaml_path", _discover_yaml_models())
def test_yaml_model_torch_jax_parity(yaml_path: Path, test_input: torch.Tensor) -> None:
    """Test that JAX model produces same output as PyTorch model."""
    # Force CPU JAX backend for stability
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    with contextlib.suppress(Exception):
        jax.config.update("jax_platforms", "cpu")

    # Load PyTorch model
    torch_model = build_model_from_yaml(yaml_path)
    torch_model.eval()

    # Adjust input if needed
    expected_input_dim = int(torch_model.layers_config[0].params.get("dim", 8))
    if test_input.shape[2] != expected_input_dim:
        test_input = _create_test_input(torch_model, batch=test_input.shape[0], seq_len=test_input.shape[1])

    # Run PyTorch forward
    torch.manual_seed(42)
    with torch.no_grad():
        y_torch, torch_hists = torch_model(test_input)

    # Convert to JAX
    from soen_toolkit.utils.port_to_jax.convert import convert_core_model_to_jax

    jax_model = convert_core_model_to_jax(torch_model)

    # Run JAX forward
    x_jax = jnp.asarray(test_input.numpy())
    y_jax, jax_hists = jax_model(x_jax)

    # Compare outputs
    y_torch_np = y_torch.detach().cpu().numpy()
    y_jax_np = jnp.asarray(y_jax)

    max_diff = float(jnp.abs(y_torch_np - y_jax_np).max())
    mean_diff = float(jnp.abs(y_torch_np - y_jax_np).mean())

    # Tolerance: should match within numerical precision
    assert max_diff < 1e-5, (
        f"PyTorch/JAX output mismatch: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}\n"
        f"Model: {yaml_path.name}, solver: {torch_model.sim_config.network_evaluation_method}"
    )

    # Compare layer histories if available
    if torch_hists and jax_hists:
        n_layers = min(len(torch_hists), len(jax_hists))
        for i in range(n_layers):
            th_np = torch_hists[i].detach().cpu().numpy()
            jh_np = jnp.asarray(jax_hists[i])

            # Handle shape mismatches (might differ by 1 timestep)
            min_b = min(th_np.shape[0], jh_np.shape[0])
            min_t = min(th_np.shape[1], jh_np.shape[1])
            min_d = min(th_np.shape[2], jh_np.shape[2])

            diff = th_np[:min_b, :min_t, :min_d] - jh_np[:min_b, :min_t, :min_d]
            layer_max_diff = float(jnp.abs(diff).max()) if diff.size > 0 else 0.0

            assert layer_max_diff < 1e-4, (
                f"Layer {i} history mismatch: max_diff={layer_max_diff:.2e}\n"
                f"Model: {yaml_path.name}, solver: {torch_model.sim_config.network_evaluation_method}"
            )


@pytest.mark.parametrize("solver", ["layerwise", "stepwise_gauss_seidel", "stepwise_jacobi"])
def test_all_solvers_with_parity_models(solver: str, test_input: torch.Tensor) -> None:
    """Test that all solvers work with parity test models.

    This test takes the first parity model and tests it with different solvers.
    """
    yaml_files = _discover_yaml_models()
    if not yaml_files:
        pytest.skip("No parity YAML models found")

    # Use first model as base
    base_yaml = yaml_files[0]

    # Load model
    model = build_model_from_yaml(base_yaml)

    # Override solver
    original_solver = model.sim_config.network_evaluation_method
    model.sim_config.network_evaluation_method = solver

    # Adjust input
    expected_input_dim = int(model.layers_config[0].params.get("dim", 8))
    if test_input.shape[2] != expected_input_dim:
        test_input = _create_test_input(model, batch=test_input.shape[0], seq_len=test_input.shape[1])

    # Test forward pass works
    model.eval()
    torch.manual_seed(42)
    with torch.no_grad():
        y, _ = model(test_input)

    # Verify output shape is reasonable
    assert y.shape[0] == test_input.shape[0], "Batch dimension mismatch"
    assert y.shape[1] >= test_input.shape[1], "Sequence length should be >= input length"
    assert y.shape[2] > 0, "Output dimension should be positive"

    # Restore original solver
    model.sim_config.network_evaluation_method = original_solver


@pytest.mark.parametrize("yaml_path", _discover_yaml_models())
def test_yaml_model_builds_and_runs(yaml_path: Path, test_input: torch.Tensor) -> None:
    """Basic smoke test: model builds and runs forward pass."""
    model = build_model_from_yaml(yaml_path)
    model.eval()

    # Adjust input if needed
    expected_input_dim = int(model.layers_config[0].params.get("dim", 8))
    if test_input.shape[2] != expected_input_dim:
        test_input = _create_test_input(model, batch=test_input.shape[0], seq_len=test_input.shape[1])

    # Run forward pass
    with torch.no_grad():
        y, histories = model(test_input)

    # Basic assertions
    assert y is not None, "Output should not be None"
    assert y.shape[0] == test_input.shape[0], "Batch dimension should match"
    assert len(histories) == len(model.layers_config), "Number of histories should match number of layers"
    assert torch.isfinite(y).all().item(), "Output should be finite"

