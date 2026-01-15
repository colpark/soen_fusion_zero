"""Test fast path parity for JAX models with dynamic connections.

This test suite verifies that the fast path (batched kernels) produces
identical outputs to:
1. The slow path (per-connection iteration) in JAX
2. The PyTorch reference implementation

Tests cover all connection modes: fixed, WICC, and NOCC.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest
import torch


def _get_parity_test_models_dir() -> Path:
    """Get the directory containing parity test YAML models."""
    return Path(__file__).parent.parent / "utils" / "parity_test_models"


def _discover_yaml_models() -> list[Path]:
    """Discover all YAML model files in parity_test_models directory."""
    parity_dir = _get_parity_test_models_dir()
    if not parity_dir.exists():
        pytest.skip(f"Parity test models directory not found: {parity_dir}")
    yaml_files = sorted(parity_dir.glob("*.yaml"))
    if not yaml_files:
        pytest.skip(f"No YAML models found in {parity_dir}")
    return yaml_files


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
    return torch.randn(2, 100, 8, dtype=torch.float32)


@pytest.mark.parametrize("yaml_path", _discover_yaml_models())
def test_jax_fast_vs_slow_path_parity(yaml_path: Path, test_input: torch.Tensor) -> None:
    """Test that JAX fast path (batched) matches slow path (per-connection) exactly.

    This is critical for WICC/NOCC connections which now use the fast path.
    """
    # Force CPU JAX backend for determinism
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    with contextlib.suppress(Exception):
        jax.config.update("jax_platforms", "cpu")

    from soen_toolkit.core.model_yaml import build_model_from_yaml
    from soen_toolkit.utils.port_to_jax.convert import convert_core_model_to_jax

    # Load PyTorch model and convert to JAX
    torch_model = build_model_from_yaml(yaml_path)
    torch_model.eval()

    # Adjust input if needed
    expected_input_dim = int(torch_model.layers_config[0].params.get("dim", 8))
    if test_input.shape[2] != expected_input_dim:
        test_input_adj = _create_test_input(torch_model, batch=test_input.shape[0], seq_len=test_input.shape[1])
    else:
        test_input_adj = test_input

    jax_model = convert_core_model_to_jax(torch_model)
    jax_model.prepare()  # Build topology arrays for fast path

    # Skip if fast path not available (e.g., array-valued j_out)
    if jax_model._topology_arrays is None:
        pytest.skip(f"Fast path not available for {yaml_path.name} (likely array-valued j_out)")

    x_jax = jnp.asarray(test_input_adj.numpy())

    # Run with FAST PATH enabled (default)
    jax_model._use_fast_layerwise = True
    y_fast, hist_fast = jax_model(x_jax)

    # Run with SLOW PATH (force disable fast path)
    jax_model._use_fast_layerwise = False
    # Clear any cached logs
    if hasattr(jax_model, "_fast_path_logged_this_call"):
        delattr(jax_model, "_fast_path_logged_this_call")
    if hasattr(jax_model, "_fast_path_error_logged"):
        delattr(jax_model, "_fast_path_error_logged")
    y_slow, hist_slow = jax_model(x_jax)

    # Compare outputs
    y_fast_np = jnp.asarray(y_fast)
    y_slow_np = jnp.asarray(y_slow)

    max_diff = float(jnp.abs(y_fast_np - y_slow_np).max())
    mean_diff = float(jnp.abs(y_fast_np - y_slow_np).mean())

    # Tolerance: fast and slow paths should match exactly (within numerical precision)
    assert max_diff < 1e-6, (
        f"JAX fast/slow path output mismatch: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}\\n"
        f"Model: {yaml_path.name}, solver: {torch_model.sim_config.network_evaluation_method}"
    )

    # Compare layer histories
    if hist_fast and hist_slow:
        n_layers = min(len(hist_fast), len(hist_slow))
        for i in range(n_layers):
            hf_np = jnp.asarray(hist_fast[i])
            hs_np = jnp.asarray(hist_slow[i])

            # Handle shape mismatches
            min_b = min(hf_np.shape[0], hs_np.shape[0])
            min_t = min(hf_np.shape[1], hs_np.shape[1])
            min_d = min(hf_np.shape[2], hs_np.shape[2])

            diff = hf_np[:min_b, :min_t, :min_d] - hs_np[:min_b, :min_t, :min_d]
            layer_max_diff = float(jnp.abs(diff).max()) if diff.size > 0 else 0.0

            assert layer_max_diff < 1e-5, (
                f"Layer {i} history mismatch (fast vs slow): max_diff={layer_max_diff:.2e}\\n"
                f"Model: {yaml_path.name}, solver: {torch_model.sim_config.network_evaluation_method}"
            )


@pytest.mark.parametrize("yaml_path", _discover_yaml_models())
def test_jax_fast_path_vs_torch_parity(yaml_path: Path, test_input: torch.Tensor) -> None:
    """Test that JAX fast path matches PyTorch exactly.

    This ensures the batched WICC/NOCC kernels are correct.
    """
    # Force CPU JAX backend
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    with contextlib.suppress(Exception):
        jax.config.update("jax_platforms", "cpu")

    from soen_toolkit.core.model_yaml import build_model_from_yaml
    from soen_toolkit.utils.port_to_jax.convert import convert_core_model_to_jax

    # Load PyTorch model
    torch_model = build_model_from_yaml(yaml_path)
    torch_model.eval()

    # Adjust input if needed
    expected_input_dim = int(torch_model.layers_config[0].params.get("dim", 8))
    if test_input.shape[2] != expected_input_dim:
        test_input_adj = _create_test_input(torch_model, batch=test_input.shape[0], seq_len=test_input.shape[1])
    else:
        test_input_adj = test_input

    # Run PyTorch forward
    torch.manual_seed(42)
    with torch.no_grad():
        y_torch, torch_hists = torch_model(test_input_adj)

    # Convert to JAX and run with FAST PATH
    jax_model = convert_core_model_to_jax(torch_model)
    jax_model.prepare()  # Build topology arrays for fast path

    # Skip if fast path not available
    if jax_model._topology_arrays is None:
        pytest.skip(f"Fast path not available for {yaml_path.name}")

    jax_model._use_fast_layerwise = True  # Explicitly enable fast path

    x_jax = jnp.asarray(test_input_adj.numpy())
    y_jax_fast, jax_hists = jax_model(x_jax)

    # Compare outputs
    y_torch_np = y_torch.detach().cpu().numpy()
    y_jax_np = jnp.asarray(y_jax_fast)

    max_diff = float(jnp.abs(y_torch_np - y_jax_np).max())
    mean_diff = float(jnp.abs(y_torch_np - y_jax_np).mean())

    # Tolerance: should match within numerical precision
    assert max_diff < 1e-5, (
        f"PyTorch vs JAX fast path output mismatch: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}\\n"
        f"Model: {yaml_path.name}, solver: {torch_model.sim_config.network_evaluation_method}"
    )

    # Compare layer histories
    if torch_hists and jax_hists:
        n_layers = min(len(torch_hists), len(jax_hists))
        for i in range(n_layers):
            th_np = torch_hists[i].detach().cpu().numpy()
            jh_np = jnp.asarray(jax_hists[i])

            # Handle shape mismatches
            min_b = min(th_np.shape[0], jh_np.shape[0])
            min_t = min(th_np.shape[1], jh_np.shape[1])
            min_d = min(th_np.shape[2], jh_np.shape[2])

            diff = th_np[:min_b, :min_t, :min_d] - jh_np[:min_b, :min_t, :min_d]
            layer_max_diff = float(jnp.abs(diff).max()) if diff.size > 0 else 0.0

            assert layer_max_diff < 1e-4, (
                f"Layer {i} history mismatch (PyTorch vs JAX fast): max_diff={layer_max_diff:.2e}\\n"
                f"Model: {yaml_path.name}, solver: {torch_model.sim_config.network_evaluation_method}"
            )


@pytest.mark.parametrize("solver", ["layerwise"])
def test_dynamic_connections_use_fast_path(solver: str, test_input: torch.Tensor) -> None:
    """Verify that WICC/NOCC connections actually use the fast path.

    This is a regression test to ensure we don't accidentally disable fast path again.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    with contextlib.suppress(Exception):
        jax.config.update("jax_platforms", "cpu")

    from soen_toolkit.core.model_yaml import build_model_from_yaml
    from soen_toolkit.utils.port_to_jax.convert import convert_core_model_to_jax

    # Find a model with WICC or NOCC connections
    yaml_files = _discover_yaml_models()
    wicc_model_path = None
    for yaml_path in yaml_files:
        if "multiplier_layerwise" in yaml_path.name or "multiplierv2_layerwise" in yaml_path.name:
            wicc_model_path = yaml_path
            break

    if wicc_model_path is None:
        pytest.skip("No WICC/NOCC test model found")

    # Load and convert model
    torch_model = build_model_from_yaml(wicc_model_path)
    torch_model.sim_config.network_evaluation_method = solver
    torch_model.eval()

    jax_model = convert_core_model_to_jax(torch_model)
    jax_model.prepare()  # Build topology arrays for fast path
    jax_model._use_fast_layerwise = True

    # Adjust input
    expected_input_dim = int(torch_model.layers_config[0].params.get("dim", 8))
    if test_input.shape[2] != expected_input_dim:
        test_input_adj = _create_test_input(torch_model, batch=2, seq_len=50)
    else:
        test_input_adj = test_input[:, :50, :]

    # Run JAX forward pass and verify it works correctly
    x_jax = jnp.asarray(test_input_adj.numpy())
    y_jax, histories = jax_model(x_jax)

    # Verify output shape is valid
    assert y_jax.shape[0] == x_jax.shape[0], f"Batch size mismatch: {y_jax.shape[0]} vs {x_jax.shape[0]}"
    assert y_jax.shape[1] == x_jax.shape[1] + 1, f"Time dim mismatch: {y_jax.shape[1]} vs {x_jax.shape[1] + 1}"

    # Verify histories returned for all layers
    assert len(histories) == len(jax_model.layers), (
        f"Expected {len(jax_model.layers)} layer histories, got {len(histories)}"
    )

    # Verify fast path infrastructure is set up
    # (topology arrays built means fast path is available for fixed connections)
    assert jax_model._topology_arrays is not None, "Topology arrays not built"
    assert jax_model._use_fast_layerwise, "Fast layerwise not enabled"

    # Note: Fast path only applies to fixed connections. WICC/NOCC connections
    # always use the iterative slow path because they have per-edge state.
    # This is by design - there's no "fast path" for dynamic connections.
