
import os
import tempfile

import pytest
import torch

from soen_toolkit.core.model_yaml import build_model_from_yaml
from soen_toolkit.utils.flux_matching_init import (
    FluxMatchingConfig,
    _compute_phi_total_per_layer,
    apply_inhibitory_fraction_preserving_flux,
    compute_flux_per_node,
    compute_total_flux_per_layer,
    initialize_model_with_flux_matching,
)


@pytest.fixture
def simple_model_yaml():
    yaml_content = """
simulation:
  dt: 0.1
layers:
  - layer_id: 0
    layer_type: Input
    params:
      dim: 20
  - layer_id: 1
    layer_type: SingleDendrite
    params:
      dim: 10
      source_function: heaviside
      phi_offset: 0.0
connections:
  - from_layer: 0
    to_layer: 1
    connection_type: dense
"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_content)
        path = f.name
    yield path
    os.unlink(path)

def test_flux_matching_convergence(simple_model_yaml):
    model = build_model_from_yaml(simple_model_yaml)

    # Generate random data
    data = torch.rand(32, 50, 20)  # [batch, seq, dim]

    # Run flux matching
    phi_target = 0.5
    result = initialize_model_with_flux_matching(
        model,
        data,
        phi_target=phi_target,
        iterations=5,
        alpha=1.0,
        verbose=False
    )

    assert result.converged

    # Verify final flux
    mean_states = result.final_mean_states
    flux_per_conn = compute_flux_per_node(model, mean_states)
    total_flux_exc = compute_total_flux_per_layer(model, flux_per_conn)
    total_phi = _compute_phi_total_per_layer(model, total_flux_exc)

    final_flux = total_phi[1]
    assert torch.allclose(final_flux, torch.tensor(phi_target), atol=1e-3)

def test_inhibitory_fraction_preservation(simple_model_yaml):
    model = build_model_from_yaml(simple_model_yaml)
    data = torch.rand(32, 50, 20)

    # 1. Run flux matching first to get baseline
    phi_target = 0.5
    result = initialize_model_with_flux_matching(
        model,
        data,
        phi_target=phi_target,
        iterations=5,
        verbose=False
    )

    # 2. Apply inhibitory fraction
    inhibitory_fraction = 0.3
    apply_inhibitory_fraction_preserving_flux(
        model,
        result.final_mean_states,
        config=FluxMatchingConfig(phi_total_target=phi_target),
        inhibitory_fraction=inhibitory_fraction,
        seed=42
    )

    # 3. Verify target flux is still preserved
    flux_per_conn = compute_flux_per_node(model, result.final_mean_states)
    total_flux_exc = compute_total_flux_per_layer(model, flux_per_conn)
    total_phi = _compute_phi_total_per_layer(model, total_flux_exc)

    final_flux = total_phi[1]
    assert torch.allclose(final_flux, torch.tensor(phi_target), atol=1e-3)

    # 4. Verify inhibitory fraction is correct
    weights = model.connections["J_0_to_1"]
    mask = model.connection_masks["J_0_to_1"]

    for i in range(weights.shape[0]):
        row_weights = weights[i][mask[i] > 0]
        num_neg = (row_weights < 0).sum().item()
        fan_in = row_weights.numel()
        expected_neg = int(fan_in * inhibitory_fraction)
        assert num_neg == expected_neg

def test_flux_matching_with_offset(simple_model_yaml):
    # Create model with non-zero offset
    model = build_model_from_yaml(simple_model_yaml)

    # Set manual offset for layer 1
    # We need to find the layer object.
    layer1 = None
    for idx, cfg in enumerate(model.layers_config):
        if cfg.layer_id == 1:
            layer1 = model.layers[idx]
            break

    phi_offset = torch.linspace(0.0, 0.2, 10)
    # Mocking standard parameter registration if it's not already there
    if hasattr(layer1, "phi_offset"):
        with torch.no_grad():
            if isinstance(layer1.phi_offset, torch.Tensor):
                layer1.phi_offset.copy_(phi_offset)
            else:
                # Some implementations might use a parameter or dict
                pass

    data = torch.rand(32, 50, 20)
    phi_target = 0.6

    result = initialize_model_with_flux_matching(
        model,
        data,
        phi_target=phi_target,
        iterations=10,
        verbose=False
    )

    # Verify final total flux (phi_exc + phi_offset) matches target
    mean_states = result.final_mean_states
    flux_per_conn = compute_flux_per_node(model, mean_states)
    total_flux_exc = compute_total_flux_per_layer(model, flux_per_conn)
    total_phi = _compute_phi_total_per_layer(model, total_flux_exc)

    final_flux = total_phi[1]
    assert torch.allclose(final_flux, torch.tensor(phi_target), atol=1e-2)
