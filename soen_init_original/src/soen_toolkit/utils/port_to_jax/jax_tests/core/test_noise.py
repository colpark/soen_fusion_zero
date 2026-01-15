"""Tests for JAX noise and perturbation functionality."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from soen_toolkit.core.model_yaml import load_model_from_yaml
from soen_toolkit.utils.port_to_jax.convert import convert_core_model_to_jax
from soen_toolkit.utils.port_to_jax.noise_jax import (
    GaussianNoiseConfig,
    NoiseConfig,
    PerturbationConfig,
    apply_connection_noise,
    apply_noise_step,
    build_noise_settings,
    precompute_perturbations,
)
from soen_toolkit.utils.port_to_jax.unified_forward import forward

# Path to test models
TEST_MODELS_DIR = "src/soen_toolkit/tests/utils/parity_test_models"


class TestNoiseBuilding:
    """Test noise configuration building."""

    def test_build_empty_settings(self):
        """Empty config returns trivial settings."""
        settings = build_noise_settings(None)
        assert settings.is_trivial()

        settings = build_noise_settings({})
        assert settings.is_trivial()

    def test_build_phi_noise(self):
        """Phi noise is correctly parsed."""
        settings = build_noise_settings({"phi": 0.01})
        assert not settings.is_trivial()
        assert settings.phi is not None
        assert settings.phi.noise is not None
        assert settings.phi.noise.std == 0.01

    def test_build_state_noise(self):
        """State noise is correctly parsed."""
        settings = build_noise_settings({"s": 0.005})
        assert not settings.is_trivial()
        assert settings.s is not None
        assert settings.s.noise is not None
        assert settings.s.noise.std == 0.005

    def test_build_relative_noise(self):
        """Relative noise flag is applied."""
        settings = build_noise_settings({"phi": 0.01, "relative": True})
        assert settings.phi.noise.relative is True

    def test_build_perturbation(self):
        """Perturbation config is correctly parsed."""
        settings = build_noise_settings({
            "phi_perturb_std": 0.05,
            "phi_perturb_mean": 0.1,
        })
        assert settings.phi is not None
        assert settings.phi.perturbation is not None
        assert settings.phi.perturbation.std == 0.05
        assert settings.phi.perturbation.mean == 0.1

    def test_build_combined_noise_and_perturbation(self):
        """Combined noise and perturbation are correctly parsed."""
        settings = build_noise_settings({
            "phi": 0.01,
            "phi_perturb_std": 0.05,
        })
        assert settings.phi.noise is not None
        assert settings.phi.noise.std == 0.01
        assert settings.phi.perturbation is not None
        assert settings.phi.perturbation.std == 0.05


class TestPerturbationPrecompute:
    """Test perturbation offset precomputation."""

    def test_no_perturbation_returns_none(self):
        """When no perturbation is configured, offsets are None."""
        settings = build_noise_settings({"phi": 0.01})  # Noise only, no perturbation
        key = jax.random.PRNGKey(42)
        offsets = precompute_perturbations(
            key, settings, batch_size=4, layer_dims={0: 10}
        )
        assert offsets.phi is None

    def test_phi_perturbation_shape(self):
        """Phi perturbation has correct shape."""
        settings = build_noise_settings({"phi_perturb_std": 0.05})
        key = jax.random.PRNGKey(42)
        offsets = precompute_perturbations(
            key, settings, batch_size=4, layer_dims={0: 10, 1: 5}
        )
        assert offsets.phi is not None
        assert offsets.phi.shape == (4, 10)  # [batch, first_layer_dim]

    def test_state_perturbation_per_layer(self):
        """State perturbation is computed per layer."""
        settings = build_noise_settings({"s_perturb_std": 0.02})
        key = jax.random.PRNGKey(42)
        offsets = precompute_perturbations(
            key, settings, batch_size=4, layer_dims={0: 10, 1: 5}
        )
        assert offsets.layer_params is not None
        assert 0 in offsets.layer_params
        assert "s" in offsets.layer_params[0]
        assert offsets.layer_params[0]["s"].shape == (4, 10)

    def test_deterministic_with_same_key(self):
        """Same key produces same offsets."""
        settings = build_noise_settings({"phi_perturb_std": 0.05})
        key = jax.random.PRNGKey(42)
        offsets1 = precompute_perturbations(
            key, settings, batch_size=4, layer_dims={0: 10}
        )
        offsets2 = precompute_perturbations(
            key, settings, batch_size=4, layer_dims={0: 10}
        )
        assert jnp.allclose(offsets1.phi, offsets2.phi)


class TestNoiseApplication:
    """Test noise application functions."""

    def test_apply_noise_no_config(self):
        """No config returns tensor unchanged."""
        key = jax.random.PRNGKey(42)
        tensor = jnp.ones((2, 10))
        result = apply_noise_step(key, tensor, None)
        assert jnp.allclose(result, tensor)

    def test_apply_noise_adds_gaussian(self):
        """Gaussian noise is added."""
        key = jax.random.PRNGKey(42)
        tensor = jnp.ones((2, 10))
        config = NoiseConfig(noise=GaussianNoiseConfig(std=0.1))
        result = apply_noise_step(key, tensor, config)
        assert not jnp.allclose(result, tensor)
        # Noise should be roughly within expected range
        diff = jnp.abs(result - tensor)
        assert diff.max() < 1.0  # 10 sigma

    def test_apply_noise_relative(self):
        """Relative noise scales with tensor magnitude."""
        key = jax.random.PRNGKey(42)
        tensor = jnp.array([[1.0, 10.0, 100.0]])
        config = NoiseConfig(noise=GaussianNoiseConfig(std=0.1, relative=True))
        result = apply_noise_step(key, tensor, config)
        diff = jnp.abs(result - tensor)
        # Larger values should have larger noise (on average)
        # Check that noise scales roughly with magnitude
        assert diff[0, 2] > diff[0, 0]  # 100x value should have larger noise

    def test_apply_perturbation_offset(self):
        """Perturbation offset is added."""
        key = jax.random.PRNGKey(42)
        tensor = jnp.zeros((2, 10))
        offset = jnp.ones((2, 10)) * 0.5
        config = NoiseConfig(perturbation=PerturbationConfig(std=0.0))
        result = apply_noise_step(key, tensor, config, offset)
        assert jnp.allclose(result, offset)

    def test_apply_connection_noise(self):
        """Connection noise is applied to weight matrices."""
        key = jax.random.PRNGKey(42)
        J = jnp.ones((5, 10))
        config = NoiseConfig(noise=GaussianNoiseConfig(std=0.01))
        result = apply_connection_noise(key, J, config)
        assert not jnp.allclose(result, J)
        assert result.shape == J.shape


class TestForwardWithNoise:
    """Test noise injection in forward pass."""

    @pytest.fixture
    def jax_model(self):
        """Load a test model."""
        model_path = f"{TEST_MODELS_DIR}/test_singledendrite_layerwise.yaml"
        torch_model = load_model_from_yaml(model_path)
        return convert_core_model_to_jax(torch_model)

    def test_forward_without_noise(self, jax_model):
        """Forward pass works without noise."""
        B, T = 2, 10
        dim = jax_model.layers[0].dim
        x = jnp.zeros((B, T, dim))

        out, _ = forward(jax_model, x)
        assert out.shape == (B, T + 1, jax_model.layers[-1].dim)

    def test_forward_with_noise_changes_output(self, jax_model):
        """Forward pass with noise produces different output."""
        key = jax.random.PRNGKey(42)
        B, T = 2, 10
        dim = jax_model.layers[0].dim
        x = jnp.zeros((B, T, dim))

        noise = build_noise_settings({"phi": 0.01, "s": 0.005})
        out_no_noise, _ = forward(jax_model, x)
        out_with_noise, _ = forward(jax_model, x, noise_settings=noise, rng_key=key)

        # Should be different due to noise
        assert not jnp.allclose(out_no_noise, out_with_noise)

    def test_forward_same_key_same_result(self, jax_model):
        """Same RNG key produces same result."""
        key = jax.random.PRNGKey(42)
        B, T = 2, 10
        dim = jax_model.layers[0].dim
        x = jnp.zeros((B, T, dim))
        noise = build_noise_settings({"phi": 0.01})

        out1, _ = forward(jax_model, x, noise_settings=noise, rng_key=key)
        out2, _ = forward(jax_model, x, noise_settings=noise, rng_key=key)
        assert jnp.allclose(out1, out2)

    def test_forward_different_key_different_result(self, jax_model):
        """Different RNG key produces different result."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(123)
        B, T = 2, 10
        dim = jax_model.layers[0].dim
        x = jnp.zeros((B, T, dim))
        noise = build_noise_settings({"phi": 0.01})

        out1, _ = forward(jax_model, x, noise_settings=noise, rng_key=key1)
        out2, _ = forward(jax_model, x, noise_settings=noise, rng_key=key2)
        assert not jnp.allclose(out1, out2)

    def test_forward_dict_noise_config(self, jax_model):
        """Forward accepts dict noise config."""
        key = jax.random.PRNGKey(42)
        B, T = 2, 10
        dim = jax_model.layers[0].dim
        x = jnp.zeros((B, T, dim))

        # Pass dict directly instead of NoiseSettings
        out, _ = forward(
            jax_model, x,
            noise_settings={"phi": 0.01, "s": 0.005},
            rng_key=key
        )
        assert out.shape == (B, T + 1, jax_model.layers[-1].dim)

    def test_forward_rejects_connection_noise_dict(self, jax_model):
        """Forward fails fast if connection noise dict is passed as noise_settings."""
        key = jax.random.PRNGKey(42)
        B, T = 2, 10
        dim = jax_model.layers[0].dim
        x = jnp.zeros((B, T, dim))

        bad = {"J_0_to_1": NoiseConfig(noise=GaussianNoiseConfig(std=0.01))}
        with pytest.raises(ValueError, match="connection_noise_settings"):
            forward(jax_model, x, noise_settings=bad, rng_key=key)

    def test_forward_requires_key_for_noise(self, jax_model):
        """Forward raises error if noise configured but no key provided."""
        B, T = 2, 10
        dim = jax_model.layers[0].dim
        x = jnp.zeros((B, T, dim))
        noise = build_noise_settings({"phi": 0.01})

        with pytest.raises(ValueError, match="rng_key must be provided"):
            forward(jax_model, x, noise_settings=noise, rng_key=None)


class TestStepwiseWithNoise:
    """Test noise injection in stepwise solvers."""

    @pytest.fixture
    def jax_model_jacobi(self):
        """Load a Jacobi solver test model."""
        model_path = f"{TEST_MODELS_DIR}/test_singledendrite_jacobi.yaml"
        torch_model = load_model_from_yaml(model_path)
        return convert_core_model_to_jax(torch_model)

    @pytest.fixture
    def jax_model_gs(self):
        """Load a Gauss-Seidel solver test model."""
        model_path = f"{TEST_MODELS_DIR}/test_singledendrite_gauss_seidel.yaml"
        torch_model = load_model_from_yaml(model_path)
        return convert_core_model_to_jax(torch_model)

    def test_jacobi_with_noise(self, jax_model_jacobi):
        """Jacobi solver works with noise."""
        key = jax.random.PRNGKey(42)
        B, T = 2, 10
        dim = jax_model_jacobi.layers[0].dim
        x = jnp.zeros((B, T, dim))
        noise = build_noise_settings({"phi": 0.01, "s": 0.005})

        out_no_noise, _ = forward(jax_model_jacobi, x)
        out_with_noise, _ = forward(
            jax_model_jacobi, x, noise_settings=noise, rng_key=key
        )
        assert not jnp.allclose(out_no_noise, out_with_noise)

    def test_gauss_seidel_with_noise(self, jax_model_gs):
        """Gauss-Seidel solver works with noise."""
        key = jax.random.PRNGKey(42)
        B, T = 2, 10
        dim = jax_model_gs.layers[0].dim
        x = jnp.zeros((B, T, dim))
        noise = build_noise_settings({"phi": 0.01, "s": 0.005})

        out_no_noise, _ = forward(jax_model_gs, x)
        out_with_noise, _ = forward(jax_model_gs, x, noise_settings=noise, rng_key=key)
        assert not jnp.allclose(out_no_noise, out_with_noise)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

