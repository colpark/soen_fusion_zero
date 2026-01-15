"""Unit tests for JAX autoregressive utilities."""

import jax.numpy as jnp
import pytest

from soen_toolkit.utils.port_to_jax.jax_training.ar_utils import (
    build_ar_targets_jax,
    pool_token_timesteps_jax,
    validate_ar_config_jax,
)


class TestPoolTokenTimestepsJax:
    """Tests for pool_token_timesteps_jax."""

    def test_basic_pooling_final(self):
        """Test 'final' pooling method."""
        # Batch=1, Tokens=2, Steps=3 -> Total=6
        # Token 0: [1, 2, 3] -> 3
        # Token 1: [4, 5, 6] -> 6
        outputs = jnp.array([[[1], [2], [3], [4], [5], [6]]], dtype=jnp.float32)
        pooled = pool_token_timesteps_jax(outputs, time_steps_per_token=3, pooling_method="final")

        assert pooled.shape == (1, 2, 1)
        expected = jnp.array([[[3], [6]]], dtype=jnp.float32)
        assert jnp.allclose(pooled, expected)

    def test_basic_pooling_mean(self):
        """Test 'mean' pooling method."""
        # Token 0: [1, 2, 3] -> 2
        # Token 1: [4, 5, 6] -> 5
        outputs = jnp.array([[[1], [2], [3], [4], [5], [6]]], dtype=jnp.float32)
        pooled = pool_token_timesteps_jax(outputs, time_steps_per_token=3, pooling_method="mean")

        expected = jnp.array([[[2], [5]]], dtype=jnp.float32)
        assert jnp.allclose(pooled, expected)

    def test_basic_pooling_max(self):
        """Test 'max' pooling method."""
        # Token 0: [1, 3, 2] -> 3
        outputs = jnp.array([[[1], [3], [2]]], dtype=jnp.float32)
        pooled = pool_token_timesteps_jax(outputs, time_steps_per_token=3, pooling_method="max")

        expected = jnp.array([[[3]]], dtype=jnp.float32)
        assert jnp.allclose(pooled, expected)

    def test_mean_last_n(self):
        """Test 'mean_last_n' pooling method."""
        # Token 0: [1, 2, 3, 4] -> mean(3, 4) = 3.5
        outputs = jnp.array([[[1], [2], [3], [4]]], dtype=jnp.float32)
        pooled = pool_token_timesteps_jax(
            outputs,
            time_steps_per_token=4,
            pooling_method="mean_last_n",
            pooling_params={"n": 2}
        )

        expected = jnp.array([[[3.5]]], dtype=jnp.float32)
        assert jnp.allclose(pooled, expected)

    def test_no_pooling(self):
        """Test time_steps_per_token=1 (no pooling)."""
        outputs = jnp.ones((2, 10, 5))
        pooled = pool_token_timesteps_jax(outputs, time_steps_per_token=1)
        assert jnp.array_equal(pooled, outputs)

    def test_truncation(self):
        """Test truncation when total steps is not a multiple of steps_per_token."""
        # 7 steps, 3 per token -> 2 tokens (6 steps), drop last 1
        outputs = jnp.zeros((1, 7, 1))
        pooled = pool_token_timesteps_jax(outputs, time_steps_per_token=3)
        assert pooled.shape == (1, 2, 1)

    def test_invalid_config(self):
        """Test invalid configurations."""
        outputs = jnp.zeros((1, 6, 1))

        with pytest.raises(ValueError, match="positive"):
            pool_token_timesteps_jax(outputs, time_steps_per_token=0)

        with pytest.raises(ValueError, match="Unknown pooling method"):
            pool_token_timesteps_jax(outputs, 3, pooling_method="invalid")

        with pytest.raises(ValueError, match="Not enough timesteps"):
            pool_token_timesteps_jax(outputs[:, :2, :], 3)  # 2 steps < 3 per token


class TestBuildARTargetsJax:
    """Tests for build_ar_targets_jax."""

    def test_basic_shift(self):
        """Test basic target shifting."""
        # Input: [0, 1, 2, 3]
        # Target: [1, 2, 3, 3] (last repeated)
        inputs = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        targets = build_ar_targets_jax(inputs)

        expected = jnp.array([[1, 2, 3, 3]], dtype=jnp.int32)
        assert jnp.array_equal(targets, expected)

    def test_batch_processing(self):
        """Test with batch size > 1."""
        inputs = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        targets = build_ar_targets_jax(inputs)

        expected = jnp.array([[1, 1], [3, 3]], dtype=jnp.int32)
        assert jnp.array_equal(targets, expected)


class TestValidateARConfigJax:
    """Tests for validate_ar_config_jax."""

    def test_valid_config(self):
        validate_ar_config_jax(4, "final")
        validate_ar_config_jax(4, "mean_last_n", {"n": 2})

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            validate_ar_config_jax(0, "final")
        with pytest.raises(ValueError):
            validate_ar_config_jax(4, "invalid")
