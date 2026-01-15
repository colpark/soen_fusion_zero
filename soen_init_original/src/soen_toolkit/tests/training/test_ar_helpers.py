"""Tests for autoregressive training utilities."""

import numpy as np
import pytest
import torch

from soen_toolkit.training.utils.ar_helpers import (
    build_multistep_ar_targets,
    get_ar_sequence_length,
    pool_token_timesteps,
    prepare_multistep_ar_dataset,
    validate_ar_config,
)


class TestPoolTokenTimesteps:
    """Tests for pool_token_timesteps function."""

    def test_final_pooling(self):
        """Test final timestep pooling."""
        # Create outputs: 2 batches, 64 tokens × 4 timesteps, 26 vocab
        outputs = torch.randn(2, 256, 26)
        pooled = pool_token_timesteps(outputs, 4, "final")

        assert pooled.shape == (2, 64, 26)

        # Verify it's actually the last timestep of each token
        # Token 0: timesteps 0-3, should use timestep 3
        # Token 1: timesteps 4-7, should use timestep 7
        assert torch.allclose(pooled[0, 0, :], outputs[0, 3, :])
        assert torch.allclose(pooled[0, 1, :], outputs[0, 7, :])

    def test_mean_pooling(self):
        """Test mean pooling over timesteps."""
        outputs = torch.randn(2, 256, 26)
        pooled = pool_token_timesteps(outputs, 4, "mean")

        assert pooled.shape == (2, 64, 26)

        # Verify it's the mean of timesteps 0-3 for token 0
        expected_mean = outputs[0, 0:4, :].mean(dim=0)
        assert torch.allclose(pooled[0, 0, :], expected_mean)

    def test_max_pooling(self):
        """Test max pooling over timesteps."""
        outputs = torch.randn(2, 256, 26)
        pooled = pool_token_timesteps(outputs, 4, "max")

        assert pooled.shape == (2, 64, 26)

        # Verify it's the max of timesteps 0-3 for token 0
        expected_max = outputs[0, 0:4, :].max(dim=0)[0]
        assert torch.allclose(pooled[0, 0, :], expected_max)

    def test_mean_last_n_pooling(self):
        """Test mean of last N timesteps pooling."""
        outputs = torch.randn(2, 256, 26)
        pooled = pool_token_timesteps(outputs, 4, "mean_last_n", {"n": 2})

        assert pooled.shape == (2, 64, 26)

        # Verify it's the mean of last 2 timesteps (2-3) for token 0
        expected_mean = outputs[0, 2:4, :].mean(dim=0)
        assert torch.allclose(pooled[0, 0, :], expected_mean)

    def test_no_pooling_when_one_step_per_token(self):
        """Test that no pooling occurs when time_steps_per_token=1."""
        outputs = torch.randn(2, 64, 26)
        pooled = pool_token_timesteps(outputs, 1, "final")

        assert torch.allclose(pooled, outputs)

    def test_truncation_warning(self, caplog):
        """Test that incomplete tokens are truncated with warning."""
        # 258 timesteps with 4 per token = 64 complete tokens + 2 extra
        outputs = torch.randn(2, 258, 26)

        import logging
        with caplog.at_level(logging.WARNING):
            pooled = pool_token_timesteps(outputs, 4, "final")

        # Should have 64 tokens (256 timesteps used, 2 discarded)
        assert pooled.shape == (2, 64, 26)

        # Check that warning was logged
        assert "Truncating outputs" in caplog.text

    def test_invalid_pooling_method(self):
        """Test error on invalid pooling method."""
        outputs = torch.randn(2, 256, 26)

        with pytest.raises(ValueError, match="Unknown pooling method"):
            pool_token_timesteps(outputs, 4, "invalid_method")

    def test_invalid_time_steps_per_token(self):
        """Test error on invalid time_steps_per_token."""
        outputs = torch.randn(2, 256, 26)

        with pytest.raises(ValueError, match="must be positive"):
            pool_token_timesteps(outputs, 0, "final")

        with pytest.raises(ValueError, match="must be positive"):
            pool_token_timesteps(outputs, -1, "final")

    def test_mean_last_n_validation(self):
        """Test validation of mean_last_n parameters."""
        outputs = torch.randn(2, 256, 26)

        # n exceeds time_steps_per_token
        with pytest.raises(ValueError, match="exceeds time_steps_per_token"):
            pool_token_timesteps(outputs, 4, "mean_last_n", {"n": 5})

        # Invalid n
        with pytest.raises(ValueError, match="positive integer"):
            pool_token_timesteps(outputs, 4, "mean_last_n", {"n": 0})

        with pytest.raises(ValueError, match="positive integer"):
            pool_token_timesteps(outputs, 4, "mean_last_n", {"n": -1})

    def test_not_enough_timesteps(self):
        """Test error when not enough timesteps for even one token."""
        outputs = torch.randn(2, 3, 26)  # Only 3 timesteps

        with pytest.raises(ValueError, match="Not enough timesteps"):
            pool_token_timesteps(outputs, 4, "final")


class TestPrepareMultistepARDataset:
    """Tests for prepare_multistep_ar_dataset function."""

    def test_one_hot_duplication(self):
        """Test duplication of one-hot encoded sequences."""
        # Create simple one-hot sequence: "hi" = [7, 8]
        seq = np.eye(26)[np.array([[7, 8]])]  # [1, 2, 26]

        extended = prepare_multistep_ar_dataset(seq, 4)

        # Should be [1, 8, 26] - each token duplicated 4 times
        assert extended.shape == (1, 8, 26)

        # Verify duplication
        assert np.array_equal(extended[0, 0:4, :], seq[0, 0:1, :].repeat(4, axis=0))
        assert np.array_equal(extended[0, 4:8, :], seq[0, 1:2, :].repeat(4, axis=0))

    def test_token_indices_duplication(self):
        """Test duplication of token index sequences."""
        seq = np.array([[7, 8, 11]])  # [1, 3]

        extended = prepare_multistep_ar_dataset(seq, 4)

        # Should be [1, 12]
        assert extended.shape == (1, 12)

        # Verify: [7,7,7,7,8,8,8,8,11,11,11,11]
        expected = np.array([[7, 7, 7, 7, 8, 8, 8, 8, 11, 11, 11, 11]])
        assert np.array_equal(extended, expected)

    def test_no_duplication_when_one_step(self):
        """Test that no duplication occurs when time_steps_per_token=1."""
        seq = np.array([[7, 8, 11]])
        extended = prepare_multistep_ar_dataset(seq, 1)

        assert np.array_equal(extended, seq)

    def test_batch_processing(self):
        """Test processing multiple sequences in batch."""
        seq = np.array([[7, 8], [11, 14]])  # [2, 2]

        extended = prepare_multistep_ar_dataset(seq, 3)

        assert extended.shape == (2, 6)
        assert np.array_equal(extended[0], [7, 7, 7, 8, 8, 8])
        assert np.array_equal(extended[1], [11, 11, 11, 14, 14, 14])

    def test_invalid_time_steps_per_token(self):
        """Test error on invalid time_steps_per_token."""
        seq = np.array([[7, 8]])

        with pytest.raises(ValueError, match="must be positive"):
            prepare_multistep_ar_dataset(seq, 0)

        with pytest.raises(ValueError, match="must be positive"):
            prepare_multistep_ar_dataset(seq, -1)

    def test_invalid_dimensions(self):
        """Test error on invalid sequence dimensions."""
        seq = np.array([7, 8])  # 1D

        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            prepare_multistep_ar_dataset(seq, 4)


class TestBuildMultistepARTargets:
    """Tests for build_multistep_ar_targets function."""

    def test_standard_target_construction(self):
        """Test that targets are shifted by 1 token."""
        tokens = torch.tensor([[7, 8, 11, 11, 14]])  # "hello"
        targets = build_multistep_ar_targets(tokens)

        # Should shift by 1: [8, 11, 11, 14, 14]
        expected = torch.tensor([[8, 11, 11, 14, 14]])
        assert torch.equal(targets, expected)

    def test_batch_processing(self):
        """Test processing multiple sequences."""
        tokens = torch.tensor([
            [7, 8, 11],
            [14, 0, 19]
        ])
        targets = build_multistep_ar_targets(tokens)

        expected = torch.tensor([
            [8, 11, 11],  # Shifted, last token repeated
            [0, 19, 19]
        ])
        assert torch.equal(targets, expected)

    def test_single_token_sequence(self):
        """Test edge case with single token."""
        tokens = torch.tensor([[7]])
        targets = build_multistep_ar_targets(tokens)

        # Should just repeat the token
        expected = torch.tensor([[7]])
        assert torch.equal(targets, expected)

    def test_time_steps_parameter_ignored(self):
        """Test that time_steps_per_token parameter doesn't affect output."""
        tokens = torch.tensor([[7, 8, 11]])

        targets1 = build_multistep_ar_targets(tokens, 1)
        targets4 = build_multistep_ar_targets(tokens, 4)

        # Should be identical
        assert torch.equal(targets1, targets4)


class TestValidateARConfig:
    """Tests for validate_ar_config function."""

    def test_valid_configs(self):
        """Test that valid configurations pass."""
        # Should not raise
        validate_ar_config(1, "final")
        validate_ar_config(4, "mean")
        validate_ar_config(4, "max")
        validate_ar_config(4, "mean_last_n", {"n": 2})

    def test_invalid_time_steps_per_token(self):
        """Test error on invalid time_steps_per_token."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_ar_config(0, "final")

        with pytest.raises(ValueError, match="must be positive"):
            validate_ar_config(-1, "final")

    def test_invalid_pooling_method(self):
        """Test error on invalid pooling method."""
        with pytest.raises(ValueError, match="Invalid token_pooling_method"):
            validate_ar_config(4, "invalid")

    def test_mean_last_n_missing_param(self):
        """Test error when mean_last_n missing 'n' parameter."""
        with pytest.raises(ValueError, match="requires 'n' parameter"):
            validate_ar_config(4, "mean_last_n", {})

    def test_mean_last_n_invalid_n(self):
        """Test error on invalid 'n' for mean_last_n."""
        with pytest.raises(ValueError, match="positive integer"):
            validate_ar_config(4, "mean_last_n", {"n": 0})

        with pytest.raises(ValueError, match="positive integer"):
            validate_ar_config(4, "mean_last_n", {"n": -1})

        with pytest.raises(ValueError, match="cannot exceed"):
            validate_ar_config(4, "mean_last_n", {"n": 5})


class TestGetARSequenceLength:
    """Tests for get_ar_sequence_length function."""

    def test_exact_division(self):
        """Test when sequence length divides evenly."""
        assert get_ar_sequence_length(256, 4) == 64
        assert get_ar_sequence_length(100, 10) == 10
        assert get_ar_sequence_length(64, 1) == 64

    def test_truncation(self):
        """Test that incomplete tokens are truncated."""
        # 258 / 4 = 64.5 -> 64 complete tokens
        assert get_ar_sequence_length(258, 4) == 64
        assert get_ar_sequence_length(99, 10) == 9

    def test_invalid_time_steps_per_token(self):
        """Test error on invalid time_steps_per_token."""
        with pytest.raises(ValueError, match="must be positive"):
            get_ar_sequence_length(256, 0)

        with pytest.raises(ValueError, match="must be positive"):
            get_ar_sequence_length(256, -1)
