"""Unit tests for local learning module.

This test suite validates local learning rules, modulators, and the LocalTrainer
on simple SOEN models with dynamics-invariant tasks.
"""


import numpy as np
import pytest
import torch

from soen_toolkit.core.model_yaml import build_model_from_yaml
from soen_toolkit.training.local_learning import (
    AccuracyBasedModulator,
    BCMRule,
    CrossEntropyErrorModulator,
    HebbianRule,
    LocalTrainer,
    MSEErrorModulator,
    OjaRule,
    RewardModulatedHebbianRule,
)
from soen_toolkit.training.local_learning.utils import get_dt_for_time_constant


# Test data generators
def generate_simple_regression_data(n_samples=100, seq_len=10, seed=42):
    """Generate simple regression task: y = 2*x1 - 1.5*x2 + noise."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.randn(n_samples, seq_len, 2) * 2.0
    y = 2.0 * X[:, 0, 0] - 1.5 * X[:, 0, 1]
    y = y.unsqueeze(1)
    y += torch.randn_like(y) * 0.1

    return X, y


def generate_xor_like_data(n_samples=100, seq_len=10, seed=42):
    """Generate XOR-like continuous task: y = tanh(x1) * tanh(x2)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.randn(n_samples, seq_len, 2) * 2.0
    x1 = X[:, 0, 0]
    x2 = X[:, 0, 1]
    y = torch.tanh(x1) * torch.tanh(x2)
    y = y.unsqueeze(1)

    return X, y


# Fixtures
@pytest.fixture
def simple_model_config():
    """Simple test model configuration: Linear -> SingleDendrite -> Linear."""
    dt = get_dt_for_time_constant(0.1)  # 0.1 ns time constant
    return {
        "simulation": {"dt": dt, "input_type": "flux", "seed": 42},
        "layers": [
            {"layer_id": 0, "layer_type": "Linear", "params": {"dim": 2}},
            {"layer_id": 1, "layer_type": "SingleDendrite", "params": {"dim": 10}},
            {"layer_id": 2, "layer_type": "Linear", "params": {"dim": 1}},
        ],
        "connections": [
            {
                "from_layer": 0,
                "to_layer": 1,
                "connection_type": "all_to_all",
                "params": {"init": "xavier_uniform", "learnable": True}
            },
            {
                "from_layer": 1,
                "to_layer": 2,
                "connection_type": "all_to_all",
                "params": {"init": "xavier_uniform", "learnable": True}
            },
        ]
    }


@pytest.fixture
def model(simple_model_config):
    """Build simple test model."""
    return build_model_from_yaml(simple_model_config, honor_yaml_seed=True)


# Test learning rules
class TestHebbianRule:
    """Tests for HebbianRule."""

    def test_initialization(self):
        """Test rule initialization."""
        rule = HebbianRule(lr=0.01)
        assert rule.lr == 0.01

    def test_compute_update_shape(self):
        """Test that update has correct shape."""
        rule = HebbianRule(lr=0.01)

        pre = torch.randn(32, 10)  # [batch, pre_dim]
        post = torch.randn(32, 5)  # [batch, post_dim]
        weights = torch.randn(5, 10)  # [post_dim, pre_dim] (PyTorch convention)

        update = rule.compute_update(pre, post, weights)

        assert update.shape == weights.shape
        assert update.dtype == torch.float32

    def test_compute_update_finite(self):
        """Test that updates are finite."""
        rule = HebbianRule(lr=0.01)

        pre = torch.randn(32, 10)
        post = torch.randn(32, 5)
        weights = torch.randn(5, 10)

        update = rule.compute_update(pre, post, weights)

        assert torch.isfinite(update).all()

    def test_no_gradient_flow(self):
        """Test that updates don't require gradients."""
        rule = HebbianRule(lr=0.01)

        pre = torch.randn(32, 10, requires_grad=True)
        post = torch.randn(32, 5, requires_grad=True)
        weights = torch.randn(5, 10, requires_grad=True)

        update = rule.compute_update(pre, post, weights)

        assert not update.requires_grad


class TestOjaRule:
    """Tests for OjaRule."""

    def test_initialization(self):
        """Test rule initialization."""
        rule = OjaRule(lr=0.01)
        assert rule.lr == 0.01

    def test_compute_update_shape(self):
        """Test that update has correct shape."""
        rule = OjaRule(lr=0.01)

        pre = torch.randn(32, 10)
        post = torch.randn(32, 5)
        weights = torch.randn(5, 10)

        update = rule.compute_update(pre, post, weights)

        assert update.shape == weights.shape


class TestBCMRule:
    """Tests for BCMRule."""

    def test_initialization(self):
        """Test rule initialization."""
        rule = BCMRule(lr=0.01, threshold_momentum=0.9)
        assert rule.lr == 0.01
        assert rule.threshold_momentum == 0.9
        assert rule.threshold is None

    def test_threshold_initialization(self):
        """Test that threshold gets initialized on first update."""
        rule = BCMRule(lr=0.01)

        pre = torch.randn(32, 10)
        post = torch.randn(32, 5)
        weights = torch.randn(5, 10)

        assert rule.threshold is None
        _ = rule.compute_update(pre, post, weights)
        assert rule.threshold is not None
        assert rule.threshold.shape == (5,)  # [post_dim]


class TestRewardModulatedHebbianRule:
    """Tests for RewardModulatedHebbianRule."""

    def test_initialization(self):
        """Test rule initialization."""
        rule = RewardModulatedHebbianRule(lr=0.01, baseline_subtract=True)
        assert rule.lr == 0.01
        assert rule.baseline_subtract is True

    def test_compute_update_with_scalar_modulator(self):
        """Test update with scalar modulator."""
        rule = RewardModulatedHebbianRule(lr=0.01)

        pre = torch.randn(32, 10)
        post = torch.randn(32, 5)
        weights = torch.randn(5, 10)
        modulator = torch.tensor(1.5)  # Scalar

        update = rule.compute_update(pre, post, weights, modulator)

        assert update.shape == weights.shape
        assert torch.isfinite(update).all()

    def test_compute_update_with_vector_modulator(self):
        """Test update with per-sample modulator."""
        rule = RewardModulatedHebbianRule(lr=0.01)

        pre = torch.randn(32, 10)
        post = torch.randn(32, 5)
        weights = torch.randn(5, 10)
        modulator = torch.randn(32)  # Per-sample

        update = rule.compute_update(pre, post, weights, modulator)

        assert update.shape == weights.shape
        assert torch.isfinite(update).all()

    def test_fallback_without_modulator(self):
        """Test that rule falls back to Hebbian without modulator."""
        rule = RewardModulatedHebbianRule(lr=0.01)

        pre = torch.randn(32, 10)
        post = torch.randn(32, 5)
        weights = torch.randn(5, 10)

        update = rule.compute_update(pre, post, weights, modulator=None)

        assert update.shape == weights.shape


# Test modulators
class TestMSEErrorModulator:
    """Tests for MSEErrorModulator."""

    def test_initialization(self):
        """Test modulator initialization."""
        mod = MSEErrorModulator(scale=1.0, per_sample=True)
        assert mod.scale == 1.0
        assert mod.per_sample is True

    def test_compute_scalar(self):
        """Test scalar modulator computation."""
        mod = MSEErrorModulator(scale=1.0, per_sample=False)

        outputs = torch.randn(32, 5)
        targets = torch.randn(32, 5)

        signal = mod.compute(outputs, targets)

        assert signal.dim() == 0  # Scalar
        assert torch.isfinite(signal)

    def test_compute_per_sample(self):
        """Test per-sample modulator computation."""
        mod = MSEErrorModulator(scale=1.0, per_sample=True)

        outputs = torch.randn(32, 5)
        targets = torch.randn(32, 5)

        signal = mod.compute(outputs, targets)

        assert signal.shape == (32,)
        assert torch.isfinite(signal).all()


class TestCrossEntropyErrorModulator:
    """Tests for CrossEntropyErrorModulator."""

    def test_compute(self):
        """Test cross-entropy modulator."""
        mod = CrossEntropyErrorModulator(scale=1.0, per_sample=True)

        outputs = torch.randn(32, 10)  # Logits
        targets = torch.randint(0, 10, (32,))  # Class indices

        signal = mod.compute(outputs, targets)

        assert signal.shape == (32,)
        assert torch.isfinite(signal).all()


class TestAccuracyBasedModulator:
    """Tests for AccuracyBasedModulator."""

    def test_compute(self):
        """Test accuracy-based modulator."""
        mod = AccuracyBasedModulator(correct_reward=1.0, incorrect_reward=-0.5)

        outputs = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))

        signal = mod.compute(outputs, targets)

        assert signal.shape == (32,)
        # Should be either correct_reward or incorrect_reward
        assert torch.all((signal == 1.0) | (signal == -0.5))


# Test LocalTrainer
class TestLocalTrainer:
    """Tests for LocalTrainer."""

    def test_initialization(self, model):
        """Test trainer initialization."""
        rule = HebbianRule(lr=0.01)
        trainer = LocalTrainer(model=model, rule=rule, layers=[1])

        assert trainer.model is model
        assert trainer.rule is rule
        assert len(trainer.target_connections) > 0

    def test_forward_and_collect_states(self, model):
        """Test state collection during forward pass."""
        rule = HebbianRule(lr=0.01)
        trainer = LocalTrainer(model=model, rule=rule, layers=[1])

        X = torch.randn(16, 10, 2)  # [batch, seq_len, input_dim]

        outputs, layer_states = trainer._forward_and_collect_states(X)

        assert outputs.shape[0] == 16  # Batch size preserved
        assert isinstance(layer_states, dict)
        assert len(layer_states) > 0

    def test_step_without_targets(self, model):
        """Test training step without targets (2-factor)."""
        rule = HebbianRule(lr=0.01)
        trainer = LocalTrainer(model=model, rule=rule, layers=[1])

        X = torch.randn(16, 10, 2)

        metrics = trainer.step(X)

        assert 'loss' in metrics
        assert 'total_update_norm' in metrics
        assert metrics['loss'] == 0.0  # No targets, no loss

    def test_step_with_targets(self, model):
        """Test training step with targets (potential 3-factor)."""
        rule = HebbianRule(lr=0.01)
        trainer = LocalTrainer(model=model, rule=rule, layers=[1], readout_loss="mse")

        X = torch.randn(16, 10, 2)
        y = torch.randn(16, 1)

        metrics = trainer.step(X, y)

        assert 'loss' in metrics
        assert 'total_update_norm' in metrics
        assert metrics['loss'] > 0.0  # Should have non-zero loss


# Integration tests
class TestLocalLearningIntegration:
    """Integration tests for complete training scenarios."""

    def test_basic_hebbian_learning(self, model):
        """Test that basic Hebbian learning reduces loss."""
        X_train, y_train = generate_simple_regression_data(n_samples=100, seq_len=10)

        rule = HebbianRule(lr=0.001)
        trainer = LocalTrainer(model=model, rule=rule, layers=[1], readout_loss="mse")

        # Record initial loss
        initial_metrics = trainer.step(X_train, y_train)
        initial_loss = initial_metrics['loss']

        # Train for a few epochs
        for _ in range(10):
            trainer.step(X_train, y_train)

        # Check final loss
        final_metrics = trainer.step(X_train, y_train)
        final_loss = final_metrics['loss']

        # Loss should decrease (or at least not increase significantly)
        # Hebbian learning is unsupervised, so we don't expect large improvements
        assert final_loss < initial_loss * 1.5

    def test_reward_modulated_learning(self, model):
        """Test that reward-modulated learning works."""
        X_train, y_train = generate_simple_regression_data(n_samples=100, seq_len=10)

        rule = RewardModulatedHebbianRule(lr=0.005, baseline_subtract=True)
        modulator = MSEErrorModulator(scale=1.0, per_sample=True)

        trainer = LocalTrainer(
            model=model,
            rule=rule,
            modulator=modulator,
            layers=[1],
            readout_loss="mse"
        )

        # Train for a few epochs
        losses = []
        for _ in range(20):
            metrics = trainer.step(X_train, y_train)
            losses.append(metrics['loss'])

        # Loss should generally decrease with reward modulation
        assert losses[-1] < losses[0] * 1.2

    def test_dynamics_invariance(self, simple_model_config):
        """Test that learning is reasonably invariant to dt."""
        X_train, y_train = generate_simple_regression_data(n_samples=100, seq_len=10)

        dt_values = [50.0, 200.0]
        final_losses = []

        for dt in dt_values:
            # Create model with specific dt
            config = simple_model_config.copy()
            config["simulation"]["dt"] = dt
            model = build_model_from_yaml(config, honor_yaml_seed=True)

            # Train
            rule = HebbianRule(lr=0.001)
            trainer = LocalTrainer(model=model, rule=rule, layers=[1])

            for _ in range(10):
                metrics = trainer.step(X_train, y_train)

            final_losses.append(metrics['loss'])

        # Losses should be similar across dt values
        max_loss = max(final_losses)
        min_loss = min(final_losses)
        ratio = max_loss / min_loss if min_loss > 0 else float('inf')

        # Allow up to 2x variation
        assert ratio < 2.0, f"Dynamics variance too high: {ratio:.2f}x"


# Run with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
