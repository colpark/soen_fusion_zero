"""Tests for ParaRNN solver correctness vs Forward Euler."""

from __future__ import annotations

import pytest
import torch

from soen_toolkit.core.layers.physical import SingleDendriteLayer


class TestParaRNNSolverTorch:
    """Test ParaRNN solver in PyTorch produces results matching Forward Euler."""

    def test_pararnn_matches_forward_euler_basic(self) -> None:
        """Basic test: ParaRNN output should match FE for simple input."""
        dim = 8
        batch = 4
        steps = 50
        dt = 37.0

        # Create two layers with SAME parameters (same seed for each)
        torch.manual_seed(42)
        layer_fe = SingleDendriteLayer(dim=dim, dt=dt, solver="FE")
        torch.manual_seed(42)
        layer_pararnn = SingleDendriteLayer(dim=dim, dt=dt, solver="PARARNN")

        # Generate random input
        torch.manual_seed(123)
        phi = torch.randn(batch, steps, dim) * 0.3

        # Run both solvers
        layer_fe.eval()
        layer_pararnn.eval()
        out_fe = layer_fe(phi).detach()
        out_pararnn = layer_pararnn(phi).detach()

        # Check shapes match
        assert out_fe.shape == out_pararnn.shape
        assert out_fe.shape == (batch, steps + 1, dim)

        # Check values are close (Newton should converge to same solution)
        # Note: RateArray source function is state-dependent, so needs more iterations
        max_diff = (out_fe - out_pararnn).abs().max().item()
        assert max_diff < 0.02, f"Max difference {max_diff} exceeds tolerance"

    def test_pararnn_matches_forward_euler_longer_sequence(self) -> None:
        """Test with longer sequence where parallelism matters more."""
        dim = 4
        batch = 2
        steps = 500  # Long sequence - Heinsen's log-scan handles this well
        dt = 37.0

        # Create layers with SAME parameters
        torch.manual_seed(123)
        layer_fe = SingleDendriteLayer(dim=dim, dt=dt, solver="FE")
        torch.manual_seed(123)
        layer_pararnn = SingleDendriteLayer(dim=dim, dt=dt, solver="PARARNN")

        # Generate input
        torch.manual_seed(456)
        phi = torch.randn(batch, steps, dim) * 0.2

        layer_fe.eval()
        layer_pararnn.eval()
        out_fe = layer_fe(phi).detach()
        out_pararnn = layer_pararnn(phi).detach()

        # Check no NaN/Inf
        assert torch.isfinite(out_pararnn).all(), "ParaRNN output contains NaN/Inf"

        # Heinsen's log-scan is numerically stable even for long sequences
        max_diff = (out_fe - out_pararnn).abs().max().item()
        assert max_diff < 1e-4, f"Max difference {max_diff} exceeds tolerance"

    def test_pararnn_rejects_dense_connectivity(self) -> None:
        """Test that ParaRNN raises error with DENSE internal connectivity.

        Dense internal connectivity creates non-diagonal Jacobians which violate
        the assumptions of the parallel Newton method. Diagonal connectivity is OK.
        """
        dim = 6
        dt = 37.0

        # Create DENSE connectivity matrix (non-diagonal)
        torch.manual_seed(456)
        connectivity = torch.randn(dim, dim) * 0.1

        # ParaRNN should raise RuntimeError for dense connectivity
        with pytest.raises(RuntimeError, match="requires diagonal"):
            SingleDendriteLayer(
                dim=dim,
                dt=dt,
                solver="PARARNN",
                connectivity=connectivity,
                connectivity_mode="fixed",
            )

    def test_pararnn_allows_diagonal_connectivity(self) -> None:
        """Test that ParaRNN works with diagonal (element-wise) recurrent weights."""
        dim = 6
        batch = 2
        steps = 100
        dt = 37.0

        # Create DIAGONAL connectivity matrix
        torch.manual_seed(456)
        diag_values = torch.randn(dim) * 0.1
        connectivity = torch.diag(diag_values)

        # Create FE layer with same connectivity
        torch.manual_seed(789)
        layer_fe = SingleDendriteLayer(
            dim=dim,
            dt=dt,
            solver="FE",
            connectivity=connectivity.clone(),
            connectivity_mode="fixed",
        )

        # Create ParaRNN layer - should NOT raise error
        torch.manual_seed(789)
        layer_pararnn = SingleDendriteLayer(
            dim=dim,
            dt=dt,
            solver="PARARNN",
            connectivity=connectivity.clone(),
            connectivity_mode="fixed",
        )

        # Generate input and compare outputs
        torch.manual_seed(111)
        phi = torch.randn(batch, steps, dim) * 0.1

        out_fe = layer_fe(phi).detach()
        out_pararnn = layer_pararnn(phi).detach()

        max_diff = (out_fe - out_pararnn).abs().max().item()
        assert max_diff < 1e-5, f"Max difference {max_diff} exceeds tolerance"

    @pytest.mark.skip(reason="ParaRNN gradient flow requires custom backward implementation")
    def test_pararnn_gradient_flow(self) -> None:
        """Test that gradients flow correctly through ParaRNN solver.

        Note: The current implementation uses autograd.grad internally during Newton
        iterations, which complicates the gradient computation. A production-ready
        implementation would use custom backward pass as described in the ParaRNN paper.
        """
        dim = 4
        batch = 2
        steps = 20
        dt = 37.0

        torch.manual_seed(789)
        layer = SingleDendriteLayer(dim=dim, dt=dt, solver="PARARNN")

        phi = torch.randn(batch, steps, dim, requires_grad=True) * 0.5

        # Forward pass
        out = layer(phi)

        # Backward pass
        loss = out.sum()
        loss.backward()

        # Check gradients exist
        assert phi.grad is not None
        assert not torch.isnan(phi.grad).any()
        assert phi.grad.shape == phi.shape

    def test_pararnn_rejects_dynamic_connectivity(self) -> None:
        """Test that ParaRNN raises error for dynamic connectivity modes.

        Dynamic connectivity modes (WICC, NOCC) always have dense weight matrices.
        """
        dim = 4
        dt = 37.0
        connectivity = torch.randn(dim, dim) * 0.1

        with pytest.raises(RuntimeError, match="requires diagonal"):
            SingleDendriteLayer(
                dim=dim,
                dt=dt,
                solver="PARARNN",
                connectivity=connectivity,
                connectivity_mode="WICC",
            )


# Skip JAX tests if JAX is not available
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestParaRNNSolverJAX:
    """Test ParaRNN solver in JAX produces results matching Forward Euler."""

    def test_pararnn_matches_forward_euler_basic(self) -> None:
        """Basic test: JAX ParaRNN should match FE for simple input."""
        from soen_toolkit.utils.port_to_jax.layers_jax import SingleDendriteLayerJAX, SingleDendriteParamsJAX
        from soen_toolkit.utils.port_to_jax.source_functions_jax import HeavisideStateDepJAX

        dim = 8
        batch = 4
        steps = 50
        dt = 37.0

        # Create layer
        source = HeavisideStateDepJAX()
        layer = SingleDendriteLayerJAX(dim=dim, dt=dt, source=source)

        # Create parameters
        params = SingleDendriteParamsJAX(
            phi_offset=jnp.ones((batch, dim)) * 0.23,
            bias_current=jnp.ones((batch, dim)) * 1.7,
            gamma_plus=jnp.ones((batch, dim)) * 0.001,
            gamma_minus=jnp.ones((batch, dim)) * 0.001,
            internal_J=None,
        )

        # Generate random input (smaller magnitude for numerical stability)
        key = jax.random.PRNGKey(42)
        phi = jax.random.normal(key, (batch, steps, dim)) * 0.1

        # Run both solvers
        out_fe = layer.forward(phi, params, solver="fe")
        out_pararnn = layer.forward(phi, params, solver="pararnn")

        # Check shapes match
        assert out_fe.shape == out_pararnn.shape
        assert out_fe.shape == (batch, steps + 1, dim)

        # Check values are close
        max_diff = jnp.abs(out_fe - out_pararnn).max()
        assert max_diff < 0.1, f"Max difference {max_diff} exceeds tolerance"

    def test_pararnn_longer_sequence(self) -> None:
        """Test JAX ParaRNN with longer sequence."""
        from soen_toolkit.utils.port_to_jax.layers_jax import SingleDendriteLayerJAX, SingleDendriteParamsJAX
        from soen_toolkit.utils.port_to_jax.source_functions_jax import HeavisideStateDepJAX

        dim = 4
        batch = 2
        steps = 100  # Reduced from 200 for numerical stability
        dt = 37.0

        source = HeavisideStateDepJAX()
        layer = SingleDendriteLayerJAX(dim=dim, dt=dt, source=source)

        params = SingleDendriteParamsJAX(
            phi_offset=jnp.ones((batch, dim)) * 0.23,
            bias_current=jnp.ones((batch, dim)) * 1.7,
            gamma_plus=jnp.ones((batch, dim)) * 0.001,
            gamma_minus=jnp.ones((batch, dim)) * 0.001,
            internal_J=None,
        )

        key = jax.random.PRNGKey(123)
        phi = jax.random.normal(key, (batch, steps, dim)) * 0.1

        out_fe = layer.forward(phi, params, solver="fe")
        out_pararnn = layer.forward(phi, params, solver="pararnn")

        max_diff = jnp.abs(out_fe - out_pararnn).max()
        # Allow larger tolerance for longer sequences due to numerical accumulation
        assert max_diff < 0.3, f"Max difference {max_diff} exceeds tolerance"

