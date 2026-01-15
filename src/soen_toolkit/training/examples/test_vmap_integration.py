"""Test script comparing standard PyTorch loop vs torch.vmap for Single Dendrite dynamics.

This demonstrates how torch.vmap can potentially speed up the forward Euler integration
by vectorizing over the time dimension instead of using a Python for-loop.
"""

from pathlib import Path
import sys
import time

import torch

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from soen_toolkit.core.layers.physical.dynamics.single_dendrite import SingleDendriteDynamics
from soen_toolkit.core.source_functions import SOURCE_FUNCTIONS


def standard_loop_integration(dynamics, state0, phi, params, dt, steps):
    """Standard Python for-loop integration (current implementation)."""
    batch, _, dim = phi.shape
    history = torch.empty(batch, steps + 1, dim, device=phi.device, dtype=phi.dtype)
    history[:, 0, :] = state0

    current_state = state0
    for t in range(steps):
        phi_t = phi[:, t, :]
        ds_dt = dynamics(current_state, phi_t, params)
        current_state = current_state + dt * ds_dt
        history[:, t + 1, :] = current_state

    return history


def vmap_integration(dynamics, state0, phi, params, dt):
    """Vectorized integration using torch.vmap.

    Note: torch.vmap vectorizes over batch dimensions, not time.
    For time-sequential operations, we still need a loop, but we can
    vectorize the dynamics computation itself.
    """
    batch, steps, dim = phi.shape

    # torch.vmap doesn't have scan, so we still need a loop over time
    # But we can use vmap to vectorize the dynamics call over batch
    from torch.func import vmap

    # Vectorize dynamics over batch dimension
    # The original dynamics function expects (batch, dim) for state and phi_t
    # vmap will map over the first dimension of state and phi_t
    # So we don't need unsqueeze/squeeze if dynamics is already batch-aware
    # Assuming dynamics(state, phi_t, params) takes (batch, dim) for state and phi_t
    batched_dynamics = vmap(dynamics, in_dims=(0, 0, None)) # Map over state and phi_t, params is broadcasted

    history = torch.empty(batch, steps + 1, dim, device=phi.device, dtype=phi.dtype)
    history[:, 0, :] = state0

    current_state = state0
    for t in range(steps):
        phi_t = phi[:, t, :]
        # This is still a loop, but the dynamics call is vectorized
        ds_dt = batched_dynamics(current_state, phi_t, params)
        current_state = current_state + dt * ds_dt
        history[:, t + 1, :] = current_state

    return history


def compiled_integration(dynamics, state0, phi, params, dt, steps):
    """Integration using torch.compile for optimization."""

    @torch.compile
    def compiled_step_loop(state0, phi, params, dt):
        batch, steps, dim = phi.shape
        history = torch.empty(batch, steps + 1, dim, device=phi.device, dtype=phi.dtype)
        history[:, 0, :] = state0

        current_state = state0
        for t in range(steps):
            phi_t = phi[:, t, :]
            ds_dt = dynamics(current_state, phi_t, params)
            current_state = current_state + dt * ds_dt
            history[:, t + 1, :] = current_state

        return history

    return compiled_step_loop(state0, phi, params, dt)



def benchmark_integration_methods():
    """Benchmark standard loop vs vmap integration."""


    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    steps = 100
    dim = 16
    dt = 1.0

    # Create dynamics
    source_func = SOURCE_FUNCTIONS["RateArray"]()
    dynamics = SingleDendriteDynamics(source_function=source_func)

    # Create test data
    state0 = torch.randn(batch_size, dim, device=device)
    phi = torch.randn(batch_size, steps, dim, device=device)
    params = {
        "gamma_plus": torch.tensor(0.001, device=device),
        "gamma_minus": torch.tensor(0.001, device=device),
        "bias_current": torch.tensor(1.7, device=device),
        "phi_offset": torch.tensor(0.23, device=device),
    }

    # Warmup
    _ = standard_loop_integration(dynamics, state0, phi, params, dt, steps)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark standard loop
    num_runs = 10

    start = time.perf_counter()
    for _ in range(num_runs):
        result_loop = standard_loop_integration(dynamics, state0, phi, params, dt, steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
    (time.perf_counter() - start) / num_runs



    # Benchmark torch.compile (PyTorch 2.0+ optimization)
    try:
        # Warmup compilation
        _ = compiled_integration(dynamics, state0, phi, params, dt, steps)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Now benchmark compiled version
        start = time.perf_counter()
        for _ in range(num_runs):
            result_compiled = compiled_integration(dynamics, state0, phi, params, dt, steps)
            if device.type == "cuda":
                torch.cuda.synchronize()
        (time.perf_counter() - start) / num_runs


        # Verify correctness
        max_diff = (result_loop - result_compiled).abs().max().item()

        if max_diff < 1e-5:
            pass
        else:
            pass

    except Exception:
        pass



if __name__ == "__main__":
    benchmark_integration_methods()
