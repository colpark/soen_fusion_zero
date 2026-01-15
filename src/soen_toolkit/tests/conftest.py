import contextlib
import os
import sys

import pytest
import torch

# Ensure 'src' is on sys.path for local runs
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
src_path = os.path.join(project_root, "src")
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)


def pytest_runtest_setup(item) -> None:
    # Set a default seed for determinism unless a test overrides it
    torch.manual_seed(0)


@pytest.fixture(scope="session", autouse=True)
def _force_jax_cpu_backend() -> None:
    # Force JAX to CPU for the entire test session to avoid METAL issues on macOS
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        import jax

        with contextlib.suppress(Exception):
            jax.config.update("jax_platforms", "cpu")
    except Exception:
        # JAX not installed or import failed; skip
        pass
