import pytest
import torch

from soen_toolkit.core.noise import (
    CompositeNoise,
    GaussianNoise,
    GaussianPerturbation,
    NoiseSettings,
    build_noise_strategies,
)


def test_gaussian_noise_zero_std_returns_input() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    n = GaussianNoise(std=0.0, relative=False)
    y = n(x)
    assert torch.allclose(y, x)


def test_gaussian_noise_relative_behavior() -> None:
    torch.manual_seed(0)
    # zeros input with relative=True should produce no change
    x0 = torch.zeros(2, 3)
    n_rel = GaussianNoise(std=0.5, relative=True)
    y0 = n_rel(x0)
    assert torch.allclose(y0, x0)

    # ones input should produce non-zero changes with fixed seed
    x1 = torch.ones(4, 5)
    y1 = n_rel(x1)
    assert not torch.allclose(y1, x1)


def test_gaussian_perturbation_constant_over_time_and_reset_changes() -> None:
    torch.manual_seed(0)
    B, T, D = 2, 6, 3
    x = torch.randn(B, T, D)

    # Deterministic offset per sample/node for all timesteps
    p = GaussianPerturbation(mean=1.0, std=0.0)
    y = p(x)
    # Compare offsets, not the input‑containing totals
    off = y - x
    assert torch.allclose(off[:, 0, :], off[:, -1, :])

    # After reset, offset should change when std>0
    p_rand = GaussianPerturbation(mean=0.0, std=0.5)
    y1 = p_rand(x)
    p_rand.reset()
    torch.manual_seed(1)  # change randomness
    y2 = p_rand(x)
    assert not torch.allclose(y1, y2)


def test_composite_noise_adds_offsets() -> None:
    x = torch.zeros(2, 3)
    n1 = GaussianPerturbation(mean=1.0, std=0.0)
    n2 = GaussianPerturbation(mean=2.0, std=0.0)
    comp = CompositeNoise([n1, n2])
    y = comp(x)
    assert torch.allclose(y, torch.full_like(x, 3.0))


def test_build_noise_strategies_basic_mapping_and_extras() -> None:
    torch.manual_seed(0)
    noise = {"phi": 0.1, "relative": False}
    perturb = {"g_mean": 1.0, "g_std": 0.0}
    ns = build_noise_strategies(noise, perturb)
    assert isinstance(ns, NoiseSettings)

    x = torch.zeros(2, 4)
    # phi GaussianNoise absolute (std>0) changes values
    y_phi = ns.apply(x, "phi")
    assert not torch.allclose(y_phi, x)

    # g GaussianPerturbation with mean=1 shifts by +1
    y_g = ns.apply(x, "g")
    assert torch.allclose(y_g, torch.ones_like(x))

    # Extras mapping
    noise2 = {"extras": {"my_key": 0.2}}
    ns2 = build_noise_strategies(noise2, None)
    y_extra = ns2.apply(x, "my_key")
    assert not torch.allclose(y_extra, x)


def test_build_noise_strategies_relative_conflict_raises() -> None:
    noise = {"relative": True, "phi": 0.1}
    perturb = {"phi_std": 0.1}
    with pytest.raises(ValueError):
        _ = build_noise_strategies(noise, perturb)


def test_extras_noise_is_time_varying_but_perturb_is_constant() -> None:
    # Build a NoiseSettings where an extra key has noise vs perturb
    torch.manual_seed(0)
    B, T, D = 2, 8, 3
    base = torch.zeros(B, T, D)

    # Case 1: extras noise -> different offsets at different timesteps
    ns_noise = build_noise_strategies({"extras": {"phi_offset": 0.3}}, None)
    y1 = ns_noise.apply(base, "phi_offset")
    # Compare timestep 0 vs last; high probability they differ for nonzero std
    assert not torch.allclose(y1[:, 0, :], y1[:, -1, :])

    # Case 2: extras perturb -> constant offset per forward
    ns_pert = build_noise_strategies(None, {"extras_std": {"phi_offset": 0.3}})
    y2 = ns_pert.apply(base, "phi_offset")
    assert torch.allclose(y2[:, 0, :], y2[:, -1, :])
