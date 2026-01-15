import math

import pytest


def _make_converter():
    from soen_toolkit.utils.physical_mappings.soen_conversion_utils import (
        PhysicalConverter,
    )

    return PhysicalConverter(I_c=100e-6, gamma_c=1.5e-9, beta_c=0.3)


def test_scalar_gamma_plus_beta_L_L_roundtrip() -> None:
    converter = _make_converter()
    two_pi = 2.0 * math.pi

    gamma_plus = 2.0
    beta_L = converter.convert(gamma_plus, src="gamma_plus", dst="beta_L")
    assert beta_L == pytest.approx(0.5)

    L = converter.convert(beta_L, src="beta_L", dst="L")
    expected_L = (beta_L * converter.Phi_0) / (two_pi * converter.I_c)
    assert pytest.approx(expected_L) == L

    # Direct gamma_plus -> L
    L2 = converter.convert(gamma_plus, src="gamma_plus", dst="L")
    assert pytest.approx(expected_L) == L2


def test_scalar_gamma_minus_tau_tau_physical_cycle() -> None:
    converter = _make_converter()
    gamma_minus = 3.0
    tau = converter.convert(gamma_minus, src="gamma_minus", dst="tau")
    assert tau == pytest.approx(1.0 / gamma_minus)

    tau_physical = converter.convert(gamma_minus, src="gamma_minus", dst="tau_physical")
    assert tau_physical == pytest.approx((1.0 / gamma_minus) / converter.omega_c)

    # Reverse
    gm_from_tp = converter.convert(tau_physical, src="tau_physical", dst="gamma_minus")
    assert gm_from_tp == pytest.approx(gamma_minus)

    tau_from_tp = converter.convert(tau_physical, src="tau_physical", dst="tau")
    assert tau_from_tp == pytest.approx(tau)


def test_scalar_alpha_rleak_current_flux_time_rate() -> None:
    converter = _make_converter()

    alpha = 0.2
    r_leak = converter.convert(alpha, src="alpha", dst="r_leak")
    assert r_leak == pytest.approx(alpha * converter.r_jj)
    alpha_back = converter.convert(r_leak, src="r_leak", dst="alpha")
    assert alpha_back == pytest.approx(alpha)

    i = 0.5
    I = converter.convert(i, src="i", dst="I")  # noqa: E741  # Physical current (capital I) vs dimensionless (lowercase i)
    assert pytest.approx(i * converter.I_c) == I
    assert converter.convert(I, src="I", dst="i") == pytest.approx(i)

    phi = 1.25
    Phi = converter.convert(phi, src="phi", dst="Phi")
    assert Phi == pytest.approx(phi * converter.Phi_0)
    assert converter.convert(Phi, src="Phi", dst="phi") == pytest.approx(phi)

    t_prime = 10.0
    t = converter.convert(t_prime, src="t_prime", dst="t")
    assert t == pytest.approx(t_prime / converter.omega_c)
    assert converter.convert(t, src="t", dst="t_prime") == pytest.approx(t_prime)

    g_fq = 123.0
    G_fq = converter.convert(g_fq, src="g_fq", dst="G_fq")
    assert G_fq == pytest.approx((g_fq * converter.omega_c) / (2.0 * math.pi))
    assert converter.convert(G_fq, src="G_fq", dst="g_fq") == pytest.approx(g_fq)


def test_numpy_array_shape_and_values() -> None:
    np = pytest.importorskip("numpy")
    converter = _make_converter()
    two_pi = 2.0 * math.pi

    gamma_plus = np.array([[0.5, 1.0, 2.0], [4.0, 0.25, 0.75]], dtype=float)
    out = converter.convert_many({"gamma_plus": gamma_plus}, ["beta_L", "L"])

    beta_L = out["beta_L"]
    L = out["L"]

    assert beta_L.shape == gamma_plus.shape
    assert L.shape == gamma_plus.shape

    expected_beta_L = 1.0 / gamma_plus
    expected_L = (expected_beta_L * converter.Phi_0) / (two_pi * converter.I_c)
    assert np.allclose(beta_L, expected_beta_L)
    assert np.allclose(L, expected_L)


def test_torch_tensor_shape_and_grad() -> None:
    torch = pytest.importorskip("torch")
    converter = _make_converter()
    two_pi = 2.0 * math.pi

    gamma_plus = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    out = converter.convert_many({"gamma_plus": gamma_plus}, ["beta_L", "L"])

    beta_L = out["beta_L"]
    L = out["L"]

    assert isinstance(beta_L, torch.Tensor)
    assert isinstance(L, torch.Tensor)
    assert beta_L.shape == gamma_plus.shape
    assert L.shape == gamma_plus.shape

    expected_beta_L = 1.0 / gamma_plus
    expected_L = (expected_beta_L * converter.Phi_0) / (two_pi * converter.I_c)
    assert torch.allclose(beta_L, expected_beta_L)
    assert torch.allclose(L, expected_L)

    # Check gradient flows
    L.sum().backward()
    assert gamma_plus.grad is not None
    assert gamma_plus.grad.shape == gamma_plus.shape


def test_ergonomic_api_to_and_inputs() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    converter = _make_converter()

    # to(): single target
    gp = 2.0
    L = converter.to("L", gamma_plus=gp)
    assert isinstance(L, float)

    # to(): multiple targets
    out = converter.to(["beta_L", "L"], gamma_plus=gp)
    assert "beta_L" in out
    assert "L" in out

    # inputs(): attribute access and caching (numpy)
    gp_np = np.array([0.5, 1.0])
    ctx = converter.inputs(gamma_plus=gp_np)
    beta_L_np = ctx.beta_L
    L_np = ctx.L
    assert np.allclose(beta_L_np, 1.0 / gp_np)
    assert np.allclose(L_np, converter.to("L", gamma_plus=gp_np))

    # inputs(): require multiple
    tau_s = 3e-9
    ctx2 = converter.inputs(tau_physical=tau_s)
    tau_dimless, gamma_minus = ctx2.require("tau", "gamma_minus")
    assert tau_dimless == pytest.approx(tau_s * converter.omega_c)
    assert gamma_minus == pytest.approx(1.0 / (tau_s * converter.omega_c))


def test_convert_many_inference_paths() -> None:
    np = pytest.importorskip("numpy")
    converter = _make_converter()
    2.0 * math.pi

    # Provide L only, infer beta_L and gamma_plus
    L = np.array([1e-9, 2e-9, 5e-9])
    out = converter.convert_many({"L": L}, ["beta_L", "gamma_plus"])
    beta_L = out["beta_L"]
    gamma_plus = out["gamma_plus"]

    expected_beta_L = (2.0 * math.pi * converter.I_c * L) / converter.Phi_0
    expected_gamma_plus = 1.0 / expected_beta_L
    assert np.allclose(beta_L, expected_beta_L)
    assert np.allclose(gamma_plus, expected_gamma_plus)

    # Provide tau_physical only, infer tau and gamma_minus
    tau_physical = 3e-9
    out2 = converter.convert_many({"tau_physical": tau_physical}, ["tau", "gamma_minus"])  # noqa
    assert out2["tau"] == pytest.approx(tau_physical * converter.omega_c)
    assert out2["gamma_minus"] == pytest.approx(1.0 / (tau_physical * converter.omega_c))

    # Provide gamma_minus + gamma_plus -> infer alpha and r_leak
    gamma_minus = np.array([0.25, 0.5, 1.0])
    gamma_plus = np.array([2.0, 4.0, 0.5])
    out3 = converter.convert_many({"gamma_minus": gamma_minus, "gamma_plus": gamma_plus}, ["alpha", "r_leak"])  # noqa
    expected_beta_L = 1.0 / gamma_plus
    expected_alpha = gamma_minus * expected_beta_L
    expected_r = expected_alpha * converter.r_jj
    assert np.allclose(out3["alpha"], expected_alpha)
    assert np.allclose(out3["r_leak"], expected_r)


@pytest.mark.parametrize(
    ("src", "dst", "aliases", "value"),
    [
        ("gamma", "betaL", None, 2.0),  # gamma -> gamma_plus, betaL -> beta_L
        ("betaL", "gamma", None, 0.5),  # reverse
        ("rleak", "alpha", None, 123.4),  # rleak -> r_leak
        ("t'", "t", None, 7.0),  # t' -> t_prime
    ],
)
def test_alias_handling(src, dst, aliases, value) -> None:
    converter = _make_converter()
    # Should not raise
    result = converter.convert(value, src=src, dst=dst)
    assert result is not None


def test_error_cases_unknown_and_unsupported() -> None:
    converter = _make_converter()
    with pytest.raises(ValueError):
        converter.convert(1.0, src="unknown_key", dst="beta_L")
    with pytest.raises(ValueError):
        converter.convert(1.0, src="beta_L", dst="unknown_target")
    # Unsupported path (no direct mapping)
    with pytest.raises(ValueError):
        converter.convert(1.0, src="i", dst="L")


def test_divide_by_zero_behaviour() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    converter = _make_converter()

    # numpy: gamma_plus=0 -> beta_L=inf
    gp_np = np.array([0.0, 1.0])
    # Expect a runtime warning from numpy division by zero; acceptable behavior
    with pytest.warns(RuntimeWarning):
        out_np = converter.convert_many({"gamma_plus": gp_np}, ["beta_L"])
    assert math.isinf(out_np["beta_L"][0])
    assert out_np["beta_L"][1] == pytest.approx(1.0)

    # torch: gamma_plus=0 -> beta_L=inf tensor value
    gp_t = torch.tensor([0.0, 1.0], dtype=torch.float64)
    # Torch may or may not emit a warning depending on build; no strict capture here
    out_t = converter.convert_many({"gamma_plus": gp_t}, ["beta_L"])
    assert torch.isinf(out_t["beta_L"][0])
    assert out_t["beta_L"][1].item() == pytest.approx(1.0)
