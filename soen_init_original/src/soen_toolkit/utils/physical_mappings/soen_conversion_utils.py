# FILEPATH: src/soen_toolkit/utils/physical_mappings/soen_conversion_utils.py

from collections.abc import Mapping, Sequence
import math
from typing import Any

# Centralized constants
from soen_toolkit.physics.constants import (
    DEFAULT_BETA_C as _DEFAULT_BETA_C,
    DEFAULT_GAMMA_C as _DEFAULT_GAMMA_C,
    DEFAULT_IC as _DEFAULT_IC,
    DEFAULT_PHI0 as _DEFAULT_PHI0,
)

# Optional array/tensor libraries
try:  # Torch is optional at import time
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except Exception:
    np = None
    _NUMPY_AVAILABLE = False


class PhysicalConverter:
    r"""Core converter class for SOEN calculations.

    Definitions (symbols and units):
    - Phi_0 [Wb]: Magnetic flux quantum = h / (2e)
    - I_c [A]: Critical current of a junction
    - c_j [F]: Junction capacitance, c_j = gamma_c * I_c
    - r_jj [Ω]: Junction (characteristic) resistance derived from beta_c
    - V_j [V]: Junction characteristic voltage, V_j = I_c * r_jj
    - omega_c [rad/s]: Josephson characteristic angular frequency
    - omega_p [rad/s]: Plasma angular frequency
    - tau_0 [s]: Characteristic time constant, tau_0 = Phi_0 / (2π I_c r_jj)

    Dimensionless parameters used throughout the toolkit:
    - beta_L (\(\beta_L\)) [-]: Inductive parameter = (2π I_c L) / Phi_0
    - gamma (\(\gamma\)) [-]: Convenience inverse of beta_L, gamma = 1 / beta_L
      We use specific symbols when context requires:
        - gamma_plus (\(\gamma_+\)) as 1 / beta_L
        - gamma_minus (\(\gamma_-\)) as 1 / tau where tau is a dimensionless time constant
    - alpha (\(\alpha\)) [-]: Dimensionless leak resistance ratio: alpha = r_leak / r_jj
    - tau (\(\tau\)) [-]: Dimensionless time constant with relationships:
        tau = beta_L / alpha,  gamma_- = 1 / tau,  alpha = gamma_- * beta_L

    Base Parameters provided to the converter upon initialization:
    - I_c: Critical current
    - gamma_c: Capacitance proportionality (F/A)
    - beta_c: Stewart–McCumber parameter (dimensionless)
    """

    # Fixed constant sourced from centralized module: Magnetic flux quantum [Wb]
    Phi_0 = _DEFAULT_PHI0

    def __init__(
        self,
        I_c: float = _DEFAULT_IC,  # Critical current [A]
        gamma_c: float = _DEFAULT_GAMMA_C,  # Proportionality between capacitance and Ic [F/A]
        beta_c: float = _DEFAULT_BETA_C,  # Stewart-McCumber parameter
    ) -> None:
        """Initialize with base physical constants."""
        # Base constants - using properties to ensure derived values are updated
        self._I_c = I_c
        self._gamma_c = gamma_c
        self._beta_c = beta_c
        self._update_derived_parameters()

    def _update_derived_parameters(self) -> None:
        """Update all derived parameters based on current base values."""
        # First calculate c_j as it's needed for other calculations
        self._c_j = self._calculate_junction_capacitance()

        # Then calculate all other derived parameters
        self._r_jj = self._calculate_junction_resistance()
        self._omega_c = self._calculate_josephson_frequency()
        self._omega_p = self._calculate_plasma_frequency()
        self._tau_0 = self._calculate_characteristic_time()
        self._V_j = self._calculate_junction_voltage()

    # Properties to ensure derived parameters update when base parameters change
    @property
    def I_c(self):
        return self._I_c

    @I_c.setter
    def I_c(self, value) -> None:
        self._I_c = value
        self._update_derived_parameters()

    @property
    def gamma_c(self):
        return self._gamma_c

    @gamma_c.setter
    def gamma_c(self, value) -> None:
        self._gamma_c = value
        self._update_derived_parameters()

    @property
    def beta_c(self):
        return self._beta_c

    @beta_c.setter
    def beta_c(self, value) -> None:
        self._beta_c = value
        self._update_derived_parameters()

    # Properties for derived parameters
    @property
    def c_j(self):
        return self._c_j

    @property
    def r_jj(self):
        return self._r_jj

    @property
    def omega_c(self):
        return self._omega_c

    @property
    def omega_p(self):
        return self._omega_p

    @property
    def tau_0(self):
        return self._tau_0

    @property
    def V_j(self):
        return self._V_j

    # ---------- Base Parameter Calculations ----------
    def _calculate_junction_capacitance(self) -> float:
        """Junction capacitance: c_j = γ_c * I_c."""
        return self.gamma_c * self.I_c

    def _calculate_junction_resistance(self) -> float:
        """Junction resistance from β_c: r_jj = sqrt((β_c * Φ_0)/(2π * c_j * I_c))."""
        return math.sqrt(
            (self.beta_c * self.Phi_0) / (2 * math.pi * self.c_j * self.I_c),
        )

    def _calculate_josephson_frequency(self) -> float:
        """Josephson frequency: ω_c = (2π * I_c * r_jj) / Φ_0."""
        return (2 * math.pi * self.I_c * self.r_jj) / self.Phi_0

    def _calculate_plasma_frequency(self) -> float:
        """Plasma frequency: ω_p = sqrt((2π * I_c)/(Φ_0 * c_j))."""
        try:
            return math.sqrt((2 * math.pi * self.I_c) / (self.Phi_0 * self.c_j))
        except (ValueError, ZeroDivisionError):
            raise

    def _calculate_characteristic_time(self) -> float:
        """Characteristic time: τ_0 = Φ_0/(2π * I_c * r_jj)."""
        return self.Phi_0 / (2 * math.pi * self.I_c * self.r_jj)

    def _calculate_junction_voltage(self) -> float:
        """Junction voltage: V_j = I_c * r_jj."""
        return self.I_c * self.r_jj

    # ---------- Physical ↔ Dimensionless Conversions ----------

    # Current
    def physical_to_dimensionless_current(self, current_I: float) -> float:
        """I = I/I_c."""
        return current_I / self.I_c

    def dimensionless_to_physical_current(self, i: float) -> float:
        """I = i * I_c."""
        return i * self.I_c

    # Flux
    def physical_to_dimensionless_flux(self, Phi: float) -> float:
        """φ = Φ/Φ_0."""
        return Phi / self.Phi_0

    def dimensionless_to_physical_flux(self, phi: float) -> float:
        """Φ = φ * Φ_0."""
        return phi * self.Phi_0

    # Inductance (and gamma)
    def physical_to_dimensionless_inductance(self, L: float) -> float:
        """β_L = (2π * I_c * L)/Φ_0."""
        return (2 * math.pi * self.I_c * L) / self.Phi_0

    def dimensionless_to_physical_inductance(self, beta_L: float) -> float:
        """L = (β_L * Φ_0)/(2π * I_c)."""
        return (beta_L * self.Phi_0) / (2 * math.pi * self.I_c)

    def beta_L_to_gamma(self, beta_L: float) -> float:
        """γ = 1/β_L."""
        return 1.0 / beta_L if beta_L != 0 else float("inf")

    def gamma_to_beta_L(self, gamma: float) -> float:
        """β_L = 1/γ."""
        return 1.0 / gamma if gamma != 0 else float("inf")

    # Resistance
    def physical_to_dimensionless_resistance(self, r_leak: float) -> float:
        """α = r_leak/r_jj."""
        return r_leak / self.r_jj

    def dimensionless_to_physical_resistance(self, alpha: float) -> float:
        """r_leak = α * r_jj."""
        return alpha * self.r_jj

    # Time
    def physical_to_dimensionless_time(self, t: float) -> float:
        """T' = t * ω_c."""
        return t * self.omega_c

    def dimensionless_to_physical_time(self, t_prime: float) -> float:
        """T = t'/ω_c."""
        return t_prime / self.omega_c

    # Flux quantum rate
    def physical_to_dimensionless_fq_rate(self, G_fq: float) -> float:
        """g_fq = (2π * G_fq)/ω_c."""
        return (2 * math.pi * G_fq) / self.omega_c

    def dimensionless_to_physical_fq_rate(self, g_fq: float) -> float:
        """G_fq = (g_fq * ω_c)/(2π)."""
        return (g_fq * self.omega_c) / (2 * math.pi)

    # Tau (physical) and Gamma Minus (dimensionless)
    def dimensionless_gamma_minus_to_physical_tau(self, gamma_minus: float) -> float:
        """Converts dimensionless gamma_minus to a physical time constant tau (in seconds).
        Dimensionless tau (tau') = 1 / gamma_minus
        Physical tau = tau' / omega_c.
        """
        if gamma_minus == 0:
            # This implies an infinite dimensionless tau
            return float("inf")

        if self.omega_c == 0:
            # If omega_c is zero, physical tau is undefined or infinite
            # for finite dimensionless tau (unless dimensionless tau is also zero,
            # which means gamma_minus is inf).
            # This state (omega_c=0) should ideally be prevented by base parameter validation.
            return float("inf")  # Or float('nan') or raise ValueError

        tau_dimensionless = 1.0 / gamma_minus
        return tau_dimensionless / self.omega_c

    def physical_tau_to_dimensionless_gamma_minus(self, tau_physical: float) -> float:
        """Converts a physical time constant tau (in seconds) to dimensionless gamma_minus.
        Dimensionless tau (tau') = tau_physical * omega_c
        gamma_minus = 1 / tau'.
        """
        if self.omega_c == 0:
            # If omega_c is zero, dimensionless tau (tau') will be zero.
            # Then gamma_minus = 1 / 0, which is infinity.
            # This is consistent: zero characteristic frequency implies infinitely fast dynamics relative to tau_physical,
            # or infinitely slow physical processes for a given dimensionless number.
            # If tau_physical is also 0, tau' is 0, gamma_minus is inf.
            return float("inf")

        tau_dimensionless = tau_physical * self.omega_c

        if tau_dimensionless == 0:
            # This happens if tau_physical is 0 (and omega_c is non-zero).
            return float("inf")

        return 1.0 / tau_dimensionless

    # ---------- Derived Dimensionless Parameters ----------
    def calculate_tau(self, beta_L: float, alpha: float) -> float:
        """τ = β_L/α."""
        return beta_L / alpha if alpha != 0 else float("inf")

    def gamma_plus_to_beta_L(self, gamma_plus: float) -> float:
        """β_L = 1/γ_+."""
        return self.gamma_to_beta_L(gamma_plus)  # Same conversion

    def beta_L_to_gamma_plus(self, beta_L: float) -> float:
        """γ_+ = 1/β_L."""
        return self.beta_L_to_gamma(beta_L)  # Same conversion

    def gamma_minus_to_tau(self, gamma_minus: float) -> float:
        """τ = 1/γ_-."""
        return 1.0 / gamma_minus if gamma_minus != 0 else float("inf")

    def tau_to_gamma_minus(self, tau: float) -> float:
        """γ_- = 1/τ."""
        return 1.0 / tau if tau != 0 else float("inf")

    def gamma_minus_to_alpha_beta_L(self, gamma_minus: float, beta_L: float) -> float:
        """α = γ_- * β_L."""
        return gamma_minus * beta_L

    def gamma_plus_gamma_minus_to_alpha(self, gamma_plus: float, gamma_minus: float) -> float:
        """α = γ_- / γ_+."""
        return gamma_minus / gamma_plus if gamma_plus != 0 else float("inf")

    def get_base_parameters(self) -> dict:
        """Return all base physical parameters."""
        return {
            "I_c": self.I_c,
            "gamma_c": self.gamma_c,
            "beta_c": self.beta_c,
            "c_j": self.c_j,
            "r_jj": self.r_jj,
            "omega_c": self.omega_c,
            "omega_p": self.omega_p,
            "tau_0": self.tau_0,
            "V_j": self.V_j,
        }

    # ---------- Array/Tensor-friendly conversion helpers ----------
    def _canonical(self, name: str) -> str:
        """Map a possibly aliased symbol name to a canonical one."""
        if not isinstance(name, str):
            msg = "Conversion names must be strings"
            raise ValueError(msg)
        key = name.strip()
        aliases = {
            # Inductance family
            "gamma_plus": "gamma_plus",
            "gamma+": "gamma_plus",
            "gamma": "gamma_plus",
            "gp": "gamma_plus",
            "beta_L": "beta_L",
            "betaL": "beta_L",
            "beta_l": "beta_L",
            "beta": "beta_L",
            "L": "L",
            # Leak/time family
            "gamma_minus": "gamma_minus",
            "gamma-": "gamma_minus",
            "gm": "gamma_minus",
            "tau": "tau",
            "tau_physical": "tau_physical",
            "tau_phys": "tau_physical",
            "tau_s": "tau_physical",
            "alpha": "alpha",
            "r_leak": "r_leak",
            "rleak": "r_leak",
            # Current/flux/time
            "i": "i",
            "I": "I",
            "phi": "phi",
            "Phi": "Phi",
            "t'": "t_prime",
            "t_prime": "t_prime",
            "t": "t",
            # Flux quantum rate
            "g_fq": "g_fq",
            "G_fq": "G_fq",
        }
        if key not in aliases:
            msg = f"Unknown conversion key '{name}'"
            raise ValueError(msg)
        return aliases[key]

    def _ensure_arraylike(self, x: Any) -> Any:
        """Return an array-like for Python lists/tuples for internal math; leave torch tensors intact.

        - torch.Tensor passes through (to preserve autograd/device)
        - numpy: ensure ndarray
        - list/tuple: convert to numpy ndarray if numpy is available
        else leave as-is (operations may fail)
        """
        if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x
        if _NUMPY_AVAILABLE:
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, (list, tuple)):
                return np.array(x)
        return x

    def _to_python_number_or_list(self, x: Any) -> Any:
        """Convert numpy arrays to nested lists for JSON friendliness; leave torch tensors unchanged.
        This is not used for in-library returns, but can be useful for endpoints.
        """
        if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x
        if _NUMPY_AVAILABLE and isinstance(x, np.ndarray):
            return x.tolist()
        return x

    def _const_like(self, x: Any, value: float) -> Any:
        """Create a constant with the same type context as x where applicable."""
        if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return torch.as_tensor(value, dtype=x.dtype, device=x.device)
        # For numpy arrays, plain Python float will broadcast fine and preserve ndarray dtype
        return value

    # Public, array/tensor-friendly API
    def convert(self, value: Any, src: str, dst: str) -> Any:
        """Convert value from src to dst.

        - Accepts scalars, numpy arrays, or torch tensors. Operations preserve shape and, for torch, autograd.
        - Supported canonical names: 'gamma_plus', 'beta_L', 'L', 'gamma_minus', 'tau', 'tau_physical',
          'alpha', 'r_leak', 'i', 'I', 'phi', 'Phi', 't_prime', 't', 'g_fq', 'G_fq'.
        - Aliases are accepted (e.g., 'gamma', 'betaL', "t'").
        """
        s = self._canonical(src)
        d = self._canonical(dst)
        if s == d:
            return value

        x = self._ensure_arraylike(value)
        two_pi = 2.0 * math.pi

        # Simple 1-1 mappings first
        if s == "gamma_plus" and d == "beta_L":
            return self._const_like(x, 1.0) / x
        if s == "beta_L" and d == "gamma_plus":
            return self._const_like(x, 1.0) / x
        if s == "beta_L" and d == "L":
            return (x * self.Phi_0) / (two_pi * self.I_c)
        if s == "L" and d == "beta_L":
            return (two_pi * self.I_c * x) / self.Phi_0
        if s == "gamma_plus" and d == "L":
            # L = ( (1/gamma_plus) * Phi_0 ) / (2*pi*I_c)
            return ((self._const_like(x, 1.0) / x) * self.Phi_0) / (two_pi * self.I_c)
        if s == "gamma_minus" and d == "tau":
            return self._const_like(x, 1.0) / x
        if s == "tau" and d == "gamma_minus":
            return self._const_like(x, 1.0) / x
        if s == "gamma_minus" and d == "tau_physical":
            return (self._const_like(x, 1.0) / x) / self.omega_c
        if s == "tau_physical" and d == "gamma_minus":
            return self._const_like(x, 1.0) / (x * self.omega_c)
        if s == "tau" and d == "tau_physical":
            return x / self.omega_c
        if s == "tau_physical" and d == "tau":
            return x * self.omega_c
        if s == "alpha" and d == "r_leak":
            return x * self.r_jj
        if s == "r_leak" and d == "alpha":
            return x / self.r_jj
        if s == "i" and d == "I":
            return x * self.I_c
        if s == "I" and d == "i":
            return x / self.I_c
        if s == "phi" and d == "Phi":
            return x * self.Phi_0
        if s == "Phi" and d == "phi":
            return x / self.Phi_0
        if s == "t_prime" and d == "t":
            return x / self.omega_c
        if s == "t" and d == "t_prime":
            return x * self.omega_c
        if s == "g_fq" and d == "G_fq":
            return (x * self.omega_c) / (two_pi)
        if s == "G_fq" and d == "g_fq":
            return (two_pi * x) / self.omega_c

        # Derived mappings via simple chains
        # Example: beta_L -> gamma_plus -> L handled above, so remaining cases are minimal.
        msg = f"Unsupported conversion path: {src} -> {dst}"
        raise ValueError(msg)

    def convert_many(self, inputs: Mapping[str, Any], targets: Sequence[str]) -> dict[str, Any]:
        """Compute multiple targets from provided inputs.

        - inputs: mapping from variable names to values (scalars/arrays/tensors). Names can use aliases.
        - targets: list of variable names desired. Names can use aliases.
        - Returns a dict mapping canonical target names to values, preserving shapes/types.

        Notes:
        - For alpha or r_leak, you can supply either directly or via gamma_minus with beta_L or gamma_plus.
        - For L, you can supply L directly, or beta_L or gamma_plus.
        - For tau/tau_physical/gamma_minus, any one determines the others given omega_c.

        """
        # Normalize keys
        cache: dict[str, Any] = {}
        for k, v in inputs.items():
            cache[self._canonical(k)] = v

        def have(name: str) -> bool:
            return name in cache and cache[name] is not None

        def setv(name: str, val: Any) -> None:
            cache[name] = val

        def get(name: str) -> Any:
            name = self._canonical(name)
            if have(name):
                return cache[name]

            # Compute on demand
            if name == "beta_L":
                if have("beta_L"):
                    return cache["beta_L"]
                if have("gamma_plus"):
                    setv("beta_L", self.convert(cache["gamma_plus"], "gamma_plus", "beta_L"))
                elif have("L"):
                    setv("beta_L", self.convert(cache["L"], "L", "beta_L"))
                else:
                    msg = "beta_L requires one of: beta_L, gamma_plus, L"
                    raise ValueError(msg)
                return cache["beta_L"]

            if name == "gamma_plus":
                if have("gamma_plus"):
                    return cache["gamma_plus"]
                if have("beta_L"):
                    setv("gamma_plus", self.convert(cache["beta_L"], "beta_L", "gamma_plus"))
                elif have("L"):
                    # L -> beta_L -> gamma_plus
                    bl = self.convert(cache["L"], "L", "beta_L")
                    setv("gamma_plus", self.convert(bl, "beta_L", "gamma_plus"))
                else:
                    msg = "gamma_plus requires one of: gamma_plus, beta_L, L"
                    raise ValueError(msg)
                return cache["gamma_plus"]

            if name == "L":
                if have("L"):
                    return cache["L"]
                if have("beta_L"):
                    setv("L", self.convert(cache["beta_L"], "beta_L", "L"))
                elif have("gamma_plus"):
                    setv("L", self.convert(cache["gamma_plus"], "gamma_plus", "L"))
                else:
                    msg = "L requires one of: L, beta_L, gamma_plus"
                    raise ValueError(msg)
                return cache["L"]

            if name in ("tau", "gamma_minus", "tau_physical"):
                # Any one determines the others
                if have("tau"):
                    tau_val = cache["tau"]
                    setv("gamma_minus", self.convert(tau_val, "tau", "gamma_minus"))
                    setv("tau_physical", self.convert(tau_val, "tau", "tau_physical"))
                elif have("gamma_minus"):
                    gm = cache["gamma_minus"]
                    setv("tau", self.convert(gm, "gamma_minus", "tau"))
                    setv("tau_physical", self.convert(gm, "gamma_minus", "tau_physical"))
                elif have("tau_physical"):
                    tp = cache["tau_physical"]
                    # tau_physical -> gamma_minus -> tau
                    gm = self.convert(tp, "tau_physical", "gamma_minus")
                    setv("gamma_minus", gm)
                    setv("tau", self.convert(gm, "gamma_minus", "tau"))
                else:
                    msg = "Provide one of tau, gamma_minus, tau_physical"
                    raise ValueError(msg)
                return cache[name]

            if name in ("alpha", "r_leak"):
                if have("alpha"):
                    setv("r_leak", self.convert(cache["alpha"], "alpha", "r_leak"))
                elif have("r_leak"):
                    setv("alpha", self.convert(cache["r_leak"], "r_leak", "alpha"))
                # Try via gamma_minus + beta_L (or gamma_plus)
                elif have("gamma_minus") and have("beta_L"):
                    alpha_val = self._ensure_arraylike(cache["gamma_minus"]) * self._ensure_arraylike(get("beta_L"))
                    setv("alpha", alpha_val)
                    setv("r_leak", self.convert(alpha_val, "alpha", "r_leak"))
                elif have("gamma_minus") and have("gamma_plus"):
                    bl = get("beta_L")  # will derive from gamma_plus
                    alpha_val = self._ensure_arraylike(cache["gamma_minus"]) * self._ensure_arraylike(bl)
                    setv("alpha", alpha_val)
                    setv("r_leak", self.convert(alpha_val, "alpha", "r_leak"))
                else:
                    msg = "alpha/r_leak require one of: alpha, r_leak, or (gamma_minus with beta_L or gamma_plus)"
                    raise ValueError(msg)
                return cache[name]

            if name in ("i", "I"):
                if have("i"):
                    setv("I", self.convert(cache["i"], "i", "I"))
                elif have("I"):
                    setv("i", self.convert(cache["I"], "I", "i"))
                else:
                    msg = "Provide one of i or I"
                    raise ValueError(msg)
                return cache[name]

            if name in ("phi", "Phi"):
                if have("phi"):
                    setv("Phi", self.convert(cache["phi"], "phi", "Phi"))
                elif have("Phi"):
                    setv("phi", self.convert(cache["Phi"], "Phi", "phi"))
                else:
                    msg = "Provide one of phi or Phi"
                    raise ValueError(msg)
                return cache[name]

            if name in ("t", "t_prime"):
                if have("t_prime"):
                    setv("t", self.convert(cache["t_prime"], "t_prime", "t"))
                elif have("t"):
                    setv("t_prime", self.convert(cache["t"], "t", "t_prime"))
                else:
                    msg = "Provide one of t or t_prime"
                    raise ValueError(msg)
                return cache[name]

            if name in ("g_fq", "G_fq"):
                if have("g_fq"):
                    setv("G_fq", self.convert(cache["g_fq"], "g_fq", "G_fq"))
                elif have("G_fq"):
                    setv("g_fq", self.convert(cache["G_fq"], "G_fq", "g_fq"))
                else:
                    msg = "Provide one of g_fq or G_fq"
                    raise ValueError(msg)
                return cache[name]

            msg = f"Don't know how to compute '{name}' from provided inputs"
            raise ValueError(msg)

        outputs: dict[str, Any] = {}
        for t in targets:
            cname = self._canonical(t)
            outputs[cname] = get(cname)
        return outputs

    # Convenience top-level-style wrapper
    def convert_to(self, value: Any, to: str | Sequence[str], src: str) -> Any:
        """Alias for convert/convert_many for ergonomic calling."""
        if isinstance(to, str):
            return self.convert(value, src=src, dst=to)
        # Multi-target: feed as inputs under src
        return self.convert_many({src: value}, list(to))

    # ---------- Modern ergonomic API ----------
    def to(self, targets: str | Sequence[str], **inputs: Any) -> Any:
        """One-shot conversion using keyword inputs, no dicts/lists needed.

        Examples:
            L = converter.to('L', gamma_plus=gp)
            results = converter.to(['beta_L', 'L'], gamma_plus=gp)

        """
        if isinstance(targets, str):
            out = self.convert_many(inputs, [targets])
            return out[self._canonical(targets)]
        return self.convert_many(inputs, list(targets))

    def inputs(self, **inputs: Any) -> "ConversionContext":
        """Create a lazy conversion context with attribute-style access.

        Examples:
            ctx = converter.inputs(gamma_plus=gp)
            beta_L = ctx.beta_L
            L = ctx.L

            tau, gamma_minus = ctx.require('tau', 'gamma_minus')

        """
        return ConversionContext(self, inputs)


def get_multiplier_v2_defaults() -> dict[str, float]:
    """Get recommended default parameters for multiplier v2.

    These defaults correspond to physical parameters:
        - beta_1 ≈ 1nH → beta = 303.85
        - beta_out ≈ 300pH → beta_out = 91.156
        - i_b ≈ 210μA → ib = 2.1
        - R ≈ 2Ω → alpha = 1.64053

    Returns:
        Dictionary with keys: beta, beta_out, alpha, ib
    """
    return {
        "beta": 303.85,
        "beta_out": 91.156,
        "alpha": 1.64053,
        "bias_current": 2.1,
    }


def beta_from_inductance(
    L: float,
    I_c: float = _DEFAULT_IC,
    Phi_0: float = _DEFAULT_PHI0,
) -> float:
    """Convert inductance to dimensionless beta parameter.

    Formula: beta_L = (2π I_c L) / Phi_0

    Args:
        L: Inductance in Henries (H)
        I_c: Critical current in Amperes (A)
        Phi_0: Flux quantum in Webers (Wb)

    Returns:
        Dimensionless beta_L parameter

    Example:
        >>> beta_from_inductance(1e-9)  # 1 nH
        303.85...
        >>> beta_from_inductance(300e-12)  # 300 pH
        91.156...
    """
    return (2 * math.pi * I_c * L) / Phi_0


def alpha_from_resistance(
    R: float,
    I_c: float = _DEFAULT_IC,
    beta_c: float = _DEFAULT_BETA_C,
    gamma_c: float = _DEFAULT_GAMMA_C,
) -> float:
    """Convert resistance to dimensionless alpha parameter.

    Formula: alpha = R / r_jj
    where r_jj is the junction characteristic resistance derived from beta_c.

    Args:
        R: Resistance in Ohms (Ω)
        I_c: Critical current in Amperes (A)
        beta_c: Stewart-McCumber parameter (dimensionless)
        gamma_c: Capacitance proportionality (F/A)

    Returns:
        Dimensionless alpha parameter (resistance ratio)

    Example:
        >>> alpha_from_resistance(2.0)  # 2 Ohms
        1.64053...
    """
    # Calculate junction resistance using the PhysicalConverter
    converter = PhysicalConverter(I_c=I_c, gamma_c=gamma_c, beta_c=beta_c)
    r_jj = converter.r_jj
    return R / r_jj


def ib_from_current(
    i_b_physical: float,
    I_c: float = _DEFAULT_IC,
) -> float:
    """Convert physical bias current to dimensionless ib parameter.

    Formula: ib = i_b_physical / (I_c / 100)

    Note: The factor of 100 is a normalization used in the toolkit.

    Args:
        i_b_physical: Bias current in Amperes (A)
        I_c: Critical current in Amperes (A)

    Returns:
        Dimensionless ib parameter

    Example:
        >>> ib_from_current(210e-6)  # 210 μA
        2.1
    """
    return i_b_physical / (I_c / 100)


class ConversionContext:
    """Lazy, cache-backed view over conversions.

    Access variables as attributes (e.g., `.beta_L`, `.L`). Missing values are
    computed from the provided inputs using the associated PhysicalConverter.
    Results are cached for subsequent access.
    """

    def __init__(self, converter: PhysicalConverter, inputs: Mapping[str, Any]) -> None:
        self._converter = converter
        # Canonicalize keys
        self._cache: dict[str, Any] = {converter._canonical(k): v for k, v in inputs.items()}

    def require(self, *targets: str, as_dict: bool = False):
        """Return one or more targets; cache results.

        - If one target and as_dict=False: returns the single value
        - If multiple targets: returns a tuple in the same order unless as_dict=True
        """
        if len(targets) == 0:
            msg = "At least one target must be specified"
            raise ValueError(msg)
        canonical_targets = [self._converter._canonical(t) for t in targets]
        out = self._converter.convert_many(self._cache, canonical_targets)
        # Update cache
        self._cache.update(out)
        if as_dict or len(canonical_targets) > 1:
            if as_dict:
                return {k: out[k] for k in canonical_targets}
            return tuple(out[k] for k in canonical_targets)
        return out[canonical_targets[0]]

    def take(self, *targets: str) -> dict[str, Any]:
        """Return a dict of requested targets; convenience wrapper for require(as_dict=True)."""
        return self.require(*targets, as_dict=True)

    def __getattr__(self, name: str) -> Any:
        # Support attribute access with aliases
        try:
            cname = self._converter._canonical(name)
        except Exception as e:
            raise AttributeError(name) from e
        # Cached?
        if cname in self._cache:
            return self._cache[cname]
        # Compute and cache
        return self.require(cname)
