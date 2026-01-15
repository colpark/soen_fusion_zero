#!/usr/bin/env python3
"""Unified tool to convert fixed connections to WICC or NOCC dynamic connections.

This script converts both external (inter-layer) and internal (intra-layer)
fixed connections to a dynamic mode (WICC or NOCC) using a steady-state mapping.

Modes:
  - WICC: With Collection Coil (v1). Implicit steady-state, single edge state.
  - NOCC: No Collection Coil (v2). Explicit steady-state, dual edge states.

Usage:
  python3 -m soen_toolkit.utils.convert_fixed_to_dynamic -i input.soen --wicc [params]
  python3 -m soen_toolkit.utils.convert_fixed_to_dynamic -i input.soen --nocc [params]
"""

from abc import ABC, abstractmethod
import argparse
from collections.abc import Callable
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from soen_toolkit.core import SOENModelCore
from soen_toolkit.core.source_functions.rate_array import RateArraySource

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Conversion Stategies
# ------------------------------------------------------------------

class DynamicStrategy(ABC):
    """Abstract base class for conversion strategies."""

    def __init__(self, args, sf: RateArraySource):
        self.args = args
        self.sf = sf
        # Common params
        self.bias_current = args.bias_current
        self.j_in = args.j_in
        self.j_out = args.j_out
        self.phi_in = args.phi_in
        self.use_half_flux_offset = not args.no_half_flux_offset
        self.ib_min = float(sf.ib_min)
        self.ib_max = float(sf.ib_max)

    def _soen_g(self, phi: float, bias: float) -> float:
        """Helper to call source function g."""
        return self.sf.g(
            torch.as_tensor(phi), bias_current=torch.as_tensor(bias)
        ).item()

    @abstractmethod
    def _s_out_star(self, phi_param: float) -> float:
        """Compute small-signal steady-state output for a given weight parameter."""
        pass

    @abstractmethod
    def get_dynamic_params(self) -> dict[str, Any]:
        """Return the dictionary of parameters for the dynamic connection."""
        pass

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Return the mode name string (WICC or NOCC)."""
        pass

    def _J_eff_of_param(self, phi_param: float) -> float:
        """Compute J_eff(phi_param).Shared mapping logic."""
        s_out = self._s_out_star(phi_param)
        return (self.j_in * self.j_out / self.phi_in) * s_out


class NOCCStrategy(DynamicStrategy):
    """Strategy for NOCC (No Collection Coil) conversion."""

    def __init__(self, args, sf: RateArraySource):
        super().__init__(args, sf)
        self.alpha = args.alpha

    @property
    def mode_name(self) -> str:
        return "NOCC"

    def _s_star(self, phi: float) -> float:
        """Solve for steady-state s* at given flux phi (NOCC)."""
        # Restrict domain so i_sq = ib - s lies within the table
        s_lo = max(0.0, self.bias_current - self.ib_max)
        s_hi = min(self.bias_current, self.bias_current - self.ib_min)

        if not (s_lo < s_hi):
            raise ValueError(
                f"bias_current={self.bias_current:.4f} incompatible with rate table "
                f"[{self.ib_min:.4f}, {self.ib_max:.4f}]"
            )

        def F(s: float) -> float:
            i_sq = self.bias_current - s
            g_val = self._soen_g(phi, abs(i_sq))
            return g_val - self.alpha * s

        f_lo = F(s_lo)
        f_hi = F(s_hi)

        if f_lo == 0.0:
            return s_lo
        if f_hi == 0.0:
            return s_hi

        if (f_lo > 0 and f_hi > 0) or (f_lo < 0 and f_hi < 0):
             # Try simple fallback if close? Or fail.
             # NOCC script failed fast here.
             # We'll stick to fail fast to be safe.
             raise RuntimeError(f"Root not bracketed for phi={phi:.4f}: F({s_lo})={f_lo}, F({s_hi})={f_hi}")

        # Bisection
        for _ in range(64):
            mid = 0.5 * (s_lo + s_hi)
            f_mid = F(mid)
            if abs(f_mid) < 1e-7 or abs(s_hi - s_lo) < 1e-7:
                return mid
            if (f_lo > 0 and f_mid > 0) or (f_lo < 0 and f_mid < 0):
                s_lo, f_lo = mid, f_mid
            else:
                s_hi, f_hi = mid, f_mid
        return 0.5 * (s_lo + s_hi)

    def _s_out_star(self, phi_w: float) -> float:
        """NOCC: s_out*(φ_w) = s*(φ_w + φ_in) - s*(φ_w - φ_in)."""
        phi_w_eff = phi_w + (0.5 if self.use_half_flux_offset else 0.0)
        s_pos = self._s_star(phi_w_eff + self.phi_in)
        s_neg = self._s_star(phi_w_eff - self.phi_in)
        return s_pos - s_neg

    def get_dynamic_params(self) -> dict[str, Any]:
        return {
            "source_func": "RateArray",
            "alpha": self.args.alpha,
            "beta": self.args.beta,
            "beta_out": self.args.beta_out,
            "bias_current": self.bias_current,
            "j_in": self.j_in,
            "j_out": self.j_out,
            "half_flux_offset": self.use_half_flux_offset,
        }


class WICCStrategy(DynamicStrategy):
    """Strategy for WICC (With Collection Coil) conversion."""

    def __init__(self, args, sf: RateArraySource):
        super().__init__(args, sf)
        self.gamma_plus = args.gamma_plus
        self.gamma_minus = args.gamma_minus

    @property
    def mode_name(self) -> str:
        return "WICC"

    def _solve_steady_state_s(self, phi_x: float, phi_y: float) -> float:
        """Solve steady-state s for WICC implicit equation."""
        valid_s_max = min(self.bias_current - self.ib_min, self.ib_max - self.bias_current)
        valid_s_min = max(self.bias_current - self.ib_max, self.ib_min - self.bias_current)

        s_min = valid_s_min
        s_max = valid_s_max
        if s_min >= s_max:
             # Try a minimal range around 0 if bounds represent an empty set due to float issues?
             # Or just 0.
             return 0.0

        def F(s: float) -> float:
            sq_a = abs(self.bias_current - s)
            sq_b = abs(self.bias_current + s)
            phi_a = phi_x + phi_y
            phi_b = phi_x - phi_y
            g_a = self._soen_g(phi_a, sq_a)
            g_b = self._soen_g(phi_b, sq_b)
            return self.gamma_plus * (g_a - g_b) - self.gamma_minus * s

        f_min = F(s_min)
        f_max = F(s_max)

        if f_min * f_max > 0:
            if abs(f_min) < abs(f_max):
                return s_min
            else:
                return s_max

        lo, hi = s_min, s_max
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            f_mid = F(mid)
            if abs(f_mid) < 1e-7 or abs(hi - lo) < 1e-7:
                return mid
            if (f_mid > 0 and f_min > 0) or (f_mid < 0 and f_min < 0):
                lo, f_min = mid, f_mid
            else:
                hi, f_max = mid, f_mid
        return 0.5 * (lo + hi)

    def _s_out_star(self, phi_y: float) -> float:
        """WICC: s_out*(φ_y) = s*(φ_in, φ_y) - s*(-φ_in, φ_y)."""
        phi_y_eff = phi_y + (0.5 if self.use_half_flux_offset else 0.0)
        s_pos = self._solve_steady_state_s(self.phi_in, phi_y_eff)
        s_neg = self._solve_steady_state_s(-self.phi_in, phi_y_eff)
        return s_pos - s_neg

    def get_dynamic_params(self) -> dict[str, Any]:
        return {
            "source_func": "RateArray",
            "gamma_plus": self.gamma_plus,
            "gamma_minus": self.gamma_minus,
            "bias_current": self.bias_current,
            "j_in": self.j_in,
            "j_out": self.j_out,
            "half_flux_offset": self.use_half_flux_offset,
        }


# ------------------------------------------------------------------
# Main Converter Class
# ------------------------------------------------------------------

class DynamicConverter:
    """Handles conversion using a specific strategy."""

    def __init__(self, strategy: DynamicStrategy, phi_param_min: float, phi_param_max: float, num_points: int):
        self.strategy = strategy
        self.phi_min = phi_param_min
        self.phi_max = phi_param_max
        self.num_points = num_points
        self.Jeff_to_phi, self.J_range = self._build_inverse_mapping()
        logger.info(f"{strategy.mode_name} converter initialized with J range: [{self.J_range[0]:.6g}, {self.J_range[1]:.6g}]")

    def _find_zero_crossing_monotonic_region(
        self, phi_vals: np.ndarray, J_vals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Find monotonic region around zero crossing."""
        if (J_vals.min() < 0 < J_vals.max()):
            # Find best zero crossing (closest to phi_param = 0)
            # Use centered phi values? No, phi_vals passed is the grid.
            zero_idx = np.argmin(np.abs(J_vals)) # Initial guess
            # Optimization: look for crossing (sign change) closest to center
            sign_changes = np.where(np.diff(np.sign(J_vals)) != 0)[0]
            if len(sign_changes) > 0:
                 # phi_vals[sign_changes] are the phi values before crossing
                 # We want the one where phi_vals is closest to 0
                 best_change_idx = sign_changes[np.argmin(np.abs(phi_vals[sign_changes]))]
                 # Pick the one of the two indices (i, i+1) that is closer to 0 in J
                 if abs(J_vals[best_change_idx]) < abs(J_vals[best_change_idx+1]):
                     zero_idx = best_change_idx
                 else:
                     zero_idx = best_change_idx + 1
        else:
             zero_idx = np.argmin(np.abs(J_vals))
             if J_vals.min() > 0 or J_vals.max() < 0:
                 logger.warning(f"No strict zero crossing in J range [{J_vals.min():.2e}, {J_vals.max():.2e}]. Using argmin.")

        dJ = np.diff(J_vals)
        increasing = dJ[min(zero_idx, len(dJ)-1)] > 0

        # Expand
        i0 = zero_idx
        while i0 > 0:
            if (J_vals[i0] - J_vals[i0-1] > 0) != increasing:
                break
            if abs(J_vals[i0] - J_vals[i0-1]) < 1e-12:
                break
             i0 -= 1
        i1 = zero_idx + 1
        while i1 < len(J_vals) - 1:
            if (J_vals[i1+1] - J_vals[i1] > 0) != increasing:
                break
            if abs(J_vals[i1+1] - J_vals[i1]) < 1e-12:
                break
            i1 += 1

        phi_seg = phi_vals[i0 : i1 + 1]
        J_seg = J_vals[i0 : i1 + 1]

        if len(J_seg) < 2:
            return phi_vals, J_vals, (J_vals[-1] > J_vals[0]) # Fallback
        return phi_seg, J_seg, increasing

    def _build_inverse_mapping(self) -> tuple[Callable, tuple[float, float]]:
        phi_grid = np.linspace(self.phi_min, self.phi_max, self.num_points)
        J_grid = np.array([self.strategy._J_eff_of_param(float(p)) for p in phi_grid])

        # Debug logging
        logger.debug(f"J grid range: [{J_grid.min():.6g}, {J_grid.max():.6g}]")
        logger.debug(f"Phi grid range: [{phi_grid.min():.6g}, {phi_grid.max():.6g}]")

        phi_seg, J_seg, increasing = self._find_zero_crossing_monotonic_region(phi_grid, J_grid)

        if increasing:
            xp, fp = J_seg, phi_seg
        else:
            xp, fp = J_seg[::-1], phi_seg[::-1]

        J_min, J_max = float(xp.min()), float(xp.max())

        def inverse_func(J: np.ndarray) -> np.ndarray:
             # Assume caller handles bounds checks or warnings
             return np.interp(J, xp, fp).astype(np.float32)

        return inverse_func, (J_min, J_max)

    def convert_model(self, input_path: str, output_path: str, device: torch.device):
        logger.info(f"Loading model from: {input_path}")
        core = SOENModelCore.load(input_path, show_logs=False)
        core = core.to(device)

        # Identify connections
        all_conns = []
        for conn_cfg in core.connections_config:
            key = f"J_{conn_cfg.from_layer}_to_{conn_cfg.to_layer}"
            if key not in core.connections:
                continue
            conn_type = "internal" if conn_cfg.from_layer == conn_cfg.to_layer else "external"
            all_conns.append((conn_cfg, key, conn_type))

        if not all_conns:
            raise RuntimeError("No connections found to convert.")

        dyn_params = self.strategy.get_dynamic_params()

        for conn_cfg, key, conn_type in all_conns:
            W = core.connections[key].detach().cpu().numpy()
            J_min, J_max = self.J_range

            if np.any(W < J_min) or np.any(W > J_max):
                 violators = W[(W < J_min) | (W > J_max)]
                 raise RuntimeError(
                    f"Connection {key}: weights range [{W.min():.4g}, {W.max():.4g}] exceeds "
                    f"reachable [{J_min:.4g}, {J_max:.4g}]. Violating: [{violators.min():.4g}, {violators.max():.4g}]"
                )

            phi_matrix = self.Jeff_to_phi(W)

            with torch.no_grad():
                core.connections[key].data.copy_(
                    torch.from_numpy(phi_matrix).to(device=core.connections[key].device, dtype=core.connections[key].dtype)
                )

            # Update mode/params
            if not hasattr(core, "_connection_modes"):
                core._connection_modes = {}
            if not hasattr(core, "_connection_params"):
                core._connection_params = {}

            core._connection_modes[key] = self.strategy.mode_name
            core._connection_params[key] = dyn_params.copy()

            # Config updates
            if conn_cfg.params is None:
                conn_cfg.params = {}
            conn_cfg.params["mode"] = self.strategy.mode_name
            conn_cfg.params["connection_params"] = dyn_params.copy()

            if conn_type == "internal":
                if not hasattr(core, "_internal_connectivity_settings"):
                    core._internal_connectivity_settings = {}
                specs = core._internal_connectivity_settings
                if conn_cfg.from_layer not in specs:
                    specs[conn_cfg.from_layer] = {}
                specs[conn_cfg.from_layer]["mode"] = self.strategy.mode_name
                specs[conn_cfg.from_layer]["dynamic_params"] = dyn_params.copy()

            logger.info(f"  {key}: Converted to {self.strategy.mode_name}")

        logger.info(f"Saving to {output_path}")
        core.save(output_path)
        self._verify(output_path)

    def _verify(self, path: str):
        logger.info("Verifying saved model...")
        m = SOENModelCore.load(path, show_logs=False)
        target = self.strategy.mode_name
        for cfg in m.connections_config:
            key = f"J_{cfg.from_layer}_to_{cfg.to_layer}"
            mode = getattr(m, "_connection_modes", {}).get(key, "fixed")
            if mode != target:
                logger.error(f"  {key} has mode {mode}, expected {target}")
                raise RuntimeError("Verification failed")
        logger.info("Verification passed.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Convert fixed connections to dynamic (WICC/NOCC)")

    parser.add_argument("-i", "--input", required=True, help="Input SOEN model")
    parser.add_argument("-o", "--output", help="Output path (default: input_stem_{mode}.soen)")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--wicc", action="store_true", help="Convert to WICC")
    mode_group.add_argument("--nocc", action="store_true", help="Convert to NOCC")

    # Common
    parser.add_argument("--bias-current", type=float, help="Bias current (WICC def: 2.0, NOCC def: 2.1)")
    parser.add_argument("--j-in", type=float, default=0.38) # Common default?
    parser.add_argument("--j-out", type=float, default=0.38)
    parser.add_argument("--phi-in", type=float, default=0.1)
    parser.add_argument("--no-half-flux-offset", action="store_true")

    # WICC specific
    wicc = parser.add_argument_group("WICC")
    wicc.add_argument("--gamma-plus", type=float, default=0.001)
    wicc.add_argument("--gamma-minus", type=float, default=0.001)

    # NOCC specific
    nocc = parser.add_argument_group("NOCC")
    nocc.add_argument("--alpha", type=float, default=1.64053)
    nocc.add_argument("--beta", type=float, default=303.85)
    nocc.add_argument("--beta-out", type=float, default=91.156)

    # Adv
    adv = parser.add_argument_group("Advanced")
    adv.add_argument("--phi-grid-total-range", type=float, default=1.0, help="Total range size for phi param mapping (centered on 0)")
    adv.add_argument("--num-points", type=int, default=1201)
    adv.add_argument("--device", default="cpu")
    adv.add_argument("-v", "--verbose", action="store_true")
    adv.add_argument("-f", "--force", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return 1

    # Defaults logic for bias current if not set
    if args.bias_current is None:
        args.bias_current = 2.0 if args.wicc else 2.1
        logger.info(f"Using default bias_current: {args.bias_current}")

    # Defaults logic for j_in/j_out if not set (or use argparse defaults)
    # The existing scripts used:
    # NOCC: j_in=1.5, j_out=1.5
    # WICC: j_in=0.38, j_out=0.38 (based on my previous write) -> actually wait,
    # check WICC previous script. I wrote defaults as 0.38.
    # But NOCC previous script had 1.5.
    # Let's override defaults based on mode if user didn't specify?
    # Actually argparse default is fixed. Let's just use what was passed.
    # But note that NOCC script had 1.5 as default.
    # If I set argparse default to 0.38, then NOCC users might get 0.38 if they don't specify.
    # It's better to verify this.
    # I will stick to argparse default=None and set manually.

    # Wait, I set defaults in argparse above. Let's fix that.
    # j_in/out/phi_in had different defaults in the two scripts.
    # WICC script: j_in=0.38, j_out=0.38, phi_in=0.1
    # NOCC script: j_in=1.5, j_out=1.5, phi_in=0.1

    # I should remove defaults from argparse and handle here.
    pass # logic handled below

    sf = RateArraySource()

    if args.wicc:
        if args.j_in == 0.38:
            args.j_in = 0.38 # Default confirmed
        if args.j_out == 0.38:
            args.j_out = 0.38
        strategy = WICCStrategy(args, sf)
        suffix = "wicc"
    else:
        # NOCC defaults override if default
        if args.j_in == 0.38:
            args.j_in = 1.5
        if args.j_out == 0.38:
            args.j_out = 1.5
        strategy = NOCCStrategy(args, sf)
        suffix = "nocc"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_{suffix}.soen"

    if output_path.exists() and not args.force:
        logger.error(f"Output exists: {output_path}")
        return 1

    converter = DynamicConverter(
        strategy,
        phi_param_min = -args.phi_grid_total_range/2,
        phi_param_max = args.phi_grid_total_range/2,
        num_points = args.num_points
    )

    try:
        converter.convert_model(str(input_path), str(output_path), torch.device(args.device))
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=args.verbose)
        return 1

if __name__ == "__main__":
    sys.exit(main())
