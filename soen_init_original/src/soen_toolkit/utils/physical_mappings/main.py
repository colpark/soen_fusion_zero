# FILEPATH: src/soen_toolkit/utils/physical_mappings/main.py

import os
import sys
import time

from flask import Flask, jsonify, render_template, request, send_from_directory

# Add necessary paths for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from soen_toolkit.physics import constants as phys_consts  # noqa: E402
from soen_toolkit.utils.physical_mappings.soen_conversion_utils import (  # noqa: E402
    PhysicalConverter,
)

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
converter = PhysicalConverter()


@app.route("/")
def index():
    return render_template("index.html", static_version=str(int(time.time())))


# Model converter page removed to keep only parameter converter UI


@app.route("/static/<path:path>")
def send_static(path):
    response = send_from_directory("static", path)
    try:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    except Exception:
        pass
    return response


@app.route("/get_constants", methods=["GET"])
def get_constants():
    """Return fundamental and default constants from centralized module."""
    constants = {
        "Phi_0": {
            "value": phys_consts.DEFAULT_PHI0,
            "latex": "\\Phi_0",
            "description": "Magnetic flux quantum (h/2e)",
            "unit": "Wb",
        },
        "h": {
            "value": phys_consts.PLANCK_H,
            "latex": "h",
            "description": "Planck constant",
            "unit": "J⋅s",
        },
        "e": {
            "value": phys_consts.ELEMENTARY_CHARGE_E,
            "latex": "e",
            "description": "Elementary charge",
            "unit": "C",
        },
        "h_over_2e": {
            "value": phys_consts.PLANCK_H / (2 * phys_consts.ELEMENTARY_CHARGE_E),
            "latex": "h/2e",
            "description": "Flux quantum calculation",
            "unit": "Wb",
        },
        "I_c_default": {
            "value": phys_consts.DEFAULT_IC,
            "latex": "I_c",
            "description": "Default critical current",
            "unit": "A",
        },
        "r_jj_default": {
            "value": phys_consts.DEFAULT_RJJ,
            "latex": "r_{jj}",
            "description": "Default junction resistance",
            "unit": "Ω",
        },
        "gamma_c_default": {
            "value": phys_consts.DEFAULT_GAMMA_C,
            "latex": "\\gamma_c",
            "description": "Default capacitance proportionality",
            "unit": "F/A",
        },
        "beta_c_default": {
            "value": phys_consts.DEFAULT_BETA_C,
            "latex": "\\beta_c",
            "description": "Default Stewart–McCumber parameter",
            "unit": "",
        },
        "omega_c_default": {
            "value": phys_consts.get_omega_c(),
            "latex": "\\omega_c",
            "description": "Default Josephson angular frequency",
            "unit": "rad/s",
        },
    }
    return jsonify(constants)


@app.route("/convert", methods=["POST"])
def convert_generic():
    """Generic conversion endpoint.

    Request JSON schema example:
    {
      "inputs": {"gamma_plus": [0.5, 1.0, 2.0]},
      "targets": ["beta_L", "L"]
    }
    Values can be scalars, lists, numpy-like nested lists, or serialized torch tensors (as lists).
    """
    try:
        data = request.json or {}
        inputs = data.get("inputs", {})
        targets = data.get("targets", [])
        if not isinstance(inputs, dict) or not isinstance(targets, list) or len(targets) == 0:
            return jsonify({"error": "Provide inputs (dict) and targets (list)"}), 400

        # Directly use converter.convert_many; it handles shape preservation for numpy/torch
        results = converter.convert_many(inputs, targets)

        # Make JSON friendly for numpy arrays
        def to_jsonable(v):
            try:
                import numpy as np  # local import

                if isinstance(v, np.ndarray):
                    return v.tolist()
            except Exception:
                pass
            try:
                import torch

                if isinstance(v, torch.Tensor):
                    return v.detach().cpu().tolist()
            except Exception:
                pass
            return v

        json_results = {k: to_jsonable(v) for k, v in results.items()}
        return jsonify({"results": json_results})
    except Exception as e:
        return jsonify({"error": f"{e}"}), 400


@app.route("/update_base_parameters", methods=["POST"])
def update_base_parameters():
    """Update base physical parameters and return all derived values."""
    data = request.json
    try:
        # Get and validate input parameters
        I_c = data.get("I_c")
        gamma_c = data.get("gamma_c")
        beta_c = data.get("beta_c")

        if any(x is not None and float(x) <= 0 for x in [I_c, gamma_c, beta_c]):
            return jsonify({"error": "All parameters must be positive"}), 400

        # Update converter parameters
        if I_c is not None:
            converter.I_c = float(I_c)
        if gamma_c is not None:
            converter.gamma_c = float(gamma_c)
        if beta_c is not None:
            converter.beta_c = float(beta_c)

        # Get updated parameters including derived ones
        params = converter.get_base_parameters()

        # Format for display with MathJax
        return jsonify(
            {
                "I_c": {"value": params["I_c"], "latex": "I_c"},
                "gamma_c": {"value": params["gamma_c"], "latex": "\\gamma_c"},
                "beta_c": {"value": params["beta_c"], "latex": "\\beta_c"},
                "c_j": {"value": params["c_j"], "latex": "c_j"},
                "r_jj": {"value": params["r_jj"], "latex": "r_{jj}"},
                "omega_c": {"value": params["omega_c"], "latex": "\\omega_c"},
                "omega_p": {"value": params["omega_p"], "latex": "\\omega_p"},
                "tau_0": {"value": params["tau_0"], "latex": "\\tau_0"},
                "V_j": {"value": params["V_j"], "latex": "V_j"},
            }
        )
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return jsonify({"error": f"Invalid input values: {e!s}"}), 400


@app.route("/convert_to_physical", methods=["POST"])
def convert_to_physical():
    """Convert dimensionless quantities to physical units."""
    data = request.json
    result = {}

    try:
        # Current
        if "i" in data and data["i"] is not None:
            i_val = float(data["i"])
            result["I"] = {
                "value": converter.dimensionless_to_physical_current(i_val),
                "latex": "I",
                "unit": "A",
            }

        # Flux
        if "phi" in data and data["phi"] is not None:
            phi_val = float(data["phi"])
            result["Phi"] = {
                "value": converter.dimensionless_to_physical_flux(phi_val),
                "latex": "\\Phi",
                "unit": "Wb",
            }

        # Inductance from beta_L
        if "beta_L" in data and data["beta_L"] is not None:
            beta_L_val = float(data["beta_L"])
            result["L"] = {
                "value": converter.dimensionless_to_physical_inductance(beta_L_val),
                "latex": "L",
                "unit": "H",
            }

        # From gamma_plus (alternative inductance representation)
        if "gamma_plus" in data and data["gamma_plus"] is not None:
            gamma_plus_val = float(data["gamma_plus"])
            beta_L = converter.gamma_to_beta_L(gamma_plus_val)
            result["L"] = {
                "value": converter.dimensionless_to_physical_inductance(beta_L),
                "latex": "L",
                "unit": "H",
            }

        # From gamma_minus and alpha (alternative time constant calculation)
        if "gamma_minus" in data and data["gamma_minus"] is not None:
            gamma_minus_val = float(data["gamma_minus"])
            tau = converter.gamma_minus_to_tau(gamma_minus_val)
            result["time_constant"] = {
                "value": tau / converter.omega_c,  # convert dimensionless tau to physical time
                "latex": "\\tau",
                "unit": "s",
            }

            # Calculate r_leak if we have beta_L (gamma_minus = alpha/beta_L)
            if "beta_L" in data and data["beta_L"] is not None:
                beta_L_val = float(data["beta_L"])
                alpha = gamma_minus_val * beta_L_val  # alpha = gamma_minus * beta_L
                result["r_leak"] = {
                    "value": converter.dimensionless_to_physical_resistance(alpha),
                    "latex": "r_{\\text{leak}}",
                    "unit": "Ω",
                }
            elif "gamma_plus" in data and data["gamma_plus"] is not None:
                gamma_plus_val = float(data["gamma_plus"])
                beta_L = converter.gamma_to_beta_L(gamma_plus_val)
                alpha = gamma_minus_val * beta_L  # alpha = gamma_minus * beta_L
                result["r_leak"] = {
                    "value": converter.dimensionless_to_physical_resistance(alpha),
                    "latex": "r_{\\text{leak}}",
                    "unit": "Ω",
                }

        # Time
        if "t_prime" in data and data["t_prime"] is not None:
            t_prime_val = float(data["t_prime"])
            result["t"] = {
                "value": converter.dimensionless_to_physical_time(t_prime_val),
                "latex": "t",
                "unit": "s",
            }

        # Resistance
        if "alpha" in data and data["alpha"] is not None:
            alpha_val = float(data["alpha"])
            result["r_leak"] = {
                "value": converter.dimensionless_to_physical_resistance(alpha_val),
                "latex": "r_{\\text{leak}}",
                "unit": "Ω",
            }

        # Flux quantum rate
        if "g_fq" in data and data["g_fq"] is not None:
            g_fq_val = float(data["g_fq"])
            result["G_fq"] = {
                "value": converter.dimensionless_to_physical_fq_rate(g_fq_val),
                "latex": "G_{fq}",
                "unit": "Hz",
            }

        return jsonify(result)
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return jsonify({"error": f"Conversion error: {e!s}"}), 400


@app.route("/convert_to_dimensionless", methods=["POST"])
def convert_to_dimensionless():
    """Convert physical quantities to dimensionless units."""
    data = request.json
    result = {}

    try:
        # Current
        if "I" in data and data["I"] is not None:
            I_val = float(data["I"])
            result["i"] = {
                "value": converter.physical_to_dimensionless_current(I_val),
                "latex": "i",
            }

        # Flux
        if "Phi" in data and data["Phi"] is not None:
            Phi_val = float(data["Phi"])
            result["phi"] = {
                "value": converter.physical_to_dimensionless_flux(Phi_val),
                "latex": "\\phi",
            }

        # Inductance (returns both beta_L and gamma)
        if "L" in data and data["L"] is not None:
            L_val = float(data["L"])
            beta_L = converter.physical_to_dimensionless_inductance(L_val)
            result["beta_L"] = {
                "value": beta_L,
                "latex": "\\beta_L",
            }
            result["gamma"] = {
                "value": converter.beta_L_to_gamma(beta_L),
                "latex": "\\gamma",
            }

        # Time
        if "t" in data and data["t"] is not None:
            t_val = float(data["t"])
            result["t_prime"] = {
                "value": converter.physical_to_dimensionless_time(t_val),
                "latex": "t'",
            }

        # Resistance
        if "r_leak" in data and data["r_leak"] is not None:
            r_leak_val = float(data["r_leak"])
            result["alpha"] = {
                "value": converter.physical_to_dimensionless_resistance(r_leak_val),
                "latex": "\\alpha",
            }

        # Flux quantum rate
        if "G_fq" in data and data["G_fq"] is not None:
            G_fq_val = float(data["G_fq"])
            result["g_fq"] = {
                "value": converter.physical_to_dimensionless_fq_rate(G_fq_val),
                "latex": "g_{fq}",
            }

        # Physical Tau to Dimensionless Gamma Minus
        if "tau_physical" in data and data["tau_physical"] is not None:
            tau_physical_val = float(data["tau_physical"])
            # Note: The method name is physical_tau_to_dimensionless_gamma_minus
            gamma_minus_from_tau = converter.physical_tau_to_dimensionless_gamma_minus(tau_physical_val)
            result["gamma_minus_from_tau"] = {  # Using a distinct key for clarity
                "value": gamma_minus_from_tau,
                "latex": "\\gamma_- \\text{ (from } \\tau_{\\mathrm{phys}} \\text{)}",  # Changed \text to \mathrm for subscript
            }

        return jsonify(result)
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return jsonify({"error": f"Conversion error: {e!s}"}), 400


@app.route("/calculate_derived", methods=["POST"])
def calculate_derived():
    """Calculate derived dimensionless parameters."""
    data = request.json
    result = {}

    try:
        # Calculate tau if we have both beta_L and alpha
        if "beta_L" in data and "alpha" in data:
            beta_L = float(data["beta_L"])
            alpha = float(data["alpha"])
            tau = converter.calculate_tau(beta_L, alpha)
            result["tau"] = {
                "value": tau,
                "latex": "\\tau",
            }
            # Also calculate gamma_minus from tau
            result["gamma_minus"] = {
                "value": converter.tau_to_gamma_minus(tau),
                "latex": "\\gamma_-",
            }

        # Calculate beta_L from gamma_plus
        if "gamma_plus" in data:
            gamma_plus = float(data["gamma_plus"])
            beta_L = converter.gamma_plus_to_beta_L(gamma_plus)
            result["beta_L"] = {
                "value": beta_L,
                "latex": "\\beta_L",
            }

            # If alpha is available, calculate tau and gamma_minus
            if "alpha" in data:
                alpha = float(data["alpha"])
                tau = converter.calculate_tau(beta_L, alpha)
                result["tau"] = {
                    "value": tau,
                    "latex": "\\tau",
                }
                result["gamma_minus"] = {
                    "value": converter.tau_to_gamma_minus(tau),
                    "latex": "\\gamma_-",
                }

        # Calculate alpha from gamma_plus and gamma_minus
        if "gamma_plus" in data and "gamma_minus" in data:
            gamma_plus = float(data["gamma_plus"])
            gamma_minus = float(data["gamma_minus"])
            alpha_val = converter.gamma_plus_gamma_minus_to_alpha(gamma_plus, gamma_minus)
            result["alpha"] = {
                "value": alpha_val,
                "latex": "\\alpha",
            }

            # Also calculate r_leak
            result["r_leak"] = {
                "value": converter.dimensionless_to_physical_resistance(alpha_val),
                "latex": "r_{\\text{leak}}",
                "unit": "Ω",
            }

        # Calculate alpha from gamma_minus and beta_L
        if "gamma_minus" in data and "beta_L" in data and "alpha" not in result:
            gamma_minus = float(data["gamma_minus"])
            beta_L = float(data["beta_L"])
            alpha_val = converter.gamma_minus_to_alpha_beta_L(gamma_minus, beta_L)
            result["alpha"] = {
                "value": alpha_val,
                "latex": "\\alpha",
            }

            # Also calculate r_leak for display
            result["r_leak"] = {
                "value": converter.dimensionless_to_physical_resistance(alpha_val),
                "latex": "r_{\\text{leak}}",
                "unit": "Ω",
            }

        # Calculate gamma_plus from beta_L
        if "beta_L" in data and "gamma_plus" not in data and "gamma_plus" not in result:
            beta_L = float(data["beta_L"])
            result["gamma_plus"] = {
                "value": converter.beta_L_to_gamma_plus(beta_L),
                "latex": "\\gamma_+",
            }

        return jsonify(result)
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return jsonify({"error": f"Calculation error: {e!s}"}), 400


if __name__ == "__main__":
    import socket

    def find_free_port(start_port=5001, max_attempts=100):
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                continue
        msg = f"Could not find a free port in range {start_port}-{start_port + max_attempts}"
        raise RuntimeError(msg)

    port = find_free_port()
    app.run(debug=True, port=port)
