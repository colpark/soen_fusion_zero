# FILEPATH: src/soen_toolkit/core/mixins/summary.py

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch

if TYPE_CHECKING:
    from torch import nn

    from soen_toolkit.core.configs import ConnectionConfig, LayerConfig


class SummaryMixin:
    """Mixin providing model summary and visualization methods.

    This mixin expects to be used with a class that implements ModelProtocol.
    """

    if TYPE_CHECKING:
        # Attributes expected from the composed class (typically SOENModelCore)
        layers_config: list[LayerConfig]
        connections_config: list[ConnectionConfig]
        layers: nn.ModuleList
        connections: nn.ParameterDict
        connection_masks: dict[str, torch.Tensor]
        num_layers: int

    @staticmethod
    def _is_notebook() -> bool:
        """Detect if running in a Jupyter notebook environment."""
        try:
            from IPython import get_ipython

            ipy = get_ipython()
            if ipy is None:
                return False
            # Check if it's a notebook kernel (not just IPython shell)
            # ipykernel will be in the kernel class name or module
            kernel_type = str(type(ipy))
            return "ipykernel" in kernel_type.lower()
        except (ImportError, AttributeError):
            return False

    def summary(
        self,
        return_df: bool = False,
        print_summary: bool = True,
        create_histograms: bool = False,
        verbose: bool = False,
        dpi: int = 300,
        notebook_view: bool | None = None,
    ) -> pd.DataFrame | None:
        layer_records = self._create_layer_records(verbose)
        layer_df = pd.DataFrame(layer_records)

        conn_records = self._create_connection_records()
        conn_df = pd.DataFrame(conn_records)

        if print_summary:
            # Auto-detect notebook if not explicitly specified
            if notebook_view is None:
                notebook_view = self._is_notebook()

            if notebook_view:
                self._print_notebook_summary(layer_df, conn_df)
            else:
                self._print_model_summary(layer_df, conn_df)

        if create_histograms:
            self._create_parameter_histograms(dpi)

        if return_df:
            return layer_df
        return None

    def compute_summary(self) -> dict[str, Any]:
        import numpy as np

        layer_df = self.summary(return_df=True, print_summary=False, verbose=False)
        if layer_df is None:
            layer_df = pd.DataFrame()

        conn_records = self._create_connection_records()

        # Use global connection mask (not weight matrix) to accurately count connections
        try:
            G = self.get_global_connection_mask().detach().cpu().numpy()
        except Exception:
            G = None

        layer_ids = [cfg.layer_id for cfg in self.layers_config]
        dims = {cfg.layer_id: cfg.params.get("dim", 0) for cfg in self.layers_config}
        degrees = {}
        fan_layer_stats = {}
        total_nodes = sum(dims.values())
        if G is not None and G.size > 0 and total_nodes == G.shape[0] == G.shape[1]:
            sorted_ids = sorted(layer_ids)
            offsets = {}
            run = 0
            for lid in sorted_ids:
                offsets[lid] = run
                run += dims[lid]

            for lid in sorted_ids:
                o = offsets[lid]
                d = dims[lid]
                rows = slice(o, o + d)
                sub_in = G[rows, :]
                sub_out = G[:, rows]
                in_deg = int(np.count_nonzero(sub_in))
                out_deg = int(np.count_nonzero(sub_out))
                degrees[lid] = {"in": in_deg, "out": out_deg}

                try:
                    fan_in_vec = np.count_nonzero(G[:, rows], axis=0)
                    fan_out_vec = np.count_nonzero(G[rows, :], axis=1)
                    fan_layer_stats[lid] = {
                        "fan_in": {
                            "min": int(fan_in_vec.min()),
                            "median": float(np.median(fan_in_vec)),
                            "max": int(fan_in_vec.max()),
                        },
                        "fan_out": {
                            "min": int(fan_out_vec.min()),
                            "median": float(np.median(fan_out_vec)),
                            "max": int(fan_out_vec.max()),
                        },
                    }
                except Exception:
                    pass

        total_params = sum(p.numel() for p in self.parameters())
        # Mask-aware trainable parameter counting:
        # - For connection weights, count ones in the mask if learnable
        # - For other learnable parameters, count full numel
        learnable_params = self._compute_mask_aware_trainable_parameters()
        nz_params = sum((p != 0).sum().item() for p in self.parameters())
        kpis = {
            "layers": len(self.layers_config),
            "connections": len(self.connections),
            "parameters": total_params,
            "trainable_parameters": learnable_params,
            "nonzero_parameters": nz_params,
        }

        return {
            "kpis": kpis,
            "layers": layer_df.to_dict(orient="records") if isinstance(layer_df, pd.DataFrame) else [],
            "connections": conn_records,
            "degrees": degrees,
            "fan": fan_layer_stats,
        }

    def _compute_mask_aware_trainable_parameters(self) -> int:
        """Return the trainable parameter count using masks for connectivity.

        Rules:
        - External connections in self.connections: if requires_grad, add mask.sum();
          fallback to param.numel() when mask missing.
        - Intra-layer connectivity (v2): if layer.connectivity.weight.requires_grad and
          the corresponding J_i_to_i (or legacy internal_i) mask exists in
          self.connection_masks, add that mask.sum()
          otherwise fallback to numel.
        - All other learnable parameters (non-connectivity): add full numel.
        """
        import re as _re

        # Total learnable parameters (unmasked baseline)
        total_learnable_unmasked = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Track how many of those belong to connection weights (to avoid double count)
        connection_unmasked_total = 0
        connection_masked_total = 0

        # External connections (including possible J_i_to_i in legacy/v1)
        for key, param in self.connections.items():
            if not param.requires_grad:
                continue
            connection_unmasked_total += int(param.numel())
            mask = self.connection_masks.get(key, None)
            if mask is not None:
                try:
                    connection_masked_total += int(mask.sum().item())
                except Exception:
                    connection_masked_total += int(mask.numel())
            else:
                connection_masked_total += int(param.numel())

        # Intra-layer connectivity in v2 lives inside each layer (not in self.connections)
        present_conn_keys = set(self.connections.keys())
        # Regex helpers to match J_i_to_i and internal_i
        pat_internal = _re.compile(r"^internal_(?P<lid>\d+)$")
        pat_j_self = _re.compile(r"^J_(?P<lid>\d+)_to_(?P<lid2>\d+)$")

        # Precompute masks available per layer id (for self-connections)
        mask_by_layer_self: dict[int, torch.Tensor] = {}
        for mkey, m in getattr(self, "connection_masks", {}).items():
            m_j = pat_j_self.match(mkey)
            if m_j:
                lid = int(m_j.group("lid"))
                lid2 = int(m_j.group("lid2"))
                if lid == lid2:
                    mask_by_layer_self[lid] = m
                continue
            m_i = pat_internal.match(mkey)
            if m_i:
                lid = int(m_i.group("lid"))
                mask_by_layer_self[lid] = m

        for cfg in self.layers_config:
            lid = cfg.layer_id
            # If an explicit connection param exists for J_lid_to_lid, it was already counted above
            j_key = f"J_{lid}_to_{lid}"
            legacy_key = f"internal_{lid}"
            if (j_key in present_conn_keys) or (legacy_key in present_conn_keys):
                continue
            # Layer-level internal connectivity
            try:
                idx = next(i for i, c in enumerate(self.layers_config) if c.layer_id == lid)
                layer = self.layers[idx]
            except Exception:
                continue
            conn_mod = getattr(layer, "connectivity", None)
            if conn_mod is None:
                continue
            weight = getattr(conn_mod, "weight", None)
            if not isinstance(weight, torch.Tensor) or not weight.requires_grad:
                continue
            # Count this internal connectivity via mask ones if available
            connection_unmasked_total += int(weight.numel())
            mask = mask_by_layer_self.get(lid)
            if mask is not None:
                try:
                    connection_masked_total += int(mask.sum().item())
                except Exception:
                    connection_masked_total += int(mask.numel())
            else:
                connection_masked_total += int(weight.numel())

        # Non-connection learnables are the remainder
        non_connection_learnable = int(total_learnable_unmasked - connection_unmasked_total)
        return int(non_connection_learnable + connection_masked_total)

    def _create_layer_records(self, verbose: bool) -> list[dict]:
        layer_records = []
        import torch

        for cfg in self.layers_config:
            idx = next(i for i, c in enumerate(self.layers_config) if c.layer_id == cfg.layer_id)
            layer = self.layers[idx]

            # Get solver name from config or layer attribute
            solver_display = cfg.params.get("solver") or getattr(layer, "_solver_name", None) or "-"

            # Get source function display name
            layer_source_attr = getattr(layer, "source_function", None) or getattr(layer, "_source_function", None)
            if layer_source_attr is None:
                source_func_display = "-"
            else:
                source_func_display = (
                    cfg.params.get("source_func") or cfg.params.get("source_func_type") or getattr(layer_source_attr, "name", None) or layer_source_attr.__class__.__name__.replace("Source", "")
                )

            base_rec = {
                "Layer ID": cfg.layer_id,
                "Layer Type": cfg.layer_type,
                "Description": cfg.description,
                "Dimension": cfg.params["dim"],
                "Solver": solver_display,
                "Source Function": source_func_display,
            }

            constraint_map = {pname: pconf for pname, pconf in cfg.params.items() if isinstance(pconf, dict)}

            if verbose:
                has_params = False
                for name, param in layer.named_parameters():
                    has_params = True
                    display_name = name.removeprefix("log_")
                    rec = base_rec.copy()
                    rec.update(
                        {
                            "Param Name": display_name,
                            "Shape": str(tuple(param.shape)),
                            "Num Elements": param.numel(),
                            "Requires Grad": param.requires_grad,
                        },
                    )

                    param_data = param.detach().cpu()
                    if name.startswith("log_"):
                        try:
                            param_data = param_data.exp()
                        except Exception:
                            param_data = torch.exp(param_data)
                    rec["Statistics"] = f"mean: {param_data.mean().item():.4f}, std: {param_data.std().item():.4f}"

                    base_param_name = name
                    if "_" in name and name[-2] == "_" and name[-1].isdigit():
                        base_param_name = name[:-2]
                    elif name.startswith("log_") and "_" in name[4:] and name[-2] == "_" and name[-1].isdigit():
                        base_param_name = "log_" + name[4:-2]

                    constraint_key = None
                    if name in constraint_map:
                        constraint_key = name
                    elif base_param_name in constraint_map:
                        constraint_key = base_param_name

                    cons = None
                    if constraint_key:
                        cons = constraint_map[constraint_key]
                    elif hasattr(layer, "_param_registry"):
                        # Fallback to registry constraints if available
                        reg_def = layer._param_registry._defs.get(base_param_name)
                        if reg_def and reg_def.constraint:
                            c = reg_def.constraint
                            cons = {"min": c.min, "max": c.max}

                    if cons:
                        rec["Constraints"] = f"min: {cons.get('min', '-inf')}, max: {cons.get('max', 'inf')}"
                        # Only show stats if they are in the config dict (registry doesn't track target stats)
                        if "mean" in cons:
                            rec["Constraints"] += f", mean: {cons.get('mean', 'N/A')}"
                        if "std" in cons:
                            rec["Constraints"] += f", std: {cons.get('std', 'N/A')}"
                    else:
                        rec["Constraints"] = ""

                    layer_records.append(rec)

                if not has_params:
                    rec = base_rec.copy()
                    rec.update(
                        {
                            "Param Name": "",
                            "Shape": "",
                            "Num Elements": 0,
                            "Requires Grad": False,
                            "Statistics": "",
                            "Constraints": "",
                        },
                    )
                    layer_records.append(rec)
            else:
                rec = base_rec.copy()
                total_params = sum(p.numel() for p in layer.parameters())
                rec["Total Parameters"] = total_params
                layer_records.append(rec)

        return layer_records

    def _create_connection_records(self) -> list[dict]:
        conn_records = []
        for idx, (key, param) in enumerate(self.connections.items()):
            num_params = param.numel()
            nonzero = (param != 0).sum().item()
            sparsity = 100 * (1 - nonzero / num_params)
            param_data = param.detach().cpu()
            stats = f"mean: {param_data.mean().item():.4f}, std: {param_data.std().item():.4f}, min: {param_data.min().item():.4f}, max: {param_data.max().item():.4f}"
            if nonzero < num_params:
                nonzero_data = param_data[param_data != 0]
                if nonzero_data.numel() > 0:
                    nonzero_stats = f"nonzero_mean: {nonzero_data.mean().item():.4f}, nonzero_std: {nonzero_data.std().item():.4f}"
                    stats += f", {nonzero_stats}"

            # Prefer runtime connection params tracked on the model (includes computed values)
            mode_str = "fixed"
            conn_mode = getattr(self, "_connection_modes", {}).get(key)
            conn_params = getattr(self, "_connection_params", {}).get(key, {})

            # Fallback to parsing from config when runtime maps are not populated
            if conn_mode is None and idx < len(self.connections_config):
                conn_cfg = self.connections_config[idx]
                params_from_cfg = conn_cfg.params or {}
                from soen_toolkit.core.utils.connection_ops import parse_connection_config

                _mode, _params = parse_connection_config(params_from_cfg)
                conn_mode = _mode
                conn_params = _params

            # Format mode string and include WICC j_out stats when available
            if conn_mode == "fixed" or conn_mode is None:
                mode_str = "fixed"
            elif conn_mode == "WICC":
                source = conn_params.get("source_func", "RateArray")
                gamma_plus = conn_params.get("gamma_plus", 1e-3)
                bias_current = conn_params.get("bias_current", 2.0)
                mode_str = f"WICC (src: {source}, γ⁺: {gamma_plus:.1e}, bias: {bias_current:.2f})"
            elif conn_mode == "NOCC":
                source = conn_params.get("source_func", "RateArray")
                alpha = conn_params.get("alpha", 1.64053)
                beta = conn_params.get("beta", 303.85)
                beta_out = conn_params.get("beta_out", 91.156)
                ib = conn_params.get("bias_current", 2.1)
                mode_str = f"NOCC (src: {source}, α: {alpha:.3g}, β: {beta:.3g}, β_out: {beta_out:.3g}, ib: {ib:.3g})"

            rec = {
                "Connection": key,
                "Mode": mode_str,
                "Learnable": param.requires_grad,
                "Total Parameters": num_params,
                "Non-zero Parameters": nonzero,
                "Sparsity (%)": round(sparsity, 1),
                "Statistics": stats,
            }

            # Add computed WICC j_out stats when available (hidden param)
            try:
                if conn_mode == "WICC":
                    j_out_val = conn_params.get("j_out")
                    if isinstance(j_out_val, torch.Tensor) and j_out_val.numel() > 0:
                        v = j_out_val.detach().cpu()
                        rec["WICC j_out (min/med/max)"] = f"{v.min().item():.3g} / {v.median().item():.3g} / {v.max().item():.3g}"
            except Exception:
                pass

            # Add connection constraints if present
            constraint_str = self._format_connection_constraints(key, param)
            if constraint_str:
                rec["Constraints"] = constraint_str

            conn_records.append(rec)
        return conn_records

    def _format_connection_constraints(self, key: str, param: torch.Tensor) -> str:
        """Format constraint information for a connection."""
        parts = []

        # Check for per-element constraint matrices (polarity constraints)
        min_mats = getattr(self, "connection_constraint_min_matrices", {})
        max_mats = getattr(self, "connection_constraint_max_matrices", {})

        if key in min_mats or key in max_mats:
            # Per-element constraints exist (from polarity or custom constraints)
            min_mat = min_mats.get(key)
            max_mat = max_mats.get(key)

            # Check if it's a polarity pattern (some cols are >=0 or <=0)
            has_excitatory = False
            has_inhibitory = False

            if min_mat is not None:
                min_vals = min_mat.detach().cpu()
                # Check for excitatory columns (min >= 0)
                if (min_vals >= 0).any():
                    has_excitatory = True

            if max_mat is not None:
                max_vals = max_mat.detach().cpu()
                # Check for inhibitory columns (max <= 0)
                if (max_vals <= 0).any():
                    has_inhibitory = True

            if has_excitatory and has_inhibitory:
                parts.append("polarity (mixed E/I)")
            elif has_excitatory:
                parts.append("polarity (excitatory)")
            elif has_inhibitory:
                parts.append("polarity (inhibitory)")
            elif min_mat is not None or max_mat is not None:
                # Generic per-element constraints
                parts.append("per-element bounds")

        # Check for scalar constraints
        scalar_constraints = getattr(self, "connection_constraints", {}).get(key, {})
        if scalar_constraints:
            min_val = scalar_constraints.get("min", -float("inf"))
            max_val = scalar_constraints.get("max", float("inf"))
            if min_val > -float("inf") or max_val < float("inf"):
                bounds = []
                if min_val > -float("inf"):
                    bounds.append(f"min: {min_val:.4g}")
                if max_val < float("inf"):
                    bounds.append(f"max: {max_val:.4g}")
                parts.append(", ".join(bounds))

        return "; ".join(parts) if parts else ""

    def _print_model_summary(self, layer_df: pd.DataFrame, conn_df: pd.DataFrame) -> None:
        """Print summary to stdout (for non-notebook environments)."""
        total_layers = self.num_layers
        total_model_params = sum(p.numel() for p in self.parameters())

        print("\n" + "=" * 80)  # noqa: T201
        print("SOEN Model Summary")  # noqa: T201
        print("=" * 80)  # noqa: T201
        print(f"Total Layers: {total_layers}")  # noqa: T201
        print(f"Total Model Parameters: {total_model_params:,}")  # noqa: T201
        print()  # noqa: T201

        if not layer_df.empty:
            print("Layer Summary:")  # noqa: T201
            print("-" * 80)  # noqa: T201
            print(layer_df.to_string(index=False))  # noqa: T201
            print()  # noqa: T201

        if not conn_df.empty:
            print("Connections Summary:")  # noqa: T201
            print("-" * 80)  # noqa: T201
            print(conn_df.to_string(index=False))  # noqa: T201
            print()  # noqa: T201

        print("=" * 80)  # noqa: T201

    def _print_notebook_summary(self, layer_df: pd.DataFrame, conn_df: pd.DataFrame) -> None:
        """Display summary with nice markdown rendering for Jupyter notebooks."""
        from IPython.display import Markdown, display

        total_layers = self.num_layers
        total_model_params = sum(p.numel() for p in self.parameters())

        # Build markdown output
        md_lines = []
        md_lines.append("## SOEN Model Summary\n")
        md_lines.append(f"**Total Layers:** {total_layers}  ")
        md_lines.append(f"**Total Model Parameters:** {total_model_params:,}\n")

        md_lines.append("### Layer Summary\n")
        md_lines.append(layer_df.to_markdown(index=False))
        md_lines.append("\n")

        md_lines.append("### Connections Summary\n")
        if not conn_df.empty:
            md_lines.append(conn_df.to_markdown(index=False))
        else:
            md_lines.append("*No connection parameters.*")

        display(Markdown("\n".join(md_lines)))

    def _create_parameter_histograms(self, dpi: int) -> None:
        import os

        import matplotlib.pyplot as plt
        import seaborn as sns

        hist_dir = "summaries/parameter_histograms"
        os.makedirs(hist_dir, exist_ok=True)

        for cfg in self.layers_config:
            idx = next(i for i, c in enumerate(self.layers_config) if c.layer_id == cfg.layer_id)
            layer = self.layers[idx]

            for name, param in layer.named_parameters():
                data = param.detach().cpu().numpy().flatten()
                filename = os.path.join(hist_dir, f"layer_{cfg.layer_id}_{name}_distribution.png")

                plt.figure()
                sns.histplot(data, bins=50, kde=True)
                plt.title(f"Layer {cfg.layer_id} - {name} Distribution")
                plt.xlabel("Parameter Value")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(filename, dpi=dpi)
                plt.close()

        conn_hist_dir = "summaries/connection_histograms"
        os.makedirs(conn_hist_dir, exist_ok=True)

        for key, param in self.connections.items():
            data = param.detach().cpu().numpy().flatten()
            nonzero_only = key.startswith("J_") and (param != 0).sum().item() < param.numel()
            if nonzero_only:
                data = data[data != 0]

            filename = os.path.join(conn_hist_dir, f"connection_{key}_distribution.png")

            plt.figure()
            sns.histplot(data, bins=50, kde=True)
            plt.title(
                f"Connection {key} Distribution" + (" (non-zeros only)" if nonzero_only else ""),
            )
            plt.xlabel("Parameter Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(filename, dpi=dpi)
            plt.close()

    def get_global_connection_matrix(self) -> torch.Tensor:
        from soen_toolkit.utils.model_tools import create_global_connection_matrix

        return create_global_connection_matrix(self)

    def get_global_connection_mask(self) -> torch.Tensor:
        from soen_toolkit.utils.model_tools import create_global_connection_mask

        return create_global_connection_mask(self)
