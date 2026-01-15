# src/soen_toolkit/utils/__init__.py


"""Utility functions for the SOEN package."""

from soen_toolkit.utils.logger import setup_logger
from soen_toolkit.utils.physical_mappings.soen_conversion_utils import PhysicalConverter

from .flux_matching_init import (
    FluxMatchingConfig,
    FluxMatchingResult,
    WeightUpdateMode,
    flux_matching_from_hdf5,
    run_flux_matching_iterations,
)
from .power_tracking import convert_energy_to_physical, convert_power_to_physical
from .pruning import (
    compute_obd_saliencies,
    estimate_fisher_diag,
    estimate_hessian_diag_hutchinson,
    post_training_prune,
    prune_connections_by_saliencies,
)
from .quantization import generate_uniform_codebook, snapped_copy


def merge_layer_configurations(*args, **kwargs):
    """Lazily import and dispatch to the layer merge helper."""
    from .merge_layers import merge_layer_configurations as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "FluxMatchingConfig",
    "FluxMatchingResult",
    "PhysicalConverter",
    "WeightUpdateMode",
    "convert_energy_to_physical",
    "convert_power_to_physical",
    "compute_obd_saliencies",
    "estimate_fisher_diag",
    "estimate_hessian_diag_hutchinson",
    "flux_matching_from_hdf5",
    "generate_uniform_codebook",
    "merge_layer_configurations",
    "post_training_prune",
    "prune_connections_by_saliencies",
    "run_flux_matching_iterations",
    "setup_logger",
    "snapped_copy",
]
