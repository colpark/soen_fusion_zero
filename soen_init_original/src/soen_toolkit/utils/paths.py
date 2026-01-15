"""This file we will define filespaths used throughout the project directory.
filepath: src/soen_toolkit/utils/paths.py.
"""

import os
from pathlib import Path


def get_project_root() -> str:
    """Get the absolute path to the project root (soen-toolkit directory).
    Returns the path as a string, with forward slashes for consistency.
    """
    # First, check if environment variable is set (for deployment flexibility)
    if "SOEN_PROJECT_ROOT" in os.environ:
        root_path = Path(os.environ["SOEN_PROJECT_ROOT"]).resolve()
        return str(root_path).replace(os.sep, "/")

    # Get the directory containing this file
    current_dir = Path(__file__).resolve().parent

    # Navigate up until we find soen-toolkit directory OR find key project files
    current_path = current_dir
    max_levels = 10  # Prevent infinite loops
    levels_up = 0

    while levels_up < max_levels:
        # Check if we found the expected directory name
        if current_path.name == "soen-toolkit":
            return str(current_path).replace(os.sep, "/")

        # Alternative: Check if this looks like project root (has pyproject.toml and src)
        if (current_path / "pyproject.toml").exists() and (current_path / "src").exists():
            return str(current_path).replace(os.sep, "/")

        # Go up one level
        if current_path.parent == current_path:  # Reached filesystem root
            break
        current_path = current_path.parent
        levels_up += 1

    # More robust fallback: if we're in src/soen_toolkit/utils, go up 3 levels
    # This file is at src/soen_toolkit/utils/paths.py, so project root is 3 levels up
    fallback_path = Path(__file__).resolve().parent.parent.parent.parent
    if fallback_path.exists() and fallback_path.name == "soen-toolkit":
        return str(fallback_path).replace(os.sep, "/")

    # Additional fallback: look for the project root by checking for key files
    # Start from current file location and go up
    search_path = Path(__file__).resolve().parent
    for _i in range(6):  # Search up to 6 levels up
        if (search_path / "pyproject.toml").exists() and (search_path / "src").exists():
            return str(search_path).replace(os.sep, "/")
        if search_path.parent == search_path:  # Reached filesystem root
            break
        search_path = search_path.parent

    # Final fallback: use current working directory
    return str(Path.cwd()).replace(os.sep, "/")


# Get the project root once at module import
PROJECT_ROOT = get_project_root()


# Function to construct a path relative to project root
def get_project_path(*parts: str) -> str:
    """Construct an absolute path relative to the project root.

    Args:
        *parts: Path components to join with the project root

    Returns:
        Absolute path as a string, with forward slashes for consistency

    Example:
        >>> get_project_path("studies", "training", "models", "my_model.pth")
        '/path/to/soen-toolkit/studies/training/models/my_model.pth'

    """
    return os.path.join(PROJECT_ROOT, *parts).replace(os.sep, "/")


# ============================================================================
# Training Data Paths
# ============================================================================
GSC_data_filepath = get_project_path("data", "GSC", "digits_melspec_padded.hdf5")  # for now they are the same. Ideally we'd remove this unless we need it
GSC_PADDED_ZEROS_DATA_FILEPATH = get_project_path("data", "GSC", "digits_melspec_padded.hdf5")

# ============================================================================
# Training Related Paths
# ============================================================================
training_checkpoints = get_project_path("examples", "training_scripts", "checkpoints")
training_logs = get_project_path("examples", "training_scripts", "logs")
default_model_path = get_project_path("studies", "training", "gsc", "time_constants", "Tiny", "v3.pth")

# ============================================================================
# Figure Paths
# ============================================================================
figure_base = get_project_path("studies", "figures")
training_figures = get_project_path("studies", "figures", "training")
state_trajectory_figures = get_project_path("studies", "figures", "state_trajectories")
power_analysis_figures = get_project_path("studies", "figures", "power_analysis")
energy_usage_figures = get_project_path("studies", "figures", "energy_usage")

# Create directories if they don't exist
# for path in [figure_base, training_figures, state_trajectory_figures,
#             power_analysis_figures, energy_usage_figures]:
#     os.makedirs(path, exist_ok=True)

# ============================================================================
# Core Paths
# ============================================================================
BASE_RATE_ARRAY_PATH = get_project_path("src", "soen_toolkit", "core", "source_functions", "base_rate_array.soen")

# ============================================================================
# AWS S3 Paths
# this needs updating
# ============================================================================
s3GSC10Class100SeqLen_filepath = "s3://trainingdata10classgsc/Training_Data/processed_speech_digits_100.h5"
s3_default_bucket = "trainingdata10classgsc"
