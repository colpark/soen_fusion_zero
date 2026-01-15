#!/bin/bash
# SOEN Toolkit - Pure pip install (no conda)
# Usage: bash install-pip.sh

set -e

# Set install location
VENV_PATH="${1:-/home/idies/workspace/Temporary/dpark1/scratch/conda/conda_envs/soen}"

echo "Creating virtual environment at: $VENV_PATH"
python3.11 -m venv "$VENV_PATH"

echo "Activating..."
source "$VENV_PATH/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch with CUDA 12.4..."
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "Installing soen_toolkit..."
pip install -e .

echo ""
echo "Done! Activate with:"
echo "  source $VENV_PATH/bin/activate"
