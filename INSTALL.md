# SOEN Toolkit Installation Guide

## Quick Start with Conda (Recommended)

### SciServer / Custom Prefix Install
```bash
cd soenre

# GPU server
conda env create -f environment.yml --prefix /home/idies/workspace/Temporary/dpark1/scratch/conda/conda_envs/soen

# OR CPU-only server
conda env create -f environment-cpu.yml --prefix /home/idies/workspace/Temporary/dpark1/scratch/conda/conda_envs/soen

# Activate
conda activate /home/idies/workspace/Temporary/dpark1/scratch/conda/conda_envs/soen
```

### Standard Install (named environment)
```bash
cd soenre
conda env create -f environment.yml      # GPU
conda env create -f environment-cpu.yml  # CPU only
conda activate soen
```

### Verify
```bash
python -c "from soen_toolkit.core import SOENModelCore; print('OK')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Alternative: pip-only (Remote Server)

### Option 1: Install from source (Recommended)

```bash
# Clone or copy the repository to your server
# Then install in editable mode:

cd soen_orig
pip install -e .
```

### Option 2: Install with minimal dependencies (headless server)

```bash
cd soen_orig
pip install -r requirements-server.txt
pip install -e . --no-deps
```

### Option 3: Build wheel locally, transfer to server

On your local machine:
```bash
cd soen_orig
pip install build
python -m build --wheel
# Creates dist/soen_toolkit-0.1.13-py3-none-any.whl
```

Transfer the wheel to server:
```bash
scp dist/soen_toolkit-*.whl user@server:/path/to/destination/
```

On the server:
```bash
pip install soen_toolkit-0.1.13-py3-none-any.whl
```

---

## Installation Options

| Method | Command | Use Case |
|--------|---------|----------|
| Full install | `pip install -e .` | Development, all features |
| Server install | `pip install -r requirements-server.txt && pip install -e . --no-deps` | Headless training |
| Wheel install | `pip install soen_toolkit-*.whl` | Production deployment |

---

## Verify Installation

```python
# Test basic import
python -c "from soen_toolkit.core import SOENModelCore; print('OK')"

# Test model building
python -c "from soen_toolkit.core.model_yaml import build_model_from_yaml; print('OK')"
```

---

## Common Issues

### PyQt6 fails on headless server
Use `requirements-server.txt` which excludes GUI dependencies.

### JAX/CUDA issues
Install JAX with appropriate CUDA support:
```bash
# CPU only
pip install jax

# CUDA 12
pip install jax[cuda12]
```

### PyTorch CUDA
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install with CUDA if needed
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Environment Setup

### Using venv (standard)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -e .
```

### Using conda
```bash
conda create -n soen python=3.11
conda activate soen
pip install -e .
```

---

## Requirements Files

| File | Description |
|------|-------------|
| `environment.yml` | Conda env with GPU support |
| `environment-cpu.yml` | Conda env, CPU only |
| `requirements.txt` | Full pip dependencies (includes GUI) |
| `requirements-server.txt` | Minimal pip for headless servers |
| `pyproject.toml` | Official package definition |
