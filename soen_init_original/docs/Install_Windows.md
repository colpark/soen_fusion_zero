---
layout: default
title: Installing on Windows
---

# Installing SOEN-Toolkit on Windows

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Getting_Started.md">&#8592; Back to Installation Hub</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
</div>

<div style="background-color: #fff7ed; border-left: 4px solid #f97316; padding: 1em; margin: 1em 0;" markdown="1">
<strong>Important:</strong> Open PowerShell as <strong>Administrator</strong> for the entire installation.
</div>

## Installation (3 Steps)

### Step 1: Install uv and Prerequisites

```powershell
# Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart PowerShell as Admin, then install prerequisites
winget install Microsoft.VCRedist.2015+.x64  # Required for PyTorch
winget install graphviz
# Or download from: https://graphviz.org/download/
```

### Step 2: Install SOEN-Toolkit

```powershell
git clone https://github.com/greatsky-ai/soen-toolkit.git
cd soen-toolkit
uv sync
```

### Step 3: Activate Your Environment

```powershell
.venv\Scripts\activate
```

Your environment is now active! You can now:
- Run the GUI: `gui`
- Run training: `soen-train --config config.yaml`
- Use Python: `python your_script.py`
- Run tests: `pytest`
- Or just explore: `python -c "import soen_toolkit; print(soen_toolkit.__version__)"`

<strong>Tip:</strong> You can activate from anywhere by using the full path: `.venv\Scripts\activate` or `C:\full\path\to\soen-toolkit\.venv\Scripts\activate`

---

### Step 4: Register Jupyter Kernel (Optional but Recommended)

For seamless IDE integration (VS Code, PyCharm, Jupyter), register the kernel:

```powershell
python -m ipykernel install --user --name soen_toolkit --display-name "Python (soen-toolkit)"
```

After this, **restart your IDE** and the `soen_toolkit` kernel will appear automatically in your notebook environment selector.

---

## Verification

```powershell
python -c "import soen_toolkit; print(soen_toolkit.__version__)"
python -c "import torch; print(torch.__version__)"
```

<strong>GPU:</strong> For NVIDIA GPUs, install CUDA-enabled PyTorch
```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Common Issues

### DLL Error When Running Any Command

If you see `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`:

```powershell
# Install Visual C++ Redistributables
winget install Microsoft.VCRedist.2015+.x64

# Reinstall PyTorch with correct Windows build
uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### PowerShell "UnauthorizedAccess" Error

If you see `cannot be loaded because running scripts is disabled on this system`:

```powershell
# Allow local scripts to run
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Next Steps

<div style="background-color: #f0fdf4; border-left: 4px solid #059669; padding: 1em; margin: 1em 0;" markdown="1">

1. <strong>[Building Models](Building_Models.md)</strong> - Create your first model
2. <strong>Tutorials</strong> - `src/soen_toolkit/tutorial_notebooks/`

</div>

<strong>Other issues?</strong> See [GitHub Issues](https://github.com/greatsky-ai/soen-toolkit/issues)

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Getting_Started.md">&#8592; Back to Installation Hub</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Building_Models.md">Next: Building Models &#8594;</a>
</div>
