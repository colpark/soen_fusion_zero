---
layout: default
title: Installing on macOS
---

# Installing SOEN-Toolkit on macOS

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Getting_Started.md">&#8592; Back to Installation Hub</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
</div>

## Installation (4 Steps)

### Step 1: Install uv and Graphviz

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH - Apple Silicon
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

# Or for Intel Macs
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

# Install uv and Graphviz
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install graphviz
```

### Step 2: Install SOEN-Toolkit

```bash
git clone https://github.com/greatsky-ai/soen-toolkit.git
cd soen-toolkit
uv sync
```

### Step 3: Activate Your Environment

```bash
source .venv/bin/activate
```

Your environment is now active! You can now:
- Run the GUI: `gui`
- Run training: `soen-train --config config.yaml`
- Use Python: `python your_script.py`
- Run tests: `pytest`
- Or just explore: `python -c "import soen_toolkit; print(soen_toolkit.__version__)"`

<strong>Tip:</strong> You can activate from anywhere by using the full path: `source /full/path/to/soen-toolkit/.venv/bin/activate`

### Step 4: Register Jupyter Kernel (Optional but Recommended)

For seamless IDE integration (VS Code, PyCharm, Jupyter), register the kernel:

```bash
python -m ipykernel install --user --name soen_toolkit --display-name "Python (soen-toolkit)"
```

After this, **restart your IDE** and the `soen_toolkit` kernel will appear automatically in your notebook environment selector.

---

## Verification

```bash
python -c "import soen_toolkit; print(soen_toolkit.__version__)"
python -c "import torch; print(torch.__version__)"
```

<strong>GPU:</strong> macOS uses MPS (Metal), not CUDA
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

---

## Next Steps

<div style="background-color: #f0fdf4; border-left: 4px solid #059669; padding: 1em; margin: 1em 0;" markdown="1">

1. <strong>[Building Models](Building_Models.md)</strong> - Create your first model
2. <strong>Tutorials</strong> - `src/soen_toolkit/tutorial_notebooks/`

</div>

<strong>Issues?</strong> See [GitHub Issues](https://github.com/greatsky-ai/soen-toolkit/issues)

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Getting_Started.md">&#8592; Back to Installation Hub</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Building_Models.md">Next: Building Models &#8594;</a>
</div>
