---
layout: default
title: WSL Setup Guide
---

# WSL Setup Guide

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Install_WSL2.md">&#8592; Back to WSL Installation</a>
</div>

This guide will help you set up Windows Subsystem for Linux (WSL) on Windows, including checking if it's installed, configuring your user account, understanding file paths, and getting started.

---

## Checking if WSL is Already Installed

Before installing WSL, check if it's already available on your system.

### Method 1: Check via PowerShell or Command Prompt

Open PowerShell or Command Prompt (as Administrator) and run:

```powershell
wsl --status
```

If WSL is installed, you'll see version information. If not, you'll get an error message.

### Method 2: Check Installed Distributions

List all installed WSL distributions:

```powershell
wsl --list --verbose
```

Or use the shorter form:

```powershell
wsl -l -v
```

If you see any distributions listed (like `Ubuntu`, `Ubuntu-22.04`, `Debian`, etc.), WSL is installed and you have at least one Linux distribution available.

---

## Installing WSL (if not already installed)

If WSL is not installed, install it using one of these methods:

### Method 1: Simple Installation (Recommended)

Run PowerShell as Administrator and execute:

```powershell
wsl --install
```

This command will:
- Enable the required Windows features
- Install WSL2 (the latest version)
- Install Ubuntu by default

After installation, **restart your computer** when prompted.

### Method 2: Manual Installation

If the simple method doesn't work, enable WSL manually:

```powershell
# Enable WSL feature
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart your computer, then set WSL2 as default
wsl --set-default-version 2
```

Then install a Linux distribution from the Microsoft Store (search for "Ubuntu" or "Debian").

---

## Setting Up Your Username and Password

After installing WSL and a Linux distribution, you'll need to set up your user account.

### First Launch

1. Launch your Linux distribution from the Start menu, or run:
   ```powershell
   wsl
   ```

2. On first launch, you'll be prompted to:
   - **Create a username**: Choose a username (lowercase letters, numbers, hyphens, and underscores only)
   - **Create a password**: Enter a password (characters won't appear as you type - this is normal)
   - **Confirm password**: Enter the password again

### Important Notes About the Password

- Your password is stored securely within WSL
- You'll use this password when running `sudo` commands (for administrative tasks)
- If you forget your password, you can reset it (see below)

### Changing Your Password Later

If you need to change your password later:

```bash
passwd
```

Enter your current password, then your new password twice.

### Resetting a Forgotten Password

If you forget your password:

1. Open PowerShell as Administrator
2. Set your distribution as default (if needed):
   ```powershell
   wsl -d Ubuntu
   ```
   (Replace `Ubuntu` with your distribution name)

3. Reset the root password:
   ```powershell
   wsl --user root
   ```

4. Inside WSL, reset your user password:
   ```bash
   passwd your_username
   ```

5. Exit and restart WSL normally.

---

## Understanding File Paths in WSL

WSL provides seamless file system integration between Windows and Linux, but understanding how paths work is important.

### Windows Paths from WSL

Access Windows files from within WSL:

```bash
# Windows C: drive is mounted at /mnt/c
cd /mnt/c/Users/YourUsername/Documents

# Windows D: drive is mounted at /mnt/d
cd /mnt/d/Projects

# List files in Windows directory
ls /mnt/c/Users/YourUsername
```

**Important**: Use forward slashes (`/`) in WSL, not backslashes (`\`).

### WSL Paths from Windows

Access WSL files from Windows:

```powershell
# Access WSL home directory
cd \\wsl$\Ubuntu\home\your_username

# Or use the full path
explorer.exe \\wsl$\Ubuntu\home\your_username
```

Replace `Ubuntu` with your distribution name.

### Home Directory

Your Linux home directory is typically:
```bash
/home/your_username
```

You can reference it with:
```bash
cd ~
# or
cd $HOME
```

### Recommended File Locations

For SOEN-Toolkit and other projects:

**Option 1: Work in WSL file system** (Recommended for better performance)
```bash
cd ~
mkdir projects
cd projects
git clone https://github.com/greatsky-ai/soen-toolkit.git
```

**Option 2: Work in Windows file system**
```bash
cd /mnt/c/Users/YourUsername/Documents
git clone https://github.com/greatsky-ai/soen-toolkit.git
```

**Performance Tip**: Files stored in the WSL file system (`~`) are typically faster than files in `/mnt/c`. Use WSL file system for active development.

---

## Starting WSL

There are several ways to start and use WSL:

### Method 1: From Start Menu

1. Search for your Linux distribution (e.g., "Ubuntu") in the Start menu
2. Click to launch
3. A terminal window opens with your Linux environment

### Method 2: From Command Prompt or PowerShell

```powershell
wsl
```

This launches your default Linux distribution.

### Method 3: Launch Specific Distribution

```powershell
wsl -d Ubuntu
```

Replace `Ubuntu` with your distribution name.

### Method 4: Run a Command Directly

Run a Linux command without opening an interactive shell:

```powershell
wsl ls -la
wsl python --version
```

### Method 5: From Windows Terminal

1. Install Windows Terminal from Microsoft Store (if not already installed)
2. Open Windows Terminal
3. Click the dropdown arrow next to the new tab button
4. Select your Linux distribution

---

## Default User and Distribution

### Set Default Distribution

If you have multiple distributions installed:

```powershell
wsl --set-default Ubuntu
```

Replace `Ubuntu` with your preferred distribution name.

### Set Default User

To automatically log in as a specific user:

```powershell
# First, set default distribution
wsl --set-default Ubuntu

# Then set default user for that distribution
ubuntu config --default-user your_username
```

For other distributions, the command varies. Check your distribution's documentation.

---

## Useful System Packages

After initial WSL setup, install useful development tools:

```bash
sudo apt update
sudo apt install -y htop tmux unzip wget curl
```

**What these do:**
- `htop` - Interactive process viewer (better than `top`)
- `tmux` - Terminal multiplexer for managing multiple sessions
- `unzip` - Extract zip archives
- `wget` - Download files from the web
- `curl` - Transfer data (may already be installed)

These are optional but recommended for a smoother development experience.

---

## Basic WSL Commands Reference

Common WSL management commands:

```powershell
# List installed distributions
wsl --list --verbose
wsl -l -v

# Set default distribution
wsl --set-default Ubuntu

# Set default WSL version
wsl --set-default-version 2

# Shutdown WSL
wsl --shutdown

# Terminate a specific distribution
wsl --terminate Ubuntu

# Update WSL
wsl --update

# Export/import distribution (for backup/restore)
wsl --export Ubuntu ubuntu-backup.tar
wsl --import Ubuntu-New C:\WSL\Ubuntu-New ubuntu-backup.tar
```

---

## GPU Support: CUDA and JAX

WSL2 supports GPU passthrough for NVIDIA GPUs, enabling CUDA applications like JAX and PyTorch to access your GPU.

### Prerequisites

1. **NVIDIA GPU drivers** must be installed on Windows (not in WSL)
2. **WSL2** must be running (not WSL1)
3. Verify GPU access from Windows first:
   ```powershell
   nvidia-smi
   ```

### Setting Up CUDA Passthrough

WSL provides a CUDA forwarder library at `/usr/lib/wsl/lib/libcuda.so.1`. Make it accessible:

```bash
# Add CUDA library path to your shell configuration
echo 'export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc

# Update library cache
sudo ldconfig
```

**Verify CUDA access:**

```bash
# Check if libcuda is visible
ldconfig -p | grep libcuda

# Should show: /usr/lib/wsl/lib/libcuda.so.1

# Test GPU access from WSL
nvidia-smi
```

You should see the same GPU information as in Windows.

### Installing JAX with CUDA Support

**Note**: JAX CUDA wheels conflict with PyTorch CUDA wheels due to different CUDA/cuDNN runtime requirements. For SOEN-Toolkit, use one of these approaches:

#### Option 1: JAX GPU + CPU-only PyTorch (Recommended)

This is the simplest approach for SOEN-Toolkit, which primarily uses JAX for training:

```bash
# After installing SOEN-Toolkit with uv (see Install_WSL2.md)
# Activate your environment
source .venv/bin/activate

# Install JAX with CUDA support
uv pip install --upgrade "jax[cuda12_local]==0.4.28"

# Install CPU-only PyTorch to avoid conflicts
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Pin compatible NumPy/SciPy versions (if needed)
uv pip install "numpy>=1.24,<2.0" "scipy>=1.10,<1.13"
```

**Test JAX GPU access:**

```bash
python -c "import jax; print('JAX devices:', jax.devices())"
python -c "import jax.numpy as jnp; x = jnp.ones((100, 100)) @ jnp.ones((100, 100)); print('JAX GPU matmul OK:', x.shape)"
```

Expected output: `JAX devices: [cuda(id=0)]` and `JAX GPU matmul OK: (100, 100)`

#### Option 2: Separate Environments

If you need both JAX GPU and PyTorch GPU, use separate virtual environments:

**JAX environment:**
```bash
# Create separate environment for JAX
python3 -m venv ~/jaxenv
source ~/jaxenv/bin/activate
pip install --upgrade pip wheel
pip install "jax[cuda12_local]==0.4.28"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**PyTorch environment:**
```bash
# Create separate environment for PyTorch
python3 -m venv ~/torchenv
source ~/torchenv/bin/activate
pip install --upgrade pip wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then register separate Jupyter kernels for each environment if using notebooks.

### GPU Memory Management

For WSL environments, set conservative GPU memory settings:

```bash
# Add to ~/.bashrc or your environment
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_PYTHON_CLIENT_ALLOCATOR=default
```

These settings help prevent GPU memory issues in WSL, especially on laptops with limited VRAM.

### Troubleshooting GPU Issues

**JAX can't find GPU:**
1. Verify CUDA passthrough is set up:
   ```bash
   ldconfig -p | grep libcuda
   nvidia-smi
   ```

2. Check CUDA library path is exported:
   ```bash
   echo $LD_LIBRARY_PATH
   # Should include /usr/lib/wsl/lib
   ```

3. Restart WSL if needed:
   ```powershell
   wsl --shutdown
   ```
   Then restart WSL and try again.

**CUDA version mismatch:**
- JAX CUDA wheels include their own CUDA runtime libraries
- These should work with any NVIDIA driver that supports CUDA 12.x
- If you see version errors, ensure you're using `jax[cuda12_local]` and that your Windows NVIDIA driver is up to date

**PyTorch/JAX conflicts:**
- If you get import errors or crashes when both are installed, use Option 1 above (CPU-only PyTorch) or separate environments
- The conflict arises from incompatible CUDA runtime libraries, not from using both frameworks simultaneously

---

## Troubleshooting

### WSL Won't Start

1. Ensure WSL features are enabled:
   ```powershell
   wsl --status
   ```

2. Update WSL:
   ```powershell
   wsl --update
   ```

3. Restart your computer if needed.

### Cannot Access Windows Files

- Ensure you're using forward slashes: `/mnt/c/Users/...`
- Check that the drive exists in Windows
- Try accessing from Windows first to ensure the path is valid

### Permission Issues

If you get permission errors when accessing Windows files:

```bash
# Change ownership (use with caution)
sudo chown -R $USER:$USER /mnt/c/path/to/folder

# Or set proper permissions
sudo chmod 755 /mnt/c/path/to/folder
```

**Note**: It's generally better to work in the WSL file system (`~`) for development projects to avoid permission issues.

### Distribution Not Found

If you can't find your distribution:

```powershell
# List all distributions (including ones that failed to install)
wsl --list --all

# Unregister a broken distribution
wsl --unregister Ubuntu

# Reinstall from Microsoft Store
```

---

## Next Steps

Once WSL is set up:

1. **[Install SOEN-Toolkit](Install_WSL2.md)** - Follow the installation guide for WSL
2. **[Building Models](Building_Models.md)** - Create your first model
3. **[Building Models](Building_Models.md)** - Create your first model

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Install_WSL2.md">&#8592; Back to WSL Installation</a>
</div>

