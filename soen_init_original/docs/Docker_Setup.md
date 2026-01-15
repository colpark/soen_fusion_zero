---
layout: default
title: Docker Setup
---

# Docker Setup for Cloud Training

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Cloud_Training.md" style="margin-right: 2em;">&#8592; Previous: Cloud Training</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
</div>

Docker packages SOEN code and dependencies into a container that runs identically on any machine. This guide covers building images for AWS SageMaker.

---

## Prerequisites

### 1. Install Docker

Choose your platform:

<details>
<summary><strong>🍎 macOS</strong></summary>

```bash
# Option A: Homebrew
brew install --cask docker

# Option B: Download from docker.com
# https://www.docker.com/products/docker-desktop

# After install, open Docker Desktop from Applications
# Wait for the whale icon to stop animating
```

</details>

<details>
<summary><strong>🪟 Windows</strong></summary>

1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Run the installer (requires WSL 2 - it will prompt to install if needed)
3. Restart your computer
4. Open Docker Desktop
5. Wait for it to start (green indicator in system tray)

**PowerShell:**
```powershell
# Verify installation
docker --version
```

</details>

<details>
<summary><strong>🐧 Linux (Ubuntu/Debian)</strong></summary>

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (avoids needing sudo)
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker --version
```

</details>

Verify Docker is running:
```bash
docker --version
# Docker version 24.x.x, build ...
```

### 2. Install AWS CLI

<details>
<summary><strong>🍎 macOS</strong></summary>

```bash
brew install awscli
```

</details>

<details>
<summary><strong>🪟 Windows</strong></summary>

Download and run the [AWS CLI MSI installer](https://aws.amazon.com/cli/).

Or with winget:
```powershell
winget install Amazon.AWSCLI
```

</details>

<details>
<summary><strong>🐧 Linux</strong></summary>

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

</details>

### 3. Configure AWS Credentials

```bash
aws configure
# Enter:
#   AWS Access Key ID: YOUR_ACCESS_KEY
#   AWS Secret Access Key: YOUR_SECRET_KEY
#   Default region: us-east-1
#   Default output format: json
```

---

## Build Docker Images

### Option A: Use the Build Script (Recommended)

The build script handles everything automatically:

```bash
cd /path/to/soen-toolkit

# Make executable (macOS/Linux only)
chmod +x src/soen_toolkit/cloud/docker/build.sh

# Run the build script
# Usage: ./build.sh <ecr-repo-uri> [version-tag]
./src/soen_toolkit/cloud/docker/build.sh \
    YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit \
    v1
```

Replace `YOUR_ACCOUNT` with your 12-digit AWS account ID.

**Windows (PowerShell):**
```powershell
# Run with bash (Git Bash or WSL)
bash src/soen_toolkit/cloud/docker/build.sh `
    YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit `
    v1
```

The script will:
1. Login to ECR
2. Build PyTorch image → `<repo>:v1-pytorch`
3. Build JAX image → `<repo>:v1-jax`
4. Push both to ECR

### Option B: Manual Build

**Step 1: Create ECR Repository**

```bash
aws ecr create-repository \
    --repository-name soen-toolkit \
    --region us-east-1
```

Save the repository URI from the output.

**Step 2: Login to ECR**

```bash
# Replace YOUR_ACCOUNT_ID
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```

**Step 3: Build Images**

```bash
cd /path/to/soen-toolkit

# PyTorch image
docker build \
    --platform linux/amd64 \
    -t soen-toolkit:pytorch \
    -f src/soen_toolkit/cloud/docker/Dockerfile.pytorch \
    .

# JAX image (if using JAX)
docker build \
    --platform linux/amd64 \
    -t soen-toolkit:jax \
    -f src/soen_toolkit/cloud/docker/Dockerfile.jax \
    .
```

> **Note:** `--platform linux/amd64` is required on Apple Silicon Macs since SageMaker runs x86_64.

**Step 4: Tag and Push**

```bash
REPO=YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit

docker tag soen-toolkit:pytorch ${REPO}:pytorch
docker tag soen-toolkit:jax ${REPO}:jax

docker push ${REPO}:pytorch
docker push ${REPO}:jax
```

---

## Configure SOEN to Use Your Images

### Option A: Environment Variables

```bash
# Add to ~/.bashrc, ~/.zshrc, or Windows environment
export SOEN_DOCKER_PYTORCH="YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit:pytorch"
export SOEN_DOCKER_JAX="YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit:jax"
```

### Option B: Cloud Config YAML

```yaml
# cloud_config.yaml
docker_images:
  pytorch: YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit:pytorch
  jax: YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit:jax
```

### Option C: Cloud GUI

1. Open: `python -m soen_toolkit.cloud_gui`
2. Go to **Credentials** tab
3. Enter image URIs
4. Click **Save as Default**

---

## When to Rebuild

**Rebuild when:**
- SOEN Toolkit is updated
- You add new dependencies
- Base images have security updates

**No rebuild needed for:**
- Training config changes
- Different datasets
- Hyperparameter changes

The training config is uploaded separately at job submission.

---

## Troubleshooting

### "Cannot connect to Docker daemon"

Docker Desktop isn't running. Start it:
- **macOS:** Open Docker Desktop from Applications
- **Windows:** Start Docker Desktop from Start menu
- **Linux:** `sudo systemctl start docker`

### "authorization token has expired"

Re-login to ECR:
```bash
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```

### "no space left on device"

Docker ran out of disk space:
```bash
# Clean up unused images and containers
docker system prune -a
```

### "repository does not exist"

Create the ECR repository first:
```bash
aws ecr create-repository --repository-name soen-toolkit --region us-east-1
```

### Slow builds

First builds are slow (~10-20 min) because Docker downloads large base images. Subsequent builds use caching and are much faster.

### Build fails on Apple Silicon

Ensure you use `--platform linux/amd64`:
```bash
docker build --platform linux/amd64 -t soen-toolkit:pytorch ...
```

---

## Quick Reference

```bash
# Full workflow
ACCOUNT_ID=YOUR_12_DIGIT_ID
REGION=us-east-1
REPO=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/soen-toolkit

# 1. Login
aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin \
    ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# 2. Build (from project root)
docker build --platform linux/amd64 -t soen-toolkit:pytorch \
    -f src/soen_toolkit/cloud/docker/Dockerfile.pytorch .

docker build --platform linux/amd64 -t soen-toolkit:jax \
    -f src/soen_toolkit/cloud/docker/Dockerfile.jax .

# 3. Tag
docker tag soen-toolkit:pytorch ${REPO}:pytorch
docker tag soen-toolkit:jax ${REPO}:jax

# 4. Push
docker push ${REPO}:pytorch
docker push ${REPO}:jax

# 5. Set environment
export SOEN_DOCKER_PYTORCH="${REPO}:pytorch"
export SOEN_DOCKER_JAX="${REPO}:jax"
```

---

## Common Docker Commands

```bash
# List local images
docker images

# List running containers
docker ps

# Stop all containers
docker stop $(docker ps -q)

# Free up disk space
docker system prune -a

# Check disk usage
docker system df

# Test image locally
docker run -it --rm soen-toolkit:pytorch bash
# Inside container:
python -c "import soen_toolkit; print('OK')"
```

<div align="center" style="margin-top: 2em;">
  <a href="Cloud_Training.md" style="margin-right: 2em;">&#8592; Previous: Cloud Training</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
</div>
