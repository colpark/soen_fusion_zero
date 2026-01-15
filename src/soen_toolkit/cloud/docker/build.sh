#!/bin/bash
# =============================================================================
# SOEN Toolkit Docker Image Builder
# =============================================================================
#
# This script builds and pushes Docker images to AWS ECR for cloud training.
#
# PREREQUISITES:
#   1. Docker Desktop installed and running
#   2. AWS CLI installed and configured (aws configure)
#   3. ECR repository created (see below)
#
# FIRST TIME SETUP:
#   # Create ECR repository (one time only):
#   aws ecr create-repository --repository-name soen-toolkit --region us-east-1
#
# USAGE:
#   ./build.sh <ecr-repo-uri> [tag-prefix]
#
# EXAMPLES:
#   # Build with 'latest' tag
#   ./build.sh 123456789012.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit
#
#   # Build with version tag
#   ./build.sh 123456789012.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit v1
#
# OUTPUT:
#   - <repo>:v1-pytorch  (or latest-pytorch)
#   - <repo>:v1-jax      (or latest-jax)
#
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Parse arguments
REPO_URI=$1
TAG_PREFIX=${2:-latest}

# Show help if no arguments
if [ -z "$REPO_URI" ]; then
    echo "=============================================="
    echo "  SOEN Toolkit Docker Image Builder"
    echo "=============================================="
    echo ""
    echo "Usage: $0 <ecr-repo-uri> [tag-prefix]"
    echo ""
    echo "Arguments:"
    echo "  ecr-repo-uri   Your ECR repository URI"
    echo "  tag-prefix     Optional tag prefix (default: latest)"
    echo ""
    echo "Example:"
    echo "  $0 123456789012.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit v1"
    echo ""
    echo "First time? You need to create the ECR repository:"
    echo "  aws ecr create-repository --repository-name soen-toolkit --region us-east-1"
    echo ""
    echo "Don't have Docker? Install Docker Desktop:"
    echo "  https://www.docker.com/products/docker-desktop"
    echo ""
    exit 1
fi

# Check prerequisites
info "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker not found. Install Docker Desktop: https://www.docker.com/products/docker-desktop"
fi

if ! docker info &> /dev/null; then
    error "Docker is not running. Please start Docker Desktop."
fi
success "Docker is running"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    error "AWS CLI not found. Install: brew install awscli"
fi
success "AWS CLI found"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    error "AWS credentials not configured. Run: aws configure"
fi
success "AWS credentials valid"

# Find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"

info "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Verify project structure
if [ ! -f "pyproject.toml" ]; then
    error "pyproject.toml not found. This script must be run from the soen-toolkit project."
fi

if [ ! -f "src/soen_toolkit/cloud/docker/Dockerfile.pytorch" ]; then
    error "Dockerfile.pytorch not found. Check project structure."
fi

# Extract region from repo URI (macOS compatible)
REGION=$(echo "$REPO_URI" | sed -n 's/.*\.ecr\.\([^.]*\)\.amazonaws.*/\1/p')
if [ -z "$REGION" ]; then
    REGION="us-east-1"
    warn "Could not extract region from URI, using default: $REGION"
fi
info "AWS Region: $REGION"

# Login to ECR
echo ""
info "Logging into ECR..."
aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "$REPO_URI" || \
    error "ECR login failed. Check your AWS credentials and repository URI."
success "ECR login successful"

# Force linux/amd64 platform for SageMaker compatibility
# This is required when building on Apple Silicon (ARM64) Macs
PLATFORM="linux/amd64"
info "Target platform: $PLATFORM (SageMaker uses x86_64)"

# Build PyTorch image
echo ""
echo "=============================================="
info "Building PyTorch image..."
echo "=============================================="
PYTORCH_TAG="${REPO_URI}:${TAG_PREFIX}-pytorch"
docker build \
    --platform "$PLATFORM" \
    -t "$PYTORCH_TAG" \
    -f src/soen_toolkit/cloud/docker/Dockerfile.pytorch \
    . || error "PyTorch image build failed"
success "PyTorch image built: $PYTORCH_TAG"

info "Pushing PyTorch image to ECR..."
docker push "$PYTORCH_TAG" || error "PyTorch image push failed"
success "PyTorch image pushed"

# Build JAX image
echo ""
echo "=============================================="
info "Building JAX image..."
echo "=============================================="
JAX_TAG="${REPO_URI}:${TAG_PREFIX}-jax"
docker build \
    --platform "$PLATFORM" \
    -t "$JAX_TAG" \
    -f src/soen_toolkit/cloud/docker/Dockerfile.jax \
    . || error "JAX image build failed"
success "JAX image built: $JAX_TAG"

info "Pushing JAX image to ECR..."
docker push "$JAX_TAG" || error "JAX image push failed"
success "JAX image pushed"

# Summary
echo ""
echo "=============================================="
echo -e "${GREEN}  BUILD COMPLETE!${NC}"
echo "=============================================="
echo ""
echo "Docker Images:"
echo "  PyTorch: $PYTORCH_TAG"
echo "  JAX:     $JAX_TAG"
echo ""
echo "Add to your cloud_config.yaml:"
echo "  docker_images:"
echo "    pytorch: $PYTORCH_TAG"
echo "    jax: $JAX_TAG"
echo ""
echo "Or set in the Cloud GUI (Credentials tab)."
echo ""

