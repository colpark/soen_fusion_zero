#!/usr/bin/env bash
# Run from soen_fusion_zero repo root, or from wavestitch_ecei with: python train.py "$@"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec python train.py "$@"
