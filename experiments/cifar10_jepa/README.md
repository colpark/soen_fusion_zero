# CIFAR-10 I-JEPA Baseline

Self-contained proof-of-concept of [I-JEPA](https://arxiv.org/abs/2301.08243)
on CIFAR-10 (32×32 images). Adapted from Meta's official codebase.

## Architecture

| Component | Details |
|-----------|---------|
| Image size | 32×32 |
| Patch size | 4 → 8×8 grid = 64 patches |
| Encoder | ViT-Tiny (dim=192, depth=6, heads=3) |
| Predictor | Transformer (dim=96, depth=4, heads=3) |
| Target encoder | EMA copy of encoder |
| Loss | Smooth L1 between predicted and target embeddings |

## Quick Start

```bash
cd experiments/cifar10_jepa

# Train (auto-downloads CIFAR-10)
python train.py

# Train with custom settings
python train.py --epochs 100 --lr 5e-4 --batch_size 128

# Linear probe evaluation
python linear_probe.py --checkpoint output/cifar10_jepa/checkpoint_latest.pth
```

## Expected Sanity Checks

| What | Expected |
|------|----------|
| Loss (epoch 1) | ~0.15–0.25 |
| Loss (epoch 100) | ~0.05–0.10 |
| Loss trend | Monotonically decreasing (with noise) |
| Linear probe (random init) | ~10% (chance) |
| Linear probe (100 ep pretrain) | >30–40% |
| Linear probe (200 ep pretrain) | >45–55% |

> Note: CIFAR-10 is low-resolution. JEPA was designed for ImageNet-scale;
> results here are for sanity-checking, not SOTA.

## Files

| File | Description |
|------|-------------|
| `models.py` | ViT encoder, predictor, mask collator |
| `train.py` | JEPA training loop (single GPU) |
| `linear_probe.py` | Freeze encoder → train linear classifier |
| `config.yaml` | Default hyperparameters |

## Requirements

- PyTorch ≥ 2.0
- torchvision
- numpy, pyyaml
