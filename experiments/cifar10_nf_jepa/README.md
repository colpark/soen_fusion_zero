# CIFAR-10 NF+JEPA (Mamba-GINR-style Neural Field + JEPA)

Proof-of-concept combining:
- **Neural field encoding**: Mamba-based set encoder for sparse (coordinate, value) observations
- **JEPA training**: EMA target encoder + predictive embedding loss
- **Optional INR reconstruction**: coordinate-conditioned decoder for PSNR monitoring

## Architecture Overview

```
Image (32×32) → flatten to 1024 (coord, RGB) pairs
                    │
    ┌───────────────┼───────────────────┐
    │               │                   │
    ▼               ▼                   ▼
Context (K=~102)  Target encoder      Target coords
obs_frac=0.1      (EMA, all 1024)     (T=~204)
    │               │                   │
    ▼               ▼                   │
Mamba Encoder     Mamba Encoder         │
    │               │                   │
    ▼               ▼                   │
ctx_tokens,       select embeddings    │
z_global          at target coords     │
    │               │                   │
    └───────┐       │                   │
            ▼       ▼                   │
         Predictor(ctx_tokens,    ◄─────┘
                   target_coords)
            │       │
            ▼       ▼
    JEPA loss: MSE(predicted, target_embeddings)

    Optional: INR Decoder(z_global, all_coords) → RGB → recon loss
```

## Quick Start

```bash
cd experiments/cifar10_nf_jepa

# Default training (Mamba encoder + JEPA + reconstruction)
python train.py

# Without Mamba (Transformer fallback)
python train.py --use_mamba false

# JEPA-only (no reconstruction loss)
python train.py --recon_weight 0.0

# Different sparsity levels
python train.py --obs_frac 0.05
python train.py --obs_frac 0.2

# Pool-based predictor (simpler, faster)
python train.py --pred_mode pool
```

### Evaluation

```bash
# Reconstruction quality at various observation fractions
python evaluate.py recon \
    --checkpoint output/cifar10_nf_jepa/checkpoint_latest.pth \
    --obs_fracs 0.01 0.05 0.1 0.2 0.5

# Linear probe on encoder features
python evaluate.py probe \
    --checkpoint output/cifar10_nf_jepa/checkpoint_latest.pth \
    --obs_frac 0.1
```

## Expected Sanity Checks

| Metric | Expected |
|--------|----------|
| JEPA loss (epoch 1) | ~0.5–1.0 |
| JEPA loss (epoch 100) | ~0.1–0.3 |
| PSNR (obs_frac=0.1, epoch 1) | ~10–15 dB |
| PSNR (obs_frac=0.1, epoch 100) | ~18–22 dB |
| PSNR (obs_frac=0.5, epoch 100) | ~22–28 dB |
| Linear probe (obs_frac=0.1) | >25–35% |
| Target encoder grad check | PASS (no gradients) |

## Key Ablation Flags

| Flag | Values | Effect |
|------|--------|--------|
| `obs_frac` | 0.01–1.0 | Sparsity level for context |
| `target_frac` | 0.1–0.5 | Target prediction set size |
| `recon_weight` | 0.0–1.0 | Reconstruction loss weight (0 = JEPA-only) |
| `pred_mode` | cross_attn, pool | Predictor architecture |
| `use_mamba` | true, false | Mamba vs Transformer encoder |
| `jepa_loss_type` | mse, cosine | Embedding distance metric |

## Mamba-SSM Installation

The Mamba encoder requires the `mamba-ssm` package which needs CUDA:

```bash
pip install mamba-ssm>=1.2.0
```

**Requirements:**
- CUDA ≥ 11.6
- PyTorch ≥ 2.0

If `mamba-ssm` is not available, the code automatically falls back to a
Transformer-based encoder (self-attention). You can also explicitly use the
fallback with `--use_mamba false`.

## Files

| File | Description |
|------|-------------|
| `models.py` | NeuralFieldEncoder, CoordPredictor, INRDecoder, coord utilities |
| `train.py` | NF+JEPA training loop (single GPU) |
| `evaluate.py` | Reconstruction metrics + linear probe |
| `config.yaml` | Default hyperparameters |
