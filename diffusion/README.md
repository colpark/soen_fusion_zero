# ECEi Diffusion Model (Generative)

Diffusion-based generative model for decimated ECEi sequences **(C=160, T≈7512)** with **AdaLN conditioning**:
1. **Clear vs. Disruption** (class_id: 0 or 1)
2. **t_disruption − 300ms** (normalised in [0,1]) as time condition for disruption samples

Architecture: **2D CNN U-Net** (treats (160, 7512) as a 2D image) with **Adaptive Layer Normalisation (AdaLN)** injecting class + t_disrupt embeddings into each block.

## Layout

```
diffusion/
├── README.md
├── requirements.txt
├── config/default.yaml
├── data/
│   ├── __init__.py
│   └── dataset.py       # DecimatedEceiMmapDataset → (x, class_id, t_disrupt_cond)
├── models/
│   ├── __init__.py
│   ├── adaln_unet.py    # UNet2DAdaLN
│   └── scheduler.py     # DDPMScheduler
├── train.py             # Training loop (DDPM)
├── sample.py            # Sample with given class_id and t_disrupt
├── evaluate.py          # Val loss + real vs generated stats
├── notebooks/
│   └── visualize_clear_vs_disruption.ipynb   # Contrast clear vs disruption
└── checkpoints/         # Saved checkpoints (gitignored)
```

## Data

- **Source**: Same prebuilt mmap as TCN subsample (`subseqs_original_mmap`), with `decimate_factor=10`.
- **Conditioning**: From `target` we compute first-disruption index and (t_disruption − 300ms) in normalised [0,1]. `labels.npy` gives clear vs disruption per sequence.

## Usage

From **repo root**:

```bash
# Train
python diffusion/train.py --prebuilt-mmap-dir ./subseqs_original_mmap --decimate-factor 10 --epochs 100

# Sample (clear)
python diffusion/sample.py --checkpoint diffusion/checkpoints/ckpt_epoch_100.pt --class-id 0 --num-samples 8

# Sample (disruption, t_disrupt=0.3)
python diffusion/sample.py --checkpoint diffusion/checkpoints/ckpt_epoch_100.pt --class-id 1 --t-disrupt 0.3 --num-samples 8

# Evaluate
python diffusion/evaluate.py --checkpoint diffusion/checkpoints/ckpt_epoch_100.pt --output-json diffusion/metrics.json
```

## Visualization (Clear vs Disruption)

Open `diffusion/notebooks/visualize_clear_vs_disruption.ipynb`:

- Load checkpoint and generate **clear** vs **disruption** samples.
- Plot 2D images (channels × time) side-by-side to contrast classes.
- Optional: compare with real data from the dataset.
- Section on **t_disrupt** effect: generate disruption at different `t_disrupt` values.

Run the notebook from repo root so `from diffusion.models` and `from diffusion.data.dataset` resolve.
