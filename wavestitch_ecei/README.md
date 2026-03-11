# WaveStitch-style diffusion for ECEi

Conditional time-series diffusion using the [WaveStitch](https://github.com/adis98/WaveStitch) backbone (SSSDS4Imputer with S4 layers) on ECEi data: **160 channels × ~7512 time steps** (decimated), with two conditions:

1. **Binary class**: 0 = clear, 1 = disruption.
2. **t_disrupt_cond**: normalised timestep in [0, 1] when disruption starts (t_disruption − 300 ms); 0 for clear.

Data is the same prebuilt mmap as in `../diffusion` (e.g. `subseqs_original_mmap` with `decimate_factor=10`).

## WaveStitch conditioning (unchanged)

- Input has **162 channels**: 160 signal + 2 condition (class_id broadcast, t_disrupt_cond broadcast).
- **Condition channels are not noised** (WaveStitch “hierarchical” columns): mask keeps channels 160 and 161 fixed; only 0..159 are noised.
- Model predicts **noise for the 160 signal channels only**; loss is MSE on that prediction.

## Usage

From the **`wavestitch_ecei`** directory (so that `utils` and `TSImputers` resolve):

```bash
# Train (default: subseqs_original_mmap, decimate 10)
python train.py --prebuilt-mmap-dir ../subseqs_original_mmap --decimate-factor 10 --batch-size 16 --epochs 100

# Train with verbose evaluation (config, model details, validation loss, sample paths)
python train.py --prebuilt-mmap-dir ../subseqs_original_mmap --decimate-factor 10 --verbose

# Samples are saved every 5 epochs by default to checkpoints_wavestitch_ecei/samples/
# (clear and disrupt conditioning). Use --sample-every N or 0 to disable.
```

**Training options:**

- `--verbose` — Print full config (JSON), dataset size, model parameter count, device; each epoch print validation loss (over up to 20 batches of test split); log checkpoint and sample paths.
- `--log-every N` — Log loss every N batches (0 = epoch only).
- `--sample-every N` — Generate and save sample visualizations every N epochs (default: 5; 0 = never). Saves `epochXXXX_clear.png` and `epochXXXX_disrupt.png` in `checkpoints_wavestitch_ecei/samples/`.
- `--num-sample-viz N` — Number of samples per visualization (default: 4).

```bash
# Sample (after training)
python sample.py --checkpoint checkpoints_wavestitch_ecei/ckpt_epoch_100.pt --num-samples 4 --class-id 1 --t-disrupt 0.5 --T 7512
```

## Layout

- **`TSImputers/`** — WaveStitch SSSDS4Imputer and S4Model (unchanged).
- **`utils/`** — WaveStitch `util.py` (only change: device-agnostic diffusion step embedding).
- **`data/dataset.py`** — ECEi loader (same logic as `../diffusion/data/dataset.py`).
- **`training_utils.py`** — MyDataset, fetchModel, fetchDiffusionConfig (no pandas).
- **`train.py`** — Builds (B, T, 162) batches for WaveStitch, conditional noising, same loss as WaveStitch; optional verbose config/model/val loss; saves samples every 5 epochs.
- **`sample.py`** — Reverse process with fixed condition channels.

See **MODIFICATIONS.md** for what was changed relative to upstream WaveStitch.
