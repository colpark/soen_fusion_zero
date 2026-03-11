# Modifications relative to WaveStitch

We keep the **core WaveStitch algorithm** unchanged: same noising (conditional mask on “hierarchical” channels), same S4 backbone, same diffusion schedule and loss (predict noise on non-hierarchical channels only). Only the following were adapted.

## 1. Data and conditioning

- **Upstream**: Tabular datasets (MetroTraffic, etc.) with pandas; “hierarchical” columns are metadata (e.g. cyclic time).
- **Here**: ECEi multivariate series (160, T) from prebuilt mmap; we **append two condition channels** so that each sample is (162, T):
  - Channel 160: binary class (0 = clear, 1 = disruption), broadcast along time.
  - Channel 161: `t_disrupt_cond` in [0, 1], broadcast along time.
- These two channels are treated as **hierarchical**: they are **not noised** and are fixed at sampling time. So the conditioning mechanism is the same as WaveStitch; only the meaning and number of condition dimensions change.

## 2. `utils/util.py`

- **Change**: In `calc_diffusion_step_embedding`, the embedding tensor is created on the **same device as `diffusion_steps`** instead of hardcoding `.cuda()`.
- **Reason**: Run on CPU or any device without assuming CUDA.

## 3. No changes to core model or training loop

- **TSImputers/SSSDS4Imputer.py** and **TSImputers/S4Model.py** are used as in the original WaveStitch repo.
- **Training**: Same forward noising formula, same mask (condition channels untouched), same loss (MSE on predicted noise for the 160 signal channels only). Timesteps are integers in [0, T-1] as in WaveStitch.
- **Sampling**: Standard DDPM reverse; the two condition channels are overwritten with the desired (class_id, t_disrupt_cond) at each step and not updated by the model.

## 4. New/removed pieces

- **New**: `data/dataset.py` for ECEi (prebuilt mmap, decimate, split, same (x, class_id, t_disrupt_cond) as in `../diffusion`).
- **New**: `train.py` and `sample.py` that use this dataset and the 162-channel layout.
- **Removed**: All pandas/Preprocessor/dataset-specific logic; no `data_utils.py` or `metasynth.py`.

If you need to re-use another WaveStitch variant (e.g. ordinal/onehot encoding, repaint, or pipeline strided synthesis), those would require separate adaptation; the **denoising model and training objective** are left as in WaveStitch.
