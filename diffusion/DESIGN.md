# Diffusion model design notes

## Current backbone: 2D U-Net (UNet2DAdaLN)

- **Input shape**: `(B, 1, H, W)` with `H=160` (ECEi channels), `W≈7512` (decimated time).
- **Structure**: 2D convs over the (160, T) “image”: encoder (conv_in + 4× DownBlock with stride-2), bottleneck (ResBlockAdaLN), decoder (4× UpBlock with ConvTranspose2d + skip concat + ResBlockAdaLN). Conditioning (timestep, class_id, t_disrupt) is fused and fed into every block via AdaLN (GroupNorm + MLP-derived scale/shift).
- **Pros**: Standard image-diffusion recipe; good at local 2D structure.
- **Cons**: (1) Very wide aspect ratio (160 vs 7512) so 2D kernels see a thin strip. (2) Long-range time dependencies rely on repeated down/up; TCN is built for long 1D sequences. (3) Different architecture from the TCN used elsewhere in the repo (classification), so no weight or block reuse.

---

## Option: TCN-style backbone for diffusion

**Idea**: Use the same **1D temporal** building blocks as in `train_tcn_ddp_original.py` (dilated causal/non-causal conv1d, InstanceNorm, etc.) as the denoising network for diffusion, with conditioning (timestep, class_id, t_disrupt) injected via AdaLN (or equivalent) into the blocks.

**Pros**

- **Consistency**: Same inductive bias and architecture family as the TCN used for disruption prediction; same data format `(B, 160, T)`.
- **1D native**: ECEi is 160 channels × time; 1D conv over time (with 160 as channels) matches the physics and matches the rest of the codebase.
- **Long range**: Dilations in TCN are designed for long sequences; no need to compress time into a small 2D grid.
- **Reuse**: Can reuse or mirror `TemporalBlockInstanceNorm` (and similar) and only add conditioning (e.g. AdaLN) and ensure input/output shape `(B, 160, T)` for the denoiser.

**Cons / design choices**

- **Causal vs non-causal**: Classification TCN is causal (chomp). For diffusion denoising we usually want the full sequence (bidirectional) so the model can use past and future. So we’d use **non-causal** 1D conv (e.g. same padding, no chomp), or a symmetric 1D U-Net in time.
- **Conditioning**: Current TCN has no AdaLN; we’d add a conditioning vector (timestep + class + t_disrupt) and inject scale/shift (AdaLN) into each block, similar to the 2D U-Net.
- **Output shape**: Denoiser must output shape `(B, 160, T)`; with 1D convs this is straightforward (same as input).

**Summary**

- **Current**: 2D U-Net is a valid baseline and is fixed (skip shapes and channel counts) so training can run.
- **Next step (if desired)**: Add an alternative backbone `TCNDiffusion` (or similar) that uses 1D TCN-style blocks + AdaLN, same in/out `(B, 160, T)`, and optionally compare it to the 2D U-Net in the same training script.

No TCN backbone has been implemented yet; the above is the intended design if we add it.
