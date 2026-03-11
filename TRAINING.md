# Training the TCN

## 1. Prebuilt (mmap) data — recommended

After running `preprocess_subseqs.py` (or notebook 4c) with `format_mode='mmap'`, train from the prebuilt dir. No HDF5 at runtime; fast DataLoader from memory-mapped files.

**Single GPU (e.g. 1 GPU on SciServer):**
```bash
torchrun --standalone --nproc_per_node=1 train_tcn_ddp.py \
  --prebuilt-subseq-dir subseqs \
  --batch-size 48 \
  --epochs 200 \
  --checkpoint-dir checkpoints_tcn_ddp
```

**Multi-GPU (4 GPUs):**
```bash
torchrun --standalone --nproc_per_node=4 train_tcn_ddp.py \
  --prebuilt-subseq-dir subseqs \
  --batch-size 48 \
  --epochs 200 \
  --checkpoint-dir checkpoints_tcn_ddp
```

Or use the helper (override prebuilt dir if needed):
```bash
bash run_train.sh --prebuilt-subseq-dir subseqs
```

**Resume from checkpoint:**
```bash
torchrun --standalone --nproc_per_node=4 train_tcn_ddp.py \
  --prebuilt-subseq-dir subseqs \
  --resume checkpoints_tcn_ddp/last.pt
```

---

## 2. From HDF5 (no prebuilt)

If you skip preprocessing and want to read decimated HDF5 on the fly:

```bash
torchrun --standalone --nproc_per_node=4 train_tcn_ddp.py \
  --root /path/to/dsrpt \
  --decimated-root /path/to/dsrpt_decimated \
  --clear-root /path/to/clear \
  --clear-decimated-root /path/to/clear_decimated \
  --batch-size 48 \
  --epochs 200
```

`num_workers` is forced to 0 in code to avoid shared-memory issues on HPC; prebuilt + mmap is faster than HDF5 with num_workers=0.

---

## 3. Important options

| Option | Default | Description |
|--------|---------|-------------|
| `--prebuilt-subseq-dir` | None | Use mmap/npz prebuilt dir (train/ + test/); no HDF5. |
| `--batch-size` | 48 | Per-GPU batch size. |
| `--epochs` | 200 | Max epochs. |
| `--lr` | (auto) | Learning rate (default from batch size). |
| `--optimizer` | adamw | adamw or sgd. |
| `--ignore-twarn` | False | Mask Twarn window from loss (learn clear only). |
| `--nsub` | 781250 | Window length (samples). Must match preprocess. |
| `--checkpoint-dir` | checkpoints_tcn_ddp | Where to save best/last.pt. |
| `--resume` | None | Resume from checkpoint path. |

---

## 4. Why training can fail: loss weighting

Training can become effectively **not trainable** (accuracy stuck near the positive-class fraction, model predicts 1 almost everywhere) if the **BCE loss is over-weighted on the positive class**.

**What goes wrong**

- The dataset can supply per-timestep weights (e.g. `pos_weight` / `neg_weight` from stratified balance, so positives get weight ~4–5 and negatives ~0.5).
- The training loop also has a **per-batch** reweighting (`batch_weights(tgt_v)`) that gives roughly 50–50 total weight to positives vs negatives in the batch.
- If **both** are applied in the loss, i.e. `weight = dataset_weight * batch_weights(target)`, then the **effective** weight on positive timesteps is much larger than on negatives (e.g. 4.5 × 2.8 vs 0.56 × 0.6). The loss is then dominated by positive timesteps.
- The optimizer minimizes loss mainly by getting positives right, so the model is pushed to **predict 1 everywhere**. Accuracy plateaus around the positive fraction (e.g. ~18%), gradients shrink after clipping, and the model does not learn a useful decision boundary.

**Correct approach (current code)**

- Use **one** source of balancing only. This code uses **only** `batch_weights(tgt_v)` in the loss; the dataset’s third element (weight) is **not** used in the loss. So we get a single, per-batch 50–50 balance and training remains stable.
- If you later use **dataset** weights (e.g. for Twarn masking or exclude_last), do **not** multiply them by `batch_weights` in the same loss. Either: use dataset weights alone, or use `batch_weights` alone.

---

## 5. Paths on SciServer

If prebuilt data lives under scratch:
```bash
PREBUILT="Temporary/dpark1/scratch/soen_fusion_zero/subseqs"
torchrun --standalone --nproc_per_node=4 train_tcn_ddp.py --prebuilt-subseq-dir "$PREBUILT" --batch-size 48
```
