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

## 4. Paths on SciServer

If prebuilt data lives under scratch:
```bash
PREBUILT="Temporary/dpark1/scratch/soen_fusion_zero/subseqs"
torchrun --standalone --nproc_per_node=4 train_tcn_ddp.py --prebuilt-subseq-dir "$PREBUILT" --batch-size 48
```
