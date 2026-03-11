# Line-by-line: Original DisruptCNN loader vs EceiDatasetOriginal

**Reference (original):** `soenre/disruptcnn/loader.py` — class `EceiDataset`  
**Current (“original” setting):** `soen_fusion_zero/disruptcnn/dataset_original.py` — class `EceiDatasetOriginal`

---

## Tick summary: four areas

| Area | Status | Detail |
|------|--------|--------|
| **Clear file** | ✓ Correct | **Optional:** when `clear_file` is provided and exists, `data_all = np.vstack((data_disrupt, data_clear))`. When omitted or missing → disrupt-only (`data_all = data_disrupt`). |
| **shots2seqs** | ✓ Correct | Same formulas as original: N, num_seq, Nseq, start adjustment, window loop. No `_file_length`, no skip by file length, no tail fallback. `num_seq` can go to 0 in else branch (matches original). |
| **_read_data** | ✓ Correct | Exact slice `LFS[..., start_idxi[index]:stop_idxi[index]][..., ::step]`; no clamping. Offset subtract and normalize same as original. |
| **calc_label_weights** | ✓ Correct | `inds=None` → `np.arange(len(self.shot_idxi))` (sequence-level). Same N, Ndisrupt, Nnondisrupt formula; `max(..., 1)` to avoid division by zero. |

---

## 1. Data loading and shot lists

| Aspect | Original (soenre loader.py) | Current (dataset_original.py) |
|--------|-----------------------------|-------------------------------|
| **Clear file** | Required: `data_clear = np.loadtxt(clear_file, skiprows=1)`, then `data_all = np.vstack((data_disrupt, data_clear))` | Optional: if `clear_file` provided and exists, same vstack; else disrupt-only (`data_all = data_disrupt`). |
| **Single-row shot list** | `if data.ndim == 1` not handled (would fail or behave oddly) | `data_disrupt.ndim == 1` → `data_disrupt = data_disrupt[np.newaxis, :]`. |
| **SNR filter** | `snr_min_threshold` in __init__; filters with `data_all = data_all[keep]` before computing indices. | Same: `snr_min_threshold` in `_parse_shot_lists`, same filter. |
| **Column indices** | Hardcoded: `data_all[:,-3]`, `data_all[:,-2]`, `[:,2]`, `[:,3]`, `[:,4]`, `[:,-1]` (t_flat_start, t_flat_last, tstart, tlast, dt, tdisrupt). | Named: `COL_T_FLAT_START=6`, `COL_T_FLAT_LAST=7`, `COL_TSTART=2`, etc. (same logical columns, 0-based). |

**Verdict:** Same segment math; current allows disrupt-only and handles single-row file.

---

## 2. Segment indices (start_idx, stop_idx, disrupt_idx, zero_idx)

| Formula | Original | Current |
|---------|---------|--------|
| disrupt_idx | `np.ceil((tdisrupt - self.Twarn - tstarts) / dt).astype(int)`; `disrupt_idx[tdisrupt<0] = -1000` | `np.ceil((tdisrupt - Twarn - tstarts) / dt).astype(int)`; `disrupt_idx[tdisrupt < 0] = -1000` |
| zero_idx | `np.ceil((0. - tstarts) / dt).astype(int)` | `np.ceil((0.0 - tstarts) / dt).astype(int)` |
| flattop start | `np.ceil((tflatstarts - tstarts) / dt).astype(int)` | Same in `_parse_shot_lists`. |
| tend (flattop) | `tend = np.maximum(tdisrupt, np.minimum(tstops, tflatstops))` | `tend = np.maximum(tdisrupt, np.minimum(tstops, tflatstops))` |
| tend (non-flattop) | `tend = np.maximum(tdisrupt, tstops)` | `tend = np.maximum(tdisrupt, tstops)` |
| stop_idx | `np.floor((tend - tstarts) / dt).astype(int)` | Same. |

**Verdict:** Mathematically identical.

---

## 3. H5 layout and file existence

| Aspect | Original | Current |
|--------|----------|--------|
| **Path** | `self.root + shot_type + '/' + str(self.shot[index]) + '.h5'` (e.g. `root/disrupt/144200.h5`). | No decimated: `Path(root) / folder / f"{shot}.h5"`. With `decimated_root`: `decimated_root / f"{shot}.h5"` (flat dir). |
| **File existence** | Not checked; assumes all shots have files. | Before `shots2seqs`: drops shots whose H5 does not exist (`_path_for_shot(...).exists()`), updates shot/start_idx/stop_idx/disrupt_idx/zero_idx. |
| **Offsets** | Assumes every H5 has `offsets`: `self.offsets = np.zeros(f['offsets'].shape + (self.shot.size,), ...)` from first file. | If `"offsets" in f` use same; else zeros (for decimated without offsets). |

**Verdict:** Same layout when using raw `root`; current adds decimated flat layout and file-existence filtering.

---

## 4. Normalization

| Aspect | Original | Current |
|--------|----------|--------|
| **Path** | `self.root + 'normalization.npz'` only. | Explicit `norm_stats_path`, or search: `root/normalization.npz`, `root/norm_stats.npz`, then decimated_root variants. |
| **Keys** | `f['mean_flat']`, `f['std_flat']` or `f['mean_all']`, `f['std_all']`. | Same; plus fallback: if key missing, use `f['mean']`, `f['std']`. |

**Verdict:** Same keys when present; current adds alternate paths and fallback.

---

## 5. shots2seqs()

| Aspect | Original | Current |
|--------|----------|--------|
| **N, num_seq, Nseq** | `N = int((stop_idx - start_idx + 1) / data_step)`; `num_seq_frac = (N - nsub) / (nsub - nrecept + 1) + 1`; `num_seq = np.ceil(num_seq_frac).astype(int)`; `if num_seq < 1: num_seq = 1`; `Nseq = nsub + (num_seq - 1)*(nsub - nrecept + 1)`. | Same formulas; uses `step = self._step_in_getitem` and `self.nsub`, `self.nrecept` (possibly decimated). |
| **Start adjustment** | If `(start_idx > zero_idx) & ((start_idx - zero_idx + 1) > (Nseq - N)*data_step)`: `start_idx -= (Nseq - N)*data_step`; else `num_seq -= 1`, `Nseq = ...`, `start_idx += (N - Nseq)*data_step`. | Same condition and updates (with `step` and in-place on `self.start_idx[s]`). |
| **Window loop** | For `m in range(num_seq)`: `start_idxi = start_idx[s] + (m*nsub - m*nrecept + m)*data_step`, `stop_idxi = ...`; disrupt_idxi from window overlap. | Same: no file-length check, no skip, no tail fallback. |
| **File length** | Not used; no check. | Same: not used; no check. |
| **Arrays** | `shot_idxi`, `start_idxi`, `stop_idxi`, `disrupt_idxi` as lists then `np.array`. | Same; plus `self.disruptedi = (self.disrupt_idxi > 0)`; `self.length = len(self.shot_idxi)` only when `test == 0`. |

**Verdict:** Matches original: same tiling, start adjustment, and window loop; no file-length logic or tail fallback.

---

## 6. calc_label_weights

| Aspect | Original | Current |
|--------|----------|--------|
| **inds** | `if inds is None: inds = np.arange(len(self.shot))` — wrong for sequence-level: should be over sequence indices. | `if inds is None: inds = np.arange(len(self.shot_idxi))` — correct for subsequence indices. |
| **N, Ndisrupt, Nnondisrupt** | `N = np.sum(stop_idxi[inds] - start_idxi[inds])`, `disinds = inds[self.disruptedi[inds]]`, `Ndisrupt = np.sum(stop_idxi[disinds] - disrupt_idxi[disinds])`, `Nnondisrupt = N - Ndisrupt`. | Same. |
| **Division** | `pos_weight = 0.5*N/Ndisrupt`, `neg_weight = 0.5*N/Nnondisrupt` — can divide by zero if no disrupt or no nondisrupt. | `pos_weight = 0.5 * N / max(Ndisrupt, 1)`, `neg_weight = 0.5 * N / max(Nnondisrupt, 1)`. |

**Verdict:** Current fixes inds (sequence vs shot) and avoids division by zero; formula otherwise same.

---

## 7. train_val_test_split

| Aspect | Original | Current |
|--------|----------|--------|
| **test > 0** | `self.train_inds = self.test_indices`, `self.val_inds = []`, `self.test_inds = []`. | Same; `val_inds`/`test_inds` as `np.array([], dtype=int)`. |
| **Stratify** | `train_test_split(..., stratify=labels, test_size=np.sum(sizes[1:]), ...)`, then val/test split. | Same; `test_size=sizes[1] + sizes[2]`, then `test_size=sizes[2]/(sizes[1]+sizes[2])`. |
| **Sequence indices** | `train_inds = np.where(np.in1d(self.shot_idxi, train_shot_inds))[0]` (and val/test). | Same. |

**Verdict:** Equivalent.

---

## 8. read_data / _read_data and __getitem__

| Aspect | Original | Current |
|--------|----------|--------|
| **Slice** | `f['LFS'][..., self.start_idxi[index]:self.stop_idxi[index]][..., ::self.data_step]` | Same: exact slice, no clamping. |
| **Offset** | Subtract `self.offsets[..., shot_index][..., np.newaxis]`. | Same. |
| **Normalize** | `(X - normalize_mean[..., np.newaxis]) / normalize_std[..., np.newaxis]`. | Same. |
| **Return type** | NumPy arrays. | Torch tensors (`torch.from_numpy(...).float()`). |
| **Target/weight** | `target[int((disrupt_idxi - start_idxi + 1) / data_step):] = 1` and same for weight. | `first_disrupt = int((disrupt_idxi - start_idxi + 1) / _step_in_getitem)` then `target[first_disrupt:] = 1`. Same logic. |
| **Return** | `return X, target, index.item(), weight` | `return (X_tensor, target_tensor, idx, weight_tensor)` — idx is sequence index (no .item() on numpy from Subset). |

**Verdict:** Label and weight logic match. Slice is exact (no clamping). Returns tensors.

---

## 9. Decimated vs raw

| Aspect | Original | Current |
|--------|----------|--------|
| **Indices** | All indices and nsub/nrecept in **raw** (1 MHz) space; `data_step` only used when slicing `[::data_step]`. | If `decimated_root` set: **converts** start_idx, stop_idx, disrupt_idx, zero_idx to decimated (divide by data_step); `nsub`, `nrecept` converted; `_step_in_getitem = 1` so no extra subsample in slice. |
| **H5 path** | Always `root/disrupt|clear/{shot}.h5`. | With decimated: `decimated_root/{shot}.h5` (flat). |

**Verdict:** Original has no decimated mode. Current adds a full decimated path with index conversion and flat dir.

---

## 10. data_generator / data_generator_original

| Aspect | Original | Current |
|--------|----------|--------|
| **Subsets** | Subset(dataset, train_inds), val_inds, test_inds. | Same. |
| **Samplers** | StratifiedSampler for train and val; test_loader no sampler (shuffle=False, drop_last=True). | Same; test_loader also no StratifiedSampler. |
| **Undersample** | If undersample: `inds = [dataset.train_inds[i] for i in train_sampler]`, `dataset.calc_label_weights(inds=inds)`. | Same. |

**Verdict:** Equivalent.

---

## Summary: does current “exactly” reflect original?

**Four areas are correctly implemented** (see tick table at top): Clear file required, shots2seqs matches (no file-length/tail), _read_data exact slice (no clamping), calc_label_weights sequence-level with safe division. Remaining differences: “when equivalent” noted where results match under typical conditions. The current dataloader is **intentionally** a superset of the original with these differences:

1. **File existence** — drops shots without an H5 file; original assumes all exist.
2. **Norm path** — multiple search paths and `mean`/`std` fallback; original only `root+'normalization.npz'`.
3. **Offsets** — can be missing (zeros) for decimated; original assumes present.
4. **Decimated mode** — index conversion, flat `decimated_root/{shot}.h5`, `_step_in_getitem=1`; original has no decimated path.
5. **Return** — tensors and sequence index; original NumPy and `index.item()`.

