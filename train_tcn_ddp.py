#!/usr/bin/env python3
"""
Distributed TCN training for disruption prediction.

Launch with torchrun:
    torchrun --nproc_per_node=4 train_tcn_ddp.py [OPTIONS]

Or via the helper script:
    bash run_train.sh

Uses PyTorch DistributedDataParallel (DDP) across multiple GPUs with
stratified batch sampling so every batch is balanced (pos/neg).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
# Use filesystem-based sharing instead of /dev/shm to avoid "Bus error"
# and "unable to allocate shared memory" on HPC nodes where /dev/shm is limited.
torch.multiprocessing.set_sharing_strategy('file_system')


def _worker_init_fn(worker_id: int) -> None:
    """Ensure each DataLoader worker also uses file_system sharing (avoids shm exhaustion)."""
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass  # already set
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, Subset

from dataset_ecei_tcn import ECEiTCNDataset, PrebuiltSubseqDataset, StratifiedBatchSampler


# ═════════════════════════════════════════════════════════════════════════
#  Model  (identical to train_tcn.ipynb)
# ═════════════════════════════════════════════════════════════════════════

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, dilation_size=2,
                 kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        if np.isscalar(dilation_size):
            dilation_size = [dilation_size ** i for i in range(num_levels)]
        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, stride=1,
                padding=(kernel_size - 1) * dilation_size[i],
                dilation=dilation_size[i], dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels,
                 kernel_size, dropout, dilation_size):
        super().__init__()
        self.tcn = TemporalConvNet(
            input_size, num_channels,
            kernel_size=kernel_size, dropout=dropout,
            dilation_size=dilation_size)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y = self.tcn(x)
        o = self.linear(y.permute(0, 2, 1))
        return torch.sigmoid(o.squeeze(-1))


# ═════════════════════════════════════════════════════════════════════════
#  Model builder (same logic as notebook)
# ═════════════════════════════════════════════════════════════════════════

def calc_receptive_field(kernel_size, dilation_sizes):
    return 1 + 2 * (kernel_size - 1) * int(np.sum(dilation_sizes))


def build_model(input_channels, n_classes, levels, nhid,
                kernel_size, dilation_base, dropout,
                nrecept_target=30_000):
    channel_sizes = [nhid] * levels
    base_dilations = [dilation_base ** i for i in range(levels - 1)]
    rf_without_last = calc_receptive_field(kernel_size, base_dilations)
    last_dilation = int(np.ceil(
        (nrecept_target - rf_without_last) / (2.0 * (kernel_size - 1))))
    last_dilation = max(last_dilation, 1)
    dilation_sizes = base_dilations + [last_dilation]
    nrecept = calc_receptive_field(kernel_size, dilation_sizes)

    model = TCN(input_channels, n_classes, channel_sizes,
                kernel_size=kernel_size, dropout=dropout,
                dilation_size=dilation_sizes)
    return model, nrecept, dilation_sizes


# ═════════════════════════════════════════════════════════════════════════
#  Distributed Stratified Batch Sampler
# ═════════════════════════════════════════════════════════════════════════

class DistributedStratifiedBatchSampler:
    """Stratified batch sampler that shards batches across DDP ranks.

    All ranks generate the same deterministic sequence of balanced batches
    (same seed), then each rank takes every ``world_size``-th batch
    starting from its ``rank``.  This guarantees no overlap and full
    coverage of the majority class each epoch.
    """

    def __init__(self, labels, indices, batch_size,
                 rank: int, world_size: int,
                 drop_last: bool = True, seed: int = 42):
        self._inner = StratifiedBatchSampler(
            labels, indices, batch_size, drop_last, seed)
        self.rank = rank
        self.world_size = world_size
        # expose for weight recomputation
        self.pos_idx = self._inner.pos_idx
        self.neg_idx = self._inner.neg_idx
        self.batch_size = batch_size

    def set_epoch(self, epoch: int):
        self._inner.set_epoch(epoch)

    def __iter__(self):
        all_batches = list(self._inner)
        for i in range(self.rank, len(all_batches), self.world_size):
            yield all_batches[i]

    def __len__(self):
        total = len(self._inner)
        return total // self.world_size + (
            1 if self.rank < total % self.world_size else 0)


# ═════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════

def batch_weights(tgt_v: torch.Tensor) -> torch.Tensor:
    """Per-timestep class weights from actual label ratio in this batch."""
    n_total = tgt_v.numel()
    n_pos = tgt_v.sum()
    n_neg = n_total - n_pos
    pw = 0.5 * n_total / n_pos.clamp(min=1)
    nw = 0.5 * n_total / n_neg.clamp(min=1)
    return torch.where(tgt_v == 1, pw, nw)


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def log(rank: int, msg: str):
    """Print only on rank 0."""
    if rank == 0:
        print(msg, flush=True)


# ═════════════════════════════════════════════════════════════════════════
#  Training
# ═════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, nrecept, device, epoch,
                    n_epochs, clip, rank, log_every=5):
    model.train()
    n_batches = len(loader)
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_pos = 0

    for batch_idx, (X, target, _weight) in enumerate(loader):
        B = X.shape[0]
        X = X.view(B, -1, X.shape[-1]).to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(X)

        out_v = output[:, nrecept - 1:]
        tgt_v = target[:, nrecept - 1:]
        wgt_v = batch_weights(tgt_v)

        loss = F.binary_cross_entropy(out_v, tgt_v, weight=wgt_v)
        loss.backward()

        gn_before = grad_norm(model.module)
        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        gn_after = grad_norm(model.module)

        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss

        with torch.no_grad():
            pred = (out_v >= 0.5).float()
            running_correct += (pred == tgt_v).sum().item()
            running_total += tgt_v.numel()
            running_pos += tgt_v.sum().item()

        if rank == 0 and ((batch_idx + 1) % log_every == 0
                          or (batch_idx + 1) == n_batches):
            avg_loss = running_loss / (batch_idx + 1)
            avg_acc = running_correct / max(running_total, 1)
            pos_frac = running_pos / max(running_total, 1)
            lr_now = optimizer.param_groups[0]['lr']
            print(f'  [{epoch}/{n_epochs}] '
                  f'batch {batch_idx+1:>4d}/{n_batches}  '
                  f'loss={batch_loss:.4e}  avg_loss={avg_loss:.4e}  '
                  f'acc={avg_acc:.4f}  pos%={pos_frac:.3f}  '
                  f'|grad|={gn_before:.3f}->{gn_after:.3f}  '
                  f'lr={lr_now:.2e}', flush=True)

    # ── all-reduce for epoch-level metrics ──
    stats = torch.tensor(
        [running_loss, running_correct, running_total, running_pos,
         n_batches],
        device=device, dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    r_loss, r_correct, r_total, r_pos, r_nbatch = stats.tolist()

    return {
        'loss': r_loss / max(r_nbatch, 1),
        'accuracy': r_correct / max(r_total, 1),
        'pos_frac': r_pos / max(r_total, 1),
    }


# ═════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, nrecept, device, thresholds=None):
    model.eval()
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    total_loss = 0.0
    n_batches = len(loader)
    total = 0
    correct_50 = 0
    total_pos = 0
    TPs = np.zeros(len(thresholds))
    TNs = np.zeros(len(thresholds))
    FPs = np.zeros(len(thresholds))
    FNs = np.zeros(len(thresholds))

    for X, target, _weight in loader:
        B = X.shape[0]
        X = X.view(B, -1, X.shape[-1]).to(device)
        target = target.to(device)

        output = model(X)
        out_v = output[:, nrecept - 1:]
        tgt_v = target[:, nrecept - 1:]
        wgt_v = batch_weights(tgt_v)

        loss = F.binary_cross_entropy(out_v, tgt_v, weight=wgt_v)
        total_loss += loss.item()
        total += tgt_v.numel()
        total_pos += tgt_v.sum().item()

        pred_50 = (out_v >= 0.5).float()
        correct_50 += (pred_50 == tgt_v).sum().item()

        for i, th in enumerate(thresholds):
            pred = (out_v >= th).float()
            TPs[i] += ((pred == 1) & (tgt_v == 1)).sum().item()
            TNs[i] += ((pred == 0) & (tgt_v == 0)).sum().item()
            FPs[i] += ((pred == 1) & (tgt_v == 0)).sum().item()
            FNs[i] += ((pred == 0) & (tgt_v == 1)).sum().item()

    # ── all-reduce across ranks ──
    scalars = torch.tensor(
        [total_loss, total, correct_50, total_pos, n_batches],
        device=device, dtype=torch.float64,
    )
    dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
    total_loss, total, correct_50, total_pos, nb_total = scalars.tolist()

    for arr_name, arr in [('tp', TPs), ('tn', TNs), ('fp', FPs), ('fn', FNs)]:
        t = torch.tensor(arr, device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if arr_name == 'tp':
            TPs = t.cpu().numpy()
        elif arr_name == 'tn':
            TNs = t.cpu().numpy()
        elif arr_name == 'fp':
            FPs = t.cpu().numpy()
        else:
            FNs = t.cpu().numpy()

    avg_loss = total_loss / max(nb_total, 1)
    precision = TPs / (TPs + FPs + 1e-10)
    recall = TPs / (TPs + FNs + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1))
    accuracy = (TPs[best_idx] + TNs[best_idx]) / max(total, 1)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'acc_at_50': correct_50 / max(total, 1),
        'f1': f1[best_idx],
        'precision': precision[best_idx],
        'recall': recall[best_idx],
        'threshold': thresholds[best_idx],
        'pos_frac': total_pos / max(total, 1),
        'n_timesteps': int(total),
    }


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Distributed TCN training for disruption prediction')

    # ── data (defaults: SciServer paths) ─────────────────────────────
    g = p.add_argument_group('data')
    g.add_argument('--root', type=str,
                   default='/home/idies/workspace/Storage/yhuang2/persistent/ecei/dsrpt')
    g.add_argument('--decimated-root', type=str,
                   default='/home/idies/workspace/Storage/yhuang2/persistent/ecei/dsrpt_decimated')
    g.add_argument('--clear-root', type=str,
                   default='/home/idies/workspace/Storage/yhuang2/persistent/ecei/clear_decimated',
                   help='Directory with non-disruptive shots (meta + .h5); use clear_decimated to load pre-decimated')
    g.add_argument('--clear-decimated-root', type=str,
                   default='/home/idies/workspace/Storage/yhuang2/persistent/ecei/clear_decimated',
                   help='Pre-decimated clear shots (same as clear-root => use decimated clear)')
    g.add_argument('--data-step', type=int, default=10)
    g.add_argument('--twarn', type=int, default=300_000,
                   help='Label as disruptive within this many samples (1 MHz) before t_disrupt (300_000 = 300 ms)')
    g.add_argument('--exclude-last-ms', type=float, default=0.0,
                   help='Do not label last N ms before disruption as 1 (e.g. 30 for mitigation; reduces FPs)')
    g.add_argument('--ignore-twarn', action='store_true',
                   help='Do not train on the Twarn window (weight=0); learn disruptive vs clear from data')
    g.add_argument('--baseline-len', type=int, default=40_000)
    g.add_argument('--nsub', type=int, default=781_250)
    g.add_argument('--prebuilt-subseq-dir', type=str, default=None,
                   help='Use pre-saved subsequence .npz from preprocess_subseqs.py (avoids shm)')

    # ── model ──
    g = p.add_argument_group('model')
    g.add_argument('--input-channels', type=int, default=160)
    g.add_argument('--levels', type=int, default=4)
    g.add_argument('--nhid', type=int, default=80)
    g.add_argument('--kernel-size', type=int, default=15)
    g.add_argument('--dilation-base', type=int, default=10)
    g.add_argument('--dropout', type=float, default=0.1)
    g.add_argument('--nrecept-target', type=int, default=30_000)

    # ── training ──
    g = p.add_argument_group('training')
    g.add_argument('--epochs', type=int, default=200,
                   help='Total epochs (default: 200; with 4 GPUs each epoch '
                        'has ~1/4 the gradient updates of single-GPU)')
    g.add_argument('--batch-size', type=int, default=48,
                   help='Per-GPU batch size (default: 48, eff. 192 on 4 GPUs)')
    g.add_argument('--num-workers', type=int, default=0,
                   help='DataLoader workers per rank (default: 0 to avoid shm; use 2–4 if /dev/shm is large)')
    g.add_argument('--optimizer', type=str, default='adamw',
                   choices=['adamw', 'sgd'],
                   help='Optimizer (default: adamw)')
    g.add_argument('--lr', type=float, default=None,
                   help='Learning rate (default: 1e-3 for AdamW, 0.5 for SGD)')
    g.add_argument('--weight-decay', type=float, default=1e-4)
    g.add_argument('--momentum', type=float, default=0.9,
                   help='SGD momentum (ignored for AdamW)')
    g.add_argument('--clip', type=float, default=0.3)
    g.add_argument('--warmup-epochs', type=int, default=20,
                   help='Warmup epochs (default: 20; ~5 single-GPU-equiv epochs)')
    g.add_argument('--warmup-factor', type=int, default=8)
    g.add_argument('--log-every', type=int, default=5)

    # ── checkpointing ──
    g = p.add_argument_group('checkpointing')
    g.add_argument('--checkpoint-dir', type=str, default='checkpoints_tcn_ddp')
    g.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')

    return p.parse_args()


def main():
    args = parse_args()

    # ── DDP initialisation ───────────────────────────────────────────────
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # set default LR based on optimizer choice
    # NOTE: for AdamW, LR does NOT need linear scaling with batch size.
    # Keep at 1e-3, same as the single-GPU baseline that achieved 0.91 F1.
    if args.lr is None:
        args.lr = 1e-3 if args.optimizer == 'adamw' else 0.5

    eff_batch = args.batch_size * world_size
    log(rank, '=' * 90)
    log(rank, f'  Distributed TCN Training')
    log(rank, f'  GPUs: {world_size}  |  batch/GPU: {args.batch_size}  |  '
              f'effective batch: {eff_batch}')
    log(rank, f'  Optimizer: {args.optimizer.upper()}  lr={args.lr}  '
              f'wd={args.weight_decay}  clip={args.clip}')
    log(rank, f'  Epochs: {args.epochs}  |  warmup: {args.warmup_epochs} ep')
    log(rank, '=' * 90)

    # ── Build model ──────────────────────────────────────────────────────
    model, nrecept, dilation_sizes = build_model(
        args.input_channels, 1, args.levels, args.nhid,
        args.kernel_size, args.dilation_base, args.dropout,
        nrecept_target=args.nrecept_target,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(rank, f'  Dilations      : {dilation_sizes}')
    log(rank, f'  Receptive field: {nrecept:,} samples')
    log(rank, f'  Parameters     : {n_params:,}')

    stride = (args.nsub // args.data_step - nrecept + 1) * args.data_step
    log(rank, f'  Stride (raw)   : {stride:,}')

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # ── Dataset ──────────────────────────────────────────────────────────
    if args.prebuilt_subseq_dir:
        train_ds = PrebuiltSubseqDataset(args.prebuilt_subseq_dir, split='train')
        val_ds = PrebuiltSubseqDataset(args.prebuilt_subseq_dir, split='test')
        if rank == 0:
            log(rank, f'  Prebuilt subseqs: train={len(train_ds)}, test={len(val_ds)}')
        ds = train_ds  # for class-weight log; train_loader uses train_ds
        train_idx = np.arange(len(train_ds))
        val_idx = np.arange(len(val_ds))
    else:
        ds = ECEiTCNDataset(
            root=args.root,
            decimated_root=args.decimated_root,
            clear_root=args.clear_root,
            clear_decimated_root=args.clear_decimated_root,
            Twarn=args.twarn,
            exclude_last_ms=args.exclude_last_ms,
            ignore_twarn=args.ignore_twarn,
            baseline_length=args.baseline_len,
            data_step=args.data_step,
            nsub=args.nsub,
            stride=stride,
            normalize=True,
        )
        if rank == 0:
            ds.summary()
        train_idx = ds.get_split_indices('train')
        val_idx = ds.get_split_indices('test')
        if len(val_idx) == 0:
            val_idx = ds.get_split_indices('val')
        train_ds = ds
        val_ds = ds

    # ── Training loader: distributed stratified batches ──────────────────
    data_for_train = train_ds
    train_sampler = DistributedStratifiedBatchSampler(
        labels=data_for_train.seq_has_disrupt.astype(int),
        indices=train_idx,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
    )

    if not args.prebuilt_subseq_dir:
        n_pos = len(train_sampler.pos_idx)
        n_neg = len(train_sampler.neg_idx)
        n_eff = min(n_pos, n_neg)
        eff_indices = np.concatenate([
            train_sampler.pos_idx[:n_eff],
            train_sampler.neg_idx[:n_eff],
        ])
        old_pw, old_nw = ds.pos_weight, ds.neg_weight
        ds._compute_class_weights(indices=eff_indices)
        log(rank, f'  Class weights (after stratified balance):')
        log(rank, f'    before: pw={old_pw:.4f} nw={old_nw:.4f}')
        log(rank, f'    after : pw={ds.pos_weight:.4f} nw={ds.neg_weight:.4f}')

    train_kw = dict(
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        train_kw["prefetch_factor"] = 2
        train_kw["persistent_workers"] = True
        train_kw["worker_init_fn"] = _worker_init_fn
    train_loader = DataLoader(data_for_train, **train_kw)

    # ── Validation loader: distributed across ranks ──────────────────────
    val_subset = Subset(val_ds, val_idx)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_subset, shuffle=False)
    val_kw = dict(
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if args.num_workers > 0:
        val_kw["prefetch_factor"] = 2
        val_kw["persistent_workers"] = True
        val_kw["worker_init_fn"] = _worker_init_fn
    val_loader = DataLoader(val_subset, **val_kw)

    log(rank, f'  Train: {len(train_loader)} batches/rank  '
              f'({len(train_idx)} subseqs total)')
    log(rank, f'  Val  : {len(val_loader)} batches/rank  '
              f'({len(val_idx)} subseqs total)')

    # ── Optimizer ────────────────────────────────────────────────────────
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, nesterov=True,
                              weight_decay=args.weight_decay)

    # ── Schedulers ───────────────────────────────────────────────────────
    warmup_iters = args.warmup_epochs * len(train_loader)
    warmup_lambda = (lambda it:
                     (1 - 1 / args.warmup_factor) / max(warmup_iters, 1) * it
                     + 1 / args.warmup_factor)
    scheduler_warmup = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lambda)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5)  # patience=10 default

    # ── Checkpoint directory ─────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    if rank == 0:
        ckpt_dir.mkdir(exist_ok=True)
    dist.barrier()

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_epoch = 1
    best_f1 = 0.0
    global_step = 0

    # initialise empty history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_threshold': [],
        'lr': [],
    }

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        # model & optimizer
        model.module.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

        # schedulers (restore internal counters, patience, best, etc.)
        if 'scheduler_warmup' in ckpt:
            scheduler_warmup.load_state_dict(ckpt['scheduler_warmup'])
        if 'scheduler_plateau' in ckpt:
            scheduler_plateau.load_state_dict(ckpt['scheduler_plateau'])

        # counters
        start_epoch = ckpt.get('epoch', 0) + 1
        best_f1 = ckpt.get('best_f1', 0.0)
        global_step = ckpt.get('global_step', 0)

        # restore training history so curves are continuous
        if 'history' in ckpt:
            history = ckpt['history']
        elif rank == 0 and (ckpt_dir / 'history.json').exists():
            # fallback: load from JSON written by a prior run
            with open(ckpt_dir / 'history.json') as f:
                history = json.load(f)

        log(rank, f'  Resumed from {args.resume}')
        log(rank, f'    epoch      = {start_epoch - 1}  →  continuing at {start_epoch}')
        log(rank, f'    best_f1    = {best_f1:.4f}')
        log(rank, f'    global_step= {global_step:,}')
        log(rank, f'    LR         = {optimizer.param_groups[0]["lr"]:.2e}')
        log(rank, f'    history pts= {len(history.get("train_loss", []))}')

    log(rank, '=' * 90)

    # ═════════════════════════════════════════════════════════════════════
    #  Training loop
    # ═════════════════════════════════════════════════════════════════════

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        log(rank, f'\n{"─" * 90}')
        log(rank, f'  EPOCH {epoch}/{args.epochs}   '
                  f'(global_step={global_step:,}  '
                  f'lr={optimizer.param_groups[0]["lr"]:.2e})')
        log(rank, f'{"─" * 90}')

        # ── TRAIN ────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, nrecept, device,
            epoch=epoch, n_epochs=args.epochs, clip=args.clip,
            rank=rank, log_every=args.log_every,
        )
        global_step += len(train_loader)

        if global_step <= warmup_iters:
            scheduler_warmup.step(global_step)

        # ── VALIDATE ─────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, nrecept, device)

        if global_step > warmup_iters:
            scheduler_plateau.step(val_metrics['loss'])

        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.perf_counter() - t0

        # ── Record history (rank 0) ─────────────────────────────────────
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_threshold'].append(val_metrics['threshold'])
        history['lr'].append(lr_now)

        # ── Checkpoint (rank 0 only) ────────────────────────────────────
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']

        if rank == 0:
            state = {
                'epoch': epoch,
                'global_step': global_step,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler_warmup': scheduler_warmup.state_dict(),
                'scheduler_plateau': scheduler_plateau.state_dict(),
                'best_f1': best_f1,
                'threshold': val_metrics['threshold'],
                'nrecept': nrecept,
                'history': history,
                'args': vars(args),
            }
            torch.save(state, ckpt_dir / 'last.pt')
            if is_best:
                torch.save(state, ckpt_dir / 'best.pt')

        dist.barrier()

        # ── Epoch summary ───────────────────────────────────────────────
        star = '  ** NEW BEST **' if is_best else ''
        log(rank, f'\n  EPOCH {epoch}/{args.epochs} SUMMARY  '
                  f'({elapsed:.1f}s){star}')
        log(rank, f'  ┌────────────────────────────────────────────'
                  f'───────────────────┐')
        log(rank, f'  │  Train loss    : {train_metrics["loss"]:.6e}'
                  f'                │')
        log(rank, f'  │  Train acc@0.5 : {train_metrics["accuracy"]:.4f}'
                  f'    pos%: {train_metrics["pos_frac"]:.3f}'
                  f'           │')
        log(rank, f'  │  ─────────────────────────────────────────'
                  f'────────────────  │')
        log(rank, f'  │  Val   loss    : {val_metrics["loss"]:.6e}'
                  f'                │')
        log(rank, f'  │  Val   acc@th  : {val_metrics["accuracy"]:.4f}'
                  f'    acc@0.5: {val_metrics["acc_at_50"]:.4f}'
                  f'          │')
        log(rank, f'  │  Val   F1      : {val_metrics["f1"]:.4f}'
                  f'    P={val_metrics["precision"]:.4f}'
                  f'  R={val_metrics["recall"]:.4f}'
                  f'  th={val_metrics["threshold"]:.2f}'
                  f'  │')
        log(rank, f'  │  LR            : {lr_now:.2e}'
                  f'                           │')
        log(rank, f'  │  Best F1 so far: {best_f1:.4f}'
                  f'                              │')
        log(rank, f'  └────────────────────────────────────────────'
                  f'───────────────────┘')

    # ── Save training history ────────────────────────────────────────────
    if rank == 0:
        # Convert numpy floats for JSON serialisation
        hist_out = {k: [float(v) for v in vs]
                    for k, vs in history.items()}
        with open(ckpt_dir / 'history.json', 'w') as f:
            json.dump(hist_out, f, indent=2)
        log(rank, f'\nHistory saved to {ckpt_dir / "history.json"}')

    log(rank, f'\n{"═" * 90}')
    log(rank, f'  TRAINING COMPLETE — {args.epochs} epochs, '
              f'best val F1 = {best_f1:.4f}')
    log(rank, f'{"═" * 90}')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
