#!/usr/bin/env python3
"""
Distributed TCN training on PCA-8 + 100× decimated + flattop H5 dataset.

Data comes from preprocessing_pca8_100x_flattop.ipynb → single H5 file with
pre-subsequenced (8, 7812) chunks, target, weight, labels per split.
The entire dataset is loaded into memory (small after 100× + PCA).

Model matches the PCA1 baseline (run_tcn_baseline_pca1_decimated_subsample.sh)
but with input_channels=8.

Usage:
    bash run_tcn_pca8_100x_flattop.sh
    torchrun --nproc_per_node=4 train_tcn_pca8_h5.py [OPTIONS]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import h5py

# Reuse model definitions and helpers from existing training script
from train_tcn_ddp_original import (
    build_model,
    calc_receptive_field,
    batch_weights,
    grad_norm,
    DistributedStratifiedBatchSampler,
)
from dataset_ecei_tcn import StratifiedBatchSampler


# ═════════════════════════════════════════════════════════════════════════
#  Dataset: in-memory from single H5
# ═════════════════════════════════════════════════════════════════════════

class EceiPCA8Dataset(Dataset):
    """Load entire PCA-8 100× flattop H5 into memory.

    H5 layout: /{split}/X (N, 8, T), target (N, T), weight (N, T), labels (N,)
    """

    def __init__(self, h5_path: str, split: str = "train"):
        with h5py.File(h5_path, "r") as f:
            g = f[split]
            self.X = torch.from_numpy(np.asarray(g["X"]))          # (N, 8, T)
            self.target = torch.from_numpy(np.asarray(g["target"]))  # (N, T)
            self.weight = torch.from_numpy(np.asarray(g["weight"]))  # (N, T)
            self.labels = np.asarray(g["labels"])                    # (N,)
            self.pos_weight = float(f.attrs.get("pos_weight", 1.0))
            self.neg_weight = float(f.attrs.get("neg_weight", 1.0))

    @property
    def seq_has_disrupt(self):
        return self.labels

    def get_split_indices(self, split_name):
        """Return all indices (dataset is already a single split)."""
        return np.arange(len(self.labels))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.target[idx], self.weight[idx]


# ═════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════

def log(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


# ═════════════════════════════════════════════════════════════════════════
#  Training & Evaluation (same logic as train_tcn_ddp_original)
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
        X = X.to(device)        # (B, 8, T)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(X)       # (B, T)

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

    stats = torch.tensor(
        [running_loss, running_correct, running_total, running_pos, n_batches],
        device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    r_loss, r_correct, r_total, r_pos, r_nbatch = stats.tolist()

    return {
        'loss': r_loss / max(r_nbatch, 1),
        'accuracy': r_correct / max(r_total, 1),
        'pos_frac': r_pos / max(r_total, 1),
    }


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
        X = X.to(device)
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

    scalars = torch.tensor(
        [total_loss, total, correct_50, total_pos, n_batches],
        device=device, dtype=torch.float64)
    dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
    total_loss, total, correct_50, total_pos, nb_total = scalars.tolist()

    for arr_name, arr in [('tp', TPs), ('tn', TNs), ('fp', FPs), ('fn', FNs)]:
        t = torch.tensor(arr, device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if arr_name == 'tp': TPs = t.cpu().numpy()
        elif arr_name == 'tn': TNs = t.cpu().numpy()
        elif arr_name == 'fp': FPs = t.cpu().numpy()
        else: FNs = t.cpu().numpy()

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
#  Args
# ═════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='TCN training on PCA-8 100× flattop H5')

    # data
    g = p.add_argument_group('data')
    g.add_argument('--h5-path', type=str,
                   default='/home/idies/workspace/Storage/yhuang2/persistent/ecei_mc/pca8_100x_flattop/all_data.h5',
                   help='Path to PCA-8 H5 file')

    # model — defaults match PCA1 baseline after decimate_extra=10 scaling
    g = p.add_argument_group('model')
    g.add_argument('--input-channels', type=int, default=8)
    g.add_argument('--levels', type=int, default=4)
    g.add_argument('--nhid', type=int, default=80)
    g.add_argument('--kernel-size', type=int, default=2,
                   help='Kernel size (default: 2, same as PCA1 after 10× scaling from 15)')
    g.add_argument('--dilation-base', type=int, default=1,
                   help='Dilation base (default: 1, same as PCA1 after 10× scaling from 10)')
    g.add_argument('--dropout', type=float, default=0.1)
    g.add_argument('--nrecept-target', type=int, default=3000,
                   help='Receptive field target in 10 kHz samples (default: 3000 = 300ms)')
    g.add_argument('--use-instance-norm', action='store_true',
                   help='Use InstanceNorm1d instead of weight normalization')
    g.add_argument('--use-prenorm', action='store_true',
                   help='Use PreNorm TCN')

    # training
    g = p.add_argument_group('training')
    g.add_argument('--epochs', type=int, default=200)
    g.add_argument('--batch-size', type=int, default=48,
                   help='Per-GPU batch size')
    g.add_argument('--optimizer', type=str, default='adamw',
                   choices=['adamw', 'sgd'])
    g.add_argument('--lr', type=float, default=None)
    g.add_argument('--weight-decay', type=float, default=1e-4)
    g.add_argument('--clip', type=float, default=0.3)
    g.add_argument('--warmup-epochs', type=int, default=20)
    g.add_argument('--warmup-factor', type=int, default=8)
    g.add_argument('--min-lr', type=float, default=1e-6)
    g.add_argument('--lr-schedule', type=str, default='plateau',
                   choices=['plateau', 'cosine_warmup'])
    g.add_argument('--early-stopping-patience', type=int, default=0)
    g.add_argument('--log-every', type=int, default=5)
    g.add_argument('--batch-neg-pos-ratio', type=float, default=1)
    g.add_argument('--dist-backend', type=str, default=None)

    # checkpointing
    g = p.add_argument_group('checkpointing')
    g.add_argument('--checkpoint-dir', type=str, default='checkpoints_tcn_pca8')
    g.add_argument('--resume', type=str, default=None)

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Checkpoint dir with timestamp
    if args.resume:
        args.checkpoint_dir = str(Path(args.resume).resolve().parent)
    else:
        config_tag = f"L{args.levels}_H{args.nhid}_pca8"
        if getattr(args, 'use_prenorm', False):
            config_tag += "_prenorm"
        elif getattr(args, 'use_instance_norm', False):
            config_tag += "_instnorm"
        start_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.checkpoint_dir = str(Path(args.checkpoint_dir) / f"{config_tag}_{start_time_str}")

    # DDP init
    backend = args.dist_backend or 'nccl'
    try:
        dist.init_process_group(backend=backend)
    except RuntimeError as e:
        if 'NCCL' in str(e):
            backend = 'gloo'
            dist.init_process_group(backend=backend)
        else:
            raise
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    if args.lr is None:
        args.lr = 5e-4 if args.optimizer == 'adamw' else 0.5

    eff_batch = args.batch_size * world_size
    log(rank, '=' * 90)
    log(rank, '  TCN Training — PCA-8 + 100× decimated + flattop H5')
    log(rank, f'  GPUs: {world_size}  |  batch/GPU: {args.batch_size}  |  eff batch: {eff_batch}')
    log(rank, f'  Optimizer: {args.optimizer.upper()}  lr={args.lr}  wd={args.weight_decay}  clip={args.clip}')
    log(rank, f'  H5: {args.h5_path}')
    log(rank, '=' * 90)

    # ── Build model ──
    model, nrecept, dilation_sizes = build_model(
        args.input_channels, 1, args.levels, args.nhid,
        args.kernel_size, args.dilation_base, args.dropout,
        nrecept_target=args.nrecept_target,
        use_instance_norm=getattr(args, 'use_instance_norm', False),
        use_prenorm=getattr(args, 'use_prenorm', False),
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(rank, f'  Dilations      : {dilation_sizes}')
    log(rank, f'  Receptive field: {nrecept:,} samples  ({nrecept/10:.1f} ms at 10 kHz)')
    log(rank, f'  Parameters     : {n_params:,}')

    # Verify receptive field fits in T_SUB
    with h5py.File(args.h5_path, 'r') as f:
        T_sub = f['train/X'].shape[-1]
    valid_output = T_sub - nrecept + 1
    log(rank, f'  T_sub          : {T_sub}')
    log(rank, f'  Valid output   : {valid_output} timesteps/subseq (after receptive field trim)')
    assert nrecept <= T_sub, f'Receptive field {nrecept} > T_sub {T_sub}!'

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # ── Load dataset ──
    log(rank, '  Loading H5 into memory...')
    train_ds = EceiPCA8Dataset(args.h5_path, 'train')
    val_ds = EceiPCA8Dataset(args.h5_path, 'val')
    test_ds = EceiPCA8Dataset(args.h5_path, 'test')

    n_train_pos = int(train_ds.labels.sum())
    n_train_neg = len(train_ds) - n_train_pos
    log(rank, f'  Train: {len(train_ds)} subseqs ({n_train_pos} disruptive, {n_train_neg} clear)')
    log(rank, f'  Val:   {len(val_ds)} subseqs')
    log(rank, f'  Test:  {len(test_ds)} subseqs')

    # ── Samplers ──
    train_indices = np.arange(len(train_ds))
    val_indices = np.arange(len(val_ds))

    train_sampler = DistributedStratifiedBatchSampler(
        labels=train_ds.labels.astype(int),
        indices=train_indices,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        neg_pos_ratio=args.batch_neg_pos_ratio,
    )
    val_sampler = DistributedStratifiedBatchSampler(
        labels=val_ds.labels.astype(int),
        indices=val_indices,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        neg_pos_ratio=1,
    )

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=0)

    # ── Optimizer & Scheduler ──
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)

    warmup_iters = args.warmup_epochs * len(train_loader)
    scheduler_cosine = None
    scheduler_warmup = None
    scheduler_plateau = None

    if args.lr_schedule == 'cosine_warmup':
        total_iters = args.epochs * len(train_loader)
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_iters, eta_min=args.min_lr)
        # warmup: linearly ramp from lr/warmup_factor to lr
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / args.warmup_factor,
            total_iters=warmup_iters)
        scheduler_cosine = optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler,
                        optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=total_iters - warmup_iters,
                            eta_min=args.min_lr)],
            milestones=[warmup_iters])
    else:
        scheduler_warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / args.warmup_factor,
            total_iters=warmup_iters)
        scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10)

    # ── Checkpoint dir ──
    ckpt_dir = Path(args.checkpoint_dir)
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
    dist.barrier()

    # ── Resume ──
    start_epoch = 1
    best_f1 = 0.0
    global_step = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_threshold': [],
        'lr': [],
    }

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler_cosine' in ckpt and scheduler_cosine is not None:
            scheduler_cosine.load_state_dict(ckpt['scheduler_cosine'])
        if 'scheduler_warmup' in ckpt and scheduler_warmup is not None:
            scheduler_warmup.load_state_dict(ckpt['scheduler_warmup'])
        if 'scheduler_plateau' in ckpt and scheduler_plateau is not None:
            scheduler_plateau.load_state_dict(ckpt['scheduler_plateau'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_f1 = ckpt.get('best_f1', 0.0)
        global_step = ckpt.get('global_step', 0)
        if 'history' in ckpt:
            history = ckpt['history']
        log(rank, f'  Resumed from {args.resume} (epoch {start_epoch - 1}, best_f1={best_f1:.4f})')

    log(rank, '=' * 90)

    # ═══════════════════════════════════════════════════════════════════
    #  Training loop
    # ═══════════════════════════════════════════════════════════════════

    epochs_without_improvement = 0
    patience = args.early_stopping_patience or 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        log(rank, f'\n{"─" * 90}')
        log(rank, f'  EPOCH {epoch}/{args.epochs}   '
                  f'(global_step={global_step:,}  lr={optimizer.param_groups[0]["lr"]:.2e})')
        log(rank, f'{"─" * 90}')

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, nrecept, device,
            epoch=epoch, n_epochs=args.epochs, clip=args.clip,
            rank=rank, log_every=args.log_every)
        global_step += len(train_loader)

        if scheduler_cosine is not None:
            scheduler_cosine.step(global_step)
        else:
            if global_step <= warmup_iters:
                scheduler_warmup.step(global_step)

        # Validate
        val_metrics = evaluate(model, val_loader, nrecept, device)

        if scheduler_cosine is None and global_step > warmup_iters:
            scheduler_plateau.step(val_metrics['loss'])

        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.perf_counter() - t0

        # History
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_threshold'].append(val_metrics['threshold'])
        history['lr'].append(lr_now)

        # Checkpoint
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if rank == 0:
            state = {
                'epoch': epoch,
                'global_step': global_step,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_f1': best_f1,
                'threshold': val_metrics['threshold'],
                'nrecept': nrecept,
                'history': history,
                'args': vars(args),
            }
            if scheduler_cosine is not None:
                state['scheduler_cosine'] = scheduler_cosine.state_dict()
            else:
                state['scheduler_warmup'] = scheduler_warmup.state_dict()
                state['scheduler_plateau'] = scheduler_plateau.state_dict()
            torch.save(state, ckpt_dir / 'last.pt')
            if is_best:
                torch.save(state, ckpt_dir / 'best.pt')
        dist.barrier()

        # Summary
        star = '  ** NEW BEST **' if is_best else ''
        log(rank, f'\n  EPOCH {epoch}/{args.epochs} SUMMARY  ({elapsed:.1f}s){star}')
        log(rank, f'  ┌──────────────────────────────────────────────────────────────┐')
        log(rank, f'  │  Train loss    : {train_metrics["loss"]:.6e}                │')
        log(rank, f'  │  Train acc@0.5 : {train_metrics["accuracy"]:.4f}'
                  f'    pos%: {train_metrics["pos_frac"]:.3f}           │')
        log(rank, f'  │  ────────────────────────────────────────────────────────  │')
        log(rank, f'  │  Val   loss    : {val_metrics["loss"]:.6e}                │')
        log(rank, f'  │  Val   acc@th  : {val_metrics["accuracy"]:.4f}'
                  f'    acc@0.5: {val_metrics["acc_at_50"]:.4f}          │')
        log(rank, f'  │  Val   F1      : {val_metrics["f1"]:.4f}'
                  f'    P={val_metrics["precision"]:.4f}'
                  f'  R={val_metrics["recall"]:.4f}'
                  f'  th={val_metrics["threshold"]:.2f}  │')
        log(rank, f'  │  LR            : {lr_now:.2e}                           │')
        log(rank, f'  │  Best F1 so far: {best_f1:.4f}                              │')
        log(rank, f'  └──────────────────────────────────────────────────────────────┘')

        if patience > 0 and epochs_without_improvement >= patience:
            log(rank, f'\n  Early stopping: no val F1 improvement for {patience} epochs.')
            break

    # ── Save history ──
    if rank == 0:
        hist_out = {k: [float(v) for v in vs] for k, vs in history.items()}
        with open(ckpt_dir / 'history.json', 'w') as f:
            json.dump(hist_out, f, indent=2)
        log(rank, f'\nHistory saved to {ckpt_dir / "history.json"}')

    log(rank, f'\n{"═" * 90}')
    log(rank, f'  TRAINING COMPLETE — {args.epochs} epochs, best val F1 = {best_f1:.4f}')
    log(rank, f'{"═" * 90}')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
