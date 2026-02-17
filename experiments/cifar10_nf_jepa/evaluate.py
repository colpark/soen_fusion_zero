#!/usr/bin/env python3
"""
Evaluation for CIFAR-10 NF+JEPA.

Two evaluation modes:
  1. Reconstruction: measure MSE / PSNR at various obs_frac values.
  2. Linear probe: freeze encoder → train linear classifier on CIFAR-10 labels.

Usage:
    # Reconstruction evaluation
    python evaluate.py recon --checkpoint output/cifar10_nf_jepa/checkpoint_latest.pth

    # Linear probe
    python evaluate.py probe --checkpoint output/cifar10_nf_jepa/checkpoint_latest.pth

    # Reconstruction ablation over obs_frac
    python evaluate.py recon --checkpoint output/cifar10_nf_jepa/checkpoint_latest.pth \
        --obs_fracs 0.01 0.05 0.1 0.2 0.5
"""

import argparse
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import (
    image_to_coord_value,
    make_nf_jepa_model,
    sample_sparse_observations,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_config(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]

    model = make_nf_jepa_model(
        d_model=cfg.get("d_model", 192),
        n_layers=cfg.get("n_layers", 4),
        num_freqs=cfg.get("num_freqs", 32),
        use_mamba=cfg.get("use_mamba", True),
        nhead=cfg.get("nhead", 4),
        pred_dim=cfg.get("pred_dim", 96),
        pred_layers=cfg.get("pred_layers", 3),
        pred_mode=cfg.get("pred_mode", "cross_attn"),
        use_inr_decoder=cfg.get("use_inr_decoder", True),
        inr_hidden=cfg.get("inr_hidden", 256),
        inr_layers=cfg.get("inr_layers", 4),
    )

    state = ckpt["model"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    model.to(device).eval()

    logger.info(f"Loaded model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model, cfg


def make_test_loader(data_root, batch_size, num_workers=4):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


def compute_psnr(mse_val):
    if mse_val == 0:
        return float("inf")
    return -10 * math.log10(mse_val)


# ---------------------------------------------------------------------------
# Reconstruction evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_reconstruction(model, loader, obs_frac, device, max_batches=None):
    """Evaluate INR reconstruction quality at a given obs_frac."""
    if model.inr_decoder is None:
        logger.error("Model has no INR decoder — cannot evaluate reconstruction.")
        return {}

    mse_sum, n_total = 0.0, 0
    for i, (imgs, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        imgs = imgs.to(device)
        B = imgs.size(0)
        all_coords, all_values = image_to_coord_value(imgs)

        ctx_coords, ctx_values, _ = sample_sparse_observations(
            all_coords, all_values, obs_frac,
        )
        _, z_global = model.forward_context(ctx_coords, ctx_values)
        pred_rgb = model.forward_reconstruct(z_global, all_coords).clamp(0, 1)

        mse_sum += F.mse_loss(pred_rgb, all_values, reduction="sum").item()
        n_total += B * all_values.size(1) * all_values.size(2)

    mse = mse_sum / max(n_total, 1)
    psnr = compute_psnr(mse)
    return {"obs_frac": obs_frac, "mse": mse, "psnr_db": psnr}


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features_nf(encoder, loader, obs_frac, device):
    """Extract mean-pooled encoder features for all images."""
    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        all_coords, all_values = image_to_coord_value(imgs)
        ctx_coords, ctx_values, _ = sample_sparse_observations(
            all_coords, all_values, obs_frac,
        )
        _, z_global = encoder(ctx_coords, ctx_values)
        all_feats.append(z_global.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_linear_probe(train_feats, train_labels, test_feats, test_labels,
                       embed_dim, epochs=100, lr=0.1, batch_size=256, device="cpu"):
    classifier = nn.Linear(embed_dim, 10).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
    )

    best_acc = 0.0
    for epoch in range(epochs):
        classifier.train()
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            loss = F.cross_entropy(classifier(feats), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        classifier.eval()
        with torch.no_grad():
            logits = classifier(test_feats.to(device))
            acc = (logits.argmax(1) == test_labels.to(device)).float().mean().item() * 100
            best_acc = max(best_acc, acc)

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Probe epoch {epoch+1}: acc={acc:.2f}%  best={best_acc:.2f}%")

    return best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # Reconstruction
    recon_p = subparsers.add_parser("recon", help="Reconstruction evaluation")
    recon_p.add_argument("--checkpoint", type=str, required=True)
    recon_p.add_argument("--data_root", type=str, default="./data")
    recon_p.add_argument("--batch_size", type=int, default=128)
    recon_p.add_argument("--obs_fracs", type=float, nargs="+",
                         default=[0.01, 0.05, 0.1, 0.2, 0.5])
    recon_p.add_argument("--max_batches", type=int, default=None)

    # Linear probe
    probe_p = subparsers.add_parser("probe", help="Linear probe evaluation")
    probe_p.add_argument("--checkpoint", type=str, required=True)
    probe_p.add_argument("--data_root", type=str, default="./data")
    probe_p.add_argument("--batch_size", type=int, default=256)
    probe_p.add_argument("--obs_frac", type=float, default=0.1)
    probe_p.add_argument("--probe_epochs", type=int, default=100)
    probe_p.add_argument("--probe_lr", type=float, default=0.1)

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    torch.manual_seed(42)
    logger.info(f"Device: {device}")

    model, cfg = load_model_and_config(args.checkpoint, device)

    if args.mode == "recon":
        loader = make_test_loader(args.data_root, args.batch_size)
        logger.info("=== Reconstruction evaluation ===")
        results = []
        for frac in args.obs_fracs:
            r = evaluate_reconstruction(model, loader, frac, device, args.max_batches)
            results.append(r)
            logger.info(f"  obs_frac={frac:.3f}  MSE={r['mse']:.6f}  PSNR={r['psnr_db']:.2f} dB")

        logger.info("\n--- Summary ---")
        logger.info(f"{'obs_frac':>10s}  {'MSE':>10s}  {'PSNR (dB)':>10s}")
        for r in results:
            logger.info(f"{r['obs_frac']:10.3f}  {r['mse']:10.6f}  {r['psnr_db']:10.2f}")

    elif args.mode == "probe":
        logger.info(f"=== Linear probe (obs_frac={args.obs_frac}) ===")

        # Train & test loaders (with label info)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_ds = torchvision.datasets.CIFAR10(
            root=args.data_root, train=True, download=True, transform=train_transform,
        )
        test_ds = torchvision.datasets.CIFAR10(
            root=args.data_root, train=False, download=True, transform=test_transform,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        logger.info("Extracting train features...")
        train_feats, train_labels = extract_features_nf(
            model.encoder, train_loader, args.obs_frac, device,
        )
        logger.info("Extracting test features...")
        test_feats, test_labels = extract_features_nf(
            model.encoder, test_loader, args.obs_frac, device,
        )
        logger.info(f"Train: {train_feats.shape}  Test: {test_feats.shape}")

        best_acc = train_linear_probe(
            train_feats, train_labels, test_feats, test_labels,
            embed_dim=cfg.get("d_model", 192),
            epochs=args.probe_epochs, lr=args.probe_lr,
            batch_size=args.batch_size, device=device,
        )
        logger.info(f"\nBest linear probe accuracy: {best_acc:.2f}%")
        logger.info("(Random baseline = 10%.)")


if __name__ == "__main__":
    main()
