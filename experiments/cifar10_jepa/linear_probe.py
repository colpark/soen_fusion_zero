#!/usr/bin/env python3
"""
Linear probe evaluation for CIFAR-10 I-JEPA.

Freezes the pretrained encoder, mean-pools patch features, and trains a
linear classifier on CIFAR-10. Reports top-1 accuracy on the test set.

Usage:
    python linear_probe.py --checkpoint output/cifar10_jepa/checkpoint_latest.pth
    python linear_probe.py --checkpoint output/cifar10_jepa/checkpoint_latest.pth --epochs 50
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import VisionTransformer

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 I-JEPA linear probe")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to JEPA checkpoint")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_dataloaders(data_root, batch_size, num_workers):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, test_loader


def load_encoder(checkpoint_path, device):
    """Load the encoder from a JEPA checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]

    encoder = VisionTransformer(
        img_size=cfg.get("crop_size", 32),
        patch_size=cfg.get("patch_size", 4),
        embed_dim=cfg.get("embed_dim", 192),
        depth=cfg.get("depth", 6),
        num_heads=cfg.get("num_heads", 3),
    )

    # Load weights (handle potential DDP prefix)
    state = ckpt.get("target_encoder", ckpt.get("encoder"))
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "")] = v
    encoder.load_state_dict(cleaned, strict=True)
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    logger.info(f"Loaded encoder from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return encoder, cfg.get("embed_dim", 192)


@torch.no_grad()
def extract_features(encoder, loader, device):
    """Mean-pool patch features for all samples."""
    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = encoder(imgs)  # (B, N, D)
        feats = feats.mean(dim=1)  # global average pool → (B, D)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    encoder, embed_dim = load_encoder(args.checkpoint, device)
    train_loader, test_loader = make_dataloaders(args.data_root, args.batch_size, args.num_workers)

    # Extract features
    logger.info("Extracting features...")
    train_feats, train_labels = extract_features(encoder, train_loader, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)
    logger.info(f"Train features: {train_feats.shape}  Test features: {test_feats.shape}")

    # Linear classifier
    classifier = nn.Linear(embed_dim, 10).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_feat_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0
        for feats, labels in train_feat_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        scheduler.step()

        train_acc = correct / total * 100

        # Test
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(test_feats.to(device))
            test_acc = (test_logits.argmax(1) == test_labels.to(device)).float().mean().item() * 100
            best_acc = max(best_acc, test_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"[Probe ep {epoch+1}/{args.epochs}] "
                f"train_loss={total_loss/total:.4f}  train_acc={train_acc:.2f}%  "
                f"test_acc={test_acc:.2f}%  best={best_acc:.2f}%"
            )

    logger.info(f"\nFinal test accuracy: {test_acc:.2f}%  Best: {best_acc:.2f}%")
    logger.info("(Random baseline = 10%.  A working JEPA should reach >40% with ViT-Tiny in ~100 epochs.)")


if __name__ == "__main__":
    main()
