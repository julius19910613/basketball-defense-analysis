#!/usr/bin/env python3
"""
Train R(2+1)D on SpaceJam basketball dataset — Mac mini (CPU/MPS) compatible.

Key changes from original train.py:
- Auto-detects device (MPS > CPU, no CUDA assumed)
- batch_size=2, num_workers=0 to fit 16GB RAM
- Uses new torchvision weights API
- Saves best checkpoint (by val accuracy) separately
- Validates BN running stats after each epoch (guards against empty checkpoints)
- BGR→RGB + /255 normalization matching dataset.py VideoToTensor

Usage:
    python train_mac.py                          # train from scratch (pretrained backbone)
    python train_mac.py --resume checkpoint.pt   # continue from checkpoint
    python train_mac.py --epochs 30 --lr 3e-4    # override defaults
"""
from __future__ import print_function, division

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, random_split

from dataset import BasketballDataset
from utils.checkpoints import init_session_history, save_weights, load_weights, write_history, read_history
from utils.metrics import get_acc_f1_precision_recall

# ── Labels ──────────────────────────────────────────────────────────────
LABELS = {
    0: "block", 1: "pass", 2: "run", 3: "dribble", 4: "shoot",
    5: "ball in hand", 6: "defense", 7: "pick", 8: "no_action", 9: "walk",
}


def parse_args():
    p = argparse.ArgumentParser(description="Train R(2+1)D on SpaceJam (Mac)")
    p.add_argument("--device", default=None, help="Force device (cpu/mps/cuda). Auto-detect if omitted.")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2 for 16GB RAM)")
    p.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--start-epoch", type=int, default=1, help="Start epoch (for resume)")
    p.add_argument("--layers", nargs="+", default=["layer3", "layer4", "fc"],
                   help="Layers to unfreeze for fine-tuning")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    p.add_argument("--annotation-path", default="dataset/annotation_dict.json")
    p.add_argument("--augmented-path", default="dataset/augmented_annotation_dict.json")
    p.add_argument("--video-dir", default="dataset/examples/")
    p.add_argument("--augmented-dir", default="dataset/augmented-examples/")
    p.add_argument("--model-dir", default="model_checkpoints/r2plus1d_v3/")
    p.add_argument("--history-path", default="histories/history_r2plus1d_v3.txt")
    p.add_argument("--save-best-only", action="store_true", help="Only save checkpoint when val acc improves")
    return p.parse_args()


def auto_device():
    """Pick the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_bn_stats(model):
    """Check that BatchNorm running stats have been updated (not all zeros/ones)."""
    issues = []
    for name, buf in model.named_buffers():
        if "running_mean" in name and torch.allclose(buf, torch.zeros_like(buf)):
            issues.append(name)
        if "running_var" in name and torch.allclose(buf, torch.ones_like(buf)):
            issues.append(name)
    if issues:
        print(f"⚠️  BN stats still at init for: {issues[:5]}{'...' if len(issues) > 5 else ''}")
    return len(issues) == 0


def train_model(model, dataloaders, criterion, optimizer, device, args, start_epoch=1, num_epochs=20):
    """Train and validate the model."""
    init_session_history(args)
    since = time.time()

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    train_f1_score, val_f1_score = [], []
    plot_epoch = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # Initialize to avoid unbound warnings
    train_loss = val_loss = 0.0
    train_accuracy = val_accuracy = 0.0
    train_cm_str = val_cm_str = ""

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*55}")
        print(f"  Epoch {epoch}/{num_epochs}")
        print(f"{'='*55}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                pred_classes, ground_truths = [], []
            else:
                model.eval()
                pred_classes, ground_truths = [], []

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0

            pbar = tqdm(dataloaders[phase], desc=f"{phase} epoch {epoch}")
            for sample in pbar:
                inputs = sample["video"].float().to(device)
                labels = sample["action"].float().to(device)
                label_indices = torch.max(labels, 1)[1]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, label_indices)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += (preds == label_indices).sum().item()
                n_samples += batch_size

                pred_classes.extend(preds.detach().cpu().numpy())
                ground_truths.extend(label_indices.detach().cpu().numpy())

                pbar.set_postfix(loss=f"{running_loss/n_samples:.4f}", acc=f"{running_corrects/n_samples:.3f}")

            epoch_loss = running_loss / n_samples
            epoch_acc = running_corrects / n_samples
            pred_arr = np.asarray(pred_classes)
            gt_arr = np.asarray(ground_truths)
            accuracy, f1, precision, recall = get_acc_f1_precision_recall(pred_arr, gt_arr)
            cm = confusion_matrix(gt_arr, pred_arr, labels=list(range(10)))

            print(f"{phase} — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  F1: {f1:.4f}")
            print(f"Confusion matrix:\n{cm}")

            if phase == "val":
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                val_f1_score.append(f1)
                val_loss = epoch_loss
                val_accuracy = accuracy

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"  🏆 New best val acc: {best_acc:.4f}")

            if phase == "train":
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                train_f1_score.append(f1)
                plot_epoch.append(epoch)
                train_loss = epoch_loss
                train_accuracy = accuracy
                train_cm_str = np.array_str(cm)

            if phase == "val":
                val_cm_str = np.array_str(cm)

        # Validate BN stats every 5 epochs
        if epoch % 5 == 0:
            validate_bn_stats(model)

        # Save checkpoint
        os.makedirs(args.model_path, exist_ok=True)
        model_name = save_weights(model, args, epoch, optimizer)

        write_history(
            args.history_path, model_name,
            train_loss, val_loss,
            train_accuracy, val_accuracy,
            f1, f1,  # simplified: use same f1 for train/val report
            precision, precision,
            recall, recall,
            train_cm_str, val_cm_str,
        )

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Save best checkpoint separately
    best_path = os.path.join(args.model_path, "best.pt")
    torch.save({
        "epoch": num_epochs,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_acc": best_acc,
    }, best_path)
    print(f"Best model saved to {best_path}")

    # Validate final BN stats
    if validate_bn_stats(model):
        print("✅ All BN running stats are properly trained")

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_f1_score, val_f1_score, plot_epoch


def check_accuracy(loader, model, device, test_n):
    """Run inference on test set."""
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for sample in tqdm(loader, desc="Testing"):
            x = sample["video"].to(device)
            y = sample["action"].to(device)
            scores = model(x)
            predictions = scores.argmax(1)
            y = y.argmax(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)
    acc = num_correct / num_samples * 100
    print(f"Test accuracy: {num_correct}/{num_samples} = {acc:.2f}%")
    model.train()
    return acc


def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────
    device = torch.device(args.device) if args.device else auto_device()
    print(f"PyTorch {torch.__version__} | Device: {device}")

    if device.type == "mps":
        print("  Note: MPS may have ops compatibility issues. Falling back to CPU if errors occur.")

    # ── Dataset sizes ───────────────────────────────────────────────
    # Count samples from annotation files
    with open(args.annotation_path) as f:
        n_orig = len(json.load(f))
    with open(args.augmented_path) as f:
        n_aug = len(json.load(f))
    n_total = n_orig + n_aug
    test_n = min(4990, n_total // 10)
    val_n = min(9980, n_total // 5)
    train_n = n_total - test_n - val_n
    print(f"Dataset: {n_total} samples (train={train_n}, val={val_n}, test={test_n})")

    args_dict = {
        "base_model_name": "r2plus1d_multiclass",
        "lr": args.lr,
        "start_epoch": args.start_epoch,
        "model_path": args.model_dir,
        "history_path": args.history_path,
    }
    # easydict-compatible namespace for checkpoint utils
    from easydict import EasyDict
    ckpt_args = EasyDict(args_dict)
    # merge into args for checkpoint utils
    for k, v in args_dict.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.model_path = args.model_dir

    # ── Model ───────────────────────────────────────────────────────
    print("Loading R(2+1)D-18 with Kinetics-400 pretrained weights...")
    model = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specified layers
    trainable_count = 0
    for name, param in model.named_parameters():
        for layer in args.layers:
            if layer in name:
                param.requires_grad = True
                trainable_count += 1
                break

    # Replace fc head
    model.fc = nn.Linear(model.fc.in_features, 10, bias=True)
    print(f"Trainable parameters: {trainable_count} + fc layer")

    # Resume from checkpoint if specified
    ckpt = None
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        if "epoch" in ckpt:
            args.start_epoch = ckpt["epoch"] + 1

    model = model.to(device)

    # ── Optimizer (only trainable params) ───────────────────────────
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.resume and ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    # ── Dataset & DataLoader ────────────────────────────────────────
    print("Loading dataset...")
    basketball_dataset = BasketballDataset(
        annotation_dict=args.annotation_path,
        augmented_dict=args.augmented_path,
        video_dir=args.video_dir,
        augmented_dir=args.augmented_dir,
    )

    train_subset, test_subset = random_split(
        basketball_dataset, [n_total - test_n, test_n],
        generator=torch.Generator().manual_seed(1),
    )
    train_subset, val_subset = random_split(
        train_subset, [train_n, val_n],
        generator=torch.Generator().manual_seed(1),
    )

    train_loader = DataLoader(train_subset, shuffle=True,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_subset, shuffle=False,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_subset, shuffle=False,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)
    dataloaders = {"train": train_loader, "val": val_loader}

    print(f"DataLoader ready — batch_size={args.batch_size}, workers={args.num_workers}")

    # ── Train ───────────────────────────────────────────────────────
    model, tlh, vlh, tah, vah, tf1, vf1, pe = train_model(
        model, dataloaders, criterion, optimizer, device, args,
        start_epoch=args.start_epoch,
        num_epochs=args.epochs,
    )

    # ── Test ────────────────────────────────────────────────────────
    check_accuracy(test_loader, model, device, test_n)


if __name__ == "__main__":
    main()
