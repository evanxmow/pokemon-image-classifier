"""
train.py -- the training loop, with support for two architectures:

  --arch scratch    the original from-scratch CNN (model.SimplePokeCNN)
  --arch resnet18   transfer learning on a pretrained ResNet18 (default,
                     and the one you should reach for if you want higher
                     accuracy -- see model.py for why)

CONCEPTS:

* An "epoch" is one full pass over the training dataset.

* Within an epoch, data is split into "batches" (e.g. 32 images at a time).
  Each batch:
    1. Forward pass: images -> model -> logits (predictions)
    2. Loss: nn.CrossEntropyLoss compares logits to the true labels and
       produces a single number -- how wrong the model was on this batch.
    3. Backward pass (`loss.backward()`): PyTorch's autograd automatically
       computes the gradient of the loss with respect to every weight in
       the model -- backpropagation.
    4. Optimizer step (`optimizer.step()`): nudges every weight a small
       amount (controlled by the learning rate) in the direction that
       reduces the loss -- gradient descent.

* We track BOTH training accuracy and validation accuracy every epoch.
    - Training accuracy climbing while validation accuracy stalls or drops
      is the signature of OVERFITTING. Dropout and data augmentation fight
      this.
    - We save the checkpoint with the BEST validation accuracy seen so far,
      not just the final epoch's.

* LEARNING RATE SCHEDULE (new): rather than using one fixed learning rate
  for the whole run, we use `CosineAnnealingLR`, which smoothly decreases
  the learning rate from its starting value down towards 0 over the course
  of training. Early on, a larger learning rate lets the model make fast
  progress; late in training, a smaller learning rate lets it settle into a
  better minimum instead of overshooting it and oscillating (which is what
  you may have noticed happening in the last few epochs of your first run).

* TWO-PHASE TRANSFER LEARNING (only when --arch resnet18):
    Phase 1, for --freeze-epochs epochs: ResNet18's pretrained layers are
    frozen (untouched); only the new final layer trains, using --lr.
    Phase 2, for the remaining epochs: everything unfreezes and the whole
    network fine-tunes together, using the much smaller --finetune-lr, so
    the pretrained features get nudged gently rather than overwritten.

Usage:
    python3 src/train.py --epochs 20
    python3 src/train.py --arch scratch --epochs 60       # the old approach
    python3 src/train.py --epochs 15 --freeze-epochs 4     # tweak the schedule
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_datasets, get_device
from model import build_model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, total_correct, total_seen = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()  # PyTorch accumulates gradients by default; reset each batch

        logits = model(images)                  # forward pass
        loss = criterion(logits, labels)

        if train:
            loss.backward()      # backward pass: compute gradients
            optimizer.step()     # gradient descent: update weights

        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_seen += labels.size(0)
        total_loss += loss.item() * labels.size(0)
    torch.set_grad_enabled(True)

    return total_loss / total_seen, total_correct / total_seen


def build_optimizer_and_scheduler(model, lr, phase_epochs):
    """
    Only pass currently-trainable parameters to the optimizer -- passing
    frozen (requires_grad=False) parameters wouldn't break anything since
    they never receive gradients, but it's clearer to only hand the
    optimizer the parameters it's actually responsible for.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(phase_epochs, 1))
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/kanto151")
    parser.add_argument("--arch", choices=["scratch", "resnet18"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=3,
                         help="resnet18 only: epochs to train just the new head before unfreezing")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for phase 1 / from-scratch training")
    parser.add_argument("--finetune-lr", type=float, default=1e-4,
                         help="resnet18 only: learning rate for phase 2 (fine-tuning the whole network)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_dataset, val_dataset, class_names = load_datasets(args.data_root)
    print(f"Arch: {args.arch} | Classes: {len(class_names)} | "
          f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    # drop_last=True: BatchNorm needs more than 1 sample per batch to compute
    # a meaningful mean/variance, so we drop a trailing partial batch of size 1
    # rather than risk a crash on odd-sized datasets.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = build_model(args.arch, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()

    # --- set up phase 1 ---
    freeze_epochs = args.freeze_epochs if args.arch == "resnet18" else 0
    if freeze_epochs > 0:
        model.set_backbone_trainable(False)
        print(f"Phase 1: training only the new classifier head for {freeze_epochs} epoch(s) "
              f"(backbone frozen), lr={args.lr}")
        optimizer, scheduler = build_optimizer_and_scheduler(model, args.lr, freeze_epochs)
    else:
        optimizer, scheduler = build_optimizer_and_scheduler(model, args.lr, args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"arch": args.arch, "num_classes": len(class_names)}, f, indent=2)

    history = []
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # --- transition into phase 2 ---
        if args.arch == "resnet18" and freeze_epochs > 0 and epoch == freeze_epochs + 1:
            remaining_epochs = args.epochs - freeze_epochs
            print(f"\nPhase 2: unfreezing backbone, fine-tuning the whole network "
                  f"for {remaining_epochs} epoch(s), lr={args.finetune_lr}\n")
            model.set_backbone_trainable(True)
            optimizer, scheduler = build_optimizer_and_scheduler(model, args.finetune_lr, remaining_epochs)

        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - start

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} | "
              f"lr={current_lr:.2e} | {elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  -> new best val_acc={val_acc:.3f}, saved to {out_dir / 'best_model.pt'}")

    torch.save(model.state_dict(), out_dir / "last_model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoints saved in: {out_dir}/")


if __name__ == "__main__":
    main()
