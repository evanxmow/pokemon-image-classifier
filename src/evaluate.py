"""
evaluate.py -- measure the trained model on the validation set in more detail
than the plain accuracy number train.py already prints.

CONCEPT: overall accuracy can hide a lot. A model might be great at
Pikachu-vs-everything but consistently confuse Nidoran-f/Nidoran-m, or
Weedle/Kakuna (visually similar sprites). A confusion matrix / per-class
breakdown shows you exactly which Pokemon the model struggles to tell apart,
which is far more actionable than one accuracy percentage.

Usage:
    python3 src/evaluate.py --checkpoint models/best_model.pt
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import load_datasets, get_device
from model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/kanto151")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--config", default="models/config.json",
                         help="tells us which architecture the checkpoint was trained with")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-mistakes", type=int, default=15)
    args = parser.parse_args()

    device = get_device()
    _, val_dataset, class_names = load_datasets(args.data_root)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    with open(args.config) as f:
        config = json.load(f)

    model = build_model(config["arch"], num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusions = defaultdict(int)  # (true_name, predicted_name) -> count
    total_correct, total_seen = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)

            for true_idx, pred_idx in zip(labels.tolist(), predictions.tolist()):
                true_name, pred_name = class_names[true_idx], class_names[pred_idx]
                per_class_total[true_name] += 1
                if true_name == pred_name:
                    per_class_correct[true_name] += 1
                    total_correct += 1
                else:
                    confusions[(true_name, pred_name)] += 1
                total_seen += 1

    print(f"Overall val accuracy: {total_correct / total_seen:.3f}  ({total_correct}/{total_seen})\n")

    print("Per-class accuracy (worst 15):")
    per_class_acc = {
        name: per_class_correct[name] / per_class_total[name]
        for name in per_class_total
    }
    for name, acc in sorted(per_class_acc.items(), key=lambda kv: kv[1])[:15]:
        print(f"  {name:15s} {acc:.2f}  ({per_class_correct[name]}/{per_class_total[name]})")

    print(f"\nTop {args.top_mistakes} most common confusions (true -> predicted):")
    for (true_name, pred_name), count in sorted(confusions.items(), key=lambda kv: -kv[1])[:args.top_mistakes]:
        print(f"  {true_name:15s} -> {pred_name:15s}  x{count}")


if __name__ == "__main__":
    main()
