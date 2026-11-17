"""
predict.py -- classify a single new image with the trained model.

CONCEPT: the model's raw output ("logits") are just unbounded numbers, one
per class, where bigger = more confident. To turn them into something
human-readable ("87% Pikachu, 6% Raichu, ..."), we apply softmax, which
exponentiates and normalizes the logits so they sum to 1.0 and can be read
as probabilities.

Usage:
    python3 src/predict.py path/to/some_pokemon.jpg
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import build_transforms, get_device
from model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--class-names", default="models/class_names.json")
    parser.add_argument("--config", default="models/config.json",
                         help="tells us which architecture the checkpoint was trained with")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    with open(args.class_names) as f:
        class_names = json.load(f)
    with open(args.config) as f:
        config = json.load(f)

    model = build_model(config["arch"], num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    transform = build_transforms(train=False)
    tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension: (3,H,W) -> (1,3,H,W)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probabilities.topk(args.top_k)

    print(f"Predictions for {args.image_path}:")
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        print(f"  {class_names[idx]:15s} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
