# Pokemon Image Classifier (Kanto 151) — a hands-on ML learning project

This project trains a convolutional neural network (CNN), built from scratch
in PyTorch, to recognize the original 151 Kanto Pokemon from images. The goal
isn't just a working classifier — it's understanding *why* each piece works,
via heavily commented code. Read the source files in this order; each one's
docstring/comments explain the concepts as they come up:

1. `src/kanto_pokemon.py` — the canonical list of 151 Pokemon names
2. `src/prepare_data.py` — turning a raw image dump into train/val folders
3. `src/dataset.py` — images -> tensors, normalization, data augmentation
4. `src/model.py` — the CNN architecture, layer by layer
5. `src/train.py` — the training loop (forward pass, loss, backprop, gradient descent)
6. `src/evaluate.py` — per-class accuracy and confusion analysis
7. `src/predict.py` — running the trained model on a new image

## 1. Set up your environment

```bash
cd ~/Projects/pokemon-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This installs PyTorch, torchvision, and Pillow. On Apple Silicon Macs,
PyTorch's `mps` backend will automatically be used for GPU-accelerated
training (see `get_device()` in `dataset.py`) — no extra setup needed.

## 2. Get the dataset

I couldn't download this for you automatically — the environment I built this
in has network restrictions that block Kaggle/GitHub/PokeAPI. You'll need to
grab a dataset yourself (a couple minutes):

**Recommended: "7,000 Labeled Pokemon" on Kaggle**
https://www.kaggle.com/datasets/lantian773030/pokemonclassification
A well-known dataset with multiple real images per Pokemon (helpful for the
model to actually generalize rather than memorize a single sprite). You may
need a free Kaggle account to download.

**Alternative: "Original-Pokemon-151-Image-Dataset"**
https://www.kaggle.com/datasets/prestigemaster/original-pokemon-151-dataset
Purpose-built for exactly the 151 Kanto Pokemon.

Either works — `prepare_data.py` doesn't care about the exact folder
structure of what you download; it searches recursively and matches
folder/file names against the Kanto list.

Once downloaded, unzip it into:
```
~/Projects/pokemon-classifier/data/raw/
```
(any internal structure is fine, just get the images in there somewhere)

## 3. Build the train/val split

```bash
python3 src/prepare_data.py
```

This scans `data/raw/`, matches images to Kanto Pokemon names (handles case
differences, punctuation, numbered filenames), and writes an 85/15 train/val
split into `data/kanto151/train/<name>/` and `data/kanto151/val/<name>/`.
It prints how many images it found per Pokemon — check this before training;
a Pokemon with only 2-3 images will be hard for the model to learn.

## 4. Train

```bash
python3 src/train.py --epochs 20
```

By default this now uses **transfer learning** on a pretrained ResNet18
(`--arch resnet18`), not the from-scratch CNN — see "Two ways to train"
below for why. The first run will download ResNet18's pretrained ImageNet
weights (~45MB), which needs internet access.

Watch `train_acc` and `val_acc` each epoch. If `train_acc` keeps climbing
while `val_acc` plateaus or drops, that's overfitting — see the comments in
`model.py` (Dropout) and `dataset.py` (augmentation) for the mechanisms
already in place to fight it, and consider gathering more images per class.

The best checkpoint (by validation accuracy) is saved to
`models/best_model.pt`, along with `models/config.json` recording which
architecture it was trained with (so `evaluate.py`/`predict.py` automatically
use the right one).

Useful flags: `--batch-size`, `--lr`, `--epochs`, `--freeze-epochs`,
`--finetune-lr` (see below), `--arch scratch` to go back to training from
scratch.

### Two ways to train

**`--arch scratch`** (the original approach): a small CNN starting from
random weights, learning every visual feature from zero using only the
~40 images/class you have. This is what produced ~21% validation accuracy
after 20 epochs — informative to understand from first principles, but
data-hungry.

**`--arch resnet18`** (now the default): starts from a network that's
already learned general visual features from 1.4 million ImageNet photos,
and only has to learn the Pokemon-specific part on top. Training happens in
two phases, controlled by `--freeze-epochs` (default 3):
  - *Phase 1* (first `--freeze-epochs` epochs): ResNet18's pretrained layers
    are frozen; only the new final layer trains, at `--lr` (default 1e-3).
  - *Phase 2* (remaining epochs): everything unfreezes and fine-tunes
    together at the much smaller `--finetune-lr` (default 1e-4), so the
    pretrained features get nudged rather than overwritten.

Each epoch also now uses a **cosine learning-rate schedule** within its
phase, smoothly decaying the learning rate so training settles down late
rather than oscillating (which you saw in the last few epochs of the
scratch run).

Expect `--arch resnet18` to reach meaningfully higher validation accuracy
than the scratch run, often within fewer epochs, since it isn't learning
basic visual concepts from zero.

**One caveat that no amount of training fixes**: if you test on an image
from a very different visual style than your training data (e.g. a 3D
game-render or fan art, when your dataset is mostly 2D sprites/box art),
the model will likely do poorly regardless of accuracy — that's a domain
mismatch, not an undertrained model. Test with images similar in style to
what's in `data/raw/` for a fair read on how well it's actually learned.

## 5. Evaluate

```bash
python3 src/evaluate.py --checkpoint models/best_model.pt
```

Shows overall accuracy, the worst-performing classes, and the most common
confusions (e.g. does it mix up Nidoran-f/Nidoran-m?).

## 6. Predict on a new image

```bash
python3 src/predict.py path/to/some_photo_or_sprite.jpg
```

Prints the top-5 predicted Pokemon with probabilities.

## Core concepts, in the order you'll encounter them

- **Tensor**: a multi-dimensional array of numbers — how images, weights,
  and activations are all represented internally.
- **Convolution**: sliding a small learnable filter over an image to detect
  local patterns (edges, textures, shapes).
- **ReLU**: a nonlinearity (`max(0, x)`) between layers — without it, stacking
  layers would be mathematically equivalent to one single layer.
- **Pooling**: downsampling by taking the max (or average) over small
  regions, reducing computation and adding a bit of positional robustness.
- **Forward pass**: running input through the network to get a prediction.
- **Loss function**: a single number measuring how wrong the prediction was
  (`CrossEntropyLoss` for classification).
- **Backpropagation**: automatically computing how much each weight
  contributed to the error (`loss.backward()`).
- **Gradient descent**: nudging each weight slightly to reduce the loss
  (`optimizer.step()`), scaled by the **learning rate**.
- **Epoch**: one full pass through the training data. **Batch**: a small
  group of images processed together per step.
- **Overfitting**: memorizing training data instead of learning generalizable
  patterns — visible as training accuracy rising while validation accuracy
  stalls. Fought here with **dropout** and **data augmentation**.
- **Softmax**: converts raw output scores ("logits") into probabilities that
  sum to 1, for human-readable predictions.
- **Transfer learning**: reusing a network already trained on a large,
  general dataset (ResNet18 on ImageNet) instead of learning every visual
  feature from scratch on your smaller dataset — see `model.py`.
- **Learning rate schedule**: changing the learning rate over the course of
  training (here, smoothly decaying it) rather than keeping it fixed, so
  training can move fast early and settle down late.

## Scaling up later

You mentioned wanting to expand beyond the 151 Kanto Pokemon once you
understand the pipeline. To do that:

1. Add more names to the list in `src/kanto_pokemon.py` (or generalize it to
   read from a file/all-generations list).
2. Get more image data covering those Pokemon and drop it in `data/raw/`.
3. Re-run `prepare_data.py` and `train.py` — nothing else needs to change,
   since the model and training loop read the number of classes dynamically
   from the dataset.

Transfer learning (`--arch resnet18`, now the default) already gives you a
big head start for this — it'll adapt to more classes with much less
training time per class than the from-scratch CNN would. If you outgrow
ResNet18 itself, a larger pretrained backbone (ResNet50, EfficientNet, etc.
via `torchvision.models`) is the natural next lever.
