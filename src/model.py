"""
model.py -- two CNN options: a from-scratch network, and a transfer-learning
network built on a pretrained ResNet18.

WHY TRANSFER LEARNING HELPS SO MUCH HERE:

SimplePokeCNN starts from random weights and has to learn every visual
concept -- edges, textures, shapes, "pointy ears", "long tail" -- from
scratch, using only ~40 images per Pokemon. That's very little data for a
network with no prior knowledge of what images even look like, which is why
it plateaus around 20% accuracy on this 151-class problem.

ResNet18 pretrained on ImageNet (1.4 million photos across 1000 everyday
categories) has already learned a rich, general-purpose set of visual
features in its early and middle layers: edges, colors, textures, simple
shapes, object parts. Reusing those features and only teaching the network
the NEW part -- "which of these 151 specific shapes is which Pokemon" -- is
a dramatically easier learning problem than starting from zero. This is
"transfer learning": transferring knowledge from one task (general image
recognition) to a related task (Pokemon classification).

TWO-PHASE TRAINING (see train.py for the implementation):
  Phase 1 ("freeze the backbone"): we keep all of ResNet18's pretrained
  weights fixed and only train the new final layer we attach on top. This is
  fast, and prevents that new layer's initially-random, large gradient
  updates from wrecking the pretrained features before they've adapted at
  all.
  Phase 2 ("fine-tune"): we unfreeze everything and keep training the whole
  network, but with a much smaller learning rate, so the pretrained features
  get gently nudged to be more Pokemon-specific rather than overwritten.

NOTE: the first time you train with --arch resnet18, torchvision will
download the pretrained ImageNet weights (~45MB) automatically. This needs
internet access on whatever machine you run train.py on.
"""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # halves H and W
    )


class SimplePokeCNN(nn.Module):
    """
    A CNN trained fully from scratch.

    CONCEPTS, layer by layer:

    * nn.Conv2d(in_channels, out_channels, kernel_size): slides a small
      learnable grid of numbers (the "kernel"/"filter") over the image. Each
      position computes a weighted sum of the pixels it covers -- this is
      "convolution". Each of the `out_channels` filters learns to detect a
      different pattern (an edge, a color blob, a texture). Early layers
      learn simple patterns; later layers combine those into more complex
      shapes (an eye, a wing, a tail).

    * nn.BatchNorm2d: rescales each channel's activations to have a stable
      mean/variance during training. Mostly a training-stability trick --
      it lets us use a higher learning rate and trains faster.

    * nn.ReLU: f(x) = max(0, x). Without a nonlinearity like this between
      layers, stacking many linear (conv/matrix) operations would collapse
      into a single linear operation no matter how many layers you stack --
      the network couldn't learn anything but straight lines.

    * nn.MaxPool2d(2): downsamples by taking the max value in each 2x2
      block, halving height and width. Reduces computation for later
      layers and gives a small amount of translation invariance.

    * nn.AdaptiveAvgPool2d((4, 4)): squashes whatever spatial size the
      feature map currently is down to exactly 4x4, by averaging. This
      decouples the network from a single hardcoded input image size --
      change IMAGE_SIZE in dataset.py and this model still works without
      editing the Linear layer's input dimension by hand.

    * nn.Flatten + nn.Linear (fully connected): once the spatial map is
      small, we flatten it into a vector and pass it through ordinary
      matrix-multiply layers to produce final class scores ("logits") --
      one number per Pokemon, higher = more likely.

    We do NOT apply softmax inside the model. nn.CrossEntropyLoss (used in
    train.py) expects raw logits and applies log-softmax internally, which
    is more numerically stable than doing it by hand.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),  # randomly zero 40% of activations during training
                                 # only -- forces the network not to rely on any
                                 # single feature, which reduces overfitting
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class ResNetPokeClassifier(nn.Module):
    """
    ResNet18 pretrained on ImageNet, with its final classification layer
    swapped out for a fresh, randomly-initialized one that outputs
    `num_classes` Pokemon instead of 1000 ImageNet categories. Every layer
    before that is reused as-is (initially -- see set_backbone_trainable).
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def set_backbone_trainable(self, trainable: bool):
        """
        Freeze (trainable=False) or unfreeze (trainable=True) every layer
        EXCEPT the final classification layer (`fc`), which always stays
        trainable since it starts randomly initialized and always needs
        training regardless of what phase we're in.
        """
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = trainable


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "scratch":
        return SimplePokeCNN(num_classes)
    if arch == "resnet18":
        return ResNetPokeClassifier(num_classes)
    raise ValueError(f"Unknown arch: {arch!r} (expected 'scratch' or 'resnet18')")
