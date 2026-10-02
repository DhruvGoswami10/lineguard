"""The two techniques being compared.

ConvAutoencoder   unsupervised: learns to rebuild defect-free bottles only.
                  A defect is a region it cannot rebuild, so the rebuild
                  error is high there. Needs zero defect images to train.
build_resnet18    supervised: ImageNet-pretrained ResNet-18 with a new
                  2-class head (good / defect). Needs labelled defect images.
"""
import torch.nn as nn
from torchvision import models

# Encoder channels per stride-2 block. The last value is the bottleneck depth:
# 8 x 8 x 64 = 4,096 numbers, 12x fewer than the 128 x 128 x 3 = 49,152 input
# pixels. Fixed before training and never tuned on test data.
AE_CHANNELS = (32, 64, 128, 64)


def down_block(c_in: int, c_out: int) -> nn.Sequential:
    """Conv with stride 2: halves height and width (128 -> 64 -> 32 -> 16 -> 8)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2, inplace=True),
    )


def up_block(c_in: int, c_out: int) -> nn.Sequential:
    """Transposed conv with stride 2: doubles height and width (mirror of down_block)."""
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """128x128x3 image -> 8x8x64 code -> 128x128x3 reconstruction.

    The narrow code forces the network to keep only what normal bottles have
    in common, so it cannot copy an unusual region (a crack, dirt) faithfully.
    """

    def __init__(self, channels: tuple = AE_CHANNELS):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.encoder = nn.Sequential(
            down_block(3, c1), down_block(c1, c2), down_block(c2, c3), down_block(c3, c4),
        )
        self.decoder = nn.Sequential(
            up_block(c4, c3), up_block(c3, c2), up_block(c2, c1),
            nn.ConvTranspose2d(c1, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # output in [0, 1], the same range as the input pixels
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_resnet18(pretrained: bool) -> nn.Module:
    """ResNet-18 with its 1000-class ImageNet head replaced by good/defect.

    pretrained=True downloads ImageNet weights (training only): the network
    starts out already knowing edges, textures and shapes, which is why 31
    defect images can be enough. The demo uses pretrained=False and then loads
    our own trained weights, so it needs no internet.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
