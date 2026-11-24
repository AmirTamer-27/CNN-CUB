"""RGB U-Net generator and a whole-image conditional discriminator."""

import torch
from torch import nn

from src.models.colorization_unet import BirdColorizationUNet


class BirdGANGenerator(BirdColorizationUNet):
    """Reuse the RGB U-Net, including its encoder and skip connections."""


class BirdDiscriminator(nn.Module):
    """One real/fake logit for each grayscale/RGB image pair."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = 6
        for out_channels in (64, 128, 256, 512):
            layers.append(nn.Conv2d(channels, out_channels, 4, stride=2, padding=1))
            if channels != 6:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1))

    def forward(self, grayscale, color):
        pair = torch.cat((grayscale, color), dim=1)
        return self.classifier(self.features(pair))
