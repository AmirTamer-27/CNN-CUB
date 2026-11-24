"""U-Net colorizer built around the Phase 1 bird encoder."""

import torch
from torch import nn
from torch.nn import functional as F

from src.models.bird_cnn_delayed_pooling import BirdCNNDelayedPooling


class DecoderBlock(nn.Module):
    """Upsample, concatenate one encoder skip, then refine with 3x3 convolutions."""

    def __init__(self, decoder_channels, skip_channels, output_channels):
        super().__init__()
        combined_channels = decoder_channels + skip_channels
        self.refine = nn.Sequential(
            nn.Conv2d(
                combined_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, decoder_features, skip_features):
        decoder_features = F.interpolate(
            decoder_features,
            size=skip_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        combined = torch.cat([decoder_features, skip_features], dim=1)
        return self.refine(combined)


class BirdColorizationUNet(nn.Module):
    """Predict a 3-channel RGB image from a grayscale bird image."""

    def __init__(self, encoder=None, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder or BirdCNNDelayedPooling()

        self.decoder5 = DecoderBlock(512, 256, 256)
        self.decoder4 = DecoderBlock(256, 128, 128)
        self.decoder3 = DecoderBlock(128, 64, 64)
        self.decoder2 = DecoderBlock(64, 32, 32)
        self.decoder1 = DecoderBlock(32, 16, 16)

        # Spatial refinement is complete; map 16 features per pixel to RGB.
        self.output_layer = nn.Conv2d(16, 3, kernel_size=1)
        self.encoder_trainable = False
        self.set_encoder_trainable(not freeze_encoder)

    def set_encoder_trainable(self, trainable):
        """Freeze or unfreeze the Phase 1 encoder."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

        self.encoder_trainable = trainable

        if trainable:
            self.encoder.train(self.training)
        else:
            self.encoder.eval()

    def train(self, mode=True):
        """Keep a frozen encoder, including its BatchNorm layers, in eval mode."""
        super().train(mode)
        if not self.encoder_trainable:
            self.encoder.eval()
        return self

    def encode(self, grayscale):
        """Return one feature map for every encoder resolution.

        The data pipeline must supply three-channel grayscale tensors.
        """
        x1 = self.encoder.block1(grayscale)  # [B, 16,  H,     W]
        x2 = self.encoder.block2(x1)   # [B, 32,  H/2,   W/2]
        x3 = self.encoder.block3(x2)   # [B, 64,  H/4,   W/4]
        x4 = self.encoder.block4(x3)   # [B, 128, H/8,   W/8]
        x5 = self.encoder.block5(x4)   # [B, 256, H/16,  W/16]
        x6 = self.encoder.block6(x5)   # [B, 512, H/32,  W/32]
        return x1, x2, x3, x4, x5, x6

    def forward(self, grayscale):
        x1, x2, x3, x4, x5, x6 = self.encode(grayscale)

        d5 = self.decoder5(x6, x5)
        d4 = self.decoder4(d5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)

        # RGB targets are expected to be normalized to [0, 1].
        return torch.sigmoid(self.output_layer(d1))
