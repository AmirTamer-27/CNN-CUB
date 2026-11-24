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

    def __init__(
        self,
        encoder=None,
        trainable_encoder_blocks=None,
        adapt_trainable_batch_norm=False,
    ):
        super().__init__()
        self.encoder = encoder or BirdCNNDelayedPooling()

        self.decoder5 = DecoderBlock(512, 256, 256)
        self.decoder4 = DecoderBlock(256, 128, 128)
        self.decoder3 = DecoderBlock(128, 64, 64)
        self.decoder2 = DecoderBlock(64, 32, 32)
        self.decoder1 = DecoderBlock(32, 16, 16)

        # Spatial refinement is complete; map 16 features per pixel to RGB.
        self.output_layer = nn.Conv2d(16, 3, kernel_size=1)
        self.trainable_encoder_blocks = []
        self.adapt_trainable_batch_norm = adapt_trainable_batch_norm
        self.set_trainable_encoder_blocks(trainable_encoder_blocks or [])

    def set_trainable_encoder_blocks(self, block_names):
        """Freeze the encoder, then unfreeze only the requested blocks."""
        valid_blocks = [f"block{index}" for index in range(1, 7)]
        invalid_blocks = [name for name in block_names if name not in valid_blocks]
        if invalid_blocks:
            raise ValueError(
                f"Invalid encoder blocks: {invalid_blocks}. "
                f"Choose from {valid_blocks}."
            )

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        for block_name in block_names:
            block = getattr(self.encoder, block_name)
            for parameter in block.parameters():
                parameter.requires_grad = True

        self.trainable_encoder_blocks = list(block_names)
        self.encoder.eval()

    def train(self, mode=True):
        """Train selected weights and optionally adapt their BatchNorm statistics."""
        super().train(mode)
        self.encoder.eval()

        if mode and self.adapt_trainable_batch_norm:
            for block_name in self.trainable_encoder_blocks:
                block = getattr(self.encoder, block_name)
                for module in block.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.train()

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
