"""U-Net predicting a color-bin score for every pixel."""

from torch import nn

from src.models.colorization_unet import BirdColorizationUNet


class BirdLabColorizationUNet(BirdColorizationUNet):
    def __init__(self, encoder=None):
        super().__init__(encoder=encoder, trainable_encoder_blocks=[])
        self.output_layer = nn.Conv2d(16, 313, kernel_size=1)

    def forward(self, lightness):
        x1, x2, x3, x4, x5, x6 = self.encode(lightness)
        d5 = self.decoder5(x6, x5)
        d4 = self.decoder4(d5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)
        return self.output_layer(d1)
