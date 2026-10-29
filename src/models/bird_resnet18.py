"""ResNet-18 written explicitly using our own residual blocks."""
from torch import nn

from src.models.bird_cnn_residual import ResidualBlock


class BirdResNet18(nn.Module):
    def __init__(self, num_classes=200, dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stage 1: two blocks, 64 channels, no further downsampling.
        self.block1 = ResidualBlock(64, 64)
        self.block2 = ResidualBlock(64, 64)

        # Stage 2: the first block halves spatial size on both paths.
        self.block3 = ResidualBlock(64, 128, stride=2)
        self.block4 = ResidualBlock(128, 128)

        # Stage 3.
        self.block5 = ResidualBlock(128, 256, stride=2)
        self.block6 = ResidualBlock(256, 256)

        # Stage 4.
        self.block7 = ResidualBlock(256, 512, stride=2)
        self.block8 = ResidualBlock(512, 512)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, images):
        x = self.stem(images)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)
