"""Six residual blocks arranged into three stages with two blocks each."""
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Both paths must produce the same shape before addition.
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        features = self.main_path(x)
        shortcut = self.shortcut(x)
        return self.relu(features + shortcut)


class BirdResidualCNN(nn.Module):
    def __init__(self, num_classes=200, dropout=0.0):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Stage 1: refine features at the original resolution.
        self.block1 = ResidualBlock(32, 32)
        self.block2 = ResidualBlock(32, 32)

        # Stage 2: downsample once, then refine at the same resolution.
        self.block3 = ResidualBlock(32, 64, stride=2)
        self.block4 = ResidualBlock(64, 64)

        # Stage 3: downsample once more, then refine.
        self.block5 = ResidualBlock(64, 128, stride=2)
        self.block6 = ResidualBlock(128, 128)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, images):
        x = self.stem(images)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)
