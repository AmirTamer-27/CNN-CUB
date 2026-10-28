"""Classify using pooled features from blocks 4, 5, and 6."""
import torch
from torch import nn

from src.models.bird_cnn_delayed_pooling import BirdCNNDelayedPooling


class BirdCNNMultiDepth(BirdCNNDelayedPooling):
    def __init__(self, num_classes=200, dropout=0.0):
       
        super().__init__(num_classes=num_classes, dropout=dropout)
        self.fc = nn.Linear(128 + 256 + 512, num_classes)

    def forward(self, images):
        x = self.block1(images)
        x = self.block2(x)
        x = self.block3(x)

        x4 = self.block4(x)
        x5 = self.block5(x4)
        x6 = self.block6(x5)

        # Each feature map becomes one value, regardless of its spatial size.
        features4 = self.flatten(self.global_pool(x4))
        features5 = self.flatten(self.global_pool(x5))
        features6 = self.flatten(self.global_pool(x6))

        # Join features for the same image, keeping batch dimension unchanged.
        features = torch.cat([features4, features5, features6], dim=1)
        features = self.dropout(features)
        return self.fc(features)
