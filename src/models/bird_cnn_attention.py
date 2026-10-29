"""Delayed-pooling CNN with learned spatial attention instead of uniform pooling."""
from torch import nn

from src.models.bird_cnn_delayed_pooling import BirdCNNDelayedPooling


class BirdCNNAttention(BirdCNNDelayedPooling):
    def __init__(self, num_classes=200, dropout=0.0):
        super().__init__(num_classes=num_classes, dropout=dropout)
        # One score per spatial position, calculated from all 512 channels.
        self.attention = nn.Conv2d(512, 1, kernel_size=1)
        # Start with uniform spatial weights, equivalent to average pooling.
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def attention_weights(self, features):
        scores = self.attention(features)
        # Positive weights sum to one over H x W, independently for each image.
        return scores.flatten(start_dim=2).softmax(dim=2).reshape_as(scores)

    def forward(self, images):
        x = self.block1(images)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        weights = self.attention_weights(x)
        features = (x * weights).sum(dim=(2, 3))
        features = self.dropout(features)
        return self.fc(features)
