import torch.nn as nn

from torchvision.models import (
    ResNeXt101_32X8D_Weights,
    resnext101_32x8d,
)


def create_resnext_classifier(
    num_classes=200,
    dropout=0.7,
):

    model = resnext101_32x8d(
        weights=ResNeXt101_32X8D_Weights.DEFAULT
    )

    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(
            model.fc.in_features,
            num_classes
        )
    )

    return model