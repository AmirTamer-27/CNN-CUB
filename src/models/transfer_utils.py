import torch.nn as nn


def freeze_backbone(model):
    """Freeze all model parameters."""
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_layer4(model):
    """Unfreeze ResNeXt's final convolutional stage."""
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True


def unfreeze_classifier(model):
    """Unfreeze the final classifier."""
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def get_trainable_parameters(model):
    """Return parameters that are currently trainable."""
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]