"""Lab inputs and RGB reconstruction for continuous a/b regression."""

import torch
from kornia.color import lab_to_rgb, rgb_to_lab


def prepare_lab(rgb):
    lab = rgb_to_lab(rgb)
    lightness = lab[:, :1]
    encoder_input = (lightness / 100.0).repeat(1, 3, 1, 1)
    target_ab = lab[:, 1:] / 128.0
    return encoder_input, lightness, target_ab


def lab_ab_to_rgb(lightness, normalized_ab):
    lab = torch.cat((lightness, normalized_ab * 128.0), dim=1)
    return lab_to_rgb(lab).clamp(0.0, 1.0)
