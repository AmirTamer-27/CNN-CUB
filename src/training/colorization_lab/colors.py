"""Lab color bin encoding and RGB reconstruction."""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from kornia.color import lab_to_rgb, rgb_to_lab


BIN_URL = (
    "https://raw.githubusercontent.com/richzhang/colorization/"
    "caffe/colorization/resources/pts_in_hull.npy"
)


def load_bins(path, device):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(BIN_URL, path)
    centers = np.load(path, allow_pickle=False)
    if centers.shape != (313, 2):
        raise ValueError(f"Expected 313 Lab ab bins; got {centers.shape}")
    return torch.tensor(centers, dtype=torch.float32, device=device)


def prepare_batch(rgb):
    lab = rgb_to_lab(rgb)
    lightness = lab[:, :1]
    return (lightness / 100).repeat(1, 3, 1, 1), lightness, lab[:, 1:]


def target_bins(ab, centers, chunk_size=16384):
    """Assign each pixel to its nearest ab center without a giant B*H*W*313 tensor."""
    batch, _, height, width = ab.shape
    pixels = ab.permute(0, 2, 3, 1).reshape(-1, 2)
    labels = []
    for chunk in pixels.split(chunk_size):
        distances = (chunk[:, None, :] - centers[None, :, :]).square().sum(-1)
        labels.append(distances.argmin(dim=1))
    return torch.cat(labels).reshape(batch, height, width)


def reconstruct_rgb(logits, lightness, centers):
    indices = logits.argmax(dim=1)
    ab = centers[indices].permute(0, 3, 1, 2)
    return lab_to_rgb(torch.cat((lightness, ab), dim=1)).clamp(0, 1)
