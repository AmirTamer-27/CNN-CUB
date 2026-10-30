"""Image preparation shared by training and validation."""
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2

def resize_and_pad(image, size=320):
    # LANCZOS resamples pixels while resizing; padding then fills unused space.
    return ImageOps.pad(
        image.convert("RGB"),
        size=(size, size),
        method=Image.Resampling.LANCZOS,
        color=(128, 128, 128),
        centering=(0.5, 0.5),
    )


def prepare_images(
    batch,
    image_size=224,
    flip_probability=0.0,
    contrast_probability=0.0,
    contrast_strength=0.2,
    brightness_probability=0.0,
    brightness_strength=0.2,
    cutout_probability=0.0,
    cutout_min_fraction=0.25,
    cutout_max_fraction=0.30,
):
    """Resize/pad, augment, convert to tensors, and optionally apply one Cutout."""
    if not 0 <= cutout_probability <= 1:
        raise ValueError("cutout_probability must be between 0 and 1.")
    if not 0 < cutout_min_fraction <= cutout_max_fraction <= 1:
        raise ValueError("Cutout side fractions must satisfy 0 < min <= max <= 1.")
    horizontal_flip = transforms.RandomHorizontalFlip(
        p=flip_probability
    )

    random_contrast = transforms.RandomApply(
        [transforms.ColorJitter(contrast=contrast_strength)],
        p=contrast_probability,
    )
    random_brightness = transforms.RandomApply(
        [transforms.ColorJitter(brightness=brightness_strength)],
        p=brightness_probability,
    )

    to_tensor = transforms.ToTensor()
    prepared_images = []

    for image in batch["image"]:
        image = resize_and_pad(image, size=image_size)

        # Each operation makes its own random decision for this image.
        image = horizontal_flip(image)
        image = random_contrast(image)
        if brightness_probability > 0:
            image = random_brightness(image)

        image = to_tensor(image)

        # One random square, independently sampled whenever this image is read.
        # The zero-probability guard keeps old configs' random sequence unchanged.
        if cutout_probability > 0 and torch.rand(()).item() < cutout_probability:
            height, width = image.shape[-2:]
            minimum = max(1, round(min(height, width) * cutout_min_fraction))
            maximum = max(minimum, round(min(height, width) * cutout_max_fraction))
            side = int(torch.randint(minimum, maximum + 1, ()).item())
            center_y = int(torch.randint(height, ()).item())
            center_x = int(torch.randint(width, ()).item())
            top = max(0, center_y - side // 2)
            bottom = min(height, center_y + (side + 1) // 2)
            left = max(0, center_x - side // 2)
            right = min(width, center_x + (side + 1) // 2)
            # Match the neutral gray padding in [0, 1] tensor units.
            image[:, top:bottom, left:right] = 128 / 255

        prepared_images.append(image)

    return {
        "image": prepared_images,
        "label": batch["label"],
    }



def get_train_transforms():

    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=(320, 320),
            scale=(0.8, 1.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(
            torch.float32,
            scale=True
        ),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms():

    return transforms.Compose([
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(
            torch.float32,
            scale=True
        ),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])