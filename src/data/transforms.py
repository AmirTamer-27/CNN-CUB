"""Image preparation shared by training and validation."""
from PIL import Image, ImageOps
from torchvision import transforms


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
):
    """Resize/pad, independently augment each image, then convert to tensors."""
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
        prepared_images.append(image)

    return {
        "image": prepared_images,
        "label": batch["label"],
    }
