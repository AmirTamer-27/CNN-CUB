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


def prepare_images(batch, image_size=224):
    """Resize and pad a batch of images, then convert them to tensors."""
    to_tensor = transforms.ToTensor()
    return {
        "image": [
            to_tensor(resize_and_pad(image, size=image_size))
            for image in batch["image"]
        ],
        "label": batch["label"],
    }
