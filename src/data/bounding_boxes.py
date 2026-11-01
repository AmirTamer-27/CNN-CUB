"""Match original CUB annotations to Hugging Face filenames and crop birds."""
import math
from pathlib import Path

import pandas as pd


def load_bounding_boxes(annotation_dir):
    annotation_dir = Path(annotation_dir)
    images = pd.read_csv(
        annotation_dir / "images.txt", sep=r"\s+",
        names=["image_id", "image_path"],
    )
    boxes = pd.read_csv(
        annotation_dir / "bounding_boxes.txt", sep=r"\s+",
        names=["image_id", "x", "y", "width", "height"],
    )
    if set(images["image_id"]) != set(boxes["image_id"]):
        raise ValueError("Image IDs in the two CUB annotation files do not match.")
    annotations = images.merge(boxes, on="image_id", validate="one_to_one")
    annotations["file_name"] = annotations["image_path"].map(lambda path: Path(path).name)
    if annotations["file_name"].duplicated().any():
        raise ValueError("Duplicate filenames in bounding-box annotations.")
    return annotations.set_index("file_name")[
        ["x", "y", "width", "height"]
    ].to_dict(orient="index")


def crop_to_bird(image, bbox):
    """Crop x/y/width/height coordinates in original pixels, without a margin."""
    x, y, width, height = (bbox[key] for key in ("x", "y", "width", "height"))
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("Bounding-box coordinates must be finite.")
    if width <= 0 or height <= 0:
        raise ValueError("Bounding-box width and height must be positive.")
    # Round outward to retain pixels touched by fractional box boundaries.
    left = max(0, math.floor(x))
    top = max(0, math.floor(y))
    right = min(image.width, math.ceil(x + width))
    bottom = min(image.height, math.ceil(y + height))
    if right <= left or bottom <= top:
        raise ValueError("Bounding box does not overlap the image.")
    return image.crop((left, top, right, bottom))


def crop_sample(sample, boxes):
    return {"image": crop_to_bird(sample["image"], boxes[sample["file_name"]])}


def crop_dataset(dataset, annotation_dir):
    """Preserve rows, order and labels while replacing images with bird crops."""
    boxes = load_bounding_boxes(annotation_dir)
    missing = set(dataset["file_name"]) - set(boxes)
    if missing:
        example = sorted(missing)[0]
        raise ValueError(f"Missing boxes for {len(missing)} images, including {example}.")
    return dataset.map(
        crop_sample, fn_kwargs={"boxes": boxes}, desc="Cropping birds",
    )
