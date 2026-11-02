"""Data loaders for transfer-learning experiments with bboxes and zero augmentations."""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.data.transforms import (
    get_val_transforms,
    resize_and_pad,
)
from src.data.bounding_boxes import crop_dataset


def make_transfer_loaders(config):
    """Create train and validation loaders with bounding boxes and no data augmentation."""

    full_train = load_dataset(
        config["train_dataset"],
        split="train",
    )

    # Apply official CUB bounding box crops if configured
    if config.get("use_bounding_boxes", False):
        annotation_dir = config.get("bbox_annotation_dir", "data/CUB_200_2011")
        print(f"Applying CUB ground-truth bounding box crops from: {annotation_dir}")
        full_train = crop_dataset(full_train, annotation_dir)

    split = full_train.train_test_split(
        test_size=config["validation_fraction"],
        stratify_by_column="label",
        seed=config["seed"],
    )

    # Use val transform (resize/pad + tensor + normalize) for both train and validation
    base_transform = get_val_transforms()

    def prepare_clean_batch(batch):
        images = []
        for image in batch["image"]:
            # Resize and pad to target size (e.g., 384) using LANCZOS
            image = resize_and_pad(
                image,
                size=config["image_size"],
            )
            image = base_transform(image)
            images.append(image)

        return {
            "image": images,
            "label": batch["label"],
        }

    train = split["train"].with_transform(prepare_clean_batch)
    validation = split["test"].with_transform(prepare_clean_batch)

    generator = torch.Generator().manual_seed(config["seed"])

    train_loader = DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        generator=generator,
    )

    validation_loader = DataLoader(
        validation,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 2),
    )

    class_names = full_train.features["label"].names

    return train_loader, validation_loader, class_names