"""Data loaders for transfer-learning experiments with bounding boxes and light augmentations."""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    resize_and_pad,
)
from src.data.bounding_boxes import crop_dataset


def make_transfer_loaders(config):
    """Create train and validation loaders with bboxes, light training augs, and clean validation."""

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

    # Assign distinct transforms for training (with light augs) vs validation (clean)
    train_transform = get_train_transforms(config["image_size"])
    val_transform = get_val_transforms()

    def prepare_train(batch):
        images = []
        for image in batch["image"]:
            image = resize_and_pad(image, size=config["image_size"])
            image = train_transform(image)
            images.append(image)
        return {
            "image": images,
            "label": batch["label"],
        }

    def prepare_validation(batch):
        images = []
        for image in batch["image"]:
            image = resize_and_pad(image, size=config["image_size"])
            image = val_transform(image)
            images.append(image)
        return {
            "image": images,
            "label": batch["label"],
        }

    train = split["train"].with_transform(prepare_train)
    validation = split["test"].with_transform(prepare_validation)

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