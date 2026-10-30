"""Data loaders for transfer-learning experiments."""

import torch
from torch.utils.data import DataLoader

from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    resize_and_pad,
)


def make_transfer_loaders(config):
    """Create train and validation loaders for transfer learning."""

    from datasets import load_dataset

    full_train = load_dataset(
        config["train_dataset"],
        split="train",
    )

    split = full_train.train_test_split(
        test_size=config["validation_fraction"],
        stratify_by_column="label",
        seed=config["seed"],
    )

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    def prepare_train(batch):
        images = []

        for image in batch["image"]:
            image = resize_and_pad(
                image,
                size=config["image_size"],
            )
            image = train_transform(image)
            images.append(image)

        return {
            "image": images,
            "label": batch["label"],
        }

    def prepare_validation(batch):
        images = []

        for image in batch["image"]:
            image = resize_and_pad(
                image,
                size=config["image_size"],
            )
            image = val_transform(image)
            images.append(image)

        return {
            "image": images,
            "label": batch["label"],
        }

    train = split["train"].with_transform(prepare_train)
    validation = split["test"].with_transform(prepare_validation)

    generator = torch.Generator().manual_seed(
        config["seed"]
    )

    train_loader = DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        generator=generator,
    )

    validation_loader = DataLoader(
        validation,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    class_names = full_train.features["label"].names

    return train_loader, validation_loader, class_names