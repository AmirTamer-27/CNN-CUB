"""Load the training source and create a repeatable validation split."""
import torch
from torch.utils.data import DataLoader
from src.data.transforms import prepare_images


def make_loaders(config):
    # Import here so loading CLI help never downloads or loads the dataset.
    from datasets import load_dataset

    full_train = load_dataset(config["train_dataset"], split="train")
    split = full_train.train_test_split(
        test_size=config["validation_fraction"],
        stratify_by_column="label",
        seed=config["seed"],
    )
    def prepare_train(batch):
        return prepare_images(
            batch,
            image_size=config["image_size"],
            flip_probability=config.get("horizontal_flip_probability", 0.0),
            contrast_probability=config.get("contrast_change_probability", 0.0),
            contrast_strength=config.get("contrast_strength", 0.2),
            brightness_probability=config.get("brightness_change_probability", 0.0),
            brightness_strength=config.get("brightness_strength", 0.2),
            cutout_probability=config.get("cutout_probability", 0.0),
            cutout_min_fraction=config.get("cutout_min_fraction", 0.25),
            cutout_max_fraction=config.get("cutout_max_fraction", 0.30),
        )

    def prepare_validation(batch):
        return prepare_images(batch, image_size=config["image_size"])

    train = split["train"].with_transform(prepare_train)
    validation = split["test"].with_transform(prepare_validation)
    generator = torch.Generator().manual_seed(config["seed"])
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation, 
        batch_size=config["batch_size"],
        shuffle=False, 
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    class_names = full_train.features["label"].names
    return train_loader, validation_loader, class_names
