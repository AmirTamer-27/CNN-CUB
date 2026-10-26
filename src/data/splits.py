"""Create CUB training/test manifests using the official image assignments."""

from pathlib import Path

import pandas as pd


def build_official_splits(data_dir, output_dir):
    """Save train.csv/test.csv and return their metadata without opening images.

    image_path is relative to data_dir / 'images'. Class IDs retain CUB's
    original 1-based numbering. Images and source annotations are never moved.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    tables = {
        "images.txt": ["image_id", "image_path"],
        "bounding_boxes.txt": ["image_id", "x", "y", "bbox_width", "bbox_height"],
        "image_class_labels.txt": ["image_id", "class_id"],
        "train_test_split.txt": ["image_id", "is_training_image"],
    }
    frames = {
        filename: pd.read_csv(data_dir / filename, sep=r"\s+", names=columns)
        for filename, columns in tables.items()
    }
    data = frames["images.txt"]
    for filename in list(tables)[1:]:
        frame = frames[filename]
        if set(frame.image_id) != set(data.image_id):
            raise ValueError(f"Image IDs in {filename} do not match images.txt")
        data = data.merge(frame, on="image_id", validate="one_to_one")
    classes = pd.read_csv(
        data_dir / "classes.txt", sep=r"\s+", names=["class_id", "class_name"]
    )
    data = data.merge(classes, on="class_id", how="left", validate="many_to_one")
    if data.class_name.isna().any() or data.image_path.duplicated().any():
        raise ValueError("Missing class names or repeated image paths")
    if not data.is_training_image.isin([0, 1]).all():
        raise ValueError("Official split flags must be 0 (test) or 1 (train)")
    missing = [p for p in data.image_path if not (data_dir / "images" / p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images, including {missing[0]}")

    train = data.loc[data.is_training_image.eq(1)].copy().sort_values("image_id")
    test = data.loc[data.is_training_image.eq(0)].copy().sort_values("image_id")
    if train.empty or test.empty:
        raise ValueError("Both official splits must contain images")
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    return train.reset_index(drop=True), test.reset_index(drop=True)
