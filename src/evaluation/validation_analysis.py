"""The simple validation loop and Pandas summary shown in the explanation."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn


def plot_samples(samples, dataset, title, filename):
    columns = min(3, len(samples))
    rows = (len(samples) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 4 * rows), squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, (_, row) in zip(axes.flat, samples.iterrows()):
        image = dataset[int(row["image_index"])]["image"]
        ax.imshow(image.cpu().permute(1, 2, 0).clamp(0, 1))
        ax.set_title(
            f"True: {row['true_class']}\n"
            f"Predicted: {row['predicted_class']}\n"
            f"Loss: {row['loss']:.3f}",
            fontsize=9,
        )

    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(filename, dpi=150)
    plt.close(figure)
    print(f"Image grid saved: {filename}")


def analyze_validation(
    model, validation_loader, class_names, device,
    output_dir="outputs/validation_analysis",
):
    model.eval()

    # "none" means return one loss per image.
    criterion = nn.CrossEntropyLoss(reduction="none")
    records = []

    with torch.no_grad():
        for batch in validation_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            scores = model(images)
            losses = criterion(scores, labels)
            predictions = scores.argmax(dim=1)

            # Move results to the CPU for ordinary Python/Pandas processing.
            labels = labels.cpu().tolist()
            predictions = predictions.cpu().tolist()
            losses = losses.cpu().tolist()

            for actual, predicted, loss in zip(labels, predictions, losses):
                records.append({
                    # The validation loader uses shuffle=False.
                    "image_index": len(records),
                    "true_class": class_names[actual],
                    "predicted_class": class_names[predicted],
                    "loss": loss,
                    "correct": actual == predicted,
                })

    if not records:
        raise ValueError("The validation dataset is empty.")
    results = pd.DataFrame(records)

    per_class = results.groupby("true_class").agg(
        samples=("loss", "size"),
        mean_loss=("loss", "mean"),
        recall=("correct", "mean"),
    )

    per_class["recall"] *= 100

    print("Classes with the highest average validation loss:")
    print(per_class.sort_values("mean_loss", ascending=False).head(20))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    highest_loss_images = results.sort_values("loss", ascending=False).head(12)
    plot_samples(
        highest_loss_images, validation_loader.dataset,
        "Highest-loss validation images",
        output_dir / "highest_loss_images.png",
    )

    worst_class = per_class["mean_loss"].idxmax()
    class_samples = results[results["true_class"] == worst_class]
    class_samples = class_samples.sample(n=min(6, len(class_samples)), random_state=42)
    print(f"Class with highest average loss: {worst_class}")
    plot_samples(
        class_samples, validation_loader.dataset,
        f"Samples from highest-average-loss class: {worst_class}",
        output_dir / "highest_loss_class.png",
    )
    return results, per_class
