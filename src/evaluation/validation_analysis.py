"""Compare true-class and predicted-class Grad-CAM on validation images."""
from pathlib import Path

import matplotlib

# The CLI saves figures to files, including on machines without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


def gradcam_maps(model, image, class_ids, device):
    # Use the last convolution in block 6's main path, before pooling.
    block = model.block6
    main_path = getattr(block, "main_path", block)
    target_layer = [layer for layer in main_path.modules()
                    if isinstance(layer, nn.Conv2d)][-1]
    activations = []
    hook = target_layer.register_forward_hook(
        lambda layer, inputs, output: activations.append(output)
    )
    try:
        # eval() keeps dropout off and BN fixed; gradients still need enabling.
        with torch.enable_grad():
            inputs = image.unsqueeze(0).to(device).requires_grad_(True)
            scores = model(inputs)
            features = activations[0]
            maps = []
            for position, class_id in enumerate(class_ids):
                gradients = torch.autograd.grad(
                    scores[0, int(class_id)], features,
                    retain_graph=position < len(class_ids) - 1,
                )[0]
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = (weights * features).sum(dim=1, keepdim=True).relu()
                cam = F.interpolate(
                    cam, size=image.shape[-2:], mode="bilinear", align_corners=False,
                )[0, 0].detach().cpu()
                # A zero map has no positive support; do not invent a hotspot.
                maximum = cam.max()
                maps.append(cam / maximum if maximum > 0 else torch.zeros_like(cam))
            return maps
    finally:
        hook.remove()


def select_samples(results):
    highest = results.nlargest(10, "loss")
    remaining = results.drop(index=highest.index)
    correct = remaining[remaining["correct"]].sample(
        n=min(5, int(remaining["correct"].sum())), random_state=42,
    )
    incorrect = remaining[~remaining["correct"]].sample(
        n=min(5, int((~remaining["correct"]).sum())), random_state=42,
    )
    random_samples = pd.concat([correct, incorrect])
    # If either group has fewer than five, fill from the other group.
    available = remaining.drop(index=random_samples.index)
    extra = available.sample(n=min(10 - len(random_samples), len(available)), random_state=42)
    random_samples = pd.concat([random_samples, extra]).sample(frac=1, random_state=42)
    return highest, random_samples


def plot_samples(samples, dataset, model, device, title, filename):
    if samples.empty:
        return
    figure, axes = plt.subplots(
        len(samples), 3, figsize=(12, 3.6 * len(samples)), squeeze=False,
    )
    try:
        for ax in axes.flat:
            ax.axis("off")
        for position, (_, row) in enumerate(samples.iterrows()):
            image = dataset[int(row["image_index"])]["image"]
            original = image.detach().cpu().permute(1, 2, 0).clamp(0, 1)
            maps = gradcam_maps(
                model, image, [row["true_label"], row["predicted_label"]], device,
            )
            for ax in axes[position]:
                ax.imshow(original)
            axes[position, 0].set_title(
                f"{'Correct' if row['correct'] else 'Incorrect'} | Loss: {row['loss']:.3f}\n"
                f"True: {row['true_class']}\nPredicted: {row['predicted_class']}",
                fontsize=9,
            )
            for column, cam, heading, label in [
                (1, maps[0], "True-class Grad-CAM", row["true_class"]),
                (2, maps[1], "Predicted-class Grad-CAM", row["predicted_class"]),
            ]:
                ax = axes[position, column]
                if cam.max() > 0:
                    ax.imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=0.45 * cam.numpy())
                    note = ""
                else:
                    note = "\nNo positive Grad-CAM response"
                ax.set_title(f"{heading}\n{label}{note}", fontsize=9)
        figure.suptitle(title + "\nEach heatmap is normalized independently; compare locations, not intensity.")
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(filename, dpi=120)
    finally:
        plt.close(figure)
    print(f"Image grid saved: {filename}")


def analyze_validation(
    model, validation_loader, class_names, device,
    output_dir="outputs/validation_analysis",
):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    records = []
    with torch.no_grad():
        for batch in validation_loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            labels = batch["label"].to(device, non_blocking=device.type == "cuda")
            scores = model(images)
            losses = criterion(scores, labels)
            predictions = scores.argmax(dim=1)
            for actual, predicted, loss in zip(
                labels.cpu().tolist(), predictions.cpu().tolist(), losses.cpu().tolist(),
            ):
                records.append({
                    # The validation loader uses shuffle=False and no augmentation.
                    "image_index": len(records),
                    "true_label": actual,
                    "predicted_label": predicted,
                    "true_class": class_names[actual],
                    "predicted_class": class_names[predicted],
                    "loss": loss,
                    "correct": actual == predicted,
                })
    if not records:
        raise ValueError("The validation dataset is empty.")
    results = pd.DataFrame(records)
    highest, random_samples = select_samples(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_samples(
        highest, validation_loader.dataset, model, device,
        "10 highest-loss validation images", output_dir / "highest_loss_gradcam.png",
    )
    plot_samples(
        random_samples, validation_loader.dataset, model, device,
        "Random validation images: 5 correct + 5 incorrect (when available)",
        output_dir / "random_predictions_gradcam.png",
    )
    return results
