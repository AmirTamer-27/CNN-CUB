"""Inspect saved model predictions; write an HTML report without JSON files."""
import base64
from html import escape
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Works in terminal commands without a desktop display.
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from src.evaluation.metrics import classification_metrics


def analyze_validation(model, loader, class_names, device, output_dir, top=12):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    records = []
    worst_images = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Analyzing validation"):
            images = batch["image"]
            labels = batch["label"].to(device)
            scores = model(images.to(device))
            losses = criterion(scores, labels).cpu()
            probabilities = scores.softmax(dim=1)
            confidence, predictions = probabilities.max(dim=1)
            true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)

            labels = labels.cpu().tolist()
            predictions = predictions.cpu().tolist()
            confidence = confidence.cpu().tolist()
            true_probability = true_probability.cpu().tolist()
            first_index = len(records)

            for i, (actual, predicted, loss) in enumerate(
                zip(labels, predictions, losses.tolist())
            ):
                records.append({
                    "validation_index": first_index + i,
                    "true_id": actual,
                    "predicted_id": predicted,
                    "true_class": class_names[actual],
                    "predicted_class": class_names[predicted],
                    "loss": loss,
                    "correct": actual == predicted,
                    "predicted_probability": confidence[i],
                    "true_probability": true_probability[i],
                })

            # Keep only the highest-loss candidates, rather than all image tensors.
            indices = losses.topk(min(top, len(losses))).indices.tolist()
            for i in indices:
                worst_images.append({
                    **records[first_index + i],
                    "image": images[i].detach().cpu().clone(),
                })
            worst_images = sorted(
                worst_images, key=lambda item: item["loss"], reverse=True
            )[:top]

    if not records:
        raise ValueError("The validation loader contains no images.")
    results = pd.DataFrame(records)
    metrics = classification_metrics(
        results["loss"].sum(), int(results["correct"].sum()),
        results["true_id"].tolist(), results["predicted_id"].tolist(),
        len(class_names),
    )
    per_class = results.groupby("true_class").agg(
        samples=("loss", "size"),
        mean_loss=("loss", "mean"),
        total_loss=("loss", "sum"),
        recall_percent=("correct", "mean"),
    )
    per_class["recall_percent"] *= 100
    per_class["loss_share_percent"] = (
        100 * per_class["total_loss"] / results["loss"].sum()
        if results["loss"].sum() > 0 else 0.0
    )
    highest_loss_classes = per_class.sort_values("mean_loss", ascending=False)
    lowest_recall_classes = per_class.sort_values(
        ["recall_percent", "mean_loss"], ascending=[True, False]
    )
    mistakes = results.loc[~results["correct"]]
    confusion_pairs = (
        mistakes.groupby(["true_class", "predicted_class"])
        .size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    highest_loss_samples = results.sort_values("loss", ascending=False)

    print(
        f"Validation images: {len(results)} | Loss: {metrics['loss']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.2f}% | "
        f"Macro recall: {metrics['macro_recall']:.2f}%"
    )
    print(f"Greatest individual loss: {results['loss'].max():.4f}")
    tables = [
        ("Highest mean loss per class", highest_loss_classes),
        ("Lowest recall per class (%)", lowest_recall_classes),
        ("Confusion pairs: actual species -> predicted species", confusion_pairs),
        ("Highest-loss validation images", highest_loss_samples),
    ]
    for title, table in tables:
        print(f"\n{title}")
        print(table.head(top).to_string(float_format=lambda value: f"{value:.4f}"))

    columns = min(3, len(worst_images))
    rows = (len(worst_images) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 4 * rows), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, item in zip(axes.flat, worst_images):
        ax.imshow(item["image"].permute(1, 2, 0).clamp(0, 1))
        ax.set_title(
            f"Validation index: {item['validation_index']}\n"
            f"True: {item['true_class']}\nPredicted: {item['predicted_class']}\n"
            f"Loss: {item['loss']:.2f} | P(true): {item['true_probability']:.1%}\n"
            f"P(predicted): {item['predicted_probability']:.1%}",
            fontsize=9,
        )
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=120)
    plt.close(figure)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")

    report = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<title>Validation analysis</title>',
        '<style>body{font-family:Arial;margin:24px}table{border-collapse:collapse}',
        'th,td{padding:7px;border:1px solid #ccc}img{max-width:100%}</style>',
        '</head><body><h1>Validation analysis</h1>',
        f'<p>Images: {len(results)} | Loss: {metrics["loss"]:.4f} | '
        f'Accuracy: {metrics["accuracy"]:.2f}% | '
        f'Macro recall: {metrics["macro_recall"]:.2f}%</p>',
        '<p>Validation indices are positions in this validation split, not original dataset IDs. '
        'Images show the resized/padded input. Class statistics with few samples are noisy. '
        'Inspect examples for similar species, small birds, lighting, and backgrounds; '
        'these causes are not identified automatically.</p>',
        f'<h2>Highest-loss images</h2><img alt="Highest-loss validation images" '
        f'src="data:image/png;base64,{encoded_image}">',
    ]
    # Include every class and prediction in the report, not just the terminal preview.
    for title, table in tables:
        report.extend([f"<h2>{escape(title)}</h2>", table.to_html(float_format="%.4f")])
    report.append("</body></html>")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nOpen the image grid and full tables: {report_path.resolve()}")
    return results, per_class, confusion_pairs, metrics
