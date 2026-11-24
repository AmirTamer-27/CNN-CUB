"""Display random validation targets beside their reconstructed color images."""

import argparse
import math
from pathlib import Path
import random

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import torch
from torchvision.transforms.functional import rgb_to_grayscale
import yaml

from src.data.dataset import make_loaders
from src.evaluation.test_evaluation import TRACKING_URI, download_run_weights
from src.models.bird_cnn_delayed_pooling import BirdCNNDelayedPooling
from src.models.colorization_unet import BirdColorizationUNet


def download_colorization_weights(run_id):
    """Download colorization weights and read linked encoder information."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    run = client.get_run(run_id)

    encoder_run_id = run.data.params.get("encoder_run_id")
    encoder_artifact = run.data.params.get("encoder_artifact")

    if encoder_run_id is None or encoder_artifact is None:
        raise ValueError(
            "The selected run does not contain encoder_run_id and encoder_artifact."
        )

    candidates = [
        artifact.path
        for artifact in client.list_artifacts(run_id, "checkpoints")
        if not artifact.is_dir and artifact.path.endswith(".pth")
    ]

    if len(candidates) != 1:
        raise ValueError(
            f"Expected one colorization checkpoint, but found {len(candidates)}."
        )

    checkpoint_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=candidates[0],
        dst_path=str(Path("outputs/downloaded_weights") / run_id),
    )

    return checkpoint_path, encoder_run_id, encoder_artifact


def build_model(config, colorization_run_id, device):
    checkpoint_path, encoder_run_id, encoder_artifact = download_colorization_weights(
        colorization_run_id
    )

    source_config, encoder_path, _, _ = download_run_weights(
        config,
        encoder_run_id,
        encoder_artifact,
    )

    if source_config["model"] != "bird_cnn_delayed_pooling":
        raise ValueError("Colorization requires a delayed-pooling encoder run.")

    encoder = BirdCNNDelayedPooling(
        num_classes=source_config["num_classes"],
        dropout=source_config.get("dropout", 0.0),
        dropout2d=source_config.get("dropout2d", 0.0),
    )
    encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu", weights_only=True)
    )

    model = BirdColorizationUNet(
        encoder=encoder,
        trainable_encoder_blocks=[],
        adapt_trainable_batch_norm=False,
    )
    colorization_weights = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    missing_keys, unexpected_keys = model.load_state_dict(
        colorization_weights,
        strict=False,
    )

    invalid_missing_keys = [
        key for key in missing_keys if not key.startswith("encoder.")
    ]
    if invalid_missing_keys or unexpected_keys:
        raise ValueError(
            "Checkpoint does not match the colorization model. "
            f"Missing: {invalid_missing_keys}; unexpected: {unexpected_keys}"
        )

    return model.to(device)


def collect_random_samples(model, loader, device, sample_count, seed):
    available_samples = len(loader.dataset)
    sample_count = min(sample_count, available_samples)
    selected_indices = set(
        random.Random(seed).sample(range(available_samples), sample_count)
    )

    samples = []
    current_index = 0
    model.eval()

    with torch.inference_mode():
        for batch in loader:
            target_rgb = batch["image"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            grayscale = rgb_to_grayscale(target_rgb, num_output_channels=3)
            predicted_rgb = model(grayscale)
            image_mae = torch.abs(predicted_rgb - target_rgb).mean(
                dim=(1, 2, 3)
            )

            for batch_index in range(target_rgb.size(0)):
                if current_index in selected_indices:
                    samples.append({
                        "index": current_index,
                        "target": target_rgb[batch_index].cpu(),
                        "prediction": predicted_rgb[batch_index].cpu(),
                        "mae": image_mae[batch_index].item(),
                    })

                current_index += 1

            if len(samples) == sample_count:
                break

    samples.sort(key=lambda sample: sample["index"])
    return samples


def save_comparison_grid(samples, output_path):
    pairs_per_row = 3
    rows = math.ceil(len(samples) / pairs_per_row)
    figure, axes = plt.subplots(
        rows,
        pairs_per_row * 2,
        figsize=(18, rows * 5),
    )
    axes = axes.reshape(rows, pairs_per_row * 2)

    for axis in axes.flat:
        axis.axis("off")

    for sample_number, sample in enumerate(samples):
        row = sample_number // pairs_per_row
        pair = sample_number % pairs_per_row
        target_axis = axes[row, pair * 2]
        prediction_axis = axes[row, pair * 2 + 1]

        target_axis.imshow(sample["target"].permute(1, 2, 0).numpy())
        target_axis.set_title(
            f"Sample {sample['index']} - Actual RGB"
        )
        target_axis.axis("off")

        prediction_axis.imshow(
            sample["prediction"].permute(1, 2, 0).numpy()
        )
        prediction_axis.set_title(
            f"Constructed RGB - MAE {sample['mae']:.4f}"
        )
        prediction_axis.axis("off")

    figure.suptitle(
        "Random Colorization Validation Samples",
        fontsize=18,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Validation comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze random colorization validation samples"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--output",
        default="outputs/validation_analysis/colorization_random_samples.png",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    model = build_model(config, args.run_id, device)
    _, validation_loader, _ = make_loaders(config)
    samples = collect_random_samples(
        model,
        validation_loader,
        device,
        args.samples,
        config["seed"],
    )
    save_comparison_grid(samples, args.output)


if __name__ == "__main__":
    main()
