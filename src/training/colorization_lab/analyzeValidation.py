"""View random Lab-bin colorizations beside the original validation images."""

import argparse
from pathlib import Path
import random

import mlflow
import torch
import yaml

from src.data.dataset import make_loaders
from src.evaluation.test_evaluation import download_run_weights
from src.models.bird_cnn_delayed_pooling import BirdCNNDelayedPooling
from src.models.colorization_lab_unet import BirdLabColorizationUNet
from src.training.colorization.analyzeValidation import (
    download_colorization_weights,
    save_comparison_grid,
)
from src.training.colorization_lab.colors import (
    prepare_batch,
    reconstruct_rgb,
    load_bins,
)


def build_model(config, run_id, device):
    checkpoint_path, encoder_run_id, encoder_artifact = download_colorization_weights(
        run_id
    )
    source_config, encoder_path, _, _ = download_run_weights(
        config, encoder_run_id, encoder_artifact
    )
    if source_config["model"] != "bird_cnn_delayed_pooling":
        raise ValueError("Expected a delayed-pooling encoder.")

    encoder = BirdCNNDelayedPooling(
        num_classes=source_config["num_classes"],
        dropout=source_config.get("dropout", 0.0),
        dropout2d=source_config.get("dropout2d", 0.0),
    )
    encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu", weights_only=True)
    )
    model = BirdLabColorizationUNet(encoder=encoder)
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or any(not name.startswith("encoder.") for name in missing):
        raise ValueError(f"Checkpoint does not match Lab model: {missing}, {unexpected}")
    return model.to(device).eval()


def download_bins(run_id, device):
    """Use the exact bin centers saved with this training run."""
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="color_bins/pts_in_hull.npy",
        dst_path=str(Path("outputs/downloaded_weights") / run_id),
    )
    return load_bins(path, device)


def collect_samples(model, loader, centers, device, count, seed):
    count = min(count, len(loader.dataset))
    chosen = set(random.Random(seed).sample(range(len(loader.dataset)), count))
    samples = []
    index = 0
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["image"].to(device, non_blocking=device.type == "cuda")
            lightness_input, lightness, _ = prepare_batch(rgb)
            predicted = reconstruct_rgb(model(lightness_input), lightness, centers)
            errors = (predicted - rgb).abs().mean(dim=(1, 2, 3))
            for position in range(rgb.size(0)):
                if index in chosen:
                    samples.append({
                        "index": index,
                        "target": rgb[position].cpu(),
                        "prediction": predicted[position].cpu(),
                        "mae": errors[position].item(),
                    })
                index += 1
            if len(samples) == count:
                break
    return sorted(samples, key=lambda sample: sample["index"])


def main():
    parser = argparse.ArgumentParser(description="Analyze Lab-bin validation samples")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True, help="Lab colorization MLflow run ID")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--output",
        default="outputs/validation_analysis/colorization_lab_random_samples.png",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero.")
    with open(args.config) as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(config, args.run_id, device)
    centers = download_bins(args.run_id, device)
    _, validation_loader, _ = make_loaders(config)
    samples = collect_samples(
        model, validation_loader, centers, device, args.samples, config["seed"]
    )
    save_comparison_grid(samples, args.output)


if __name__ == "__main__":
    main()
