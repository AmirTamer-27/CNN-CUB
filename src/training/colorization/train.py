"""Train the U-Net colorizer using a Phase 1 encoder from MLflow."""

import argparse
import math
from pathlib import Path

import mlflow
import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale
from tqdm.auto import tqdm
import yaml

from src.data.dataset import make_loaders
from src.evaluation.test_evaluation import TRACKING_URI, download_run_weights
from src.models.bird_cnn_delayed_pooling import BirdCNNDelayedPooling
from src.models.colorization_unet import BirdColorizationUNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_mae = 0.0
    total_images = 0

    for batch in tqdm(loader, desc="Colorization training"):
        target_rgb = batch["image"].to(
            device, non_blocking=device.type == "cuda"
        )
        grayscale = rgb_to_grayscale(target_rgb, num_output_channels=3)

        optimizer.zero_grad()
        predicted_rgb = model(grayscale)
        mae = criterion(predicted_rgb, target_rgb)
        mae.backward()
        optimizer.step()

        batch_size = target_rgb.size(0)
        total_mae += mae.item() * batch_size
        total_images += batch_size

    return total_mae / total_images


def validate(model, loader, criterion, device):
    model.eval()
    total_mae = 0.0
    total_images = 0
    total_squared_error = 0.0
    total_values = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Colorization validation"):
            target_rgb = batch["image"].to(
                device, non_blocking=device.type == "cuda"
            )
            grayscale = rgb_to_grayscale(target_rgb, num_output_channels=3)

            predicted_rgb = model(grayscale)
            mae = criterion(predicted_rgb, target_rgb)

            batch_size = target_rgb.size(0)
            total_mae += mae.item() * batch_size
            total_images += batch_size
            total_squared_error += torch.sum(
                (predicted_rgb - target_rgb) ** 2
            ).item()
            total_values += target_rgb.numel()

    validation_mae = total_mae / total_images
    validation_mse = total_squared_error / total_values

    if validation_mse == 0:
        validation_psnr = float("inf")
    else:
        validation_psnr = 10 * math.log10(1.0 / validation_mse)

    return validation_mae, validation_psnr


def save_decoder_weights(model, checkpoint_path):
    """Save the decoder and RGB output layer without the encoder weights."""
    decoder_weights = {}

    for name, weights in model.state_dict().items():
        if not name.startswith("encoder."):
            decoder_weights[name] = weights

    torch.save(decoder_weights, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Train the bird colorization U-Net")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-path")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    source_config, weights_path, _, source_artifact = download_run_weights(
        config,
        args.run_id,
        args.artifact_path,
    )
    if source_config["model"] != "bird_cnn_delayed_pooling":
        raise ValueError("Colorization currently requires a delayed-pooling encoder run.")

    encoder = BirdCNNDelayedPooling(
        num_classes=source_config["num_classes"],
        dropout=source_config.get("dropout", 0.0),
        dropout2d=source_config.get("dropout2d", 0.0),
    )
    encoder.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )

    model = BirdColorizationUNet(
        encoder=encoder,
        freeze_encoder=config.get("freeze_encoder", True),
    ).to(device)

    train_loader, validation_loader, _ = make_loaders(config)
    criterion = nn.L1Loss()

    trainable_parameters = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    checkpoint_path = Path(config["decoder_checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_mae = float("inf")
    best_validation_psnr = 0.0
    early_stopping_best_mae = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = config["early_stopping_patience"]
    early_stopping_min_delta = config["early_stopping_min_delta"]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("bird-colorization")

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params({
            "encoder_run_id": args.run_id,
            "encoder_artifact": source_artifact,
            "freeze_encoder": config.get("freeze_encoder", True),
            "image_size": config["image_size"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config.get("weight_decay", 0.0),
            "loss": "mae",
            "use_bounding_boxes": config["use_bounding_boxes"],
            "seed": config["seed"],
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        })

        for epoch in range(config["epochs"]):
            train_mae = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            validation_mae, validation_psnr = validate(
                model, validation_loader, criterion, device
            )

            print(
                f"Epoch {epoch + 1}/{config['epochs']} | "
                f"Train MAE: {train_mae:.4f} | "
                f"Validation MAE: {validation_mae:.4f} | "
                f"Validation PSNR: {validation_psnr:.2f} dB"
            )
            mlflow.log_metrics(
                {
                    "train_mae": train_mae,
                    "validation_mae": validation_mae,
                    "validation_psnr": validation_psnr,
                },
                step=epoch + 1,
            )

            if validation_mae < best_validation_mae:
                best_validation_mae = validation_mae
                save_decoder_weights(model, checkpoint_path)
                print(f"Best decoder checkpoint saved: {checkpoint_path}")

            best_validation_psnr = max(best_validation_psnr, validation_psnr)

            improvement = early_stopping_best_mae - validation_mae

            if improvement > early_stopping_min_delta:
                early_stopping_best_mae = validation_mae
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(
                    "Epochs without sufficient improvement: "
                    f"{epochs_without_improvement}/{early_stopping_patience}"
                )

                if epochs_without_improvement >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        mlflow.log_metric("best_validation_mae", best_validation_mae)
        mlflow.log_metric("best_validation_psnr", best_validation_psnr)
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")


if __name__ == "__main__":
    main()
