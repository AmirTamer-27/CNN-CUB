"""Train Lab color-bin U-Net with a frozen Phase 1 encoder."""

import argparse
import math
from pathlib import Path

import mlflow
import torch
from torch import nn
from tqdm.auto import tqdm
import yaml

from src.data.dataset import make_loaders
from src.evaluation.test_evaluation import TRACKING_URI, download_run_weights
from src.models.bird_cnn_delayed_pooling import BirdCNNDelayedPooling
from src.models.colorization_lab_unet import BirdLabColorizationUNet
from src.training.colorization_lab.colors import (
    BIN_URL,
    load_bins,
    prepare_batch,
    reconstruct_rgb,
    target_bins,
)


def train_one_epoch(model, loader, criterion, optimizer, centers, device):
    model.train()
    total_loss = 0.0
    total_pixels = 0

    for batch in tqdm(loader, desc="Colorization training"):
        target_rgb = batch["image"].to(
            device, non_blocking=device.type == "cuda"
        )
        lightness_input, _, target_ab = prepare_batch(target_rgb)
        targets = target_bins(target_ab, centers)

        optimizer.zero_grad()
        logits = model(lightness_input)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        pixels = targets.numel()
        total_loss += loss.item() * pixels
        total_pixels += pixels

    return total_loss / total_pixels


def validate(model, loader, criterion, centers, device):
    model.eval()
    total_mae = 0.0
    total_images = 0
    total_squared_error = 0.0
    total_values = 0
    total_loss = 0.0
    total_pixels = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Colorization validation"):
            target_rgb = batch["image"].to(
                device, non_blocking=device.type == "cuda"
            )
            lightness_input, lightness, target_ab = prepare_batch(target_rgb)
            targets = target_bins(target_ab, centers)

            logits = model(lightness_input)
            loss = criterion(logits, targets)
            predicted_rgb = reconstruct_rgb(logits, lightness, centers)
            mae = torch.mean(torch.abs(predicted_rgb - target_rgb))
            pixels = targets.numel()
            total_loss += loss.item() * pixels
            total_pixels += pixels

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

    return total_loss / total_pixels, validation_mae, validation_psnr


def save_colorization_weights(model, checkpoint_path):
    """The source encoder is restored separately from its MLflow run."""
    weights = {name: value for name, value in model.state_dict().items()
               if not name.startswith("encoder.")}
    torch.save(weights, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Train the bird Lab color-bin U-Net")
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

    model = BirdLabColorizationUNet(encoder=encoder).to(device)
    print(
        "Trainable encoder blocks: "
        f"{model.trainable_encoder_blocks or 'none'}"
    )
    print(
        "Adapt BatchNorm in trainable blocks: "
        f"{model.adapt_trainable_batch_norm}"
    )

    train_loader, validation_loader, _ = make_loaders(config)
    centers = load_bins(config["color_bins_path"], device)
    criterion = nn.CrossEntropyLoss()

    decoder_learning_rate = config["decoder_learning_rate"]
    if decoder_learning_rate <= 0:
        raise ValueError("Decoder learning rate must be positive.")

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=decoder_learning_rate,
        weight_decay=config.get("weight_decay", 0.0),
    )

    checkpoint_path = Path(config["checkpoint_path"])
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
            "trainable_encoder_blocks": ",".join(
                model.trainable_encoder_blocks
            ) or "none",
            "adapt_trainable_batch_norm": model.adapt_trainable_batch_norm,
            "image_size": config["image_size"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "decoder_learning_rate": decoder_learning_rate,
            "color_bins": 313,
            "color_bins_source": BIN_URL,
            "weight_decay": config.get("weight_decay", 0.0),
            "loss": "cross_entropy_nearest_bin",
            "use_bounding_boxes": config["use_bounding_boxes"],
            "seed": config["seed"],
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        })

        for epoch in range(config["epochs"]):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, centers, device
            )
            validation_loss, validation_mae, validation_psnr = validate(
                model, validation_loader, criterion, centers, device
            )

            print(
                f"Epoch {epoch + 1}/{config['epochs']} | "
                f"Train CE: {train_loss:.4f} | Validation CE: {validation_loss:.4f} | "
                f"Validation MAE: {validation_mae:.4f} | "
                f"Validation PSNR: {validation_psnr:.2f} dB"
            )
            mlflow.log_metrics(
                {
                    "train_cross_entropy": train_loss,
                    "validation_cross_entropy": validation_loss,
                    "validation_mae": validation_mae,
                    "validation_psnr": validation_psnr,
                },
                step=epoch + 1,
            )

            if validation_mae < best_validation_mae:
                best_validation_mae = validation_mae
                save_colorization_weights(model, checkpoint_path)
                print(f"Best colorization checkpoint saved: {checkpoint_path}")

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
        mlflow.log_artifact(config["color_bins_path"], artifact_path="color_bins")


if __name__ == "__main__":
    main()
