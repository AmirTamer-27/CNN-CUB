"""Train an RGB U-Net generator with a whole-image conditional discriminator."""

import argparse
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
from src.models.colorization_gan import BirdDiscriminator, BirdGANGenerator
from src.training.colorization.train import save_colorization_weights, validate


def train_one_epoch(generator, discriminator, loader, generator_optimizer,
                    discriminator_optimizer, lambda_mae, lambda_adversarial, device):
    generator.train()
    discriminator.train()
    bce = nn.BCEWithLogitsLoss()
    mae_loss = nn.L1Loss()
    totals = {"generator_mae": 0.0, "generator_adversarial": 0.0,
              "discriminator": 0.0}
    images = 0

    for batch in tqdm(loader, desc="GAN colorization training"):
        target = batch["image"].to(device, non_blocking=device.type == "cuda")
        grayscale = rgb_to_grayscale(target, num_output_channels=3)
        generated = generator(grayscale)

        # Update the discriminator using real pairs and detached generated pairs.
        discriminator.train()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)
        discriminator_optimizer.zero_grad()
        real_logits = discriminator(grayscale, target)
        fake_logits = discriminator(grayscale, generated.detach())
        discriminator_loss = 0.5 * (
            bce(real_logits, torch.ones_like(real_logits))
            + bce(fake_logits, torch.zeros_like(fake_logits))
        )
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Keep discriminator weights and BatchNorm statistics fixed for G's update.
        discriminator.eval()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        generator_optimizer.zero_grad()
        fake_logits = discriminator(grayscale, generated)
        adversarial_loss = bce(fake_logits, torch.ones_like(fake_logits))
        mae = mae_loss(generated, target)
        generator_loss = lambda_mae * mae + lambda_adversarial * adversarial_loss
        generator_loss.backward()
        generator_optimizer.step()

        count = target.size(0)
        totals["generator_mae"] += mae.item() * count
        totals["generator_adversarial"] += adversarial_loss.item() * count
        totals["discriminator"] += discriminator_loss.item() * count
        images += count

    return {name: value / images for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train conditional whole-image colorization GAN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True, help="Phase 1 encoder run ID")
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
        config, args.run_id, args.artifact_path
    )
    if source_config["model"] != "bird_cnn_delayed_pooling":
        raise ValueError("GAN colorization requires a delayed-pooling encoder.")
    encoder = BirdCNNDelayedPooling(
        num_classes=source_config["num_classes"],
        dropout=source_config.get("dropout", 0.0),
        dropout2d=source_config.get("dropout2d", 0.0),
    )
    encoder.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    generator = BirdGANGenerator(
        encoder=encoder,
        trainable_encoder_blocks=config["trainable_encoder_blocks"],
        adapt_trainable_batch_norm=config["adapt_trainable_batch_norm"],
    ).to(device)
    discriminator = BirdDiscriminator().to(device)
    print(f"Trainable encoder blocks: {generator.trainable_encoder_blocks or 'none'}")

    decoder_params = [p for name, p in generator.named_parameters()
                      if not name.startswith("encoder.") and p.requires_grad]
    encoder_params = [p for name, p in generator.named_parameters()
                      if name.startswith("encoder.") and p.requires_grad]
    groups = [{"params": decoder_params, "lr": config["decoder_learning_rate"]}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": config["encoder_learning_rate"]})
    for key in ("decoder_learning_rate", "discriminator_learning_rate"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if encoder_params and config["encoder_learning_rate"] <= 0:
        raise ValueError("encoder_learning_rate must be positive.")
    if config["lambda_mae"] < 0 or config["lambda_adversarial"] < 0:
        raise ValueError("Loss weights must be nonnegative.")
    if config["lambda_mae"] + config["lambda_adversarial"] == 0:
        raise ValueError("At least one generator loss must have positive weight.")
    generator_optimizer = torch.optim.Adam(groups, weight_decay=config["weight_decay"])
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config["discriminator_learning_rate"],
        weight_decay=config["weight_decay"],
    )
    train_loader, validation_loader, _ = make_loaders(config)
    generator_path = Path(config["checkpoint_path"])
    discriminator_path = Path(config["discriminator_checkpoint_path"])
    generator_path.parent.mkdir(parents=True, exist_ok=True)
    discriminator_path.parent.mkdir(parents=True, exist_ok=True)
    best_mae = float("inf")
    stopping_reference = float("inf")
    stale_epochs = 0

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("bird-colorization")
    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params({
            "encoder_run_id": args.run_id, "encoder_artifact": source_artifact,
            "trainable_encoder_blocks": ",".join(generator.trainable_encoder_blocks) or "none",
            "adapt_trainable_batch_norm": generator.adapt_trainable_batch_norm,
            "discriminator": "whole_image", "loss": "mae_plus_bce_logits",
            "lambda_mae": config["lambda_mae"],
            "lambda_adversarial": config["lambda_adversarial"],
            "decoder_learning_rate": config["decoder_learning_rate"],
            "encoder_learning_rate": config["encoder_learning_rate"],
            "discriminator_learning_rate": config["discriminator_learning_rate"],
            "batch_size": config["batch_size"], "epochs": config["epochs"],
            "image_size": config["image_size"], "seed": config["seed"],
            "use_bounding_boxes": config["use_bounding_boxes"],
            "early_stopping_patience": config["early_stopping_patience"],
            "early_stopping_min_delta": config["early_stopping_min_delta"],
        })
        for epoch in range(config["epochs"]):
            training = train_one_epoch(
                generator, discriminator, train_loader,
                generator_optimizer, discriminator_optimizer,
                config["lambda_mae"], config["lambda_adversarial"], device,
            )
            validation_mae, validation_psnr = validate(
                generator, validation_loader, nn.L1Loss(), device
            )
            print(f"Epoch {epoch + 1}/{config['epochs']} | "
                  f"G MAE: {training['generator_mae']:.4f} | "
                  f"G adv: {training['generator_adversarial']:.4f} | "
                  f"D loss: {training['discriminator']:.4f} | "
                  f"Val MAE: {validation_mae:.4f} | Val PSNR: {validation_psnr:.2f} dB")
            mlflow.log_metrics({**training, "validation_mae": validation_mae,
                                "validation_psnr": validation_psnr}, step=epoch + 1)

            if validation_mae < best_mae:
                best_mae = validation_mae
                save_colorization_weights(generator, generator_path)
                torch.save(discriminator.state_dict(), discriminator_path)
            if stopping_reference - validation_mae > config["early_stopping_min_delta"]:
                stopping_reference = validation_mae
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config["early_stopping_patience"]:
                    print("Early stopping triggered.")
                    break
        mlflow.log_metric("best_validation_mae", best_mae)
        mlflow.log_artifact(str(generator_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(discriminator_path), artifact_path="discriminator_checkpoints")


if __name__ == "__main__":
    main()
