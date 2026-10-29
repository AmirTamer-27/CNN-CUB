"""Start training from the project root with python -m src.cli train --config PATH."""
import argparse

import torch
from torch import nn
import yaml
import mlflow
from src.data.dataset import make_loaders
from src.models.bird_cnn import BirdCNN
from src.models.bird_cnn_delayed_pooling import BirdCNNDelayedPooling
from src.models.bird_cnn_multidepth import BirdCNNMultiDepth
from src.models.bird_cnn_residual import BirdResidualCNN
from src.training.trainer import fit


def main():
    # 1. Read the command and configuration file.
    parser = argparse.ArgumentParser(description="Train a bird classifier")
    commands = parser.add_subparsers(dest="command", required=True)
    train_command = commands.add_parser("train")
    train_command.add_argument("--config", required=True)
    analysis_command = commands.add_parser(
        "analyze-validation", help="Inspect a saved model on the validation split"
    )
    analysis_command.add_argument("--config", required=True)
    analysis_command.add_argument("--weights", required=True)
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Only these choices have been implemented so far.
    if config["model"] not in (
        "bird_cnn", "bird_cnn_delayed_pooling", "bird_cnn_multidepth", "bird_cnn_residual",
    ):
        raise ValueError(
            "Choose model: bird_cnn, bird_cnn_delayed_pooling, "
            "bird_cnn_multidepth, or bird_cnn_residual."
        )
    if args.command == "train" and config["optimizer"] != "adam":
        raise ValueError("This version supports optimizer: adam.")

    # 2. Set the PyTorch seed and select the device.
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 3. Create the network structure selected in YAML.
    if config["model"] == "bird_cnn":
        model_class = BirdCNN
    elif config["model"] == "bird_cnn_delayed_pooling":
        model_class = BirdCNNDelayedPooling
    elif config["model"] == "bird_cnn_multidepth":
        model_class = BirdCNNMultiDepth
    else:
        model_class = BirdResidualCNN
    model = model_class(
        num_classes=config["num_classes"], dropout=config.get("dropout", 0.0)
    ).to(device)

    if args.command == "analyze-validation":
        # Load onto CPU first so files saved on a GPU also work on other devices.
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    # Recreate the same seeded split and preprocessing used during training.
    train_loader, validation_loader, class_names = make_loaders(config)
    if len(class_names) != config["num_classes"]:
        raise ValueError("num_classes must match the number of dataset classes.")
    print(f"Device: {device}")

    if args.command == "analyze-validation":
        from src.evaluation.validation_analysis import analyze_validation

        analyze_validation(
            model, validation_loader, class_names, device,
        )
        return

    # 4. Training creates an optimizer; analysis never updates the weights.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    # 5. Configure MLflow and start one run for this training attempt.
    mlflow.set_tracking_uri(
        "https://dagshub.com/AmirTamer-27/CNN-CUD.mlflow"
    )
    mlflow.set_experiment("bird-classification")

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params({
            "model": config["model"],
            "dropout": config.get("dropout", 0.0),
            "image_size": config["image_size"],
            "horizontal_flip_probability": config.get("horizontal_flip_probability", 0.0),
            "contrast_change_probability": config.get("contrast_change_probability", 0.0),
            "contrast_strength": config.get("contrast_strength", 0.2),
            "brightness_change_probability": config.get("brightness_change_probability", 0.0),
            "brightness_strength": config.get("brightness_strength", 0.2),
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "optimizer": config["optimizer"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config.get("weight_decay", 0.0),
            "seed": config["seed"],
            "validation_fraction": config["validation_fraction"],
            "train_dataset": config["train_dataset"],
        })

        history = fit(
            model, train_loader, validation_loader,
            criterion, optimizer, device, config,
        )
        weights_path = f"{config['run_name']}.pth"
        torch.save(model.state_dict(), weights_path)

        # Keep a copy in the current MLflow run.
        mlflow.log_artifact(weights_path, artifact_path="weights")

        # Upload the per-epoch history after training finishes.
        for result in history:
            mlflow.log_metrics(
                {
                    "train_loss": result["train"]["loss"],
                    "train_accuracy": result["train"]["accuracy"],
                    "train_macro_recall": result["train"]["macro_recall"],
                    "val_loss": result["validation"]["loss"],
                    "val_accuracy": result["validation"]["accuracy"],
                    "val_macro_recall": result["validation"]["macro_recall"],
                },
                step=result["epoch"],
            )


if __name__ == "__main__":
    main()
