"""Start training from the project root with:
python -m src.cli train --config PATH

Analyze validation:
python -m src.cli analyze-validation --config PATH --weights PATH
"""

import argparse
import math

import torch
from torch import nn
import yaml
import mlflow

# Existing project imports
from src.data.dataset import make_loaders
from src.models.bird_cnn import BirdCNN
from src.models.bird_cnn_delayed_pooling import BirdCNNDelayedPooling
from src.models.bird_cnn_multidepth import BirdCNNMultiDepth
from src.models.bird_cnn_residual import BirdResidualCNN
from src.models.bird_resnet18 import BirdResNet18
from src.training.trainer import fit

# ResNeXt imports
from src.data.transfer_dataset import make_transfer_loaders
from src.models.transfer import create_resnext_classifier
from src.training.transfer_trainer import fit_transfer


def main():
    parser = argparse.ArgumentParser(description="Train a bird classifier")
    commands = parser.add_subparsers(dest="command", required=True)

    train_command = commands.add_parser("train")
    train_command.add_argument("--config", required=True)

    analysis_command = commands.add_parser(
        "analyze-validation",
        help="Inspect a saved model on the validation split",
    )
    analysis_command.add_argument("--config", required=True)
    analysis_command.add_argument("--weights", required=True)

    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # 1. Validate the selected model.
    supported_models = (
        "bird_cnn",
        "bird_cnn_delayed_pooling",
        "bird_cnn_multidepth",
        "bird_cnn_residual",
        "bird_resnet18",
        "resnext101",
    )

    if config["model"] not in supported_models:
        raise ValueError(
            f"Unsupported model. Choose one of: {supported_models}"
        )

    # Validate optimizer and scheduler only when training scratch models.
    if args.command == "train" and config["model"] != "resnext101":
        if config["optimizer"] not in ("adam", "sgd"):
            raise ValueError("Choose optimizer: adam or sgd.")

        if config.get("lr_scheduler", "none") not in (
            "none", "step", "plateau"
        ):
            raise ValueError(
                "Choose lr_scheduler: none, step, or plateau."
            )

        if config.get("loss", "cross_entropy") not in ("cross_entropy", "combined"):
            raise ValueError("Choose loss: cross_entropy or combined.")
        cosine_weight = config.get("cosine_weight", 1.0)
        if not math.isfinite(cosine_weight) or cosine_weight < 0:
            raise ValueError("cosine_weight must be finite and nonnegative.")

    # 2. Set the PyTorch seed and select the device.
    torch.manual_seed(config["seed"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 3. Create the selected model.
    if config["model"] == "resnext101":
        model = create_resnext_classifier(
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        ).to(device)

    else:
        model_classes = {
            "bird_cnn": BirdCNN,
            "bird_cnn_delayed_pooling": BirdCNNDelayedPooling,
            "bird_cnn_multidepth": BirdCNNMultiDepth,
            "bird_cnn_residual": BirdResidualCNN,
            "bird_resnet18": BirdResNet18,
        }

        model_class = model_classes[config["model"]]

        model = model_class(
            num_classes=config["num_classes"],
            dropout=config.get("dropout", 0.0),
        ).to(device)

    # 4. Prepare the appropriate data loaders.
    if config["model"] == "resnext101":
        train_loader, validation_loader, class_names = (
            make_transfer_loaders(config)
        )
    else:
        train_loader, validation_loader, class_names = make_loaders(config)

    if len(class_names) != config["num_classes"]:
        raise ValueError(
            "num_classes must match the number of dataset classes."
        )

    print(f"Model: {config['model']}")
    print(f"Device: {device}")

    # 5. Handle validation analysis.
    if args.command == "analyze-validation":
        if config["model"] == "resnext101":
            raise NotImplementedError(
                "analyze-validation currently supports the five "
                "existing scratch models, not ResNeXt checkpoints."
            )

        # Load onto CPU first so GPU-saved weights work on other devices.
        state = torch.load(
            args.weights,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)

        from src.evaluation.validation_analysis import analyze_validation

        analyze_validation(
            model,
            validation_loader,
            class_names,
            device,
        )
        return

    # 6. Configure MLflow.
    mlflow.set_tracking_uri(
        "https://dagshub.com/AmirTamer-27/CNN-CUD.mlflow"
    )
    mlflow.set_experiment("bird-classification")

    with mlflow.start_run(run_name=config["run_name"]):

        # -------------------------------------------------
        # Existing scratch-model training pipeline
        # -------------------------------------------------
        if config["model"] != "resnext101":

            criterion = nn.CrossEntropyLoss()

            if config["optimizer"] == "adam":
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config["learning_rate"],
                    weight_decay=config.get("weight_decay", 0.0),
                )
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=config["learning_rate"],
                    momentum=config.get("momentum", 0.9),
                    weight_decay=config.get("weight_decay", 0.0),
                )

            # Existing teammate parameter logging.
            mlflow.log_params({
                "model": config["model"],
                "dropout": config.get("dropout", 0.0),
                "image_size": config["image_size"],
                "horizontal_flip_probability": config.get(
                    "horizontal_flip_probability", 0.0
                ),
                "contrast_change_probability": config.get(
                    "contrast_change_probability", 0.0
                ),
                "contrast_strength": config.get(
                    "contrast_strength", 0.2
                ),
                "brightness_change_probability": config.get(
                    "brightness_change_probability", 0.0
                ),
                "brightness_strength": config.get(
                    "brightness_strength", 0.2
                ),
                "cutout_probability": config.get(
                    "cutout_probability", 0.0
                ),
                "cutout_min_fraction": config.get(
                    "cutout_min_fraction", 0.25
                ),
                "cutout_max_fraction": config.get(
                    "cutout_max_fraction", 0.30
                ),
                "cutout_fill": 128 / 255,
                "batch_size": config["batch_size"],
                "epochs": config["epochs"],
                "optimizer": config["optimizer"],
                "loss": config.get("loss", "cross_entropy"),
                "cosine_weight": (
                    config.get("cosine_weight", 1.0)
                    if config.get("loss", "cross_entropy") == "combined" else 0.0
                ),
                "momentum": config.get("momentum", 0.0),
                "pretrained": False,
                "lr_scheduler": config.get("lr_scheduler", "none"),
                "lr_step_size": config.get("lr_step_size", 30),
                "lr_gamma": config.get("lr_gamma", 0.1),
                "lr_plateau_monitor": "val_loss",
                "lr_plateau_factor": config.get(
                    "lr_plateau_factor", 0.5
                ),
                "lr_plateau_patience": config.get(
                    "lr_plateau_patience", 5
                ),
                "lr_plateau_threshold": config.get(
                    "lr_plateau_threshold", 0.001
                ),
                "lr_plateau_threshold_mode": "abs",
                "lr_plateau_min_lr": config.get(
                    "lr_plateau_min_lr", 0.000001
                ),
                "learning_rate": config["learning_rate"],
                "weight_decay": config.get("weight_decay", 0.0),
                "seed": config["seed"],
                "validation_fraction": config["validation_fraction"],
                "train_dataset": config["train_dataset"],
            })

            history = fit(
                model,
                train_loader,
                validation_loader,
                criterion,
                optimizer,
                device,
                config,
            )

            # Existing teammate checkpoint saving.
            weights_path = f"{config['run_name']}.pth"
            torch.save(model.state_dict(), weights_path)
            mlflow.log_artifact(weights_path, artifact_path="weights")

            # Existing teammate metric logging.
            for result in history:
                # Preserve CE metrics and record combined-loss components separately.
                component_metrics = {}
                for split, prefix in (("train", "train"), ("validation", "val")):
                    for name in ("cosine_loss", "total_loss"):
                        if name in result[split]:
                            component_metrics[f"{prefix}_{name}"] = result[split][name]
                mlflow.log_metrics(
                    {
                        **component_metrics,
                        "train_loss": result["train"]["loss"],
                        "train_accuracy": result["train"]["accuracy"],
                        "train_macro_recall": result["train"]["macro_recall"],
                        "val_loss": result["validation"]["loss"],
                        "val_accuracy": result["validation"]["accuracy"],
                        "val_macro_recall": result["validation"]["macro_recall"],
                        "learning_rate": result.get(
                            "learning_rate", config["learning_rate"]
                        ),
                    },
                    step=result["epoch"],
                )

        # -------------------------------------------------
        # Your ResNeXt transfer-learning pipeline
        # -------------------------------------------------
        else:

            mlflow.log_params({
                key: value
                for key, value in config.items()
                if isinstance(value, (str, int, float, bool))
            })

            mlflow.log_param("device", str(device))
            mlflow.log_param("optimizer", "AdamW")

            model, history, checkpoint_path = fit_transfer(
                model,
                train_loader,
                validation_loader,
                device,
                config,
                class_names,
            )

            for step, result in enumerate(history, start=1):
                mlflow.log_metrics(
                    {
                        "train_loss": result["train"]["loss"],
                        "train_accuracy": result["train"]["accuracy"],
                        "train_macro_recall": result["train"]["macro_recall"],
                        "val_loss": result["validation"]["loss"],
                        "val_accuracy": result["validation"]["accuracy"],
                        "val_macro_recall": result["validation"]["macro_recall"],
                    },
                    step=step,
                )

            mlflow.log_artifact(
                checkpoint_path,
                artifact_path="checkpoints",
            )


if __name__ == "__main__":
    main()