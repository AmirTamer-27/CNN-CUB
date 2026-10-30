
"""Two-stage training workflow for ResNeXt transfer learning."""

from pathlib import Path

import torch
import torch.nn as nn

from src.training.trainer import train_one_epoch, validate
from src.models.transfer_utils import (
    freeze_backbone,
    unfreeze_layer4,
    unfreeze_classifier,
)


def _run_epoch(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device,
    num_classes,
):
    """Run one training epoch followed by validation."""

    train_metrics = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        num_classes,
    )

    validation_metrics = validate(
        model,
        validation_loader,
        criterion,
        device,
        num_classes,
    )

    return train_metrics, validation_metrics


def _save_checkpoint(
    model,
    checkpoint_path,
    stage,
    epoch,
    validation_metrics,
    class_names,
    config,
):
    """Save the current best model checkpoint."""

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stage": stage,
            "epoch": epoch,
            "validation_metrics": validation_metrics,
            "class_names": class_names,
            "config": config,
        },
        checkpoint_path,
    )


def fit_transfer(
    model,
    train_loader,
    validation_loader,
    device,
    config,
    class_names,
):
    """
    Train ResNeXt in two stages.

    Stage 1: Train only the new classifier.
    Stage 2: Fine-tune layer4 and the classifier.
    """

    criterion = nn.CrossEntropyLoss()
    num_classes = config["num_classes"]

    checkpoint_dir = Path(
        config.get("checkpoint_dir", "outputs/checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "resnext101_best.pth"

    history = []

    best_macro_recall = float("-inf")
    best_validation_loss = float("inf")
    best_state = None

    # --------------------------------------------------
    # STAGE 1: Train the classifier only
    # --------------------------------------------------

    print("\nStarting Stage 1: Classifier training")

    freeze_backbone(model)
    unfreeze_classifier(model)

    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=config["stage1_learning_rate"],
        weight_decay=config["stage1_weight_decay"],
    )

    for epoch in range(config["stage1_epochs"]):

        print(
            f"\nStage 1 - Epoch {epoch + 1}/"
            f"{config['stage1_epochs']}"
        )

        train_metrics, validation_metrics = _run_epoch(
            model,
            train_loader,
            validation_loader,
            criterion,
            optimizer,
            device,
            num_classes,
        )

        history.append({
            "stage": 1,
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        })

        print(
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train accuracy: {train_metrics['accuracy']:.2f}% | "
            f"Val loss: {validation_metrics['loss']:.4f} | "
            f"Val accuracy: {validation_metrics['accuracy']:.2f}% | "
            f"Val macro recall: "
            f"{validation_metrics['macro_recall']:.2f}%"
        )

        current_recall = validation_metrics["macro_recall"]
        current_loss = validation_metrics["loss"]

        # Select the best checkpoint using macro recall.
        if (
            current_recall > best_macro_recall
            or (
                current_recall == best_macro_recall
                and current_loss < best_validation_loss
            )
        ):
            best_macro_recall = current_recall
            best_validation_loss = current_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            _save_checkpoint(
                model,
                checkpoint_path,
                stage=1,
                epoch=epoch + 1,
                validation_metrics=validation_metrics,
                class_names=class_names,
                config=config,
            )

            print("Best checkpoint updated.")


    print("\nStarting Stage 2: Fine-tuning")

    freeze_backbone(model)
    unfreeze_layer4(model)
    unfreeze_classifier(model)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.layer4.parameters(),
                "lr": config["layer4_learning_rate"],
            },
            {
                "params": model.fc.parameters(),
                "lr": config["classifier_learning_rate"],
            },
        ],
        weight_decay=config["stage2_weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["scheduler_patience"],
        min_lr=config.get("minimum_learning_rate", 1e-7),
    )

    stage2_best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config["stage2_epochs"]):

        print(
            f"\nStage 2 - Epoch {epoch + 1}/"
            f"{config['stage2_epochs']}"
        )

        train_metrics, validation_metrics = _run_epoch(
            model,
            train_loader,
            validation_loader,
            criterion,
            optimizer,
            device,
            num_classes,
        )

        history.append({
            "stage": 2,
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        })

        current_loss = validation_metrics["loss"]
        current_recall = validation_metrics["macro_recall"]

        print(
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train accuracy: {train_metrics['accuracy']:.2f}% | "
            f"Val loss: {current_loss:.4f} | "
            f"Val accuracy: {validation_metrics['accuracy']:.2f}% | "
            f"Val macro recall: {current_recall:.2f}%"
        )

        # Adjust learning rates based on validation loss.
        scheduler.step(current_loss)

        print(
            "Current learning rates: "
            f"Layer4={optimizer.param_groups[0]['lr']:.7f}, "
            f"Classifier={optimizer.param_groups[1]['lr']:.7f}"
        )

        # Save the best model based on validation macro recall.
        if (
            current_recall > best_macro_recall
            or (
                current_recall == best_macro_recall
                and current_loss < best_validation_loss
            )
        ):
            best_macro_recall = current_recall
            best_validation_loss = current_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            _save_checkpoint(
                model,
                checkpoint_path,
                stage=2,
                epoch=epoch + 1,
                validation_metrics=validation_metrics,
                class_names=class_names,
                config=config,
            )

            print("Best checkpoint updated.")

        # Early stopping monitors validation loss.
        min_delta = config.get("early_stopping_min_delta", 0.0001)

        if current_loss < stage2_best_loss - min_delta:
            stage2_best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= config["early_stopping_patience"]
        ):
            print("Early stopping triggered.")
            break

    # Restore the best model, rather than leaving the last epoch's weights.
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nTransfer learning completed.")
    print(f"Best validation macro recall: {best_macro_recall:.2f}%")
    print(f"Best checkpoint: {checkpoint_path}")

    return model, history, str(checkpoint_path)
