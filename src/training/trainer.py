"""Explicit training, validation, and epoch loops."""
import torch
from tqdm.auto import tqdm

from src.evaluation.metrics import classification_metrics


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0.0
    total_correct = 0
    all_labels = []
    all_predictions = []

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        labels = batch["label"].to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad()
        scores = model(images)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        predictions = scores.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    return classification_metrics(
        total_loss, total_correct, all_labels, all_predictions, num_classes,
    )


def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    all_labels = []
    all_predictions = []

    # Validation measures performance without calculating gradients or updating weights.
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            labels = batch["label"].to(device, non_blocking=device.type == "cuda")
            scores = model(images)
            loss = criterion(scores, labels)

            predictions = scores.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return classification_metrics(
        total_loss, total_correct, all_labels, all_predictions, num_classes,
    )


def fit(model, train_loader, validation_loader, criterion, optimizer, device, config):
    history = []
    scheduler = None
    if config.get("lr_scheduler", "none") == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.get("lr_step_size", 30),
            gamma=config.get("lr_gamma", 0.1),
        )

    for epoch in range(config["epochs"]):
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{config['epochs']} | Learning rate: {learning_rate:g}")
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config["num_classes"],
        )
        validation_metrics = validate(
            model, validation_loader, criterion, device, config["num_classes"],
        )

        # Keep measurements in memory only; no JSON files are written.
        history.append({
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
        })

        print(
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train accuracy: {train_metrics['accuracy']:.2f}% | "
            f"Val loss: {validation_metrics['loss']:.4f} | "
            f"Val accuracy: {validation_metrics['accuracy']:.2f}% | "
            f"Val macro recall: {validation_metrics['macro_recall']:.2f}%"
        )
        # The reduced rate applies to the next epoch, not this epoch's metrics.
        if scheduler is not None:
            scheduler.step()

    return history
