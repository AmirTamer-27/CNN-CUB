"""Explicit training, validation, and epoch loops."""
import torch
from tqdm.auto import tqdm

from src.evaluation.metrics import classification_metrics
from src.training.losses import combined_loss


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes,
                    cosine_weight=0.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    all_labels = []
    all_predictions = []
    total_objective = 0.0
    total_cosine = 0.0

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        labels = batch["label"].to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad()
        scores = model(images)
        if cosine_weight > 0:
            loss, ce_loss, cosine_loss = combined_loss(scores, labels, cosine_weight)
            total_objective += loss.item() * labels.size(0)
            total_cosine += cosine_loss.item() * labels.size(0)
        else:
            ce_loss = loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        predictions = scores.argmax(dim=1)
        # Keep "loss" as CE for comparison with earlier experiments.
        total_loss += ce_loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    metrics = classification_metrics(
        total_loss, total_correct, all_labels, all_predictions, num_classes,
    )
    if cosine_weight > 0:
        metrics["cosine_loss"] = total_cosine / len(all_labels)
        metrics["total_loss"] = total_objective / len(all_labels)
    return metrics


def validate(model, loader, criterion, device, num_classes, cosine_weight=0.0):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    all_labels = []
    all_predictions = []
    total_objective = 0.0
    total_cosine = 0.0

    # Validation measures performance without calculating gradients or updating weights.
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            labels = batch["label"].to(device, non_blocking=device.type == "cuda")
            scores = model(images)
            if cosine_weight > 0:
                objective, loss, cosine_loss = combined_loss(scores, labels, cosine_weight)
                total_objective += objective.item() * labels.size(0)
                total_cosine += cosine_loss.item() * labels.size(0)
            else:
                loss = criterion(scores, labels)

            predictions = scores.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    metrics = classification_metrics(
        total_loss, total_correct, all_labels, all_predictions, num_classes,
    )
    if cosine_weight > 0:
        metrics["cosine_loss"] = total_cosine / len(all_labels)
        metrics["total_loss"] = total_objective / len(all_labels)
    return metrics


def fit(model, train_loader, validation_loader, criterion, optimizer, device, config):
    history = []
    cosine_weight = (
        config.get("cosine_weight", 1.0)
        if config.get("loss", "cross_entropy") == "combined" else 0.0
    )
    scheduler = None
    if config.get("lr_scheduler", "none") == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.get("lr_step_size", 30),
            gamma=config.get("lr_gamma", 0.1),
        )
    elif config.get("lr_scheduler", "none") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("lr_plateau_factor", 0.5),
            patience=config.get("lr_plateau_patience", 5),
            threshold=config.get("lr_plateau_threshold", 0.001),
            threshold_mode="abs",
            min_lr=config.get("lr_plateau_min_lr", 0.000001),
        )

    for epoch in range(config["epochs"]):
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{config['epochs']} | Learning rate: {learning_rate:g}")
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config["num_classes"],
            cosine_weight=cosine_weight,
        )
        validation_metrics = validate(
            model, validation_loader, criterion, device, config["num_classes"],
            cosine_weight=cosine_weight,
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
        if cosine_weight > 0:
            print(
                f"Train cosine loss: {train_metrics['cosine_loss']:.4f} | "
                f"Train total loss: {train_metrics['total_loss']:.4f} | "
                f"Val cosine loss: {validation_metrics['cosine_loss']:.4f} | "
                f"Val total loss: {validation_metrics['total_loss']:.4f}"
            )
        # The reduced rate applies to the next epoch, not this epoch's metrics.
        if scheduler is not None:
            if config.get("lr_scheduler") == "plateau":
                scheduler.step(validation_metrics["loss"])
            else:
                scheduler.step()
            next_rate = optimizer.param_groups[0]["lr"]
            if next_rate < learning_rate:
                print(f"Learning rate reduced: {learning_rate:g} -> {next_rate:g} (next epoch)")

    return history
