"""Download a saved MLflow state dict and evaluate the official CUB test split."""
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report
import torch
from torch import nn
from tqdm.auto import tqdm

from src.evaluation.metrics import classification_metrics


TRACKING_URI = "https://dagshub.com/AmirTamer-27/CNN-CUD.mlflow"


def download_run_weights(config, run_id, artifact_path=None):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    run = client.get_run(run_id)
    effective = dict(config)
    # Training-only settings do not affect inference. Restore the input/model
    # settings so a search trial can be evaluated using its original base YAML.
    converters = {"model": str, "image_size": int, "num_classes": int,
                  "dropout": float, "dropout2d": float}
    for key, convert in converters.items():
        if key in run.data.params:
            effective[key] = convert(run.data.params[key])
    if "use_bounding_boxes" in run.data.params:
        value = run.data.params["use_bounding_boxes"].lower()
        if value not in ("true", "false"):
            raise ValueError("Invalid use_bounding_boxes parameter in source run.")
        effective["use_bounding_boxes"] = value == "true"
    if effective["model"] == "resnext101":
        raise ValueError("test currently supports scratch CNNs, not the separate ResNeXt pipeline.")

    if artifact_path is None:
        candidates = [entry.path for entry in client.list_artifacts(run_id, "weights")
                      if not entry.is_dir and entry.path.endswith(".pth")]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one .pth file under weights/, found {len(candidates)}. "
                "Use --artifact-path weights/EXACT_FILENAME.pth to select one."
            )
        artifact_path = candidates[0]
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path,
        dst_path=str(Path("outputs/downloaded_weights") / run_id),
    )
    if not Path(path).is_file():
        raise ValueError("The selected artifact must be a weights file, not a directory.")
    print(f"Source run: {run_id}\nWeights: {path}")
    print(f"Test input: {effective['image_size']}px; bounding boxes: "
          f"{effective.get('use_bounding_boxes', False)}")
    return effective, path, run.info.experiment_id, artifact_path


def evaluate_test(model, loader, class_names, device):
    model.eval()  # Disable dropout and freeze BatchNorm statistics.
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    labels = []
    predictions = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Test"):
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            targets = batch["label"].to(device, non_blocking=device.type == "cuda")
            scores = model(images)
            predicted = scores.argmax(dim=1)
            total_loss += criterion(scores, targets).item() * targets.size(0)
            total_correct += (predicted == targets).sum().item()
            labels.extend(targets.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
    if not labels:
        raise ValueError("Test split contains no samples.")
    metrics = classification_metrics(
        total_loss, total_correct, labels, predictions, len(class_names),
    )
    report = classification_report(
        labels, predictions, labels=list(range(len(class_names))),
        target_names=class_names, digits=4, zero_division=0,
    )
    print(f"Test loss: {metrics['loss']:.4f} | Test accuracy: {metrics['accuracy']:.2f}% | "
          f"Test macro recall: {metrics['macro_recall']:.2f}%\n\n{report}")
    return metrics, report, len(labels)


def log_test_results(metrics, report, count, config, source_run_id,
                     experiment_id, artifact_path):
    # A separate run preserves the completed training run and its epoch curves.
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"test_{source_run_id[:8]}"):
        mlflow.set_tags({"evaluation_split": "test", "source_run_id": source_run_id})
        mlflow.log_params({
            "weights_artifact": artifact_path,
            "model": config["model"], "image_size": config["image_size"],
            "use_bounding_boxes": config.get("use_bounding_boxes", False),
            "bbox_annotation_dir": config.get("bbox_annotation_dir", "data/CUB_200_2011"),
            "test_dataset": "Multimodal-Fatima/CUB_test", "test_samples": count,
            "batch_size": config["batch_size"],
        })
        mlflow.log_metrics({f"test_{name}": value for name, value in metrics.items()})
        mlflow.log_text(report, "test/classification_report.txt")
