"""Classification metrics; accuracy and recall are returned as percentages."""
from sklearn.metrics import recall_score


def classification_metrics(total_loss, total_correct, labels, predictions, num_classes):
    return {
        "loss": total_loss / len(labels),
        "accuracy": 100 * total_correct / len(labels),
        "macro_recall": 100 * recall_score(
            labels, predictions, labels=list(range(num_classes)),
            average="macro", zero_division=0,
        ),
    }
