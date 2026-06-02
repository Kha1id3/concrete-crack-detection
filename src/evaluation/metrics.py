import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute common metrics from true labels and predicted probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_prob = np.asarray(y_pred_prob).reshape(-1)

    # Convert probabilities such as 0.83 into binary labels such as 1.
    y_pred = (y_pred_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
