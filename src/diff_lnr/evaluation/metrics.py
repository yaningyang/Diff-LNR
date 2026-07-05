"""Binary classification metrics for imbalanced deletion data."""

from __future__ import annotations

import math
from collections.abc import Sequence

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    pathogenic_probabilities: Sequence[float],
) -> dict[str, float | list[list[int]]]:
    """Compute threshold and ranking metrics from one evaluation pass."""

    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="binary", zero_division=0)),
        "f1": float(f1_score(labels, predictions, average="binary", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    if len(set(labels)) < 2:
        metrics["auroc"] = math.nan
        metrics["auprc"] = math.nan
    else:
        metrics["auroc"] = float(roc_auc_score(labels, pathogenic_probabilities))
        metrics["auprc"] = float(average_precision_score(labels, pathogenic_probabilities))
    return metrics
