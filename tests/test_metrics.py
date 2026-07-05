import pytest

from diff_lnr.evaluation.metrics import compute_binary_metrics


def test_metrics_include_imbalance_aware_scores():
    metrics = compute_binary_metrics(
        labels=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        pathogenic_probabilities=[0.1, 0.7, 0.8, 0.9],
    )
    assert metrics["mcc"] > 0
    # Ranking remains perfect even though the 0.5 threshold creates one FP.
    assert metrics["auroc"] == pytest.approx(1.0)
    assert "auprc" in metrics
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
