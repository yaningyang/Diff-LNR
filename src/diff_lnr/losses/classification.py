"""Classification objective."""

from torch import nn


def build_classification_loss() -> nn.Module:
    """Return the two-logit cross-entropy objective used in the paper."""

    return nn.CrossEntropyLoss()
