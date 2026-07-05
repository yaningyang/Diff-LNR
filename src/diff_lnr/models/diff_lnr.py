"""Diff-LNR classifier."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..config import ModelConfig
from ..layers import SharedSequenceEncoder


class DiffLNRClassifier(nn.Module):
    """Shared genomic encoder followed by the paper's 768→256→2 MLP."""

    def __init__(self, backbone: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = SharedSequenceEncoder(backbone, pooling=config.pooling)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, config.classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_size, 2),
        )

    @property
    def backbone(self) -> nn.Module:
        return self.encoder.backbone

    def forward(self, **batch: Any) -> torch.Tensor:
        encoder_inputs = {
            key: value
            for key, value in batch.items()
            if key.endswith("_input_ids") or key.endswith("_attention_mask")
        }
        embedding = self.encoder(**encoder_inputs)
        return self.classifier(embedding)
