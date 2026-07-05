from types import SimpleNamespace

import torch
from torch import nn

from diff_lnr.config import ModelConfig
from diff_lnr.models.diff_lnr import DiffLNRClassifier
from diff_lnr.models.fine_tuning import configure_selective_fine_tuning


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embeddings = nn.Embedding(16, 4)
        self.encoder = TinyEncoder()
        self.pooler = nn.Linear(4, 4)

    def forward(self, input_ids, attention_mask):
        hidden = self.embeddings(input_ids)
        pooled = self.pooler(hidden[:, 0, :])
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=pooled,
        )


def test_forward_and_selective_fine_tuning_policy():
    config = ModelConfig(
        classifier_hidden_size=3,
        unfreeze_encoder_layers=[6, 11],
    )
    model = DiffLNRClassifier(TinyBackbone(), config)
    logits = model(
        ref_input_ids=torch.tensor([[1, 2, 3]]),
        ref_attention_mask=torch.tensor([[1, 1, 1]]),
        alt_input_ids=torch.tensor([[1, 4, 3]]),
        alt_attention_mask=torch.tensor([[1, 1, 1]]),
    )
    assert logits.shape == (1, 2)

    summary = configure_selective_fine_tuning(model, [6, 11])
    assert summary["trainable_parameters"] > 0
    assert model.backbone.embeddings.weight.requires_grad
    assert not model.backbone.encoder.layer[5].weight.requires_grad
    assert model.backbone.encoder.layer[6].weight.requires_grad
    assert not model.backbone.pooler.weight.requires_grad
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
