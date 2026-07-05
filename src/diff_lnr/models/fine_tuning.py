"""Selective fine-tuning policy used by the paper implementation."""

from __future__ import annotations

from torch import nn

from .diff_lnr import DiffLNRClassifier


def configure_selective_fine_tuning(
    model: DiffLNRClassifier,
    layer_range: list[int],
) -> dict[str, int | float]:
    """Train embeddings, encoder layers [start, end], and the classifier.

    All other parameters, including the lower encoder blocks and pooler, remain
    frozen. This exactly captures the supplied implementation's fine-tuning
    boundary.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    backbone = model.backbone
    embeddings = _require_module(backbone, "embeddings")
    encoder = _require_module(backbone, "encoder")
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise AttributeError("backbone.encoder.layer is required by this fine-tuning policy")

    for parameter in embeddings.parameters():
        parameter.requires_grad = True

    start, end = layer_range
    if end >= len(layers):
        raise ValueError(
            f"requested encoder layers {start}--{end}, but backbone has {len(layers)} layers"
        )
    for layer_index in range(start, end + 1):
        for parameter in layers[layer_index].parameters():
            parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_fraction": trainable / max(total, 1),
    }


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _require_module(module: nn.Module, name: str) -> nn.Module:
    child = getattr(module, name, None)
    if child is None:
        raise AttributeError(f"backbone.{name} is required by this implementation")
    return child
