"""Shared DNABERT-2 encoding and reference-minus-alternate contrast."""

from __future__ import annotations

import torch
from torch import nn


class SharedSequenceEncoder(nn.Module):
    """Encode single sequences or a shared-weight sequence pair."""

    def __init__(self, backbone: nn.Module, pooling: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.output_dim = int(backbone.config.hidden_size)

    def pooled_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if self.pooling == "cls":
            return outputs.last_hidden_state[:, 0, :]
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is None:
            try:
                pooler_output = outputs[1]
            except (IndexError, TypeError) as error:
                raise RuntimeError(
                    "pooling='pooler' requested, but backbone returned no pooler output"
                ) from error
        return pooler_output

    def strand_average(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        reverse_input_ids: torch.Tensor | None,
        reverse_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        forward = self.pooled_embedding(input_ids, attention_mask)
        if reverse_input_ids is None:
            return forward
        reverse = self.pooled_embedding(reverse_input_ids, reverse_attention_mask)
        return 0.5 * (forward + reverse)

    def forward(
        self,
        *,
        ref_input_ids: torch.Tensor | None = None,
        ref_attention_mask: torch.Tensor | None = None,
        alt_input_ids: torch.Tensor | None = None,
        alt_attention_mask: torch.Tensor | None = None,
        single_input_ids: torch.Tensor | None = None,
        single_attention_mask: torch.Tensor | None = None,
        ref_rc_input_ids: torch.Tensor | None = None,
        ref_rc_attention_mask: torch.Tensor | None = None,
        alt_rc_input_ids: torch.Tensor | None = None,
        alt_rc_attention_mask: torch.Tensor | None = None,
        single_rc_input_ids: torch.Tensor | None = None,
        single_rc_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if single_input_ids is not None:
            if single_attention_mask is None:
                raise ValueError("single_attention_mask is required")
            return self.strand_average(
                single_input_ids,
                single_attention_mask,
                single_rc_input_ids,
                single_rc_attention_mask,
            )
        if (
            ref_input_ids is None
            or ref_attention_mask is None
            or alt_input_ids is None
            or alt_attention_mask is None
        ):
            raise ValueError("paired input requires ref_* and alt_* tensors")
        reference = self.strand_average(
            ref_input_ids,
            ref_attention_mask,
            ref_rc_input_ids,
            ref_rc_attention_mask,
        )
        alternate = self.strand_average(
            alt_input_ids,
            alt_attention_mask,
            alt_rc_input_ids,
            alt_rc_attention_mask,
        )
        return reference - alternate
