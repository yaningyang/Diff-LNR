"""Load the genomic backbone and enforce the N-token policy."""

from __future__ import annotations

from typing import Any

from transformers import AutoModel, AutoTokenizer, BertConfig

from ..config import ModelConfig


def load_backbone_and_tokenizer(
    config: ModelConfig,
    *,
    tokenizer_path: str | None = None,
) -> tuple[Any, Any]:
    """Load DNABERT-2 and, when requested, add a trainable N token."""

    backbone_config = BertConfig.from_pretrained(config.name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or config.name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    backbone = AutoModel.from_pretrained(
        config.name_or_path,
        trust_remote_code=config.trust_remote_code,
        config=backbone_config,
    )

    has_n_token = "N" in tokenizer.get_vocab()
    if not has_n_token and config.add_n_token:
        added = tokenizer.add_tokens(["N"])
        if added != 1:
            raise RuntimeError(f"expected to add one N token, tokenizer added {added}")
        backbone.resize_token_embeddings(len(tokenizer))
    elif not has_n_token and config.representation in {"lnr", "diff_lnr"}:
        raise ValueError(
            "LNR representations require token 'N'; set model.add_n_token=true "
            "or provide a tokenizer that already contains it"
        )
    if backbone.get_input_embeddings().num_embeddings != len(tokenizer):
        backbone.resize_token_embeddings(len(tokenizer))
    return backbone, tokenizer
