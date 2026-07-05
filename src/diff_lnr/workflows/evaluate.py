"""Checkpoint evaluation workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import ProjectConfig
from ..data.dataset import TokenizedDeletionDataset, collate_deletions
from ..data.splits import SplitIndices, select_records
from ..models import DiffLNRClassifier, load_backbone_and_tokenizer
from ..training import evaluate_loader
from ..utils.logging import configure_logging
from ..utils.reproducibility import resolve_device, set_global_seed
from .common import load_configured_records


def load_checkpoint_model(
    config: ProjectConfig,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[DiffLNRClassifier, object, Path]:
    checkpoint = Path(checkpoint_path).resolve()
    run_dir = checkpoint.parent.parent
    tokenizer_directory = run_dir / "tokenizer"
    backbone, tokenizer = load_backbone_and_tokenizer(
        config.model,
        tokenizer_path=str(tokenizer_directory) if tokenizer_directory.exists() else None,
    )
    model = DiffLNRClassifier(backbone, config.model)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, tokenizer, run_dir


def make_evaluation_loader(
    records: list,
    tokenizer: object,
    config: ProjectConfig,
) -> DataLoader:
    dataset = TokenizedDeletionDataset(
        records,
        tokenizer,
        representation=config.model.representation,
        reverse_complement_average=config.data.reverse_complement_average,
    )
    padding_token_id = tokenizer.pad_token_id or 0
    from functools import partial

    return DataLoader(
        dataset,
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=partial(
            collate_deletions,
            padding_token_id=padding_token_id,
        ),
    )


def run_evaluation(
    config: ProjectConfig,
    *,
    checkpoint_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    run_dir = checkpoint.parent.parent
    evaluation_dir = (
        Path(output_dir).resolve()
        if output_dir
        else run_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(evaluation_dir)
    set_global_seed(config.training.seed, deterministic=config.runtime.deterministic)
    device = resolve_device(config.runtime.device)

    load_result = load_configured_records(config)
    splits = SplitIndices.load(split_path)
    test_records = select_records(load_result.records, splits.test)
    model, tokenizer, _ = load_checkpoint_model(config, checkpoint, device)
    loader = make_evaluation_loader(test_records, tokenizer, config)
    metrics, _ = evaluate_loader(
        model,
        loader,
        device,
        output_dir=evaluation_dir,
        save_predictions=config.output.save_predictions,
    )
    logger.info("evaluation metrics: %s", metrics)
    return evaluation_dir
