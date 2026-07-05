"""Holdout training workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from ..data import stratified_holdout_split
from ..data.dataset import build_dataloaders
from ..data.splits import select_records
from ..models import (
    DiffLNRClassifier,
    configure_selective_fine_tuning,
    load_backbone_and_tokenizer,
)
from ..models.fine_tuning import trainable_parameter_names
from ..training import TrainingEngine, evaluate_loader
from ..utils.environment import collect_environment
from ..utils.io import write_json, write_yaml
from ..utils.logging import configure_logging
from ..utils.reproducibility import resolve_device, set_global_seed
from .common import class_counts, create_run_directory, load_configured_records


def run_training(
    config: ProjectConfig,
    *,
    verbose: bool = False,
) -> Path:
    os.environ["TOKENIZERS_PARALLELISM"] = str(config.runtime.tokenizer_parallelism).lower()
    set_global_seed(
        config.training.seed,
        deterministic=config.runtime.deterministic,
    )
    device = resolve_device(config.runtime.device)
    run_dir = create_run_directory(config)
    logger = configure_logging(run_dir, verbose=verbose)
    write_yaml(run_dir / "config.resolved.yaml", config.to_dict())
    write_json(run_dir / "environment.json", collect_environment())

    load_result = load_configured_records(config)
    records = load_result.records
    if not records:
        raise RuntimeError("no records remained after filtering")
    write_json(run_dir / "data_loading.json", load_result.statistics)
    logger.info("loaded data: %s", class_counts(records))

    splits = stratified_holdout_split(records, config.training)
    splits.save(run_dir / "splits.json")
    train_records = select_records(records, splits.train)
    validation_records = select_records(records, splits.validation)
    test_records = select_records(records, splits.test)
    write_json(
        run_dir / "split_summary.json",
        {
            "train": class_counts(train_records),
            "validation": class_counts(validation_records),
            "test": class_counts(test_records),
        },
    )

    backbone, tokenizer = load_backbone_and_tokenizer(config.model)
    tokenizer.save_pretrained(run_dir / "tokenizer")
    model = DiffLNRClassifier(backbone, config.model)
    parameter_summary = configure_selective_fine_tuning(
        model,
        config.model.unfreeze_encoder_layers,
    )
    write_json(
        run_dir / "trainable_parameters.json",
        {
            **parameter_summary,
            "parameter_names": trainable_parameter_names(model),
        },
    )
    logger.info(
        "trainable parameters: %d / %d (%.2f%%)",
        parameter_summary["trainable_parameters"],
        parameter_summary["total_parameters"],
        100 * parameter_summary["trainable_fraction"],
    )

    train_loader, validation_loader, test_loader = build_dataloaders(
        train_records,
        validation_records,
        test_records,
        tokenizer,
        config,
    )
    engine = TrainingEngine(model, config, device, run_dir, logger)
    best_metrics = engine.fit(train_loader, validation_loader)
    test_metrics, _ = evaluate_loader(
        model,
        test_loader,
        device,
        output_dir=run_dir / "test",
        save_predictions=config.output.save_predictions,
    )
    write_json(
        run_dir / "summary.json",
        {
            "status": "complete",
            "best_validation": best_metrics,
            "test": test_metrics,
            "run_directory": str(run_dir),
        },
    )
    logger.info(
        "test: F1=%.4f AUROC=%.4f MCC=%.4f",
        test_metrics["f1"],
        test_metrics["auroc"],
        test_metrics["mcc"],
    )
    return run_dir


def validate_inputs(config: ProjectConfig) -> dict[str, Any]:
    """Validate configuration and processed data without loading the model."""

    if not config.data.context_file.exists():
        raise FileNotFoundError(config.data.context_file)
    result = load_configured_records(config)
    if not result.records:
        raise RuntimeError("no records remained after filtering")
    return {
        "statistics": result.statistics,
        "class_counts": class_counts(result.records),
        "first_record": result.records[0].to_dict(),
    }
