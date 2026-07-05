"""Stratified cross-validation workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..config import ProjectConfig
from ..data.dataset import build_dataloaders
from ..data.splits import select_records
from ..models import (
    DiffLNRClassifier,
    configure_selective_fine_tuning,
    load_backbone_and_tokenizer,
)
from ..training import TrainingEngine, evaluate_loader
from ..utils.environment import collect_environment
from ..utils.io import write_json, write_yaml
from ..utils.logging import configure_logging
from ..utils.reproducibility import resolve_device, set_global_seed
from .common import class_counts, create_run_directory, load_configured_records


def run_cross_validation(
    config: ProjectConfig,
    *,
    folds: int = 5,
) -> Path:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    os.environ["TOKENIZERS_PARALLELISM"] = str(config.runtime.tokenizer_parallelism).lower()
    run_dir = create_run_directory(config, prefix=f"cv{folds}")
    logger = configure_logging(run_dir)
    write_yaml(run_dir / "config.resolved.yaml", config.to_dict())
    write_json(run_dir / "environment.json", collect_environment())
    load_result = load_configured_records(config)
    records = load_result.records
    labels = np.asarray([record.label for record in records])
    write_json(run_dir / "data_loading.json", load_result.statistics)
    logger.info("loaded data: %s", class_counts(records))

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config.training.seed,
    )
    fold_results: list[dict[str, Any]] = []
    for fold_index, (train_validation_indices, test_indices) in enumerate(
        splitter.split(np.arange(len(records)), labels),
        start=1,
    ):
        fold_seed = config.training.seed + fold_index
        set_global_seed(fold_seed, deterministic=config.runtime.deterministic)
        inner_labels = labels[train_validation_indices]
        validation_share = config.training.validation_fraction / (
            config.training.train_fraction + config.training.validation_fraction
        )
        train_indices, validation_indices = train_test_split(
            train_validation_indices,
            test_size=validation_share,
            random_state=fold_seed,
            stratify=inner_labels,
        )
        fold_dir = run_dir / f"fold_{fold_index:02d}"
        fold_dir.mkdir(parents=True)
        write_json(
            fold_dir / "splits.json",
            {
                "train": train_indices.tolist(),
                "validation": validation_indices.tolist(),
                "test": test_indices.tolist(),
                "seed": fold_seed,
            },
        )
        train_records = select_records(records, train_indices)
        validation_records = select_records(records, validation_indices)
        test_records = select_records(records, test_indices)
        write_json(
            fold_dir / "split_summary.json",
            {
                "train": class_counts(train_records),
                "validation": class_counts(validation_records),
                "test": class_counts(test_records),
            },
        )

        backbone, tokenizer = load_backbone_and_tokenizer(config.model)
        if fold_index == 1:
            tokenizer.save_pretrained(run_dir / "tokenizer")
        model = DiffLNRClassifier(backbone, config.model)
        parameter_summary = configure_selective_fine_tuning(
            model,
            config.model.unfreeze_encoder_layers,
        )
        write_json(fold_dir / "trainable_parameters.json", parameter_summary)
        train_loader, validation_loader, test_loader = build_dataloaders(
            train_records,
            validation_records,
            test_records,
            tokenizer,
            config,
        )
        device = resolve_device(config.runtime.device)
        engine = TrainingEngine(model, config, device, fold_dir, logger)
        best_metrics = engine.fit(train_loader, validation_loader)
        test_metrics, _ = evaluate_loader(
            model,
            test_loader,
            device,
            output_dir=fold_dir / "test",
            save_predictions=config.output.save_predictions,
        )
        fold_result = {
            "fold": fold_index,
            "best_validation": best_metrics,
            "test": test_metrics,
        }
        fold_results.append(fold_result)
        write_json(fold_dir / "result.json", fold_result)
        logger.info(
            "fold=%d test_f1=%.4f test_auroc=%.4f test_mcc=%.4f",
            fold_index,
            test_metrics["f1"],
            test_metrics["auroc"],
            test_metrics["mcc"],
        )

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "mcc",
        "loss",
    ]
    summary: dict[str, Any] = {"folds": folds, "fold_results": fold_results}
    for metric in metric_names:
        values = np.asarray([result["test"][metric] for result in fold_results], dtype=float)
        summary[metric] = {
            "values": values.tolist(),
            "mean": float(np.nanmean(values)),
            "standard_deviation": float(np.nanstd(values, ddof=1)),
        }
    write_json(run_dir / "cv_summary.json", summary)
    logger.info("cross-validation complete: %s", run_dir)
    return run_dir
