"""Shared workflow assembly."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from ..data.records import DeletionRecord, LoadResult, load_deletion_records
from ..data.splits import apply_subset_indices


def load_configured_records(config: ProjectConfig) -> LoadResult:
    result = load_deletion_records(
        config.data.context_file,
        skip_uncertain_labels=config.data.skip_uncertain_labels,
        skip_likely_labels=config.data.skip_likely_labels,
        max_ref_allele_length=config.data.max_ref_allele_length,
    )
    selected = apply_subset_indices(
        result.records,
        config.data.subset_indices_file,
    )
    statistics = dict(result.statistics)
    statistics["rows_after_subset"] = len(selected)
    return LoadResult(records=selected, statistics=statistics)


def create_run_directory(config: ProjectConfig, *, prefix: str = "run") -> Path:
    name = config.output.run_name
    if not name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{prefix}_{config.model.representation}_{timestamp}"
    output_dir = config.output.root / name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def class_counts(records: list[DeletionRecord]) -> dict[str, int]:
    benign = sum(record.label == 0 for record in records)
    pathogenic = sum(record.label == 1 for record in records)
    return {
        "total": len(records),
        "benign": benign,
        "pathogenic": pathogenic,
    }


def without_confusion_matrix(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "confusion_matrix"}
