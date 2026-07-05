"""Reproducible train/validation/test partitions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from ..config import TrainingConfig
from .records import DeletionRecord


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    validation: list[int]
    test: list[int]
    seed: int

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SplitIndices:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


def stratified_holdout_split(
    records: Sequence[DeletionRecord],
    config: TrainingConfig,
) -> SplitIndices:
    labels = [record.label for record in records]
    all_indices = list(range(len(records)))
    holdout_fraction = config.validation_fraction + config.test_fraction
    train_indices, holdout_indices = train_test_split(
        all_indices,
        test_size=holdout_fraction,
        random_state=config.seed,
        stratify=labels,
    )
    holdout_labels = [labels[index] for index in holdout_indices]
    test_share = config.test_fraction / holdout_fraction
    validation_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=test_share,
        random_state=config.seed,
        stratify=holdout_labels,
    )
    return SplitIndices(
        train=list(train_indices),
        validation=list(validation_indices),
        test=list(test_indices),
        seed=config.seed,
    )


def select_records(
    records: Sequence[DeletionRecord],
    indices: Sequence[int],
) -> list[DeletionRecord]:
    return [records[index] for index in indices]


def apply_subset_indices(
    records: Sequence[DeletionRecord],
    path: str | Path | None,
) -> list[DeletionRecord]:
    if path is None:
        return list(records)
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    indices = payload["indices"] if isinstance(payload, dict) else payload
    return select_records(records, indices)
