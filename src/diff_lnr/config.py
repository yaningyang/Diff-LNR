"""Typed configuration for Diff-LNR experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPRESENTATIONS = {
    "ldr",
    "lr",
    "lnr",
    "diff_lr",
    "diff_lnr",
    "d_only",
}


@dataclass
class DataConfig:
    context_file: Path
    subset_indices_file: Path | None = None
    max_ref_allele_length: int | None = 1000
    skip_uncertain_labels: bool = True
    skip_likely_labels: bool = False
    reverse_complement_average: bool = False


@dataclass
class ModelConfig:
    name_or_path: str = "zhihan1996/DNABERT-2-117M"
    trust_remote_code: bool = True
    add_n_token: bool = True
    representation: str = "diff_lnr"
    pooling: str = "pooler"
    classifier_hidden_size: int = 256
    classifier_dropout: float = 0.1
    unfreeze_encoder_layers: list[int] = field(default_factory=lambda: [6, 11])


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.05
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    batch_size_by_max_deletion: dict[int, int] = field(
        default_factory=lambda: {50: 32, 100: 32, 200: 16, 500: 8, 1000: 4}
    )
    evaluation_batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    save_checkpoint: bool = True


@dataclass
class RuntimeConfig:
    device: str = "auto"
    deterministic: bool = True
    tokenizer_parallelism: bool = False


@dataclass
class OutputConfig:
    root: Path = Path("runs")
    run_name: str | None = None
    save_predictions: bool = True


@dataclass
class ProjectConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    source_path: Path | None = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProjectConfig:
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        if "data" not in raw or "context_file" not in raw["data"]:
            raise ValueError("config must define data.context_file")

        base_dir = config_path.parent
        data_raw = dict(raw.get("data", {}))
        output_raw = dict(raw.get("output", {}))
        data_raw["context_file"] = _resolve_local_path(base_dir, data_raw["context_file"])
        subset = data_raw.get("subset_indices_file")
        data_raw["subset_indices_file"] = _resolve_local_path(base_dir, subset) if subset else None
        output_raw["root"] = _resolve_local_path(base_dir, output_raw.get("root", "runs"))

        training_raw = dict(raw.get("training", {}))
        batch_sizes = training_raw.get("batch_size_by_max_deletion")
        if batch_sizes is not None:
            training_raw["batch_size_by_max_deletion"] = {
                int(boundary): int(size) for boundary, size in batch_sizes.items()
            }

        config = cls(
            data=DataConfig(**data_raw),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**training_raw),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
            output=OutputConfig(**output_raw),
            source_path=config_path,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.data.skip_uncertain_labels:
            raise ValueError(
                "binary Diff-LNR training requires data.skip_uncertain_labels=true"
            )
        if self.model.representation not in REPRESENTATIONS:
            raise ValueError(f"model.representation must be one of {sorted(REPRESENTATIONS)}")
        if self.model.pooling not in {"pooler", "cls"}:
            raise ValueError("model.pooling must be 'pooler' or 'cls'")
        if len(self.model.unfreeze_encoder_layers) != 2:
            raise ValueError("model.unfreeze_encoder_layers must be [start, end]")
        first, last = self.model.unfreeze_encoder_layers
        if first < 0 or last < first:
            raise ValueError("unfreeze encoder layer range is invalid")
        if self.model.classifier_hidden_size <= 0:
            raise ValueError("model.classifier_hidden_size must be positive")
        if not 0 <= self.model.classifier_dropout < 1:
            raise ValueError("model.classifier_dropout must be in [0, 1)")

        train = self.training
        fractions = train.train_fraction + train.validation_fraction + train.test_fraction
        if abs(fractions - 1.0) > 1e-8:
            raise ValueError("train/validation/test fractions must sum to 1")
        if train.epochs <= 0 or train.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        if train.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if train.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if not train.batch_size_by_max_deletion:
            raise ValueError("batch_size_by_max_deletion cannot be empty")
        boundaries = list(train.batch_size_by_max_deletion)
        if boundaries != sorted(boundaries) or any(boundary <= 0 for boundary in boundaries):
            raise ValueError("batch-size boundaries must be sorted positive integers")
        if any(size <= 0 for size in train.batch_size_by_max_deletion.values()):
            raise ValueError("all batch sizes must be positive")
        if train.evaluation_batch_size <= 0 or train.num_workers < 0:
            raise ValueError("evaluation_batch_size must be positive and num_workers non-negative")

    def to_dict(self) -> dict[str, Any]:
        return _paths_to_strings(asdict(self))


def _resolve_local_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_paths_to_strings(item) for item in value]
    return value
