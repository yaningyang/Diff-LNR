"""PyTorch dataset and dataloader construction."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ..config import ProjectConfig
from .records import DeletionRecord
from .samplers import LengthBucketBatchSampler
from .sequence import build_sequence_inputs, reverse_complement


class TokenizedDeletionDataset(Dataset):
    """Tokenize one of the six deletion representations on demand."""

    def __init__(
        self,
        records: Sequence[DeletionRecord],
        tokenizer: Any,
        *,
        representation: str,
        reverse_complement_average: bool = False,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.representation = representation
        self.reverse_complement_average = reverse_complement_average
        self.deletion_lengths = [record.deletion_length for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )
        return (
            encoded["input_ids"].squeeze(0),
            encoded["attention_mask"].squeeze(0),
        )

    def _add_sequence(
        self,
        sample: dict[str, Any],
        prefix: str,
        sequence: str,
    ) -> None:
        input_ids, attention_mask = self._encode(sequence)
        sample[f"{prefix}_input_ids"] = input_ids
        sample[f"{prefix}_attention_mask"] = attention_mask
        if self.reverse_complement_average:
            rc_ids, rc_mask = self._encode(reverse_complement(sequence))
            sample[f"{prefix}_rc_input_ids"] = rc_ids
            sample[f"{prefix}_rc_attention_mask"] = rc_mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        inputs = build_sequence_inputs(record, self.representation)
        sample: dict[str, Any] = {
            "label": torch.tensor(record.label, dtype=torch.long),
            "index": index,
            "source_index": record.source_index,
            "record_id": record.record_id,
        }
        if inputs.single is not None:
            self._add_sequence(sample, "single", inputs.single)
        else:
            if inputs.reference is None or inputs.alternate is None:
                raise RuntimeError("paired representation did not provide both sequences")
            self._add_sequence(sample, "ref", inputs.reference)
            self._add_sequence(sample, "alt", inputs.alternate)
        return sample


def collate_deletions(
    batch: list[dict[str, Any]],
    *,
    padding_token_id: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in batch[0]:
        if key.endswith("_input_ids"):
            result[key] = pad_sequence(
                [item[key] for item in batch],
                batch_first=True,
                padding_value=padding_token_id,
            )
        elif key.endswith("_attention_mask"):
            result[key] = pad_sequence(
                [item[key] for item in batch],
                batch_first=True,
                padding_value=0,
            )
    result["label"] = torch.stack([item["label"] for item in batch])
    result["index"] = torch.tensor([item["index"] for item in batch], dtype=torch.long)
    result["source_index"] = torch.tensor(
        [item["source_index"] for item in batch], dtype=torch.long
    )
    result["record_id"] = [item["record_id"] for item in batch]
    return result


def build_dataloaders(
    train_records: Sequence[DeletionRecord],
    validation_records: Sequence[DeletionRecord],
    test_records: Sequence[DeletionRecord],
    tokenizer: Any,
    config: ProjectConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_kwargs = {
        "representation": config.model.representation,
        "reverse_complement_average": config.data.reverse_complement_average,
    }
    train_dataset = TokenizedDeletionDataset(train_records, tokenizer, **dataset_kwargs)
    validation_dataset = TokenizedDeletionDataset(validation_records, tokenizer, **dataset_kwargs)
    test_dataset = TokenizedDeletionDataset(test_records, tokenizer, **dataset_kwargs)

    padding_token_id = tokenizer.pad_token_id
    if padding_token_id is None:
        padding_token_id = 0
    collate = partial(collate_deletions, padding_token_id=padding_token_id)
    sampler = LengthBucketBatchSampler(
        train_dataset.deletion_lengths,
        config.training.batch_size_by_max_deletion,
        shuffle=True,
        drop_last=False,
        seed=config.training.seed,
    )
    common = {
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
        "collate_fn": collate,
    }
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, **common)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader, test_loader
