"""Deterministic deletion-length bucket sampler."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group examples by deletion length and use bucket-specific batch sizes."""

    def __init__(
        self,
        deletion_lengths: Sequence[int],
        batch_size_by_boundary: dict[int, int],
        *,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        self.deletion_lengths = list(deletion_lengths)
        self.batch_size_by_boundary = dict(sorted(batch_size_by_boundary.items()))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.buckets = self._assign_buckets()

    def _assign_buckets(self) -> dict[int, list[int]]:
        buckets: dict[int, list[int]] = defaultdict(list)
        largest = max(self.batch_size_by_boundary)
        for index, length in enumerate(self.deletion_lengths):
            boundary = next(
                (value for value in self.batch_size_by_boundary if length <= value),
                None,
            )
            if boundary is None:
                raise ValueError(
                    f"deletion length {length} exceeds largest bucket boundary {largest}"
                )
            buckets[boundary].append(index)
        return dict(buckets)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        for boundary, source_indices in self.buckets.items():
            indices = list(source_indices)
            if self.shuffle:
                rng.shuffle(indices)
            batch_size = self.batch_size_by_boundary[boundary]
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                if len(batch) == batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for boundary, indices in self.buckets.items():
            batch_size = self.batch_size_by_boundary[boundary]
            total += (
                len(indices) // batch_size
                if self.drop_last
                else math.ceil(len(indices) / batch_size)
            )
        return total
