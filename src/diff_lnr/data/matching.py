"""Greedy one-to-one deletion-length matching used by the control analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .records import DeletionRecord


@dataclass(frozen=True)
class MatchingResult:
    indices: list[int]
    statistics: dict[str, object]


def match_deletion_lengths(
    records: Sequence[DeletionRecord],
    *,
    max_difference: int = 5,
    seed: int = 42,
) -> MatchingResult:
    """Match the minority class one-to-one to unused majority examples."""

    class_indices = {
        0: [index for index, record in enumerate(records) if record.label == 0],
        1: [index for index, record in enumerate(records) if record.label == 1],
    }
    if len(class_indices[0]) <= len(class_indices[1]):
        anchor_label, pool_label = 0, 1
    else:
        anchor_label, pool_label = 1, 0
    anchors = class_indices[anchor_label]
    pool = sorted(
        class_indices[pool_label],
        key=lambda index: records[index].deletion_length,
    )
    pool_lengths = np.asarray([records[index].deletion_length for index in pool], dtype=int)

    rng = np.random.default_rng(seed)
    used_pool_positions: set[int] = set()
    matched_anchors: list[int] = []
    matched_pool: list[int] = []
    for anchor_position in rng.permutation(len(anchors)):
        anchor_index = anchors[int(anchor_position)]
        target = records[anchor_index].deletion_length
        insertion = int(np.searchsorted(pool_lengths, target))
        best: tuple[int, int] | None = None
        offset = 0
        while True:
            candidates = (insertion - 1 - offset, insertion + offset)
            valid = [position for position in candidates if 0 <= position < len(pool)]
            if not valid:
                break
            for position in valid:
                difference = abs(int(pool_lengths[position]) - target)
                if difference <= max_difference and position not in used_pool_positions:
                    if best is None or difference < best[0]:
                        best = (difference, position)
            if best is not None:
                break
            nearest_difference = min(
                abs(int(pool_lengths[position]) - target) for position in valid
            )
            if nearest_difference > max_difference:
                break
            offset += 1
        if best is None:
            continue
        matched_anchors.append(anchor_index)
        matched_pool.append(pool[best[1]])
        used_pool_positions.add(best[1])

    matched_indices = matched_anchors + matched_pool
    rng.shuffle(matched_indices)
    statistics = _matching_statistics(
        records,
        class_indices,
        matched_indices,
        anchor_label,
        pool_label,
        max_difference,
    )
    return MatchingResult(indices=matched_indices, statistics=statistics)


def _matching_statistics(
    records: Sequence[DeletionRecord],
    class_indices: dict[int, list[int]],
    matched_indices: list[int],
    anchor_label: int,
    pool_label: int,
    max_difference: int,
) -> dict[str, object]:
    original = {
        label: [records[index].deletion_length for index in indices]
        for label, indices in class_indices.items()
    }
    matched = {
        label: [
            records[index].deletion_length
            for index in matched_indices
            if records[index].label == label
        ]
        for label in (0, 1)
    }
    return {
        "original_benign_count": len(class_indices[0]),
        "original_pathogenic_count": len(class_indices[1]),
        "matched_benign_count": len(matched[0]),
        "matched_pathogenic_count": len(matched[1]),
        "anchor_label": anchor_label,
        "pool_label": pool_label,
        "max_difference": max_difference,
        "total_matched": len(matched_indices),
        "original_mean_difference": float(np.mean(original[1]) - np.mean(original[0])),
        "matched_mean_difference": float(np.mean(matched[1]) - np.mean(matched[0])),
    }
