"""ClinVar deletion record parsing.

The parser preserves the sequence reconstruction used by the original
research code. The input context is the alternate allele window, and the
reference window is reconstructed by replacing ALT with REF at var_position.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LABEL_MAPPING = {
    "Benign": 0,
    "Likely_benign": 0,
    "Uncertain_significance": 2,
    "Likely_pathogenic": 1,
    "Pathogenic": 1,
    "Pathogenic/Likely_pathogenic": 1,
    "Benign/Likely_benign": 0,
    "Conflicting_interpretations_of_pathogenicity": 2,
    "unknown": 2,
}

LIKELY_LABELS = {
    "Likely_benign",
    "Likely_pathogenic",
    "Pathogenic/Likely_pathogenic",
    "Benign/Likely_benign",
}


@dataclass(frozen=True)
class DeletionRecord:
    """One processed deletion and its reconstructed sequence windows."""

    source_index: int
    chromosome: str
    position: int
    ref_allele: str
    alt_allele: str
    clinical_significance: str
    reference_sequence: str
    alternate_sequence: str
    deleted_sequence: str
    label: int
    deletion_start: int
    deletion_end: int

    @property
    def record_id(self) -> str:
        return f"{self.chromosome}:{self.position}:{self.ref_allele}>{self.alt_allele}"

    @property
    def deletion_length(self) -> int:
        return len(self.reference_sequence) - len(self.alternate_sequence)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["record_id"] = self.record_id
        result["deletion_length"] = self.deletion_length
        return result


@dataclass(frozen=True)
class LoadResult:
    records: list[DeletionRecord]
    statistics: dict[str, int]


def load_deletion_records(
    path: str | Path,
    *,
    skip_uncertain_labels: bool = True,
    skip_likely_labels: bool = False,
    max_ref_allele_length: int | None = 1000,
) -> LoadResult:
    """Load the seven-column processed ClinVar deletion TSV.

    The label behavior defaults to the supplied implementation: uncertain and
    conflicting records are excluded, while "likely" labels are retained.
    Set ``skip_likely_labels=true`` in the configuration to exclude them.
    """

    input_path = Path(path)
    records: list[DeletionRecord] = []
    statistics = {
        "rows_seen": 0,
        "rows_loaded": 0,
        "malformed_rows": 0,
        "unknown_labels": 0,
        "uncertain_labels": 0,
        "likely_labels": 0,
        "ref_too_long": 0,
        "invalid_positions": 0,
    }

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for source_index, row in enumerate(reader):
            statistics["rows_seen"] += 1
            if len(row) < 7:
                statistics["malformed_rows"] += 1
                continue

            chromosome, position_text, ref, alt, label_text, context, var_pos_text = row[:7]
            if label_text not in LABEL_MAPPING:
                statistics["unknown_labels"] += 1
                continue
            if skip_likely_labels and label_text in LIKELY_LABELS:
                statistics["likely_labels"] += 1
                continue

            label = LABEL_MAPPING[label_text]
            if skip_uncertain_labels and label == 2:
                statistics["uncertain_labels"] += 1
                continue
            if max_ref_allele_length is not None and len(ref) > max_ref_allele_length:
                statistics["ref_too_long"] += 1
                continue

            try:
                position = int(position_text)
                var_position = int(var_pos_text)
            except ValueError:
                statistics["invalid_positions"] += 1
                continue
            if var_position < 0 or var_position + len(alt) > len(context):
                statistics["invalid_positions"] += 1
                continue

            reference_sequence = context[:var_position] + ref + context[var_position + len(alt) :]
            record = DeletionRecord(
                source_index=source_index,
                chromosome=chromosome,
                position=position,
                ref_allele=ref,
                alt_allele=alt,
                clinical_significance=label_text,
                reference_sequence=reference_sequence.upper(),
                alternate_sequence=context.upper(),
                deleted_sequence=ref[len(alt) :].upper(),
                label=label,
                deletion_start=var_position,
                deletion_end=var_position + len(ref),
            )
            records.append(record)

    statistics["rows_loaded"] = len(records)
    return LoadResult(records=records, statistics=statistics)
