"""Deletion sequence representations used in the manuscript."""

from __future__ import annotations

from dataclasses import dataclass

from .records import DeletionRecord

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@dataclass(frozen=True)
class SequenceInputs:
    """Raw sequence inputs for one model example."""

    reference: str | None = None
    alternate: str | None = None
    single: str | None = None


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def build_aligned_alternate(record: DeletionRecord) -> str:
    """Create L + N^k + R with exact base-space length preservation.

    The N run starts immediately after the retained ALT anchor, matching the
    original implementation and standard VCF deletion representation.
    """

    deletion_length = record.deletion_length
    insert_at = record.deletion_start + len(record.alt_allele)
    aligned = (
        record.alternate_sequence[:insert_at]
        + "N" * deletion_length
        + record.alternate_sequence[insert_at:]
    )
    if len(aligned) != len(record.reference_sequence):
        raise ValueError(
            f"alignment failed for {record.record_id}: "
            f"reference={len(record.reference_sequence)}, aligned={len(aligned)}"
        )
    return aligned


def build_sequence_inputs(record: DeletionRecord, representation: str) -> SequenceInputs:
    """Map the six paper representations to raw sequence inputs."""

    if representation == "ldr":
        return SequenceInputs(single=record.reference_sequence)
    if representation == "lr":
        return SequenceInputs(single=record.alternate_sequence)
    if representation == "lnr":
        return SequenceInputs(single=build_aligned_alternate(record))
    if representation == "diff_lr":
        return SequenceInputs(
            reference=record.reference_sequence,
            alternate=record.alternate_sequence,
        )
    if representation == "diff_lnr":
        return SequenceInputs(
            reference=record.reference_sequence,
            alternate=build_aligned_alternate(record),
        )
    if representation == "d_only":
        return SequenceInputs(single=record.deleted_sequence)
    raise ValueError(f"unknown representation: {representation!r}")
