from diff_lnr.data.matching import match_deletion_lengths
from diff_lnr.data.records import DeletionRecord


def make_record(index: int, label: int, deletion_length: int) -> DeletionRecord:
    ref = "A" + "C" * deletion_length
    alt = "A"
    return DeletionRecord(
        source_index=index,
        chromosome="1",
        position=100 + index,
        ref_allele=ref,
        alt_allele=alt,
        clinical_significance="Pathogenic" if label else "Benign",
        reference_sequence="T" + ref + "G",
        alternate_sequence="T" + alt + "G",
        deleted_sequence="C" * deletion_length,
        label=label,
        deletion_start=1,
        deletion_end=1 + len(ref),
    )


def test_matching_is_balanced_and_reproducible():
    records = [
        make_record(0, 0, 2),
        make_record(1, 0, 5),
        make_record(2, 1, 1),
        make_record(3, 1, 3),
        make_record(4, 1, 5),
    ]
    first = match_deletion_lengths(records, max_difference=1, seed=42)
    second = match_deletion_lengths(records, max_difference=1, seed=42)
    assert first.indices == second.indices
    labels = [records[index].label for index in first.indices]
    assert labels.count(0) == labels.count(1) == 2
