from diff_lnr.data.records import DeletionRecord
from diff_lnr.data.sequence import build_aligned_alternate, build_sequence_inputs


def record() -> DeletionRecord:
    return DeletionRecord(
        source_index=0,
        chromosome="1",
        position=101,
        ref_allele="ATG",
        alt_allele="A",
        clinical_significance="Pathogenic",
        reference_sequence="CCCATGTTT",
        alternate_sequence="CCCATTT",
        deleted_sequence="TG",
        label=1,
        deletion_start=3,
        deletion_end=6,
    )


def test_aligned_alternate_preserves_reference_length_and_anchor():
    aligned = build_aligned_alternate(record())
    assert aligned == "CCCANNTTT"
    assert len(aligned) == len(record().reference_sequence)


def test_six_representations_map_to_expected_sequences():
    item = record()
    assert build_sequence_inputs(item, "ldr").single == "CCCATGTTT"
    assert build_sequence_inputs(item, "lr").single == "CCCATTT"
    assert build_sequence_inputs(item, "lnr").single == "CCCANNTTT"
    assert build_sequence_inputs(item, "d_only").single == "TG"
    assert build_sequence_inputs(item, "diff_lr").alternate == "CCCATTT"
    assert build_sequence_inputs(item, "diff_lnr").alternate == "CCCANNTTT"
