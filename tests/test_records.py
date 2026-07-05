from pathlib import Path

from diff_lnr.data.records import load_deletion_records


def test_loader_reconstructs_reference_and_maps_labels(tmp_path: Path):
    source = tmp_path / "records.tsv"
    source.write_text(
        "CHROM\tPOS\tREF\tALT\tlabels\tcontext_sequence\tvar_position\n"
        "1\t101\tATG\tA\tPathogenic\tCCCATTT\t3\n"
        "1\t201\tCT\tC\tUncertain_significance\tAAACGG\t3\n",
        encoding="utf-8",
    )
    result = load_deletion_records(source)
    assert result.statistics["rows_loaded"] == 1
    item = result.records[0]
    assert item.reference_sequence == "CCCATGTTT"
    assert item.deleted_sequence == "TG"
    assert item.deletion_length == 2
    assert item.label == 1
