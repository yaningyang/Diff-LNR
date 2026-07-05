"""Coding/noncoding labels derived from ClinVar molecular consequences."""

from __future__ import annotations

import gzip
from pathlib import Path

CODING_CONSEQUENCES = {
    "exon_variant",
    "missense_variant",
    "nonsense_variant",
    "frameshift_variant",
    "inframe_deletion",
    "outofframe_deletion",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "splice_region_variant",
    "start_lost",
    "stop_gained",
    "stop_lost",
    "coding_sequence_variant",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
}


def build_coding_lookup(
    vcf_path: str | Path,
    wanted_positions: set[tuple[str, int]],
) -> dict[tuple[str, int], bool]:
    """Return coding status for requested positions using the INFO/MC field."""

    path = Path(vcf_path)
    opener = gzip.open if path.suffix == ".gz" else open
    lookup: dict[tuple[str, int], bool] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            try:
                key = (fields[0], int(fields[1]))
            except ValueError:
                continue
            if key not in wanted_positions:
                continue
            info = fields[7]
            consequences: set[str] = set()
            for entry in info.split(";"):
                if not entry.startswith("MC="):
                    continue
                for term in entry[3:].split(","):
                    consequences.add(term.rsplit("|", 1)[-1])
            lookup[key] = bool(consequences & CODING_CONSEQUENCES)
    return lookup
