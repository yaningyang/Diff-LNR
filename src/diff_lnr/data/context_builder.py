"""Generate alternate-allele context TSVs from a reference FASTA."""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO


def load_variant_templates(path: str | Path) -> list[dict[str, Any]]:
    variants = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 7:
                continue
            chromosome, position, ref, alt, labels = row[:5]
            variants.append(
                {
                    "chromosome": chromosome,
                    "position": int(position),
                    "ref": ref.upper(),
                    "alt": alt.upper(),
                    "labels": labels,
                }
            )
    return variants


def stream_fasta(path: str | Path) -> Iterator[tuple[str, bytes]]:
    fasta_path = Path(path)
    opener = gzip.open if fasta_path.suffix == ".gz" else open
    name: str | None = None
    sequence = bytearray()
    with opener(fasta_path, "rb") as handle:
        for line in handle:
            if line.startswith(b">"):
                if name is not None:
                    yield name, bytes(sequence).upper()
                name = line[1:].decode("ascii", errors="replace").split()[0]
                sequence.clear()
            else:
                sequence.extend(line.rstrip(b"\r\n"))
    if name is not None:
        yield name, bytes(sequence).upper()


def build_context_files(
    variant_file: str | Path,
    fasta_file: str | Path,
    output_dir: str | Path,
    flank_sizes: list[int],
) -> dict[int, dict[str, int | str]]:
    """Write one TSV for each per-side flank length."""

    variants = load_variant_templates(variant_file)
    by_chromosome: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        by_chromosome.setdefault(variant["chromosome"], []).append(variant)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    statistics = {
        flank: {
            "output": str(destination / f"clinvar_deletions_context_{2 * flank}.txt"),
            "kept": 0,
            "out_of_bounds": 0,
            "reference_mismatch": 0,
        }
        for flank in flank_sizes
    }

    with ExitStack() as stack:
        writers: dict[int, Any] = {}
        for flank in flank_sizes:
            handle: TextIO = stack.enter_context(
                Path(statistics[flank]["output"]).open("w", encoding="utf-8", newline="")
            )
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                [
                    "CHROM",
                    "POS",
                    "REF",
                    "ALT",
                    "labels",
                    "context_sequence",
                    "var_position",
                ]
            )
            writers[flank] = writer

        for chromosome, sequence in stream_fasta(fasta_file):
            chromosome_variants = by_chromosome.get(chromosome)
            if not chromosome_variants:
                continue
            for flank, writer in writers.items():
                _write_chromosome_contexts(
                    sequence,
                    chromosome_variants,
                    flank,
                    writer,
                    statistics[flank],
                )
    return statistics


def _write_chromosome_contexts(
    sequence: bytes,
    variants: list[dict[str, Any]],
    flank: int,
    writer: Any,
    statistics: dict[str, int | str],
) -> None:
    sequence_length = len(sequence)
    for variant in variants:
        reference_start = variant["position"] - 1
        reference_end = reference_start + len(variant["ref"])
        if reference_start < 0 or reference_end > sequence_length:
            statistics["out_of_bounds"] += 1
            continue
        genome_reference = sequence[reference_start:reference_end].decode("ascii")
        if genome_reference.upper() != variant["ref"]:
            statistics["reference_mismatch"] += 1
            continue
        upstream_start = reference_start - flank
        downstream_start = reference_start + len(variant["alt"])
        downstream_end = downstream_start + flank
        if upstream_start < 0 or downstream_end > sequence_length:
            statistics["out_of_bounds"] += 1
            continue
        upstream = sequence[upstream_start:reference_start].decode("ascii")
        downstream = sequence[downstream_start:downstream_end].decode("ascii")
        context = upstream + variant["alt"] + downstream
        writer.writerow(
            [
                variant["chromosome"],
                variant["position"],
                variant["ref"],
                variant["alt"],
                variant["labels"],
                context,
                flank,
            ]
        )
        statistics["kept"] += 1
