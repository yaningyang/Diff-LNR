"""Build processed context TSVs from a template TSV and reference FASTA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diff_lnr.data.context_builder import build_context_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", required=True, help="existing seven-column TSV")
    parser.add_argument("--fasta", required=True, help="plain or gzip-compressed FASTA")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--flank",
        type=int,
        nargs="+",
        default=[150, 200, 250, 300],
        help="per-side flank sizes in bp",
    )
    args = parser.parse_args()
    statistics = build_context_files(
        args.variants,
        args.fasta,
        args.output_dir,
        args.flank,
    )
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
