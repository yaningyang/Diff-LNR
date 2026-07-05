"""Create the index file for the one-to-one deletion-length control."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diff_lnr.config import ProjectConfig  # noqa: E402
from diff_lnr.data.matching import match_deletion_lengths  # noqa: E402
from diff_lnr.utils.io import write_json  # noqa: E402
from diff_lnr.workflows.common import load_configured_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-difference", type=int, default=5)
    parser.add_argument(
        "--output",
        default="data/length_matched_indices.json",
    )
    args = parser.parse_args()
    config = ProjectConfig.from_yaml(args.config)
    records = load_configured_records(config).records
    result = match_deletion_lengths(
        records,
        max_difference=args.max_difference,
        seed=config.training.seed,
    )
    write_json(
        args.output,
        {
            "indices": result.indices,
            "statistics": result.statistics,
        },
    )
    print(f"Matched subset written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
