"""Evaluate one checkpoint within coding and noncoding test subsets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diff_lnr.config import ProjectConfig  # noqa: E402
from diff_lnr.workflows.region_control import run_region_control  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--vcf", required=True)
    parser.add_argument(
        "--keep-unmatched-separate",
        action="store_true",
        help="do not assign positions absent from the VCF to noncoding",
    )
    args = parser.parse_args()
    output = run_region_control(
        ProjectConfig.from_yaml(args.config),
        checkpoint_path=args.checkpoint,
        split_path=args.splits,
        vcf_path=args.vcf,
        missing_as_noncoding=not args.keep_unmatched_separate,
    )
    print(f"Region control completed: {output}")


if __name__ == "__main__":
    main()
