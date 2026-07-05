"""Command-line interface for checkpoint evaluation."""

from __future__ import annotations

import argparse

from ..config import ProjectConfig
from ..workflows.evaluate import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Diff-LNR checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    output = run_evaluation(
        ProjectConfig.from_yaml(args.config),
        checkpoint_path=args.checkpoint,
        split_path=args.splits,
        output_dir=args.output_dir,
    )
    print(f"Evaluation completed: {output}")


if __name__ == "__main__":
    main()
