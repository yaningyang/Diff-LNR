"""Command-line interface for cross-validation."""

from __future__ import annotations

import argparse

from ..config import ProjectConfig
from ..workflows.cross_validate import run_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Diff-LNR cross-validation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    output = run_cross_validation(
        ProjectConfig.from_yaml(args.config),
        folds=args.folds,
    )
    print(f"Cross-validation completed: {output}")


if __name__ == "__main__":
    main()
