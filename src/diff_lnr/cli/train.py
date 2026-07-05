"""Command-line interface for holdout training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import ProjectConfig
from ..workflows.train import run_training, validate_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Diff-LNR")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration")
    parser.add_argument("--data", help="override data.context_file")
    parser.add_argument("--model", help="override model.name_or_path")
    parser.add_argument("--output-root", help="override output.root")
    parser.add_argument("--run-name", help="override output.run_name")
    parser.add_argument("--device", help="override runtime.device")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate configuration and processed data without loading DNABERT-2",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ProjectConfig.from_yaml(args.config)
    if args.data:
        config.data.context_file = Path(args.data).expanduser().resolve()
    if args.model:
        config.model.name_or_path = args.model
    if args.output_root:
        config.output.root = Path(args.output_root).expanduser().resolve()
    if args.run_name:
        config.output.run_name = args.run_name
    if args.device:
        config.runtime.device = args.device
    config.validate()
    if args.validate_only:
        print(json.dumps(validate_inputs(config), indent=2, ensure_ascii=False))
        return
    output_dir = run_training(config, verbose=args.verbose)
    print(f"Run completed: {output_dir}")


if __name__ == "__main__":
    main()
