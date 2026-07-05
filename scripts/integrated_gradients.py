"""Run the paper's branchwise Integrated Gradients analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diff_lnr.config import ProjectConfig  # noqa: E402
from diff_lnr.interpretability import run_integrated_gradients  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--convergence-steps", type=int, nargs="*")
    args = parser.parse_args()
    output = run_integrated_gradients(
        ProjectConfig.from_yaml(args.config),
        checkpoint_path=args.checkpoint,
        split_path=args.splits,
        confidence_threshold=args.confidence,
        sample_count=args.samples,
        integration_steps=args.steps,
        alignment_window=args.window,
        convergence_steps=args.convergence_steps,
    )
    print(f"Integrated Gradients completed: {output}")


if __name__ == "__main__":
    main()
