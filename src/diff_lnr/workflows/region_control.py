"""Coding/noncoding stratified evaluation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from ..data.regions import build_coding_lookup
from ..data.splits import SplitIndices, select_records
from ..training import evaluate_loader
from ..utils.io import write_json
from ..utils.logging import configure_logging
from ..utils.reproducibility import resolve_device, set_global_seed
from .common import class_counts, load_configured_records
from .evaluate import load_checkpoint_model, make_evaluation_loader


def run_region_control(
    config: ProjectConfig,
    *,
    checkpoint_path: str | Path,
    split_path: str | Path,
    vcf_path: str | Path,
    missing_as_noncoding: bool = True,
) -> Path:
    set_global_seed(config.training.seed, deterministic=config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    model, tokenizer, run_dir = load_checkpoint_model(config, checkpoint_path, device)
    output_dir = run_dir / f"region_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True)
    logger = configure_logging(output_dir)

    records = load_configured_records(config).records
    test_records = select_records(records, SplitIndices.load(split_path).test)
    positions = {(record.chromosome, record.position) for record in test_records}
    lookup = build_coding_lookup(vcf_path, positions)
    groups: dict[str, list] = {"coding": [], "noncoding": [], "unmatched": []}
    for record in test_records:
        key = (record.chromosome, record.position)
        if key not in lookup:
            group = "noncoding" if missing_as_noncoding else "unmatched"
        else:
            group = "coding" if lookup[key] else "noncoding"
        groups[group].append(record)

    result: dict[str, Any] = {
        "missing_as_noncoding": missing_as_noncoding,
        "groups": {},
    }
    for name in ("coding", "noncoding"):
        subset = groups[name]
        if not subset:
            result["groups"][name] = {"counts": class_counts(subset), "metrics": None}
            continue
        loader = make_evaluation_loader(subset, tokenizer, config)
        metrics, _ = evaluate_loader(
            model,
            loader,
            device,
            output_dir=output_dir / name,
            save_predictions=config.output.save_predictions,
        )
        result["groups"][name] = {
            "counts": class_counts(subset),
            "metrics": metrics,
        }
        logger.info("%s: counts=%s metrics=%s", name, class_counts(subset), metrics)
    result["unmatched_count"] = len(groups["unmatched"])
    write_json(output_dir / "summary.json", result)
    return output_dir
