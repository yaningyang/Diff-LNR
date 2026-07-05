"""Integrated Gradients analysis for the Diff-LNR contrast."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import ProjectConfig
from ..data.sequence import build_aligned_alternate
from ..data.splits import SplitIndices, select_records
from ..utils.io import write_json
from ..utils.reproducibility import resolve_device, set_global_seed
from ..workflows.common import load_configured_records
from ..workflows.evaluate import load_checkpoint_model


class PathogenicLogitWrapper(nn.Module):
    """Expose one sequence branch at a time to Captum."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.direction = "reference"
        self.fixed_reference: torch.Tensor | None = None
        self.fixed_alternate: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.model.encoder.pooled_embedding(input_ids, attention_mask)
        if self.direction == "reference":
            if self.fixed_alternate is None:
                raise RuntimeError("fixed alternate embedding was not set")
            contrast = embedding - self.fixed_alternate
        else:
            if self.fixed_reference is None:
                raise RuntimeError("fixed reference embedding was not set")
            contrast = self.fixed_reference - embedding
        return self.model.classifier(contrast)


def run_integrated_gradients(
    config: ProjectConfig,
    *,
    checkpoint_path: str | Path,
    split_path: str | Path,
    confidence_threshold: float = 0.95,
    sample_count: int = 200,
    integration_steps: int = 32,
    alignment_window: int = 200,
    convergence_steps: list[int] | None = None,
) -> Path:
    """Compute normalized |IG(reference)-IG(aligned alternate)| profiles."""

    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError as error:
        raise RuntimeError(
            "Captum is required. Install with: pip install -e .[interpretability]"
        ) from error

    if config.model.representation != "diff_lnr":
        raise ValueError("Integrated Gradients analysis requires representation=diff_lnr")
    set_global_seed(config.training.seed, deterministic=config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    model, tokenizer, run_dir = load_checkpoint_model(config, checkpoint_path, device)
    records = load_configured_records(config).records
    test_records = select_records(records, SplitIndices.load(split_path).test)
    output_dir = run_dir / f"integrated_gradients_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True)

    wrapper = PathogenicLogitWrapper(model).to(device).eval()
    embedding_layer = model.backbone.embeddings.word_embeddings
    layer_ig = LayerIntegratedGradients(wrapper, embedding_layer)
    pad_token_id = tokenizer.pad_token_id or 0
    aligned_sum = np.zeros(2 * alignment_window + 1, dtype=np.float64)
    aligned_count = np.zeros_like(aligned_sum)
    used = 0
    selected_records = []

    for record in test_records:
        if used >= sample_count:
            break
        if record.label != 1:
            continue
        profile, center, probability = _difference_attribution(
            record,
            tokenizer,
            model,
            wrapper,
            layer_ig,
            device,
            pad_token_id,
            integration_steps,
        )
        if probability < confidence_threshold:
            continue
        for position, value in enumerate(profile):
            relative = position - center
            if -alignment_window <= relative <= alignment_window:
                aligned_sum[relative + alignment_window] += value
                aligned_count[relative + alignment_window] += 1
        selected_records.append(
            {
                "record_id": record.record_id,
                "pathogenic_probability": probability,
            }
        )
        used += 1

    if used == 0:
        raise RuntimeError("no pathogenic test records passed the confidence threshold")
    mean_attribution = np.divide(
        aligned_sum,
        aligned_count,
        out=np.zeros_like(aligned_sum),
        where=aligned_count > 0,
    )
    relative_positions = np.arange(-alignment_window, alignment_window + 1)
    np.savez(
        output_dir / "aligned_attribution.npz",
        relative_positions=relative_positions,
        mean_attribution=mean_attribution,
        counts=aligned_count,
    )
    _plot_profile(output_dir, relative_positions, mean_attribution, used)
    near = float(mean_attribution[(relative_positions >= -25) & (relative_positions <= 25)].mean())
    far = float(
        mean_attribution[
            (np.abs(relative_positions) >= 50) & (np.abs(relative_positions) <= 100)
        ].mean()
    )
    result: dict[str, Any] = {
        "method": "absolute_difference_of_branchwise_layer_integrated_gradients",
        "sample_count": used,
        "confidence_threshold": confidence_threshold,
        "integration_steps": integration_steps,
        "alignment_window": alignment_window,
        "mean_attribution_within_25bp": near,
        "mean_attribution_50_to_100bp": far,
        "near_to_far_ratio": near / far if far > 0 else None,
        "selected_records": selected_records,
    }
    if convergence_steps:
        result["convergence"] = _convergence_check(
            test_records,
            tokenizer,
            model,
            wrapper,
            layer_ig,
            device,
            pad_token_id,
            confidence_threshold,
            convergence_steps,
            min(30, sample_count),
        )
    write_json(output_dir / "summary.json", result)
    return output_dir


def _difference_attribution(
    record: Any,
    tokenizer: Any,
    model: nn.Module,
    wrapper: PathogenicLogitWrapper,
    layer_ig: Any,
    device: torch.device,
    pad_token_id: int,
    integration_steps: int,
) -> tuple[np.ndarray, int, float]:
    reference = record.reference_sequence
    alternate = build_aligned_alternate(record)
    reference_encoding = tokenizer(
        reference,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=False,
    )
    alternate_encoding = tokenizer(
        alternate,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=False,
    )
    reference_ids = reference_encoding["input_ids"].to(device)
    reference_mask = reference_encoding["attention_mask"].to(device)
    alternate_ids = alternate_encoding["input_ids"].to(device)
    alternate_mask = alternate_encoding["attention_mask"].to(device)
    reference_offsets = reference_encoding["offset_mapping"].squeeze(0).tolist()
    alternate_offsets = alternate_encoding["offset_mapping"].squeeze(0).tolist()

    with torch.no_grad():
        reference_embedding = model.encoder.pooled_embedding(reference_ids, reference_mask)
        alternate_embedding = model.encoder.pooled_embedding(alternate_ids, alternate_mask)
        probability = float(
            torch.softmax(model.classifier(reference_embedding - alternate_embedding), dim=1)[
                0, 1
            ].item()
        )

    wrapper.direction = "reference"
    wrapper.fixed_alternate = alternate_embedding.detach()
    reference_attr = layer_ig.attribute(
        inputs=reference_ids,
        baselines=torch.full_like(reference_ids, pad_token_id),
        additional_forward_args=(reference_mask,),
        target=1,
        n_steps=integration_steps,
    )
    wrapper.direction = "alternate"
    wrapper.fixed_reference = reference_embedding.detach()
    alternate_attr = layer_ig.attribute(
        inputs=alternate_ids,
        baselines=torch.full_like(alternate_ids, pad_token_id),
        additional_forward_args=(alternate_mask,),
        target=1,
        n_steps=integration_steps,
    )
    reference_base = _token_to_base(
        _summarize_token_attribution(reference_attr),
        reference_offsets,
        len(reference),
    )
    alternate_base = _token_to_base(
        _summarize_token_attribution(alternate_attr),
        alternate_offsets,
        len(alternate),
    )
    length = min(len(reference_base), len(alternate_base))
    profile = np.abs(reference_base[:length] - alternate_base[:length])
    if profile.max() > 0:
        profile /= profile.max()
    true_start = record.deletion_start + len(record.alt_allele)
    true_end = true_start + record.deletion_length
    center = (true_start + true_end) // 2
    return profile, center, probability


def _summarize_token_attribution(attribution: torch.Tensor) -> np.ndarray:
    return attribution.sum(dim=-1).squeeze(0).detach().cpu().numpy()


def _token_to_base(
    token_attribution: np.ndarray,
    offsets: list[list[int]],
    sequence_length: int,
) -> np.ndarray:
    base_attribution = np.zeros(sequence_length, dtype=np.float64)
    for token_index, (start, end) in enumerate(offsets):
        if end <= start or start >= sequence_length:
            continue
        end = min(end, sequence_length)
        base_attribution[start:end] += token_attribution[token_index] / (end - start)
    return base_attribution


def _convergence_check(
    records: list,
    tokenizer: Any,
    model: nn.Module,
    wrapper: PathogenicLogitWrapper,
    layer_ig: Any,
    device: torch.device,
    pad_token_id: int,
    threshold: float,
    steps: list[int],
    sample_count: int,
) -> dict[str, Any]:
    candidates = [record for record in records if record.label == 1]
    result = {}
    for step_count in steps:
        values = []
        for record in candidates:
            if len(values) >= sample_count:
                break
            profile, center, probability = _difference_attribution(
                record,
                tokenizer,
                model,
                wrapper,
                layer_ig,
                device,
                pad_token_id,
                step_count,
            )
            if probability < threshold:
                continue
            lower = max(0, center - 25)
            upper = min(len(profile), center + 26)
            values.append(float(profile[lower:upper].mean()))
        result[str(step_count)] = {
            "mean_attribution_within_25bp": float(np.mean(values)) if values else None,
            "sample_count": len(values),
        }
    return result


def _plot_profile(
    output_dir: Path,
    positions: np.ndarray,
    attribution: np.ndarray,
    sample_count: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.plot(positions, attribution, color="#3B82F6", linewidth=1.2)
    axis.axvline(0, color="#E85D75", linestyle="--", linewidth=1.0)
    axis.axvspan(-25, 25, color="#F6C6A8", alpha=0.25)
    axis.set_xlabel("Position relative to deletion center (bp)")
    axis.set_ylabel("Normalized difference attribution")
    axis.set_title(f"Diff-LNR Integrated Gradients (n={sample_count})")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "attribution_profile.png", dpi=200)
    plt.close(figure)
