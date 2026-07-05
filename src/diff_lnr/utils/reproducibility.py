"""Randomness control and runtime selection."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def move_model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key.endswith("_input_ids") or key.endswith("_attention_mask")
    }
