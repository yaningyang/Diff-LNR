"""Capture enough runtime metadata to audit an experiment."""

from __future__ import annotations

import platform
import sys
from importlib import metadata


def collect_environment() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "PyYAML", "scikit-learn", "torch", "transformers", "tqdm"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
