"""Console and file logging."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(output_dir: str | Path, *, verbose: bool = False) -> logging.Logger:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("diff_lnr")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(output_path / "run.log", encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
