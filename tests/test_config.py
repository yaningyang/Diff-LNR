from pathlib import Path

import pytest

from diff_lnr.config import ProjectConfig


def test_config_resolves_paths_and_integer_bucket_keys(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  context_file: data/input.tsv
model:
  representation: diff_lnr
training:
  batch_size_by_max_deletion:
    50: 4
    100: 2
output:
  root: runs
""",
        encoding="utf-8",
    )
    config = ProjectConfig.from_yaml(path)
    assert config.data.context_file == (tmp_path / "data/input.tsv").resolve()
    assert config.output.root == (tmp_path / "runs").resolve()
    assert config.training.batch_size_by_max_deletion == {50: 4, 100: 2}


def test_invalid_representation_is_rejected(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  context_file: data.tsv\nmodel:\n  representation: invalid\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="representation"):
        ProjectConfig.from_yaml(path)
