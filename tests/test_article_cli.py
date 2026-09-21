from pathlib import Path

import pytest

from scripts.run_article_audit import validate_config


def test_validate_config_reads_article_template() -> None:
    summary = validate_config(Path("configs/article.yaml"))

    assert summary["model"] == "changemamba"
    assert summary["dataset_count"] >= 1
    assert summary["canary_count"] == 4


def test_validate_config_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="configuration file"):
        validate_config(tmp_path / "missing.yaml")
