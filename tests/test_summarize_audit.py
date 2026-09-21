import csv
from pathlib import Path

from scripts.summarize_audit import summarize_run


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_run_computes_canary_and_evidence_ratios(tmp_path: Path) -> None:
    run = tmp_path / "target"
    _write_csv(
        run / "canary_scores.csv",
        ["canary_recall"],
        [{"canary_recall": "0.4"}, {"canary_recall": "0.6"}],
    )
    _write_csv(
        run / "passive_scores.csv",
        ["evidence_coverage"],
        [{"evidence_coverage": "0.5"}, {"evidence_coverage": "0.7"}],
    )

    result = summarize_run(run, source_canary_mean=1.0, source_evidence_mean=0.6)

    assert result["canary_mean"] == 0.5
    assert result["evidence_mean"] == 0.6
    assert result["canary_ratio"] == 0.5
    assert result["evidence_ratio"] == 1.0
    assert result["decision"] == "I"
