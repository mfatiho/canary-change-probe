"""Convert pipeline CSV artifacts into source-normalized audit decisions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts._bootstrap import add_src_to_path
except ModuleNotFoundError:
    from _bootstrap import add_src_to_path

add_src_to_path()

from scp.d9.k2_rules import PRIMARY_EQ4_RULE, apply_rule


def _mean_column(path: Path, column: str) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        values = [float(row[column]) for row in csv.DictReader(handle)]
    if not values:
        raise ValueError(f"artifact has no rows: {path}")
    return sum(values) / len(values)


def summarize_run(
    run_directory: Path,
    source_canary_mean: float,
    source_evidence_mean: float,
) -> dict[str, Any]:
    """Compute mean responses, source-normalized ratios, and the fixed decision."""
    if source_canary_mean <= 0.0 or source_evidence_mean <= 0.0:
        raise ValueError("source means must be positive")
    canary_mean = _mean_column(run_directory / "canary_scores.csv", "canary_recall")
    evidence_mean = _mean_column(run_directory / "passive_scores.csv", "evidence_coverage")
    canary_ratio = canary_mean / source_canary_mean
    evidence_ratio = evidence_mean / source_evidence_mean
    return {
        "canary_mean": canary_mean,
        "evidence_mean": evidence_mean,
        "canary_ratio": canary_ratio,
        "evidence_ratio": evidence_ratio,
        "decision": apply_rule(canary_ratio, evidence_ratio, PRIMARY_EQ4_RULE),
    }


def _parse_run(raw: str) -> tuple[str, str, str]:
    try:
        name, value = raw.split("=", 1)
        dataset, split = value.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("run must have NAME=DATASET:SPLIT form") from error
    if not name or not dataset or not split:
        raise argparse.ArgumentTypeError("run must have NAME=DATASET:SPLIT form")
    return name, dataset, split


def _run_directory(results_root: Path, model: str, dataset: str, split: str) -> Path:
    return results_root / "silent_failure" / "canary_scores" / model / dataset / split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize canary and evidence audit artifacts.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--model", default="changemamba")
    parser.add_argument("--source", required=True, type=_parse_run)
    parser.add_argument("--target", action="append", required=True, type=_parse_run)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    source_name, source_dataset, source_split = args.source
    source_directory = _run_directory(args.results_root, args.model, source_dataset, source_split)
    source_canary_mean = _mean_column(source_directory / "canary_scores.csv", "canary_recall")
    source_evidence_mean = _mean_column(
        source_directory / "passive_scores.csv", "evidence_coverage"
    )
    records: list[dict[str, Any]] = []
    for name, dataset, split in args.target:
        record = summarize_run(
            _run_directory(args.results_root, args.model, dataset, split),
            source_canary_mean,
            source_evidence_mean,
        )
        records.append({"run": name, "dataset": dataset, "split": split, **record})
    payload = {
        "source": {
            "run": source_name,
            "dataset": source_dataset,
            "split": source_split,
            "canary_mean": source_canary_mean,
            "evidence_mean": source_evidence_mean,
        },
        "rule": PRIMARY_EQ4_RULE,
        "targets": records,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
