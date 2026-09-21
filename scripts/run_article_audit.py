"""Validate the article configuration or run the reproducibility pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_config(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _repository_root() / candidate


def _load_config(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    os.chdir(_repository_root())
    for import_root in (_repository_root(), _repository_root() / "src"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from scp.utils import load_config

    return load_config(path)


def validate_config(path: Path) -> dict[str, Any]:
    """Parse an article YAML file and return a compact summary."""
    config = _load_config(path)
    return {
        "model": "changemamba",
        "dataset_count": len(config.datasets),
        "canary_count": len(config.canary.enabled_types),
        "threshold": config.threshold,
        "results_root": str(config.paths.results_root),
    }


def _check_inputs(path: Path) -> None:
    config = _load_config(path)
    missing: list[str] = []
    for name, dataset in config.datasets.items():
        if not dataset.path.exists():
            missing.append(f"dataset {name}: {dataset.path}")
    for name in ("bit", "changeformer", "cdmamba", "changemamba", "tinycd"):
        checkpoint = getattr(config.models, name).checkpoint
        if checkpoint is not None and not checkpoint.exists():
            missing.append(f"checkpoint {name}: {checkpoint}")
    for name in ("bit_repo", "changeformer_repo", "cdmamba_repo", "changemamba_repo", "tinycd_repo"):
        repository = getattr(config.paths, name)
        if not repository.exists():
            missing.append(f"model repository {name}: {repository}")
    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"configured article inputs are missing:\n{details}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canary-change-probe article audit.")
    parser.add_argument("--config", default="configs/article.yaml")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--model", default="changemamba")
    parser.add_argument("--source", default="levir_val=levir-256:val")
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-analyses", action="store_true")
    parser.add_argument("--label-free", action="store_true")
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--skip-canary-bank", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run validation or delegate to the article pipeline."""
    args = _build_parser().parse_args(argv)
    config_path = _resolve_config(args.config)
    summary = validate_config(config_path)
    if args.validate_only:
        print("Article configuration is valid.")
        for key, value in summary.items():
            print(f"{key}: {value}")
        return 0
    if args.check_inputs:
        _check_inputs(config_path)
    os.chdir(_repository_root())
    if str(_repository_root()) not in sys.path:
        sys.path.insert(0, str(_repository_root()))
    pipeline_args = ["--config", str(config_path), "--model", args.model, "--source", args.source]
    for run in args.runs or []:
        pipeline_args.extend(["--run", run])
    if args.limit is not None:
        pipeline_args.extend(["--limit", str(args.limit)])
    for enabled, option in (
        (args.skip_analyses, "--skip-analyses"),
        (args.label_free, "--label-free"),
        (args.skip_env_check, "--skip-env-check"),
        (args.skip_canary_bank, "--skip-canary-bank"),
        (args.dry_run, "--dry-run"),
    ):
        if enabled:
            pipeline_args.append(option)
    from scripts.run_pipeline import main as pipeline_main

    return pipeline_main(pipeline_args)


if __name__ == "__main__":
    raise SystemExit(main())
