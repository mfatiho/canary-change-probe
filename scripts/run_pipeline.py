"""Run the full SCP inference and analysis pipeline with progress reporting."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts._bootstrap import add_src_to_path
except ModuleNotFoundError:
    from _bootstrap import add_src_to_path

add_src_to_path()

DEFAULT_RUNS = (
    "levir_test=levir-256:test",
    "levirplus=levirplus-256:test",
    "whu=whu-256:test",
    "sysu=sysu-256:test",
    "s2looking=s2looking-256:test",
    "dsifn=dsifn-256:test",
    "clcd=clcd-256:test",
)
PIPELINE_STAGES = ("prob_maps", "passive", "canary", "labels")
ANALYSIS_NAMES = (
    "taxonomy",
    "threshold_sensitivity",
    "deploy_sensitivity",
    "canary_ablation",
)
CSV_FILENAMES = {
    "passive": "passive_scores.csv",
    "canary": "canary_scores.csv",
    "labels": "eval_labels.csv",
}
REPORT_FILENAMES = {
    "taxonomy": "taxonomy_report.json",
    "threshold_sensitivity": "threshold_sensitivity_report.json",
    "deploy_sensitivity": "deploy_sensitivity_report.json",
    "canary_ablation": "canary_ablation_report.json",
}


@dataclass(frozen=True)
class RunSpec:
    """One named dataset split processed by the pipeline."""

    name: str
    dataset: str
    split: str


@dataclass(frozen=True)
class StepResult:
    """Result row for one pipeline step."""

    step: str
    target: str
    status: str
    elapsed_seconds: float
    output: Path | None = None
    note: str = ""


def build_parser() -> argparse.ArgumentParser:
    """Build the full-pipeline CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the full SCP pipeline and report per-step status."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--model", default="changemamba", help="Registered model adapter name.")
    parser.add_argument(
        "--source",
        default="levir_val=levir-256:val",
        metavar="NAME=DATASET:SPLIT",
        help="Source/calibration run used by analyses.",
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        metavar="NAME=DATASET:SPLIT",
        help="Target run to process. Can be repeated. Defaults to all paper domains.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional pair limit per run.")
    parser.add_argument(
        "--canary-bank-dataset",
        default="levir-256",
        help="Dataset used to build the canary crop bank when missing.",
    )
    parser.add_argument(
        "--canary-bank-split",
        default="train",
        help="Split used to build the canary crop bank when missing.",
    )
    parser.add_argument(
        "--canary-bank-limit",
        type=int,
        default=None,
        help="Optional pair limit for canary-bank construction.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Pipeline staging/report root. Defaults to results/changemamba_pipeline/<model>.",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip model runtime import checks.",
    )
    parser.add_argument(
        "--skip-canary-bank",
        action="store_true",
        help="Do not build the configured canary bank if it is missing.",
    )
    parser.add_argument(
        "--skip-analyses",
        action="store_true",
        help="Only generate per-run artifacts; skip taxonomy and ablation reports.",
    )
    parser.add_argument(
        "--label-free",
        action="store_true",
        help=(
            "Generate and stage only passive/canary evidence. Refuse pre-existing "
            "label artifacts and require --skip-analyses."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining steps after a failure and summarize all errors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned steps without executing them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the configured pipeline and return a shell exit status."""
    from scp.utils import load_config

    args = build_parser().parse_args(argv)
    _validate_label_free_args(args)
    config_path = Path(args.config)
    config = load_config(config_path)
    source_run = _parse_run_spec(args.source)
    target_runs = [_parse_run_spec(raw_run) for raw_run in (args.runs or DEFAULT_RUNS)]
    runs = _deduplicate_runs([source_run, *target_runs])
    _validate_source_canary_bank(source_run, args.canary_bank_dataset)
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else config.paths.results_root / "changemamba_pipeline" / args.model
    )
    if args.label_free:
        for run in runs:
            _assert_label_free_paths(config, args.model, run, output_root)

    planned_steps = _planned_step_count(
        run_count=len(runs),
        include_env=not args.skip_env_check,
        include_bank=not args.skip_canary_bank,
        include_analyses=not args.skip_analyses,
        include_labels=not args.label_free,
    )
    print(f"Pipeline model={args.model} config={config_path} steps={planned_steps}")
    print(f"Pipeline output root: {output_root}")
    if args.dry_run:
        _print_dry_run(runs, source_run, args, output_root)
        return 0

    progress = Progress(total=planned_steps)
    results: list[StepResult] = []

    def run_step(step: str, target: str, action: Callable[[], Path | None]) -> bool:
        result = _execute_step(step, target, progress, action)
        results.append(result)
        if result.status == "OK":
            return True
        if not args.continue_on_error:
            _print_summary(results)
            return False
        return True

    if not args.skip_env_check and not run_step(
        "env_check",
        args.model,
        lambda: _check_runtime(args.model, config, source_run),
    ):
        return 1

    if not args.skip_canary_bank and not run_step(
        "canary_bank",
        f"{args.canary_bank_dataset}:{args.canary_bank_split}",
        lambda: _ensure_canary_bank(
            config=config,
            dataset_name=args.canary_bank_dataset,
            split=args.canary_bank_split,
            limit=args.canary_bank_limit,
        ),
    ):
        return 1

    active_adapter: Any = None

    def _load_active_adapter() -> None:
        nonlocal active_adapter
        active_adapter = _build_shared_adapter(config, args.model)
        return None

    if not run_step("load_adapter", args.model, _load_active_adapter):
        return 1

    for run in runs:
        if not run_step(
            "prob_maps",
            run.name,
            lambda run=run: _generate_probability_maps(
                config,
                args.model,
                run,
                config_path,
                args.limit,
                active_adapter,
            ),
        ):
            return 1
        if not run_step(
            "passive",
            run.name,
            lambda run=run: _compute_passive_scores(
                config,
                args.model,
                run,
                config_path,
                args.limit,
                active_adapter,
            ),
        ):
            return 1
        if not run_step(
            "canary",
            run.name,
            lambda run=run: _compute_canary_scores(
                config,
                args.model,
                run,
                config_path,
                args.limit,
                active_adapter,
            ),
        ):
            return 1
        if not args.label_free:
            if not run_step(
                "labels",
                run.name,
                lambda run=run: _compute_eval_labels(
                    config, args.model, run, config_path, args.limit, active_adapter
                ),
            ):
                return 1
        if not run_step(
            "stage",
            run.name,
            lambda run=run: _stage_run_artifacts(
                config,
                args.model,
                run,
                output_root,
                include_labels=not args.label_free,
            ),
        ):
            return 1

    if not args.skip_analyses:
        run_dirs = {run.name: output_root / "runs" / run.name for run in runs}
        for analysis_name in ANALYSIS_NAMES:
            if not run_step(
                analysis_name,
                source_run.name,
                lambda analysis_name=analysis_name: _run_analysis(
                    analysis_name=analysis_name,
                    run_dirs=run_dirs,
                    source_name=source_run.name,
                    output_root=output_root,
                ),
            ):
                return 1

    _print_summary(results)
    return 0 if all(result.status == "OK" for result in results) else 1


class Progress:
    """Small terminal progress bar with no third-party dependency."""

    def __init__(self, total: int, width: int = 28) -> None:
        self._total = max(total, 1)
        self._width = width
        self._current = 0

    def advance(self, label: str, status: str) -> None:
        """Advance the progress bar by one step and print status."""
        self._current += 1
        filled = min(self._width, int(self._width * self._current / self._total))
        bar = "#" * filled + "-" * (self._width - filled)
        print(f"[{bar}] {self._current:>3}/{self._total:<3} {status:<4} {label}")


def _execute_step(
    step: str,
    target: str,
    progress: Progress,
    action: Callable[[], Path | None],
) -> StepResult:
    label = f"{step}:{target}"
    start_time = time.perf_counter()
    try:
        output = action()
    except Exception as error:  # noqa: BLE001 - final table must report any failure.
        elapsed = time.perf_counter() - start_time
        progress.advance(label, "FAIL")
        return StepResult(
            step=step,
            target=target,
            status="FAIL",
            elapsed_seconds=elapsed,
            note=f"{type(error).__name__}: {error}",
        )
    elapsed = time.perf_counter() - start_time
    progress.advance(label, "OK")
    return StepResult(
        step=step,
        target=target,
        status="OK",
        elapsed_seconds=elapsed,
        output=output,
    )


def _check_runtime(model_name: str, config: Any, source_run: RunSpec) -> Path | None:
    if model_name == "changemamba":
        try:
            from scripts.check_changemamba_env import (
                _check_checkpoint_dataset,
                _check_cuda_runtime,
                _check_imports,
            )
        except ModuleNotFoundError:
            from check_changemamba_env import (  # type: ignore[no-redef]
                _check_checkpoint_dataset,
                _check_cuda_runtime,
                _check_imports,
            )

        _check_cuda_runtime(config.device)
        _check_imports(config.paths.changemamba_repo)
        _check_checkpoint_dataset(
            checkpoint_path=config.models.changemamba.checkpoint,
            configured_dataset=config.models.changemamba.trained_dataset,
            source_dataset=source_run.dataset,
        )
    if model_name == "cdmamba":
        try:
            from scripts.check_cdmamba_env import (
                _check_checkpoint_config,
                _check_checkpoint_dataset,
                _check_cuda_runtime,
                _check_file,
                _check_imports,
                _check_upstream_config_dataset,
            )
        except ModuleNotFoundError:
            from check_cdmamba_env import (  # type: ignore[no-redef]
                _check_checkpoint_config,
                _check_checkpoint_dataset,
                _check_cuda_runtime,
                _check_file,
                _check_imports,
                _check_upstream_config_dataset,
        )

        _check_cuda_runtime(config.device)
        _check_file(config.models.cdmamba.config_path, "CDMamba config")
        _check_file(config.models.cdmamba.checkpoint, "CDMamba checkpoint")
        _check_checkpoint_dataset(
            configured_dataset=config.models.cdmamba.trained_dataset,
            source_dataset=source_run.dataset,
        )
        _check_upstream_config_dataset(
            config_path=config.models.cdmamba.config_path,
            configured_dataset=config.models.cdmamba.trained_dataset,
        )
        _check_checkpoint_config(
            checkpoint_path=config.models.cdmamba.checkpoint,
            config_path=config.models.cdmamba.config_path,
        )
        _check_imports(config.paths.cdmamba_repo)
    return None


def _validate_source_canary_bank(source_run: RunSpec, canary_bank_dataset: str) -> None:
    if source_run.dataset != canary_bank_dataset:
        raise ValueError(
            "source/canary-bank mismatch: "
            f"source dataset is {source_run.dataset}, but canary bank dataset is "
            f"{canary_bank_dataset}. Build the canary bank from the source dataset."
        )


def _validate_label_free_args(args: argparse.Namespace) -> None:
    """Reject CLI combinations that could access labels during stage one."""
    if args.label_free and not args.skip_analyses:
        raise ValueError("--label-free requires --skip-analyses")


def _assert_label_free_paths(
    config: Any,
    model_name: str,
    run: RunSpec,
    output_root: Path,
) -> None:
    """Fail if a label artifact already exists in a label-free run tree."""
    label_paths = (
        _artifact_dir(config, "eval_labels", model_name, run) / CSV_FILENAMES["labels"],
        output_root / "runs" / run.name / CSV_FILENAMES["labels"],
    )
    existing_paths = [str(path) for path in label_paths if path.exists()]
    if existing_paths:
        joined_paths = ", ".join(existing_paths)
        raise FileExistsError(
            "label-free execution requires a clean unlabeled run tree; "
            f"found label artifacts: {joined_paths}"
        )


def _ensure_canary_bank(
    config: Any,
    dataset_name: str,
    split: str,
    limit: int | None,
) -> Path:
    from scp.canary.bank import build_canary_bank
    from scp.data import get_dataset
    from scp.utils import ConfigError

    if config.canary.bank_path is None:
        raise ConfigError("canary.bank_path is not configured")
    output_path = config.canary.bank_path
    if output_path.is_file():
        return output_path
    dataset_config = config.datasets.get(dataset_name)
    if dataset_config is None:
        raise ConfigError(f"dataset is not configured: {dataset_name}")
    build_canary_bank(
        dataset=get_dataset(dataset_name, dataset_config),
        dataset_name=dataset_name,
        split=split,
        output_path=output_path,
        min_area=64,
        max_area=4096,
        limit=limit,
    )
    return output_path


def _build_shared_adapter(config: Any, model_name: str) -> Any:
    """Build one model adapter to reuse across every stage and run.

    Each stage function accepts an optional pre-built adapter; without one
    the model checkpoint would be reloaded onto the device once per stage
    per dataset instead of once for the whole pipeline invocation.
    """
    from scp.pipeline import build_model_adapter

    return build_model_adapter(config, model_name)


def _generate_probability_maps(
    config: Any,
    model_name: str,
    run: RunSpec,
    config_path: Path,
    limit: int | None,
    adapter: Any,
) -> Path:
    from scp.pipeline import generate_probability_maps

    return generate_probability_maps(
        config=config,
        model_name=model_name,
        dataset_name=run.dataset,
        split=run.split,
        config_path=config_path,
        limit=limit,
        adapter=adapter,
    )


def _compute_passive_scores(
    config: Any,
    model_name: str,
    run: RunSpec,
    config_path: Path,
    limit: int | None,
    adapter: Any,
) -> Path:
    from scp.pipeline import compute_passive_scores

    return compute_passive_scores(
        config=config,
        model_name=model_name,
        dataset_name=run.dataset,
        split=run.split,
        config_path=config_path,
        limit=limit,
        adapter=adapter,
    )


def _compute_canary_scores(
    config: Any,
    model_name: str,
    run: RunSpec,
    config_path: Path,
    limit: int | None,
    adapter: Any,
) -> Path:
    from scp.pipeline import compute_canary_scores

    return compute_canary_scores(
        config=config,
        model_name=model_name,
        dataset_name=run.dataset,
        split=run.split,
        config_path=config_path,
        limit=limit,
        adapter=adapter,
    )


def _compute_eval_labels(
    config: Any,
    model_name: str,
    run: RunSpec,
    config_path: Path,
    limit: int | None,
    adapter: Any,
) -> Path:
    from scp.pipeline import compute_eval_labels

    return compute_eval_labels(
        config=config,
        model_name=model_name,
        dataset_name=run.dataset,
        split=run.split,
        config_path=config_path,
        limit=limit,
        adapter=adapter,
    )


def _stage_run_artifacts(
    config: Any,
    model_name: str,
    run: RunSpec,
    output_root: Path,
    *,
    include_labels: bool = True,
) -> Path:
    staged_dir = output_root / "runs" / run.name
    staged_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "passive": (
            _artifact_dir(config, "passive_scores", model_name, run)
            / CSV_FILENAMES["passive"]
        ),
        "canary": _artifact_dir(config, "canary_scores", model_name, run) / CSV_FILENAMES["canary"],
    }
    if include_labels:
        source_paths["labels"] = (
            _artifact_dir(config, "eval_labels", model_name, run) / CSV_FILENAMES["labels"]
        )
    for stage_name, source_path in source_paths.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"missing {stage_name} artifact: {source_path}")
        target_name = CSV_FILENAMES[stage_name]
        shutil.copy2(source_path, staged_dir / target_name)
    return staged_dir


def _artifact_dir(config: Any, artifact_name: str, model_name: str, run: RunSpec) -> Path:
    return (
        config.paths.results_root
        / "silent_failure"
        / artifact_name
        / model_name
        / run.dataset
        / run.split
    )


def _run_analysis(
    analysis_name: str,
    run_dirs: dict[str, Path],
    source_name: str,
    output_root: Path,
) -> Path:
    output_dir = output_root / analysis_name
    if analysis_name == "taxonomy":
        from scp.analysis import analyze_taxonomy

        return analyze_taxonomy(runs=run_dirs, source=source_name, output_dir=output_dir)
    if analysis_name == "threshold_sensitivity":
        from scp.threshold_sensitivity import analyze_threshold_sensitivity

        return analyze_threshold_sensitivity(
            runs=run_dirs,
            source=source_name,
            output_dir=output_dir,
        )
    if analysis_name == "deploy_sensitivity":
        from scp.deploy_sensitivity import analyze_deploy_sensitivity

        return analyze_deploy_sensitivity(
            runs=run_dirs,
            source=source_name,
            output_dir=output_dir,
        )
    if analysis_name == "canary_ablation":
        from scp.canary_ablation import analyze_canary_ablation

        return analyze_canary_ablation(
            runs=run_dirs,
            source=source_name,
            output_dir=output_dir,
        )
    raise ValueError(f"unknown analysis: {analysis_name}")


def _parse_run_spec(raw_spec: str) -> RunSpec:
    name, separator, dataset_split = raw_spec.partition("=")
    dataset, split_separator, split = dataset_split.partition(":")
    if not name or separator != "=" or not dataset or split_separator != ":" or not split:
        raise argparse.ArgumentTypeError(
            f"run must use NAME=DATASET:SPLIT syntax: {raw_spec}"
        )
    return RunSpec(name=name, dataset=dataset, split=split)


def _deduplicate_runs(runs: Sequence[RunSpec]) -> list[RunSpec]:
    seen: set[str] = set()
    deduplicated: list[RunSpec] = []
    for run in runs:
        if run.name in seen:
            continue
        seen.add(run.name)
        deduplicated.append(run)
    return deduplicated


def _planned_step_count(
    run_count: int,
    include_env: bool,
    include_bank: bool,
    include_analyses: bool,
    include_labels: bool = True,
) -> int:
    stage_count = len(PIPELINE_STAGES) if include_labels else len(PIPELINE_STAGES) - 1
    count = run_count * (stage_count + 1) + 1  # +1: shared adapter load
    if include_env:
        count += 1
    if include_bank:
        count += 1
    if include_analyses:
        count += len(ANALYSIS_NAMES)
    return count


def _print_dry_run(
    runs: Sequence[RunSpec],
    source_run: RunSpec,
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    print("Dry run only. Planned run order:")
    print(f"  source: {source_run.name}={source_run.dataset}:{source_run.split}")
    for run in runs:
        print(f"  run:    {run.name}={run.dataset}:{run.split}")
    print(f"  output: {output_root}")
    print(f"  limit:  {args.limit if args.limit is not None else 'full'}")
    print(f"  labels: {'excluded' if args.label_free else 'included'}")


def _print_summary(results: Sequence[StepResult]) -> None:
    print()
    print("Pipeline summary")
    rows = [
        [
            str(index),
            result.step,
            result.target,
            result.status,
            f"{result.elapsed_seconds:.1f}s",
            _row_count_note(result.output) if result.status == "OK" else result.note,
        ]
        for index, result in enumerate(results, start=1)
    ]
    _print_table(["#", "Step", "Target", "Status", "Time", "Output / note"], rows)


def _row_count_note(path: Path | None) -> str:
    if path is None:
        return ""
    if path.is_file():
        return str(path)
    if path.is_dir():
        csv_counts = [
            f"{csv_path.name}:{_count_csv_rows(csv_path)}"
            for csv_path in sorted(path.glob("*.csv"))
        ]
        if csv_counts:
            return f"{path} ({', '.join(csv_counts)})"
        reports = [REPORT_FILENAMES[name] for name in ANALYSIS_NAMES]
        report_names = [report for report in reports if (path / report).is_file()]
        if report_names:
            return f"{path} ({', '.join(report_names)})"
        return str(path)
    return str(path)


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        row_count = sum(1 for _row in reader)
    return max(row_count - 1, 0)


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [
        max(len(headers[column_index]), *(len(row[column_index]) for row in rows))
        for column_index in range(len(headers))
    ]
    separator = "+".join("-" * (width + 2) for width in widths)
    print(separator)
    print(_format_row(headers, widths))
    print(separator)
    for row in rows:
        print(_format_row(row, widths))
    print(separator)


def _format_row(values: Sequence[str], widths: Sequence[int]) -> str:
    return "|".join(f" {value:<{width}} " for value, width in zip(values, widths))


if __name__ == "__main__":
    raise SystemExit(main())
