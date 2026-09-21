"""Experiment orchestration helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
import yaml

from scp.canary import CanaryGenerator, get_canary_generator
from scp.data import ChangeDataset, ImagePair, get_dataset
from scp.eval import (
    canary_precision,
    compute_patch_eval_label,
    counterfactual_canary_metrics,
)
from scp.models import CDModelAdapter, get_model_adapter
from scp.signals import SIGNAL_REGISTRY, PassiveSignal, get_signal
from scp.utils import AppConfig, ConfigError

CsvValue: TypeAlias = str | float | int
CsvRow: TypeAlias = dict[str, CsvValue]
CANARY_SCORE_COLUMNS = [
    "pair_id",
    "canary_type",
    "canary_index",
    "canary_area_fraction",
    "canary_recall",
    "canary_precision",
    "canary_mean_response",
    "original_canary_recall",
    "canary_recall_delta",
    "original_canary_mean_response",
    "canary_mean_response_delta",
    "canary_new_activation_fraction",
    "canary_deactivation_fraction",
]
CANARY_UNIT_INTERVAL_COLUMNS = (
    "canary_area_fraction",
    "canary_recall",
    "canary_precision",
    "canary_mean_response",
    "original_canary_recall",
    "original_canary_mean_response",
    "canary_new_activation_fraction",
    "canary_deactivation_fraction",
)
CANARY_SIGNED_COLUMNS = (
    "canary_recall_delta",
    "canary_mean_response_delta",
)


def generate_probability_maps(
    config: AppConfig,
    model_name: str,
    dataset_name: str,
    split: str,
    config_path: Path | str | None = None,
    limit: int | None = None,
    adapter: CDModelAdapter | None = None,
    dataset: ChangeDataset | None = None,
) -> Path:
    """Generate and save model probability maps for a dataset split.

    Args:
        config: Resolved application configuration.
        model_name: Registered model adapter name.
        dataset_name: Registered dataset name.
        split: Dataset split to process.
        config_path: Optional source YAML path for traceability.
        limit: Optional maximum number of image pairs to process.
        adapter: Optional injected adapter for tests or custom runners.
        dataset: Optional injected dataset for tests or custom runners.

    Returns:
        Directory containing `.npy` probability maps and metadata files.

    Raises:
        ConfigError: If model, dataset, or limit inputs are invalid.
    """
    if limit is not None and limit < 0:
        raise ConfigError("limit must be non-negative when provided")
    output_dir = config.paths.results_root / "prob_maps" / model_name / dataset_name / split
    output_dir.mkdir(parents=True, exist_ok=True)

    active_dataset = dataset or _build_dataset(config, dataset_name)
    active_adapter = adapter or build_model_adapter(config, model_name)
    count = _write_probability_maps(
        active_dataset,
        active_adapter,
        split,
        output_dir,
        limit,
        config.inference_batch_size,
    )
    _write_run_config(output_dir, config, config_path)
    _write_manifest(output_dir, model_name, dataset_name, split, count, config.seed)
    return output_dir


def compute_passive_scores(
    config: AppConfig,
    model_name: str,
    dataset_name: str,
    split: str,
    config_path: Path | str | None = None,
    limit: int | None = None,
    adapter: CDModelAdapter | None = None,
    dataset: ChangeDataset | None = None,
) -> Path:
    """Compute registered passive signal scores for a dataset split.

    Args:
        config: Resolved application configuration.
        model_name: Registered model adapter name.
        dataset_name: Registered dataset name.
        split: Dataset split to process.
        config_path: Optional source YAML path for traceability.
        limit: Optional maximum number of image pairs to process.
        adapter: Optional injected adapter for tests or custom runners.
        dataset: Optional injected dataset for tests or custom runners.

    Returns:
        Directory containing `passive_scores.csv` and `run_config.yaml`.

    Raises:
        ConfigError: If model, dataset, limit, or yielded pair IDs are invalid.
    """
    if limit is not None and limit < 0:
        raise ConfigError("limit must be non-negative when provided")
    output_dir = _silent_failure_output_dir(
        config=config,
        artifact_name="passive_scores",
        model_name=model_name,
        dataset_name=dataset_name,
        split=split,
    )
    active_dataset = dataset or _build_dataset(config, dataset_name)
    active_adapter = adapter or build_model_adapter(config, model_name)
    signals = _registered_signals()
    rows = _passive_score_rows(
        active_dataset,
        active_adapter,
        signals,
        split,
        limit,
        config.inference_batch_size,
    )
    _write_csv(output_dir / "passive_scores.csv", ["pair_id", *SIGNAL_REGISTRY.names()], rows)
    _write_run_config(output_dir, config, config_path)
    return output_dir


def compute_eval_labels(
    config: AppConfig,
    model_name: str,
    dataset_name: str,
    split: str,
    config_path: Path | str | None = None,
    limit: int | None = None,
    adapter: CDModelAdapter | None = None,
    dataset: ChangeDataset | None = None,
) -> Path:
    """Compute GT-based offline evaluation labels for a dataset split.

    Args:
        config: Resolved application configuration.
        model_name: Registered model adapter name.
        dataset_name: Registered dataset name.
        split: Dataset split to process.
        config_path: Optional source YAML path for traceability.
        limit: Optional maximum number of image pairs to process.
        adapter: Optional injected adapter for tests or custom runners.
        dataset: Optional injected dataset for tests or custom runners.

    Returns:
        Directory containing `eval_labels.csv` and `run_config.yaml`.

    Raises:
        ConfigError: If inputs are invalid or prediction and mask streams misalign.
    """
    if limit is not None and limit < 0:
        raise ConfigError("limit must be non-negative when provided")
    output_dir = _silent_failure_output_dir(
        config=config,
        artifact_name="eval_labels",
        model_name=model_name,
        dataset_name=dataset_name,
        split=split,
    )
    active_dataset = dataset or _build_dataset(config, dataset_name)
    active_adapter = adapter or build_model_adapter(config, model_name)
    rows = _eval_label_rows(
        active_dataset,
        active_adapter,
        split,
        config.threshold,
        limit,
        config.inference_batch_size,
    )
    _write_csv(
        output_dir / "eval_labels.csv",
        [
            "pair_id",
            "gt_change_density",
            "pred_change_density",
            "fn_rate",
            "empty_mask_failure",
            "low_response_failure",
        ],
        rows,
    )
    _write_run_config(output_dir, config, config_path)
    return output_dir


def compute_canary_scores(
    config: AppConfig,
    model_name: str,
    dataset_name: str,
    split: str,
    config_path: Path | str | None = None,
    limit: int | None = None,
    adapter: CDModelAdapter | None = None,
    dataset: ChangeDataset | None = None,
) -> Path:
    """Compute active canary probing scores for a dataset split.

    Args:
        config: Resolved application configuration.
        model_name: Registered model adapter name.
        dataset_name: Registered dataset name.
        split: Dataset split to process.
        config_path: Optional source YAML path for traceability.
        limit: Optional maximum number of image pairs to process.
        adapter: Optional injected adapter for tests or custom runners.
        dataset: Optional injected dataset for tests or custom runners.

    Returns:
        Directory containing `canary_scores.csv` and `run_config.yaml`. The CSV
        retains absolute post-intervention scores and adds original-pair,
        post-minus-pre, and activation-transition metrics.

    Raises:
        ConfigError: If inputs are invalid or predictions are malformed.
    """
    if limit is not None and limit < 0:
        raise ConfigError("limit must be non-negative when provided")
    output_dir = _silent_failure_output_dir(
        config=config,
        artifact_name="canary_scores",
        model_name=model_name,
        dataset_name=dataset_name,
        split=split,
    )
    active_dataset = dataset or _build_dataset(config, dataset_name)
    active_adapter = adapter or build_model_adapter(config, model_name)
    canary_generators = _canary_generators(config)
    rows = _canary_score_rows(
        dataset=active_dataset,
        adapter=active_adapter,
        canary_generators=canary_generators,
        split=split,
        threshold=config.threshold,
        canaries_per_pair=config.canary.canaries_per_pair,
        seed=config.seed,
        limit=limit,
        batch_size=config.inference_batch_size,
    )
    _write_csv(output_dir / "canary_scores.csv", CANARY_SCORE_COLUMNS, rows)
    _write_run_config(output_dir, config, config_path)
    return output_dir


def _build_dataset(config: AppConfig, dataset_name: str) -> ChangeDataset:
    dataset_config = config.datasets.get(dataset_name)
    if dataset_config is None:
        raise ConfigError(f"dataset is not configured: {dataset_name}")
    return get_dataset(dataset_name, dataset_config)


def build_model_adapter(config: AppConfig, model_name: str) -> CDModelAdapter:
    """Construct the registered adapter for `model_name` from resolved config.

    The returned adapter lazily loads its checkpoint on first `predict`/
    `predict_batch` call and caches it for its lifetime, so callers driving
    multiple pipeline stages for the same model should build one adapter and
    reuse it across stages instead of calling this once per stage.

    Args:
        config: Resolved application configuration.
        model_name: Registered model adapter name.

    Returns:
        A lazily-initialized model adapter.

    Raises:
        ConfigError: If `model_name` is not a registered model.
    """
    if model_name == "bit":
        model_config = config.models.bit
        return get_model_adapter(
            "bit",
            checkpoint_path=model_config.checkpoint,
            repo_path=config.paths.bit_repo,
            device=config.device,
            net_g=model_config.net_g,
            image_size=model_config.image_size,
        )
    if model_name == "changeformer":
        model_config = config.models.changeformer
        return get_model_adapter(
            "changeformer",
            checkpoint_path=model_config.checkpoint,
            repo_path=config.paths.changeformer_repo,
            device=config.device,
            net_g=model_config.net_g,
            embed_dim=model_config.embed_dim,
            image_size=model_config.image_size,
        )
    if model_name == "cdmamba":
        model_config = config.models.cdmamba
        return get_model_adapter(
            "cdmamba",
            checkpoint_path=model_config.checkpoint,
            repo_path=config.paths.cdmamba_repo,
            device=config.device,
            config_path=model_config.config_path,
        )
    if model_name == "tinycd":
        model_config = config.models.tinycd
        return get_model_adapter(
            "tinycd",
            checkpoint_path=model_config.checkpoint,
            repo_path=config.paths.tinycd_repo,
            device=config.device,
            backbone=model_config.backbone,
            output_layer_backbone=model_config.output_layer_backbone,
            pretrained_backbone=model_config.pretrained_backbone,
            freeze_backbone=model_config.freeze_backbone,
        )
    if model_name == "changemamba":
        model_config = config.models.changemamba
        return get_model_adapter(
            "changemamba",
            checkpoint_path=model_config.checkpoint,
            repo_path=config.paths.changemamba_repo,
            device=config.device,
            cfg_path=model_config.cfg_path,
        )
    raise ConfigError(f"model is not configured: {model_name}")


def _write_probability_maps(
    dataset: ChangeDataset,
    adapter: CDModelAdapter,
    split: str,
    output_dir: Path,
    limit: int | None,
    batch_size: int,
) -> int:
    count = 0
    for pair_batch in _iter_pair_batches(dataset, split, limit, batch_size):
        predictions = _predict_in_chunks(adapter, pair_batch, batch_size)
        for pair, prediction in zip(pair_batch, predictions):
            if pair.pair_id is None:
                raise ConfigError("dataset yielded a pair without pair_id")
            probability_map = np.asarray(prediction, dtype=np.float32)
            np.save(output_dir / f"{pair.pair_id}.npy", probability_map)
            count += 1
    return count


def _silent_failure_output_dir(
    config: AppConfig,
    artifact_name: str,
    model_name: str,
    dataset_name: str,
    split: str,
) -> Path:
    output_dir = (
        config.paths.results_root
        / "silent_failure"
        / artifact_name
        / model_name
        / dataset_name
        / split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _registered_signals() -> tuple[PassiveSignal, ...]:
    return tuple(get_signal(name) for name in SIGNAL_REGISTRY.names())


def _iter_pair_batches(
    dataset: ChangeDataset,
    split: str,
    limit: int | None,
    batch_size: int,
) -> Iterator[list[ImagePair]]:
    if batch_size <= 0:
        raise ConfigError("inference_batch_size must be positive")

    batch: list[ImagePair] = []
    yielded_count = 0
    for pair in dataset.iter_split(split):
        if limit is not None and yielded_count >= limit:
            break
        batch.append(pair)
        yielded_count += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _predict_in_chunks(
    adapter: CDModelAdapter,
    pairs: Sequence[ImagePair],
    batch_size: int,
) -> list[NDArray[np.float32]]:
    if batch_size <= 0:
        raise ConfigError("inference_batch_size must be positive")

    predictions: list[NDArray[np.float32]] = []
    for start_index in range(0, len(pairs), batch_size):
        batch = pairs[start_index : start_index + batch_size]
        batch_predictions = adapter.predict_batch(batch)
        if len(batch_predictions) != len(batch):
            raise ConfigError(
                f"{adapter.name}.predict_batch returned {len(batch_predictions)} predictions "
                f"for {len(batch)} image pairs"
            )
        predictions.extend(batch_predictions)
    return predictions


def _passive_score_rows(
    dataset: ChangeDataset,
    adapter: CDModelAdapter,
    signals: tuple[PassiveSignal, ...],
    split: str,
    limit: int | None,
    batch_size: int,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for pair_batch in _iter_pair_batches(dataset, split, limit, batch_size):
        prediction_pairs: list[ImagePair] = []
        for pair in pair_batch:
            prediction_pairs.extend((pair, _swap_pair(pair)))
        predictions = _predict_in_chunks(adapter, prediction_pairs, batch_size)
        for pair_index, pair in enumerate(pair_batch):
            pair_id = _required_pair_id(pair)
            prediction = _finite_prediction(predictions[pair_index * 2], pair_id)
            swapped_prediction = _finite_prediction(predictions[pair_index * 2 + 1], pair_id)
            row: CsvRow = {"pair_id": pair_id}
            for signal in signals:
                value = float(signal.compute(pair, prediction, swapped_prediction))
                if not np.isfinite(value):
                    raise ConfigError(f"signal '{signal.name}' returned a non-finite value")
                row[signal.name] = value
            rows.append(row)
    return rows


def _eval_label_rows(
    dataset: ChangeDataset,
    adapter: CDModelAdapter,
    split: str,
    threshold: float,
    limit: int | None,
    batch_size: int,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    mask_iterator = dataset.iter_masks(split)
    for pair_batch in _iter_pair_batches(dataset, split, limit, batch_size):
        masks: list[tuple[str, NDArray[np.uint8]]] = []
        for pair in pair_batch:
            pair_id = _required_pair_id(pair)
            mask_pair_id, mask = _next_mask(mask_iterator, pair_id)
            if mask_pair_id != pair_id:
                raise ConfigError(f"mask stream pair_id mismatch: {pair_id} vs {mask_pair_id}")
            masks.append((pair_id, mask))
        predictions = _predict_in_chunks(adapter, pair_batch, batch_size)
        for prediction, (pair_id, mask) in zip(predictions, masks):
            rows.append(
                compute_patch_eval_label(
                    pair_id,
                    _finite_prediction(prediction, pair_id),
                    mask,
                    threshold,
                )
            )
    return rows


def _canary_generators(config: AppConfig) -> tuple[CanaryGenerator, ...]:
    generators: list[CanaryGenerator] = []
    for canary_type in config.canary.enabled_types:
        params: dict[str, Any] = dict(config.canary.generator_params.get(canary_type, {}))
        if canary_type == "building_crop_add":
            params["bank_path"] = config.canary.bank_path
        generators.append(get_canary_generator(canary_type, **params))
    return tuple(generators)


def _canary_score_rows(
    dataset: ChangeDataset,
    adapter: CDModelAdapter,
    canary_generators: tuple[CanaryGenerator, ...],
    split: str,
    threshold: float,
    canaries_per_pair: int,
    seed: int,
    limit: int | None,
    batch_size: int,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for pair_batch in _iter_pair_batches(dataset, split, limit, batch_size):
        canary_items: list[tuple[str, str, int, ImagePair, NDArray[np.bool_]]] = []
        for pair in pair_batch:
            pair_id = _required_pair_id(pair)
            for canary_generator in canary_generators:
                for canary_index in range(canaries_per_pair):
                    rng = _canary_rng(seed, pair_id, canary_generator.name, canary_index)
                    canary_pair, canary_mask = canary_generator.generate(pair, rng)
                    canary_items.append(
                        (
                            pair_id,
                            canary_generator.name,
                            canary_index,
                            canary_pair,
                            canary_mask,
                        )
                    )
        original_count = len(pair_batch)
        predictions = _predict_in_chunks(
            adapter,
            [*pair_batch, *(canary_item[3] for canary_item in canary_items)],
            batch_size,
        )
        original_predictions = predictions[:original_count]
        canary_predictions = predictions[original_count:]
        original_by_pair_id = {
            _required_pair_id(pair): _finite_prediction(prediction, _required_pair_id(pair))
            for pair, prediction in zip(pair_batch, original_predictions)
        }
        for prediction, (pair_id, canary_type, canary_index, _canary_pair, canary_mask) in zip(
            canary_predictions,
            canary_items,
        ):
            rows.append(
                _canary_score_row(
                    pair_id=pair_id,
                    canary_type=canary_type,
                    canary_index=canary_index,
                    original_prediction=original_by_pair_id[pair_id],
                    prediction=_finite_prediction(prediction, pair_id),
                    canary_mask=canary_mask,
                    threshold=threshold,
                )
            )
    return rows


def _canary_rng(
    seed: int,
    pair_id: str,
    canary_type: str,
    canary_index: int,
) -> np.random.Generator:
    seed_material = f"{seed}\0{pair_id}\0{canary_type}\0{canary_index}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    return np.random.default_rng(stable_seed)


def _canary_score_row(
    pair_id: str,
    canary_type: str,
    canary_index: int,
    original_prediction: NDArray[np.float32],
    prediction: NDArray[np.float32],
    canary_mask: NDArray[np.bool_],
    threshold: float,
) -> CsvRow:
    truth_mask = np.asarray(canary_mask, dtype=bool)
    if original_prediction.shape != truth_mask.shape:
        raise ConfigError(
            f"original prediction shape for pair '{pair_id}' does not match canary mask: "
            f"{original_prediction.shape} vs {truth_mask.shape}"
        )
    if prediction.shape != truth_mask.shape:
        raise ConfigError(
            f"prediction shape for pair '{pair_id}' does not match canary mask: "
            f"{prediction.shape} vs {truth_mask.shape}"
        )
    if not np.any(truth_mask):
        raise ConfigError(f"canary '{canary_type}' for pair '{pair_id}' produced an empty mask")

    prediction_mask = prediction >= threshold
    counterfactual_metrics = counterfactual_canary_metrics(
        original_prediction,
        prediction,
        truth_mask,
        threshold,
    )
    row = {
        "pair_id": pair_id,
        "canary_type": canary_type,
        "canary_index": canary_index,
        "canary_area_fraction": float(np.mean(truth_mask)),
        "canary_recall": counterfactual_metrics.post_recall,
        "canary_precision": canary_precision(prediction_mask, truth_mask),
        "canary_mean_response": counterfactual_metrics.post_mean_response,
        "original_canary_recall": counterfactual_metrics.original_recall,
        "canary_recall_delta": counterfactual_metrics.recall_delta,
        "original_canary_mean_response": counterfactual_metrics.original_mean_response,
        "canary_mean_response_delta": counterfactual_metrics.mean_response_delta,
        "canary_new_activation_fraction": counterfactual_metrics.newly_activated_fraction,
        "canary_deactivation_fraction": counterfactual_metrics.newly_deactivated_fraction,
    }
    _validate_canary_row(row)
    return row


def _validate_canary_row(row: CsvRow) -> None:
    for key in (*CANARY_UNIT_INTERVAL_COLUMNS, *CANARY_SIGNED_COLUMNS):
        value = row[key]
        if not isinstance(value, int | float) or not np.isfinite(value):
            raise ConfigError(f"{key} must be finite")
    for key in CANARY_UNIT_INTERVAL_COLUMNS:
        value = float(row[key])
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"{key} must be between 0.0 and 1.0")
    for key in CANARY_SIGNED_COLUMNS:
        value = float(row[key])
        if not -1.0 <= value <= 1.0:
            raise ConfigError(f"{key} must be between -1.0 and 1.0")


def _required_pair_id(pair: Any) -> str:
    pair_id = pair.pair_id
    if pair_id is None:
        raise ConfigError("dataset yielded a pair without pair_id")
    return pair_id


def _swap_pair(pair: ImagePair) -> ImagePair:
    return pair.copy_with(before=pair.after, after=pair.before)


def _finite_prediction(
    prediction: NDArray[np.floating],
    pair_id: str,
) -> NDArray[np.float32]:
    prediction_array = np.asarray(prediction, dtype=np.float32)
    if not np.all(np.isfinite(prediction_array)):
        raise ConfigError(f"prediction for pair '{pair_id}' contains non-finite values")
    return prediction_array


def _next_mask(
    mask_iterator: Any,
    pair_id: str,
) -> tuple[str, NDArray[np.uint8]]:
    try:
        return next(mask_iterator)
    except StopIteration as error:
        raise ConfigError(f"mask stream ended before pair '{pair_id}'") from error


def _write_csv(csv_path: Path, fieldnames: list[str], rows: list[CsvRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_config(
    output_dir: Path,
    config: AppConfig,
    config_path: Path | str | None,
) -> None:
    snapshot = _to_yamlable(config)
    if config_path is not None:
        snapshot["config_path"] = str(config_path)
    (output_dir / "run_config.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=True),
        encoding="utf-8",
    )


def _write_manifest(
    output_dir: Path,
    model_name: str,
    dataset_name: str,
    split: str,
    count: int,
    seed: int,
) -> None:
    manifest = {
        "model": model_name,
        "dataset": dataset_name,
        "split": split,
        "count": count,
        "seed": seed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _to_yamlable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_yamlable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_yamlable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_yamlable(item) for item in value]
    return value
