"""Build and load source-split canary crop banks from ground-truth masks.

Banks must be built from source training splits only. Target-domain ground truth
is reserved for evaluation and must not be used to construct canary probes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from scp.data import ChangeDataset, ImagePair


@dataclass(frozen=True)
class CanaryBank:
    """Loaded bank arrays and per-crop metadata."""

    crops: tuple[NDArray[np.floating] | NDArray[np.integer], ...]
    masks: tuple[NDArray[np.bool_], ...]
    metadata: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CanaryBankManifest:
    """Summary written next to a generated canary bank."""

    dataset: str
    split: str
    count: int
    scanned_pairs: int
    area: dict[str, float | int | None]
    height: dict[str, float | int | None]
    width: dict[str, float | int | None]


@dataclass(frozen=True)
class _ComponentCrop:
    crop: NDArray[np.floating] | NDArray[np.integer]
    mask: NDArray[np.bool_]
    metadata: dict[str, Any]


def build_canary_bank(
    dataset: ChangeDataset,
    dataset_name: str,
    split: str,
    output_path: Path | str,
    min_area: int = 64,
    max_area: int = 4096,
    limit: int | None = None,
) -> CanaryBankManifest:
    """Extract connected GT components into a compressed crop bank.

    Args:
        dataset: Dataset exposing aligned `iter_split` and `iter_masks` streams.
        dataset_name: Source dataset identifier for manifest traceability.
        split: Source split to scan, normally `train`.
        output_path: `.npz` path to write.
        min_area: Inclusive minimum connected-component area in pixels.
        max_area: Inclusive maximum connected-component area in pixels.
        limit: Optional maximum number of image pairs to scan.

    Returns:
        Manifest summary for the written bank.

    Raises:
        ValueError: If area bounds, limits, pair IDs, or mask alignment are invalid.
    """
    _validate_build_inputs(min_area, max_area, limit)
    crops: list[_ComponentCrop] = []
    scanned_pairs = 0
    mask_iterator = dataset.iter_masks(split)
    for pair in dataset.iter_split(split):
        if limit is not None and scanned_pairs >= limit:
            break
        pair_id = _required_pair_id(pair)
        mask_pair_id, mask = _next_mask(mask_iterator, pair_id)
        if mask_pair_id != pair_id:
            raise ValueError(f"mask stream pair_id mismatch: {pair_id} vs {mask_pair_id}")
        crops.extend(_component_crops(pair, np.asarray(mask), min_area, max_area))
        scanned_pairs += 1

    bank_path = Path(output_path)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    _write_bank(bank_path, crops)
    manifest = _manifest(dataset_name, split, crops, scanned_pairs)
    _write_manifest(bank_path.with_suffix(".json"), manifest)
    return manifest


def load_canary_bank(path: Path | str) -> CanaryBank:
    """Load a canary crop bank written by `build_canary_bank`.

    Args:
        path: `.npz` bank path.

    Returns:
        Crop arrays, binary masks, and JSON metadata entries.
    """
    with np.load(Path(path), allow_pickle=True) as bank:
        crops = tuple(np.asarray(crop) for crop in bank["crops"])
        masks = tuple(np.asarray(mask, dtype=bool) for mask in bank["masks"])
        metadata = tuple(json.loads(str(item)) for item in bank["metadata"])
    return CanaryBank(crops=crops, masks=masks, metadata=metadata)


def _component_crops(
    pair: ImagePair,
    mask: NDArray[np.integer] | NDArray[np.bool_],
    min_area: int,
    max_area: int,
) -> Iterator[_ComponentCrop]:
    if mask.shape != pair.spatial_shape:
        raise ValueError(
            f"mask shape for pair '{_required_pair_id(pair)}' does not match image shape: "
            f"{mask.shape} vs {pair.spatial_shape}"
        )
    labels, component_count = ndimage.label(mask.astype(bool))
    objects = ndimage.find_objects(labels)
    pair_id = _required_pair_id(pair)
    for label_index in range(1, component_count + 1):
        component_slice = objects[label_index - 1]
        if component_slice is None:
            continue
        row_slice, col_slice = component_slice
        component_mask = labels[row_slice, col_slice] == label_index
        area = int(component_mask.sum())
        if area < min_area or area > max_area:
            continue
        crop = pair.after[row_slice, col_slice].copy()
        metadata = {
            "pair_id": pair_id,
            "component_label": label_index,
            "area": area,
            "row_min": int(row_slice.start),
            "row_max": int(row_slice.stop),
            "col_min": int(col_slice.start),
            "col_max": int(col_slice.stop),
            "height": int(component_mask.shape[0]),
            "width": int(component_mask.shape[1]),
        }
        yield _ComponentCrop(crop=crop, mask=component_mask.astype(bool), metadata=metadata)


def _write_bank(path: Path, crops: list[_ComponentCrop]) -> None:
    crop_array = np.empty(len(crops), dtype=object)
    mask_array = np.empty(len(crops), dtype=object)
    metadata_array = np.empty(len(crops), dtype=object)
    for index, component in enumerate(crops):
        crop_array[index] = component.crop
        mask_array[index] = component.mask
        metadata_array[index] = json.dumps(component.metadata, sort_keys=True)
    np.savez_compressed(path, crops=crop_array, masks=mask_array, metadata=metadata_array)


def _manifest(
    dataset_name: str,
    split: str,
    crops: list[_ComponentCrop],
    scanned_pairs: int,
) -> CanaryBankManifest:
    areas = [component.metadata["area"] for component in crops]
    heights = [component.metadata["height"] for component in crops]
    widths = [component.metadata["width"] for component in crops]
    return CanaryBankManifest(
        dataset=dataset_name,
        split=split,
        count=len(crops),
        scanned_pairs=scanned_pairs,
        area=_stats(areas),
        height=_stats(heights),
        width=_stats(widths),
    )


def _stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(np.mean(values)),
    }


def _write_manifest(path: Path, manifest: CanaryBankManifest) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": manifest.dataset,
                "split": manifest.split,
                "count": manifest.count,
                "scanned_pairs": manifest.scanned_pairs,
                "area": manifest.area,
                "height": manifest.height,
                "width": manifest.width,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _validate_build_inputs(min_area: int, max_area: int, limit: int | None) -> None:
    if min_area <= 0:
        raise ValueError("min_area must be positive")
    if max_area < min_area:
        raise ValueError("max_area must be greater than or equal to min_area")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative when provided")


def _required_pair_id(pair: ImagePair) -> str:
    if pair.pair_id is None:
        raise ValueError("dataset yielded a pair without pair_id")
    return pair.pair_id


def _next_mask(
    mask_iterator: Iterator[tuple[str, NDArray[np.uint8]]],
    pair_id: str,
) -> tuple[str, NDArray[np.uint8]]:
    try:
        return next(mask_iterator)
    except StopIteration as error:
        raise ValueError(f"mask stream ended before pair '{pair_id}'") from error
