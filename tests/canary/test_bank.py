import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from scp.canary.bank import build_canary_bank, load_canary_bank
from scp.data import ImagePair


class TinyMaskDataset:
    """Small split-stream dataset for canary-bank tests."""

    @property
    def name(self) -> str:
        return "tiny"

    def iter_split(self, split: str) -> Iterator[ImagePair]:
        if split != "train":
            raise ValueError(split)
        before = np.zeros((8, 8, 3), dtype=np.uint8)
        after = before.copy()
        after[1:4, 1:4] = [50, 60, 70]
        after[5:7, 5:8] = [100, 110, 120]
        yield ImagePair(before=before, after=after, pair_id="pair-a")

    def iter_masks(self, split: str) -> Iterator[tuple[str, np.ndarray]]:
        if split != "train":
            raise ValueError(split)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:4, 1:4] = 1
        mask[5:7, 5:8] = 1
        yield "pair-a", mask


def test_build_canary_bank_extracts_components_and_manifest(tmp_path: Path) -> None:
    bank_path = tmp_path / "tiny_train.npz"

    manifest = build_canary_bank(
        dataset=TinyMaskDataset(),
        dataset_name="tiny",
        split="train",
        output_path=bank_path,
        min_area=6,
        max_area=9,
    )

    loaded = load_canary_bank(bank_path)
    manifest_json = json.loads(bank_path.with_suffix(".json").read_text(encoding="utf-8"))

    assert manifest.count == 2
    assert manifest_json["dataset"] == "tiny"
    assert manifest_json["split"] == "train"
    assert manifest_json["count"] == 2
    assert loaded.crops[0].shape == (3, 3, 3)
    assert loaded.masks[0].shape == (3, 3)
    assert loaded.metadata[0]["pair_id"] == "pair-a"
    assert loaded.metadata[0]["area"] == 9
    np.testing.assert_array_equal(loaded.crops[0][loaded.masks[0]], np.array([[50, 60, 70]] * 9))
    np.testing.assert_array_equal(loaded.crops[1][loaded.masks[1]], np.array([[100, 110, 120]] * 6))


def test_build_canary_bank_rejects_misaligned_mask_stream(tmp_path: Path) -> None:
    class BadMaskDataset(TinyMaskDataset):
        def iter_masks(self, split: str) -> Iterator[tuple[str, np.ndarray]]:
            yield "wrong-id", np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(ValueError, match="mask stream pair_id mismatch"):
        build_canary_bank(
            dataset=BadMaskDataset(),
            dataset_name="tiny",
            split="train",
            output_path=tmp_path / "bad.npz",
            min_area=1,
            max_area=64,
        )
