from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scp.data.folder import FolderLayout, FolderLayoutChangeDataset
from scp.utils import PatchingConfig


LAYOUTS = {
    "levir-256": FolderLayout("A", "B", "label", ("train", "val", "test")),
    "whu-256": FolderLayout("A", "B", "label", ("train", "test")),
    "dsifn-256": FolderLayout("t1", "t2", "mask", ("train", "val", "test")),
    "sysu-256": FolderLayout("time1", "time2", "label", ("train", "val", "test")),
    "s2looking-256": FolderLayout("Image1", "Image2", "label", ("train", "val", "test")),
}


def _write_image(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values).save(path)


def _make_layout_tree(root: Path, layout: FolderLayout, split: str = "test") -> None:
    before = np.full((4, 4, 3), 10, dtype=np.uint8)
    after = np.full((4, 4, 3), 20, dtype=np.uint8)
    mask = np.array(
        [
            [0, 2, 0, 255],
            [0, 0, 3, 0],
            [4, 0, 0, 0],
            [0, 0, 0, 5],
        ],
        dtype=np.uint8,
    )
    _write_image(root / split / layout.before_dir / "scene_01.png", before)
    _write_image(root / split / layout.after_dir / "scene_01.jpg", after)
    _write_image(root / split / layout.mask_dir / "scene_01.tif", mask)


@pytest.mark.parametrize(("dataset_name", "layout"), LAYOUTS.items())
def test_folder_layout_loads_pairs_and_binarized_masks(
    tmp_path: Path, dataset_name: str, layout: FolderLayout
) -> None:
    _make_layout_tree(tmp_path, layout)
    dataset = FolderLayoutChangeDataset(dataset_name, tmp_path, layout)

    pairs = list(dataset.iter_split("test"))
    masks = list(dataset.iter_masks("test"))

    assert dataset.available_splits == layout.available_splits
    assert pairs[0].pair_id == "scene_01"
    assert pairs[0].before.dtype == np.uint8
    assert int(pairs[0].before[0, 0, 0]) == 10
    assert int(pairs[0].after[0, 0, 0]) == 20
    assert masks[0][0] == "scene_01"
    assert masks[0][1].dtype == np.uint8
    assert set(np.unique(masks[0][1]).tolist()) == {0, 1}


def test_folder_layout_rejects_unavailable_split(tmp_path: Path) -> None:
    layout = LAYOUTS["whu-256"]
    _make_layout_tree(tmp_path, layout, split="test")
    dataset = FolderLayoutChangeDataset("whu-256", tmp_path, layout)

    with pytest.raises(ValueError, match="split 'val' is not available"):
        list(dataset.iter_split("val"))


def test_folder_layout_raises_when_mask_stem_is_missing(tmp_path: Path) -> None:
    layout = LAYOUTS["levir-256"]
    _make_layout_tree(tmp_path, layout)
    (tmp_path / "test" / layout.mask_dir / "scene_01.tif").unlink()
    dataset = FolderLayoutChangeDataset("levir-256", tmp_path, layout)

    with pytest.raises(FileNotFoundError, match="missing mask stems.*scene_01"):
        list(dataset.iter_masks("test"))


def test_folder_layout_allows_maskless_pair_stream(tmp_path: Path) -> None:
    layout = FolderLayout("A", "B", "label", ("test",))
    before = np.full((4, 4, 3), 10, dtype=np.uint8)
    after = np.full((4, 4, 3), 20, dtype=np.uint8)
    _write_image(tmp_path / "test" / layout.before_dir / "scene_01.png", before)
    _write_image(tmp_path / "test" / layout.after_dir / "scene_01.png", after)
    dataset = FolderLayoutChangeDataset("levir-256", tmp_path, layout)

    pairs = list(dataset.iter_split("test"))

    assert len(dataset) == 1
    assert pairs[0].pair_id == "scene_01"
    np.testing.assert_array_equal(pairs[0].before, before)
    with pytest.raises(FileNotFoundError, match="missing mask directory"):
        list(dataset.iter_masks("test"))


def test_folder_layout_raises_when_after_has_extra_stem(tmp_path: Path) -> None:
    layout = LAYOUTS["levir-256"]
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    _write_image(tmp_path / "test" / layout.before_dir / "a.png", image)
    _write_image(tmp_path / "test" / layout.after_dir / "a.png", image)
    _write_image(tmp_path / "test" / layout.after_dir / "b.png", image)
    _write_image(tmp_path / "test" / layout.mask_dir / "a.png", np.zeros((2, 2), dtype=np.uint8))
    dataset = FolderLayoutChangeDataset("levir-256", tmp_path, layout)

    with pytest.raises(FileNotFoundError, match="after-only stems.*b"):
        list(dataset.iter_split("test"))


def test_folder_layout_raises_when_mask_stems_do_not_cover_pairs(tmp_path: Path) -> None:
    layout = LAYOUTS["levir-256"]
    _make_layout_tree(tmp_path, layout)
    _write_image(tmp_path / "test" / layout.before_dir / "scene_02.png", np.zeros((4, 4, 3), dtype=np.uint8))
    _write_image(tmp_path / "test" / layout.after_dir / "scene_02.png", np.zeros((4, 4, 3), dtype=np.uint8))
    dataset = FolderLayoutChangeDataset("levir-256", tmp_path, layout)

    with pytest.raises(FileNotFoundError, match="missing mask stems.*scene_02"):
        list(dataset.iter_masks("test"))


def test_folder_layout_patching_yields_stable_patch_ids(tmp_path: Path) -> None:
    layout = LAYOUTS["dsifn-256"]
    before = np.arange(5 * 5 * 3, dtype=np.uint8).reshape(5, 5, 3)
    after = np.flip(before, axis=1).copy()
    mask = np.zeros((5, 5), dtype=np.uint8)
    _write_image(tmp_path / "test" / layout.before_dir / "tile.png", before)
    _write_image(tmp_path / "test" / layout.after_dir / "tile.png", after)
    _write_image(tmp_path / "test" / layout.mask_dir / "tile.png", mask)
    dataset = FolderLayoutChangeDataset(
        "dsifn-256",
        tmp_path,
        layout,
        patching=PatchingConfig(enabled=True, patch_size=3, stride=2),
    )

    pairs = list(dataset.iter_split("test"))

    assert [pair.pair_id for pair in pairs] == [
        "tile_r0_c0",
        "tile_r0_c2",
        "tile_r2_c0",
        "tile_r2_c2",
    ]
    assert all(pair.spatial_shape == (3, 3) for pair in pairs)
    np.testing.assert_array_equal(pairs[0].before, before[:3, :3])
