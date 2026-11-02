import numpy as np
import pytest

from scp.data import ChangeDataset, DatasetRegistry, ImagePair


class DemoDataset(ChangeDataset):
    """Small in-memory dataset used to exercise the registry contract."""

    def __init__(self, pair: ImagePair) -> None:
        self._pair = pair

    @property
    def name(self) -> str:
        return "demo"

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> ImagePair:
        if index != 0:
            raise IndexError(index)
        return self._pair


def make_pair() -> ImagePair:
    before = np.zeros((4, 4, 3), dtype=np.float32)
    after = np.ones((4, 4, 3), dtype=np.float32)
    return ImagePair(before=before, after=after, pair_id="p0")


def test_image_pair_validates_matching_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ImagePair(before=np.zeros((2, 2)), after=np.zeros((3, 3)))


def test_image_pair_exposes_spatial_shape() -> None:
    assert make_pair().spatial_shape == (4, 4)


def test_image_pair_copy_with_revalidates_data() -> None:
    pair = make_pair()
    changed = pair.copy_with(after=np.full((4, 4, 3), 7.0, dtype=np.float32))

    assert changed.pair_id == "p0"
    assert float(changed.after.max()) == 7.0


def test_dataset_registry_registers_and_creates_dataset() -> None:
    registry = DatasetRegistry()
    registry.register("demo", DemoDataset)

    dataset = registry.create("demo", make_pair())

    assert dataset.name == "demo"
    assert len(dataset) == 1


def test_dataset_registry_fails_on_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown"):
        DatasetRegistry().get("unknown")
