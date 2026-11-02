"""Dataset registry."""

from __future__ import annotations

from collections.abc import Callable

from scp.data.dataset import ChangeDataset
from scp.data.folder import (
    FolderLayout,
    FolderLayoutChangeDataset,
    SemanticPairFolderChangeDataset,
    SemanticPairLayout,
)
from scp.utils import DatasetConfig

DatasetFactory = Callable[..., ChangeDataset]


class DatasetRegistry:
    """Register and instantiate dataset loaders without importing datasets at module import."""

    def __init__(self) -> None:
        self._factories: dict[str, DatasetFactory] = {}

    def register(self, name: str, factory: DatasetFactory) -> None:
        """Register a dataset factory under a unique name."""
        if not name:
            raise ValueError("dataset name must be non-empty")
        if name in self._factories:
            raise KeyError(f"dataset already registered: {name}")
        self._factories[name] = factory

    def get(self, name: str) -> DatasetFactory:
        """Return a registered dataset factory."""
        try:
            return self._factories[name]
        except KeyError as error:
            raise KeyError(f"unknown dataset: {name}") from error

    def create(self, name: str, *args: object, **kwargs: object) -> ChangeDataset:
        """Instantiate a registered dataset factory."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered dataset names in sorted order."""
        return tuple(sorted(self._factories))


def _folder_factory(name: str, layout: FolderLayout) -> DatasetFactory:
    def create_dataset(config: DatasetConfig) -> ChangeDataset:
        return FolderLayoutChangeDataset(
            name=name,
            root=config.path,
            layout=layout,
            patching=config.patching,
        )

    return create_dataset


DATASET_REGISTRY = DatasetRegistry()
DATASET_REGISTRY.register(
    "levir-256", _folder_factory("levir-256", FolderLayout("A", "B", "label", ("train", "val", "test")))
)
DATASET_REGISTRY.register(
    "whu-256",
    _folder_factory("whu-256", FolderLayout("A", "B", "label", ("train", "val", "test"))),
)
DATASET_REGISTRY.register(
    "dsifn-256", _folder_factory("dsifn-256", FolderLayout("t1", "t2", "mask", ("train", "val", "test")))
)
DATASET_REGISTRY.register(
    "sysu-256", _folder_factory("sysu-256", FolderLayout("time1", "time2", "label", ("train", "val", "test")))
)
DATASET_REGISTRY.register(
    "s2looking-256",
    _folder_factory("s2looking-256", FolderLayout("Image1", "Image2", "label", ("train", "val", "test"))),
)
DATASET_REGISTRY.register(
    "clcd-256",
    _folder_factory("clcd-256", FolderLayout("time1", "time2", "label", ("train", "val", "test"))),
)
DATASET_REGISTRY.register(
    "levirplus-256",
    _folder_factory("levirplus-256", FolderLayout("A", "B", "label", ("train", "test"))),
)
DATASET_REGISTRY.register(
    "egy-256",
    _folder_factory(
        "egy-256",
        FolderLayout("A", "B", "label", ("source", "target_mansoura", "target_tiba")),
    ),
)
DATASET_REGISTRY.register(
    "d1-v3-256",
    _folder_factory(
        "d1-v3-256",
        FolderLayout("A", "B", "label", ("source", "hrcus_test", "bandon_ood")),
    ),
)
DATASET_REGISTRY.register(
    "d8-panel-256",
    _folder_factory(
        "d8-panel-256",
        FolderLayout("A", "B", "label", ("hiucd_val", "hrscd", "oscd_test")),
    ),
)
DATASET_REGISTRY.register(
    "d8-hiucd-source-256",
    _folder_factory(
        "d8-hiucd-source-256",
        FolderLayout("A", "B", "label", ("hiucd_model_val",)),
    ),
)
DATASET_REGISTRY.register(
    "second-semantic-256",
    lambda config: SemanticPairFolderChangeDataset(
        name="second-semantic-256",
        root=config.path,
        layout=SemanticPairLayout(
            "rgb/A",
            "rgb/B",
            "sem/A",
            "sem/B",
            ("train", "val", "test"),
        ),
        class_colors=(
            (255, 255, 255),
            (128, 128, 128),
            (0, 128, 0),
            (0, 255, 0),
            (128, 0, 0),
            (0, 0, 255),
        ),
        patching=config.patching,
    ),
)


def get_dataset(name: str, *args: object, **kwargs: object) -> ChangeDataset:
    """Create a dataset from the global registry."""
    return DATASET_REGISTRY.create(name, *args, **kwargs)
