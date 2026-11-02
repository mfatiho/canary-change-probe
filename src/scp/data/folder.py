"""Folder-layout dataset loaders for remote-sensing change datasets."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from scp.data.dataset import ChangeDataset
from scp.data.pair import ImagePair
from scp.utils import PatchingConfig

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})


@dataclass(frozen=True)
class FolderLayout:
    """Names of split-local folders that contain paired images and masks."""

    before_dir: str
    after_dir: str
    mask_dir: str
    available_splits: tuple[str, ...]


@dataclass(frozen=True)
class SemanticPairLayout:
    """Split-local RGB pairs and before/after semantic-label folders."""

    before_dir: str
    after_dir: str
    before_semantic_dir: str
    after_semantic_dir: str
    available_splits: tuple[str, ...]


@dataclass(frozen=True)
class _PairFiles:
    stem: str
    before_path: Path
    after_path: Path


@dataclass(frozen=True)
class _PatchWindow:
    row: int
    col: int
    size: int


class FolderLayoutChangeDataset(ChangeDataset):
    """Lazy loader for datasets organized as split/before-after-mask folders."""

    def __init__(
        self,
        name: str,
        root: Path | str,
        layout: FolderLayout,
        patching: PatchingConfig | None = None,
    ) -> None:
        self._name = name
        self._root = Path(root)
        self._layout = layout
        self._patching = patching

    @property
    def name(self) -> str:
        """Human-readable dataset identifier."""
        return self._name

    @property
    def available_splits(self) -> tuple[str, ...]:
        """Return available split names in the configured deterministic order."""
        return self._layout.available_splits

    def __len__(self) -> int:
        """Return the number of whole-image pairs across available splits."""
        return sum(1 for split in self.available_splits for _ in self._scan_split(split))

    def __getitem__(self, index: int) -> ImagePair:
        """Return a pair by index over the first available split."""
        if index < 0:
            raise IndexError(index)
        for current_index, pair in enumerate(self.iter_split(self.available_splits[0])):
            if current_index == index:
                return pair
        raise IndexError(index)

    def iter_split(self, split: str) -> Iterator[ImagePair]:
        """Yield image pairs for a split, optionally tiled into deterministic patches."""
        for files in self._scan_split(split):
            before = _load_rgb(files.before_path)
            after = _load_rgb(files.after_path)
            _validate_pair_spatial_match(files.stem, before.shape, after.shape)
            for patch_id, before_patch, after_patch in self._iter_arrays(
                files.stem, before, after
            ):
                yield ImagePair(before=before_patch, after=after_patch, pair_id=patch_id)

    def iter_masks(self, split: str) -> Iterator[tuple[str, NDArray[np.uint8]]]:
        """Yield binarized masks separately from image-pair prediction streams.

        Reads only image headers (not pixel data) for the before/after spatial
        validation, since callers that also drive `iter_split` for the same
        split (`build_canary_bank`, `compute_eval_labels`) would otherwise pay
        for decoding every before/after image twice.
        """
        pair_files = list(self._scan_split(split))
        mask_dir = self._root / split / self._layout.mask_dir
        _validate_directory(mask_dir, "mask")
        mask_files = _files_by_stem(mask_dir)
        _validate_stem_sets(
            {files.stem for files in pair_files},
            set(mask_files),
            "pair",
            "mask",
        )
        for files in pair_files:
            before_shape = _rgb_image_shape(files.before_path)
            after_shape = _rgb_image_shape(files.after_path)
            mask = _load_mask(mask_files[files.stem])
            _validate_pair_spatial_match(files.stem, before_shape, after_shape)
            _validate_mask_spatial_match(files.stem, before_shape[:2], mask)
            yield from _iter_mask_patches(files.stem, before_shape[:2], mask, self._patching)

    def _scan_split(self, split: str) -> Iterator[_PairFiles]:
        self._validate_split(split)
        split_root = self._root / split
        before_dir = split_root / self._layout.before_dir
        after_dir = split_root / self._layout.after_dir
        _validate_directory(before_dir, "before")
        _validate_directory(after_dir, "after")

        before_files = _files_by_stem(before_dir)
        after_files = _files_by_stem(after_dir)
        _validate_stem_sets(set(before_files), set(after_files), "before", "after")
        for stem in sorted(before_files):
            yield _PairFiles(stem, before_files[stem], after_files[stem])

    def _validate_split(self, split: str) -> None:
        if split not in self.available_splits:
            choices = ", ".join(self.available_splits)
            raise ValueError(f"split '{split}' is not available for {self.name}: {choices}")

    def _iter_arrays(
        self,
        stem: str,
        before: NDArray[np.uint8],
        after: NDArray[np.uint8],
    ) -> Iterator[tuple[str, NDArray[np.uint8], NDArray[np.uint8]]]:
        if self._patching is None or not self._patching.enabled:
            yield stem, before, after
            return
        for window in _patch_windows(before.shape[:2], self._patching):
            patch_id = f"{stem}_r{window.row}_c{window.col}"
            row_end = window.row + window.size
            col_end = window.col + window.size
            yield (
                patch_id,
                before[window.row : row_end, window.col : col_end],
                after[window.row : row_end, window.col : col_end],
            )


class SemanticPairFolderChangeDataset(FolderLayoutChangeDataset):
    """Derive binary change masks from paired semantic-label rasters."""

    def __init__(
        self,
        name: str,
        root: Path | str,
        layout: SemanticPairLayout,
        class_colors: tuple[tuple[int, int, int], ...],
        patching: PatchingConfig | None = None,
    ) -> None:
        if not class_colors:
            raise ValueError("class_colors must include at least one RGB class")
        pair_layout = FolderLayout(
            before_dir=layout.before_dir,
            after_dir=layout.after_dir,
            mask_dir=layout.before_semantic_dir,
            available_splits=layout.available_splits,
        )
        super().__init__(name=name, root=root, layout=pair_layout, patching=patching)
        self._semantic_layout = layout
        self._class_colors = np.asarray(class_colors, dtype=np.int32)

    def iter_masks(self, split: str) -> Iterator[tuple[str, NDArray[np.uint8]]]:
        """Yield masks whose positive pixels changed semantic class."""
        pair_files = list(self._scan_split(split))
        split_root = self._root / split
        before_dir = split_root / self._semantic_layout.before_semantic_dir
        after_dir = split_root / self._semantic_layout.after_semantic_dir
        _validate_directory(before_dir, "before semantic mask")
        _validate_directory(after_dir, "after semantic mask")
        before_files = _files_by_stem(before_dir)
        after_files = _files_by_stem(after_dir)
        pair_stems = {files.stem for files in pair_files}
        _validate_stem_sets(pair_stems, set(before_files), "pair", "before semantic mask")
        _validate_stem_sets(pair_stems, set(after_files), "pair", "after semantic mask")

        for files in pair_files:
            image_shape = _rgb_image_shape(files.before_path)
            before_semantic = _load_rgb(before_files[files.stem])
            after_semantic = _load_rgb(after_files[files.stem])
            _validate_pair_spatial_match(
                files.stem,
                before_semantic.shape,
                after_semantic.shape,
            )
            _validate_pair_spatial_match(files.stem, image_shape, before_semantic.shape)
            before_classes = _nearest_color_classes(before_semantic, self._class_colors)
            after_classes = _nearest_color_classes(after_semantic, self._class_colors)
            change_mask = (before_classes != after_classes).astype(np.uint8)
            yield from _iter_mask_patches(
                files.stem,
                image_shape[:2],
                change_mask,
                self._patching,
            )


def _files_by_stem(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in files:
            raise FileExistsError(f"duplicate image stem '{path.stem}' in {directory}")
        files[path.stem] = path
    return files


def _validate_directory(directory: Path, role: str) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing {role} directory: {directory}")


def _validate_stem_sets(
    left_stems: set[str],
    right_stems: set[str],
    left_role: str,
    right_role: str,
) -> None:
    left_only = sorted(left_stems - right_stems)
    right_only = sorted(right_stems - left_stems)
    if not left_only and not right_only:
        return
    details = []
    if left_only:
        label = _left_only_stem_label(left_role, right_role)
        details.append(f"{label}: {_format_stems(left_only)}")
    if right_only:
        details.append(f"{right_role}-only stems: {_format_stems(right_only)}")
    raise FileNotFoundError("; ".join(details))


def _left_only_stem_label(left_role: str, right_role: str) -> str:
    if left_role == "pair" and right_role == "mask":
        return "missing mask stems"
    return f"{left_role}-only stems"


def _format_stems(stems: list[str]) -> str:
    return ", ".join(stems)


def _load_rgb(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _rgb_image_shape(path: Path) -> tuple[int, int, int]:
    """Return the (height, width, 3) shape `_load_rgb` would produce.

    Reads only the image header via `Image.size`, not the pixel data, since
    `_load_rgb` always forces a 3-channel RGB conversion regardless of the
    source image's native mode.
    """
    with Image.open(path) as image:
        width, height = image.size
    return height, width, 3


def _load_mask(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    return (mask > 0).astype(np.uint8)


def _nearest_color_classes(
    semantic_rgb: NDArray[np.uint8],
    class_colors: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Quantize possibly augmented semantic colors without integer overflow."""
    pixels = semantic_rgb.reshape(-1, 3).astype(np.int32)
    squared_distances = np.square(pixels[:, None, :] - class_colors[None, :, :]).sum(axis=2)
    return squared_distances.argmin(axis=1).reshape(semantic_rgb.shape[:2])


def _iter_mask_patches(
    stem: str,
    spatial_shape: tuple[int, int],
    mask: NDArray[np.uint8],
    patching: PatchingConfig | None,
) -> Iterator[tuple[str, NDArray[np.uint8]]]:
    if patching is None or not patching.enabled:
        yield stem, mask
        return
    for window in _patch_windows(spatial_shape, patching):
        patch_id = f"{stem}_r{window.row}_c{window.col}"
        row_end = window.row + window.size
        col_end = window.col + window.size
        yield patch_id, mask[window.row : row_end, window.col : col_end]


def _validate_pair_spatial_match(
    stem: str,
    before_shape: tuple[int, ...],
    after_shape: tuple[int, ...],
) -> None:
    if before_shape != after_shape:
        raise ValueError(f"before/after shape mismatch for '{stem}': {before_shape} vs {after_shape}")


def _validate_mask_spatial_match(
    stem: str,
    image_shape: tuple[int, int],
    mask: NDArray[np.uint8],
) -> None:
    if image_shape != mask.shape:
        raise ValueError(f"mask shape mismatch for '{stem}': {mask.shape} vs {image_shape}")


def _patch_windows(
    spatial_shape: tuple[int, int],
    patching: PatchingConfig,
) -> Iterator[_PatchWindow]:
    height, width = spatial_shape
    patch_size = patching.patch_size
    stride = patching.stride or patch_size
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"patch_size {patch_size} exceeds image spatial shape {(height, width)}"
        )
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            yield _PatchWindow(row=row, col=col, size=patch_size)
