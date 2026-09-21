"""Lazy wrappers for third-party change-detection backbones."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scp.data import ImagePair
from scp.models.base import CDModelAdapter, ModelLoadError
from scp.utils import ConfigError

BIT_NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
BIT_NORMALIZATION_STD = (0.5, 0.5, 0.5)
IMAGENET_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORMALIZATION_STD = (0.229, 0.224, 0.225)
IMAGENET_PIXEL_NORMALIZATION_MEAN = (123.675, 116.28, 103.53)
IMAGENET_PIXEL_NORMALIZATION_STD = (58.395, 57.12, 57.375)
CHANGE_CLASS_INDEX = 1
CHANGEMAMBA_DEFAULT_CFG_PATH = Path(
    "third_party/ChangeMamba/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
)
CDMAMBA_DEFAULT_CONFIG_PATH = Path("third_party/CDMamba/config/mamba/levir_test_cdmamba.json")


@dataclass
class _TorchModelState:
    torch: ModuleType
    model: Any
    device: Any


def _load_trusted_checkpoint(torch: ModuleType, checkpoint_path: Path, device: Any) -> Any:
    """Load a provenance-verified legacy checkpoint under modern PyTorch.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    The registered upstream checkpoints include legacy NumPy scalars and cannot
    be read by that restricted unpickler. Callers should verify checkpoint
    provenance and integrity before inference.
    """
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


class _LazyTorchAdapter(CDModelAdapter):
    """Base adapter that keeps torch and upstream imports out of import time."""

    def __init__(self, checkpoint_path: Path | str | None = None, device: str = "auto") -> None:
        self._checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self._device_name = device
        self._state: _TorchModelState | None = None

    def predict(
        self,
        pair: ImagePair | NDArray[np.integer] | NDArray[np.floating],
        after: NDArray[np.integer] | NDArray[np.floating] | None = None,
    ) -> NDArray[np.float32]:
        """Predict a float32 change-probability map for an image pair."""
        image_pair = _coerce_pair(pair, after)
        state = self._get_state()
        before_tensor = self._to_input_tensor(image_pair.before, state.torch, state.device)
        after_tensor = self._to_input_tensor(image_pair.after, state.torch, state.device)
        with state.torch.no_grad():
            output = state.model(before_tensor, after_tensor)
        probability_map = self._output_to_probability_map(output, state.torch)
        if probability_map.shape != image_pair.spatial_shape:
            raise ModelLoadError(
                f"{self.name} returned shape {probability_map.shape}, "
                f"expected {image_pair.spatial_shape}"
            )
        return probability_map

    def predict_batch(self, pairs: Sequence[ImagePair]) -> list[NDArray[np.float32]]:
        """Predict float32 probability maps for image pairs with one model forward pass."""
        if not pairs:
            return []
        state = self._get_state()
        before_batch = state.torch.cat(
            [self._to_input_tensor(pair.before, state.torch, state.device) for pair in pairs],
            dim=0,
        )
        after_batch = state.torch.cat(
            [self._to_input_tensor(pair.after, state.torch, state.device) for pair in pairs],
            dim=0,
        )
        with state.torch.inference_mode():
            output = state.model(before_batch, after_batch)

        probability_maps: list[NDArray[np.float32]] = []
        for index, pair in enumerate(pairs):
            single_output = _select_batch_item(output, index)
            probability_map = self._output_to_probability_map(single_output, state.torch)
            if probability_map.shape != pair.spatial_shape:
                raise ModelLoadError(
                    f"{self.name} returned shape {probability_map.shape}, "
                    f"expected {pair.spatial_shape}"
                )
            probability_maps.append(probability_map)
        return probability_maps

    def _get_state(self) -> _TorchModelState:
        if self._state is None:
            self._state = self._load_state()
        return self._state

    def _load_state(self) -> _TorchModelState:
        checkpoint_path = self._validated_checkpoint_path()
        torch = _import_torch()
        device = _resolve_device(torch, self._device_name)
        _ensure_torchvision_models_utils()
        with _UpstreamImportContext(self.repo_path):
            model = self._build_model(torch)
            checkpoint = _load_trusted_checkpoint(torch, checkpoint_path, device)
            model.load_state_dict(self._normalize_state_dict(_unwrap_state_dict(checkpoint)))
            model.to(device)
            model.eval()
        return _TorchModelState(torch=torch, model=model, device=device)

    def _validated_checkpoint_path(self) -> Path:
        if self._checkpoint_path is None:
            raise ConfigError(f"{self.name} checkpoint is not configured")
        if not self._checkpoint_path.is_file():
            raise ConfigError(f"{self.name} checkpoint does not exist: {self._checkpoint_path}")
        return self._checkpoint_path

    @property
    def repo_path(self) -> Path:
        """Return the upstream repository path."""
        raise NotImplementedError

    def _normalize_state_dict(self, state_dict: Any) -> Any:
        """Hook for adapter-specific legacy checkpoint key migrations."""
        return state_dict

    def _build_model(self, torch: ModuleType) -> Any:
        raise NotImplementedError

    def _to_input_tensor(
        self,
        image: NDArray[np.integer] | NDArray[np.floating],
        torch: ModuleType,
        device: Any,
    ) -> Any:
        raise NotImplementedError

    def _output_to_probability_map(self, output: Any, torch: ModuleType) -> NDArray[np.float32]:
        raise NotImplementedError

    @staticmethod
    def _logits_to_probability_map(output: Any, torch: ModuleType) -> NDArray[np.float32]:
        """Convert a binary two-channel logit tensor to a clipped probability map."""
        logits = output[-1] if isinstance(output, list | tuple) else output
        if logits.ndim == 4 and logits.shape[1] > 1:
            probabilities = torch.nn.functional.softmax(logits, dim=1)[:, CHANGE_CLASS_INDEX]
        elif logits.ndim == 4 and logits.shape[1] == 1:
            probabilities = torch.sigmoid(logits[:, 0])
        elif logits.ndim == 3:
            probabilities = torch.sigmoid(logits)
        else:
            raise ModelLoadError(f"unsupported output tensor shape: {tuple(logits.shape)}")
        array = probabilities.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return np.clip(array, 0.0, 1.0).astype(np.float32)


class BITAdapter(_LazyTorchAdapter):
    """Lazy adapter for the BIT_CD third-party repository."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        repo_path: Path | str = Path("third_party/BIT_CD"),
        device: str = "auto",
        net_g: str = "base_transformer_pos_s4_dd8_dedim8",
        image_size: int = 256,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._repo_path = Path(repo_path)
        self._net_g = net_g
        self._image_size = image_size

    @property
    def name(self) -> str:
        """Adapter registry name."""
        return "bit"

    @property
    def repo_path(self) -> Path:
        """Return the upstream repository path."""
        return self._repo_path

    def _build_model(self, torch: ModuleType) -> Any:
        networks = _import_upstream_module("models.networks", self.name)
        args = SimpleNamespace(net_G=self._net_g, gpu_ids=[])
        return networks.define_G(args=args, gpu_ids=[])

    def _to_input_tensor(
        self,
        image: NDArray[np.integer] | NDArray[np.floating],
        torch: ModuleType,
        device: Any,
    ) -> Any:
        return _normalize_image(image, torch, device, BIT_NORMALIZATION_MEAN, BIT_NORMALIZATION_STD)

    def _output_to_probability_map(self, output: Any, torch: ModuleType) -> NDArray[np.float32]:
        return self._logits_to_probability_map(output, torch)


class ChangeFormerAdapter(_LazyTorchAdapter):
    """Lazy adapter for the ChangeFormer third-party repository."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        repo_path: Path | str = Path("third_party/ChangeFormer"),
        device: str = "auto",
        net_g: str = "ChangeFormerV6",
        embed_dim: int = 256,
        image_size: int = 256,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._repo_path = Path(repo_path)
        self._net_g = net_g
        self._embed_dim = embed_dim
        self._image_size = image_size

    @property
    def name(self) -> str:
        """Adapter registry name."""
        return "changeformer"

    @property
    def repo_path(self) -> Path:
        """Return the upstream repository path."""
        return self._repo_path

    def _build_model(self, torch: ModuleType) -> Any:
        networks = _import_upstream_module("models.networks", self.name)
        args = SimpleNamespace(net_G=self._net_g, embed_dim=self._embed_dim, gpu_ids=[])
        return networks.define_G(args=args, gpu_ids=[])

    def _to_input_tensor(
        self,
        image: NDArray[np.integer] | NDArray[np.floating],
        torch: ModuleType,
        device: Any,
    ) -> Any:
        return _normalize_image(image, torch, device, BIT_NORMALIZATION_MEAN, BIT_NORMALIZATION_STD)

    def _output_to_probability_map(self, output: Any, torch: ModuleType) -> NDArray[np.float32]:
        return self._logits_to_probability_map(output, torch)


class TinyCDAdapter(_LazyTorchAdapter):
    """Lazy adapter for the Tiny_model_4_CD third-party repository."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        repo_path: Path | str = Path("third_party/Tiny_model_4_CD"),
        device: str = "auto",
        backbone: str = "efficientnet_b4",
        output_layer_backbone: str = "3",
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._repo_path = Path(repo_path)
        self._backbone = backbone
        self._output_layer_backbone = output_layer_backbone
        self._pretrained_backbone = pretrained_backbone
        self._freeze_backbone = freeze_backbone

    @property
    def name(self) -> str:
        """Adapter registry name."""
        return "tinycd"

    @property
    def repo_path(self) -> Path:
        """Return the upstream repository path."""
        return self._repo_path

    def _normalize_state_dict(self, state_dict: Any) -> Any:
        """Migrate legacy TinyCD checkpoints to the current upstream layout.

        Older upstream revisions wrapped the THIRD mixing block (a plain
        MixingBlock today) in an attention wrapper, nesting its keys under
        `_mixing_mask.2._mixing` (e.g. whu_best.pth); tensors are identical.
        The first blocks are genuine attention blocks whose `._mixing._convmix`
        keys are correct in both layouts and must not be touched.
        """
        if not isinstance(state_dict, dict):
            return state_dict
        legacy_prefix = "_mixing_mask.2._mixing._convmix."
        current_prefix = "_mixing_mask.2._convmix."
        return {
            (current_prefix + key[len(legacy_prefix):] if key.startswith(legacy_prefix) else key): value
            for key, value in state_dict.items()
        }

    def _build_model(self, torch: ModuleType) -> Any:
        module = _import_upstream_module("models.change_classifier", self.name)
        return module.ChangeClassifier(
            bkbn_name=self._backbone,
            pretrained=self._pretrained_backbone,
            output_layer_bkbn=self._output_layer_backbone,
            freeze_backbone=self._freeze_backbone,
        )

    def _to_input_tensor(
        self,
        image: NDArray[np.integer] | NDArray[np.floating],
        torch: ModuleType,
        device: Any,
    ) -> Any:
        return _normalize_image(
            image,
            torch,
            device,
            IMAGENET_NORMALIZATION_MEAN,
            IMAGENET_NORMALIZATION_STD,
        )

    def _output_to_probability_map(self, output: Any, torch: ModuleType) -> NDArray[np.float32]:
        probabilities = output[-1] if isinstance(output, list | tuple) else output
        if probabilities.ndim == 4 and probabilities.shape[1] == 1:
            probabilities = probabilities[:, 0]
        elif probabilities.ndim != 3:
            raise ModelLoadError(f"unsupported output tensor shape: {tuple(probabilities.shape)}")
        array = probabilities.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return np.clip(array, 0.0, 1.0).astype(np.float32)


class ChangeMambaAdapter(_LazyTorchAdapter):
    """Lazy adapter for the ChangeMamba binary change-detection model."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        repo_path: Path | str = Path("third_party/ChangeMamba"),
        device: str = "auto",
        cfg_path: Path | str = CHANGEMAMBA_DEFAULT_CFG_PATH,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._repo_path = Path(repo_path)
        self._cfg_path = Path(cfg_path)

    @property
    def name(self) -> str:
        """Adapter registry name."""
        return "changemamba"

    @property
    def repo_path(self) -> Path:
        """Return the upstream repository path."""
        return self._repo_path

    def _load_state(self) -> _TorchModelState:
        checkpoint_path = self._validated_checkpoint_path()
        torch = _import_torch()
        device = _resolve_device(torch, self._device_name)
        if getattr(device, "type", None) != "cuda":
            raise ModelLoadError("ChangeMamba inference requires CUDA selective-scan kernels.")
        with _UpstreamImportContext(self.repo_path):
            model = self._build_model(torch)
            checkpoint_loader = _import_upstream_module("changedetection.checkpoints", self.name)
            checkpoint_loader.load_model_weights(model, str(checkpoint_path))
            model.to(device)
            model.eval()
        return _TorchModelState(torch=torch, model=model, device=device)

    def _build_model(self, torch: ModuleType) -> Any:
        config_path = self._validated_cfg_path()
        config_module = _import_upstream_module("changedetection.configs.config", self.name)
        script_utils = _import_upstream_module("changedetection.script.script_utils", self.name)
        model_module = _import_upstream_module(
            "changedetection.models.ChangeMambaBCD",
            self.name,
        )
        args = SimpleNamespace(
            cfg=str(config_path),
            opts=None,
            batch_size=None,
            data_path=None,
            zip=False,
            cache_mode=None,
            pretrained=None,
            encoder_pretrained_path=None,
            model_checkpoint_path=None,
            resume=None,
            resume_training_path=None,
            accumulation_steps=None,
            use_checkpoint=False,
            disable_amp=None,
            output="",
            tag="scp",
            eval=False,
            throughput=False,
            enable_amp=None,
            fused_layernorm=False,
            optim=None,
        )
        config = config_module.get_config(args)
        return model_module.ChangeMambaBCD(
            pretrained=None,
            **script_utils.get_vssm_kwargs(config),
        )

    def _validated_cfg_path(self) -> Path:
        if not self._cfg_path.is_file():
            raise ConfigError(f"{self.name} config does not exist: {self._cfg_path}")
        return self._cfg_path.resolve()

    def _to_input_tensor(
        self,
        image: NDArray[np.integer] | NDArray[np.floating],
        torch: ModuleType,
        device: Any,
    ) -> Any:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected HxWx3 image array, got shape {array.shape}")
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device=device)
        mean_tensor = torch.tensor(
            IMAGENET_PIXEL_NORMALIZATION_MEAN,
            dtype=tensor.dtype,
            device=device,
        ).view(1, 3, 1, 1)
        std_tensor = torch.tensor(
            IMAGENET_PIXEL_NORMALIZATION_STD,
            dtype=tensor.dtype,
            device=device,
        ).view(1, 3, 1, 1)
        return (tensor - mean_tensor) / std_tensor

    def _output_to_probability_map(self, output: Any, torch: ModuleType) -> NDArray[np.float32]:
        return self._logits_to_probability_map(output, torch)


class CDMambaAdapter(_LazyTorchAdapter):
    """Lazy adapter for the upstream CDMamba binary change-detection model."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        repo_path: Path | str = Path("third_party/CDMamba"),
        device: str = "auto",
        config_path: Path | str = CDMAMBA_DEFAULT_CONFIG_PATH,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._repo_path = Path(repo_path)
        self._config_path = Path(config_path)

    @property
    def name(self) -> str:
        """Adapter registry name."""
        return "cdmamba"

    @property
    def repo_path(self) -> Path:
        """Return the upstream repository path."""
        return self._repo_path

    def _load_state(self) -> _TorchModelState:
        checkpoint_path = self._validated_checkpoint_path()
        torch = _import_torch()
        device = _resolve_device(torch, self._device_name)
        with _UpstreamImportContext(self.repo_path):
            model = self._build_model(torch)
            checkpoint = _load_trusted_checkpoint(torch, checkpoint_path, device)
            load_result = model.load_state_dict(_unwrap_state_dict(checkpoint), strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                raise ModelLoadError(
                    "CDMamba checkpoint is incompatible with the configured model: "
                    f"missing={load_result.missing_keys}, "
                    f"unexpected={load_result.unexpected_keys}"
                )
            model.to(device)
            model.eval()
        return _TorchModelState(torch=torch, model=model, device=device)

    def _build_model(self, torch: ModuleType) -> Any:
        options = _load_json_with_comments(self._validated_config_path())
        model_options = options.get("model")
        if not isinstance(model_options, dict):
            raise ConfigError("models.cdmamba.config_path must contain a 'model' mapping")
        _ensure_mamba_ssm_selective_scan_compat()
        model_module = _import_upstream_module("models.CDMamba", self.name)
        _import_upstream_module("models.mamba_customer", self.name)
        return model_module.CDMamba(
            spatial_dims=model_options["spatial_dims"],
            in_channels=model_options["in_channels"],
            init_filters=model_options["init_filters"],
            out_channels=model_options["n_classes"],
            mode=model_options["mode"],
            conv_mode=model_options["conv_mode"],
            up_mode=model_options["up_mode"],
            up_conv_mode=model_options["up_conv_mode"],
            norm=model_options["norm"],
            blocks_down=model_options["blocks_down"],
            blocks_up=model_options["blocks_up"],
            resdiual=model_options["resdiual"],
            diff_abs=model_options["diff_abs"],
            stage=model_options["stage"],
            mamba_act=model_options["mamba_act"],
            local_query_model=model_options["local_query_model"],
        )

    def _validated_config_path(self) -> Path:
        if not self._config_path.is_file():
            raise ConfigError(f"{self.name} config does not exist: {self._config_path}")
        return self._config_path.resolve()

    def _to_input_tensor(
        self,
        image: NDArray[np.integer] | NDArray[np.floating],
        torch: ModuleType,
        device: Any,
    ) -> Any:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected HxWx3 image array, got shape {array.shape}")
        if array.max(initial=0.0) > 1.0:
            array = array / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device=device)
        return tensor * 2.0 - 1.0

    def _output_to_probability_map(self, output: Any, torch: ModuleType) -> NDArray[np.float32]:
        return self._logits_to_probability_map(output, torch)


class _UpstreamImportContext:
    """Temporarily prioritize an upstream repository for absolute imports."""

    def __init__(self, repo_path: Path) -> None:
        self._repo_path = repo_path
        self._repo_string = str(repo_path.resolve())

    def __enter__(self) -> None:
        while self._repo_string in sys.path:
            sys.path.remove(self._repo_string)
        sys.path.insert(0, self._repo_string)
        _purge_upstream_modules()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        while self._repo_string in sys.path:
            sys.path.remove(self._repo_string)
        _purge_upstream_modules()
        return None


def _purge_upstream_modules() -> None:
    prefixes = (
        "changedetection",
        "core",
        "data",
        "dataset",
        "datasets",
        "metrics",
        "misc",
        "models",
    )
    for module_name in tuple(sys.modules):
        if module_name in prefixes or module_name.startswith(
            tuple(f"{prefix}." for prefix in prefixes)
        ):
            sys.modules.pop(module_name, None)


def _import_torch() -> ModuleType:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ModelLoadError(
            "Install torch and the selected third-party model dependencies before prediction."
        ) from error


def _ensure_torchvision_models_utils() -> None:
    """Shim `torchvision.models.utils` for legacy upstream code.

    torchvision >= 0.13 removed `torchvision.models.utils`; older third_party
    repos (e.g. BIT_CD's resnet.py) still import `load_state_dict_from_url`
    from it. third_party is read-only, so the alias is installed
    programmatically instead of patching upstream files.
    """
    if "torchvision.models.utils" in sys.modules:
        return
    try:
        importlib.import_module("torchvision.models.utils")
        return
    except ImportError:
        pass
    try:
        hub = importlib.import_module("torch.hub")
    except ImportError:
        return
    shim = ModuleType("torchvision.models.utils")
    shim.load_state_dict_from_url = hub.load_state_dict_from_url  # type: ignore[attr-defined]
    sys.modules["torchvision.models.utils"] = shim


def _ensure_mamba_ssm_transformers_compat() -> None:
    """Alias transformers generation classes that mamba_ssm's __init__ expects.

    `mamba_ssm/__init__.py` eagerly imports `mamba_ssm.utils.generation`,
    which imports `GreedySearchDecoderOnlyOutput` and
    `SampleDecoderOnlyOutput` from `transformers.generation`. Newer
    transformers releases renamed both to `GenerateDecoderOnlyOutput`. CDMamba
    never uses mamba_ssm's LM generation utilities, so the legacy names are
    aliased on the installed transformers module to unblock the import chain.
    """
    try:
        generation_module = importlib.import_module("transformers.generation")
    except ImportError:
        return
    if not hasattr(generation_module, "GenerateDecoderOnlyOutput"):
        return
    for name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
        if not hasattr(generation_module, name):
            setattr(generation_module, name, generation_module.GenerateDecoderOnlyOutput)


def _ensure_mamba_ssm_selective_scan_compat() -> None:
    """Backfill selective-scan callables newer mamba_ssm releases dropped.

    Recent mamba_ssm versions removed `bimamba_inner_fn` and
    `mamba_inner_fn_no_out_proj` from `selective_scan_interface`. The vendored
    CDMamba `mamba_customer.py` catches the resulting ImportError, but its
    fallback assignment has 5 values for 4 names and raises ValueError
    instead of degrading gracefully. third_party is read-only, so the missing
    names are backfilled on the installed mamba_ssm module instead, letting
    the upstream `from ... import ...` succeed with `None` placeholders.
    """
    _ensure_mamba_ssm_transformers_compat()
    try:
        module = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    except ImportError:
        return
    for name in ("bimamba_inner_fn", "mamba_inner_fn_no_out_proj"):
        if not hasattr(module, name):
            setattr(module, name, None)


def _import_upstream_module(module_name: str, adapter_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        missing_dependency = _format_missing_dependency(error)
        raise ModelLoadError(
            f"Could not import {adapter_name} upstream module '{module_name}'. "
            "Install its third-party runtime dependencies before prediction."
            f"{missing_dependency}"
        ) from error


def _format_missing_dependency(error: ImportError) -> str:
    """Return a short missing-dependency suffix for an upstream import error."""
    if error.name is None:
        return ""
    if _is_binary_extension_load_error(error):
        return (
            f" Incompatible binary dependency: {error.name} failed to load. "
            "Rebuild or reinstall that extension against the active torch/CUDA runtime."
        )
    return f" Missing dependency: {error.name}."


def _is_binary_extension_load_error(error: ImportError) -> bool:
    """Return whether an import failed while loading an installed binary extension."""
    error_message = str(error)
    binary_markers = (
        ".so:",
        "undefined symbol:",
        "cannot open shared object file",
        "dynamic module does not define module export function",
    )
    return any(marker in error_message for marker in binary_markers)


def _resolve_device(torch: ModuleType, device_name: str) -> Any:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError(
            f"CUDA device '{device_name}' was requested, but torch reports CUDA unavailable"
        )
    return device


def _unwrap_state_dict(checkpoint: Any) -> Any:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_G_state_dict", "state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if value is not None:
            return value
    return checkpoint


def _select_batch_item(output: Any, index: int) -> Any:
    if isinstance(output, list):
        return [_select_batch_item(item, index) for item in output]
    if isinstance(output, tuple):
        return tuple(_select_batch_item(item, index) for item in output)
    return output[index : index + 1]


def _coerce_pair(
    pair: ImagePair | NDArray[np.integer] | NDArray[np.floating],
    after: NDArray[np.integer] | NDArray[np.floating] | None,
) -> ImagePair:
    if isinstance(pair, ImagePair):
        if after is not None:
            raise ValueError("after must be omitted when the first argument is an ImagePair")
        return pair
    if after is None:
        raise ValueError("after image is required when predicting from arrays")
    return ImagePair(before=np.asarray(pair), after=np.asarray(after))


def _normalize_image(
    image: NDArray[np.integer] | NDArray[np.floating],
    torch: ModuleType,
    device: Any,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Any:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image array, got shape {array.shape}")
    if array.max(initial=0.0) > 1.0:
        array = array / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device=device)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=device).view(1, 3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def _load_json_with_comments(path: Path) -> dict[str, Any]:
    """Load a JSON file that may contain JavaScript-style line comments."""
    try:
        return json.loads(_strip_json_line_comments(path.read_text(encoding="utf-8")))
    except OSError as error:
        raise ConfigError(f"Could not read JSON config: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Could not parse JSON config: {path}") from error


def _strip_json_line_comments(text: str) -> str:
    """Remove // line comments while preserving quoted string content."""
    stripped_lines = []
    for line in text.splitlines():
        in_string = False
        is_escaped = False
        output_chars = []
        index = 0
        while index < len(line):
            char = line[index]
            next_char = line[index + 1] if index + 1 < len(line) else ""
            if char == '"' and not is_escaped:
                in_string = not in_string
            if not in_string and char == "/" and next_char == "/":
                break
            output_chars.append(char)
            is_escaped = char == "\\" and not is_escaped
            if char != "\\":
                is_escaped = False
            index += 1
        stripped_lines.append("".join(output_chars))
    return "\n".join(stripped_lines)
