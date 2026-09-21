"""Typed YAML configuration loading for experiment settings."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scp.scoring import DecisionThresholds

MIN_PROBABILITY = 0.0
MAX_PROBABILITY = 1.0
DEFAULT_DECISION_MARGIN = 0.2
DEFAULT_CANARY_TYPES = (
    "building_crop_add",
    "building_remove_fill",
    "linear_road_add",
    "texture_replace",
)
DEFAULT_CANARY_GENERATOR_PARAMS = {
    "small_building_add": {"size": 4},
    "building_remove": {"size": 4},
    "building_crop_add": {},
    "building_remove_fill": {"size": 4, "ring_width": 2},
    "linear_road_add": {"width": 2},
    "texture_replace": {"size": 4},
}
DEFAULT_INFERENCE_BATCH_SIZE = 8
CHANGEMAMBA_DEFAULT_CFG_PATH = Path(
    "third_party/ChangeMamba/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
)
CDMAMBA_DEFAULT_CONFIG_PATH = Path("third_party/CDMamba/config/mamba/levir_test_cdmamba.json")


class ConfigError(ValueError):
    """Raised when a configuration file is missing required or valid values."""


@dataclass(frozen=True)
class PatchingConfig:
    """Patch extraction settings for datasets that require tiling."""

    enabled: bool
    patch_size: int
    stride: int | None = None

    def __post_init__(self) -> None:
        """Validate patching dimensions early."""
        if self.patch_size <= 0:
            raise ConfigError("patch_size must be positive")
        if self.stride is not None and self.stride <= 0:
            raise ConfigError("stride must be positive when provided")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing options grouped under a dataset entry."""

    patching: PatchingConfig | None = None


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for one named dataset root and preprocessing pipeline."""

    path: Path
    preprocessing: PreprocessingConfig

    @property
    def patching(self) -> PatchingConfig | None:
        """Return patching settings without exposing preprocessing nesting to callers."""
        return self.preprocessing.patching


@dataclass(frozen=True)
class BITModelConfig:
    """Configuration required to construct and run the upstream BIT model."""

    checkpoint: Path | None
    net_g: str = "base_transformer_pos_s4_dd8_dedim8"
    image_size: int = 256

    def __post_init__(self) -> None:
        """Validate model construction settings."""
        if not self.net_g:
            raise ConfigError("models.bit.net_g must be a non-empty string")
        if self.image_size <= 0:
            raise ConfigError("models.bit.image_size must be positive")


@dataclass(frozen=True)
class ChangeFormerModelConfig:
    """Configuration required to construct and run the upstream ChangeFormer model."""

    checkpoint: Path | None
    net_g: str = "ChangeFormerV6"
    embed_dim: int = 256
    image_size: int = 256

    def __post_init__(self) -> None:
        """Validate model construction settings."""
        if not self.net_g:
            raise ConfigError("models.changeformer.net_g must be a non-empty string")
        if self.embed_dim <= 0:
            raise ConfigError("models.changeformer.embed_dim must be positive")
        if self.image_size <= 0:
            raise ConfigError("models.changeformer.image_size must be positive")


@dataclass(frozen=True)
class TinyCDModelConfig:
    """Configuration required to construct and run the upstream TinyCD model."""

    checkpoint: Path | None
    backbone: str = "efficientnet_b4"
    output_layer_backbone: str = "3"
    pretrained_backbone: bool = False
    freeze_backbone: bool = False

    def __post_init__(self) -> None:
        """Validate TinyCD backbone settings."""
        if not self.backbone:
            raise ConfigError("models.tinycd.backbone must be a non-empty string")
        if not self.output_layer_backbone:
            raise ConfigError("models.tinycd.output_layer_backbone must be a non-empty string")


@dataclass(frozen=True)
class ChangeMambaModelConfig:
    """Configuration required to construct and run the upstream ChangeMamba model."""

    checkpoint: Path | None
    cfg_path: Path = CHANGEMAMBA_DEFAULT_CFG_PATH
    trained_dataset: str | None = None

    def __post_init__(self) -> None:
        """Validate ChangeMamba construction settings."""
        if not str(self.cfg_path):
            raise ConfigError("models.changemamba.cfg_path must be a non-empty path")
        if self.trained_dataset is not None and not self.trained_dataset:
            raise ConfigError("models.changemamba.trained_dataset must be non-empty")


@dataclass(frozen=True)
class CDMambaModelConfig:
    """Configuration required to construct and run the upstream CDMamba model."""

    checkpoint: Path | None
    config_path: Path = CDMAMBA_DEFAULT_CONFIG_PATH
    trained_dataset: str | None = None

    def __post_init__(self) -> None:
        """Validate CDMamba construction settings."""
        if not str(self.config_path):
            raise ConfigError("models.cdmamba.config_path must be a non-empty path")
        if self.trained_dataset is not None and not self.trained_dataset:
            raise ConfigError("models.cdmamba.trained_dataset must be non-empty")


@dataclass(frozen=True)
class ModelsConfig:
    """Configuration for all supported change-detection backbones."""

    bit: BITModelConfig
    cdmamba: CDMambaModelConfig
    changeformer: ChangeFormerModelConfig
    changemamba: ChangeMambaModelConfig
    tinycd: TinyCDModelConfig


@dataclass(frozen=True)
class CanaryConfig:
    """Configuration for synthetic canary probing."""

    enabled_types: tuple[str, ...]
    canaries_per_pair: int
    bank_path: Path | None
    generator_params: dict[str, dict[str, int | float]]

    def __post_init__(self) -> None:
        """Validate canary type selection and generator constructor parameters."""
        if not self.enabled_types:
            raise ConfigError("canary.enabled_types must include at least one canary")
        if self.canaries_per_pair <= 0:
            raise ConfigError("canary.canaries_per_pair must be positive")

        registered_types = _registered_canary_names()
        for canary_type in self.enabled_types:
            if canary_type not in registered_types:
                raise ConfigError(f"unknown canary type: {canary_type}")

        for canary_type, params in self.generator_params.items():
            if canary_type not in registered_types:
                raise ConfigError(f"unknown canary generator params target: {canary_type}")
            allowed_params = _canary_constructor_params(canary_type)
            unknown_params = sorted(set(params) - allowed_params)
            if unknown_params:
                names = ", ".join(unknown_params)
                raise ConfigError(f"unknown canary generator params for {canary_type}: {names}")
            for param_name, value in params.items():
                if isinstance(value, bool) or not isinstance(value, int | float):
                    label = f"canary.generators.{canary_type}.{param_name}"
                    raise ConfigError(f"{label} must be numeric")


@dataclass(frozen=True)
class PathsConfig:
    """Repository and output paths used by adapters and experiment runners."""

    bit_repo: Path
    cdmamba_repo: Path
    changeformer_repo: Path
    changemamba_repo: Path
    tinycd_repo: Path
    data_root: Path
    results_root: Path


@dataclass(frozen=True)
class AppConfig:
    """Resolved top-level experiment configuration."""

    seed: int
    device: str
    threshold: float
    inference_batch_size: int
    paths: PathsConfig
    datasets: dict[str, DatasetConfig]
    models: ModelsConfig
    canary: CanaryConfig
    decision_thresholds: DecisionThresholds


def load_config(path: Path | str = Path("configs/default.yaml")) -> AppConfig:
    """Load and validate a YAML experiment configuration.

    Args:
        path: YAML file path.

    Returns:
        A typed, validated application configuration.

    Raises:
        ConfigError: If the YAML cannot be parsed or required fields are invalid.
    """
    config_path = Path(path)
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Could not read config: {config_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Could not parse YAML config: {config_path}") from error

    if not isinstance(raw_config, dict):
        raise ConfigError("config must be a YAML mapping")

    paths = _parse_paths(_required_mapping(raw_config, "paths"))
    datasets = _parse_datasets(_required_mapping(raw_config, "datasets"))
    threshold = _required_probability(raw_config, "threshold")
    models = _parse_models(_required_mapping(raw_config, "models"))
    return AppConfig(
        seed=_required_int(raw_config, "seed"),
        device=_required_str(raw_config, "device"),
        threshold=threshold,
        inference_batch_size=_optional_positive_int(
            raw_config,
            "inference_batch_size",
            DEFAULT_INFERENCE_BATCH_SIZE,
        ),
        paths=paths,
        datasets=datasets,
        models=models,
        canary=_parse_canary(raw_config.get("canary", {})),
        decision_thresholds=_decision_thresholds_from_threshold(threshold),
    )


def _parse_paths(raw_paths: dict[str, Any]) -> PathsConfig:
    required_fields = ["bit_repo", "changeformer_repo", "tinycd_repo", "data_root", "results_root"]
    missing_fields = [field for field in required_fields if field not in raw_paths]
    if missing_fields:
        raise ConfigError(f"paths missing required fields: {', '.join(missing_fields)}")
    return PathsConfig(
        bit_repo=Path(_required_str(raw_paths, "bit_repo")),
        cdmamba_repo=Path(_optional_str(raw_paths, "cdmamba_repo", "third_party/CDMamba")),
        changeformer_repo=Path(_required_str(raw_paths, "changeformer_repo")),
        changemamba_repo=Path(
            _optional_str(raw_paths, "changemamba_repo", "third_party/ChangeMamba")
        ),
        tinycd_repo=Path(_required_str(raw_paths, "tinycd_repo")),
        data_root=Path(_required_str(raw_paths, "data_root")),
        results_root=Path(_required_str(raw_paths, "results_root")),
    )


def _parse_datasets(raw_datasets: dict[str, Any]) -> dict[str, DatasetConfig]:
    if not raw_datasets:
        raise ConfigError("datasets must include at least one dataset")

    datasets: dict[str, DatasetConfig] = {}
    for name, raw_dataset in raw_datasets.items():
        if not isinstance(raw_dataset, dict):
            raise ConfigError(f"dataset '{name}' must be a mapping")
        datasets[str(name)] = DatasetConfig(
            path=Path(_required_str(raw_dataset, "path")),
            preprocessing=_parse_preprocessing(raw_dataset.get("preprocessing", {})),
        )
    return datasets


def _parse_models(raw_models: dict[str, Any]) -> ModelsConfig:
    return ModelsConfig(
        bit=_parse_bit_model(_required_mapping(raw_models, "bit")),
        cdmamba=_parse_cdmamba_model(
            _optional_mapping(raw_models, "cdmamba"),
        ),
        changeformer=_parse_changeformer_model(_required_mapping(raw_models, "changeformer")),
        changemamba=_parse_changemamba_model(
            _optional_mapping(raw_models, "changemamba"),
        ),
        tinycd=_parse_tinycd_model(_required_mapping(raw_models, "tinycd")),
    )


def _parse_canary(raw_canary: object) -> CanaryConfig:
    if raw_canary is None:
        raw_canary = {}
    if not isinstance(raw_canary, dict):
        raise ConfigError("canary must be a mapping when provided")

    return CanaryConfig(
        enabled_types=_parse_canary_types(raw_canary.get("enabled_types")),
        canaries_per_pair=_optional_positive_int(raw_canary, "canaries_per_pair", 1),
        bank_path=_optional_path(raw_canary, "bank_path", "canary.bank_path"),
        generator_params=_parse_canary_generator_params(raw_canary.get("generators", {})),
    )


def _parse_canary_types(raw_enabled_types: object) -> tuple[str, ...]:
    if raw_enabled_types is None:
        return DEFAULT_CANARY_TYPES
    if not isinstance(raw_enabled_types, list) or not raw_enabled_types:
        raise ConfigError("canary.enabled_types must be a non-empty list")

    enabled_types: list[str] = []
    for value in raw_enabled_types:
        if not isinstance(value, str) or not value:
            raise ConfigError("canary.enabled_types entries must be non-empty strings")
        enabled_types.append(value)
    return tuple(enabled_types)


def _parse_canary_generator_params(
    raw_generators: object,
) -> dict[str, dict[str, int | float]]:
    if raw_generators is None:
        return _default_canary_generator_params()
    if not isinstance(raw_generators, dict):
        raise ConfigError("canary.generators must be a mapping when provided")

    generator_params = _default_canary_generator_params()
    for canary_type, raw_params in raw_generators.items():
        if not isinstance(canary_type, str) or not canary_type:
            raise ConfigError("canary.generators keys must be non-empty strings")
        if not isinstance(raw_params, dict):
            raise ConfigError(f"canary.generators.{canary_type} must be a mapping")
        generator_params[canary_type] = {
            str(param_name): _numeric_canary_param(canary_type, str(param_name), value)
            for param_name, value in raw_params.items()
        }
    return generator_params


def _default_canary_generator_params() -> dict[str, dict[str, int | float]]:
    return {
        canary_type: dict(params)
        for canary_type, params in DEFAULT_CANARY_GENERATOR_PARAMS.items()
    }


def _numeric_canary_param(canary_type: str, param_name: str, value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"canary.generators.{canary_type}.{param_name} must be numeric")
    return value


def _parse_bit_model(raw_model: dict[str, Any]) -> BITModelConfig:
    return BITModelConfig(
        checkpoint=_optional_path(raw_model, "checkpoint", "models.bit.checkpoint"),
        net_g=_optional_str(raw_model, "net_g", "base_transformer_pos_s4_dd8_dedim8"),
        image_size=_optional_int_with_default(raw_model, "image_size", 256),
    )


def _parse_changeformer_model(raw_model: dict[str, Any]) -> ChangeFormerModelConfig:
    return ChangeFormerModelConfig(
        checkpoint=_optional_path(raw_model, "checkpoint", "models.changeformer.checkpoint"),
        net_g=_optional_str(raw_model, "net_g", "ChangeFormerV6"),
        embed_dim=_optional_int_with_default(raw_model, "embed_dim", 256),
        image_size=_optional_int_with_default(raw_model, "image_size", 256),
    )


def _parse_tinycd_model(raw_model: dict[str, Any]) -> TinyCDModelConfig:
    return TinyCDModelConfig(
        checkpoint=_optional_path(raw_model, "checkpoint", "models.tinycd.checkpoint"),
        backbone=_optional_str(raw_model, "backbone", "efficientnet_b4"),
        output_layer_backbone=_optional_str(raw_model, "output_layer_backbone", "3"),
        pretrained_backbone=_optional_bool(raw_model, "pretrained_backbone", False),
        freeze_backbone=_optional_bool(raw_model, "freeze_backbone", False),
    )


def _parse_changemamba_model(raw_model: dict[str, Any]) -> ChangeMambaModelConfig:
    return ChangeMambaModelConfig(
        checkpoint=_optional_path(raw_model, "checkpoint", "models.changemamba.checkpoint"),
        cfg_path=_optional_path(
            raw_model,
            "cfg_path",
            "models.changemamba.cfg_path",
        )
        or CHANGEMAMBA_DEFAULT_CFG_PATH,
        trained_dataset=_optional_nullable_str(raw_model, "trained_dataset"),
    )


def _parse_cdmamba_model(raw_model: dict[str, Any]) -> CDMambaModelConfig:
    return CDMambaModelConfig(
        checkpoint=_optional_path(raw_model, "checkpoint", "models.cdmamba.checkpoint"),
        config_path=_optional_path(
            raw_model,
            "config_path",
            "models.cdmamba.config_path",
        )
        or CDMAMBA_DEFAULT_CONFIG_PATH,
        trained_dataset=_optional_nullable_str(raw_model, "trained_dataset"),
    )


def _parse_preprocessing(raw_preprocessing: object) -> PreprocessingConfig:
    if not isinstance(raw_preprocessing, dict):
        raise ConfigError("preprocessing must be a mapping when provided")
    raw_patching = raw_preprocessing.get("patching")
    if raw_patching is None:
        return PreprocessingConfig()
    if not isinstance(raw_patching, dict):
        raise ConfigError("patching must be a mapping when provided")
    return PreprocessingConfig(
        patching=PatchingConfig(
            enabled=_required_bool(raw_patching, "enabled"),
            patch_size=_required_int(raw_patching, "patch_size"),
            stride=_optional_int(raw_patching, "stride"),
        )
    )


def _decision_thresholds_from_threshold(threshold: float) -> DecisionThresholds:
    deploy = max(MIN_PROBABILITY, threshold - DEFAULT_DECISION_MARGIN)
    reject = min(MAX_PROBABILITY, threshold + DEFAULT_DECISION_MARGIN)
    return DecisionThresholds(deploy=deploy, reject=reject)


def _required_mapping(raw_config: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be provided as a mapping")
    return value


def _optional_mapping(raw_config: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw_config.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be provided as a mapping")
    return value


def _required_str(raw_config: dict[str, Any], key: str) -> str:
    value = raw_config.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _required_int(raw_config: dict[str, Any], key: str) -> int:
    value = raw_config.get(key)
    if not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def _optional_int(raw_config: dict[str, Any], key: str) -> int | None:
    if key not in raw_config:
        return None
    return _required_int(raw_config, key)


def _optional_int_with_default(raw_config: dict[str, Any], key: str, default: int) -> int:
    if key not in raw_config:
        return default
    return _required_int(raw_config, key)


def _optional_positive_int(raw_config: dict[str, Any], key: str, default: int) -> int:
    value = default if key not in raw_config else _required_int(raw_config, key)
    if value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value


def _optional_str(raw_config: dict[str, Any], key: str, default: str) -> str:
    if key not in raw_config:
        return default
    return _required_str(raw_config, key)


def _optional_nullable_str(raw_config: dict[str, Any], key: str) -> str | None:
    if key not in raw_config or raw_config[key] is None:
        return None
    return _required_str(raw_config, key)


def _optional_bool(raw_config: dict[str, Any], key: str, default: bool) -> bool:
    if key not in raw_config:
        return default
    return _required_bool(raw_config, key)


def _optional_path(raw_config: dict[str, Any], key: str, label: str) -> Path | None:
    if key not in raw_config or raw_config[key] is None:
        return None
    value = raw_config[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty string or null")
    return Path(value)


def _required_bool(raw_config: dict[str, Any], key: str) -> bool:
    value = raw_config.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _required_probability(raw_config: dict[str, Any], key: str) -> float:
    value = raw_config.get(key)
    if not isinstance(value, int | float):
        raise ConfigError(f"{key} must be numeric")
    probability = float(value)
    if not MIN_PROBABILITY <= probability <= MAX_PROBABILITY:
        raise ConfigError(f"{key} must be between {MIN_PROBABILITY} and {MAX_PROBABILITY}")
    return probability


def _canary_constructor_params(canary_type: str) -> set[str]:
    from scp.canary import CANARY_REGISTRY

    signature = inspect.signature(CANARY_REGISTRY.get(canary_type))
    return {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }


def _registered_canary_names() -> set[str]:
    from scp.canary import CANARY_REGISTRY

    return set(CANARY_REGISTRY.names())
