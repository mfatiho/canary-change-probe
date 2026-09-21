from pathlib import Path

import pytest

from scp.utils import ConfigError, load_config


def _base_config(models_section: str) -> str:
    return f"""
seed: 7
device: auto
threshold: 0.5
paths:
  bit_repo: third_party/BIT_CD
  changeformer_repo: third_party/ChangeFormer
  tinycd_repo: third_party/Tiny_model_4_CD
  data_root: data
  results_root: results
datasets:
  demo:
    path: data/demo
{models_section}
"""


def test_load_config_parses_model_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _base_config(
            """
models:
  bit:
    checkpoint: D:\\datasets\\LEVIR-CD\\BIT_LEVIR\\best_ckpt.pt
    net_g: base_transformer_pos_s4_dd8_dedim8
    image_size: 256
  changeformer:
    checkpoint: null
    net_g: ChangeFormerV6
    embed_dim: 256
    image_size: 256
  cdmamba:
    checkpoint: third_party/CDMamba/checkpoints/LEVIR-CD/best_cd_model_gen.pth
    trained_dataset: levir-256
  tinycd:
    checkpoint: null
    backbone: efficientnet_b4
    pretrained_backbone: false
  changemamba:
    checkpoint: third_party/ChangeMamba/pretrained_weight/MambaBCD_Tiny_WHU_F1_0.9409.pth
    trained_dataset: whu-256
"""
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.models.bit.checkpoint == Path(
        "D:\\datasets\\LEVIR-CD\\BIT_LEVIR\\best_ckpt.pt"
    )
    assert config.models.bit.net_g == "base_transformer_pos_s4_dd8_dedim8"
    assert config.models.changeformer.embed_dim == 256
    assert config.models.cdmamba.trained_dataset == "levir-256"
    assert config.models.tinycd.pretrained_backbone is False
    assert config.models.changemamba.trained_dataset == "whu-256"


def test_load_config_defaults_tinycd_pretrained_backbone_to_false(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _base_config(
            """
models:
  bit:
    checkpoint: null
  changeformer:
    checkpoint: null
  tinycd:
    checkpoint: null
"""
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.models.tinycd.pretrained_backbone is False


def test_load_config_rejects_missing_models_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_base_config(""), encoding="utf-8")

    with pytest.raises(ConfigError, match="models"):
        load_config(config_path)


def test_load_config_rejects_invalid_model_checkpoint_type(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _base_config(
            """
models:
  bit:
    checkpoint: 3
  changeformer:
    checkpoint: null
  tinycd:
    checkpoint: null
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="checkpoint"):
        load_config(config_path)
