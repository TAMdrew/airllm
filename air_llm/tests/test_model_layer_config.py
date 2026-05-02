"""Test suite for ModelLayerConfig dataclass.

Tests the configuration object that maps logical layer roles
to weight name prefixes.
"""

from __future__ import annotations

from air_llm.airllm.airllm_base import ModelLayerConfig


def test_default_values() -> None:
    """Test that ModelLayerConfig has correct default values."""
    # Arrange & Act
    config = ModelLayerConfig()

    # Assert
    assert config.embed_tokens == "model.embed_tokens"
    assert config.layer_prefix == "model.layers"
    assert config.norm == "model.norm"
    assert config.lm_head == "lm_head"
    assert config.rotary_pos_emb is None


def test_custom_values() -> None:
    """Test that ModelLayerConfig accepts custom values."""
    # Arrange & Act
    config = ModelLayerConfig(
        embed_tokens="custom.embed",
        layer_prefix="custom.layers",
        norm="custom.norm",
        lm_head="custom.head",
        rotary_pos_emb="custom.rotary",
    )

    # Assert
    assert config.embed_tokens == "custom.embed"
    assert config.layer_prefix == "custom.layers"
    assert config.norm == "custom.norm"
    assert config.lm_head == "custom.head"
    assert config.rotary_pos_emb == "custom.rotary"


def test_to_dict_returns_correct_keys() -> None:
    """Test that to_dict() returns a dict with expected keys."""
    # Arrange
    config = ModelLayerConfig()

    # Act
    result = config.to_dict()

    # Assert
    assert isinstance(result, dict)
    assert "embed" in result
    assert "layer_prefix" in result
    assert "norm" in result
    assert "lm_head" in result
    assert result["embed"] == "model.embed_tokens"
    assert result["layer_prefix"] == "model.layers"
    assert result["norm"] == "model.norm"
    assert result["lm_head"] == "lm_head"


def test_to_dict_includes_rotary_pos_emb_when_set() -> None:
    """Test that to_dict() includes rotary_pos_emb when it's not None."""
    # Arrange
    config = ModelLayerConfig(rotary_pos_emb="custom.rotary")

    # Act
    result = config.to_dict()

    # Assert
    assert "rotary_pos_emb" in result
    assert result["rotary_pos_emb"] == "custom.rotary"


def test_to_dict_excludes_rotary_pos_emb_when_none() -> None:
    """Test that to_dict() excludes rotary_pos_emb when it's None."""
    # Arrange
    config = ModelLayerConfig(rotary_pos_emb=None)

    # Act
    result = config.to_dict()

    # Assert
    assert "rotary_pos_emb" not in result


def test_config_equality() -> None:
    """Test that two configs with same values are equal."""
    # Arrange
    config1 = ModelLayerConfig(
        embed_tokens="test.embed",
        layer_prefix="test.layers",
    )
    config2 = ModelLayerConfig(
        embed_tokens="test.embed",
        layer_prefix="test.layers",
    )

    # Act & Assert
    assert config1 == config2


def test_config_inequality() -> None:
    """Test that two configs with different values are not equal."""
    # Arrange
    config1 = ModelLayerConfig(embed_tokens="test1.embed")
    config2 = ModelLayerConfig(embed_tokens="test2.embed")

    # Act & Assert
    assert config1 != config2


def test_config_is_dataclass() -> None:
    """Test that ModelLayerConfig is a dataclass."""
    # Arrange & Act
    config = ModelLayerConfig()

    # Assert
    assert hasattr(config, "__dataclass_fields__")


def test_to_dict_round_trip_compatibility() -> None:
    """Test that to_dict() output matches legacy layer_names_dict format."""
    # Arrange
    config = ModelLayerConfig(
        embed_tokens="model.embed_tokens",
        layer_prefix="model.layers",
        norm="model.norm",
        lm_head="lm_head",
    )

    # Act
    result = config.to_dict()

    # Assert - Should match the format expected by legacy code
    expected_keys = {"embed", "layer_prefix", "norm", "lm_head"}
    assert set(result.keys()) == expected_keys
