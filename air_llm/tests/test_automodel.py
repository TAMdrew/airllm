"""Test suite for AutoModel factory class.

Tests the AutoModel.from_pretrained() method that dispatches to
the correct AirLLM model class based on the architecture.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from air_llm.airllm.auto_model import AutoModel
from air_llm.airllm.constants import IS_ON_MAC_OS


def test_auto_model_cannot_be_instantiated_directly() -> None:
    """Test that AutoModel raises OSError when instantiated directly."""
    # Arrange & Act & Assert
    with pytest.raises(OSError, match="AutoModel is designed to be instantiated"):
        AutoModel()


@pytest.mark.skipif(IS_ON_MAC_OS, reason="macOS uses MLX backend")
@patch("air_llm.airllm.auto_model.AutoConfig")
def test_auto_model_from_pretrained_returns_correct_class(mock_auto_config) -> None:
    """Test that from_pretrained returns the correct model class."""
    # Arrange
    mock_config = MagicMock()
    mock_config.architectures = ["LlamaForCausalLM"]
    mock_auto_config.from_pretrained.return_value = mock_config

    mock_model_class = MagicMock()
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance

    with patch("air_llm.airllm.auto_model.ModelRegistry") as mock_registry:
        mock_registry.get.return_value = mock_model_class

        # Act
        result = AutoModel.from_pretrained("fake-model-path")

        # Assert
        assert result is mock_instance
        mock_model_class.assert_called_once()


@pytest.mark.skipif(IS_ON_MAC_OS, reason="macOS uses MLX backend")
@patch("air_llm.airllm.auto_model.AutoConfig")
def test_auto_model_from_pretrained_with_hf_token(mock_auto_config) -> None:
    """Test that from_pretrained passes hf_token to config loading."""
    # Arrange
    mock_config = MagicMock()
    mock_config.architectures = ["LlamaForCausalLM"]
    mock_auto_config.from_pretrained.return_value = mock_config

    mock_model_class = MagicMock()
    with patch("air_llm.airllm.auto_model.ModelRegistry") as mock_registry:
        mock_registry.get.return_value = mock_model_class

        # Act
        AutoModel.from_pretrained("fake-model-path", hf_token="test-token")

        # Assert
        mock_auto_config.from_pretrained.assert_called_once()
        call_kwargs = mock_auto_config.from_pretrained.call_args[1]
        assert "token" in call_kwargs
        assert call_kwargs["token"] == "test-token"


@pytest.mark.skipif(IS_ON_MAC_OS, reason="macOS uses MLX backend")
@patch("air_llm.airllm.auto_model.AutoConfig")
def test_auto_model_from_pretrained_with_unknown_architecture(mock_auto_config) -> None:
    """Test that unknown architecture falls back to AirLLMLlama2."""
    # Arrange
    mock_config = MagicMock()
    mock_config.architectures = ["CompletelyUnknownArchitecture"]
    mock_auto_config.from_pretrained.return_value = mock_config

    mock_model_class = MagicMock()
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance

    # Unknown arch triggers ValueError from registry, then falls back
    with patch("air_llm.airllm.auto_model.ModelRegistry") as mock_registry:
        mock_registry.get.side_effect = ValueError("Unknown")
        with patch("air_llm.airllm.airllm.AirLLMLlama2", mock_model_class):
            # Act
            result = AutoModel.from_pretrained("fake-model-path")

            # Assert - Should fall back to AirLLMLlama2
            assert result is mock_instance


@pytest.mark.skipif(IS_ON_MAC_OS, reason="macOS uses MLX backend")
@patch("air_llm.airllm.auto_model.AutoConfig")
def test_auto_model_from_pretrained_forwards_args_and_kwargs(mock_auto_config) -> None:
    """Test that from_pretrained forwards all args and kwargs to the model class."""
    # Arrange
    mock_config = MagicMock()
    mock_config.architectures = ["LlamaForCausalLM"]
    mock_auto_config.from_pretrained.return_value = mock_config

    mock_model_class = MagicMock()
    with patch("air_llm.airllm.auto_model.ModelRegistry") as mock_registry:
        mock_registry.get.return_value = mock_model_class

        # Act
        AutoModel.from_pretrained(
            "fake-model-path",
            device="cuda:1",
            compression="4bit",
            custom_arg="custom_value",
        )

        # Assert
        mock_model_class.assert_called_once_with(
            "fake-model-path",
            device="cuda:1",
            compression="4bit",
            custom_arg="custom_value",
        )


@pytest.mark.skipif(not IS_ON_MAC_OS, reason="macOS-specific test")
def test_auto_model_on_macos_returns_mlx_model() -> None:
    """Test that AutoModel returns AirLLMLlamaMlx on macOS."""
    # Arrange — patch at the import target inside auto_model.from_pretrained
    with patch("air_llm.airllm.auto_model.AirLLMLlamaMlx", create=True) as mock_mlx_class:
        mock_instance = MagicMock()
        mock_mlx_class.return_value = mock_instance

        # Also patch the lazy import inside from_pretrained
        with patch.dict(
            "sys.modules",
            {"air_llm.airllm.airllm_llama_mlx": MagicMock(AirLLMLlamaMlx=mock_mlx_class)},
        ):
            # Act
            result = AutoModel.from_pretrained("fake-model-path")

            # Assert
            assert result is mock_instance


@pytest.mark.skipif(IS_ON_MAC_OS, reason="Non-macOS test")
@patch("air_llm.airllm.auto_model.AutoConfig")
def test_auto_model_trust_remote_code_in_config_loading(mock_auto_config) -> None:
    """Test that trust_remote_code=True is passed to config loading."""
    # Arrange
    mock_config = MagicMock()
    mock_config.architectures = ["LlamaForCausalLM"]
    mock_auto_config.from_pretrained.return_value = mock_config

    mock_model_class = MagicMock()
    with patch("air_llm.airllm.auto_model.ModelRegistry") as mock_registry:
        mock_registry.get.return_value = mock_model_class

        # Act
        AutoModel.from_pretrained("fake-model-path")

        # Assert
        call_kwargs = mock_auto_config.from_pretrained.call_args[1]
        # trust_remote_code defaults to False
        assert call_kwargs.get("trust_remote_code") is False


@pytest.mark.skipif(IS_ON_MAC_OS, reason="Non-macOS test")
@patch("air_llm.airllm.auto_model.AutoConfig")
def test_auto_model_uses_first_architecture_from_list(mock_auto_config) -> None:
    """Test that AutoModel uses the first architecture from the config list."""
    # Arrange
    mock_config = MagicMock()
    mock_config.architectures = ["MistralForCausalLM", "LlamaForCausalLM"]
    mock_auto_config.from_pretrained.return_value = mock_config

    mock_model_class = MagicMock()
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance

    with patch("air_llm.airllm.auto_model.ModelRegistry") as mock_registry:
        mock_registry.get.return_value = mock_model_class

        # Act
        result = AutoModel.from_pretrained("fake-model-path")

        # Assert - Should use first architecture (Mistral)
        assert result is mock_instance
        mock_registry.get.assert_called_once_with("MistralForCausalLM")
