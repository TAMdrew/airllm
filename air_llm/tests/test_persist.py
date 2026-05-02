"""Test suite for persister classes.

Tests the ModelPersister interface and SafetensorModelPersister implementation.
"""

from __future__ import annotations

import pytest
import torch

from air_llm.airllm.constants import IS_ON_MAC_OS
from air_llm.airllm.persist.model_persister import ModelPersister


def _try_get_persister():
    """Helper to get persister, skipping if mlx is unavailable on macOS."""
    try:
        return ModelPersister.get_model_persister()
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mlx not installed — cannot instantiate MlxModelPersister")


def test_model_persister_get_model_persister_returns_singleton() -> None:
    """Test that get_model_persister returns the same instance."""
    # Arrange & Act
    persister1 = _try_get_persister()
    persister2 = _try_get_persister()

    # Assert
    assert persister1 is persister2


@pytest.mark.skipif(IS_ON_MAC_OS, reason="macOS uses MLX persister")
def test_model_persister_on_non_macos_returns_safetensor() -> None:
    """Test that non-macOS systems get SafetensorModelPersister."""
    # Arrange
    from air_llm.airllm.persist.safetensor_model_persister import SafetensorModelPersister

    # Act
    persister = ModelPersister.get_model_persister()

    # Assert
    assert isinstance(persister, SafetensorModelPersister)


@pytest.mark.skipif(not IS_ON_MAC_OS, reason="macOS-specific test")
def test_model_persister_on_macos_returns_mlx() -> None:
    """Test that macOS systems get MlxModelPersister."""
    try:
        from air_llm.airllm.persist.mlx_model_persister import MlxModelPersister
    except ImportError:
        pytest.skip("mlx not installed")

    # Act
    persister = ModelPersister.get_model_persister()

    # Assert
    assert isinstance(persister, MlxModelPersister)


def test_base_persister_model_persist_exist_not_implemented() -> None:
    """Test that base class model_persist_exist raises TypeError."""
    with pytest.raises(TypeError):
        ModelPersister()


def test_base_persister_persist_model_not_implemented() -> None:
    """Test that base class persist_model raises TypeError."""
    with pytest.raises(TypeError):
        ModelPersister()


def test_base_persister_load_model_not_implemented() -> None:
    """Test that base class load_model raises TypeError."""
    with pytest.raises(TypeError):
        ModelPersister()


@pytest.mark.skipif(IS_ON_MAC_OS, reason="SafetensorModelPersister not used on macOS")
def test_safetensor_persister_model_persist_exist_checks_files(temp_model_dir) -> None:
    """Test that SafetensorModelPersister checks for both .safetensors and .done files."""
    # Arrange
    from air_llm.airllm.persist.safetensor_model_persister import SafetensorModelPersister

    persister = SafetensorModelPersister()
    layer_name = "test.layer."

    # Create the files
    (temp_model_dir / f"{layer_name}safetensors").touch()
    (temp_model_dir / f"{layer_name}safetensors.done").touch()

    # Act
    exists = persister.model_persist_exist(layer_name, temp_model_dir)

    # Assert
    assert exists is True


@pytest.mark.skipif(IS_ON_MAC_OS, reason="SafetensorModelPersister not used on macOS")
def test_safetensor_persister_model_persist_exist_returns_false_if_missing_done(
    temp_model_dir,
) -> None:
    """Test that model_persist_exist returns False if .done marker is missing."""
    # Arrange
    from air_llm.airllm.persist.safetensor_model_persister import SafetensorModelPersister

    persister = SafetensorModelPersister()
    layer_name = "test.layer."

    # Create only the safetensors file, not the .done marker
    (temp_model_dir / f"{layer_name}safetensors").touch()

    # Act
    exists = persister.model_persist_exist(layer_name, temp_model_dir)

    # Assert
    assert exists is False


@pytest.mark.skipif(IS_ON_MAC_OS, reason="SafetensorModelPersister not used on macOS")
def test_safetensor_persister_model_persist_exist_returns_false_if_missing_safetensors(
    temp_model_dir,
) -> None:
    """Test that model_persist_exist returns False if .safetensors file is missing."""
    # Arrange
    from air_llm.airllm.persist.safetensor_model_persister import SafetensorModelPersister

    persister = SafetensorModelPersister()
    layer_name = "test.layer."

    # Create only the .done marker, not the safetensors file
    (temp_model_dir / f"{layer_name}safetensors.done").touch()

    # Act
    exists = persister.model_persist_exist(layer_name, temp_model_dir)

    # Assert
    assert exists is False


@pytest.mark.skipif(IS_ON_MAC_OS, reason="SafetensorModelPersister not used on macOS")
def test_safetensor_persister_persist_model_creates_files(temp_model_dir) -> None:
    """Test that persist_model creates both .safetensors and .done files."""
    # Arrange
    from air_llm.airllm.persist.safetensor_model_persister import SafetensorModelPersister

    persister = SafetensorModelPersister()
    layer_name = "test.layer."
    state_dict = {"weight": torch.randn(10, 10, dtype=torch.float32)}

    # Act
    persister.persist_model(state_dict, layer_name, temp_model_dir)

    # Assert
    assert (temp_model_dir / f"{layer_name}safetensors").exists()
    assert (temp_model_dir / f"{layer_name}safetensors.done").exists()


@pytest.mark.skipif(IS_ON_MAC_OS, reason="SafetensorModelPersister not used on macOS")
def test_safetensor_persister_load_model_returns_state_dict(temp_model_dir) -> None:
    """Test that load_model returns a valid state dict."""
    # Arrange
    from air_llm.airllm.persist.safetensor_model_persister import SafetensorModelPersister

    persister = SafetensorModelPersister()
    layer_name = "test.layer"
    original_state_dict = {
        "weight": torch.randn(10, 10, dtype=torch.float32),
        "bias": torch.randn(10, dtype=torch.float32),
    }

    # Save and load must use the same layer_name (including trailing dot)
    full_layer_name = layer_name + "."
    persister.persist_model(original_state_dict, full_layer_name, temp_model_dir)

    # Act
    loaded_state_dict = persister.load_model(full_layer_name, temp_model_dir)

    # Assert
    assert isinstance(loaded_state_dict, dict)
    assert "weight" in loaded_state_dict
    assert "bias" in loaded_state_dict
    assert torch.allclose(loaded_state_dict["weight"], original_state_dict["weight"])
    assert torch.allclose(loaded_state_dict["bias"], original_state_dict["bias"])


@pytest.mark.skipif(IS_ON_MAC_OS, reason="SafetensorModelPersister not used on macOS")
def test_safetensor_persister_round_trip_preserves_data(temp_model_dir) -> None:
    """Test that save -> load preserves tensor data exactly."""
    # Arrange
    from air_llm.airllm.persist.safetensor_model_persister import SafetensorModelPersister

    persister = SafetensorModelPersister()
    layer_name = "round.trip.layer"
    original = {
        "weight": torch.randn(32, 64, dtype=torch.float32),
        "bias": torch.randn(64, dtype=torch.float32),
    }

    # Act — save and load must use identical layer_name
    full_layer_name = layer_name + "."
    persister.persist_model(original, full_layer_name, temp_model_dir)
    loaded = persister.load_model(full_layer_name, temp_model_dir)

    # Assert
    for key in original:
        assert key in loaded
        assert torch.equal(original[key], loaded[key])


def test_persister_singleton_is_cached() -> None:
    """Test that the persister singleton is cached globally."""
    # Arrange
    # Clear the singleton
    import air_llm.airllm.persist.model_persister as mp_module

    original_persister = mp_module._model_persister
    mp_module._model_persister = None

    try:
        # Act
        first_call = _try_get_persister()
        second_call = _try_get_persister()

        # Assert
        assert first_call is second_call
        assert mp_module._model_persister is not None
    finally:
        # Cleanup - restore original
        mp_module._model_persister = original_persister
