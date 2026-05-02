"""Test suite for utility functions.

Tests memory management, compression/decompression, and file operations.
Mocks external dependencies to avoid requiring GPU/CUDA.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from air_llm.airllm import utils
from air_llm.airllm.constants import BYTES_PER_GB
from air_llm.airllm.persist import ModelPersister


def test_clean_memory_does_not_raise() -> None:
    """Test that clean_memory runs without errors."""
    # Arrange & Act & Assert - Should not raise
    utils.clean_memory()


def test_clean_memory_calls_gc_collect() -> None:
    """Test that clean_memory triggers garbage collection."""
    # Arrange
    with patch("gc.collect") as mock_gc_collect:
        # Act
        utils.clean_memory()

        # Assert
        mock_gc_collect.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_clean_memory_empties_cuda_cache_when_available() -> None:
    """Test that clean_memory clears CUDA cache when available."""
    # Arrange
    with patch("torch.cuda.empty_cache") as mock_empty_cache:
        with patch("torch.cuda.is_available", return_value=True):
            # Act
            utils.clean_memory()

            # Assert
            mock_empty_cache.assert_called_once()


def test_uncompress_layer_state_dict_returns_uncompressed_unchanged() -> None:
    """Test that uncompressed state dicts are returned as-is."""
    # Arrange
    state_dict = {
        "weight": torch.randn(10, 10, dtype=torch.float16),
        "bias": torch.randn(10, dtype=torch.float16),
    }

    # Act
    result = utils.uncompress_layer_state_dict(state_dict)

    # Assert
    assert result is state_dict
    assert torch.equal(result["weight"], state_dict["weight"])
    assert torch.equal(result["bias"], state_dict["bias"])


@pytest.mark.skipif(not utils.bitsandbytes_installed, reason="bitsandbytes not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compress_layer_state_dict_4bit() -> None:
    """Test 4-bit compression creates expected keys."""
    # Arrange
    state_dict = {
        "weight": torch.randn(32, 64, dtype=torch.float16).cuda(),
    }

    # Act
    compressed = utils.compress_layer_state_dict(state_dict, compression="4bit")

    # Assert
    assert "weight" in compressed
    # Should have quant state keys
    assert any("4bit" in key for key in compressed.keys())
    # Original weight should be quantized (smaller)
    assert compressed["weight"].dtype == torch.uint8


@pytest.mark.skipif(not utils.bitsandbytes_installed, reason="bitsandbytes not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compress_layer_state_dict_8bit() -> None:
    """Test 8-bit compression creates expected keys."""
    # Arrange
    state_dict = {
        "weight": torch.randn(32, 64, dtype=torch.float16).cuda(),
    }

    # Act
    compressed = utils.compress_layer_state_dict(state_dict, compression="8bit")

    # Assert
    assert "weight" in compressed
    # Should have 8bit absmax and code keys
    assert "weight.8bit.absmax" in compressed
    assert "weight.8bit.code" in compressed


def test_compress_layer_state_dict_none_returns_unchanged() -> None:
    """Test that compression=None returns the state dict unchanged."""
    # Arrange
    state_dict = {
        "weight": torch.randn(10, 10, dtype=torch.float16),
    }

    # Act
    result = utils.compress_layer_state_dict(state_dict, compression=None)

    # Assert
    assert result is state_dict


@pytest.mark.skipif(not utils.bitsandbytes_installed, reason="bitsandbytes not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compress_uncompress_4bit_round_trip() -> None:
    """Test that 4-bit compress -> uncompress produces similar values."""
    # Arrange
    original = {
        "weight": torch.randn(32, 64, dtype=torch.float16).cuda(),
    }

    # Act
    compressed = utils.compress_layer_state_dict(original, compression="4bit")
    decompressed = utils.uncompress_layer_state_dict(compressed)

    # Assert
    assert "weight" in decompressed
    # Values should be similar (not exact due to quantization)
    mse_loss = torch.nn.functional.mse_loss(
        decompressed["weight"].float(), original["weight"].float()
    )
    rmse = torch.sqrt(mse_loss).item()
    assert rmse < 0.1, f"RMSE {rmse} too high for 4-bit compression"


@pytest.mark.skipif(not utils.bitsandbytes_installed, reason="bitsandbytes not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compress_uncompress_8bit_round_trip() -> None:
    """Test that 8-bit compress -> uncompress produces similar values."""
    # Arrange
    original = {
        "weight": torch.randn(32, 64, dtype=torch.float16).cuda(),
    }

    # Act
    compressed = utils.compress_layer_state_dict(original, compression="8bit")
    decompressed = utils.uncompress_layer_state_dict(compressed)

    # Assert
    assert "weight" in decompressed
    # Values should be similar (not exact due to quantization)
    mse_loss = torch.nn.functional.mse_loss(
        decompressed["weight"].float(), original["weight"].float()
    )
    rmse = torch.sqrt(mse_loss).item()
    assert rmse < 0.1, f"RMSE {rmse} too high for 8-bit compression"


def test_not_enough_space_exception_is_exception() -> None:
    """Test that NotEnoughSpaceException can be raised."""
    # Arrange & Act & Assert
    with pytest.raises(utils.NotEnoughSpaceException):
        raise utils.NotEnoughSpaceException("Test error")


def test_check_space_raises_when_insufficient(temp_model_dir) -> None:
    """Test that check_space raises NotEnoughSpaceException when space is low."""
    # Arrange - Mock disk_usage to return insufficient space
    with patch("shutil.disk_usage") as mock_disk_usage:
        # Simulate: 1GB total, 0.5GB used, 0.5GB free
        mock_disk_usage.return_value = (
            1 * BYTES_PER_GB,  # total
            int(0.5 * BYTES_PER_GB),  # used
            int(0.5 * BYTES_PER_GB),  # free
        )

        # glob must return file paths so os.path.getsize is called
        with patch("air_llm.airllm.utils.core.glob", return_value=["shard1.bin", "shard2.bin"]):
            with patch("os.path.getsize", return_value=int(50 * BYTES_PER_GB)):
                # Act & Assert — model needs ~100GB but only 0.5GB free
                with pytest.raises(utils.NotEnoughSpaceException, match="Not enough space"):
                    utils.check_space(temp_model_dir)


def test_check_space_succeeds_when_sufficient(temp_model_dir) -> None:
    """Test that check_space succeeds when there's enough disk space."""
    # Arrange - Mock disk_usage to return sufficient space
    with patch("shutil.disk_usage") as mock_disk_usage:
        # Simulate: 1TB total, 100GB used, 900GB free
        # Model needs 10GB
        mock_disk_usage.return_value = (
            1000 * BYTES_PER_GB,  # total
            100 * BYTES_PER_GB,  # used
            900 * BYTES_PER_GB,  # free
        )

        with patch("glob.glob", return_value=[]):
            with patch("os.path.getsize", return_value=10 * BYTES_PER_GB):
                # Act & Assert - Should not raise
                utils.check_space(temp_model_dir)


def test_load_layer_returns_state_dict(temp_model_dir) -> None:
    """Test that load_layer returns a state dict."""
    # Arrange
    mock_state_dict = {"weight": torch.randn(10, 10)}

    with patch.object(
        ModelPersister,
        "get_model_persister",
        return_value=MagicMock(load_model=MagicMock(return_value=mock_state_dict)),
    ):
        # Act
        result = utils.load_layer(temp_model_dir, "test_layer", profiling=False)

        # Assert
        assert isinstance(result, dict)
        assert "weight" in result


def test_load_layer_with_profiling_returns_tuple(temp_model_dir) -> None:
    """Test that load_layer with profiling=True returns (state_dict, time)."""
    # Arrange
    mock_state_dict = {"weight": torch.randn(10, 10)}

    with patch.object(
        ModelPersister,
        "get_model_persister",
        return_value=MagicMock(load_model=MagicMock(return_value=mock_state_dict)),
    ):
        # Act
        result = utils.load_layer(temp_model_dir, "test_layer", profiling=True)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        state_dict, elapsed_time = result
        assert isinstance(state_dict, dict)
        assert isinstance(elapsed_time, float)
        assert elapsed_time >= 0


def test_remove_real_and_linked_file_removes_regular_file(temp_model_dir) -> None:
    """Test that remove_real_and_linked_file removes a regular file."""
    import os
    from pathlib import Path

    # Arrange — resolve the path to avoid macOS /tmp → /private/var symlink mismatch
    real_dir = Path(os.path.realpath(temp_model_dir))
    test_file = real_dir / "test_file.txt"
    test_file.write_text("test content")
    assert test_file.exists()

    # Act
    utils.remove_real_and_linked_file(test_file)

    # Assert
    assert not test_file.exists()


def test_save_quant_state_to_dict_returns_dict() -> None:
    """Test that save_quant_state_to_dict returns a dictionary."""
    # Arrange
    if not utils.bitsandbytes_installed:
        pytest.skip("bitsandbytes not installed")

    mock_quant_state = MagicMock()
    mock_quant_state.quant_type = "nf4"
    mock_quant_state.absmax = torch.randn(10)
    mock_quant_state.blocksize = 64
    mock_quant_state.code = torch.randn(16)
    mock_quant_state.dtype = torch.float16
    mock_quant_state.shape = (32, 64)
    mock_quant_state.nested = False

    # Act
    result = utils.save_quant_state_to_dict(mock_quant_state, packed=False)

    # Assert
    assert isinstance(result, dict)
    assert "quant_type" in result
    assert "absmax" in result
    assert "blocksize" in result
