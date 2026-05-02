"""Test suite for constants module.

Tests that all constants are properly defined and have expected types/values.
"""

from __future__ import annotations

from air_llm.airllm import constants


def test_is_on_mac_os_is_bool() -> None:
    """Test that IS_ON_MAC_OS is a boolean value."""
    # Arrange & Act
    value = constants.IS_ON_MAC_OS

    # Assert
    assert isinstance(value, bool)


def test_supported_compressions_is_tuple() -> None:
    """Test that SUPPORTED_COMPRESSIONS is a tuple."""
    # Arrange & Act
    value = constants.SUPPORTED_COMPRESSIONS

    # Assert
    assert isinstance(value, tuple)
    assert len(value) == 2
    assert "4bit" in value
    assert "8bit" in value


def test_compression_block_sizes_are_positive() -> None:
    """Test that compression block sizes are positive integers."""
    # Arrange & Act
    block_size_4bit = constants.DEFAULT_COMPRESSION_BLOCK_SIZE_4BIT
    block_size_8bit = constants.DEFAULT_COMPRESSION_BLOCK_SIZE_8BIT

    # Assert
    assert isinstance(block_size_4bit, int)
    assert isinstance(block_size_8bit, int)
    assert block_size_4bit > 0
    assert block_size_8bit > 0


def test_compression_ratios_are_valid() -> None:
    """Test that compression ratios are positive floats less than 1."""
    # Arrange & Act
    ratio_4bit = constants.COMPRESSION_RATIO_4BIT
    ratio_8bit = constants.COMPRESSION_RATIO_8BIT

    # Assert
    assert isinstance(ratio_4bit, float)
    assert isinstance(ratio_8bit, float)
    assert 0 < ratio_4bit < 1
    assert 0 < ratio_8bit < 1


def test_bytes_per_gb_value() -> None:
    """Test that BYTES_PER_GB equals 1024^3."""
    # Arrange
    expected = 1024 * 1024 * 1024

    # Act
    value = constants.BYTES_PER_GB

    # Assert
    assert value == expected
    assert isinstance(value, int)


def test_bytes_per_mb_value() -> None:
    """Test that BYTES_PER_MB equals 1024^2."""
    # Arrange
    expected = 1024 * 1024

    # Act
    value = constants.BYTES_PER_MB

    # Assert
    assert value == expected
    assert isinstance(value, int)


def test_default_max_length_is_positive() -> None:
    """Test that DEFAULT_MAX_LENGTH is a positive integer."""
    # Arrange & Act
    value = constants.DEFAULT_MAX_LENGTH

    # Assert
    assert isinstance(value, int)
    assert value > 0


def test_default_max_seq_len_is_positive() -> None:
    """Test that DEFAULT_MAX_SEQ_LEN is a positive integer."""
    # Arrange & Act
    value = constants.DEFAULT_MAX_SEQ_LEN

    # Assert
    assert isinstance(value, int)
    assert value > 0


def test_default_device_is_string() -> None:
    """Test that DEFAULT_DEVICE is a string in expected format."""
    # Arrange & Act
    value = constants.DEFAULT_DEVICE

    # Assert
    assert isinstance(value, str)
    assert value.startswith("cuda")


def test_splitted_model_dir_name_is_non_empty_string() -> None:
    """Test that SPLITTED_MODEL_DIR_NAME is a non-empty string."""
    # Arrange & Act
    value = constants.SPLITTED_MODEL_DIR_NAME

    # Assert
    assert isinstance(value, str)
    assert len(value) > 0


def test_index_file_names_are_strings() -> None:
    """Test that index file name constants are non-empty strings."""
    # Arrange & Act
    pytorch_index = constants.PYTORCH_INDEX_FILE
    safetensors_index = constants.SAFETENSORS_INDEX_FILE

    # Assert
    assert isinstance(pytorch_index, str)
    assert isinstance(safetensors_index, str)
    assert len(pytorch_index) > 0
    assert len(safetensors_index) > 0
    assert pytorch_index.endswith(".json")
    assert safetensors_index.endswith(".json")


def test_profiler_initial_min_free_mem_is_large() -> None:
    """Test that PROFILER_INITIAL_MIN_FREE_MEM is a large sentinel value."""
    # Arrange & Act
    value = constants.PROFILER_INITIAL_MIN_FREE_MEM

    # Assert
    assert isinstance(value, int)
    # Should be a very large value (1 TB sentinel)
    assert value >= 1024 * 1024 * 1024 * 1024
