"""Test suite for ModelRegistry class.

Tests the decorator-based model registration system that maps
HuggingFace architecture names to AirLLM model classes.
"""

from __future__ import annotations

import pytest

from air_llm.airllm.airllm_base import AirLLMBaseModel
from air_llm.airllm.model_registry import ModelRegistry


class DummyModel(AirLLMBaseModel):
    """Minimal test model for registry testing."""

    pass


class AnotherDummyModel(AirLLMBaseModel):
    """Second test model for multi-registration testing."""

    pass


def test_register_single_architecture() -> None:
    """Test registering a class with a single architecture name."""
    # Arrange
    ModelRegistry._clear()

    # Act
    @ModelRegistry.register("TestArchForCausalLM")
    class TestModel(AirLLMBaseModel):
        pass

    # Assert
    retrieved_class = ModelRegistry.get("TestArchForCausalLM")
    assert retrieved_class is TestModel
    assert retrieved_class.__name__ == "TestModel"


def test_register_multiple_architectures() -> None:
    """Test registering a class with multiple architecture names."""
    # Arrange
    ModelRegistry._clear()

    # Act
    @ModelRegistry.register("Arch1ForCausalLM", "Arch2ForCausalLM", "Arch3ForCausalLM")
    class MultiArchModel(AirLLMBaseModel):
        pass

    # Assert
    for arch_name in ["Arch1ForCausalLM", "Arch2ForCausalLM", "Arch3ForCausalLM"]:
        retrieved_class = ModelRegistry.get(arch_name)
        assert retrieved_class is MultiArchModel


def test_get_unknown_architecture_raises_value_error() -> None:
    """Test that looking up an unknown architecture raises ValueError."""
    # Arrange
    ModelRegistry._clear()

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported architecture"):
        ModelRegistry.get("CompletelyUnknownArchitecture")


def test_list_supported_returns_sorted_list() -> None:
    """Test that list_supported returns a sorted list of registered names."""
    # Arrange
    ModelRegistry._clear()

    @ModelRegistry.register("ZebraArch")
    class Model1(AirLLMBaseModel):
        pass

    @ModelRegistry.register("AlphaArch")
    class Model2(AirLLMBaseModel):
        pass

    @ModelRegistry.register("MidArch")
    class Model3(AirLLMBaseModel):
        pass

    # Act
    supported = ModelRegistry.list_supported()

    # Assert
    assert supported == ["AlphaArch", "MidArch", "ZebraArch"]
    assert isinstance(supported, list)


def test_register_decorator_returns_original_class() -> None:
    """Test that the register decorator doesn't modify the class."""
    # Arrange
    ModelRegistry._clear()

    # Act
    @ModelRegistry.register("DecoratorTestArch")
    class OriginalClass(AirLLMBaseModel):
        custom_attr = "test_value"

    # Assert
    assert OriginalClass.custom_attr == "test_value"
    assert OriginalClass.__name__ == "OriginalClass"

    retrieved = ModelRegistry.get("DecoratorTestArch")
    assert retrieved is OriginalClass


def test_overwrite_registration_with_same_class_succeeds() -> None:
    """Test that re-registering the same class object to the same name succeeds."""
    # Arrange
    ModelRegistry._clear()

    class SameModel(AirLLMBaseModel):
        pass

    # Act — register the same class object twice
    ModelRegistry.register("OverwriteArch")(SameModel)
    ModelRegistry.register("OverwriteArch")(SameModel)

    # Assert - Should not raise, and should return the same class
    retrieved = ModelRegistry.get("OverwriteArch")
    assert retrieved is SameModel


def test_overwrite_registration_with_different_class_raises() -> None:
    """Test that registering a different class to an existing name raises ValueError."""
    # Arrange
    ModelRegistry._clear()

    @ModelRegistry.register("ConflictArch")
    class FirstModel(AirLLMBaseModel):
        pass

    # Act & Assert
    with pytest.raises(ValueError, match="already registered"):

        @ModelRegistry.register("ConflictArch")
        class SecondModel(AirLLMBaseModel):
            pass


def test_clear_registry_empties_registry() -> None:
    """Test that _clear() removes all registered architectures."""
    # Arrange
    ModelRegistry._clear()

    @ModelRegistry.register("ToClear1", "ToClear2")
    class ClearTestModel(AirLLMBaseModel):
        pass

    assert len(ModelRegistry.list_supported()) == 2

    # Act
    ModelRegistry._clear()

    # Assert
    assert len(ModelRegistry.list_supported()) == 0
    assert ModelRegistry._registry == {}


def test_substring_fallback_match() -> None:
    """Test that substring matching works for backward compatibility."""
    # Arrange
    ModelRegistry._clear()

    @ModelRegistry.register("QWen")
    class QWenModel(AirLLMBaseModel):
        pass

    # Act - Substring match: "QWenLMHeadModel" contains "QWen"
    retrieved = ModelRegistry.get("QWenLMHeadModel")

    # Assert
    assert retrieved is QWenModel


def test_exact_match_takes_precedence_over_substring() -> None:
    """Test that exact matches are preferred over substring matches."""
    # Arrange
    ModelRegistry._clear()

    @ModelRegistry.register("Llama")
    class SubstringMatchModel(AirLLMBaseModel):
        pass

    @ModelRegistry.register("LlamaForCausalLM")
    class ExactMatchModel(AirLLMBaseModel):
        pass

    # Act
    retrieved = ModelRegistry.get("LlamaForCausalLM")

    # Assert - Should get exact match, not substring match
    assert retrieved is ExactMatchModel


def test_registry_persists_across_lookups() -> None:
    """Test that registry state persists across multiple get() calls."""
    # Arrange
    ModelRegistry._clear()

    @ModelRegistry.register("PersistArch")
    class PersistModel(AirLLMBaseModel):
        pass

    # Act
    first_lookup = ModelRegistry.get("PersistArch")
    second_lookup = ModelRegistry.get("PersistArch")

    # Assert
    assert first_lookup is second_lookup
    assert first_lookup is PersistModel


def test_get_with_empty_registry_raises_value_error() -> None:
    """Test that get() with an empty registry provides helpful error."""
    # Arrange
    ModelRegistry._clear()

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported architecture.*Supported: \\[\\]"):
        ModelRegistry.get("AnyArchitecture")
