"""Test suite for all model backend classes.

Tests that all model architectures are properly registered and
have expected behavior without loading actual models.
"""

from __future__ import annotations

import pytest

# Import all backends to trigger @ModelRegistry.register() decorators.
# The registry_cleanup fixture clears state before each test, so we
# need a helper to re-register all backends within the test scope.
import air_llm.airllm.airllm
import air_llm.airllm.airllm_baichuan
import air_llm.airllm.airllm_chatglm
import air_llm.airllm.airllm_cohere
import air_llm.airllm.airllm_deepseek
import air_llm.airllm.airllm_falcon3
import air_llm.airllm.airllm_gemma
import air_llm.airllm.airllm_gemma4
import air_llm.airllm.airllm_glm4
import air_llm.airllm.airllm_granite
import air_llm.airllm.airllm_internlm
import air_llm.airllm.airllm_jamba
import air_llm.airllm.airllm_llama4
import air_llm.airllm.airllm_mistral
import air_llm.airllm.airllm_mixtral
import air_llm.airllm.airllm_olmo2
import air_llm.airllm.airllm_phi
import air_llm.airllm.airllm_qwen
import air_llm.airllm.airllm_qwen2
import air_llm.airllm.airllm_qwen3
import air_llm.airllm.airllm_zamba  # noqa: F401
from air_llm.airllm.constants import IS_ON_MAC_OS
from air_llm.airllm.model_registry import ModelRegistry

# Skip all backend tests on macOS (only MLX backend is used)
pytestmark = pytest.mark.skipif(IS_ON_MAC_OS, reason="Backend tests not applicable on macOS")


@pytest.mark.parametrize(
    "arch_name,expected_class_name",
    [
        # Llama
        ("LlamaForCausalLM", "AirLLMLlama2"),
        ("LLaMAForCausalLM", "AirLLMLlama2"),
        # Gemma 4
        ("Gemma4ForCausalLM", "AirLLMGemma4"),
        ("Gemma4ForConditionalGeneration", "AirLLMGemma4"),
        # Gemma (v1/v2/v3)
        ("GemmaForCausalLM", "AirLLMGemma"),
        ("Gemma2ForCausalLM", "AirLLMGemma"),
        ("Gemma3ForCausalLM", "AirLLMGemma"),
        # Qwen 3
        ("Qwen3ForCausalLM", "AirLLMQwen3"),
        # DeepSeek
        ("DeepseekV3ForCausalLM", "AirLLMDeepSeek"),
        ("DeepseekV2ForCausalLM", "AirLLMDeepSeek"),
        # GLM 4
        ("GlmForCausalLM", "AirLLMGlm4"),
        ("Glm4ForCausalLM", "AirLLMGlm4"),
        # Phi
        ("Phi3ForCausalLM", "AirLLMPhi"),
        ("PhiForCausalLM", "AirLLMPhi"),
        # Cohere
        ("CohereForCausalLM", "AirLLMCohere"),
        ("Cohere2ForCausalLM", "AirLLMCohere"),
        # Zamba 2
        ("Zamba2ForCausalLM", "AirLLMZamba2"),
        # Mistral
        ("MistralForCausalLM", "AirLLMMistral"),
        ("Mistral4ForCausalLM", "AirLLMMistral"),
        ("Mistral3ForConditionalGeneration", "AirLLMMistral"),
        # Mixtral
        ("MixtralForCausalLM", "AirLLMMixtral"),
        # QWen v1
        ("QWenLMHeadModel", "AirLLMQWen"),
        # Qwen 2
        ("Qwen2ForCausalLM", "AirLLMQWen2"),
        # ChatGLM
        ("ChatGLMModel", "AirLLMChatGLM"),
        # Baichuan
        ("BaichuanForCausalLM", "AirLLMBaichuan"),
        # InternLM
        ("InternLMForCausalLM", "AirLLMInternLM"),
    ],
)
def test_architecture_is_registered(arch_name: str, expected_class_name: str) -> None:
    """Test that architecture names are registered in ModelRegistry."""
    # Arrange - Import modules to trigger registration
    _import_all_backends()

    # Act
    supported = ModelRegistry.list_supported()

    # Assert
    assert arch_name in supported, f"{arch_name} not in registry"


@pytest.mark.parametrize(
    "arch_name,expected_class_name",
    [
        ("LlamaForCausalLM", "AirLLMLlama2"),
        ("Gemma4ForCausalLM", "AirLLMGemma4"),
        ("Qwen3ForCausalLM", "AirLLMQwen3"),
        ("DeepseekV3ForCausalLM", "AirLLMDeepSeek"),
        ("GlmForCausalLM", "AirLLMGlm4"),
        ("Phi3ForCausalLM", "AirLLMPhi"),
        ("CohereForCausalLM", "AirLLMCohere"),
        ("Zamba2ForCausalLM", "AirLLMZamba2"),
        ("GemmaForCausalLM", "AirLLMGemma"),
        ("MistralForCausalLM", "AirLLMMistral"),
        ("MixtralForCausalLM", "AirLLMMixtral"),
        ("QWenLMHeadModel", "AirLLMQWen"),
        ("Qwen2ForCausalLM", "AirLLMQWen2"),
        ("ChatGLMModel", "AirLLMChatGLM"),
        ("BaichuanForCausalLM", "AirLLMBaichuan"),
        ("InternLMForCausalLM", "AirLLMInternLM"),
    ],
)
def test_architecture_resolves_to_correct_class(arch_name: str, expected_class_name: str) -> None:
    """Test that architecture names resolve to the expected model class."""
    # Arrange - Import modules to trigger registration
    _import_all_backends()

    # Act
    model_class = ModelRegistry.get(arch_name)

    # Assert
    assert model_class.__name__ == expected_class_name


def test_all_llama_class_has_expected_base() -> None:
    """Test that AirLLMLlama2 inherits from AirLLMBaseModel."""
    # Arrange
    from air_llm.airllm.airllm import AirLLMLlama2
    from air_llm.airllm.airllm_base import AirLLMBaseModel

    # Act & Assert
    assert issubclass(AirLLMLlama2, AirLLMBaseModel)


def test_gemma4_get_use_better_transformer_returns_false() -> None:
    """Test that Gemma4 disables BetterTransformer."""
    # Arrange
    from air_llm.airllm.airllm_gemma4 import AirLLMGemma4

    # Act - Create a minimal instance without full init
    # We test the class method directly to avoid full model initialization
    result = AirLLMGemma4.get_use_better_transformer(None)  # type: ignore[arg-type]

    # Assert
    assert result is False


def test_mistral_get_use_better_transformer_returns_false() -> None:
    """Test that Mistral disables BetterTransformer."""
    # Arrange
    from air_llm.airllm.airllm_mistral import AirLLMMistral

    # Act
    result = AirLLMMistral.get_use_better_transformer(None)  # type: ignore[arg-type]

    # Assert
    assert result is False


def test_qwen_get_use_better_transformer_returns_false() -> None:
    """Test that QWen disables BetterTransformer."""
    # Arrange
    from air_llm.airllm.airllm_qwen import AirLLMQWen

    # Act
    result = AirLLMQWen.get_use_better_transformer(None)  # type: ignore[arg-type]

    # Assert
    assert result is False


def test_llama_get_use_better_transformer_returns_true() -> None:
    """Test that Llama enables BetterTransformer by default."""
    # Arrange
    from air_llm.airllm.airllm import AirLLMLlama2

    # Act
    result = AirLLMLlama2.get_use_better_transformer(None)  # type: ignore[arg-type]

    # Assert
    assert result is True


def test_all_model_classes_imported() -> None:
    """Test that all model backend modules can be imported."""
    # Arrange & Act & Assert - Should not raise


def test_registry_contains_all_expected_architectures() -> None:
    """Test that the registry contains all expected architecture names."""
    # Arrange
    _import_all_backends()
    expected_min_count = 27  # Based on the parametrize list above

    # Act
    supported = ModelRegistry.list_supported()

    # Assert
    assert len(supported) >= expected_min_count, (
        f"Expected at least {expected_min_count} architectures, found {len(supported)}: {supported}"
    )


def test_no_duplicate_registrations() -> None:
    """Test that each architecture is registered exactly once."""
    # Arrange
    _import_all_backends()

    # Act
    supported = ModelRegistry.list_supported()

    # Assert
    # list_supported returns a list, so duplicates would show up
    assert len(supported) == len(set(supported))


# Helper function to ensure all backends are imported
def _import_all_backends() -> None:
    """Import all model backend modules to trigger @register decorators."""
    from air_llm.airllm import (
        airllm,  # noqa: F401
        airllm_baichuan,  # noqa: F401
        airllm_chatglm,  # noqa: F401
        airllm_cohere,  # noqa: F401
        airllm_deepseek,  # noqa: F401
        airllm_gemma,  # noqa: F401
        airllm_gemma4,  # noqa: F401
        airllm_glm4,  # noqa: F401
        airllm_internlm,  # noqa: F401
        airllm_mistral,  # noqa: F401
        airllm_mixtral,  # noqa: F401
        airllm_phi,  # noqa: F401
        airllm_qwen,  # noqa: F401
        airllm_qwen2,  # noqa: F401
        airllm_qwen3,  # noqa: F401
        airllm_zamba,  # noqa: F401
    )
