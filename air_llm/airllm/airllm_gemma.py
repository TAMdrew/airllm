"""AirLLM Gemma model implementations.

Supports Gemma, Gemma 2, and Gemma 3 architectures.
All Gemma variants use base class defaults (BetterTransformer disabled,
bare ``GenerationConfig``).

Note:
    Gemma 4 is implemented in :pymod:`airllm_gemma4` with dedicated
    support for alternating attention and Per-Layer Embeddings (PLE).
"""

from __future__ import annotations

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry


@ModelRegistry.register("GemmaForCausalLM")
class AirLLMGemma(AirLLMBaseModel):
    """AirLLM implementation for Gemma models."""


@ModelRegistry.register("Gemma2ForCausalLM")
class AirLLMGemma2(AirLLMBaseModel):
    """AirLLM implementation for Gemma 2 models."""


@ModelRegistry.register("Gemma3ForCausalLM", "Gemma3ForConditionalGeneration")
class AirLLMGemma3(AirLLMBaseModel):
    """AirLLM implementation for Gemma 3 models."""
