"""AirLLM Llama model implementation.

Supports LlamaForCausalLM and LLaMAForCausalLM architectures.
Llama is the only architecture that supports BetterTransformer and
uses a pretrained ``GenerationConfig``, so it overrides both methods.
"""

from __future__ import annotations

from transformers import GenerationConfig

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry


@ModelRegistry.register("LlamaForCausalLM", "LLaMAForCausalLM")
class AirLLMLlama2(AirLLMBaseModel):
    """AirLLM implementation for Llama-family models (Llama 2, Llama 3, etc.)."""

    def get_use_better_transformer(self) -> bool:
        """Llama supports BetterTransformer / SDPA wrapping."""
        return True

    def get_generation_config(self) -> GenerationConfig:
        """Load the pretrained generation config for Llama.

        Falls back to a bare ``GenerationConfig`` if the pretrained
        config is unavailable.
        """
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except (OSError, ValueError):
            return GenerationConfig()
