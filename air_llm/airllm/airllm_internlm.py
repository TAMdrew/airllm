"""AirLLM InternLM model implementation.

Supports InternLMForCausalLM architecture.  Uses base class defaults
(BetterTransformer disabled, bare ``GenerationConfig``).
"""

from __future__ import annotations

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry


@ModelRegistry.register("InternLMForCausalLM")
class AirLLMInternLM(AirLLMBaseModel):
    """AirLLM implementation for InternLM models."""
