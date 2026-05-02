"""AirLLM Mistral model implementation.

Supports MistralForCausalLM architecture.  Uses base class defaults
(BetterTransformer disabled, bare ``GenerationConfig``).

Note:
    Mistral Small 4 (119B MoE) uses a different architecture —
    ``Mistral4ForCausalLM`` or ``Mistral3ForConditionalGeneration`` — and
    is registered here for routing convenience, but its MoE layers may
    require the Mixtral backend pattern for optimal performance on
    very-low-VRAM systems.
"""

from __future__ import annotations

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry


@ModelRegistry.register(
    "MistralForCausalLM",
    "Mistral4ForCausalLM",
    "Mistral3ForConditionalGeneration",
)
class AirLLMMistral(AirLLMBaseModel):
    """AirLLM implementation for Mistral models.

    Covers Mistral 7B, Mistral Small 3, and Mistral Small 4 variants.
    """
