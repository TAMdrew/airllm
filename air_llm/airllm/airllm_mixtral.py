"""AirLLM Mixtral (Mixture of Experts) model implementation.

Supports MixtralForCausalLM architecture.  Uses base class defaults
(BetterTransformer disabled, bare ``GenerationConfig``).
"""

from __future__ import annotations

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry


@ModelRegistry.register("MixtralForCausalLM")
class AirLLMMixtral(AirLLMBaseModel):
    """AirLLM implementation for Mixtral MoE models."""
