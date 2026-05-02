"""AirLLM backend for Meta Llama 4 architecture.

Llama 4 introduces two flagship models with Mixture-of-Experts (MoE):

- **Scout** (109B total, 17B active): 16 experts per MoE layer,
  1 expert selected per token. 10M token context window.
- **Maverick** (400B total, 17B active): 128 experts per MoE layer,
  1 expert selected per token. Optimized for quality.

Both use a standard ``model.layers`` / ``model.embed_tokens`` / ``model.norm``
/ ``lm_head`` naming convention, so the default ``ModelLayerConfig`` works
without overrides.

Key architectural features:
    - Interleaved dense + MoE layers (not all layers are MoE)
    - iRoPE: alternating no-RoPE / local-RoPE / global-RoPE attention
    - Native multimodal support (``Llama4ForConditionalGeneration``)
    - Standard tokenizer (``PreTrainedTokenizerFast``)

.. note::

    Llama 4 MoE layers are large but only 1 expert is active per token.
    With ``compression='4bit'``, Scout runs comfortably on 8 GB VRAM
    in layer-by-layer mode.
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("Llama4ForCausalLM", "Llama4ForConditionalGeneration")
class AirLLMLlama4(AirLLMBaseModel):
    """AirLLM implementation for Meta Llama 4 (Scout / Maverick).

    Supports layer-by-layer inference for Llama 4 MoE models.
    Expert routing is handled natively by the HuggingFace model
    implementation; AirLLM manages the per-layer loading/offloading.

    Tested variants: Llama-4-Scout-17B-16E (109B), Llama-4-Maverick-17B-128E (400B).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._log_architecture_info()

    def _log_architecture_info(self) -> None:
        """Log Llama 4-specific configuration details."""
        num_layers = getattr(self.config, "num_hidden_layers", "unknown")
        num_experts = getattr(self.config, "num_local_experts", "unknown")
        num_experts_per_tok = getattr(self.config, "num_experts_per_tok", "unknown")

        logger.info(
            "Llama 4 config: %s layers, %s experts, %s experts/token",
            num_layers,
            num_experts,
            num_experts_per_tok,
        )
