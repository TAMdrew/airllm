"""AirLLM backend for Qwen3 architecture.

Qwen3 builds on the Qwen2 foundation with several enhancements:

- **Group Query Attention (GQA)**: Reduces KV-cache memory by sharing
  key/value heads across multiple query heads.
- **Dynamic RoPE with YaRN**: Extends context length dynamically using
  Yet Another RoPE extensioN (YaRN) scaling, enabling efficient
  long-context inference without fine-tuning.
- **Standard layer naming**: Uses the same ``model.layers`` /
  ``model.embed_tokens`` / ``model.norm`` / ``lm_head`` layout as Llama.

All Qwen3 variants are compatible with AirLLM's layer-by-layer execution.
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("Qwen3ForCausalLM")
class AirLLMQwen3(AirLLMBaseModel):
    """AirLLM implementation for Qwen3 architecture.

    Supports layer-by-layer inference for Qwen3 models on low-VRAM GPUs.
    Qwen3 uses GQA and dynamic RoPE with YaRN for extended context.

    Tested variants: 0.5B, 1.5B, 4B, 8B, 14B, 32B, 72B.

    VRAM Requirements (layer-by-layer):
        - 4GB: 0.5B, 1.5B, 4B, 8B, 14B
        - 8GB: 32B, 72B (larger per-layer footprint)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._log_architecture_info()

    def _log_architecture_info(self) -> None:
        """Log Qwen3-specific configuration details."""
        num_layers = getattr(self.config, "num_hidden_layers", "unknown")
        num_kv_heads = getattr(self.config, "num_key_value_heads", "unknown")
        rope_scaling = getattr(self.config, "rope_scaling", None)

        logger.info(
            "Qwen3 config: %s layers, num_kv_heads=%s, rope_scaling=%s",
            num_layers,
            num_kv_heads,
            rope_scaling,
        )
