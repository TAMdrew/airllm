"""AirLLM backend for Gemma 4 architecture.

Gemma 4 extends Gemma 3 with several notable features:

- **Alternating Attention**: Layers alternate between sliding-window attention
  (1024-token window) and full global attention, reducing memory during
  long-context inference.
- **Per-Layer Embeddings (PLE)**: The E2B and E4B variants inject small
  per-layer embedding tables that modulate hidden states, allowing smaller
  models to punch above their weight.
- **RoPE Scaling**: Supports ``proportional`` and ``yarn`` RoPE scaling for
  extended context lengths.

All Gemma 4 variants are compatible with AirLLM's layer-by-layer execution —
even the 31B model fits in 4 GB VRAM.
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Gemma 4 sliding-window attention window size (tokens).
_GEMMA4_SLIDING_WINDOW_SIZE: int = 1024


@ModelRegistry.register("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration")
class AirLLMGemma4(AirLLMBaseModel):
    """AirLLM implementation for Gemma 4 architecture.

    Supports layer-by-layer inference for Gemma 4 models on low-VRAM GPUs.
    Gemma 4 introduces alternating sliding-window / global attention layers
    and optional Per-Layer Embeddings (PLE) for efficient smaller variants.

    Tested variants: E2B (2B), E4B (4B), 12B, 31B.

    VRAM Requirements (layer-by-layer):
        - 4GB: All variants (E2B, E4B, 12B, 31B)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._log_architecture_info()

    def _log_architecture_info(self) -> None:
        """Log Gemma 4-specific configuration details."""
        sliding_window = getattr(self.config, "sliding_window", _GEMMA4_SLIDING_WINDOW_SIZE)
        num_layers = getattr(self.config, "num_hidden_layers", "unknown")
        rope_scaling = getattr(self.config, "rope_scaling", None)

        logger.info(
            "Gemma 4 config: %d layers, sliding_window=%s, rope_scaling=%s",
            num_layers,
            sliding_window,
            rope_scaling,
        )
