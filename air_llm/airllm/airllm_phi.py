"""AirLLM backend for Phi-3/Phi-4 architecture.

The Phi family of models from Microsoft are dense decoder-only transformers
that use the standard HuggingFace layer naming convention:

- ``model.embed_tokens`` for token embeddings
- ``model.layers`` for transformer blocks
- ``model.norm`` for final layer normalization
- ``lm_head`` for the language model head

Phi models use RoPE positional embeddings and are fully compatible with
AirLLM's layer-by-layer execution.

Note:
    Phi-4 uses the ``Phi3ForCausalLM`` architecture string in HuggingFace
    (not ``Phi4ForCausalLM``), but we register both for forward
    compatibility.
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("Phi3ForCausalLM", "PhiForCausalLM", "Phi4ForCausalLM")
class AirLLMPhi(AirLLMBaseModel):
    """AirLLM implementation for Phi-3/Phi-4 architecture.

    Supports layer-by-layer inference for Phi models on low-VRAM GPUs.
    Phi models are dense decoders with RoPE and standard HF layer naming.

    Tested variants: Phi-4 (14B), Phi-3 Mini/Small/Medium.

    VRAM Requirements (layer-by-layer):
        - 4GB: All Phi-3 variants (up to 14B)
        - 4GB: Phi-4 (14B)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._log_architecture_info()

    def _log_architecture_info(self) -> None:
        """Log Phi-specific configuration details."""
        num_layers = getattr(self.config, "num_hidden_layers", "unknown")
        hidden_size = getattr(self.config, "hidden_size", "unknown")
        rope_scaling = getattr(self.config, "rope_scaling", None)

        logger.info(
            "Phi config: %s layers, hidden_size=%s, rope_scaling=%s",
            num_layers,
            hidden_size,
            rope_scaling,
        )
