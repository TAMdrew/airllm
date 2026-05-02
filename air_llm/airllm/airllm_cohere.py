"""AirLLM backend for Cohere Command-R architecture.

Command-R and Command-R+ are dense decoder-only transformers from Cohere,
optimized for RAG (Retrieval-Augmented Generation) workloads.  They use
the standard HuggingFace layer naming convention:

- ``model.embed_tokens`` for token embeddings
- ``model.layers`` for transformer blocks
- ``model.norm`` for final layer normalization
- ``lm_head`` for the language model head

Command-R uses RoPE positional embeddings and is fully compatible with
AirLLM's layer-by-layer execution.

This module also registers the ``Cohere2ForCausalLM`` architecture string
for forward compatibility with the Command-R2 family.
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("CohereForCausalLM", "Cohere2ForCausalLM")
class AirLLMCohere(AirLLMBaseModel):
    """AirLLM implementation for Cohere Command-R architecture.

    Supports layer-by-layer inference for Command-R models on low-VRAM GPUs.
    Command-R is a dense decoder optimized for RAG with RoPE embeddings.

    Tested variants: Command-R (35B), Command-R+ (104B).

    VRAM Requirements (layer-by-layer):
        - 4GB: Command-R (35B)
        - 8GB: Command-R+ (104B) — larger per-layer footprint
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._log_architecture_info()

    def _log_architecture_info(self) -> None:
        """Log Cohere-specific configuration details."""
        num_layers = getattr(self.config, "num_hidden_layers", "unknown")
        hidden_size = getattr(self.config, "hidden_size", "unknown")

        logger.info(
            "Cohere config: %s layers, hidden_size=%s",
            num_layers,
            hidden_size,
        )
