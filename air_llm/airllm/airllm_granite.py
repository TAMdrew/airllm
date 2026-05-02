"""AirLLM backend for IBM Granite architecture.

Granite is IBM's family of open-source enterprise LLMs, available in
3B, 8B, and 34B sizes. Designed for enterprise RAG, code generation,
and multi-language tasks.

Uses standard ``model.layers`` / ``model.embed_tokens`` / ``model.norm``
/ ``lm_head`` naming convention (Llama-compatible layout).
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("GraniteForCausalLM")
class AirLLMGranite(AirLLMBaseModel):
    """AirLLM implementation for IBM Granite.

    Uses standard Llama-compatible layer naming.

    Tested variants: Granite-3.1-8B, Granite-3.1-3B.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        logger.info(
            "Granite config: %s layers",
            getattr(self.config, "num_hidden_layers", "unknown"),
        )
