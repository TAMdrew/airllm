"""AirLLM backend for AI2 OLMo 2 architecture.

OLMo 2 is a fully open-source language model from the Allen Institute for AI,
with fully reproducible training data, code, and intermediate checkpoints.

Available sizes: 1B, 7B, 13B, 32B.

Uses standard ``model.layers`` / ``model.embed_tokens`` / ``model.norm``
/ ``lm_head`` naming convention (Llama-compatible layout).
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("OLMo2ForCausalLM")
class AirLLMOLMo2(AirLLMBaseModel):
    """AirLLM implementation for AI2 OLMo 2.

    Uses standard Llama-compatible layer naming, so no overrides needed.

    Tested variants: OLMo-2-7B, OLMo-2-13B, OLMo-2-32B.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        logger.info(
            "OLMo 2 config: %s layers",
            getattr(self.config, "num_hidden_layers", "unknown"),
        )
