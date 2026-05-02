"""AirLLM backend for TII Falcon 3 architecture.

Falcon 3 is a family of open-source models from the Technology Innovation
Institute (TII), available in 1B, 3B, 7B, and 10B sizes.

Uses standard ``model.layers`` / ``model.embed_tokens`` / ``model.norm``
/ ``lm_head`` naming convention (Llama-compatible layout).
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("FalconForCausalLM")
class AirLLMFalcon3(AirLLMBaseModel):
    """AirLLM implementation for TII Falcon 3.

    Uses standard Llama-compatible layer naming.

    Tested variants: Falcon-3-7B, Falcon-3-10B.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        logger.info(
            "Falcon 3 config: %s layers",
            getattr(self.config, "num_hidden_layers", "unknown"),
        )
