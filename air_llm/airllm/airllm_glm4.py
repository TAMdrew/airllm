"""AirLLM backend for GLM-4 architecture.

GLM-4 is a standard HuggingFace-compatible causal language model that no
longer requires ``trust_remote_code=True`` (unlike its predecessor ChatGLM).
It uses the standard layer naming convention:

- ``model.embed_tokens`` for token embeddings
- ``model.layers`` for transformer blocks
- ``model.norm`` for final layer normalization
- ``lm_head`` for the language model head

GLM-4 uses RoPE positional embeddings and is fully compatible with
AirLLM's layer-by-layer execution.

Note:
    The older ChatGLM models (GLM-3 and earlier) use a completely different
    architecture with ``transformer.encoder.layers`` naming.  Those are
    handled by :pymod:`airllm_chatglm`.
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("GlmForCausalLM", "Glm4ForCausalLM", "ChatGLM4Model")
class AirLLMGlm4(AirLLMBaseModel):
    """AirLLM implementation for GLM-4 architecture.

    Supports layer-by-layer inference for GLM-4 models on low-VRAM GPUs.
    GLM-4 uses the standard HuggingFace architecture with RoPE embeddings.

    Tested variants: GLM-4-9B.

    VRAM Requirements (layer-by-layer):
        - 4GB: 9B variant
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._log_architecture_info()

    def _log_architecture_info(self) -> None:
        """Log GLM-4-specific configuration details."""
        num_layers = getattr(self.config, "num_hidden_layers", "unknown")
        hidden_size = getattr(self.config, "hidden_size", "unknown")

        logger.info(
            "GLM-4 config: %s layers, hidden_size=%s",
            num_layers,
            hidden_size,
        )
