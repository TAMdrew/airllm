"""AirLLM QWen2 model implementation.

Supports Qwen2ForCausalLM and Qwen2_5ForCausalLM architectures.  QWen2 uses
the standard Llama-style layer naming and loads a pretrained
``GenerationConfig``.

Note:
    QwQ-32B uses the ``Qwen2ForCausalLM`` architecture and is automatically
    supported by this backend.
"""

from __future__ import annotations

from transformers import GenerationConfig

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry


@ModelRegistry.register("Qwen2ForCausalLM", "Qwen2_5ForCausalLM")
class AirLLMQWen2(AirLLMBaseModel):
    """AirLLM implementation for QWen2/QwQ models.

    Covers Qwen2, Qwen2.5, and QwQ variants (which share the same
    ``Qwen2ForCausalLM`` architecture string).
    """

    def get_generation_config(self) -> GenerationConfig:
        """Load the pretrained generation config for QWen2.

        Falls back to a bare ``GenerationConfig`` if the pretrained
        config is unavailable.
        """
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except (OSError, ValueError):
            return GenerationConfig()
