"""AirLLM backend for AI21 Jamba architecture.

Jamba is a hybrid SSM-Transformer model from AI21 Labs that interleaves
Mamba (state-space model) layers with standard Transformer attention layers.
This hybrid design provides:

- **Long context**: Efficient O(n) SSM layers for long-range dependencies
- **Quality**: Standard attention layers where fine-grained reasoning matters
- **MoE**: Some layers use Mixture-of-Experts for capacity

Available sizes: Jamba-1.5-Mini (52B total, 12B active),
Jamba-1.5-Large (398B total, 94B active).

Uses standard ``model.layers`` / ``model.embed_tokens`` / ``model.norm``
/ ``lm_head`` naming convention.

.. note::

    Jamba requires ``trust_remote_code=True`` for the custom Mamba layer
    implementations in HuggingFace transformers.
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("JambaForCausalLM")
class AirLLMJamba(AirLLMBaseModel):
    """AirLLM implementation for AI21 Jamba (SSM + Transformer hybrid).

    Supports layer-by-layer inference for Jamba hybrid models.
    Both Mamba (SSM) and Transformer attention layers are loaded
    and offloaded identically in the layer-by-layer loop.

    .. warning::

        Jamba models require ``trust_remote_code=True`` due to custom
        Mamba layer implementations.

    Tested variants: Jamba-1.5-Mini (52B), Jamba-1.5-Large (398B).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Jamba requires trust_remote_code for custom Mamba layers.
        # SEC: Log a warning since this enables arbitrary code execution
        # from the HuggingFace repo (ARCH-7).
        kwargs.setdefault("trust_remote_code", True)
        if kwargs.get("trust_remote_code"):
            logger.warning(
                "Jamba requires trust_remote_code=True for custom Mamba layers. "
                "This allows arbitrary code execution from the model repository. "
                "Only use models from trusted sources."
            )
        super().__init__(*args, **kwargs)
        self._log_architecture_info()

    def _log_architecture_info(self) -> None:
        """Log Jamba-specific configuration details."""
        num_layers = getattr(self.config, "num_hidden_layers", "unknown")
        num_experts = getattr(self.config, "num_experts", "unknown")
        num_experts_per_tok = getattr(self.config, "num_experts_per_tok", "unknown")

        logger.info(
            "Jamba config: %s layers (SSM+Transformer hybrid), %s experts, %s experts/token",
            num_layers,
            num_experts,
            num_experts_per_tok,
        )
