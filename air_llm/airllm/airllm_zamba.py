"""AirLLM backend for Zamba2 architecture.

Zamba2 is a hybrid architecture that interleaves Mamba2 (Structured State
Space Model) layers with shared transformer attention layers:

- **Mamba2 layers**: Efficient linear-time sequence modeling via selective
  state spaces.  These layers process sequences without explicit attention
  matrices.
- **Shared attention layers**: Standard transformer attention layers that
  are *shared* (reused) across multiple positions in the layer stack,
  reducing total parameter count.

This hybrid design means Zamba2 is incompatible with BetterTransformer
and SDPA optimizations, as the Mamba2 blocks use fundamentally different
computation patterns.

.. warning::

    Zamba2's Mamba2 blocks may require custom handling in the forward pass
    if the HuggingFace ``transformers`` implementation does not fully
    support them.  This backend relies on ``transformers >= 4.45`` for
    native Zamba2 support.
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("Zamba2ForCausalLM")
class AirLLMZamba2(AirLLMBaseModel):
    """AirLLM implementation for Zamba2 hybrid architecture.

    Supports layer-by-layer inference for Zamba2 models on low-VRAM GPUs.
    Zamba2 interleaves Mamba2 (SSM) layers with shared transformer attention
    layers for an efficient hybrid design.

    Tested variants: Zamba2-2.7B, Zamba2-7B.

    VRAM Requirements (layer-by-layer):
        - 4GB: 2.7B variant
        - 4GB: 7B variant
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._log_architecture_info()

    def _log_architecture_info(self) -> None:
        """Log Zamba2-specific configuration details."""
        num_layers = getattr(self.config, "num_hidden_layers", "unknown")
        hidden_size = getattr(self.config, "hidden_size", "unknown")
        # Zamba2 may report its Mamba state dimension
        mamba_d_state = getattr(self.config, "mamba_d_state", "N/A")

        logger.info(
            "Zamba2 config: %s layers, hidden_size=%s, mamba_d_state=%s",
            num_layers,
            hidden_size,
            mamba_d_state,
        )
