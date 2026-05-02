"""AirLLM backend for DeepSeek V2/V3 architecture.

DeepSeek V3 and R1 are 671B-parameter Mixture-of-Experts (MoE) models with
several unique architectural features:

- **MoE with 256 routed experts + 1 shared expert**: Each MoE layer selects
  a small subset of experts per token, keeping compute tractable despite
  the enormous parameter count.
- **Multi-head Latent Attention (MLA)**: Compresses key-value projections
  into a low-rank latent space, dramatically reducing KV-cache memory.
- **Standard layer naming**: Despite the architectural complexity, DeepSeek
  uses the standard ``model.layers`` / ``model.embed_tokens`` / ``model.norm``
  / ``lm_head`` naming convention.

.. warning::

    DeepSeek V3/R1 MoE layers are very large (~22 GB per layer in FP16).
    Systems with < 16 GB VRAM will experience significant disk I/O overhead
    during layer-by-layer execution.  Consider using ``compression='4bit'``
    to reduce per-layer memory footprint.

Note:
    DeepSeek R1-Distill variants (e.g. R1-Distill-Qwen-32B) use the
    underlying Qwen or Llama architecture and should be loaded via
    their respective backends, not this one.
"""

from __future__ import annotations

import logging
from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("DeepseekV3ForCausalLM", "DeepseekV2ForCausalLM")
class AirLLMDeepSeek(AirLLMBaseModel):
    """AirLLM implementation for DeepSeek V2/V3 architecture.

    Supports layer-by-layer inference for DeepSeek MoE models.  The MLA
    attention mechanism and expert routing are handled natively by the
    HuggingFace model implementation; AirLLM only manages the per-layer
    loading/offloading cycle.

    Tested variants: DeepSeek-V3 (671B), DeepSeek-R1 (671B).

    VRAM Requirements (layer-by-layer):
        - 16GB+: Recommended (MoE layers are ~22 GB in FP16)
        - 8GB: Possible with ``compression='4bit'``
        - 4GB: Possible with ``compression='4bit'`` but very slow

    Note:
        DeepSeek R1-Distill variants use Llama or Qwen backends.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._log_architecture_info()

    def _log_architecture_info(self) -> None:
        """Log DeepSeek-specific configuration details."""
        num_layers = getattr(self.config, "num_hidden_layers", "unknown")
        num_experts = getattr(self.config, "n_routed_experts", "unknown")
        num_shared = getattr(self.config, "n_shared_experts", "unknown")
        num_experts_per_tok = getattr(self.config, "num_experts_per_tok", "unknown")

        logger.info(
            "DeepSeek config: %s layers, %s routed experts, %s shared experts, %s experts/token",
            num_layers,
            num_experts,
            num_shared,
            num_experts_per_tok,
        )
