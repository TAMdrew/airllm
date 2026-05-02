"""AirLLM QWen (v1) model implementation.

Supports QWenLMHeadModel architecture.  QWen has a non-standard layer
naming scheme (``transformer.wte``, ``transformer.h``, etc.) and custom
rotary positional embedding handling.
"""

from __future__ import annotations

from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry


@ModelRegistry.register("QWenLMHeadModel")
class AirLLMQWen(AirLLMBaseModel):
    """AirLLM implementation for QWen v1 models."""

    def get_past_key_values_cache_seq_len(self, past_key_values: Any) -> int:
        """QWen stores KV cache with sequence length in dim 1."""
        return past_key_values[0][0].shape[1]

    def set_layer_names_dict(self) -> None:
        """QWen uses ``transformer.wte`` / ``transformer.h`` naming."""
        self.layer_names_dict = {
            "embed": "transformer.wte",
            "layer_prefix": "transformer.h",
            "norm": "transformer.ln_f",
            "lm_head": "lm_head",
        }

    def get_pos_emb_args(self, len_p: int, len_s: int) -> dict[str, Any]:
        """Compute QWen-specific rotary positional embeddings.

        Args:
            len_p: Length of past (cached) sequence.
            len_s: Length of current input sequence.

        Returns:
            Dict with ``rotary_pos_emb_list`` keyword argument.
        """
        if self.model.transformer.use_dynamic_ntk:
            ntk_alpha_list = [1.0]
        elif len_p + len_s != len_s:
            ntk_alpha_list = self.model.transformer.rotary_emb._ntk_alpha_cached_list
        else:
            ntk_alpha_list = []
            ntk_alpha = self.model.transformer.get_ntk_alpha(len_p + len_s)
            ntk_alpha_list.append(ntk_alpha)

        self.model.transformer.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
        rotary_pos_emb_list = [
            self.model.transformer.rotary_emb(len_p + len_s, ntk_alpha=ntk_alpha)
            for ntk_alpha in ntk_alpha_list
        ]
        return {"rotary_pos_emb_list": rotary_pos_emb_list}

    def get_past_key_value_args(self, k_cache: Any, v_cache: Any) -> dict[str, Any]:
        """QWen uses ``layer_past`` instead of ``past_key_value``."""
        return {"layer_past": (k_cache, v_cache)}

    def get_attention_mask_args(
        self, full_attention_mask: Any, len_p: int, len_s: int
    ) -> dict[str, Any]:
        """QWen does not use an explicit attention mask."""
        return {"attention_mask": None}

    def get_position_ids_args(
        self, full_position_ids: Any, len_p: int, len_s: int
    ) -> dict[str, Any]:
        """QWen does not use explicit position IDs."""
        return {}


# Backward-compatible alias with consistent casing (POLA fix — CODE-12).
# Users can import either AirLLMQWen or AirLLMQwen.
AirLLMQwen = AirLLMQWen
