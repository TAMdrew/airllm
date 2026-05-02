"""AirLLM ChatGLM model implementation.

Supports ChatGLMModel architecture.  ChatGLM uses a completely different
layer naming scheme (``transformer.embedding.word_embeddings``,
``transformer.encoder.layers``, etc.) and has its own rotary positional
embedding handling.  Sequence length is in dim 0 rather than dim 1.
"""

from __future__ import annotations

from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry


@ModelRegistry.register("ChatGLMModel")
class AirLLMChatGLM(AirLLMBaseModel):
    """AirLLM implementation for ChatGLM models."""

    def get_sequence_len(self, seq: Any) -> int:
        """ChatGLM stores sequence length in dim 0."""
        return seq.shape[0]

    def get_past_key_values_cache_seq_len(self, past_key_values: Any) -> int:
        """ChatGLM stores cached sequence length in dim 0."""
        return past_key_values[0][0].shape[0]

    def set_layer_names_dict(self) -> None:
        """ChatGLM uses a unique ``transformer.encoder`` naming scheme."""
        self.layer_names_dict = {
            "embed": "transformer.embedding.word_embeddings",
            "layer_prefix": "transformer.encoder.layers",
            "norm": "transformer.encoder.final_layernorm",
            "lm_head": "transformer.output_layer",
            "rotary_pos_emb": "transformer.rotary_pos_emb",
        }

    def get_pos_emb_args(self, len_p: int, len_s: int) -> dict[str, Any]:
        """Compute ChatGLM rotary positional embeddings.

        Args:
            len_p: Length of past (cached) sequence.
            len_s: Length of current input sequence.

        Returns:
            Dict with ``rotary_pos_emb`` keyword argument.
        """
        rotary_pos_emb = self.model.transformer.rotary_pos_emb(self.config.seq_length)
        rotary_pos_emb = rotary_pos_emb[None, :len_s]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        return {"rotary_pos_emb": rotary_pos_emb}

    def get_past_key_value_args(self, k_cache: Any, v_cache: Any) -> dict[str, Any]:
        """ChatGLM uses ``kv_cache`` instead of ``past_key_value``."""
        return {"kv_cache": (k_cache, v_cache)}

    def get_attention_mask_args(
        self, full_attention_mask: Any, len_p: int, len_s: int
    ) -> dict[str, Any]:
        """ChatGLM does not use an explicit attention mask."""
        return {"attention_mask": None}

    def get_position_ids_args(
        self, full_position_ids: Any, len_p: int, len_s: int
    ) -> dict[str, Any]:
        """ChatGLM does not use explicit position IDs."""
        return {}
