"""AirLLM Baichuan model implementation.

Supports BaichuanForCausalLM architecture.  Uses a custom tokenizer
(``BaichuanTokenizer``) as a workaround for upstream tokenizer bugs.
"""

from __future__ import annotations

from typing import Any

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry
from .tokenization_baichuan import BaichuanTokenizer


@ModelRegistry.register("BaichuanForCausalLM")
class AirLLMBaichuan(AirLLMBaseModel):
    """AirLLM implementation for Baichuan models."""

    def get_tokenizer(self, hf_token: Any = None) -> BaichuanTokenizer:
        """Load the Baichuan-specific tokenizer.

        Uses ``BaichuanTokenizer`` as a workaround for:
        https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/discussions/2

        Args:
            hf_token: Ignored (Baichuan tokenizer does not need auth).

        Returns:
            A ``BaichuanTokenizer`` instance.
        """
        # Baichuan tokenizer inherently requires trust_remote_code for its
        # custom tokenization logic. Use the instance setting.
        return BaichuanTokenizer.from_pretrained(
            self.model_local_path, use_fast=False, trust_remote_code=self.trust_remote_code
        )
