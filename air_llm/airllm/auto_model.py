"""AutoModel dispatcher — 100% local inference factory.

``AutoModel.from_pretrained`` reads the HuggingFace ``config.json`` (locally)
to determine the model architecture, then looks up the appropriate AirLLM
class via :pyclass:`ModelRegistry`.

All operations run entirely on your local hardware:
- First call downloads weights from HuggingFace Hub (one-time).
- Subsequent calls use cached local weights — no network required.
- Set ``HF_HUB_OFFLINE=1`` for fully air-gapped operation.
"""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoConfig

from .constants import IS_ON_MAC_OS
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Ensure all model modules are imported so their @ModelRegistry.register
# decorators execute.  These imports are intentionally side-effect-only.
if not IS_ON_MAC_OS:
    from . import (
        airllm,  # noqa: F401  — registers LlamaForCausalLM
        airllm_baichuan,  # noqa: F401
        airllm_chatglm,  # noqa: F401
        airllm_cohere,  # noqa: F401
        airllm_deepseek,  # noqa: F401
        airllm_gemma,  # noqa: F401
        airllm_gemma4,  # noqa: F401
        airllm_glm4,  # noqa: F401
        airllm_internlm,  # noqa: F401
        airllm_mistral,  # noqa: F401
        airllm_mixtral,  # noqa: F401
        airllm_phi,  # noqa: F401
        airllm_qwen,  # noqa: F401
        airllm_qwen2,  # noqa: F401
        airllm_qwen3,  # noqa: F401
        airllm_zamba,  # noqa: F401
    )


class AutoModel:
    """Factory that selects the correct AirLLM model class at runtime.

    This class should not be instantiated directly — use
    :pymeth:`AutoModel.from_pretrained` instead.
    """

    def __init__(self) -> None:
        raise OSError(
            "AutoModel is designed to be instantiated using the "
            "`AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> Any:
        """Load model weights for completely local inference.

        On macOS, this always returns an ``AirLLMLlamaMlx`` instance.
        On other platforms, the HuggingFace config ``architectures`` list is
        used to look up the correct class via ``ModelRegistry``.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID (for initial
                download) OR local directory path (for offline operation).
            *args: Forwarded to the model constructor.
            **kwargs: Forwarded to the model constructor.  ``hf_token`` is
                extracted for config loading if present.

        Keyword Args (new in v3.0.0):
            kv_compression: Enable KV cache compression.
                Options: ``"turboquant"``, ``"4bit"``, ``"3bit"``.
            speculative_config: :class:`SpeculativeConfig` or ``dict`` for
                self-speculative decoding.  Dict keys: ``exit_layer_ratio``,
                ``num_speculations``.

        Returns:
            An initialised AirLLM model instance.

        Note:
            First call downloads weights from HuggingFace Hub.
            Subsequent calls use cached local weights.
            Set ``HF_HUB_OFFLINE=1`` for fully offline operation.
        """
        if IS_ON_MAC_OS:
            from .airllm_llama_mlx import AirLLMLlamaMlx

            return AirLLMLlamaMlx(pretrained_model_name_or_path, *args, **kwargs)

        # Load HF config to determine the architecture
        # SECURITY: respect caller's trust_remote_code setting (defaults False)
        trust_remote_code = kwargs.get("trust_remote_code", False)
        config_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        hf_token = kwargs.get("hf_token")
        if hf_token is not None:
            config_kwargs["token"] = hf_token
            logger.debug("Using hf_token for config loading")

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)

        architectures = getattr(config, "architectures", None)
        if not architectures:
            logger.warning(
                "config.json has no 'architectures' field for %r — falling back to AirLLMLlama2",
                pretrained_model_name_or_path,
            )
            from .airllm import AirLLMLlama2

            return AirLLMLlama2(pretrained_model_name_or_path, *args, **kwargs)

        architecture = architectures[0]
        logger.info("Detected architecture: %s", architecture)

        try:
            model_class = ModelRegistry.get(architecture)
        except ValueError:
            logger.warning(
                "Unknown architecture %r — falling back to AirLLMLlama2",
                architecture,
            )
            from .airllm import AirLLMLlama2

            model_class = AirLLMLlama2

        return model_class(pretrained_model_name_or_path, *args, **kwargs)
