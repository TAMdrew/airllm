"""AirLLM — 100% local LLM inference on a single 4GB GPU.

Run 70B+ parameter models entirely on your local hardware. No cloud APIs,
no subscriptions, no data leaves your machine.  HuggingFace libraries are
used locally for config parsing, tokenization, and one-time weight download.
After initial download, everything runs offline (``HF_HUB_OFFLINE=1``).
"""

from __future__ import annotations

# Single-source version from pyproject.toml via installed package metadata.
# Eliminates DRY violation of maintaining version in two places (ARCH-4).
try:
    from importlib.metadata import version as _get_version

    __version__: str = _get_version("airllm")
except Exception:
    __version__ = "3.0.0"  # Fallback for editable installs without metadata

from .constants import IS_ON_MAC_OS
from .model_registry import ModelRegistry

if IS_ON_MAC_OS:
    try:
        from .airllm_llama_mlx import AirLLMLlamaMlx
    except ImportError:
        AirLLMLlamaMlx = None  # type: ignore[assignment,misc]
    from .auto_model import AutoModel
else:
    AirLLMLlamaMlx = None  # type: ignore[assignment,misc]  # MLX only on macOS
    from .auto_model import AutoModel

# Always import non-MLX backends (they only require torch/transformers)
from .airllm import AirLLMLlama2
from .airllm_baichuan import AirLLMBaichuan
from .airllm_base import AirLLMBaseModel
from .airllm_chatglm import AirLLMChatGLM
from .airllm_cohere import AirLLMCohere
from .airllm_deepseek import AirLLMDeepSeek
from .airllm_falcon3 import AirLLMFalcon3
from .airllm_gemma import AirLLMGemma, AirLLMGemma2, AirLLMGemma3
from .airllm_gemma4 import AirLLMGemma4
from .airllm_glm4 import AirLLMGlm4
from .airllm_granite import AirLLMGranite
from .airllm_internlm import AirLLMInternLM
from .airllm_jamba import AirLLMJamba
from .airllm_llama4 import AirLLMLlama4
from .airllm_mistral import AirLLMMistral
from .airllm_mixtral import AirLLMMixtral
from .airllm_olmo2 import AirLLMOLMo2
from .airllm_phi import AirLLMPhi
from .airllm_qwen import AirLLMQWen, AirLLMQwen
from .airllm_qwen2 import AirLLMQWen2
from .airllm_qwen3 import AirLLMQwen3
from .airllm_zamba import AirLLMZamba2
from .async_loader import AsyncLayerLoader
from .io.downloader import download_model, resolve_model_path
from .kv_cache import CompressedKVCache, KVCacheCompressor, PolarQuantConfig
from .moe_loader import ExpertRouter, MoEConfig
from .paged_kv_cache import PageConfig, PagedKVCache
from .quantization import (
    QuantizationMethod,
    detect_quantization,
    get_available_methods,
    is_gguf_available,
    is_turboquant_available,
)
from .speculative import SpeculativeConfig, estimate_speedup, verify_draft_tokens
from .utils import NotEnoughSpaceException, split_and_save_layers

__all__ = [
    # Base
    "AirLLMBaseModel",
    # KV Cache compression (TurboQuant)
    "CompressedKVCache",
    "KVCacheCompressor",
    "PolarQuantConfig",
    # Paged KV Cache
    "PageConfig",
    "PagedKVCache",
    # Speculative decoding
    "SpeculativeConfig",
    "estimate_speedup",
    "verify_draft_tokens",
    # Async layer loading
    "AsyncLayerLoader",
    # MoE expert loading
    "ExpertRouter",
    "MoEConfig",
    # Model implementations (alphabetical)
    "AirLLMBaichuan",
    "AirLLMChatGLM",
    "AirLLMCohere",
    "AirLLMDeepSeek",
    "AirLLMFalcon3",
    "AirLLMGemma",
    "AirLLMGemma2",
    "AirLLMGemma3",
    "AirLLMGemma4",
    "AirLLMGlm4",
    "AirLLMGranite",
    "AirLLMInternLM",
    "AirLLMJamba",
    "AirLLMLlama2",
    "AirLLMLlama4",
    "AirLLMLlamaMlx",
    "AirLLMMistral",
    "AirLLMMixtral",
    "AirLLMOLMo2",
    "AirLLMPhi",
    "AirLLMQWen",
    "AirLLMQWen2",
    "AirLLMQwen",
    "AirLLMQwen3",
    "AirLLMZamba2",
    # Auto-model & registry
    "AutoModel",
    "ModelRegistry",
    # Downloader (works without huggingface-hub)
    "download_model",
    "resolve_model_path",
    # Quantization
    "QuantizationMethod",
    "detect_quantization",
    "get_available_methods",
    "is_gguf_available",
    "is_turboquant_available",
    # Utilities
    "NotEnoughSpaceException",
    "__version__",
    "split_and_save_layers",
]
