"""Centralized constants for the airllm package.

This module provides all magic numbers, strings, platform detection,
and offline operation configuration used throughout the codebase,
ensuring a single source of truth.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
IS_ON_MAC_OS: bool = sys.platform == "darwin"
"""True when running on macOS (Darwin)."""

# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------
OFFLINE_MODE: bool = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
"""True when HF_HUB_OFFLINE=1 is set — prevents any network calls."""

# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------
SUPPORTED_COMPRESSIONS: tuple[str, ...] = ("4bit", "8bit")
"""Compression formats supported by the bitsandbytes quantization utilities."""

# ---------------------------------------------------------------------------
# Quantization methods (all backends)
# ---------------------------------------------------------------------------
QUANTIZATION_METHODS: tuple[str, ...] = (
    "none",
    "4bit",
    "8bit",
    "awq",
    "gptq",
    "pre_quantized",
    "turboquant",
    "gguf",
)
"""All quantization methods across all backends (bnb, AWQ, GPTQ, TurboQuant, GGUF)."""

DEFAULT_COMPRESSION_BLOCK_SIZE_4BIT: int = 64
"""Default block size for NF4 quantization."""

DEFAULT_COMPRESSION_BLOCK_SIZE_8BIT: int = 2048
"""Default block size for 8-bit block-wise quantization."""

COMPRESSION_RATIO_4BIT: float = 0.2813
"""Approximate compression ratio (compressed / original) for 4-bit."""

COMPRESSION_RATIO_8BIT: float = 0.5
"""Approximate compression ratio (compressed / original) for 8-bit."""

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_LENGTH: int = 128
"""Default maximum generation length."""

DEFAULT_MAX_SEQ_LEN: int = 512
"""Default maximum sequence length for the model context window."""

DEFAULT_DEVICE: str = "cuda:0"
"""Default device for model execution."""

SPLITTED_MODEL_DIR_NAME: str = "splitted_model"
"""Default directory name for split model shards."""

# ---------------------------------------------------------------------------
# File format markers
# ---------------------------------------------------------------------------
PYTORCH_INDEX_FILE: str = "pytorch_model.bin.index.json"
"""Index file name for PyTorch checkpoint format."""

SAFETENSORS_INDEX_FILE: str = "model.safetensors.index.json"
"""Index file name for safetensors checkpoint format."""

# ---------------------------------------------------------------------------
# Profiler defaults
# ---------------------------------------------------------------------------
PROFILER_INITIAL_MIN_FREE_MEM: int = 1024 * 1024 * 1024 * 1024  # 1 TB sentinel
"""Initial sentinel value for minimum free memory tracking."""

# ---------------------------------------------------------------------------
# Bytes-to-human helpers
# ---------------------------------------------------------------------------
BYTES_PER_GB: int = 1024 * 1024 * 1024
"""Number of bytes in one gigabyte."""

BYTES_PER_MB: int = 1024 * 1024
"""Number of bytes in one megabyte."""

# ---------------------------------------------------------------------------
# Bandwidth constants (from 2026 MCP research)
# ---------------------------------------------------------------------------
PCIE_4_BANDWIDTH_GBPS: float = 16.0
"""PCIe 4.0 unidirectional bandwidth in GB/s."""

PCIE_5_BANDWIDTH_GBPS: float = 32.0
"""PCIe 5.0 unidirectional bandwidth in GB/s."""

NVME_GEN4_BANDWIDTH_GBPS: float = 7.0
"""NVMe Gen4 sequential read bandwidth in GB/s."""

NVME_GEN5_BANDWIDTH_GBPS: float = 14.0
"""NVMe Gen5 sequential read bandwidth in GB/s."""

DDR5_BANDWIDTH_GBPS: float = 76.8
"""DDR5 dual-channel (DDR5-4800) bandwidth in GB/s."""

# ---------------------------------------------------------------------------
# Apple Silicon memory bandwidth (GB/s) — unified memory architecture
# See docs/HARDWARE_GUIDE.md for full chip comparison table.
# ---------------------------------------------------------------------------
APPLE_M1_BANDWIDTH_GBPS: float = 68.0
"""Apple M1 unified memory bandwidth in GB/s."""

APPLE_M1_PRO_BANDWIDTH_GBPS: float = 200.0
"""Apple M1 Pro unified memory bandwidth in GB/s."""

APPLE_M1_MAX_BANDWIDTH_GBPS: float = 400.0
"""Apple M1 Max unified memory bandwidth in GB/s."""

APPLE_M1_ULTRA_BANDWIDTH_GBPS: float = 800.0
"""Apple M1 Ultra unified memory bandwidth in GB/s."""

APPLE_M4_BANDWIDTH_GBPS: float = 120.0
"""Apple M4 unified memory bandwidth in GB/s."""

APPLE_M4_PRO_BANDWIDTH_GBPS: float = 273.0
"""Apple M4 Pro unified memory bandwidth in GB/s."""

APPLE_M4_MAX_BANDWIDTH_GBPS: float = 546.0
"""Apple M4 Max unified memory bandwidth in GB/s."""

# ---------------------------------------------------------------------------
# AMD Strix Halo bandwidth
# ---------------------------------------------------------------------------
AMD_STRIX_HALO_BANDWIDTH_GBPS: float = 215.0
"""AMD Ryzen AI Max+ (Strix Halo) LPDDR5X memory bandwidth in GB/s."""

# ---------------------------------------------------------------------------
# KV Cache compression defaults (TurboQuant — ICLR 2026)
# ---------------------------------------------------------------------------
DEFAULT_KV_CACHE_BITS: int = 3
"""Default bit width for PolarQuant KV cache compression (2, 3, or 4).

3-bit is the TurboQuant default: provides ~5x compression with 99.5%
attention fidelity. The rotation-based approach eliminates per-block
scale/zero-point overhead that traditional methods require.
"""

DEFAULT_KV_CACHE_GROUP_SIZE: int = 128
"""Legacy group size — unused by PolarQuant (kept for backward compat)."""

DEFAULT_QJL_DIM: int = 64
"""Default number of QJL projection dimensions (sign bits per vector).

Higher values improve compressed-domain attention accuracy but use more
memory. Set to 0 to disable QJL projections and use standard attention.
64 provides a good balance: ~8x attention speedup with <1% quality loss.
"""

DEFAULT_ROTATION_SEED: int = 42
"""Default seed for TurboQuant rotation/projection matrix generation.

Using a fixed seed ensures that compress/decompress produce consistent
results across different calls and process restarts.
"""

# ---------------------------------------------------------------------------
# Supported model architectures (HuggingFace config.architectures values)
# ---------------------------------------------------------------------------
SUPPORTED_ARCHITECTURES: tuple[str, ...] = (
    # Llama family
    "LlamaForCausalLM",
    "LLaMAForCausalLM",
    "Llama4ForCausalLM",
    "Llama4ForConditionalGeneration",
    # Qwen family
    "QWenLMHeadModel",
    "Qwen2ForCausalLM",
    "Qwen2_5ForCausalLM",
    "Qwen3ForCausalLM",
    # Gemma family
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    # Mistral family
    "MistralForCausalLM",
    "Mistral4ForCausalLM",
    "Mistral3ForConditionalGeneration",
    "MixtralForCausalLM",
    # DeepSeek
    "DeepseekV3ForCausalLM",
    "DeepseekV2ForCausalLM",
    # Phi family
    "Phi3ForCausalLM",
    "PhiForCausalLM",
    "Phi4ForCausalLM",
    # Cohere
    "CohereForCausalLM",
    # ChatGLM / GLM
    "ChatGLMModel",
    "Glm4ForCausalLM",
    # Others
    "BaichuanForCausalLM",
    "InternLMForCausalLM",
    "Zamba2ForCausalLM",
    # New in v3.1
    "FalconForCausalLM",
    "GraniteForCausalLM",
    "JambaForCausalLM",
    "OLMo2ForCausalLM",
)
"""All HuggingFace architecture strings supported by AirLLM model backends."""
