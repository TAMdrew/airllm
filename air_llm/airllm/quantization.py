"""Quantization backends for AirLLM — supports bitsandbytes, AWQ, GPTQ, and TurboQuant.

Provides a unified interface for loading quantized model weights during
layer-by-layer inference. Supports:

- **bitsandbytes**: 4-bit NF4 and 8-bit dynamic quantization
- **AWQ**: Activation-aware weight quantization (4x faster inference)
- **GPTQ**: Optimal brain quantization
- **Pre-quantized**: Load pre-quantized safetensor weights directly
- **TurboQuant**: KV cache compression via PolarQuant + QJL (3-bit, ICLR 2026)

Note: The built-in TurboQuant KV cache compressor lives in ``kv_cache.py``
(pure PyTorch, no external dependency). The ``turboquant`` external package
is only needed for the full PolarQuant rotation matrices.

Based on 2026 MCP research findings:

- AWQ with Marlin kernels: ~700+ tokens/sec on Qwen2.5-32B
- GPTQ with Marlin: ~712 tokens/sec
- bitsandbytes: ~168 tokens/sec
- AWQ/GPTQ offer ~4x better throughput than bitsandbytes
- TurboQuant: 8x attention speedup with 3-bit KV cache (99.5% fidelity)
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)


class QuantizationMethod(Enum):
    """Supported quantization methods.

    Each variant maps to a string identifier used in config files and CLI args.

    Note on TurboQuant: Unlike AWQ/GPTQ/bitsandbytes which compress model
    **weights**, TurboQuant compresses the **KV cache** during inference using
    PolarQuant + QJL to achieve 3-bit KV cache with 99.5% attention fidelity
    and up to 8x attention speedup (ICLR 2026).
    """

    NONE = "none"
    BITSANDBYTES_4BIT = "4bit"
    BITSANDBYTES_8BIT = "8bit"
    AWQ = "awq"
    GPTQ = "gptq"
    PRE_QUANTIZED = "pre_quantized"
    TURBOQUANT_KV = "turboquant"
    GGUF = "gguf"  # llama.cpp GGUF format


# Quantization method string aliases for user-facing APIs
_QUANT_METHOD_ALIASES: dict[str, QuantizationMethod] = {
    "none": QuantizationMethod.NONE,
    "4bit": QuantizationMethod.BITSANDBYTES_4BIT,
    "8bit": QuantizationMethod.BITSANDBYTES_8BIT,
    "awq": QuantizationMethod.AWQ,
    "gptq": QuantizationMethod.GPTQ,
    "exl2": QuantizationMethod.GPTQ,  # EXL2 uses GPTQ-style weights
    "bitsandbytes": QuantizationMethod.BITSANDBYTES_4BIT,
    "bnb": QuantizationMethod.BITSANDBYTES_4BIT,
    "pre_quantized": QuantizationMethod.PRE_QUANTIZED,
    "turboquant": QuantizationMethod.TURBOQUANT_KV,
    "turboquant_kv": QuantizationMethod.TURBOQUANT_KV,
    "gguf": QuantizationMethod.GGUF,
}


def detect_quantization(model_path: str) -> QuantizationMethod:
    """Auto-detect quantization method from model config.

    Reads ``config.json`` and ``quantization_config`` to determine the method.

    Args:
        model_path: Path to model directory.

    Returns:
        Detected :class:`QuantizationMethod`.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return QuantizationMethod.NONE

    with open(config_path) as f:
        config = json.load(f)

    quant_config = config.get("quantization_config", {})
    quant_method = quant_config.get("quant_method", "").lower()

    if quant_method == "awq":
        return QuantizationMethod.AWQ
    if quant_method in ("gptq", "exl2"):
        return QuantizationMethod.GPTQ
    if quant_method in ("bitsandbytes", "bnb"):
        bits = quant_config.get("bits", 4)
        if bits == 4:
            return QuantizationMethod.BITSANDBYTES_4BIT
        return QuantizationMethod.BITSANDBYTES_8BIT
    if quant_method == "gguf":
        return QuantizationMethod.GGUF

    # Check for GGUF files in the model directory (no config entry needed)
    gguf_files = [f for f in os.listdir(model_path) if f.endswith(".gguf")]
    if gguf_files:
        return QuantizationMethod.GGUF

    return QuantizationMethod.NONE


def parse_quantization_method(method: str) -> QuantizationMethod:
    """Parse a user-provided quantization method string.

    Args:
        method: Case-insensitive method name (e.g., ``"awq"``, ``"4bit"``).

    Returns:
        The corresponding :class:`QuantizationMethod`.

    Raises:
        ValueError: If the method string is not recognized.
    """
    normalized = method.strip().lower()
    result = _QUANT_METHOD_ALIASES.get(normalized)
    if result is None:
        valid = sorted(set(_QUANT_METHOD_ALIASES.keys()))
        msg = f"Unknown quantization method '{method}'. Valid options: {valid}"
        raise ValueError(msg)
    return result


def is_awq_available() -> bool:
    """Check if AutoAWQ is installed.

    Returns:
        True if the ``awq`` package is importable.
    """
    try:
        import awq  # noqa: F401

        return True
    except ImportError:
        return False


def is_gptq_available() -> bool:
    """Check if AutoGPTQ is installed.

    Returns:
        True if the ``auto_gptq`` package is importable.
    """
    try:
        import auto_gptq  # noqa: F401

        return True
    except ImportError:
        return False


def is_bitsandbytes_available() -> bool:
    """Check if bitsandbytes is installed.

    Returns:
        True if the ``bitsandbytes`` package is importable.
    """
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def is_turboquant_available() -> bool:
    """Check if the external TurboQuant package is installed.

    Note: AirLLM includes a built-in KV cache compressor in ``kv_cache.py``
    (``KVCacheCompressor``) that implements the core PolarQuant + QJL
    approach using pure PyTorch — no external dependency needed.

    This function checks for the *external* ``turboquant`` package, which
    provides the full PolarQuant rotation matrices and advanced features.

    Returns:
        True if the ``turboquant`` package is importable.
    """
    try:
        import turboquant  # noqa: F401

        return True
    except ImportError:
        return False


def is_gguf_available() -> bool:
    """Check if gguf Python library is installed.

    The ``gguf`` package provides reading/writing support for the
    llama.cpp GGUF model format. Install with: ``pip install airllm[gguf]``

    Returns:
        True if the ``gguf`` package is importable.
    """
    try:
        import gguf  # noqa: F401

        return True
    except ImportError:
        return False


def get_available_methods() -> list[str]:
    """List all available quantization methods on this system.

    Checks which quantization backends are installed and returns the
    corresponding method names.

    Returns:
        Sorted list of available method name strings.
    """
    methods = ["none", "pre_quantized"]

    if is_bitsandbytes_available():
        methods.extend(["4bit", "8bit"])

    if is_awq_available():
        methods.append("awq")

    if is_gptq_available():
        methods.append("gptq")

    if is_turboquant_available():
        methods.append("turboquant")

    if is_gguf_available():
        methods.append("gguf")

    return sorted(methods)


def load_awq_layer_weights(
    layer_path: str,
    device: str = "cuda",
) -> dict:
    """Load pre-quantized AWQ layer weights.

    AWQ weights are stored as quantized integers + scales.
    The Marlin kernel handles dequantization during matmul.

    Args:
        layer_path: Path to the layer safetensor file.
        device: Target device.

    Returns:
        State dict with quantized tensors.

    Raises:
        ImportError: If safetensors is not installed.
    """
    from safetensors.torch import load_file

    return load_file(layer_path, device=device)


def load_gptq_layer_weights(
    layer_path: str,
    device: str = "cuda",
) -> dict:
    """Load pre-quantized GPTQ layer weights.

    GPTQ weights use column-wise quantization with optimal rounding.

    Args:
        layer_path: Path to the layer safetensor file.
        device: Target device.

    Returns:
        State dict with quantized tensors.

    Raises:
        ImportError: If safetensors is not installed.
    """
    from safetensors.torch import load_file

    return load_file(layer_path, device=device)


def validate_quantization_backend(method: QuantizationMethod) -> None:
    """Validate that the required backend for a quantization method is installed.

    Args:
        method: The quantization method to validate.

    Raises:
        ImportError: If the required backend is not installed.
    """
    if method == QuantizationMethod.NONE:
        return
    if method == QuantizationMethod.PRE_QUANTIZED:
        return

    if (
        method in (QuantizationMethod.BITSANDBYTES_4BIT, QuantizationMethod.BITSANDBYTES_8BIT)
        and not is_bitsandbytes_available()
    ):
        raise ImportError(
            "bitsandbytes is required for 4-bit/8-bit quantization. "
            "Install with: pip install airllm[quantization]"
        )

    if method == QuantizationMethod.AWQ and not is_awq_available():
        raise ImportError(
            "AutoAWQ is required for AWQ quantization. Install with: pip install airllm[awq]"
        )

    if method == QuantizationMethod.GPTQ and not is_gptq_available():
        raise ImportError(
            "AutoGPTQ is required for GPTQ quantization. Install with: pip install airllm[gptq]"
        )

    if method == QuantizationMethod.TURBOQUANT_KV and not is_turboquant_available():
        raise ImportError(
            "TurboQuant is required for KV cache compression. Install with: pip install turboquant"
        )

    if method == QuantizationMethod.GGUF and not is_gguf_available():
        raise ImportError(
            "gguf package is required for GGUF format support. "
            "Install with: pip install airllm[gguf]"
        )
