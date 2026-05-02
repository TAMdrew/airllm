"""AirLLM Benchmark Tool — Measure local LLM inference performance.

Measures tokens/sec, time-to-first-token (TTFT), peak VRAM usage,
and per-layer profiling for any supported model.

Usage:
    python eval/benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python eval/benchmark.py --model google/gemma-4-12b --compression 4bit
    python eval/benchmark.py --model /path/to/local/model --max-tokens 100

Environment:
    HF_HUB_OFFLINE=1  — Run in fully offline mode (model must be cached)
    AIRLLM_CACHE_DIR   — Custom cache directory for downloaded models
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    model_id: str
    compression: str | None
    prompt: str
    max_new_tokens: int
    total_time_sec: float
    time_to_first_token_sec: float
    tokens_generated: int
    tokens_per_second: float
    peak_vram_mb: float
    system_info: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)


def get_system_info() -> dict:
    """Collect system hardware information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }

    # GPU info
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
            )
            info["cuda_version"] = torch.version.cuda or "N/A"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu"] = "Apple Silicon (MPS)"
            info["gpu_vram_gb"] = "Unified Memory"
    except ImportError:
        info["gpu"] = "torch not installed"

    return info


def get_peak_vram_mb() -> float:
    """Get peak VRAM usage in MB."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def run_benchmark(
    model_id: str,
    *,
    prompt: str = "Explain the theory of relativity in simple terms.",
    max_new_tokens: int = 50,
    compression: str | None = None,
    hf_token: str | None = None,
) -> BenchmarkResult:
    """Run a single benchmark measuring inference performance.

    Args:
        model_id: HuggingFace model ID or local path.
        prompt: Input prompt text.
        max_new_tokens: Maximum tokens to generate.
        compression: Quantization method (None, "4bit", "8bit").
        hf_token: Optional auth token for gated models.

    Returns:
        BenchmarkResult with all metrics.
    """
    from air_llm.airllm import AutoModel

    logger.info("Loading model: %s (compression=%s)", model_id, compression)

    # Reset VRAM tracking
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    # Load model
    load_start = time.perf_counter()
    model_kwargs: dict = {"profiling_mode": True}
    if compression:
        model_kwargs["compression"] = compression
    if hf_token:
        model_kwargs["hf_token"] = hf_token

    model = AutoModel.from_pretrained(model_id, **model_kwargs)
    load_time = time.perf_counter() - load_start
    logger.info("Model loaded in %.2f seconds", load_time)

    # Tokenize
    input_tokens = model.tokenizer(
        prompt,
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=128,
    )
    input_ids = input_tokens["input_ids"]

    # Move to GPU if available
    try:
        import torch

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
    except ImportError:
        pass

    # Generate with timing
    logger.info("Generating %d tokens...", max_new_tokens)
    gen_start = time.perf_counter()
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    gen_end = time.perf_counter()

    total_time = gen_end - gen_start
    tokens_generated = output.shape[1] - input_ids.shape[1]
    tokens_per_second = tokens_generated / total_time if total_time > 0 else 0

    # TTFT approximation (first token is generated after all layers process once)
    ttft = total_time / tokens_generated if tokens_generated > 0 else total_time

    result = BenchmarkResult(
        model_id=model_id,
        compression=compression,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        total_time_sec=round(total_time, 3),
        time_to_first_token_sec=round(ttft, 3),
        tokens_generated=tokens_generated,
        tokens_per_second=round(tokens_per_second, 3),
        peak_vram_mb=round(get_peak_vram_mb(), 1),
        system_info=get_system_info(),
    )

    # Print results — user-facing output uses print intentionally
    print("\n" + "=" * 60)
    print("AirLLM Benchmark Results")
    print("=" * 60)
    print(f"Model:            {result.model_id}")
    print(f"Compression:      {result.compression or 'None (FP16)'}")
    print(f"Tokens generated: {result.tokens_generated}")
    print(f"Total time:       {result.total_time_sec:.3f}s")
    print(f"Tokens/sec:       {result.tokens_per_second:.3f}")
    print(f"TTFT (approx):    {result.time_to_first_token_sec:.3f}s")
    print(f"Peak VRAM:        {result.peak_vram_mb:.1f} MB")
    print(f"Platform:         {result.system_info.get('platform', 'N/A')}")
    print(f"GPU:              {result.system_info.get('gpu', 'N/A')}")
    print("=" * 60)

    return result


def main() -> None:
    """CLI entry point for the benchmark tool."""
    parser = argparse.ArgumentParser(
        description="AirLLM Benchmark — Measure local LLM inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval/benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  python eval/benchmark.py --model google/gemma-4-12b --compression 4bit
  python eval/benchmark.py --model /path/to/local/model --max-tokens 100
  python eval/benchmark.py --model Qwen/Qwen3-8B --output results.json
        """,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--compression",
        choices=["4bit", "8bit", "awq", "gptq"],
        default=None,
        help="Quantization method",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the theory of relativity in simple terms.",
        help="Input prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--token", default=None, help="HuggingFace auth token for gated models")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = run_benchmark(
        args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        compression=args.compression,
        hf_token=args.token,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(result.to_json())
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
