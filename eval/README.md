# Eval — Model Evaluation & Benchmarking Tools

> **⚠️ Research Utility**
> This directory contains evaluation and benchmarking tools for AirLLM,
> **separate from** the core inference library.
> No cloud APIs are required — everything runs locally on your hardware.

## Contents

| File                                                    | Description                                                                                                                                                   |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `benchmark.py`                                          | **AirLLM Benchmark Tool** — measures tokens/sec, TTFT, peak VRAM, and per-layer profiling for any supported model.                                            |
| `elo_tournanment_all_models_on_translated_vicuna.ipynb` | _(Legacy)_ Elo rating tournament evaluation — compares multiple Chinese LLMs (Anima 33B, Belle, Chinese Vicuna, ChatGPT-3.5) using the translated Vicuna set. |

## Benchmark Tool

The benchmark tool (`benchmark.py`) measures local LLM inference performance on your hardware.

### Quick Start

```bash
# Basic benchmark
python eval/benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# With quantization
python eval/benchmark.py --model google/gemma-4-12b --compression 4bit

# Save results to JSON
python eval/benchmark.py --model Qwen/Qwen3-8B --output results.json

# Custom prompt and token count
python eval/benchmark.py --model microsoft/phi-4 --prompt "Hello world" --max-tokens 100
```

### CLI Arguments

| Argument        | Default                                 | Description                                        |
| --------------- | --------------------------------------- | -------------------------------------------------- |
| `--model`       | _(required)_                            | HuggingFace model ID or local path                 |
| `--compression` | `None`                                  | Quantization method: `4bit`, `8bit`, `awq`, `gptq` |
| `--prompt`      | `"Explain the theory of relativity..."` | Input prompt text                                  |
| `--max-tokens`  | `50`                                    | Maximum tokens to generate                         |
| `--output`      | `None`                                  | Save results to a JSON file                        |
| `--token`       | `None`                                  | HuggingFace auth token for gated models            |

### Metrics Measured

| Metric          | Description                                 |
| --------------- | ------------------------------------------- |
| **Tokens/sec**  | Generation throughput                       |
| **TTFT**        | Time to first token (latency approximation) |
| **Peak VRAM**   | Maximum GPU memory allocated (MB)           |
| **Total time**  | End-to-end generation time (seconds)        |
| **System info** | Platform, GPU, CUDA version, Python version |

### Example Output

```
============================================================
AirLLM Benchmark Results
============================================================
Model:            TinyLlama/TinyLlama-1.1B-Chat-v1.0
Compression:      None (FP16)
Tokens generated: 50
Total time:       12.345s
Tokens/sec:       4.050
TTFT (approx):    0.247s
Peak VRAM:        1234.5 MB
Platform:         Linux-6.5.0-x86_64
GPU:              NVIDIA RTX 4090
============================================================
```

### Programmatic Usage

```python
from eval.benchmark import run_benchmark

result = run_benchmark(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=20,
    compression="4bit",
)

print(result.tokens_per_second)
print(result.to_json())
```

### Environment Variables

| Variable           | Description                                      |
| ------------------ | ------------------------------------------------ |
| `HF_HUB_OFFLINE=1` | Run fully offline (model must already be cached) |
| `AIRLLM_CACHE_DIR` | Custom cache directory for downloaded models     |

## Elo Tournament (Legacy)

The Elo tournament notebook is a standalone research evaluation tool from an earlier version of the project.

### Methodology

The evaluation uses the **Elo rating tournament** method recommended by
[QLoRA](https://arxiv.org/abs/2305.14314):

- 300 rounds of random evaluation
- Random model ordering to offset position bias
- Random seed: 42
- K=32, initial rating=1000

> **Note:** The Elo tournament notebook uses the legacy API and has not been updated
> to the v3.0.0 `AutoModel` interface. See [Features & Capabilities](../README.md#-features--capabilities)
> in the project README.

## Relationship to AirLLM

These are standalone evaluation tools.
For inference with AirLLM, see the [Quickstart Guide](../docs/QUICKSTART.md) or the [Usage Guide](../docs/USAGE_GUIDE.md).
