# 🌬️ AirLLM v3.0

> Run any large language model on any GPU — even with just 4 GB VRAM.

[![PyPI](https://img.shields.io/pypi/v/airllm)](https://pypi.org/project/airllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Downloads](https://static.pepy.tech/personalized-badge/airllm?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/airllm)

## Table of Contents

- [🌬️ AirLLM v3.0](#️-airllm-v30)
  - [Table of Contents](#table-of-contents)
  - [✨ What is AirLLM?](#-what-is-airllm)
  - [🏠 Why Local?](#-why-local)
  - [🚀 Quick Start](#-quick-start)
  - [📋 Supported Models](#-supported-models)
    - [Model Families](#model-families)
  - [💻 VRAM Requirements](#-vram-requirements)
  - [💻 Hardware Guide](#-hardware-guide)
    - [Quick Reference](#quick-reference)
  - [🔧 Installation](#-installation)
    - [Standard Installation](#standard-installation)
    - [With Quantization Support](#with-quantization-support)
    - [With macOS MLX Support](#with-macos-mlx-support)
    - [Development Installation](#development-installation)
    - [Prerequisites](#prerequisites)
  - [📖 Usage Examples](#-usage-examples)
    - [Basic Inference](#basic-inference)
    - [With 4-bit Quantization (3× Faster)](#with-4-bit-quantization-3-faster)
    - [Running Gemma 4](#running-gemma-4)
    - [Running Qwen 3](#running-qwen-3)
    - [Running DeepSeek R1](#running-deepseek-r1)
    - [macOS with MLX](#macos-with-mlx)
    - [Gated Models (HF Token for Download Only)](#gated-models-hf-token-for-download-only)
  - [⚙️ Configuration](#️-configuration)
    - [Compression Details](#compression-details)
  - [🏗️ Architecture](#️-architecture)
  - [✅ Features \& Capabilities](#-features--capabilities)
  - [🧪 Testing](#-testing)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)
  - [📚 Citing AirLLM](#-citing-airllm)

## ✨ What is AirLLM?

AirLLM enables **completely local, free LLM inference**. No API keys, no subscriptions, no cloud dependencies. Your models run entirely on your hardware.

AirLLM is a **layer-by-layer inference engine** that enables running 70B+ parameter LLMs on consumer GPUs with as little as 4 GB VRAM.
Instead of loading the entire model into GPU memory, AirLLM loads one transformer layer at a time, runs the forward pass, then offloads it before loading the next layer.

**Key features:**

- **100% local** — no cloud APIs, no subscriptions, no internet required after initial download.
- **No model degradation** — runs full-precision weights (FP16) with zero accuracy loss.
- **Optional quantization** — 4-bit and 8-bit block-wise compression for up to 3× faster inference.
- **30 supported architectures** — from Llama 3 and Gemma 4 to DeepSeek R1 and Qwen 3.
- **macOS support** — Apple Silicon inference via MLX backend.
- **Prefetching** — overlaps disk I/O with GPU compute for improved throughput.
- **Drop-in API** — follows the HuggingFace `transformers` `GenerationMixin` interface.

## 🏠 Why Local?

Running LLMs locally with AirLLM provides significant advantages over cloud API services:

|                  | **AirLLM (Local)**              | **Cloud APIs (OpenAI, Anthropic, etc.)** |
| ---------------- | ------------------------------- | ---------------------------------------- |
| **Cost**         | Free after hardware             | $20–200+/month or per-token fees         |
| **Privacy**      | Data never leaves your machine  | Data sent to third-party servers         |
| **Availability** | Works offline, no rate limits   | Requires internet, subject to outages    |
| **Control**      | Full model access, any prompt   | Content filtering, terms of service      |
| **Models**       | 30+ architectures, any HF model | Limited to provider's offerings          |

> **HuggingFace** libraries (`transformers`, `safetensors`, `huggingface-hub`) run entirely locally on your machine — they are used for model config parsing, tokenization, and one-time weight download. No cloud APIs, no paid services, no data leaves your machine. After the first download, everything runs 100% offline (set `HF_HUB_OFFLINE=1`).
>
> **Unlike server-based tools** (e.g., Ollama), AirLLM runs as a native Python library with no daemon process, no separate server, and no port management. Just `import` and run.

## 🚀 Quick Start

```bash
pip install airllm
```

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

input_tokens = model.tokenizer(
    ["What is the capital of France?"],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

> **Note:** On first run, the model is decomposed and saved layer-wise.
> Ensure sufficient disk space in your HuggingFace cache directory.

For a step-by-step walkthrough, see the [Quickstart Guide](docs/QUICKSTART.md).

## 📋 Supported Models

AirLLM v3.0 supports **33 architecture strings** across **23 model backends**.
[`AutoModel.from_pretrained()`](air_llm/airllm/auto_model.py:57) automatically detects the correct backend from the HuggingFace `config.json`.

### Model Families

| Family           | Backend Class                                            | Architecture Strings                                                            | Example Models                        | Min VRAM      |
| ---------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------- | ------------- |
| **Llama**        | [`AirLLMLlama2`](air_llm/airllm/airllm.py:17)            | `LlamaForCausalLM`, `LLaMAForCausalLM`                                          | Llama 3.1 8B/70B/405B                 | 4 GB          |
| **Gemma**        | [`AirLLMGemma`](air_llm/airllm/airllm_gemma.py:24)       | `GemmaForCausalLM`                                                              | Gemma 7B                              | 4 GB          |
| **Gemma 2**      | [`AirLLMGemma2`](air_llm/airllm/airllm_gemma.py:40)      | `Gemma2ForCausalLM`                                                             | Gemma 2 9B/27B                        | 4 GB          |
| **Gemma 3**      | [`AirLLMGemma3`](air_llm/airllm/airllm_gemma.py:56)      | `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration`                           | Gemma 3 4B/12B/27B                    | 4 GB          |
| **Gemma 4**      | [`AirLLMGemma4`](air_llm/airllm/airllm_gemma4.py:35)     | `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`                           | Gemma 4 E2B/E4B/12B/31B               | 4 GB          |
| **Qwen**         | [`AirLLMQWen`](air_llm/airllm/airllm_qwen.py:19)         | `QWenLMHeadModel`                                                               | Qwen 7B/14B/72B                       | 4 GB          |
| **Qwen 2 / 2.5** | [`AirLLMQWen2`](air_llm/airllm/airllm_qwen2.py:21)       | `Qwen2ForCausalLM`, `Qwen2_5ForCausalLM`                                        | Qwen2.5 7B/72B, QwQ-32B               | 4 GB          |
| **Qwen 3**       | [`AirLLMQwen3`](air_llm/airllm/airllm_qwen3.py:30)       | `Qwen3ForCausalLM`                                                              | Qwen3 8B/32B/72B                      | 4 GB          |
| **DeepSeek**     | [`AirLLMDeepSeek`](air_llm/airllm/airllm_deepseek.py:42) | `DeepseekV3ForCausalLM`, `DeepseekV2ForCausalLM`                                | DeepSeek-V3, DeepSeek-R1 (671B)       | 16 GB         |
| **Mistral**      | [`AirLLMMistral`](air_llm/airllm/airllm_mistral.py:29)   | `MistralForCausalLM`, `Mistral4ForCausalLM`, `Mistral3ForConditionalGeneration` | Mistral 7B, Mistral Small 3/4         | 4 GB          |
| **Mixtral**      | [`AirLLMMixtral`](air_llm/airllm/airllm_mixtral.py:18)   | `MixtralForCausalLM`                                                            | Mixtral 8x7B, 8x22B                   | 4 GB          |
| **GLM-4**        | [`AirLLMGlm4`](air_llm/airllm/airllm_glm4.py:35)         | `GlmForCausalLM`, `Glm4ForCausalLM`, `ChatGLM4Model`                            | GLM-4-9B                              | 4 GB          |
| **ChatGLM**      | [`AirLLMChatGLM`](air_llm/airllm/airllm_chatglm.py:20)   | `ChatGLMModel`                                                                  | ChatGLM3-6B                           | 4 GB          |
| **Phi**          | [`AirLLMPhi`](air_llm/airllm/airllm_phi.py:34)           | `Phi3ForCausalLM`, `PhiForCausalLM`, `Phi4ForCausalLM`                          | Phi-4 14B, Phi-3 Mini/Small           | 4 GB          |
| **Cohere**       | [`AirLLMCohere`](air_llm/airllm/airllm_cohere.py:33)     | `CohereForCausalLM`, `Cohere2ForCausalLM`                                       | Command-R 35B, Command-R+ 104B        | 4 GB          |
| **Zamba 2**      | [`AirLLMZamba2`](air_llm/airllm/airllm_zamba.py:39)      | `Zamba2ForCausalLM`                                                             | Zamba2-2.7B, Zamba2-7B                | 4 GB          |
| **Baichuan**     | [`AirLLMBaichuan`](air_llm/airllm/airllm_baichuan.py:19) | `BaichuanForCausalLM`                                                           | Baichuan2-7B/13B                      | 4 GB          |
| **InternLM**     | [`AirLLMInternLM`](air_llm/airllm/airllm_internlm.py:18) | `InternLMForCausalLM`                                                           | InternLM-20B                          | 4 GB          |
| **Llama 4**      | [`AirLLMLlama4`](air_llm/airllm/airllm_llama4.py)        | `Llama4ForCausalLM`, `Llama4ForConditionalGeneration`                           | Llama 4 Scout (109B), Maverick (400B) | 16 GB         |
| **Falcon 3**     | [`AirLLMFalcon3`](air_llm/airllm/airllm_falcon3.py)      | `FalconForCausalLM`                                                             | Falcon 3 1B/3B/7B/10B                 | 4 GB          |
| **OLMo 2**       | [`AirLLMOLMo2`](air_llm/airllm/airllm_olmo2.py)          | `OLMo2ForCausalLM`                                                              | OLMo 2 7B/13B/32B                     | 4 GB          |
| **Granite**      | [`AirLLMGranite`](air_llm/airllm/airllm_granite.py)      | `GraniteForCausalLM`                                                            | Granite 3B/8B/34B                     | 4 GB          |
| **Jamba**        | [`AirLLMJamba`](air_llm/airllm/airllm_jamba.py)          | `JambaForCausalLM`                                                              | Jamba 1.5 Mini (52B), Large (398B)    | 16 GB         |
| **Llama MLX**    | [`AirLLMLlamaMlx`](air_llm/airllm/airllm_llama_mlx.py)   | _(macOS only)_                                                                  | Any Llama-compatible model            | Apple Silicon |

For detailed model information including prompt formats and per-variant VRAM, see [Supported Models Reference](docs/SUPPORTED_MODELS.md).

## 💻 VRAM Requirements

AirLLM's layer-by-layer approach means VRAM usage depends on the **largest single layer**, not the total model size.
Most dense models work on 4 GB VRAM.
MoE models require more due to their expert-heavy layers.

| VRAM Tier  | Dense Models                                                        | MoE Models                                               | Notes                                |
| ---------- | ------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------ |
| **4 GB**   | Llama ≤405B, Gemma ≤31B, Qwen ≤72B, Phi ≤14B, GLM-4, Cohere, Zamba2 | —                                                        | FP16, no quantization needed         |
| **8 GB**   | All dense models                                                    | Mixtral 8x7B (with 4-bit)                                | Comfortable headroom for most models |
| **12 GB**  | All dense models                                                    | Mixtral 8x22B (with 4-bit), Mistral Small 4 (with 4-bit) | —                                    |
| **16 GB**  | All dense models                                                    | DeepSeek-V3/R1 (with 4-bit)                              | Recommended for DeepSeek MoE         |
| **24 GB+** | All dense models                                                    | DeepSeek-V3/R1 (FP16)                                    | MoE layers are ~22 GB each in FP16   |

> **Tip:** Use `compression='4bit'` to reduce per-layer memory by ~70%, enabling larger MoE models on smaller GPUs.

## 💻 Hardware Guide

See [docs/HARDWARE_GUIDE.md](docs/HARDWARE_GUIDE.md) for comprehensive recommendations covering Apple Silicon (M1–M4), AMD Ryzen AI Max+, and NVIDIA GPUs.

### Quick Reference

| Hardware                         | Best For                                |
| -------------------------------- | --------------------------------------- |
| Mac Mini M4 Pro (64GB)           | 70B models locally, no layer offloading |
| Mac Studio M3 Ultra (512GB)      | 405B–671B models, enterprise local AI   |
| AMD Ryzen AI Max+ laptop (128GB) | Portable 70B inference at ~$2,500       |
| RTX 4090 (24GB)                  | Fastest layer-by-layer with AWQ         |
| RTX 4060 (8GB)                   | Budget layer-by-layer inference         |

## 🔧 Installation

### Standard Installation

```bash
pip install airllm
```

### With Quantization Support

```bash
pip install "airllm[quantization]"
```

### With macOS MLX Support

```bash
pip install "airllm[mlx]"
```

### Development Installation

```bash
git clone https://github.com/lyogavin/airllm.git
cd airllm
pip install -e ".[dev]"
```

### Prerequisites

- Python 3.10+
- PyTorch 2.6+
- CUDA-capable GPU (Linux/Windows) or Apple Silicon Mac
- Sufficient disk space for model shards (typically 2× model size on first run)

## 📖 Usage Examples

### Basic Inference

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

input_tokens = model.tokenizer(
    ["Explain quantum computing in simple terms."],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=50,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### With 4-bit Quantization (3× Faster)

```bash
pip install bitsandbytes
```

```python
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    compression="4bit",  # or "8bit"
)

input_tokens = model.tokenizer(
    ["What are the benefits of renewable energy?"],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=50,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### Running Gemma 4

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("google/gemma-4-12b-it")

input_tokens = model.tokenizer(
    ["Write a haiku about machine learning."],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=True,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=30,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### Running Qwen 3

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("Qwen/Qwen3-8B")

input_tokens = model.tokenizer(
    ["Describe the water cycle."],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=50,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### Running DeepSeek R1

```python
from airllm import AutoModel

# DeepSeek R1 MoE — recommend 16 GB+ VRAM or use 4-bit compression
model = AutoModel.from_pretrained(
    "deepseek-ai/DeepSeek-R1",
    compression="4bit",
)

input_tokens = model.tokenizer(
    ["Solve: What is 25 * 37?"],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=100,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### macOS with MLX

```python
from airllm import AutoModel

# On macOS with Apple Silicon, AutoModel automatically uses the MLX backend
model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

input_tokens = model.tokenizer(
    ["What is the meaning of life?"],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"],  # No .cuda() needed on macOS
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### Gated Models (HF Token for Download Only)

Some models (Llama 3, Gemma) require accepting a license on HuggingFace before downloading.
The token is used **only for the initial weight download** — inference runs entirely locally with no internet.

```python
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    hf_token="hf_YOUR_TOKEN_HERE",  # Only needed once to download weights
)
```

## ⚙️ Configuration

All configuration is passed through [`AutoModel.from_pretrained()`](air_llm/airllm/auto_model.py:57):

| Parameter                  | Type          | Default         | Description                                                                                    |
| -------------------------- | ------------- | --------------- | ---------------------------------------------------------------------------------------------- |
| `compression`              | `str \| None` | `None`          | `"4bit"` or `"8bit"` for block-wise quantization. Requires `bitsandbytes`.                     |
| `hf_token`                 | `str \| None` | `None`          | HuggingFace token for downloading gated model weights (download only, not used for inference). |
| `prefetching`              | `bool`        | `True`          | Overlap disk I/O with GPU compute. Auto-disabled with compression.                             |
| `layer_shards_saving_path` | `str \| None` | `None`          | Custom directory for split model shards.                                                       |
| `profiling_mode`           | `bool`        | `False`         | Enable per-layer timing instrumentation.                                                       |
| `delete_original`          | `bool`        | `False`         | Delete original HF model after splitting to save disk space.                                   |
| `device`                   | `str`         | `"cuda:0"`      | Target device for execution.                                                                   |
| `dtype`                    | `torch.dtype` | `torch.float16` | Weight precision.                                                                              |
| `max_seq_len`              | `int`         | `512`           | Maximum sequence length for the model context window.                                          |

### Compression Details

AirLLM uses block-wise weight quantization (not activation quantization) to reduce disk I/O:

| Mode           | Compression Ratio | Speed Improvement | Accuracy Impact |
| -------------- | ----------------- | ----------------- | --------------- |
| `None` (FP16)  | 1.0×              | Baseline          | None            |
| `"8bit"`       | 0.50×             | ~2× faster        | Negligible      |
| `"4bit"` (NF4) | 0.28×             | ~3× faster        | Minimal         |

Since AirLLM's bottleneck is disk-to-GPU transfer, compressing only the weights (not activations) preserves accuracy while dramatically improving speed.

## 🏗️ Architecture

AirLLM is built on a clean, extensible architecture:

```
┌──────────────────────────────────────────────────────────────┐
│                    AutoModel.from_pretrained()                │
│                   ┌─────────────────────┐                    │
│                   │   ModelRegistry      │                    │
│                   │  (30 architectures)  │                    │
│                   └────────┬────────────┘                    │
│                            │ resolves                        │
│         ┌──────────────────┼───────────────────┐             │
│         ▼                  ▼                   ▼             │
│   AirLLMLlama2     AirLLMGemma4       AirLLMDeepSeek        │
│   AirLLMMistral    AirLLMQwen3        AirLLMCohere    ...   │
│         │                  │                   │             │
│         └──────────────────┼───────────────────┘             │
│                            ▼                                 │
│                   AirLLMBaseModel                             │
│              (layer-by-layer forward pass)                    │
│                            │                                 │
│              ┌─────────────┼─────────────┐                   │
│              ▼             ▼             ▼                   │
│         load_layer   move_to_device   offload                │
│         (disk→CPU)   (CPU→GPU)        (GPU→meta)             │
└──────────────────────────────────────────────────────────────┘
```

For detailed architecture documentation:

- [VRAM Management](docs/VRAM_MANAGEMENT.md) — memory optimization strategies.
- [Model Integration Guide](docs/MODEL_INTEGRATION.md) — how to add new model backends.
- [API Reference](docs/API_REFERENCE.md) — public classes and methods.

## ✅ Features & Capabilities

AirLLM v3.0 is production-ready for layer-by-layer inference. All modules are fully integrated into the forward loop:

| Feature                                                                | Module                                                  | Status                                            |
| ---------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------- |
| **TurboQuant KV cache compression** (ICLR 2026)                        | [`kv_cache.py`](air_llm/airllm/kv_cache.py)             | ✅ Integrated — PolarQuant + QJL, ~4.4× reduction |
| **Self-speculative decoding**                                          | [`speculative.py`](air_llm/airllm/speculative.py)       | ✅ Integrated                                     |
| **Async layer prefetching**                                            | [`async_loader.py`](air_llm/airllm/async_loader.py)     | ✅ Integrated                                     |
| **Paged KV cache (CPU offloading)**                                    | [`paged_kv_cache.py`](air_llm/airllm/paged_kv_cache.py) | ✅ Integrated                                     |
| **MoE expert-by-expert loading**                                       | [`moe_loader.py`](air_llm/airllm/moe_loader.py)         | ✅ Integrated                                     |
| **GGUF format support**                                                | [`quantization.py`](air_llm/airllm/quantization.py)     | ✅ Integrated                                     |
| **23+ model backends** (Llama 2–4, Qwen 3, Gemma 4, DeepSeek R1, etc.) | [`airllm_*.py`](air_llm/airllm/)                        | ✅ 36 architectures                               |
| **311 unit tests**                                                     | [`tests/`](air_llm/tests/)                              | ✅ All passing                                    |

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=airllm --cov-report=term-missing

# Run a specific test file
pytest air_llm/tests/test_model_registry.py -v
```

The test suite (311 tests) covers:

- [`test_model_registry.py`](air_llm/tests/test_model_registry.py) — registry decorator and lookup.
- [`test_automodel.py`](air_llm/tests/test_automodel.py) — AutoModel dispatching.
- [`test_constants.py`](air_llm/tests/test_constants.py) — constants and platform detection.
- [`test_model_layer_config.py`](air_llm/tests/test_model_layer_config.py) — `ModelLayerConfig` dataclass.
- [`test_model_backends.py`](air_llm/tests/test_model_backends.py) — all model backend classes.
- [`test_compression.py`](air_llm/tests/test_compression.py) — quantization utilities.
- [`test_persist.py`](air_llm/tests/test_persist.py) — model persistence layer.
- [`test_kv_cache.py`](air_llm/tests/test_kv_cache.py) — TurboQuant KV cache compression.
- [`test_quantization.py`](air_llm/tests/test_quantization.py) — quantization method registry and GGUF support.
- [`test_speculative.py`](air_llm/tests/test_speculative.py) — self-speculative decoding config and verification.
- [`test_async_loader.py`](air_llm/tests/test_async_loader.py) — async layer prefetching.
- [`test_paged_kv_cache.py`](air_llm/tests/test_paged_kv_cache.py) — paged KV cache page management.
- [`test_moe_loader.py`](air_llm/tests/test_moe_loader.py) — MoE expert router and selective loading.
- [`test_downloader.py`](air_llm/tests/test_downloader.py) — model download and path resolution.
- [`test_utils.py`](air_llm/tests/test_utils.py) — utility functions.

## 🤝 Contributing

Contributions are welcome!
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Setting up the development environment.
- Adding a new model backend.
- Code style (Google Python Style, ruff, mypy).
- Running tests and CI.
- The pull request process.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

AirLLM builds on the work of many talented researchers and engineers:

- **SimJeg** — the original Kaggle competition code that inspired AirLLM's layer-by-layer approach.
  [GitHub](https://github.com/SimJeg) ·
  [Kaggle notebook](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag) ·
  [Discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446414)
- **HuggingFace** — the `transformers`, `accelerate`, and `safetensors` libraries.
- **NavodPeiris** — CPU inference and non-sharded model support.

## 📚 Citing AirLLM

```bibtex
@software{airllm2024,
  author = {Gavin Li},
  title = {AirLLM: scaling large language models on low-end commodity computers},
  url = {https://github.com/lyogavin/airllm/},
  version = {3.0.0},
  year = {2024},
}
```

---

⭐ If you find AirLLM useful, please star the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=lyogavin/airllm&type=Timeline)](https://star-history.com/#lyogavin/airllm&Timeline)
