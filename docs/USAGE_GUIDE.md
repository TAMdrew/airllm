# AirLLM Usage Guide

## Zero Account Required

**AirLLM requires NO accounts, NO API keys, and NO subscriptions.** All inference runs 100% locally on your hardware.

- Most models (Qwen 3, Gemma 4, DeepSeek R1, Mistral, Phi-4, Falcon 3, etc.) are **publicly downloadable** — no account needed
- Only Meta Llama models require a free HuggingFace account to accept Meta's license (one-time)
- After the first download, set `HF_HUB_OFFLINE=1` to run fully air-gapped — zero network calls
- You can also download model weights from **any source** (ModelScope, Kaggle, direct HTTP) and point AirLLM at the local directory

```python
# No account needed — just point to local model files
model = AutoModel.from_pretrained("/path/to/local/model")
```

## Quick Start

### Installation

```bash
# Basic install (minimal dependencies)
pip install -e .

# With all optional features
pip install -e ".[all]"

# Specific extras
pip install -e ".[hub]"            # HuggingFace Hub integration
pip install -e ".[awq]"            # AWQ quantization (4x faster)
pip install -e ".[gptq]"           # GPTQ quantization
pip install -e ".[quantization]"   # bitsandbytes quantization
pip install -e ".[mlx]"            # Apple Silicon MLX
pip install -e ".[dev]"            # Development tools (pytest, ruff, mypy)
```

### Basic Inference

```python
from air_llm.airllm import AutoModel

# Load any supported model (downloads once, runs locally forever)
model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Tokenize input
inputs = model.tokenizer("What is machine learning?", return_tensors="pt")

# Generate (on GPU if available)
output = model.generate(
    inputs["input_ids"].cuda(),  # .to("mps") for Apple Silicon
    max_new_tokens=50,
)

print(model.tokenizer.decode(output[0], skip_special_tokens=True))
```

### Offline Mode

```bash
# After first download, run fully offline
export HF_HUB_OFFLINE=1
python your_script.py
```

### Using Local Model Files

```python
# Point directly to a local directory (no network calls at all)
model = AutoModel.from_pretrained("/path/to/local/model/directory")
```

## Testing the Project

### Run All Tests

```bash
pip install -e ".[dev]"
pytest air_llm/tests/ -v --tb=short
```

### Run Specific Test Files

```bash
pytest air_llm/tests/test_model_registry.py -v
pytest air_llm/tests/test_quantization.py -v
pytest air_llm/tests/test_downloader.py -v
```

### Run Linting

```bash
ruff check air_llm/
```

### Verify All Imports

```bash
python -c "
from air_llm.airllm import AutoModel, ModelRegistry
from air_llm.airllm.quantization import get_available_methods
print(f'Architectures: {len(ModelRegistry.list_supported())}')
print(f'Quantization: {get_available_methods()}')
print('All imports OK')
"
```

## Benchmarking

### Run the Benchmark Tool

```bash
# Basic benchmark
python eval/benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# With quantization
python eval/benchmark.py --model google/gemma-4-12b --compression 4bit

# Save results
python eval/benchmark.py --model Qwen/Qwen3-8B --output results.json
```

### Metrics Measured

| Metric     | Description                   |
| ---------- | ----------------------------- |
| Tokens/sec | Generation throughput         |
| TTFT       | Time to first token (latency) |
| Peak VRAM  | Maximum GPU memory used (MB)  |
| Total time | End-to-end generation time    |

## Using Different Quantization Methods

### bitsandbytes (4-bit/8-bit)

```bash
pip install airllm[quantization]
```

```python
model = AutoModel.from_pretrained("meta-llama/Llama-3.1-70B", compression="4bit")
```

### AWQ (4x faster inference)

```bash
pip install airllm[awq]
```

```python
# Use a pre-quantized AWQ model
model = AutoModel.from_pretrained("TheBloke/Llama-2-70B-AWQ")
```

### GPTQ

```bash
pip install airllm[gptq]
```

```python
model = AutoModel.from_pretrained("TheBloke/Llama-2-70B-GPTQ")
```

### TurboQuant KV Cache Compression (ICLR 2026)

Reduce KV cache memory by ~5× using PolarQuant + QJL. No calibration needed.

```python
# Enable TurboQuant with default settings (3-bit PolarQuant + 64-dim QJL)
model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    kv_compression="turboquant",
)

# Customize bit width and QJL dimensions
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-8B",
    kv_compression="turboquant",
    kv_bits=4,        # 4-bit for higher fidelity
    kv_qjl_dim=128,   # More QJL dims for better approximate attention
)

# PolarQuant only (no QJL compressed attention)
model = AutoModel.from_pretrained(
    "google/gemma-4-12b",
    kv_compression="3bit",  # Simple 3-bit PolarQuant, no QJL
)
```

#### Using PolarQuantConfig Directly

```python
from air_llm.airllm import KVCacheCompressor, PolarQuantConfig

# Create compressor for standalone use
config = PolarQuantConfig(bits=3, qjl_dim=64)
compressor = KVCacheCompressor(config)

# Check compression quality on sample data
import torch
sample = torch.randn(1, 32, 512, 128, dtype=torch.float16)
report = compressor.compression_fidelity_report(sample)
print(f"Cosine similarity: {report['cosine_similarity']:.4f}")
print(f"Compression ratio: {report['compression_ratio']:.1f}x")
```

## Model-Specific Examples

### Gemma 4

```python
model = AutoModel.from_pretrained("google/gemma-4-12b")
```

### Qwen 3

```python
model = AutoModel.from_pretrained("Qwen/Qwen3-8B")
```

### DeepSeek R1 (requires 16GB+ VRAM)

```python
model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-R1")
```

### Phi-4

```python
model = AutoModel.from_pretrained("microsoft/phi-4")
```

### Llama 4 Scout (MoE)

```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    compression="4bit",
    kv_compression="turboquant",
)
```

### OLMo 2

```python
model = AutoModel.from_pretrained("allenai/OLMo-2-7B")
```

### Jamba (SSM + Transformer hybrid)

```python
# trust_remote_code is set automatically
model = AutoModel.from_pretrained("ai21labs/AI21-Jamba-1.5-Mini")
```

## Python Version Compatibility

| Python | Status                              |
| ------ | ----------------------------------- |
| 3.9    | ❌ Not supported (EOL October 2025) |
| 3.10   | ❌ Not supported (dropped in v3.0)  |
| 3.11   | ✅ Minimum supported version        |
| 3.12   | ✅ Fully supported                  |
| 3.13   | ✅ Fully supported (tested in CI)   |

## Troubleshooting

### "Model not found" in offline mode

```bash
# Download first with internet
python -c "from air_llm.airllm import AutoModel; AutoModel.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"

# Then use offline
export HF_HUB_OFFLINE=1
python your_script.py
```

### CUDA Out of Memory

Use quantization to reduce VRAM usage:

```python
model = AutoModel.from_pretrained("model-id", compression="4bit")
```

### Permission denied (gated model)

```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    hf_token="hf_YOUR_TOKEN_HERE",
)
```
