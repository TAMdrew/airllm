# AirLLM Quickstart Guide

This guide will walk you through setting up AirLLM and running your first large language model entirely on your local hardware — no cloud APIs, no subscriptions, no internet required after the initial model download. Zero subscription fees — your hardware, your models, forever.

## Prerequisites

Before you begin, ensure you have the following:

- **Python:** 3.11 or newer.
- **Hardware:** A CUDA-capable GPU (NVIDIA) with at least 4 GB of VRAM, or an Apple Silicon Mac (M1/M2/M3).
- **Disk Space:** Sufficient storage for the model weights. The model will be downloaded and then split into shards. You generally need about 2x the model's size in free space during the initial setup.

## Installation

Install the base `airllm` package using pip:

```bash
pip install airllm
```

### Optional Dependencies

For significantly faster inference (up to 3x) and reduced memory footprint, install the quantization dependencies:

```bash
pip install "airllm[quantization]"
```

If you are on an Apple Silicon Mac, install the MLX dependencies:

```bash
pip install "airllm[mlx]"
```

## Your First Inference

Let's run a Llama 3.1 8B model. The `AutoModel` class automatically handles downloading the model weights from HuggingFace Hub (one-time download, cached locally), splitting them into layers, and managing the layer-by-layer execution entirely on your local hardware. No API keys required for inference — only for downloading gated models.

After the first download, set `HF_HUB_OFFLINE=1` for fully offline operation.

Create a file named `run_llama.py` and add the following code:

```python
from airllm import AutoModel

# 1. Initialize the model
# The first time you run this, it will download model weights from HuggingFace
# and split them into layer shards. After this one-time setup, no internet is needed.
# All inference runs locally on your GPU — no cloud APIs are called.
model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 2. Prepare your input
prompt = "Explain the concept of layer-by-layer inference in simple terms."
input_tokens = model.tokenizer(
    [prompt],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

# 3. Generate text
# The .cuda() call moves the input tokens to the GPU.
# (Omit .cuda() if you are on a Mac using MLX)
print("Generating response...")
output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=100,
    use_cache=True,
    return_dict_in_generate=True,
)

# 4. Decode and print the output
response = model.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
print("\nResponse:\n")
print(response)
```

Run the script:

```bash
python run_llama.py
```

## Trying Different Models

AirLLM supports a wide variety of models. You can switch models simply by changing the Hugging Face repository ID in `AutoModel.from_pretrained()`.

For example, to run a Gemma 4 model:

```python
model = AutoModel.from_pretrained("google/gemma-4-12b-it")
```

Or a Qwen 3 model:

```python
model = AutoModel.from_pretrained("Qwen/Qwen3-8B")
```

See the [Supported Models Reference](SUPPORTED_MODELS.md) for a complete list.

## Using Quantization (Highly Recommended)

AirLLM's primary bottleneck is transferring data from disk to GPU memory. By using block-wise weight quantization, you can drastically reduce the amount of data transferred, speeding up inference by up to 3x with negligible impact on accuracy.

To enable quantization, pass the `compression` argument:

```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    compression="4bit"  # Options: "4bit" or "8bit"
)
```

_Note: This requires the `bitsandbytes` package, which is installed via `pip install "airllm[quantization]"`._

## Testing Your Installation

After installing, verify everything works correctly:

### Quick Smoke Test

```bash
python -c "
from air_llm.airllm import AutoModel, ModelRegistry
from air_llm.airllm.quantization import get_available_methods
print(f'Architectures: {len(ModelRegistry.list_supported())}')
print(f'Quantization: {get_available_methods()}')
print('All imports OK')
"
```

### Run the Test Suite

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest air_llm/tests/ -v --tb=short

# Run a specific test file
pytest air_llm/tests/test_model_registry.py -v
```

### Run Linting

```bash
ruff check air_llm/
```

## Benchmarking

Use the benchmark tool to measure inference performance on your hardware:

```bash
# Basic benchmark
python eval/benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# With quantization
python eval/benchmark.py --model google/gemma-4-12b --compression 4bit

# Save results to JSON
python eval/benchmark.py --model Qwen/Qwen3-8B --output results.json
```

The benchmark measures tokens/sec, time-to-first-token (TTFT), and peak VRAM usage.
See [eval/README.md](../eval/README.md) for full CLI options and metrics, or the
[Usage Guide](USAGE_GUIDE.md) for more details.

## Troubleshooting Common Issues

### 1. Out of Disk Space (`MetadataIncompleteBuffer` error)

The process of splitting the model into shards requires significant disk space. If you encounter an error like `safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer`, you have likely run out of space.

**Solution:**

- Ensure you have enough free space (roughly 2x the model size).
- Clear your Hugging Face cache (`~/.cache/huggingface/hub`).
- Use the `delete_original=True` parameter to delete the original downloaded weights after the split shards are created:
  ```python
  model = AutoModel.from_pretrained("model-id", delete_original=True)
  ```

### 2. Gated Models (401 Client Error)

Some models (like Llama 3 or Gemma) require you to accept their license agreement on HuggingFace before downloading weights. The HuggingFace token is used **only for the initial weight download** — all inference runs entirely locally with no internet connection required.

**Solution:**

1. Go to the model's page on HuggingFace and accept the agreement.
2. Create an Access Token in your HuggingFace account settings.
3. Pass the token to `AutoModel` (only needed once for the download):
   ```python
   model = AutoModel.from_pretrained(
       "meta-llama/Llama-3.1-8B-Instruct",
       hf_token="your_token_here",  # Download only — not used for inference
   )
   ```

### 3. Tokenizer Padding Errors

If you see `ValueError: Asking to pad but the tokenizer does not have a padding token`, the model's tokenizer isn't configured for padding.

**Solution:**
Turn off padding in the tokenizer call:

```python
input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    padding=False  # <-- Set this to False
)
```
