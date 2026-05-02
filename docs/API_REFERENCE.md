# API Reference

This document provides a comprehensive reference for the public API of AirLLM — a 100% local inference engine. No API keys required for inference. All operations run entirely on your hardware.

## `AutoModel`

The primary entry point for loading models. It acts as a factory, inspecting the model's configuration (locally) and instantiating the correct backend class. First call downloads weights from HuggingFace Hub (one-time); subsequent calls use cached local weights. Set `HF_HUB_OFFLINE=1` for fully offline operation.

### `AutoModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)`

Load model weights for completely local inference.

**Arguments:**

- `pretrained_model_name_or_path` (str): HuggingFace model ID (for initial download) OR local directory path (for offline operation). Example: `"meta-llama/Llama-3.1-8B-Instruct"` or `"/path/to/local/model"`.
- `device` (str, optional): The target device for execution. Defaults to `"cuda:0"`.
- `dtype` (torch.dtype, optional): The weight precision to use. Defaults to `torch.float16`.
- `max_seq_len` (int, optional): The maximum sequence length for the model context window. Defaults to `512`.
- `layer_shards_saving_path` (str | Path, optional): A custom directory path to save the split model shards. If `None`, a default path based on the model name is used.
- `profiling_mode` (bool, optional): If `True`, enables per-layer timing instrumentation and prints profiling information during inference. Defaults to `False`.
- `compression` (str, optional): Enables block-wise weight quantization. Supported values are `"4bit"` or `"8bit"`. Requires the `bitsandbytes` package. Defaults to `None` (no compression).
- `hf_token` (str, optional): A Hugging Face API token. Required only for downloading gated model weights (download only — not used for inference).
- `prefetching` (bool, optional): If `True`, overlaps disk I/O (loading the next layer) with GPU compute (processing the current layer). This provides a speedup but is automatically disabled if `compression` is used (unless `AsyncLayerLoader` is available). Defaults to `True`.
- `delete_original` (bool, optional): If `True`, deletes the original Hugging Face model files after they have been split into shards to save disk space. Defaults to `False`.

**New in v3.0.0:**

- `kv_compression` (str, optional): Enable KV cache compression to reduce memory usage during generation. Options: `"turboquant"` (4-bit, ICLR 2026), `"4bit"`, `"3bit"`. Defaults to `None` (no KV compression).
- `speculative_config` (`SpeculativeConfig` | dict, optional): Enable self-speculative decoding for faster generation. When passed as a dict, accepted keys are `exit_layer_ratio` (float, default `0.5`) and `num_speculations` (int, default `3`). Defaults to `None`.

**Returns:**

- An initialized instance of an `AirLLMBaseModel` subclass (e.g., `AirLLMLlama2`, `AirLLMGemma4`). On macOS, it always returns an `AirLLMLlamaMlx` instance.

---

## `AirLLMBaseModel`

The base class for all layer-by-layer inference models — 100% local inference engine. It inherits from Hugging Face's `GenerationMixin` (a local library, not a cloud service), meaning it supports standard generation methods like `.generate()`. All inference runs entirely on your local hardware.

### `generate(inputs, **kwargs)`

Generates sequences of token ids for models with a language modeling head. This method is inherited from `transformers.GenerationMixin`.

**Arguments:**

- `inputs` (torch.Tensor): The input token IDs.
- `**kwargs`: Standard Hugging Face generation parameters (e.g., `max_new_tokens`, `use_cache`, `return_dict_in_generate`, `temperature`, `top_p`).

**Returns:**

- A `GenerateOutput` object or a tuple, depending on the parameters passed.

### `forward(...)`

The core layer-by-layer forward pass. It loads each transformer layer from disk, runs it, and then offloads it to free GPU memory for the next layer. You typically do not call this directly; it is invoked by `.generate()`.

---

## `ModelRegistry`

A centralized registry mapping Hugging Face architecture names to specific AirLLM model classes.

### `@ModelRegistry.register(*architecture_names)`

A class decorator used to register a model backend class.

**Arguments:**

- `*architecture_names` (str): One or more architecture strings as they appear in a Hugging Face `config.json` under the `"architectures"` key.

**Example:**

```python
@ModelRegistry.register("LlamaForCausalLM", "LLaMAForCausalLM")
class AirLLMLlama2(AirLLMBaseModel):
    ...
```

### `ModelRegistry.get(architecture_name)`

Looks up the model class for a given architecture name.

**Arguments:**

- `architecture_name` (str): The architecture string from the Hugging Face config.

**Returns:**

- The registered model class (type).

**Raises:**

- `ValueError`: If no registered architecture matches.

### `ModelRegistry.list_supported()`

Returns a sorted list of all registered architecture names.

---

## `ModelLayerConfig`

A dataclass used to declaratively map logical layer roles to their specific weight-name prefixes in a model's state dictionary.

**Attributes:**

- `embed_tokens` (str): Prefix for the embedding layer. Defaults to `"model.embed_tokens"`.
- `layer_prefix` (str): Prefix for the transformer blocks. Defaults to `"model.layers"`.
- `norm` (str): Prefix for the final layer normalization. Defaults to `"model.norm"`.
- `lm_head` (str): Prefix for the language model head. Defaults to `"lm_head"`.
- `rotary_pos_emb` (str | None): Optional prefix for rotary positional embeddings. Defaults to `None`.

---

## `downloader` Module

Direct model weight downloader — eliminates the mandatory `huggingface-hub` dependency. Supports downloading safetensor model weights directly via HTTPS from HuggingFace Hub using only Python stdlib.

### `resolve_model_path(model_id_or_path, *, token=None, cache_dir=None)`

Resolve a model ID or local path to a local directory. This is the primary entry point for model resolution.

**Resolution Order:**

1. Local path (if exists as directory)
2. AirLLM cache (`AIRLLM_CACHE_DIR`)
3. HuggingFace Hub cache (if `huggingface-hub` installed)
4. Direct HTTPS download (no `huggingface-hub` needed)

**Arguments:**

- `model_id_or_path` (str): Local path or HuggingFace model ID (e.g., `"google/gemma-4-12b"`).
- `token` (str | None): Auth token for gated models.
- `cache_dir` (str | None): Cache directory for downloads. Defaults to `~/.cache/airllm` or `AIRLLM_CACHE_DIR`.

**Returns:**

- (str): Local directory path containing model files.

**Raises:**

- `FileNotFoundError`: If model not found and offline mode is enabled.
- `PermissionError`: If model is gated and no valid token is provided.

### `download_model(repo_id, *, revision="main", token=None, cache_dir=None, allow_patterns=None)`

Download a model from HuggingFace Hub via direct HTTPS. Does **not** require the `huggingface-hub` package.

**Arguments:**

- `repo_id` (str): HuggingFace model ID.
- `revision` (str): Git revision (default: `"main"`).
- `token` (str | None): Auth token for gated models.
- `cache_dir` (str | None): Local directory to cache downloads.
- `allow_patterns` (list[str] | None): Glob patterns to filter files (e.g., `["*.safetensors", "*.json"]`).

**Returns:**

- (str): Local directory path containing the downloaded model files.

### `download_file(url, dest_path, *, token=None, chunk_size=8388608)`

Download a single file from a URL with progress logging and atomic write.

### `get_cache_dir()`

Returns the model cache directory, respecting the `AIRLLM_CACHE_DIR` environment variable.

---

## `quantization` Module

Unified quantization abstraction supporting bitsandbytes, AWQ, GPTQ, and TurboQuant.

### `QuantizationMethod` (Enum)

Supported quantization methods:

| Value | String | Description |
|-------|--------|-------------|
| `NONE` | `"none"` | No quantization (FP16) |
| `BITSANDBYTES_4BIT` | `"4bit"` | bitsandbytes NF4 4-bit |
| `BITSANDBYTES_8BIT` | `"8bit"` | bitsandbytes dynamic 8-bit |
| `AWQ` | `"awq"` | Activation-aware weight quantization |
| `GPTQ` | `"gptq"` | Optimal brain quantization |
| `PRE_QUANTIZED` | `"pre_quantized"` | Pre-quantized safetensor weights |
| `TURBOQUANT_KV` | `"turboquant"` | KV cache compression (ICLR 2026) |

### `detect_quantization(model_path)`

Auto-detect quantization method from model `config.json`.

**Arguments:**

- `model_path` (str): Path to model directory.

**Returns:**

- `QuantizationMethod`: The detected method (or `NONE`).

### `parse_quantization_method(method)`

Parse a user-provided quantization method string to enum value.

**Arguments:**

- `method` (str): Case-insensitive method name. Supported aliases: `"4bit"`, `"8bit"`, `"awq"`, `"gptq"`, `"bnb"`, `"bitsandbytes"`, `"exl2"`, `"turboquant"`, `"turboquant_kv"`.

**Returns:**

- `QuantizationMethod`: The corresponding enum value.

**Raises:**

- `ValueError`: If the method string is not recognized.

### `validate_quantization_backend(method)`

Validate that the required backend for a quantization method is installed.

**Arguments:**

- `method` (QuantizationMethod): The quantization method to validate.

**Raises:**

- `ImportError`: If the required backend is not installed.

### `get_available_methods()`

Returns a sorted list of quantization method strings available on the current system.

### Backend Availability Checks

- `is_bitsandbytes_available()` → `bool`
- `is_awq_available()` → `bool`
- `is_gptq_available()` → `bool`
- `is_turboquant_available()` → `bool`

---

## `kv_cache` Module — TurboQuant (ICLR 2026)

Faithful implementation of Google's TurboQuant KV cache compression using PolarQuant + QJL.

### `PolarQuantConfig` (Frozen Dataclass)

Immutable configuration for TurboQuant KV cache compression.

**Attributes:**

- `bits` (int): Quantization bit width for direction vectors (2, 3, or 4). Default: `3`.
- `qjl_dim` (int): Number of QJL projection dimensions. Set to `0` to disable. Default: `64`.
- `rotation_seed` (int): Fixed seed for rotation matrix generation. Default: `42`.

### `KVCacheCompressor`

Compresses KV cache tensors using PolarQuant (random rotation → polar decomposition → uniform quantization) with optional QJL sign-bit projections for compressed-domain attention.

**Constructor:**

```python
KVCacheCompressor(
    config: PolarQuantConfig | None = None,
    *,
    bits: int = 3,           # Legacy compat
    use_residual: bool = True,  # Legacy: maps to qjl_dim > 0
    device: str | None = None,
)
```

**Methods:**

- `compress(tensor)` → `CompressedKVCache`: Compress a KV tensor via PolarQuant + optional QJL.
- `decompress(compressed, *, dtype, device)` → `torch.Tensor`: Reconstruct from compressed cache.
- `compressed_attention(query, compressed_keys)` → `torch.Tensor`: Compute approximate attention scores directly on compressed keys via QJL Hamming distance (8× speedup).
- `memory_reduction_ratio()` → `float`: Theoretical compression ratio vs FP16.
- `compression_fidelity_report(tensor)` → `dict`: Measure cosine similarity, relative error, and compression ratio.

### `CompressedKVCache` (Frozen Dataclass)

Container for PolarQuant-compressed KV cache tensors.

**Attributes:**

- `magnitudes` (torch.Tensor): Per-vector L2 norms (FP16).
- `quantized_directions` (torch.Tensor): Uniformly quantized unit direction vectors (uint8).
- `shape` (tuple): Original tensor shape.
- `bits` (int): Bit width used.
- `qjl_signs` (torch.Tensor | None): Packed QJL sign bits.

**Class Methods:**

- `cat(caches, dim=0)` → `CompressedKVCache`: Concatenate compressed caches along batch dimension.

### New Model Backend Classes (v3.0)

| Class | Architecture Strings | Models |
|-------|---------------------|--------|
| `AirLLMLlama4` | `Llama4ForCausalLM`, `Llama4ForConditionalGeneration` | Meta Llama 4 Scout/Maverick |
| `AirLLMOLMo2` | `OLMo2ForCausalLM` | AI2 OLMo 2 |
| `AirLLMFalcon3` | `FalconForCausalLM` | TII Falcon 3 |
| `AirLLMGranite` | `GraniteForCausalLM` | IBM Granite |
| `AirLLMJamba` | `JambaForCausalLM` | AI21 Jamba (SSM+Transformer) |

---

## Benchmark Tool

The benchmark tool (`eval/benchmark.py`) measures local LLM inference performance.

### `BenchmarkResult` (Dataclass)

**Attributes:**

- `model_id` (str): Model identifier.
- `compression` (str | None): Quantization method used.
- `prompt` (str): Input prompt.
- `max_new_tokens` (int): Maximum tokens requested.
- `total_time_sec` (float): End-to-end generation time.
- `time_to_first_token_sec` (float): TTFT approximation.
- `tokens_generated` (int): Actual tokens generated.
- `tokens_per_second` (float): Throughput metric.
- `peak_vram_mb` (float): Peak GPU memory allocated.
- `system_info` (dict): Platform, GPU, and Python details.

**Methods:**

- `to_json()` → `str`: Serialize to JSON string.

### `run_benchmark(model_id, *, prompt, max_new_tokens, compression, hf_token)`

Run a single benchmark and return a `BenchmarkResult`.

### `get_system_info()`

Collect system hardware information (platform, GPU, CUDA version).

### `get_peak_vram_mb()`

Get peak VRAM usage in MB (CUDA only).

---

## Constants

Centralized configuration constants available in `airllm.constants`.

- `IS_ON_MAC_OS` (bool): `True` if running on macOS.
- `OFFLINE_MODE` (bool): `True` when `HF_HUB_OFFLINE=1` is set — prevents any network calls.
- `SUPPORTED_COMPRESSIONS` (tuple): `("4bit", "8bit")`.
- `DEFAULT_MAX_LENGTH` (int): `128`.
- `DEFAULT_MAX_SEQ_LEN` (int): `512`.
- `DEFAULT_DEVICE` (str): `"cuda:0"`.
