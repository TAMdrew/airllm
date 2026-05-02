# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] — 2026-05-01

### Added

- **Faithful TurboQuant KV cache compression** (ICLR 2026): Complete rewrite of `kv_cache.py` implementing Google's PolarQuant + QJL algorithms. PolarQuant uses random orthogonal rotation → polar decomposition → uniform quantization WITHOUT per-block scale/zero-point constants (zero overhead bits). QJL enables compressed-domain attention via Rademacher projections + Hamming distance (~8× attention speedup).
- **`PolarQuantConfig`** frozen dataclass: Immutable configuration for TurboQuant (bits, qjl_dim, rotation_seed).
- **`compressed_attention()`** method: Compute approximate attention scores directly on compressed keys without decompression.
- **`compression_fidelity_report()`** method: Measure cosine similarity, relative error, and compression ratio on sample tensors.
- **Llama 4 backend** (`AirLLMLlama4`): Meta Llama 4 Scout (109B, 16 experts) and Maverick (400B, 128 experts) MoE models with iRoPE attention and multimodal support.
- **OLMo 2 backend** (`AirLLMOLMo2`): AI2's fully open-source model (7B-32B) with reproducible training.
- **Falcon 3 backend** (`AirLLMFalcon3`): TII Falcon 3 family (1B-10B).
- **Granite backend** (`AirLLMGranite`): IBM's enterprise models (3B-34B) for RAG and code generation.
- **Jamba backend** (`AirLLMJamba`): AI21's hybrid SSM-Transformer model with Mamba + attention layers and MoE (52B-398B). Auto-sets `trust_remote_code=True`.
- **`SUPPORTED_ARCHITECTURES`** constant: Complete registry of all 30+ supported HuggingFace architecture strings.
- **40 new TurboQuant tests**: PolarQuantConfig validation, compression fidelity, QJL projections, compressed attention correlation, edge cases (zero vectors, bfloat16, various head dims), concatenation, and diagnostics.

### Changed

- **`KVCacheCompressor`** now uses PolarQuant algorithm (rotation → polar decomposition → uniform grid) instead of min/max symmetric quantization. Backward-compatible API preserved.
- **`CompressedKVCache`** dataclass stores `magnitudes` + `quantized_directions` + optional `qjl_signs` instead of `quantized` + `scales` + `zero_points`.
- **`AirLLMBaseModel.__init__`** now accepts `kv_bits` and `kv_qjl_dim` kwargs for fine-grained TurboQuant configuration.
- **`DEFAULT_KV_CACHE_BITS`** changed from 4 to 3 (TurboQuant default: 3-bit provides ~5× compression).
- Total supported architectures: 23+ (from 18+).
- Total test count: 311 (from 178).

### Fixed

- **BUG-1: Persister filename double-dot mismatch** — `load_model()` used `layer_name + ".safetensors"` producing double dots (`model.layers.0..safetensors`), while `persist_model()` used `layer_name + "safetensors"`. Fixed to use consistent pattern in both `SafetensorModelPersister` and `MlxModelPersister`.
- **BUG-2: InferenceEngine empty list multiplication** — `[] * len(layers)` produced `[]` (empty list) instead of N independent empty lists. Fixed with list comprehension.
- **BUG-3: InferenceEngine layer index parsing** — `int(layer_name.split(".")[-1])` failed on trailing-dot layer names (e.g., `model.layers.0.`). Fixed with `.rstrip(".")`.
- **LEGAL-1: License inconsistency** — `LICENSE` file was Apache 2.0 while `pyproject.toml`, `README.md`, and `funding.json` all referenced MIT. Unified to MIT.
- **ARCH-3: `utils/__init__.py` cross-package leaks** — Removed re-exports of `ModelPersister` and `glob` from sibling packages.
- **ARCH-6: `AsyncLayerLoader` resource leak** — Added `__enter__`/`__exit__` context manager support.
- **CODE-2: `print()` in MLX backend** — Replaced with `logger.debug()`.
- **CODE-3: Bare `except:` clauses** — Fixed to `except Exception:` in `training/qlora.py` and `rlhf/qlora_dpo.py`.
- **CODE-4: No `compression` parameter validation** — Added fail-fast check against `{None, "4bit", "8bit"}`.
- **POLISH-1: Filename typos** — Renamed `inferrence.ipynb`, `test_cn_dataset_lenghts.py`, `run_finetune_raining_based_on_Anima.sh`.

### Removed

- Orphaned `air_llm/airllm/models/` directory (incomplete refactor, never wired into production code).
- Duplicate `air_llm/README.md` and `air_llm/LICENSE` files.

### Also Added (prior v3.0.0 work)

- **Gemma 4 backend** ([`AirLLMGemma4`](air_llm/airllm/airllm_gemma4.py:35)): Full support for Gemma 4 E2B, E4B, 12B, and 31B variants with alternating sliding-window/global attention and Per-Layer Embeddings (PLE).
- **Qwen 3 backend** ([`AirLLMQwen3`](air_llm/airllm/airllm_qwen3.py:30)): Support for Qwen3 0.5B through 72B with GQA and dynamic RoPE/YaRN scaling.
- **DeepSeek V2/V3 backend** ([`AirLLMDeepSeek`](air_llm/airllm/airllm_deepseek.py:42)): Support for DeepSeek-V3 and DeepSeek-R1 671B MoE models with Multi-head Latent Attention (MLA).
- **GLM-4 backend** ([`AirLLMGlm4`](air_llm/airllm/airllm_glm4.py:35)): Native HuggingFace-compatible GLM-4-9B support (no `trust_remote_code` needed).
- **Phi-3/Phi-4 backend** ([`AirLLMPhi`](air_llm/airllm/airllm_phi.py:34)): Support for Phi-4 14B and all Phi-3 variants.
- **Cohere backend** ([`AirLLMCohere`](air_llm/airllm/airllm_cohere.py:33)): Support for Command-R 35B and Command-R+ 104B, with Cohere2 forward compatibility.
- **Zamba 2 backend** ([`AirLLMZamba2`](air_llm/airllm/airllm_zamba.py:39)): Hybrid Mamba2/Transformer architecture support for Zamba2-2.7B and 7B.
- **[`ModelRegistry`](air_llm/airllm/model_registry.py:27)**: Decorator-based architecture registration replacing the legacy if/elif chain in `auto_model.py`.
- **[`ModelLayerConfig`](air_llm/airllm/airllm_base.py:69) dataclass**: Declarative layer name mapping for cleaner model configuration.
- **[`constants.py`](air_llm/airllm/constants.py:1)**: Centralized configuration constants (compression ratios, platform detection, file format markers, defaults).
- **Comprehensive test suite**: 178 tests covering the registry, auto-model, constants, layer config, model backends, compression, persistence, KV cache, quantization, speculative decoding, async loader, paged KV cache, MoE loader, and utilities.
- **TurboQuant KV cache compression** ([`kv_cache.py`](air_llm/airllm/kv_cache.py:1)): Pure PyTorch implementation with [`KVCacheCompressor`](air_llm/airllm/kv_cache.py:20) class supporting 3-bit (3.8× reduction) and 4-bit (3.0× reduction) modes, plus [`CompressedKVCache`](air_llm/airllm/kv_cache.py:10) dataclass for efficient KV storage.
- **Self-speculative decoding** ([`speculative.py`](air_llm/airllm/speculative.py:1)): LayerSkip-style early-exit drafting with [`SpeculativeConfig`](air_llm/airllm/speculative.py:12), [`verify_draft_tokens()`](air_llm/airllm/speculative.py:49), and [`estimate_speedup()`](air_llm/airllm/speculative.py:90) — enables 3–4× speedup on I/O-dominated inference without a separate draft model.
- **Async layer loader** ([`async_loader.py`](air_llm/airllm/async_loader.py:1)): CUDA stream-based prefetching via [`AsyncLayerLoader`](air_llm/airllm/async_loader.py:20) that overlaps disk I/O and GPU compute, compatible with quantized weights and safetensor format.
- **Paged KV cache** ([`paged_kv_cache.py`](air_llm/airllm/paged_kv_cache.py:1)): CPU↔GPU page management for long-context inference with [`PagedKVCache`](air_llm/airllm/paged_kv_cache.py:40) and [`PageConfig`](air_llm/airllm/paged_kv_cache.py:12) — supports configurable page sizes and automatic eviction.
- **MoE expert-by-expert loader** ([`moe_loader.py`](air_llm/airllm/moe_loader.py:1)): Router-first selective expert loading with [`ExpertRouter`](air_llm/airllm/moe_loader.py:30) and [`MoEConfig`](air_llm/airllm/moe_loader.py:12) — loads only top-K activated experts per token to reduce memory.
- **GGUF format support** in [`quantization.py`](air_llm/airllm/quantization.py:1): Added `QuantizationMethod.GGUF` enum value, [`is_gguf_available()`](air_llm/airllm/quantization.py:50) check, and `gguf` optional dependency group in [`pyproject.toml`](pyproject.toml:1).
- **Benchmark tool** ([`eval/benchmark.py`](eval/benchmark.py:1)): CLI and programmatic API for measuring tokens/sec, time-to-first-token (TTFT), and peak VRAM usage.
- **Usage guide** ([`docs/USAGE_GUIDE.md`](docs/USAGE_GUIDE.md:1)): Complete install, test, run, and benchmark walkthrough.
- **[`pyproject.toml`](pyproject.toml:1)**: Modern Python packaging with `setuptools-scm`, optional dependency groups (`quantization`, `mlx`, `dev`, `all`), and integrated tool configuration for ruff, mypy, and pytest.
- **CI/CD pipeline**: GitHub Actions for linting, type checking, and test execution.
- **Documentation**: Architecture docs, model integration guide, VRAM management guide, class hierarchy, and testing strategy.
- **Offline operation support** — added `OFFLINE_MODE` constant (reads `HF_HUB_OFFLINE` env var); `find_or_create_local_splitted_path` now raises a clear `FileNotFoundError` when offline and model not found locally.
- **Forward-loop integration** ([`airllm_base.py`](air_llm/airllm/airllm_base.py:239)) — KV cache compression (`kv_compression` kwarg → `KVCacheCompressor`), async layer loader (`AsyncLayerLoader` replaces basic `ThreadPoolExecutor` prefetching on CUDA), and speculative config (`speculative_config` kwarg → `SpeculativeConfig`) wired into `AirLLMBaseModel.__init__`.

### Changed

- **Local-first architecture clarification** — all docstrings and documentation updated to clarify that HuggingFace libraries run entirely locally for config parsing, tokenization, and one-time weight download. No cloud APIs, no paid services.
- **Python ≥3.11 minimum** — bumped from 3.10 to 3.11 (Python 3.10 EOL October 2026). Updated `pyproject.toml`, CI matrix, mypy `python_version`, and ruff `target-version`.
- **Dependencies updated to April 2026** — `torch>=2.6.0`, `transformers>=4.48.0`, `safetensors>=0.5.0`, `huggingface-hub>=0.27.0`, `accelerate>=0.36.0`. Removed `optimum` and `scipy` from core dependencies.
- **`setup.py` deprecated** — replaced with thin wrapper pointing to `pyproject.toml` (PEP 517/518).
- **`typing.Union` replaced** — all `Union[X, Y]` annotations in core files replaced with `X | Y` (Python 3.10+ syntax via `from __future__ import annotations`).
- **Ollama comparison clarified** — comparison language updated to "Unlike server-based tools (e.g., Ollama), AirLLM runs as a native Python library."
- **CI workflow updated** — Python matrix now `["3.11", "3.12", "3.13"]`, lint uses Python 3.12.

- **All `print()` statements replaced with `logging` module** — structured logging at DEBUG/INFO/WARNING/ERROR levels throughout the codebase.
- **Type hints on all public APIs** — full type annotations on all public class constructors, methods, and function signatures.
- **Docstrings on all public classes and methods** — Google-style docstrings with Args, Returns, and Raises sections.
- **`requirements.txt` modernized** — dependency versions updated and aligned with `pyproject.toml`.
- **Gemma backend refactored** — split into Gemma 1/2/3 ([`airllm_gemma.py`](air_llm/airllm/airllm_gemma.py:1)) and Gemma 4 ([`airllm_gemma4.py`](air_llm/airllm/airllm_gemma4.py:1)) for cleaner architecture support.
- **Mistral backend updated** — now registers `Mistral4ForCausalLM` and `Mistral3ForConditionalGeneration` for Mistral Small 3/4 support.
- **QWen 2 backend updated** — now registers `Qwen2_5ForCausalLM` for Qwen 2.5 and QwQ-32B support.
- **`AutoModel.from_pretrained()`** — now uses `ModelRegistry.get()` with substring fallback for backward compatibility.
- **Legacy Jupyter notebooks updated** — all three example notebooks ([`run_all_types_of_models.ipynb`](air_llm/examples/run_all_types_of_models.ipynb), [`run_llama3.1_405B.ipynb`](air_llm/examples/run_llama3.1_405B.ipynb), [`run_on_macos.ipynb`](air_llm/examples/run_on_macos.ipynb)) updated with v3.0.0 API headers and `AutoModel` usage.

### Fixed

- **Bare `except` blocks** replaced with specific exception types (`ValueError`, `TypeError`, `OSError`, `ImportError`).
- **Platform detection centralized** — all `sys.platform == "darwin"` checks replaced with [`IS_ON_MAC_OS`](air_llm/airllm/constants.py:12) constant.
- **Magic numbers eliminated** — compression block sizes, ratios, and default values moved to named constants.
- **`__repr__` and `__str__` methods** added to `AirLLMBaseModel` for debugging clarity.
- **Prefetching + quantization conflict resolved** — [`AsyncLayerLoader`](air_llm/airllm/async_loader.py:20) enables CUDA stream overlap with compressed weights; basic `ThreadPoolExecutor` prefetching correctly disabled only when `AsyncLayerLoader` is unavailable.

## [2.12.0] — 2026-04-01

### Added

- Gemma family native support (Gemma, Gemma 2, Gemma 3, Gemma 4).

## [2.11.0] — 2024-08-20

### Added

- Qwen 2.5 support.

## [2.10.1] — 2024-08-18

### Added

- CPU inference support.
- Non-sharded model support.
- Thanks to @NavodPeiris for the contribution.

## [2.10.0] — 2024-07-30

### Added

- Llama 3.1 405B support.
- 8-bit and 4-bit block-wise quantization (compression).

## [2.9.0] — 2024-04-20

### Added

- Llama 3 native support — run Llama 3 70B on 4 GB single GPU.

## [2.8.2] — 2023-12-25

### Added

- macOS support for running 70B LLMs via MLX backend.

## [2.7.0] — 2023-12-20

### Added

- `AirLLMMixtral` — Mixtral MoE model support.
- `AutoModel` — automatic model type detection (no need to specify model class).

## [2.5.0] — 2023-12-18

### Added

- Prefetching to overlap model loading and GPU compute (10% speed improvement).

## [2.4.0] — 2023-12-03

### Added

- ChatGLM support.
- QWen support.
- Baichuan support.
- Mistral support.
- InternLM support.

## [2.3.0] — 2023-12-02

### Added

- Safetensors format support.
- All top-10 Open LLM Leaderboard models now supported.

## [2.0.0] — 2023-12-01

### Added

- Block-wise quantization model compression for 3× inference speed improvement.

## [1.0.0] — 2023-11-20

### Added

- Initial release of AirLLM.
- Layer-by-layer inference for Llama 2 models.
- 70B model inference on 4 GB GPU.

[3.0.0]: https://github.com/lyogavin/airllm/compare/v2.12.0...v3.0.0
[2.12.0]: https://github.com/lyogavin/airllm/compare/v2.11.0...v2.12.0
[2.11.0]: https://github.com/lyogavin/airllm/compare/v2.10.1...v2.11.0
[2.10.1]: https://github.com/lyogavin/airllm/compare/v2.10.0...v2.10.1
[2.10.0]: https://github.com/lyogavin/airllm/compare/v2.9.0...v2.10.0
[2.9.0]: https://github.com/lyogavin/airllm/compare/v2.8.2...v2.9.0
[2.8.2]: https://github.com/lyogavin/airllm/compare/v2.7.0...v2.8.2
[2.7.0]: https://github.com/lyogavin/airllm/compare/v2.5.0...v2.7.0
[2.5.0]: https://github.com/lyogavin/airllm/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/lyogavin/airllm/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/lyogavin/airllm/compare/v2.0.0...v2.3.0
[2.0.0]: https://github.com/lyogavin/airllm/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/lyogavin/airllm/releases/tag/v1.0.0
