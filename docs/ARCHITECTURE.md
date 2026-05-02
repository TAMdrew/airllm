# Architecture Overview

AirLLM v3.0 — Repository structure and design decisions.

## High-Level Design

AirLLM enables running 70B+ parameter models on a single 4GB GPU via
**layer-by-layer inference**: weights are loaded from disk one transformer
layer at a time, the forward pass runs for that layer, and then the weights
are freed before the next layer is loaded.

```text
┌─────────────────────────────────────────────────────┐
│                    User Code                        │
│   AutoModel.from_pretrained("meta-llama/...")       │
└────────────────────┬────────────────────────────────┘
                     │
              ┌──────▼──────┐
              │  AutoModel   │  Reads config.json, dispatches
              │              │  to correct backend class
              └──────┬──────┘
                     │
          ┌──────────▼──────────┐
          │   ModelRegistry     │  Decorator-based registry
          │   @register(arch)   │  maps HF architecture strings
          └──────────┬──────────┘  to backend classes
                     │
         ┌───────────▼───────────┐
         │  AirLLMBaseModel      │  Core layer-by-layer engine
         │  (931 lines)          │  Handles: init, forward,
         │                       │  layer loading, generation
         └───────────┬───────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼───┐      ┌────▼────┐     ┌─────▼─────┐
│ Llama │      │  Qwen   │     │  ChatGLM  │  ...21 more
│Backend│      │ Backend │     │  Backend  │  backend classes
└───────┘      └─────────┘     └───────────┘
```

## Directory Structure

```text
air_llm/
├── __init__.py              # Top-level re-export
└── airllm/
    ├── __init__.py           # Public API, __all__, version
    ├── airllm_base.py        # AirLLMBaseModel (core engine)
    ├── airllm.py             # AirLLMLlama2 (minimal subclass)
    ├── airllm_*.py           # Architecture-specific backends
    ├── auto_model.py         # AutoModel dispatcher
    ├── model_registry.py     # Decorator-based model registry
    ├── constants.py          # Shared constants
    ├── async_loader.py       # CUDA stream prefetching
    ├── kv_cache.py           # KV cache compression (TurboQuant)
    ├── paged_kv_cache.py     # Paged KV cache
    ├── speculative.py        # Self-speculative decoding
    ├── moe_loader.py         # MoE expert routing
    ├── profiler.py           # Per-layer timing
    ├── quantization.py       # Quantization method detection
    ├── tokenization_baichuan.py  # Custom Baichuan tokenizer
    ├── engine/
    │   └── inference_engine.py   # Decoupled forward pass
    ├── io/
    │   └── downloader.py     # HuggingFace model download
    ├── persist/
    │   ├── model_persister.py        # ABC for persistence
    │   ├── safetensor_model_persister.py  # CUDA/Linux
    │   └── mlx_model_persister.py    # macOS/Apple Silicon
    └── utils/
        └── core.py           # Shared utilities
```

## Key Design Patterns

### Decorator-Based Model Registry

Backend classes register themselves with HuggingFace architecture strings:

```python
@ModelRegistry.register("LlamaForCausalLM")
class AirLLMLlama2(AirLLMBaseModel):
    pass
```

`AutoModel.from_pretrained()` reads the model's `config.json`, extracts
`architectures[0]`, and looks it up in the registry to instantiate the
correct backend class.

### Strategy Pattern for Persistence

`ModelPersister` is an abstract base class with two implementations:

- `SafetensorModelPersister` — for CUDA/Linux (`.safetensors` format)
- `MlxModelPersister` — for macOS/Apple Silicon (`.mlx.npz` format)

The factory method `ModelPersister.get_model_persister()` selects the
correct implementation based on the platform.

### Template Method for Layer Naming

Each backend can override hook methods to customize layer naming:

- `set_layer_names_dict()` — Maps logical roles to weight prefixes
- `get_pos_emb_args()` — Custom positional embedding arguments
- `get_past_key_value_args()` — Custom KV cache format
- `get_attention_mask_args()` — Custom attention mask handling

## Design Decisions

1. **Layer-by-layer vs. model parallelism**: We chose layer-by-layer
   offloading because it works on ANY GPU, even 4GB. Model parallelism
   requires multiple GPUs.

2. **Empty model scaffold**: We create a meta-device model scaffold and
   load real weights per-layer. This avoids ever needing the full model
   in memory.

3. **Separate persisters**: macOS Apple Silicon uses MLX for inference,
   which requires `.npz` format. Linux uses safetensors. The strategy
   pattern cleanly separates these.
