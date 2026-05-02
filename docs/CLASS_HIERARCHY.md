# Class Hierarchy

AirLLM v3.0 — Inheritance tree and method override points.

## Backend Class Hierarchy

```text
GenerationMixin (HuggingFace)
└── AirLLMBaseModel                    # Core layer-by-layer engine
    ├── AirLLMLlama2                   # Llama 2/3 (standard layout)
    ├── AirLLMQWen                     # Qwen v1 (custom rotary, layer naming)
    ├── AirLLMQWen2                    # Qwen 2/2.5/QwQ (custom GenerationConfig)
    ├── AirLLMQwen3                    # Qwen 3 (logging only)
    ├── AirLLMChatGLM                  # ChatGLM (unique layout)
    ├── AirLLMBaichuan                 # Baichuan (custom tokenizer)
    ├── AirLLMMistral                  # Mistral/Mistral4
    ├── AirLLMPhi                      # Phi-2/3/4
    ├── AirLLMDeepSeek                 # DeepSeek V2/V3
    ├── AirLLMFalcon3                  # Falcon 3
    ├── AirLLMGemma                    # Gemma 1 (marker class)
    ├── AirLLMGemma2                   # Gemma 2 (marker class)
    ├── AirLLMGemma3                   # Gemma 3 (marker class)
    ├── AirLLMGemma4                   # Gemma 4 (custom embed/head)
    ├── AirLLMLlama4                   # Llama 4 (custom embed/head)
    ├── AirLLMCohere                   # Cohere Command R
    ├── AirLLMGranite                  # IBM Granite
    ├── AirLLMInternLM                 # InternLM (marker class)
    ├── AirLLMJamba                    # AI21 Jamba (SSM hybrid)
    ├── AirLLMMixtral                  # Mixtral MoE (marker class)
    ├── AirLLMOLMo2                    # OLMo 2 (marker class)
    ├── AirLLMGlm4                     # GLM-4 (marker class)
    └── AirLLMZamba2                   # Zamba 2 (marker class)

AirLLMLlamaMlx                         # Standalone MLX backend (macOS)
```

## Override Points

Methods that subclasses can override to customize behavior:

### Layer Naming

| Method                      | Purpose                               | Who Overrides |
| --------------------------- | ------------------------------------- | ------------- |
| `set_layer_names_dict()`    | Define embed/layer/norm/head prefixes | QWen, ChatGLM |
| `get_pos_emb_args()`        | Positional embedding kwargs           | QWen, ChatGLM |
| `get_past_key_value_args()` | KV cache format                       | QWen, ChatGLM |
| `get_attention_mask_args()` | Attention mask format                 | QWen, ChatGLM |
| `get_position_ids_args()`   | Position ID format                    | QWen, ChatGLM |

### Model Setup

| Method                         | Purpose                       | Who Overrides        |
| ------------------------------ | ----------------------------- | -------------------- |
| `get_generation_config()`      | Custom GenerationConfig       | QWen2                |
| `get_tokenizer()`              | Custom tokenizer loading      | Baichuan             |
| `get_use_better_transformer()` | Enable BetterTransformer      | None (default False) |
| `init_model()`                 | Model scaffold initialization | None                 |

### Forward Pass

| Method                                | Purpose                     | Who Overrides  |
| ------------------------------------- | --------------------------- | -------------- |
| `run_lm_head()`                       | Run language model head     | Gemma4, Llama4 |
| `run_norm()`                          | Run normalization layer     | None           |
| `get_sequence_len()`                  | Sequence length from tensor | ChatGLM        |
| `get_past_key_values_cache_seq_len()` | KV cache seq length         | QWen, ChatGLM  |

## Persistence Hierarchy

```text
ModelPersister (ABC)
├── SafetensorModelPersister    # CUDA / Linux
└── MlxModelPersister           # macOS / Apple Silicon
```

## Support Class Hierarchy

```text
PolarQuantConfig (dataclass)     # KV compression config
KVCacheCompressor                # KV cache compress/decompress
CompressedKVCache (dataclass)    # Compressed cache container
PageConfig (dataclass)           # Paged KV cache config
PagedKVCache                     # Paged cache manager
SpeculativeConfig (dataclass)    # Self-speculative config
AsyncLayerLoader                 # CUDA stream prefetching
ExpertRouter                     # MoE expert routing
MoEConfig (dataclass)            # MoE configuration
LayeredProfiler                  # Per-layer timing
ModelLayerConfig (dataclass)     # Layer name mapping
```
