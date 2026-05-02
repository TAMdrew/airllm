# Hardware Guide for Local LLM Inference

This guide covers hardware recommendations for running LLMs locally with AirLLM. Whether you're using Apple Silicon, AMD, or NVIDIA hardware, this document helps you choose the right configuration for your target model sizes.

All inference runs **100% locally** — no cloud APIs, no subscriptions, no internet required after initial model download.

## Table of Contents

- [Apple Silicon — Unified Memory Architecture](#apple-silicon--unified-memory-architecture)
- [AMD — Ryzen AI Max+ (Strix Halo)](#amd--ryzen-ai-max-strix-halo)
- [NVIDIA Consumer GPUs](#nvidia-consumer-gpus)
- [When is AirLLM Layer-by-Layer Needed?](#when-is-airllm-layer-by-layer-needed)
- [Budget Recommendations](#budget-recommendations)
- [Key Concepts](#key-concepts)

---

## Apple Silicon — Unified Memory Architecture

Apple Silicon eliminates the PCIe bottleneck by giving the GPU direct access to all system RAM via Unified Memory Architecture (UMA). Memory bandwidth is the key metric for tokens/sec performance.

> **Why Apple Silicon excels for LLMs:** There is no PCIe bus separating CPU and GPU memory. The GPU can access the full system RAM pool directly, meaning a 192GB M2 Ultra can load a 192GB model entirely "in VRAM" — something impossible on discrete GPU systems without multi-GPU setups.

### Complete Apple Silicon Comparison

| Chip         | GPU Cores | Bandwidth    | Max RAM | Max Model (FP16) | Max Model (INT4) | Est. tok/s (7B Q4) | Products                   |
| ------------ | --------- | ------------ | ------- | ---------------- | ---------------- | ------------------ | -------------------------- |
| **M1**       | 8         | 68 GB/s      | 16GB    | 8B               | 14B              | ~20                | MacBook Air/Pro, Mac Mini  |
| **M1 Pro**   | 14–16     | 200 GB/s     | 32GB    | 14B              | 32B              | ~40                | MacBook Pro 14/16          |
| **M1 Max**   | 24–32     | 400 GB/s     | 64GB    | 32B              | 70B              | ~55                | MacBook Pro 16, Mac Studio |
| **M1 Ultra** | 48–64     | 800 GB/s     | 128GB   | 70B              | 140B             | ~70                | Mac Studio, Mac Pro        |
| **M2**       | 8–10      | 100 GB/s     | 24GB    | 12B              | 24B              | ~30                | MacBook Air/Pro, Mac Mini  |
| **M2 Pro**   | 16–19     | 200 GB/s     | 32GB    | 14B              | 32B              | ~45                | MacBook Pro, Mac Mini      |
| **M2 Max**   | 30–38     | 400 GB/s     | 96GB    | 48B              | 96B              | ~60                | MacBook Pro 16, Mac Studio |
| **M2 Ultra** | 60–76     | 800 GB/s     | 192GB   | 96B              | 192B             | ~75                | Mac Studio, Mac Pro        |
| **M3**       | 8–10      | 100 GB/s     | 24GB    | 12B              | 24B              | ~35                | MacBook Air/Pro, iMac      |
| **M3 Pro**   | 14–18     | 150 GB/s     | 36GB    | 18B              | 36B              | ~42                | MacBook Pro 14/16          |
| **M3 Max**   | 30–40     | 300–400 GB/s | 128GB   | 64B              | 128B             | ~65                | MacBook Pro 16, Mac Studio |
| **M3 Ultra** | 60–80     | 800 GB/s     | 512GB   | 256B             | 512B             | ~80                | Mac Studio                 |
| **M4**       | 10        | 120 GB/s     | 32GB    | 16B              | 32B              | ~40                | MacBook Pro, iPad Pro      |
| **M4 Pro**   | 16–20     | 273 GB/s     | 64GB    | 32B              | 70B              | ~55                | MacBook Pro, Mac Mini      |
| **M4 Max**   | 32–40     | 410–546 GB/s | 128GB   | 64B              | 128B             | ~75                | MacBook Pro 16, Mac Studio |

### Apple Silicon Notes

- **Framework recommendation:** Use [MLX](https://github.com/ml-explore/mlx) (Apple's native ML framework) for best performance on Apple Silicon. AirLLM's [`AirLLMLlamaMlx`](../air_llm/airllm/airllm_llama_mlx.py) backend provides MLX integration.
- **Bandwidth is king:** The tok/s estimates above are for 7B Q4 models. Larger models are proportionally slower due to more bytes per token generation step.
- **INT4 capacity:** INT4 quantization reduces model size by ~4× compared to FP16, so a 64GB Mac can fit a ~70B parameter model quantized to INT4.

---

## AMD — Ryzen AI Max+ (Strix Halo)

AMD's Strix Halo APUs bring unified memory to x86 laptops, offering a compelling alternative to Apple Silicon for portable LLM inference.

| Chip                  | GPU Cores   | Bandwidth | Max RAM | Max Model (INT4) | Est. tok/s (7B Q4) |
| --------------------- | ----------- | --------- | ------- | ---------------- | ------------------ |
| **Ryzen AI Max+ 395** | 40 RDNA 3.5 | ~215 GB/s | 128GB   | 128B             | ~35                |
| **Ryzen AI Max 390**  | 32 RDNA 3.5 | ~215 GB/s | 128GB   | 128B             | ~30                |

### Advantages over Apple Silicon

- **128GB RAM at ~$2,500 laptop price** (vs $4,000+ Mac Studio for equivalent RAM)
- **ROCm GPU compute support** for CUDA-like acceleration
- **x86 compatibility** — broader software ecosystem and driver support
- **Upgradeable** in some desktop configurations

### Disadvantages

- **Lower bandwidth** (215 GB/s vs 273+ GB/s for M4 Pro, 546 GB/s for M4 Max)
- **Less mature ML framework support** than MLX (Apple) or CUDA (NVIDIA)
- **Higher power consumption** than Apple Silicon equivalents
- **ROCm ecosystem** is less polished than CUDA for ML workloads

### Best Use Case

Budget-conscious users who need 70B+ model inference in a portable form factor. The 128GB unified memory at laptop prices makes Strix Halo uniquely positioned for large model inference where Apple's pricing is prohibitive.

---

## NVIDIA Consumer GPUs

NVIDIA GPUs offer the highest memory bandwidth and the most mature ML software ecosystem (CUDA, cuDNN, TensorRT). With AirLLM's layer-by-layer inference, **any model can run on any NVIDIA GPU** — VRAM determines maximum context length, not model size.

| GPU               | VRAM | Bandwidth  | Max Model (Layer-by-Layer INT4) | Est. tok/s (7B, layer-by-layer) |
| ----------------- | ---- | ---------- | ------------------------------- | ------------------------------- |
| **RTX 5090**      | 32GB | 1,792 GB/s | Any (via AirLLM)                | ~100+                           |
| **RTX 4090**      | 24GB | 1,008 GB/s | Any (via AirLLM)                | ~60                             |
| **RTX 4080**      | 16GB | 717 GB/s   | Any (via AirLLM)                | ~40                             |
| **RTX 4070 Ti**   | 12GB | 504 GB/s   | Any (via AirLLM)                | ~30                             |
| **RTX 4060**      | 8GB  | 272 GB/s   | Any (via AirLLM)                | ~15                             |
| **RTX 3090**      | 24GB | 936 GB/s   | Any (via AirLLM)                | ~50                             |
| **RTX 3060 12GB** | 12GB | 360 GB/s   | Any (via AirLLM)                | ~20                             |
| **GTX 1650**      | 4GB  | 128 GB/s   | Dense models only               | ~5                              |

> **Note:** With AirLLM layer-by-layer inference, **ANY model can run on ANY NVIDIA GPU**. The VRAM determines maximum context length, not model size. A 4GB GPU can run a 405B model — it will just be slower due to layer swapping.

### NVIDIA Advantages

- **Highest bandwidth** — RTX 5090 at 1,792 GB/s is 3× faster than any unified memory system
- **CUDA ecosystem** — most mature ML software stack (PyTorch, TensorRT, Flash Attention)
- **AirLLM sweet spot** — layer-by-layer inference leverages high bandwidth for fast layer swaps
- **AWQ/GPTQ support** — hardware-accelerated quantized inference via Marlin kernels

### NVIDIA Disadvantages

- **VRAM is fixed** — cannot be expanded (unlike unified memory systems)
- **PCIe bottleneck** — layer weights must traverse PCIe bus from system RAM to GPU
- **Power hungry** — RTX 4090 draws 450W vs ~30W for Apple Silicon
- **Multi-GPU cost** — scaling beyond single GPU VRAM requires expensive NVLink setups

---

## When is AirLLM Layer-by-Layer Needed?

AirLLM's layer-by-layer approach shines when the model does **not** fit entirely in available memory. Here's a decision guide:

### Apple Silicon Decision Matrix

| Scenario                             | Layer-by-Layer Needed?    | Recommendation            |
| ------------------------------------ | ------------------------- | ------------------------- |
| 7B model on M1 (16GB)                | No                        | Load fully via MLX        |
| 70B model on M1 Max (64GB)           | Only at FP16; No at INT4  | Use INT4 quantization     |
| 70B model on M2 Ultra (192GB)        | No                        | Load fully via MLX        |
| 405B model on M3 Ultra (512GB)       | Only at FP16; No at INT4  | Use INT4 via MLX          |
| 671B DeepSeek-R1 on M3 Ultra (512GB) | Yes even at INT4 (~400GB) | Use AirLLM layer-by-layer |

### NVIDIA Decision Matrix

| Scenario                       | Layer-by-Layer Needed? | Recommendation                    |
| ------------------------------ | ---------------------- | --------------------------------- |
| 7B model on RTX 4060 (8GB)     | No at INT4 (~3.5GB)    | Load fully with AWQ               |
| 7B model on GTX 1650 (4GB)     | Yes at FP16 (~14GB)    | Use AirLLM with 4-bit compression |
| 70B model on RTX 4090 (24GB)   | Yes (~35GB INT4)       | Use AirLLM with AWQ               |
| 405B model on any consumer GPU | Yes                    | Use AirLLM with 4-bit compression |

### General Rule

> **Use AirLLM layer-by-layer when:** `model_size_bytes > available_vram`
>
> **Use direct loading when:** `model_size_bytes ≤ available_vram`

---

## Budget Recommendations

### Best Value for Local LLM Inference

| Budget   | Best Option                   | Models Supported               | Notes                                 |
| -------- | ----------------------------- | ------------------------------ | ------------------------------------- |
| ~$600    | Mac Mini M4 (16GB)            | Up to 8B (FP16), 14B (INT4)    | Entry-level local AI                  |
| ~$800    | Mac Mini M4 Pro (24GB)        | Up to 12B (FP16), 24B (INT4)   | Sweet spot for small models           |
| ~$1,400  | Mac Mini M4 Pro (48GB)        | Up to 24B (FP16), 48B (INT4)   | Good for Qwen3-32B                    |
| ~$2,000  | Mac Mini M4 Pro (64GB)        | Up to 32B (FP16), 70B (INT4)   | Best value for 70B models             |
| ~$2,500  | AMD Strix Halo laptop (128GB) | Up to 70B (FP16), 128B (INT4)  | Portable 70B+ inference               |
| ~$4,000  | Mac Studio M4 Max (128GB)     | Up to 64B (FP16), 128B (INT4)  | Professional local AI workstation     |
| ~$8,000+ | Mac Studio M3 Ultra (512GB)   | Up to 256B (FP16), 671B (INT4) | Enterprise — runs DeepSeek-R1 locally |

### Budget NVIDIA Options (Layer-by-Layer with AirLLM)

| Budget  | GPU                | Best For                                        |
| ------- | ------------------ | ----------------------------------------------- |
| ~$300   | RTX 4060 (8GB)     | Budget layer-by-layer inference for any model   |
| ~$600   | RTX 4070 Ti (12GB) | Comfortable layer-by-layer with longer contexts |
| ~$1,200 | RTX 4090 (24GB)    | Fastest consumer layer-by-layer with AWQ        |
| ~$2,000 | RTX 5090 (32GB)    | Ultimate layer-by-layer speed                   |

---

## Key Concepts

### Memory Bandwidth vs VRAM Capacity

- **VRAM capacity** determines the maximum model size you can load at once (or with AirLLM, the maximum context length).
- **Memory bandwidth** determines generation speed (tokens/sec). Higher bandwidth = faster token generation.

### Quantization Impact on Model Size

| Precision | Bytes per Parameter | 7B Model | 70B Model | 405B Model |
| --------- | ------------------: | -------: | --------: | ---------: |
| FP16      |                 2.0 |    14 GB |    140 GB |     810 GB |
| INT8      |                 1.0 |     7 GB |     70 GB |     405 GB |
| INT4      |                 0.5 |   3.5 GB |     35 GB |     203 GB |

### AirLLM Layer-by-Layer Overhead

When using AirLLM's layer-by-layer approach, expect:

- **Throughput:** 1–10 tokens/sec (vs 30–100+ when model fits in memory)
- **Bottleneck:** Disk → CPU RAM → GPU transfer speed (PCIe bandwidth)
- **Optimization:** Use `compression="4bit"` to reduce transfer size by ~4×
- **Best with:** NVMe Gen4/Gen5 SSDs for fastest layer loading

### Related Documentation

- [VRAM Management Architecture](VRAM_MANAGEMENT.md) — detailed memory optimization strategies
- [Supported Models Reference](SUPPORTED_MODELS.md) — per-model VRAM requirements
- [Scientific Review](SCIENTIFIC_REVIEW.md) — mathematical analysis of throughput and latency
- [2026 Research](2026_LLM_INFERENCE_RESEARCH.md) — latest hardware and software findings
