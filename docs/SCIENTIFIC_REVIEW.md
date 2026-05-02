# Scientific and Mathematical Review of AirLLM

This document provides a rigorous mathematical foundation and scientific analysis of the AirLLM inference engine — a 100% local inference system requiring no cloud APIs, no subscriptions, and no internet after initial model download. It covers the theoretical underpinnings of layer-by-layer inference, memory complexity, quantization mathematics, and architectural adaptations for state-of-the-art Large Language Models (LLMs).

## Table of Contents

- [Scientific and Mathematical Review of AirLLM](#scientific-and-mathematical-review-of-airllm)
  - [Table of Contents](#table-of-contents)
  - [1. Layer-by-Layer Inference — Mathematical Foundation](#1-layer-by-layer-inference--mathematical-foundation)
    - [Memory Complexity Analysis](#memory-complexity-analysis)
    - [Throughput Analysis](#throughput-analysis)
    - [Latency Breakdown](#latency-breakdown)
  - [2. Quantization Mathematics](#2-quantization-mathematics)
    - [Block-wise Absmax Quantization](#block-wise-absmax-quantization)
    - [Impact on Model Quality](#impact-on-model-quality)
    - [Block Size Impact](#block-size-impact)
  - [3. KV Cache Analysis](#3-kv-cache-analysis)
    - [KV Cache Size Formula](#kv-cache-size-formula)
    - [Architectural Variations](#architectural-variations)
    - [VRAM Budget and Maximum Sequence Length](#vram-budget-and-maximum-sequence-length)
  - [4. MoE Architecture Considerations](#4-moe-architecture-considerations)
    - [Dense vs. MoE Parameters](#dense-vs-moe-parameters)
    - [Expert-by-Expert Loading Strategy](#expert-by-expert-loading-strategy)
  - [5. Prefetching Pipeline Analysis](#5-prefetching-pipeline-analysis)
    - [Optimal Prefetching](#optimal-prefetching)
    - [Bandwidth Considerations](#bandwidth-considerations)
    - [Prefetching and Quantization](#prefetching-and-quantization)
  - [6. Attention Mechanism Variants](#6-attention-mechanism-variants)
  - [7. 2026 State-of-the-Art Techniques](#7-2026-state-of-the-art-techniques)
    - [7.1 TurboQuant — KV Cache Compression (ICLR 2026)](#71-turboquant--kv-cache-compression-iclr-2026)
      - [PolarQuant — Rotation-Based Normalization-Free Quantization](#polarquant--rotation-based-normalization-free-quantization)
      - [QJL — Johnson-Lindenstrauss Residual Compression](#qjl--johnson-lindenstrauss-residual-compression)
      - [Combined Memory Reduction](#combined-memory-reduction)
  - [8. Performance Benchmarks Framework](#8-performance-benchmarks-framework)
    - [Benchmark Methodology Template](#benchmark-methodology-template)
    - [Expected Performance Ranges](#expected-performance-ranges)

---

## 1. Layer-by-Layer Inference — Mathematical Foundation

For a transformer model with $L$ layers, traditional inference requires loading all $L$ layers simultaneously into Video RAM (VRAM). AirLLM circumvents this by loading layers sequentially, as implemented in the core loop of [`AirLLMBaseModel.forward`](../air_llm/airllm/airllm_base.py).

### Memory Complexity Analysis

Let $P$ be the total number of parameters in the model.
- **Traditional Inference:** The memory complexity is $O(P)$.
- **AirLLM Inference:** The memory complexity is reduced to $O(P/L + KV + A)$, where:
  - $P/L$ is the memory required for a single transformer layer.
  - $KV$ is the memory required for the Key-Value cache.
  - $A$ is the memory required for intermediate activations.

### Throughput Analysis

The throughput (tokens per second) in a layer-by-layer architecture is a function of disk read speed, PCIe bandwidth, and GPU compute time:

$$ \text{Throughput} = f(B_{\text{disk}}, B_{\text{pcie}}, t_{\text{compute}}) $$

Where $B_{\text{disk}}$ is the disk bandwidth and $B_{\text{pcie}}$ is the PCIe bandwidth.

### Latency Breakdown

The total time $t_{\text{total}}$ to generate a single token is the sum of the times taken for each layer, minus the time saved through prefetching overlap:

$$ t_{\text{total}} = L \times (t_{\text{disk\_to\_cpu}} + t_{\text{cpu\_to\_gpu}} + t_{\text{compute}} + t_{\text{cleanup}}) - (L-1) \times t_{\text{prefetch\_overlap}} $$

Where:
- $t_{\text{disk\_to\_cpu}}$: Time to load the layer from NVMe to CPU RAM.
- $t_{\text{cpu\_to\_gpu}}$: Time to transfer the layer over PCIe to VRAM.
- $t_{\text{compute}}$: GPU execution time for the forward pass.
- $t_{\text{cleanup}}$: Time to offload the layer to the `meta` device and run garbage collection.
- $t_{\text{prefetch\_overlap}}$: Time saved by asynchronously loading layer $N+1$ while layer $N$ is computing.

---

## 2. Quantization Mathematics

AirLLM supports block-wise quantization (4-bit NF4 and 8-bit) via `bitsandbytes`, as implemented in [`utils.py`](../air_llm/airllm/utils.py). This significantly reduces disk I/O and PCIe transfer times at the cost of minor precision loss.

### Block-wise Absmax Quantization

For a block of weights $x$ of size $B$, the absolute maximum quantization works mathematically as follows:

1. **Scaling Factor:** Calculate the scale based on the maximum absolute value in the block.
   $$ \text{scale} = \frac{\max(|x|)}{2^{\text{bits}-1} - 1} $$
2. **Quantization:**
   $$ q = \text{round}\left(\frac{x}{\text{scale}}\right) $$
3. **Dequantization:**
   $$ \hat{x} = q \times \text{scale} $$
4. **Error Bound:** The maximum quantization error for any single weight is:
   $$ |x - \hat{x}| \le \frac{\text{scale}}{2} $$

### Impact on Model Quality

Research indicates that 8-bit quantization introduces negligible perplexity degradation, while 4-bit quantization (specifically NormalFloat4) introduces minimal degradation. The degradation curve is non-linear and typically more pronounced in smaller models (<10B parameters) than in larger models (>70B parameters).

### Block Size Impact

The block size $B$ (default is 64 for 4-bit) dictates the granularity of the scaling factor.
- **Larger Blocks:** Reduce memory overhead for storing scales but increase the quantization error, as outliers skew the scale for the entire block.
- **Smaller Blocks:** Minimize quantization error but increase the metadata overhead (more scales to store and transfer).

---

## 3. KV Cache Analysis

The Key-Value (KV) cache is the primary VRAM bottleneck for long-context inference.

### KV Cache Size Formula

The total size of the KV cache in bytes is calculated as:

$$ KV_{\text{bytes}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{dtype\_bytes} $$

Where:
- $2$ accounts for both Key and Value tensors.
- $n_{\text{kv\_heads}}$ is the number of KV heads.
- $d_{\text{head}}$ is the dimension of each head.
- $\text{dtype\_bytes}$ is 2 for FP16/BF16.

### Architectural Variations

- **Grouped Query Attention (GQA):** In models like Llama 3 and Qwen, $n_{\text{kv\_heads}} < n_{\text{heads}}$. This proportionally reduces the KV cache size, allowing for longer contexts within the same VRAM budget.
- **Multi-head Latent Attention (MLA):** DeepSeek models utilize MLA, which compresses the KV representation into a latent vector, reducing the cache size by a factor of 4x to 8x.

### VRAM Budget and Maximum Sequence Length

The available VRAM for the KV cache is:

$$ \text{Available\_VRAM} = \text{Total\_VRAM} - \text{Layer\_VRAM} - \text{Embedding\_VRAM} - \text{Overhead} $$

The theoretical maximum sequence length is therefore:

$$ \text{max\_seq} = \frac{\text{Available\_VRAM}}{2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{dtype\_bytes}} $$

*(Note: This formula assumes the KV cache is kept entirely in VRAM. Future implementations may offload older KV blocks to CPU RAM, as detailed in [VRAM_MANAGEMENT.md](VRAM_MANAGEMENT.md).)*

---

## 4. MoE Architecture Considerations

Mixture of Experts (MoE) models present unique challenges for layer-by-layer inference due to the massive size of individual layers.

### Dense vs. MoE Parameters

- **Standard Dense Layer:** Contains one Feed-Forward Network (FFN).
  $$ \text{Params}_{\text{dense}} = 2 \times d_{\text{model}} \times d_{\text{ffn}} \times 2 $$
  *(Assuming gated SwiGLU architecture: gate, up, down projections).*
- **MoE Layer:** Contains $N$ expert FFNs plus a router.
  $$ \text{Params}_{\text{moe}} = N \times (2 \times d_{\text{model}} \times d_{\text{ffn}} \times 2) + \text{Params}_{\text{router}} $$

For example, DeepSeek-V3 has 256 experts per layer, resulting in approximately 11B parameters (~22GB in FP16) for a *single layer*.

### Expert-by-Expert Loading Strategy

To run MoE models on low VRAM (e.g., 8GB), AirLLM implements an expert-by-expert loading strategy:
1. Load the router weights.
2. Compute routing probabilities to determine the top-$K$ active experts for the current sequence.
3. Load only the $K$ activated experts from disk to VRAM.

**Theoretical VRAM Complexity:** The VRAM requirement drops from $O(N \times \text{FFN\_params})$ to $O(K \times \text{FFN\_params})$, where $K \ll N$.

---

## 5. Prefetching Pipeline Analysis

AirLLM uses asynchronous prefetching to hide disk latency. The pipeline consists of four stages: Disk $\rightarrow$ CPU RAM $\rightarrow$ GPU VRAM $\rightarrow$ Compute.

### Optimal Prefetching

The optimal scenario occurs when the time to load layer $N+1$ from disk to CPU is completely hidden by the GPU computation of layer $N$:

$$ t_{\text{prefetch}} = t_{\text{disk\_to\_cpu}} $$

For full overlap, the condition $t_{\text{compute}} \ge t_{\text{disk\_to\_cpu}}$ must hold.

### Bandwidth Considerations

- **Typical NVMe Bandwidth:** ~3.5 GB/s.
- Loading a 100MB layer takes approximately $28$ ms.
- If $t_{\text{compute}} < 28$ ms (common for small batch sizes), the system is I/O bound.

### Prefetching and Quantization

Currently, `self.prefetching` is disabled when `compression` is active. This is because dynamic dequantization (or the state dict manipulation required by `bitsandbytes`) occurs on the CPU, adding significant synchronous overhead that disrupts the asynchronous CUDA stream, leading to pipeline stalls.

---

## 6. Attention Mechanism Variants

AirLLM supports various attention mechanisms, each with distinct computational complexities:

- **Standard Multi-Head Attention (MHA):** Complexity is $O(n^2 \times d)$ per head, where $n$ is sequence length and $d$ is head dimension.
- **Grouped Query Attention (GQA):** Reduces memory bandwidth requirements by sharing KV heads across multiple query heads. Compute complexity remains similar to MHA, but memory transfer is reduced.
- **Multi-head Latent Attention (MLA):** Compresses the KV cache into a latent space, significantly reducing memory footprint during generation.
- **Sliding Window Attention (SWA):** Used in Mistral and Gemma 4. Complexity is reduced to $O(n \times w \times d)$, where $w$ is the fixed window size.
- **Alternating Patterns:** Gemma 4 uses a mix of sliding window and full global attention layers, requiring dynamic KV cache management.

---

## 7. 2026 State-of-the-Art Techniques

AirLLM v3.0 integrates several cutting-edge techniques:

- **FP8 Quantization:** Native support for FP8 (e.g., DeepSeek-V3), utilizing hardware-accelerated FP8 tensor cores on modern GPUs.
- **Per-Layer Embeddings (PLE):** Supported for Gemma 4 (E2B/E4B), where embeddings are specific to layers rather than shared.
- **Dynamic RoPE / YaRN Scaling:** Automatically adjusts Rotary Positional Embeddings for context windows extending beyond the model's native training length.
- **Mamba/SSM Hybrid Architectures:** Support for models like Zamba 2, which interleave state-space models (Mamba2) with shared attention layers, requiring specialized state persistence between layers.
- **Flash Attention v3:** Integration opportunities exist to further optimize the $t_{\text{compute}}$ phase, particularly for the attention block.

### 7.1 TurboQuant — KV Cache Compression (ICLR 2026)

TurboQuant is a KV cache compression method that reduces the cache to 3 bits per element while preserving 99.5% of attention fidelity. Unlike weight quantization (AWQ/GPTQ/bitsandbytes), TurboQuant targets the **KV cache** — the persistent memory that grows with sequence length during generation.

#### PolarQuant — Rotation-Based Normalization-Free Quantization

Traditional KV cache quantization requires computing channel-wise statistics (mean, variance) for normalization, which is expensive for streaming inference. PolarQuant eliminates this overhead:

1. **Random Rotation:** Apply a fixed random orthogonal matrix $R$ to the KV vectors:
   $$ \tilde{k} = R \cdot k, \quad \tilde{v} = R \cdot v $$
   By the Johnson-Lindenstrauss lemma, the rotation distributes the energy uniformly across dimensions, eliminating outlier channels.

2. **Polar Coordinate Conversion:** Convert the rotated vector to polar form:
   $$ \tilde{k} = r \cdot \hat{u}, \quad r = \|\tilde{k}\|_2, \quad \hat{u} = \tilde{k} / r $$
   The unit direction $\hat{u}$ is quantized independently from the magnitude $r$.

3. **Uniform Quantization of Direction:** Since the rotation eliminates outliers, the direction components are approximately uniformly distributed in $[-1, 1]$, enabling efficient uniform quantization:
   $$ \hat{u}_q = \text{round}\left(\hat{u} \cdot (2^{b-1} - 1)\right) / (2^{b-1} - 1) $$
   where $b$ is the target bit-width (typically 3).

4. **Key Advantage:** No per-channel statistics computation, no normalization overhead — the random rotation provides a "free" normalization effect.

#### QJL — Johnson-Lindenstrauss Residual Compression

The quantization residual $\epsilon = \hat{u} - \hat{u}_q$ is compressed using random projection:

1. **JL Projection:** Project the residual onto a random sign matrix $S \in \{-1, +1\}^{m \times d}$:
   $$ z = \text{sign}(S \cdot \epsilon) \in \{-1, +1\}^m $$
   Each projected component requires only 1 bit to store.

2. **Theoretical Guarantee:** The Johnson-Lindenstrauss lemma ensures that for any two vectors $a, b$:
   $$ (1 - \varepsilon) \|a - b\|^2 \le \frac{d}{m} \|S \cdot a - S \cdot b\|^2 \le (1 + \varepsilon) \|a - b\|^2 $$
   with high probability, where $m = O(\varepsilon^{-2} \log n)$.

3. **Attention Score Preservation:** Since attention computes $\text{softmax}(QK^T / \sqrt{d})$, the inner product geometry preserved by JL projection directly translates to preserved attention scores.

#### Combined Memory Reduction

For a KV cache at FP16 precision:

$$ \text{Original:} \quad 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times 16 \text{ bits} $$

$$ \text{TurboQuant:} \quad 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times 3 \text{ bits} + O(\text{magnitudes}) $$

This yields approximately a **5× reduction** in KV cache memory, enabling:
- Longer context windows within the same VRAM budget
- Reduced memory pressure during AirLLM layer-by-layer inference
- Up to **8× speedup** in attention computation due to reduced memory bandwidth for KV reads

---

## 8. Performance Benchmarks Framework

To rigorously evaluate AirLLM, we define the following standard metrics:

1. **Tokens/sec:** The generation speed during the decoding phase.
2. **Time-to-First-Token (TTFT):** The latency of the initial prompt processing (prefill phase).
3. **Peak VRAM:** The maximum GPU memory allocated during the forward pass.
4. **Disk I/O Utilization:** The average read bandwidth sustained during inference.

### Benchmark Methodology Template

When comparing models or hardware, use the following framework:
- **Hardware:** Specify CPU, RAM, GPU, and exact NVMe SSD model.
- **Model:** Specify architecture, parameter count, and quantization level.
- **Prompt:** Use a standardized prompt length (e.g., 512 tokens).
- **Generation:** Generate a fixed number of tokens (e.g., 128 tokens).

### Expected Performance Ranges

| VRAM Tier | Disk Speed | Expected Throughput (70B FP16) | Expected Throughput (70B 4-bit) |
| :--- | :--- | :--- | :--- |
| PCIe Gen3 | ~3.0 GB/s | 0.5 - 1.0 tokens/sec | 1.5 - 2.5 tokens/sec |
| PCIe Gen4 | ~7.0 GB/s | 1.0 - 2.0 tokens/sec | 3.0 - 5.0 tokens/sec |
| PCIe Gen5 | ~12.0 GB/s | 2.0 - 3.5 tokens/sec | 5.0 - 8.0 tokens/sec |

*Note: Throughput is heavily bottlenecked by disk read speeds in the layer-by-layer paradigm.*
