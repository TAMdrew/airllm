# Supported Models Reference

AirLLM v3.0 supports a wide range of Large Language Models (LLMs) out of the box — all running **100% locally** on your hardware with no cloud API dependencies. The [`AutoModel.from_pretrained()`](../air_llm/airllm/auto_model.py:62) method automatically detects the architecture from the HuggingFace `config.json` (parsed locally) and routes it to the correct backend.

HuggingFace libraries run entirely locally for config parsing, tokenization, and one-time weight download. No API keys or subscriptions are required for inference. After initial download, everything runs offline (`HF_HUB_OFFLINE=1`). Zero subscription fees — your hardware, your models, forever.

This document provides a detailed breakdown of supported architectures, their VRAM requirements, and specific usage notes.

## No Account Required

**The vast majority of supported models are publicly available and require NO account, NO API key, and NO subscription to download or use.** AirLLM downloads model weights once via direct HTTPS, then runs 100% offline forever.

| Access Level | Models | Account Needed? |
|---|---|---|
| ✅ **Public** (no account) | Qwen 3, Gemma 2/4, DeepSeek R1, Mistral, Phi-4, Falcon 3, OLMo 2, Granite, Jamba, Mixtral, Cohere, Zamba 2, and more | **NO** |
| ⚠️ **Gated** (free account) | Meta Llama 3/4 (requires Meta license acceptance) | Free HF account + license click |

For gated models, you only need a free HuggingFace account to accept the license agreement once. No paid tier is ever required.

## Table of Contents

- [Supported Models Reference](#supported-models-reference)
  - [Table of Contents](#table-of-contents)
  - [Llama Family](#llama-family)
    - [VRAM Requirements](#vram-requirements)
  - [Gemma Family](#gemma-family)
    - [VRAM Requirements](#vram-requirements-1)
  - [Qwen Family](#qwen-family)
    - [VRAM Requirements](#vram-requirements-2)
  - [DeepSeek Family](#deepseek-family)
    - [VRAM Requirements](#vram-requirements-3)
  - [Mistral \& Mixtral](#mistral--mixtral)
    - [VRAM Requirements](#vram-requirements-4)
  - [GLM \& ChatGLM](#glm--chatglm)
    - [VRAM Requirements](#vram-requirements-5)
  - [Phi Family](#phi-family)
    - [VRAM Requirements](#vram-requirements-6)
  - [Cohere](#cohere)
    - [VRAM Requirements](#vram-requirements-7)
  - [Zamba 2](#zamba-2)
    - [VRAM Requirements](#vram-requirements-8)
  - [Llama 4 Family](#llama-4-family)
    - [VRAM Requirements](#vram-requirements-9)
  - [Falcon 3](#falcon-3)
  - [OLMo 2](#olmo-2)
  - [Granite](#granite)
  - [Jamba](#jamba)
    - [VRAM Requirements](#vram-requirements-10)
  - [Other Models](#other-models)

---

## Llama Family

The standard Llama architecture is the foundation for many open-source models.

- **Backend Class:** `AirLLMLlama2`
- **Architecture Strings:** `LlamaForCausalLM`, `LLaMAForCausalLM`
- **Example Models:** `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`, `meta-llama/Llama-3.1-405B-Instruct`

### VRAM Requirements

| Variant | FP16 | 8-bit | 4-bit |
| ------- | ---- | ----- | ----- |
| 8B      | 4 GB | 4 GB  | 4 GB  |
| 70B     | 4 GB | 4 GB  | 4 GB  |
| 405B    | 8 GB | 8 GB  | 8 GB  |

**Notes:** The 405B model requires slightly more VRAM due to the sheer size of the embedding and LM head layers, even when processed layer-by-layer.

---

## Gemma Family

Google's Gemma models, including the latest Gemma 4 with alternating attention.

- **Backend Classes:** `AirLLMGemma`, `AirLLMGemma2`, `AirLLMGemma3`, `AirLLMGemma4`
- **Architecture Strings:** `GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration`, `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`
- **Example Models:** `google/gemma-4-12b-it`, `google/gemma-2-27b-it`

### VRAM Requirements

| Variant | FP16 | 8-bit | 4-bit |
| ------- | ---- | ----- | ----- |
| ≤ 12B   | 4 GB | 4 GB  | 4 GB  |
| 27B/31B | 4 GB | 4 GB  | 4 GB  |

**Notes:** Gemma 4 supports Per-Layer Embeddings (PLE) and alternating sliding-window/global attention natively in AirLLM.

---

## Qwen Family

Alibaba's Qwen models, featuring GQA and YaRN RoPE scaling in newer versions.

- **Backend Classes:** `AirLLMQWen`, `AirLLMQWen2`, `AirLLMQwen3`
- **Architecture Strings:** `QWenLMHeadModel`, `Qwen2ForCausalLM`, `Qwen2_5ForCausalLM`, `Qwen3ForCausalLM`
- **Example Models:** `Qwen/Qwen3-8B`, `Qwen/Qwen2.5-72B-Instruct`, `Qwen/QwQ-32B-Preview`

### VRAM Requirements

| Variant | FP16 | 8-bit | 4-bit |
| ------- | ---- | ----- | ----- |
| ≤ 14B   | 4 GB | 4 GB  | 4 GB  |
| 32B     | 4 GB | 4 GB  | 4 GB  |
| 72B     | 4 GB | 4 GB  | 4 GB  |

**Notes:** QwQ-32B uses the `Qwen2ForCausalLM` architecture and is fully supported. Older QWen v1 models use a custom layer naming scheme handled automatically by `AirLLMQWen`.

---

## DeepSeek Family

DeepSeek's massive Mixture-of-Experts (MoE) models with Multi-head Latent Attention (MLA).

- **Backend Class:** `AirLLMDeepSeek`
- **Architecture Strings:** `DeepseekV3ForCausalLM`, `DeepseekV2ForCausalLM`
- **Example Models:** `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1`

### VRAM Requirements

| Variant | FP16  | 8-bit | 4-bit |
| ------- | ----- | ----- | ----- |
| 671B    | 24 GB | 16 GB | 16 GB |

**Warning:** DeepSeek V3/R1 MoE layers are extremely large (~22 GB per layer in FP16). Systems with less than 16 GB VRAM will experience significant disk I/O overhead. Using `compression="4bit"` is highly recommended.

_Note: DeepSeek R1-Distill variants (e.g., R1-Distill-Qwen-32B) use the underlying Qwen or Llama architecture and are handled by those backends._

---

## Mistral & Mixtral

Mistral AI's dense and MoE models.

- **Backend Classes:** `AirLLMMistral`, `AirLLMMixtral`
- **Architecture Strings:** `MistralForCausalLM`, `Mistral4ForCausalLM`, `Mistral3ForConditionalGeneration`, `MixtralForCausalLM`
- **Example Models:** `mistralai/Mistral-7B-Instruct-v0.2`, `mistralai/Mixtral-8x7B-Instruct-v0.1`

### VRAM Requirements

| Variant | FP16  | 8-bit | 4-bit |
| ------- | ----- | ----- | ----- |
| 7B      | 4 GB  | 4 GB  | 4 GB  |
| 8x7B    | 8 GB  | 6 GB  | 4 GB  |
| 8x22B   | 16 GB | 12 GB | 8 GB  |

---

## GLM & ChatGLM

Zhipu AI's GLM models. GLM-4 uses standard Hugging Face architecture, while older versions use a custom structure.

- **Backend Classes:** `AirLLMGlm4`, `AirLLMChatGLM`
- **Architecture Strings:** `GlmForCausalLM`, `Glm4ForCausalLM`, `ChatGLM4Model`, `ChatGLMModel`
- **Example Models:** `THUDM/glm-4-9b-chat`, `THUDM/chatglm3-6b`

### VRAM Requirements

| Variant | FP16 | 8-bit | 4-bit |
| ------- | ---- | ----- | ----- |
| 6B/9B   | 4 GB | 4 GB  | 4 GB  |

---

## Phi Family

Microsoft's highly capable small language models.

- **Backend Class:** `AirLLMPhi`
- **Architecture Strings:** `Phi3ForCausalLM`, `PhiForCausalLM`, `Phi4ForCausalLM`
- **Example Models:** `microsoft/phi-4`, `microsoft/Phi-3-mini-4k-instruct`

### VRAM Requirements

| Variant | FP16 | 8-bit | 4-bit |
| ------- | ---- | ----- | ----- |
| ≤ 14B   | 4 GB | 4 GB  | 4 GB  |

---

## Cohere

Cohere's Command-R models, optimized for RAG.

- **Backend Class:** `AirLLMCohere`
- **Architecture Strings:** `CohereForCausalLM`, `Cohere2ForCausalLM`
- **Example Models:** `CohereForAI/c4ai-command-r-v01`

### VRAM Requirements

| Variant | FP16 | 8-bit | 4-bit |
| ------- | ---- | ----- | ----- |
| 35B     | 4 GB | 4 GB  | 4 GB  |
| 104B    | 8 GB | 8 GB  | 8 GB  |

---

## Zamba 2

Zyphra's hybrid architecture interleaving Mamba2 layers with shared attention.

- **Backend Class:** `AirLLMZamba2`
- **Architecture Strings:** `Zamba2ForCausalLM`
- **Example Models:** `Zyphra/Zamba2-7B-instruct`

### VRAM Requirements

| Variant | FP16 | 8-bit | 4-bit |
| ------- | ---- | ----- | ----- |
| 2.7B/7B | 4 GB | 4 GB  | 4 GB  |

---

## Llama 4 Family

Meta's Llama 4 introduces Mixture-of-Experts (MoE) architecture with interleaved dense + MoE layers and iRoPE attention.

- **Backend Class:** `AirLLMLlama4`
- **Architecture Strings:** `Llama4ForCausalLM`, `Llama4ForConditionalGeneration`
- **Example Models:** `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct`

### VRAM Requirements

| Variant | FP16 | 8-bit | 4-bit |
| ------- | ---- | ----- | ----- |
| Scout (109B total, 17B active) | 8 GB | 6 GB | 4 GB |
| Maverick (400B total, 17B active) | 12 GB | 8 GB | 6 GB |

---

## Falcon 3

TII's Falcon 3 family of open-source models.

- **Backend Class:** `AirLLMFalcon3`
- **Architecture Strings:** `FalconForCausalLM`
- **Example Models:** `tiiuae/Falcon3-7B-Instruct`, `tiiuae/Falcon3-10B-Instruct`

---

## OLMo 2

AI2's fully open-source language model with reproducible training data, code, and checkpoints.

- **Backend Class:** `AirLLMOLMo2`
- **Architecture Strings:** `OLMo2ForCausalLM`
- **Example Models:** `allenai/OLMo-2-7B`, `allenai/OLMo-2-13B`, `allenai/OLMo-2-32B`

---

## Granite

IBM's enterprise-focused open-source models for RAG, code generation, and multi-language tasks.

- **Backend Class:** `AirLLMGranite`
- **Architecture Strings:** `GraniteForCausalLM`
- **Example Models:** `ibm-granite/granite-3.1-8b-instruct`

---

## Jamba

AI21's hybrid SSM-Transformer model interleaving Mamba (state-space) layers with Transformer attention.

- **Backend Class:** `AirLLMJamba`
- **Architecture Strings:** `JambaForCausalLM`
- **Example Models:** `ai21labs/AI21-Jamba-1.5-Mini`, `ai21labs/AI21-Jamba-1.5-Large`
- **Note:** Requires `trust_remote_code=True` (set automatically by the backend)

### VRAM Requirements

| Variant | FP16 | 4-bit |
| ------- | ---- | ----- |
| Mini (52B total, 12B active) | 8 GB | 4 GB |
| Large (398B total, 94B active) | 16 GB | 8 GB |

---

## Other Models

- **Baichuan:** `BaichuanForCausalLM` (e.g., `baichuan-inc/Baichuan2-7B-Base`) - 4 GB VRAM
- **InternLM:** `InternLMForCausalLM` (e.g., `internlm/internlm-20b`) - 4 GB VRAM
