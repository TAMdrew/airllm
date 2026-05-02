# New Model Integration Plan

## 1. Overview

AirLLM supports a wide range of Large Language Models (LLMs) — all running **100% locally** with no cloud API dependencies. The project abstracts model architectures into a standard layer-by-layer execution flow. This document provides a step-by-step guide and configuration-driven approach for integrating new models, ensuring consistency and ease of maintenance.

> **Important:** When adding new model support, ensure all operations remain local. HuggingFace libraries are used locally for config parsing and tokenization. No cloud inference APIs should ever be introduced.

## 2. Configuration-Driven Model Definition

Instead of writing boilerplate Python classes for every new model, we will move towards a configuration-driven approach. Most dense models share the same fundamental structure (Embeddings $\rightarrow$ N Layers $\rightarrow$ Norm $\rightarrow$ LM Head) but use different attribute names in their Hugging Face implementations.

### 2.1. The Layer Mapping Dictionary

The core of adding a new model is defining its `layer_names_dict`. This maps the standard `AirLLM` internal names to the specific attribute paths used by the model's `transformers` implementation.

```python
# Default mapping (used by Llama, Qwen, Mistral, Gemma, etc.)
default_layer_names = {
    'embed': 'model.embed_tokens',
    'layer_prefix': 'model.layers',
    'norm': 'model.norm',
    'lm_head': 'lm_head',
}
```

If a new model uses this exact structure, it requires **zero** new code. It simply needs to be registered.

### 2.2. Registering a New Model

To add a new model, create a new file in `airllm/models/` (e.g., `airllm_phi4.py`) and use the `@ModelRegistry.register` decorator.

```python
# airllm/models/airllm_phi4.py
from airllm.models.registry import ModelRegistry
from airllm.core.base import AirLLMStandardModel

@ModelRegistry.register("Phi3ForCausalLM") # The architecture string from config.json
class AirLLMPhi4(AirLLMStandardModel):
    pass # Uses default layer names and standard dense execution
```

If the model has non-standard layer names (like ChatGLM), override `set_layer_names_dict`:

```python
# airllm/models/airllm_chatglm.py
from airllm.models.registry import ModelRegistry
from airllm.core.base import AirLLMCustomModel

@ModelRegistry.register("ChatGLMModel")
class AirLLMChatGLM(AirLLMCustomModel):
    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'transformer.embedding.word_embeddings',
            'layer_prefix': 'transformer.encoder.layers',
            'norm': 'transformer.encoder.final_layernorm',
            'lm_head': 'transformer.output_layer',
            'rotary_pos_emb': 'transformer.rotary_pos_emb'
        }

    # Override other methods if the forward pass logic differs significantly
```

## 3. Step-by-Step Guide for Adding a New Model

### Step 1: Identify the Architecture String
Find the target model on Hugging Face (e.g., `Qwen/Qwen2.5-72B-Instruct`). Check its `config.json` for the `architectures` array (e.g., `["Qwen2ForCausalLM"]`).

### Step 2: Analyze the Model Structure
Look at the model's `modeling_*.py` file in the `transformers` library (or the model's repo if it uses `trust_remote_code=True`).
- Does it have standard `embed_tokens`, `layers`, `norm`, and `lm_head`?
- Is it a dense model or a Mixture of Experts (MoE)?
- Does it use standard RoPE (Rotary Position Embedding) or something custom (like DeepSeek's MLA or Gemma's PLE)?

### Step 3: Create the Model Class
Create a new file in `airllm/models/`.
- If it's a standard dense model, inherit from `AirLLMStandardModel`.
- If it's an MoE model, inherit from `AirLLMMoEModel` and implement expert loading logic if necessary.
- If it's highly custom, inherit from `AirLLMCustomModel` and override `forward()` or `get_pos_emb_args()`.

### Step 4: Register the Architecture
Use the `@ModelRegistry.register("ArchitectureString")` decorator on your class.

### Step 5: Test the Integration
Run the standard integration test suite (see Section 4) against a small version of the model (e.g., a 0.5B or 1.5B variant).

## 4. Testing Strategy for Each Model

Every new model must pass the following tests before being merged:

1.  **Instantiation Test:** Can `AutoModel.from_pretrained` correctly identify and instantiate the model class based on the repo ID?
2.  **Splitting Test:** Can the model be successfully downloaded, split into layer shards, and saved to disk (with and without compression)?
3.  **Inference Test (Dense):** Does a basic forward pass produce coherent text output?
4.  **Inference Test (Quantized):** Does the model produce coherent text when `compression='4bit'` or `'8bit'` is enabled?
5.  **Equivalence Test:** (Optional but recommended) Does the output of `AirLLM` match the output of standard `transformers` (using a small model that fits in VRAM normally)?

We will create a parameterized pytest fixture in `tests/test_models.py` that automatically runs these tests for a predefined list of "tiny" model checkpoints representing each architecture family.

## 5. Prompt Formatting Reference

Different model families require specific prompt templates (chat templates). While `transformers` handles this via `tokenizer.apply_chat_template()`, users often construct prompts manually. We will document the standard formats for the newly supported models:

| Model Family | System Prompt | User Prompt | Assistant Response |
| :--- | :--- | :--- | :--- |
| **Gemma** | N/A | `<start_of_turn>user\n{prompt}<end_of_turn>\n` | `<start_of_turn>model\n{response}<end_of_turn>\n` |
| **Qwen** | `<\|im_start\|>system\n{system}<\|im_end\|>\n` | `<\|im_start\|>user\n{prompt}<\|im_end\|>\n` | `<\|im_start\|>assistant\n{response}<\|im_end\|>\n` |
| **Mistral** | N/A | `[INST] {prompt} [/INST]` | `{response}` |
| **DeepSeek** | `<\|system\|>{system}` | `<\|user\|>{prompt}` | `<\|assistant\|>{response}` |
| **GLM-4** | `<\|system\|>\n{system}` | `<\|user\|>\n{prompt}` | `<\|assistant\|>\n{response}` |
| **Phi-4** | `<\|system\|>\n{system}<\|end\|>\n` | `<\|user\|>\n{prompt}<\|end\|>\n` | `<\|assistant\|>\n{response}<\|end\|>\n` |
| **Command-R** | `<\|START_OF_TURN_TOKEN\|><\|SYSTEM_TOKEN\|>{system}<\|END_OF_TURN_TOKEN\|>` | `<\|START_OF_TURN_TOKEN\|><\|USER_TOKEN\|>{prompt}<\|END_OF_TURN_TOKEN\|>` | `<\|START_OF_TURN_TOKEN\|><\|CHATBOT_TOKEN\|>{response}<\|END_OF_TURN_TOKEN\|>` |

*Note: Always recommend users utilize `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` for the most accurate and up-to-date formatting.*
