# AirLLM Examples

This directory contains example notebooks and scripts demonstrating AirLLM usage.

> **Local-first:** All examples run entirely on your local hardware.
> HuggingFace is used only to download model weights on first run — no API keys,
> no subscriptions, no cloud dependencies.

## Example Notebooks

### Root-Level Examples

| Notebook                                        | Description                                                                                                                  | Status                                                                                        |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [`examples/inferrence.ipynb`](inferrence.ipynb) | Original Anima 33B inference notebook using PEFT adapters. Demonstrates loading a base Guanaco 33B model with LoRA adapters. | ⚠️ **Needs update for v3.0** — uses legacy `LlamaForCausalLM` imports instead of `AutoModel`. |

### Core AirLLM Examples (`air_llm/examples/`)

| Notebook                                                                             | Description                                                                                                                                                         | Status                                                                                                                                        |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| [`run_all_types_of_models.ipynb`](../air_llm/examples/run_all_types_of_models.ipynb) | Demonstrates running multiple model architectures (Llama, ChatGLM, QWen, Baichuan, Mixtral, InternLM) with AirLLM. Shows both HuggingFace Hub and local path usage. | ⚠️ **Needs update for v3.0** — should use `AutoModel` instead of architecture-specific classes, and add Gemma 4 / Qwen 3 / DeepSeek examples. |
| [`run_llama3.1_405B.ipynb`](../air_llm/examples/run_llama3.1_405B.ipynb)             | Shows how to run the massive Llama 3.1 405B model on a single GPU using AirLLM with 4-bit compression.                                                              | ✅ Uses `AutoModel` API. May need token/auth updates.                                                                                         |
| [`run_on_macos.ipynb`](../air_llm/examples/run_on_macos.ipynb)                       | Demonstrates running AirLLM on macOS with Apple Silicon (M1/M2/M3) via the MLX backend.                                                                             | ⚠️ **Needs update for v3.0** — should use `AutoModel` and note MLX auto-detection.                                                            |

### Evaluation Examples (`eval/`)

| Notebook                                                                                                                 | Description                                                                                                                                                     | Status                                                                               |
| ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| [`elo_tournanment_all_models_on_translated_vicuna.ipynb`](../eval/elo_tournanment_all_models_on_translated_vicuna.ipynb) | Elo rating tournament evaluating multiple Chinese LLMs (Anima 33B, Belle, Chinese Vicuna, ChatGPT-3.5) on the translated Vicuna benchmark using GPT-4 as judge. | 📚 **Research artifact** — historical evaluation, not part of core AirLLM inference. |

### Data Processing (`data/`)

| Notebook                                                                               | Description                                                                 | Status                                                                                          |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [`gpt4_translate_vicuna_eval_set.ipynb`](../data/gpt4_translate_vicuna_eval_set.ipynb) | Uses GPT-4 API to translate the English Vicuna evaluation set into Chinese. | 📚 **Research artifact** — requires OpenAI API key. Not related to core AirLLM local inference. |

## Example Scripts

| Script                                                            | Description                                                                                                                                                      |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`air_llm/inference_example.py`](../air_llm/inference_example.py) | **Recommended starting point.** Demonstrates AirLLM v3.0 local inference with `AutoModel`, logging, type hints, and examples for Llama 3.1, Gemma 4, and Qwen 3. |

## Quick Start

The fastest way to try AirLLM:

```bash
pip install airllm
```

```python
from airllm import AutoModel

# Downloads weights once, then runs entirely locally
model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

input_tokens = model.tokenizer(
    ["What is the capital of France?"],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

## Notes

- **No cloud APIs required** — AirLLM runs completely locally.
- **HuggingFace tokens** are only needed for gated models (Llama 3, Gemma) to download weights. After download, no internet is needed.
- **VRAM:** Most models run on 4 GB VRAM. Use `compression="4bit"` for larger models.
