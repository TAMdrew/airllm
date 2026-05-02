"""AirLLM inference example — basic usage with AutoModel.

Demonstrates the standard workflow:
1. Load a model via ``AutoModel.from_pretrained()``
2. Tokenize input text
3. Generate output tokens
4. Decode and print the result

Requires: ``pip install airllm``
"""

from __future__ import annotations

from air_llm.airllm import AutoModel


def main() -> None:
    """Run a basic inference example."""
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    model = AutoModel.from_pretrained(model_id)

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


if __name__ == "__main__":
    main()
