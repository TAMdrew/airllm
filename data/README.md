# Data — Evaluation Datasets

> **⚠️ Research Utility**
> This directory contains evaluation datasets and data-processing notebooks,
> **separate from** the core AirLLM inference library.
> No cloud APIs are required to use these datasets.

## Contents

| File                                   | Description                                                                                                                 |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `translated_vicuna_eval_set.json`      | Chinese-translated Vicuna evaluation dataset (GPT-4 translated). Used for Elo rating tournament evaluation of Chinese LLMs. |
| `gpt4_translate_vicuna_eval_set.ipynb` | Jupyter notebook that performs the GPT-4 translation of the Vicuna evaluation set from English to Chinese.                  |

## Usage

These datasets are used by the evaluation notebooks in [`eval/`](../eval/) for
model comparison via Elo rating tournaments.

## Relationship to AirLLM

These are standalone research artifacts.
For inference with AirLLM, see the [core library documentation](../docs/QUICKSTART.md).
