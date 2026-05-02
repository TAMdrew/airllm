# Scripts — Research Utilities

> **⚠️ Research Utility**
> This directory contains helper scripts for training data analysis,
> **separate from** the core AirLLM inference library.
> These scripts run locally and do not call any cloud APIs.

## Contents

| File                         | Description                                                                                                                                                                                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_cn_dataset_lenghts.py` | Analyzes token length distributions for Chinese training datasets (e.g., `guanaco_belle_merge_v1.0`). Used to determine optimal `source_max_len` and `target_max_len` hyperparameters for QLoRA training. |

## Usage

```bash
# Requires: pip install datasets transformers
python scripts/test_cn_dataset_lenghts.py
```

The script outputs quantile statistics (0.8, 0.85, 0.9, 0.95, 0.98) for
source and target token lengths, helping to choose sequence length
hyperparameters that cover most training samples without excessive truncation.

## Relationship to AirLLM

These are standalone research scripts.
For inference with AirLLM, see the [core library documentation](../docs/QUICKSTART.md).
