"""Test Chinese dataset token length distributions.

This is a research/training utility script, separate from the core AirLLM
inference library. It analyzes token length distributions for Chinese training
datasets to inform max sequence length hyperparameter choices.

NOTE: This script requires the ``datasets`` and ``transformers`` packages
(both run locally — no cloud APIs).

NOTE: Filename typo "lenghts" (should be "lengths") is preserved to avoid
breaking git history and any external references.
"""

from __future__ import annotations

import logging
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL_ID: str = "timdettmers/guanaco-33b-merged"
DATASET_ID: str = "Chinese-Vicuna/guanaco_belle_merge_v1.0"

SOURCE_TEMPLATE: str = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
)

QUANTILES: list[float] = [0.8, 0.85, 0.9, 0.95, 0.98]


def main() -> None:
    """Compute and log token length quantiles for source and target columns."""
    logger.info("Loading tokenizer: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    logger.info("Loading dataset: %s", DATASET_ID)
    ds = load_dataset(DATASET_ID)

    ds = ds.map(
        lambda x: {
            "source_length": len(tokenizer.encode(SOURCE_TEMPLATE.format(**x))),
            "target_length": len(tokenizer.encode(x["output"])),
        }
    )

    df = ds["train"].to_pandas()

    for qt in QUANTILES:
        logger.info("source len @qt%.2f: %.1f", qt, df["source_length"].quantile(qt))
        logger.info("target len @qt%.2f: %.1f", qt, df["target_length"].quantile(qt))


if __name__ == "__main__":
    main()
