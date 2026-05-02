"""MLX-based model persister for macOS (Apple Silicon) platforms.

Persists each layer as a ``.mlx.npz`` file alongside a ``.done``
marker file.  Weights are automatically mapped from the HuggingFace
naming convention to the MLX naming convention on load.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_unflatten

from .model_persister import ModelPersister

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HuggingFace -> MLX weight-name mapping
# ---------------------------------------------------------------------------
_WEIGHT_NAME_REPLACEMENTS: list[tuple[str, str]] = [
    ("model.", ""),
    ("mlp", "feed_forward"),
    ("down_proj", "w2"),
    ("up_proj", "w3"),
    ("gate_proj", "w1"),
    ("input_layernorm", "attention_norm"),
    ("post_attention_layernorm", "ffn_norm"),
    ("lm_head", "output"),
    ("embed_tokens", "tok_embeddings"),
    ("self_attn", "attention"),
    ("q_proj", "wq"),
    ("k_proj", "wk"),
    ("v_proj", "wv"),
    ("o_proj", "wo"),
]


def map_torch_to_mlx(model: dict[str, Any]) -> dict[str, Any]:
    """Rename weight keys from HuggingFace convention to MLX convention.

    Args:
        model: State dict with HuggingFace-style key names.

    Returns:
        A new dict with MLX-style key names.
    """
    for old, new in _WEIGHT_NAME_REPLACEMENTS:
        model = {k.replace(old, new): v for k, v in model.items()}
    return model


# ---------------------------------------------------------------------------
# Persister
# ---------------------------------------------------------------------------
class MlxModelPersister(ModelPersister):
    """Persister implementation using NumPy ``.npz`` format for MLX."""

    def model_persist_exist(self, layer_name: str, saving_path: str | Path) -> bool:
        """Check if the ``.mlx.npz`` file and done marker exist.

        Args:
            layer_name: Logical layer name.
            saving_path: Directory containing layer files.

        Returns:
            ``True`` if both the ``.mlx.npz`` file and its
            ``.mlx.done`` marker exist.
        """
        saving_path = Path(saving_path)
        npz_exists = os.path.exists(saving_path / (layer_name + "mlx.npz"))
        done_marker_exists = os.path.exists(saving_path / (layer_name + "mlx.done"))
        return npz_exists and done_marker_exists

    def persist_model(
        self,
        state_dict: dict[str, Any],
        layer_name: str,
        saving_path: str | Path,
    ) -> None:
        """Save a layer state dict as a ``.mlx.npz`` file.

        Weights are converted to ``float16`` numpy arrays before saving.

        Args:
            state_dict: The tensor dict to save.
            layer_name: Logical layer name.
            saving_path: Target directory.
        """
        saving_path = Path(saving_path)
        weights = {k: v.to(torch.float16).numpy() for k, v in state_dict.items()}
        output_file = saving_path / (layer_name + "mlx")
        np.savez(output_file, **weights)
        logger.info("Saved layer: %s", output_file)

        # Write done marker
        (saving_path / (layer_name + "mlx.done")).touch()

    def load_model(self, layer_name: str, path: str | Path) -> dict[str, Any]:
        """Load a layer state dict from a ``.mlx.npz`` file.

        The loaded weights are renamed from HuggingFace convention to MLX
        convention and unflattened into a nested dict structure.

        Args:
            layer_name: Logical layer name.
            path: Directory containing layer files.

        Returns:
            The loaded and restructured weight dict.

        Raises:
            OSError: If the file cannot be read.
        """
        # BUG-FIX: layer names end with "." so we must NOT add another dot.
        # The save path uses `layer_name + "mlx"` (no leading dot).
        to_load_path = Path(path) / (layer_name + "mlx.npz")
        try:
            layer_state_dict = mx.load(str(to_load_path))
            layer_state_dict = map_torch_to_mlx(layer_state_dict)
            weights: dict[str, Any] = tree_unflatten(list(layer_state_dict.items()))
            return weights
        except Exception:
            logger.exception("Failed to load layer %r from %s", layer_name, path)
            raise
