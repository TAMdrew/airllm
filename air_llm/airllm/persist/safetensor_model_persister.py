"""Safetensors-based model persister for CUDA / Linux platforms.

Persists each layer as a ``.safetensors`` file alongside a ``.done``
marker file to track completion.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from .model_persister import ModelPersister

logger = logging.getLogger(__name__)


class SafetensorModelPersister(ModelPersister):
    """Persister implementation using the ``safetensors`` format."""

    def model_persist_exist(self, layer_name: str, saving_path: str | Path) -> bool:
        """Check if the safetensors file and done marker exist.

        Args:
            layer_name: Logical layer name.
            saving_path: Directory containing layer files.

        Returns:
            ``True`` if both the ``.safetensors`` file and its
            ``.safetensors.done`` marker exist.
        """
        saving_path = Path(saving_path)
        safetensor_exists = os.path.exists(saving_path / (layer_name + "safetensors"))
        done_marker_exists = os.path.exists(saving_path / (layer_name + "safetensors.done"))
        return safetensor_exists and done_marker_exists

    def persist_model(
        self,
        state_dict: dict[str, Any],
        layer_name: str,
        saving_path: str | Path,
    ) -> None:
        """Save a layer state dict as a safetensors file.

        A ``.done`` marker file is created after a successful save to
        allow crash-safe resumption.

        Args:
            state_dict: The tensor dict to save.
            layer_name: Logical layer name (used to derive filename).
            saving_path: Target directory.
        """
        saving_path = Path(saving_path)
        output_file = saving_path / (layer_name + "safetensors")
        save_file(state_dict, output_file)
        logger.info("Saved layer: %s", output_file)

        # Write done marker
        (saving_path / (layer_name + "safetensors.done")).touch()

    def load_model(self, layer_name: str, path: str | Path) -> dict[str, Any]:
        """Load a layer state dict from a safetensors file.

        Args:
            layer_name: Logical layer name.
            path: Directory containing layer files.

        Returns:
            The loaded state dict with tensors on CPU.
        """
        # BUG-FIX: layer names end with "." so we must NOT add another dot.
        # The save path uses `layer_name + "safetensors"` (no leading dot).
        return load_file(Path(path) / (layer_name + "safetensors"), device="cpu")
