"""Abstract base class for model persistence backends.

``ModelPersister`` defines the interface for loading and saving per-layer
model weights.  Concrete implementations handle format-specific logic
(safetensors on Linux/CUDA, MLX on macOS).

The singleton persister instance is lazily created by
:pymeth:`ModelPersister.get_model_persister` based on the current platform.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_model_persister: ModelPersister | None = None
_persister_lock = threading.Lock()


class ModelPersister(ABC):
    """Abstract persister interface for layer-level model I/O.

    Subclasses must implement :pymeth:`model_persist_exist`,
    :pymeth:`persist_model`, and :pymeth:`load_model`.
    """

    @classmethod
    def get_model_persister(cls) -> ModelPersister:
        """Return the singleton persister for the current platform.

        On macOS (Darwin) this returns an ``MlxModelPersister``.
        On all other platforms it returns a ``SafetensorModelPersister``.

        Returns:
            The platform-appropriate ``ModelPersister`` singleton.
        """
        global _model_persister
        if _model_persister is not None:
            return _model_persister

        with _persister_lock:
            if _model_persister is not None:
                return _model_persister

            from ..constants import IS_ON_MAC_OS

            if IS_ON_MAC_OS:
                from .mlx_model_persister import MlxModelPersister

                _model_persister = MlxModelPersister()
                logger.debug("Using MlxModelPersister (macOS detected)")
            else:
                from .safetensor_model_persister import SafetensorModelPersister

                _model_persister = SafetensorModelPersister()
                logger.debug("Using SafetensorModelPersister")

            return _model_persister

    @abstractmethod
    def model_persist_exist(self, layer_name: str, saving_path: str | Path) -> bool:
        """Check whether a persisted layer file exists.

        Args:
            layer_name: The logical layer name.
            saving_path: Directory containing persisted layer files.

        Returns:
            ``True`` if the layer file (and its done marker) exist.
        """
        pass

    @abstractmethod
    def persist_model(
        self,
        state_dict: dict[str, Any],
        layer_name: str,
        path: str | Path,
    ) -> None:
        """Persist a layer's state dict to disk.

        Args:
            state_dict: The tensor dict to save.
            layer_name: Logical layer name (used to derive filename).
            path: Target directory.
        """
        pass

    @abstractmethod
    def load_model(self, layer_name: str, path: str | Path) -> dict[str, Any]:
        """Load a layer's state dict from disk.

        Args:
            layer_name: Logical layer name.
            path: Directory containing persisted layer files.

        Returns:
            The loaded state dict.
        """
        pass
