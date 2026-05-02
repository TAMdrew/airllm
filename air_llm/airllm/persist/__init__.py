"""Model persistence backends for AirLLM.

Provides platform-specific serialisation for per-layer model weights:
- ``SafetensorModelPersister`` for CUDA / Linux (safetensors format).
- ``MlxModelPersister`` for macOS / Apple Silicon (NumPy ``.npz`` format).

Use :pymeth:`ModelPersister.get_model_persister` to obtain the singleton
persister for the current platform.
"""

from .model_persister import ModelPersister

__all__ = ["ModelPersister"]
