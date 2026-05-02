"""Shared pytest fixtures for airllm test suite.

Provides:
- mock_config: Minimal HuggingFace-style config object
- temp_model_dir: Temporary directory for test artifacts
- registry_cleanup: Auto-cleanup of ModelRegistry state between tests
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def mock_config() -> MagicMock:
    """Return a minimal HuggingFace-style config object for testing.

    Returns:
        MagicMock configured with common config attributes.
    """
    config = MagicMock()
    config.architectures = ["LlamaForCausalLM"]
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.vocab_size = 1000
    config.max_position_embeddings = 512
    config.quantization_config = None
    config.attn_implementation = None
    return config


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts.

    Yields:
        Path to temporary directory (auto-cleaned after test).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def registry_cleanup() -> Generator[None, None, None]:
    """Save and restore ModelRegistry state around each test.

    Preserves pre-existing registrations (from module-level @register
    decorators) while isolating test-specific mutations.
    """
    from air_llm.airllm.model_registry import ModelRegistry

    # Save current registry state before test
    saved = ModelRegistry._registry.copy() if hasattr(ModelRegistry, "_registry") else {}

    yield

    # Restore registry to pre-test state (undo any test-specific changes)
    if hasattr(ModelRegistry, "_registry"):
        ModelRegistry._registry.clear()
        ModelRegistry._registry.update(saved)


@pytest.fixture
def sample_state_dict() -> dict[str, torch.Tensor]:
    """Return a small state dict for compression testing.

    Returns:
        Dict with two small random tensors.
    """
    return {
        "weight_0": torch.randn(32, 64, dtype=torch.float16),
        "weight_1": torch.randn(64, 32, dtype=torch.float16),
    }


@pytest.fixture
def mock_layer_config() -> dict[str, str]:
    """Return a standard layer names configuration dict.

    Returns:
        Dict mapping logical layer roles to weight name prefixes.
    """
    return {
        "embed": "model.embed_tokens",
        "layer_prefix": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head",
    }
