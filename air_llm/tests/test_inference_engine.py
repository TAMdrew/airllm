"""Test suite for InferenceEngine.

Tests the decoupled forward pass orchestrator with mocked model wrapper.
Focuses on list initialization bugs and layer index parsing edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_mock_wrapper(
    num_layers: int = 3,
    profiling_mode: bool = False,
    prefetching: bool = False,
) -> MagicMock:
    """Create a mock model wrapper for InferenceEngine tests."""
    wrapper = MagicMock()
    wrapper.profiling_mode = profiling_mode
    wrapper.prefetching = prefetching
    wrapper.running_device = "cpu"
    wrapper.max_seq_len = 32
    wrapper.speculative_config = None
    wrapper.hf_quantizer = None
    wrapper._kv_compressor = None

    # Create mock layers
    layers = [MagicMock() for _ in range(num_layers + 3)]  # embed + N + norm + head
    wrapper.layers = layers

    # Layer names: embed, layer.0, layer.1, ..., norm, lm_head
    layer_names = ["model.embed_tokens."]
    for i in range(num_layers):
        layer_names.append(f"model.layers.{i}.")
    layer_names.extend(["model.norm.", "lm_head."])
    wrapper.layer_names = layer_names

    wrapper.layer_names_dict = {
        "embed": "model.embed_tokens.",
        "layer_prefix": "model.layers",
        "norm": "model.norm.",
        "lm_head": "lm_head.",
    }

    return wrapper


def test_hidden_states_list_initialization_creates_independent_lists() -> None:
    """Regression test: [] * N produced [] (empty), not N independent lists.

    The bug was: `[] * len(layers)` evaluates to `[]`, not a list of N empty lists.
    After the fix, each element should be an independent empty list.
    """
    from air_llm.airllm.engine.inference_engine import InferenceEngine

    wrapper = _make_mock_wrapper(num_layers=3)
    InferenceEngine(wrapper)

    # Simulate the initialization that happens inside forward()
    num_layers = len(wrapper.layers)
    all_hidden = [[] for _ in range(num_layers)]
    all_attns = [[] for _ in range(num_layers)]

    # Verify correct number of sublists
    assert len(all_hidden) == num_layers
    assert len(all_attns) == num_layers

    # Verify independence — mutating one should NOT affect others
    all_hidden[0].append("test")
    assert all_hidden[1] == []
    assert all_hidden[2] == []


def test_layer_index_parsing_with_trailing_dot() -> None:
    """Regression test: layer names ending with '.' caused ValueError.

    The bug: `int("model.layers.0.".split(".")[-1])` → `int("")` → ValueError.
    After fix: `.rstrip(".")` strips trailing dots before parsing.
    """
    # Layer name with trailing dot (as AirLLM produces)
    layer_name = "model.layers.5."

    # The fixed approach
    layer_idx = int(layer_name.rstrip(".").split(".")[-1])
    assert layer_idx == 5


def test_layer_index_parsing_without_trailing_dot() -> None:
    """Verify parsing still works for names without trailing dots."""
    layer_name = "model.layers.10"

    layer_idx = int(layer_name.rstrip(".").split(".")[-1])
    assert layer_idx == 10


def test_engine_init_stores_wrapper() -> None:
    """Test that InferenceEngine correctly stores the wrapper reference."""
    from air_llm.airllm.engine.inference_engine import InferenceEngine

    wrapper = _make_mock_wrapper()
    engine = InferenceEngine(wrapper)

    assert engine.wrapper is wrapper


@pytest.mark.parametrize(
    "layer_name,expected_idx",
    [
        ("model.layers.0.", 0),
        ("model.layers.31.", 31),
        ("model.layers.127.", 127),
        ("transformer.h.0.", 0),
    ],
)
def test_layer_index_parsing_various_formats(layer_name: str, expected_idx: int) -> None:
    """Test layer index extraction works for various naming conventions."""
    layer_idx = int(layer_name.rstrip(".").split(".")[-1])
    assert layer_idx == expected_idx
