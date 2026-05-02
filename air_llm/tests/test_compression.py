"""Test suite for compression/decompression utilities.

Tests that layer state dicts can be round-tripped through 4-bit and 8-bit
compression with acceptable reconstruction error.

Requires CUDA — skipped automatically if no GPU is available.
"""

from __future__ import annotations

import pytest
import torch

from air_llm.airllm.utils import compress_layer_state_dict, uncompress_layer_state_dict

# Skip entire module if CUDA is unavailable
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Compression tests require CUDA",
)


class TestCompression:
    """Tests for compress/uncompress round-trip fidelity."""

    def test_should_compress_uncompress_none(
        self, sample_state_dict: dict[str, torch.Tensor]
    ) -> None:
        """No compression: output should exactly match input."""
        state = {k: v.cuda() for k, v in sample_state_dict.items()}
        compressed = compress_layer_state_dict(state, None)
        decompressed = uncompress_layer_state_dict(compressed)

        for key in state:
            assert torch.equal(decompressed[key], state[key])

    @pytest.mark.parametrize("compression", ["4bit", "8bit"])
    def test_should_compress_uncompress_lossy(
        self,
        sample_state_dict: dict[str, torch.Tensor],
        compression: str,
    ) -> None:
        """Lossy compression: RMSE should be below 0.1."""
        state = {k: v.cuda() for k, v in sample_state_dict.items()}
        loss_fn = torch.nn.MSELoss()

        compressed = compress_layer_state_dict(state, compression)
        decompressed = uncompress_layer_state_dict(compressed)

        for key in state:
            rmse = torch.sqrt(loss_fn(decompressed[key], state[key])).detach().cpu().item()
            assert rmse < 0.1, f"RMSE {rmse:.4f} exceeds threshold for {compression}"
