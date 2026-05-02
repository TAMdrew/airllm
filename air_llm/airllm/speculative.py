"""Self-Speculative Decoding for AirLLM — LayerSkip-inspired early-exit drafting.

Generates draft tokens using only the first N layers (kept in VRAM),
then verifies them by running the full model. This reduces the number
of full layer-loading cycles from disk, providing 2-5x speedup.

Based on LayerSkip (Meta, 2024) and ConfLayers (2025) research.
Optimized for layer-by-layer inference where disk I/O is the bottleneck.

Reference:
    - LayerSkip: https://arxiv.org/abs/2404.16710
    - Self-Speculative Decoding: https://arxiv.org/abs/2311.08263
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for self-speculative decoding.

    Attributes:
        exit_layer_ratio: Fraction of layers to use for drafting (0.0-1.0).
            Default 0.33 means use the first 1/3 of layers.
        num_speculations: Number of draft tokens per verification cycle.
        temperature: Sampling temperature for draft tokens (0 = greedy).
        acceptance_threshold: Minimum probability ratio to accept a draft token.
    """

    exit_layer_ratio: float = 0.33
    num_speculations: int = 4
    temperature: float = 0.0
    acceptance_threshold: float = 0.0

    def get_exit_layer(self, total_layers: int) -> int:
        """Calculate the exit layer index.

        Args:
            total_layers: Total number of transformer layers.

        Returns:
            Layer index to exit at for drafting.
        """
        exit_layer = max(1, int(total_layers * self.exit_layer_ratio))
        logger.debug(
            "Speculative exit layer: %d / %d (%.0f%%)",
            exit_layer,
            total_layers,
            self.exit_layer_ratio * 100,
        )
        return exit_layer


def verify_draft_tokens(
    draft_tokens: list[int],
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    acceptance_threshold: float = 0.0,
) -> tuple[list[int], int]:
    """Verify draft tokens against full-model target logits.

    Uses greedy acceptance: accept draft token if it matches the
    argmax of the target logits at that position.

    Args:
        draft_tokens: List of drafted token IDs.
        draft_logits: Logits from the draft (early-exit) model.
            Shape: (num_draft_tokens, vocab_size). Currently used for
            future stochastic acceptance extensions.
        target_logits: Logits from the full model verification.
            Shape: (num_draft_tokens, vocab_size).
        acceptance_threshold: Minimum probability ratio for acceptance.
            Currently unused (reserved for stochastic acceptance).

    Returns:
        Tuple of (accepted_tokens, num_accepted).
        accepted_tokens includes the first rejected token's correction.
    """
    # Suppress unused-argument lint — draft_logits and acceptance_threshold
    # are part of the public API for future stochastic acceptance extensions.
    _ = draft_logits
    _ = acceptance_threshold

    accepted: list[int] = []

    for i, draft_token in enumerate(draft_tokens):
        target_token = torch.argmax(target_logits[i]).item()

        if draft_token == target_token:
            accepted.append(draft_token)
        else:
            # Reject: use the target model's prediction instead
            accepted.append(target_token)
            break

    # Count how many draft tokens were truly accepted (excluding correction)
    num_accepted = len(accepted) - (
        1 if accepted and accepted[-1] != draft_tokens[len(accepted) - 1] else 0
    )

    logger.debug(
        "Speculative decoding: %d/%d draft tokens accepted",
        num_accepted,
        len(draft_tokens),
    )
    return accepted, num_accepted


def estimate_speedup(
    num_speculations: int,
    acceptance_rate: float,
    layer_load_time_ms: float,
    compute_time_ms: float,
) -> float:
    """Estimate the speedup from speculative decoding.

    Args:
        num_speculations: Number of draft tokens per cycle.
        acceptance_rate: Average fraction of accepted draft tokens (0-1).
        layer_load_time_ms: Time to load all layers from disk (ms).
        compute_time_ms: Time for a single forward pass (ms).

    Returns:
        Estimated speedup factor (e.g., 3.0 means 3x faster).
    """
    # Without speculative: each token requires full layer loading
    baseline_per_token = layer_load_time_ms + compute_time_ms

    # With speculative: draft tokens are cheap (early exit)
    # Only verification requires full layer loading
    avg_accepted = num_speculations * acceptance_rate
    # Early exit is ~1/3 cost since we only run 1/3 of layers
    draft_cost = avg_accepted * compute_time_ms * 0.33
    verify_cost = layer_load_time_ms + compute_time_ms  # Full verification

    tokens_per_cycle = max(1, avg_accepted + 1)  # +1 for correction token
    speculative_per_token = (draft_cost + verify_cost) / tokens_per_cycle

    speedup = baseline_per_token / speculative_per_token if speculative_per_token > 0 else 1.0
    return round(speedup, 2)
