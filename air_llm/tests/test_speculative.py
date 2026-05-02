"""Tests for self-speculative decoding module."""

import torch

from air_llm.airllm.speculative import (
    SpeculativeConfig,
    estimate_speedup,
    verify_draft_tokens,
)


class TestSpeculativeConfig:
    """Tests for the SpeculativeConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config has sensible values."""
        config = SpeculativeConfig()
        assert config.exit_layer_ratio == 0.33
        assert config.num_speculations == 4
        assert config.temperature == 0.0
        assert config.acceptance_threshold == 0.0

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        config = SpeculativeConfig(
            exit_layer_ratio=0.5,
            num_speculations=8,
            temperature=0.7,
            acceptance_threshold=0.1,
        )
        assert config.exit_layer_ratio == 0.5
        assert config.num_speculations == 8
        assert config.temperature == 0.7
        assert config.acceptance_threshold == 0.1

    def test_get_exit_layer_32_layers(self) -> None:
        """Exit layer is 1/3 of 32 = 10."""
        config = SpeculativeConfig(exit_layer_ratio=0.33)
        assert config.get_exit_layer(32) == 10

    def test_get_exit_layer_80_layers(self) -> None:
        """Exit layer is 1/3 of 80 = 26."""
        config = SpeculativeConfig(exit_layer_ratio=0.33)
        assert config.get_exit_layer(80) == 26

    def test_get_exit_layer_minimum_is_one(self) -> None:
        """Exit layer is at least 1, even with tiny ratio."""
        config = SpeculativeConfig(exit_layer_ratio=0.001)
        assert config.get_exit_layer(10) == 1

    def test_get_exit_layer_half(self) -> None:
        """Exit layer at 50% ratio."""
        config = SpeculativeConfig(exit_layer_ratio=0.5)
        assert config.get_exit_layer(40) == 20


class TestVerifyDraftTokens:
    """Tests for the verify_draft_tokens function."""

    def test_all_accepted_greedy(self) -> None:
        """All draft tokens accepted when they match target argmax."""
        draft_tokens = [10, 20, 30]
        # Target logits where argmax matches draft tokens
        target_logits = torch.zeros(3, 100)
        target_logits[0, 10] = 10.0
        target_logits[1, 20] = 10.0
        target_logits[2, 30] = 10.0
        draft_logits = target_logits.clone()

        accepted, num_accepted = verify_draft_tokens(draft_tokens, draft_logits, target_logits)
        assert accepted == [10, 20, 30]
        assert num_accepted == 3

    def test_first_rejected(self) -> None:
        """First token rejected gives correction from target."""
        draft_tokens = [10, 20, 30]
        target_logits = torch.zeros(3, 100)
        target_logits[0, 99] = 10.0  # Draft says 10, target says 99
        target_logits[1, 20] = 10.0
        target_logits[2, 30] = 10.0
        draft_logits = target_logits.clone()

        accepted, num_accepted = verify_draft_tokens(draft_tokens, draft_logits, target_logits)
        assert accepted == [99]  # Only correction token
        assert num_accepted == 0

    def test_partial_acceptance(self) -> None:
        """First two accepted, third rejected with correction."""
        draft_tokens = [10, 20, 30]
        target_logits = torch.zeros(3, 100)
        target_logits[0, 10] = 10.0  # Match
        target_logits[1, 20] = 10.0  # Match
        target_logits[2, 50] = 10.0  # Mismatch: draft=30, target=50
        draft_logits = target_logits.clone()

        accepted, num_accepted = verify_draft_tokens(draft_tokens, draft_logits, target_logits)
        assert accepted == [10, 20, 50]
        assert num_accepted == 2

    def test_single_token_accepted(self) -> None:
        """Single draft token fully accepted."""
        draft_tokens = [42]
        target_logits = torch.zeros(1, 100)
        target_logits[0, 42] = 5.0
        draft_logits = target_logits.clone()

        accepted, num_accepted = verify_draft_tokens(draft_tokens, draft_logits, target_logits)
        assert accepted == [42]
        assert num_accepted == 1

    def test_single_token_rejected(self) -> None:
        """Single draft token rejected gives correction."""
        draft_tokens = [42]
        target_logits = torch.zeros(1, 100)
        target_logits[0, 7] = 5.0  # Target says 7, not 42
        draft_logits = target_logits.clone()

        accepted, num_accepted = verify_draft_tokens(draft_tokens, draft_logits, target_logits)
        assert accepted == [7]
        assert num_accepted == 0

    def test_empty_draft(self) -> None:
        """Empty draft list returns empty accepted list."""
        draft_tokens: list[int] = []
        target_logits = torch.zeros(0, 100)
        draft_logits = target_logits.clone()

        accepted, num_accepted = verify_draft_tokens(draft_tokens, draft_logits, target_logits)
        assert accepted == []
        assert num_accepted == 0


class TestEstimateSpeedup:
    """Tests for the estimate_speedup function."""

    def test_perfect_acceptance_gives_speedup(self) -> None:
        """100% acceptance rate gives significant speedup."""
        speedup = estimate_speedup(
            num_speculations=4,
            acceptance_rate=1.0,
            layer_load_time_ms=1000.0,
            compute_time_ms=100.0,
        )
        assert speedup > 2.0

    def test_zero_acceptance_no_speedup(self) -> None:
        """0% acceptance rate gives minimal or no speedup."""
        speedup = estimate_speedup(
            num_speculations=4,
            acceptance_rate=0.0,
            layer_load_time_ms=1000.0,
            compute_time_ms=100.0,
        )
        assert speedup >= 0.9  # Should be close to 1.0 (no speedup)

    def test_returns_float(self) -> None:
        """Result is always a float."""
        speedup = estimate_speedup(
            num_speculations=4,
            acceptance_rate=0.75,
            layer_load_time_ms=1000.0,
            compute_time_ms=100.0,
        )
        assert isinstance(speedup, float)

    def test_higher_acceptance_more_speedup(self) -> None:
        """Higher acceptance rates give better speedup."""
        low = estimate_speedup(4, 0.25, 1000.0, 100.0)
        high = estimate_speedup(4, 0.75, 1000.0, 100.0)
        assert high > low

    def test_io_dominated_benefits_more(self) -> None:
        """IO-dominated workloads benefit more from speculative decoding."""
        io_heavy = estimate_speedup(4, 0.75, 5000.0, 100.0)
        compute_heavy = estimate_speedup(4, 0.75, 100.0, 5000.0)
        assert io_heavy > compute_heavy
