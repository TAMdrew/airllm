"""Tests for MoE expert-by-expert loader module."""

import torch

from air_llm.airllm.moe_loader import ExpertRouter, MoEConfig


class TestMoEConfig:
    """Tests for the MoEConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config has sensible values."""
        config = MoEConfig()
        assert config.num_experts == 8
        assert config.top_k == 2
        assert config.expert_cache_size == 4

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        config = MoEConfig(num_experts=64, top_k=4, expert_cache_size=8)
        assert config.num_experts == 64
        assert config.top_k == 4
        assert config.expert_cache_size == 8


class TestExpertRouterInit:
    """Tests for ExpertRouter initialization."""

    def test_init(self) -> None:
        """Init creates router with empty cache."""
        config = MoEConfig(num_experts=4, top_k=2, expert_cache_size=2)
        router = ExpertRouter(config, device="cpu")
        assert router.device == "cpu"
        assert len(router._expert_cache) == 0
        assert len(router._cache_order) == 0


class TestExpertRouterRoute:
    """Tests for the routing function."""

    def test_route_returns_correct_shapes(self) -> None:
        """Route returns correct shapes for weights and indices."""
        config = MoEConfig(num_experts=4, top_k=2)
        router = ExpertRouter(config, device="cpu")

        hidden_states = torch.randn(1, 8, 32)  # (batch, seq_len, hidden_dim)
        router_weights = torch.randn(4, 32)  # (num_experts, hidden_dim)

        routing_weights, selected = router.route(hidden_states, router_weights)

        assert routing_weights.shape == (1, 8, 2)  # (batch, seq_len, top_k)
        assert selected.shape == (1, 8, 2)  # (batch, seq_len, top_k)

    def test_route_weights_sum_to_one(self) -> None:
        """Routing weights are softmax-normalized (sum to ~1)."""
        config = MoEConfig(num_experts=4, top_k=2)
        router = ExpertRouter(config, device="cpu")

        hidden_states = torch.randn(1, 4, 16)
        router_weights = torch.randn(4, 16)

        routing_weights, _ = router.route(hidden_states, router_weights)

        # Softmax over top_k dimension should sum to 1
        sums = routing_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_route_selects_top_k_experts(self) -> None:
        """Selected expert indices are valid."""
        config = MoEConfig(num_experts=8, top_k=3)
        router = ExpertRouter(config, device="cpu")

        hidden_states = torch.randn(2, 4, 32)
        router_weights = torch.randn(8, 32)

        _, selected = router.route(hidden_states, router_weights)

        # All selected indices should be in [0, num_experts)
        assert (selected >= 0).all()
        assert (selected < 8).all()
        assert selected.shape[-1] == 3  # top_k=3


class TestExpertRouterCache:
    """Tests for the expert LRU cache."""

    def _make_expert_weights(self, dim: int = 16) -> dict[str, torch.Tensor]:
        """Create dummy expert weights on CPU."""
        return {
            "gate_proj.weight": torch.randn(dim, dim),
            "up_proj.weight": torch.randn(dim, dim),
            "down_proj.weight": torch.randn(dim, dim),
        }

    def test_load_expert_caches(self) -> None:
        """Loading an expert puts it in the cache."""
        config = MoEConfig(num_experts=4, top_k=2, expert_cache_size=4)
        router = ExpertRouter(config, device="cpu")

        weights = self._make_expert_weights()
        gpu_weights = router.load_expert(0, weights)

        assert 0 in router._expert_cache
        assert len(router._cache_order) == 1
        assert "gate_proj.weight" in gpu_weights

    def test_load_expert_cache_hit(self) -> None:
        """Loading a cached expert returns the cached version."""
        config = MoEConfig(num_experts=4, top_k=2, expert_cache_size=4)
        router = ExpertRouter(config, device="cpu")

        weights = self._make_expert_weights()
        result1 = router.load_expert(0, weights)
        result2 = router.load_expert(0, weights)

        # Should be the same object (cache hit)
        assert result1 is result2

    def test_load_expert_eviction(self) -> None:
        """Loading beyond cache_size evicts oldest expert."""
        config = MoEConfig(num_experts=8, top_k=2, expert_cache_size=2)
        router = ExpertRouter(config, device="cpu")

        # Fill cache
        router.load_expert(0, self._make_expert_weights())
        router.load_expert(1, self._make_expert_weights())
        assert len(router._expert_cache) == 2

        # Load one more — should evict expert 0
        router.load_expert(2, self._make_expert_weights())
        assert len(router._expert_cache) == 2
        assert 0 not in router._expert_cache
        assert 1 in router._expert_cache
        assert 2 in router._expert_cache

    def test_load_expert_lru_reorder(self) -> None:
        """Accessing a cached expert moves it to end of LRU."""
        config = MoEConfig(num_experts=8, top_k=2, expert_cache_size=2)
        router = ExpertRouter(config, device="cpu")

        router.load_expert(0, self._make_expert_weights())
        router.load_expert(1, self._make_expert_weights())

        # Access expert 0 — moves to end
        router.load_expert(0, self._make_expert_weights())

        # Now load expert 2 — should evict expert 1 (not 0)
        router.load_expert(2, self._make_expert_weights())

        assert 0 in router._expert_cache
        assert 1 not in router._expert_cache
        assert 2 in router._expert_cache


class TestExpertRouterComputeMoEOutput:
    """Tests for compute_moe_output."""

    def test_compute_moe_output_shape(self) -> None:
        """MoE output has same shape as input hidden_states."""
        config = MoEConfig(num_experts=4, top_k=2, expert_cache_size=4)
        router = ExpertRouter(config, device="cpu")

        hidden_states = torch.randn(1, 4, 16)
        routing_weights = torch.ones(1, 4, 2) * 0.5
        selected_experts = torch.tensor([[[0, 1], [1, 2], [0, 3], [2, 3]]])

        expert_weights_cpu = [{"w": torch.randn(16, 16)} for _ in range(4)]

        def expert_fn(h: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
            return h  # Identity for shape test

        output = router.compute_moe_output(
            hidden_states,
            routing_weights,
            selected_experts,
            expert_weights_cpu,
            expert_fn,
        )

        assert output.shape == hidden_states.shape

    def test_compute_moe_output_loads_only_needed(self) -> None:
        """Only experts referenced in selected_experts are loaded."""
        config = MoEConfig(num_experts=8, top_k=1, expert_cache_size=8)
        router = ExpertRouter(config, device="cpu")

        hidden_states = torch.randn(1, 2, 8)
        routing_weights = torch.ones(1, 2, 1)
        # Only experts 3 and 5 are selected
        selected_experts = torch.tensor([[[3], [5]]])

        expert_weights_cpu = [{"w": torch.randn(8, 8)} for _ in range(8)]

        def expert_fn(h: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
            return h

        router.compute_moe_output(
            hidden_states,
            routing_weights,
            selected_experts,
            expert_weights_cpu,
            expert_fn,
        )

        # Only experts 3 and 5 should be cached
        assert 3 in router._expert_cache
        assert 5 in router._expert_cache
        assert len(router._expert_cache) == 2


class TestExpertRouterClearCache:
    """Tests for cache clearing."""

    def test_clear_cache_empties_all(self) -> None:
        """clear_cache removes all cached experts."""
        config = MoEConfig(num_experts=4, top_k=2, expert_cache_size=4)
        router = ExpertRouter(config, device="cpu")

        router.load_expert(0, {"w": torch.randn(4, 4)})
        router.load_expert(1, {"w": torch.randn(4, 4)})

        assert len(router._expert_cache) == 2

        router.clear_cache()

        assert len(router._expert_cache) == 0
        assert len(router._cache_order) == 0
