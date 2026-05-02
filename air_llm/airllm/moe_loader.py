"""MoE Expert-by-Expert Loader — Selective expert loading for limited VRAM.

For Mixture-of-Experts (MoE) models like DeepSeek-V3 (256 experts) and
Mistral Small 4 (128 experts), loading all experts simultaneously exceeds
consumer GPU VRAM.

This module implements the Router-First Evaluation pattern:
1. Keep router weights on GPU (tiny: ~1MB per layer)
2. Evaluate router to determine which experts are needed
3. Load ONLY the activated experts (top-K) from CPU/disk
4. Compute expert outputs
5. Free expert VRAM immediately

Based on PowerInfer (2024) and MoE-Infinity (2025) research.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F  # noqa: N812

from .utils import clean_memory

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class MoEConfig:
    """Configuration for MoE expert loading.

    Attributes:
        num_experts: Total number of experts in the MoE layer.
        top_k: Number of experts activated per token.
        expert_cache_size: Max experts to keep cached on GPU.
    """

    num_experts: int = 8
    top_k: int = 2
    expert_cache_size: int = 4


class ExpertRouter:
    """Routes tokens to experts and manages selective loading.

    Evaluates the gating function to determine which experts
    are needed, then loads only those experts on-demand.

    Args:
        config: MoE configuration.
        device: GPU device for router and active experts.
    """

    def __init__(
        self,
        config: MoEConfig,
        device: str = "cuda:0",
    ) -> None:
        self.config = config
        self.device = device

        # LRU cache for recently used experts: expert_id -> weights dict
        self._expert_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._cache_order: list[int] = []

        logger.debug(
            "ExpertRouter: %d experts, top-%d, cache=%d",
            config.num_experts,
            config.top_k,
            config.expert_cache_size,
        )

    def route(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights and select top-K experts.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).
            router_weights: Router weight matrix (num_experts, hidden_dim).

        Returns:
            Tuple of (routing_weights, selected_expert_ids).
            routing_weights: (batch, seq_len, top_k) — softmax weights.
            selected_expert_ids: (batch, seq_len, top_k) — expert indices.
        """
        # Compute router logits
        router_logits = F.linear(hidden_states, router_weights)

        # Select top-K experts
        routing_weights, selected = torch.topk(router_logits, self.config.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)

        return routing_weights, selected

    def load_expert(
        self,
        expert_id: int,
        expert_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Load a single expert's weights to GPU.

        Uses an LRU cache to avoid reloading frequently-used experts.

        Args:
            expert_id: Index of the expert to load.
            expert_weights: CPU-resident weight dict for this expert.

        Returns:
            GPU-resident weight dict.
        """
        # Check cache
        if expert_id in self._expert_cache:
            # Move to end of LRU
            self._cache_order.remove(expert_id)
            self._cache_order.append(expert_id)
            return self._expert_cache[expert_id]

        # Evict if cache is full
        while len(self._expert_cache) >= self.config.expert_cache_size:
            evict_id = self._cache_order.pop(0)
            del self._expert_cache[evict_id]
            clean_memory()
            logger.debug("Evicted expert %d from cache", evict_id)

        # Load to GPU
        gpu_weights = {
            name: tensor.to(self.device, non_blocking=True)
            for name, tensor in expert_weights.items()
        }

        self._expert_cache[expert_id] = gpu_weights
        self._cache_order.append(expert_id)
        logger.debug(
            "Loaded expert %d to GPU (cache: %d/%d)",
            expert_id,
            len(self._expert_cache),
            self.config.expert_cache_size,
        )

        return gpu_weights

    def compute_moe_output(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        all_expert_weights_cpu: list[dict[str, torch.Tensor]],
        expert_forward_fn: Callable[[torch.Tensor, dict[str, torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor:
        """Compute the MoE layer output with selective expert loading.

        Args:
            hidden_states: (batch, seq_len, hidden_dim).
            routing_weights: (batch, seq_len, top_k) — softmax weights.
            selected_experts: (batch, seq_len, top_k) — expert indices.
            all_expert_weights_cpu: List of CPU-resident expert weight dicts.
            expert_forward_fn: Function(hidden_states, expert_weights) -> output.

        Returns:
            Combined expert output tensor.
        """
        output = torch.zeros_like(hidden_states)

        # Get unique expert IDs needed for this batch
        unique_experts = selected_experts.unique().tolist()

        for expert_id in unique_experts:
            expert_id = int(expert_id)

            # Load expert to GPU (cached if recently used)
            gpu_weights = self.load_expert(expert_id, all_expert_weights_cpu[expert_id])

            # Create mask for tokens routed to this expert
            mask = selected_experts == expert_id

            # Find which top-k slot this expert occupies
            for k in range(self.config.top_k):
                slot_mask = mask[:, :, k]  # (batch, seq_len)
                if not slot_mask.any():
                    continue

                # Compute expert output for all tokens
                expert_output = expert_forward_fn(hidden_states, gpu_weights)

                # Weight by routing probability
                weight = routing_weights[:, :, k].unsqueeze(-1)  # (batch, seq_len, 1)
                slot_contribution = expert_output * weight * slot_mask.unsqueeze(-1).float()
                output = output + slot_contribution

        return output

    def clear_cache(self) -> None:
        """Clear the expert cache and free GPU memory."""
        self._expert_cache.clear()
        self._cache_order.clear()
        clean_memory()
        logger.debug("Expert cache cleared")
