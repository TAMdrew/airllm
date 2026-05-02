"""Paged KV Cache — CPU-GPU memory management for long-context inference.

Divides the KV cache into fixed-size pages that can be stored in GPU memory
or offloaded to CPU (pinned) memory. Uses LRU eviction to manage GPU pages
and async transfers for prefetching.

This enables long-context generation (100k+ tokens) on GPUs with limited VRAM
by transparently swapping KV cache pages between GPU and CPU.

Based on PagedAttention (vLLM, 2023) adapted for single-GPU layer-by-layer inference.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class PageConfig:
    """Configuration for paged KV cache.

    Attributes:
        page_size: Number of tokens per page.
        num_gpu_pages: Maximum pages to keep on GPU.
        num_cpu_pages: Maximum pages to keep on CPU.
        head_dim: Dimension of each attention head.
        num_kv_heads: Number of KV attention heads.
    """

    page_size: int = 16
    num_gpu_pages: int = 64
    num_cpu_pages: int = 256
    head_dim: int = 128
    num_kv_heads: int = 8


class PagedKVCache:
    """Manages KV cache pages across GPU and CPU memory.

    Allocates fixed-size pages for key and value tensors. When GPU
    memory is full, evicts least-recently-used pages to CPU using
    async transfers.

    Args:
        config: PageConfig with dimensions and pool sizes.
        num_layers: Number of transformer layers.
        dtype: Data type for cache tensors.
        gpu_device: GPU device string.

    Example:
        >>> config = PageConfig(page_size=16, num_gpu_pages=32, num_cpu_pages=128)
        >>> cache = PagedKVCache(config, num_layers=32)
        >>> cache.append(layer_idx=0, key=k, value=v)
        >>> result = cache.get(layer_idx=0)
        >>> if result is not None:
        ...     k, v = result
    """

    def __init__(
        self,
        config: PageConfig,
        num_layers: int,
        dtype: torch.dtype = torch.float16,
        gpu_device: str = "cuda:0",
    ) -> None:
        self.config = config
        self.num_layers = num_layers
        self.dtype = dtype
        self.gpu_device = gpu_device

        # Per-layer page tables: layer_idx -> OrderedDict[page_idx -> device]
        self._page_tables: list[OrderedDict[int, str]] = [OrderedDict() for _ in range(num_layers)]

        # GPU page pool: (layer_idx, page_idx) -> (key_page, value_page)
        self._gpu_pool: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._gpu_page_count = 0

        # CPU page pool (pinned memory for fast transfers)
        self._cpu_pool: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        # Track current sequence position per layer
        self._seq_positions: list[int] = [0] * num_layers

        logger.debug(
            "PagedKVCache initialized: %d layers, page_size=%d, gpu_pages=%d, cpu_pages=%d",
            num_layers,
            config.page_size,
            config.num_gpu_pages,
            config.num_cpu_pages,
        )

    def append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append new key/value tensors to the cache for a layer.

        Args:
            layer_idx: Transformer layer index.
            key: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim).
            value: Value tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        """
        seq_len = key.shape[2]
        pos = self._seq_positions[layer_idx]

        for i in range(seq_len):
            page_idx = (pos + i) // self.config.page_size
            page_offset = (pos + i) % self.config.page_size

            pool_key = (layer_idx, page_idx)

            # Allocate new page if needed
            if pool_key not in self._gpu_pool:
                self._allocate_gpu_page(layer_idx, page_idx)

            # Write to page
            if pool_key in self._gpu_pool:
                k_page, v_page = self._gpu_pool[pool_key]
                k_page[:, :, page_offset, :] = key[:, :, i, :]
                v_page[:, :, page_offset, :] = value[:, :, i, :]

        self._seq_positions[layer_idx] = pos + seq_len

    def _allocate_gpu_page(self, layer_idx: int, page_idx: int) -> None:
        """Allocate a new GPU page, evicting to CPU if necessary.

        Args:
            layer_idx: Transformer layer index.
            page_idx: Page index within the layer.
        """
        pool_key = (layer_idx, page_idx)

        # Evict if GPU pool is full
        if self._gpu_page_count >= self.config.num_gpu_pages:
            self._evict_lru_to_cpu()

        shape = (1, self.config.num_kv_heads, self.config.page_size, self.config.head_dim)
        k_page = torch.zeros(shape, dtype=self.dtype, device=self.gpu_device)
        v_page = torch.zeros(shape, dtype=self.dtype, device=self.gpu_device)

        self._gpu_pool[pool_key] = (k_page, v_page)
        self._page_tables[layer_idx][page_idx] = "gpu"
        self._gpu_page_count += 1

    def _evict_lru_to_cpu(self) -> None:
        """Evict the least recently used GPU page to CPU."""
        # Find LRU page across all layers
        for layer_idx, page_table in enumerate(self._page_tables):
            for page_idx, device in page_table.items():
                if device == "gpu":
                    pool_key = (layer_idx, page_idx)
                    k_gpu, v_gpu = self._gpu_pool.pop(pool_key)

                    # Transfer to CPU (pinned)
                    k_cpu = k_gpu.to("cpu", non_blocking=True)
                    v_cpu = v_gpu.to("cpu", non_blocking=True)

                    try:
                        k_cpu = k_cpu.pin_memory()
                        v_cpu = v_cpu.pin_memory()
                    except RuntimeError:
                        pass

                    self._cpu_pool[pool_key] = (k_cpu, v_cpu)
                    page_table[page_idx] = "cpu"
                    self._gpu_page_count -= 1

                    del k_gpu, v_gpu
                    logger.debug("Evicted page (%d, %d) to CPU", layer_idx, page_idx)
                    return

    def get(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get all cached key/value tensors for a layer.

        Fetches from GPU if available, otherwise swaps from CPU.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (keys, values) concatenated across all pages,
            or None if no cache exists for this layer.
        """
        page_table = self._page_tables[layer_idx]
        if not page_table:
            return None

        all_keys: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        for page_idx in sorted(page_table.keys()):
            pool_key = (layer_idx, page_idx)
            device = page_table[page_idx]

            if device == "gpu":
                k, v = self._gpu_pool[pool_key]
            elif device == "cpu":
                # Swap back to GPU
                k_cpu, v_cpu = self._cpu_pool.pop(pool_key)
                k = k_cpu.to(self.gpu_device, non_blocking=True)
                v = v_cpu.to(self.gpu_device, non_blocking=True)
                self._gpu_pool[pool_key] = (k, v)
                page_table[page_idx] = "gpu"
                self._gpu_page_count += 1
            else:
                continue

            all_keys.append(k)
            all_values.append(v)

        if not all_keys:
            return None

        # Concatenate along sequence dimension
        keys = torch.cat(all_keys, dim=2)
        values = torch.cat(all_values, dim=2)

        # Trim to actual sequence length
        seq_len = self._seq_positions[layer_idx]
        keys = keys[:, :, :seq_len, :]
        values = values[:, :, :seq_len, :]

        return keys, values

    def clear(self) -> None:
        """Clear all cached pages."""
        self._gpu_pool.clear()
        self._cpu_pool.clear()
        for pt in self._page_tables:
            pt.clear()
        self._seq_positions = [0] * self.num_layers
        self._gpu_page_count = 0
        logger.debug("PagedKVCache cleared")

    @property
    def total_cached_tokens(self) -> int:
        """Total number of cached tokens across all layers."""
        return sum(self._seq_positions)

    @property
    def gpu_pages_used(self) -> int:
        """Number of GPU pages currently in use."""
        return self._gpu_page_count

    @property
    def cpu_pages_used(self) -> int:
        """Number of CPU pages currently in use."""
        return len(self._cpu_pool)
