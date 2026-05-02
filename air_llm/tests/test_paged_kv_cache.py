"""Tests for paged KV cache module."""

import torch

from air_llm.airllm.paged_kv_cache import PageConfig, PagedKVCache


class TestPageConfig:
    """Tests for the PageConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config has sensible values."""
        config = PageConfig()
        assert config.page_size == 16
        assert config.num_gpu_pages == 64
        assert config.num_cpu_pages == 256
        assert config.head_dim == 128
        assert config.num_kv_heads == 8

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        config = PageConfig(
            page_size=32,
            num_gpu_pages=128,
            num_cpu_pages=512,
            head_dim=64,
            num_kv_heads=4,
        )
        assert config.page_size == 32
        assert config.num_gpu_pages == 128
        assert config.num_cpu_pages == 512
        assert config.head_dim == 64
        assert config.num_kv_heads == 4


class TestPagedKVCacheInit:
    """Tests for PagedKVCache initialization."""

    def test_init_creates_page_tables(self) -> None:
        """Init creates per-layer page tables."""
        config = PageConfig(page_size=4, num_gpu_pages=8, num_kv_heads=2, head_dim=8)
        cache = PagedKVCache(config, num_layers=4, gpu_device="cpu")
        assert len(cache._page_tables) == 4
        assert cache.gpu_pages_used == 0
        assert cache.cpu_pages_used == 0
        assert cache.total_cached_tokens == 0

    def test_init_seq_positions_zero(self) -> None:
        """All sequence positions start at zero."""
        config = PageConfig(page_size=4, num_gpu_pages=8, num_kv_heads=2, head_dim=8)
        cache = PagedKVCache(config, num_layers=3, gpu_device="cpu")
        assert cache._seq_positions == [0, 0, 0]


class TestPagedKVCacheAppendGet:
    """Tests for append and get operations."""

    def _make_cache(
        self,
        num_layers: int = 2,
        page_size: int = 4,
        num_gpu_pages: int = 16,
        num_kv_heads: int = 2,
        head_dim: int = 8,
    ) -> PagedKVCache:
        """Create a test cache on CPU."""
        config = PageConfig(
            page_size=page_size,
            num_gpu_pages=num_gpu_pages,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        return PagedKVCache(config, num_layers=num_layers, gpu_device="cpu")

    def test_append_single_token(self) -> None:
        """Appending a single token creates one page."""
        cache = self._make_cache()
        key = torch.randn(1, 2, 1, 8, dtype=torch.float16)
        value = torch.randn(1, 2, 1, 8, dtype=torch.float16)

        cache.append(layer_idx=0, key=key, value=value)

        assert cache.total_cached_tokens == 1
        assert cache.gpu_pages_used == 1

    def test_append_multiple_tokens(self) -> None:
        """Appending multiple tokens works across pages."""
        cache = self._make_cache(page_size=4)
        # Append 6 tokens — should create 2 pages (4 + 2)
        key = torch.randn(1, 2, 6, 8, dtype=torch.float16)
        value = torch.randn(1, 2, 6, 8, dtype=torch.float16)

        cache.append(layer_idx=0, key=key, value=value)

        assert cache._seq_positions[0] == 6
        assert cache.gpu_pages_used == 2

    def test_get_returns_correct_shape(self) -> None:
        """Get returns concatenated tensors with correct shape."""
        cache = self._make_cache(page_size=4)
        key = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        value = torch.randn(1, 2, 3, 8, dtype=torch.float16)

        cache.append(layer_idx=0, key=key, value=value)
        result = cache.get(layer_idx=0)

        assert result is not None
        k, v = result
        assert k.shape == (1, 2, 3, 8)
        assert v.shape == (1, 2, 3, 8)

    def test_get_empty_layer_returns_none(self) -> None:
        """Get on a layer with no cached data returns None."""
        cache = self._make_cache()
        assert cache.get(layer_idx=0) is None

    def test_append_preserves_values(self) -> None:
        """Appended values are retrievable."""
        cache = self._make_cache(page_size=4)
        key = torch.ones(1, 2, 2, 8, dtype=torch.float16) * 3.0
        value = torch.ones(1, 2, 2, 8, dtype=torch.float16) * 7.0

        cache.append(layer_idx=0, key=key, value=value)
        result = cache.get(layer_idx=0)

        assert result is not None
        k, v = result
        assert torch.allclose(k, key)
        assert torch.allclose(v, value)

    def test_append_incremental(self) -> None:
        """Multiple appends accumulate correctly."""
        cache = self._make_cache(page_size=4)

        k1 = torch.randn(1, 2, 2, 8, dtype=torch.float16)
        v1 = torch.randn(1, 2, 2, 8, dtype=torch.float16)
        cache.append(layer_idx=0, key=k1, value=v1)

        k2 = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        v2 = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        cache.append(layer_idx=0, key=k2, value=v2)

        result = cache.get(layer_idx=0)
        assert result is not None
        k, v = result
        assert k.shape == (1, 2, 5, 8)  # 2 + 3 = 5 tokens
        assert v.shape == (1, 2, 5, 8)

    def test_multiple_layers_independent(self) -> None:
        """Each layer's cache is independent."""
        cache = self._make_cache(num_layers=3, page_size=4)

        k0 = torch.randn(1, 2, 2, 8, dtype=torch.float16)
        v0 = torch.randn(1, 2, 2, 8, dtype=torch.float16)
        cache.append(layer_idx=0, key=k0, value=v0)

        k2 = torch.randn(1, 2, 5, 8, dtype=torch.float16)
        v2 = torch.randn(1, 2, 5, 8, dtype=torch.float16)
        cache.append(layer_idx=2, key=k2, value=v2)

        assert cache._seq_positions[0] == 2
        assert cache._seq_positions[1] == 0
        assert cache._seq_positions[2] == 5

        assert cache.get(layer_idx=1) is None


class TestPagedKVCacheEviction:
    """Tests for GPU→CPU page eviction."""

    def test_eviction_occurs_when_gpu_full(self) -> None:
        """Pages are evicted to CPU when GPU pool is full."""
        config = PageConfig(page_size=2, num_gpu_pages=2, num_kv_heads=1, head_dim=4)
        cache = PagedKVCache(config, num_layers=1, gpu_device="cpu")

        # Fill 2 GPU pages (4 tokens)
        k1 = torch.randn(1, 1, 4, 4, dtype=torch.float16)
        v1 = torch.randn(1, 1, 4, 4, dtype=torch.float16)
        cache.append(layer_idx=0, key=k1, value=v1)
        assert cache.gpu_pages_used == 2

        # Add more — should trigger eviction
        k2 = torch.randn(1, 1, 2, 4, dtype=torch.float16)
        v2 = torch.randn(1, 1, 2, 4, dtype=torch.float16)
        cache.append(layer_idx=0, key=k2, value=v2)

        assert cache.cpu_pages_used >= 1

    def test_get_swaps_from_cpu(self) -> None:
        """Getting data from an evicted page swaps it back to GPU."""
        config = PageConfig(page_size=2, num_gpu_pages=2, num_kv_heads=1, head_dim=4)
        cache = PagedKVCache(config, num_layers=1, gpu_device="cpu")

        # Fill GPU, force eviction, then retrieve
        k = torch.randn(1, 1, 6, 4, dtype=torch.float16)
        v = torch.randn(1, 1, 6, 4, dtype=torch.float16)
        cache.append(layer_idx=0, key=k, value=v)

        result = cache.get(layer_idx=0)
        assert result is not None
        k_out, _v_out = result
        assert k_out.shape == (1, 1, 6, 4)


class TestPagedKVCacheClear:
    """Tests for cache clearing."""

    def test_clear_resets_everything(self) -> None:
        """Clear removes all pages and resets positions."""
        config = PageConfig(page_size=4, num_gpu_pages=8, num_kv_heads=2, head_dim=8)
        cache = PagedKVCache(config, num_layers=2, gpu_device="cpu")

        k = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        v = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        cache.append(layer_idx=0, key=k, value=v)
        cache.append(layer_idx=1, key=k, value=v)

        cache.clear()

        assert cache.gpu_pages_used == 0
        assert cache.cpu_pages_used == 0
        assert cache.total_cached_tokens == 0
        assert cache.get(layer_idx=0) is None
        assert cache.get(layer_idx=1) is None


class TestPagedKVCacheProperties:
    """Tests for cache properties."""

    def test_total_cached_tokens(self) -> None:
        """total_cached_tokens sums across layers."""
        config = PageConfig(page_size=4, num_gpu_pages=16, num_kv_heads=2, head_dim=8)
        cache = PagedKVCache(config, num_layers=2, gpu_device="cpu")

        k1 = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        v1 = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        cache.append(layer_idx=0, key=k1, value=v1)

        k2 = torch.randn(1, 2, 5, 8, dtype=torch.float16)
        v2 = torch.randn(1, 2, 5, 8, dtype=torch.float16)
        cache.append(layer_idx=1, key=k2, value=v2)

        assert cache.total_cached_tokens == 8  # 3 + 5

    def test_gpu_pages_used(self) -> None:
        """gpu_pages_used tracks allocated GPU pages."""
        config = PageConfig(page_size=4, num_gpu_pages=16, num_kv_heads=2, head_dim=8)
        cache = PagedKVCache(config, num_layers=1, gpu_device="cpu")

        assert cache.gpu_pages_used == 0

        k = torch.randn(1, 2, 5, 8, dtype=torch.float16)
        v = torch.randn(1, 2, 5, 8, dtype=torch.float16)
        cache.append(layer_idx=0, key=k, value=v)

        assert cache.gpu_pages_used == 2  # 5 tokens across 2 pages of size 4

    def test_cpu_pages_used(self) -> None:
        """cpu_pages_used tracks evicted pages."""
        config = PageConfig(page_size=4, num_gpu_pages=16, num_kv_heads=2, head_dim=8)
        cache = PagedKVCache(config, num_layers=1, gpu_device="cpu")
        assert cache.cpu_pages_used == 0
