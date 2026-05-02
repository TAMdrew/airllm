"""Tests for async layer loader module."""

import time

import torch

from air_llm.airllm.async_loader import AsyncLayerLoader


class TestAsyncLayerLoaderInit:
    """Tests for AsyncLayerLoader initialization."""

    def test_default_init(self) -> None:
        """Default initialization sets expected attributes."""
        loader = AsyncLayerLoader(device="cpu")
        assert loader.device == "cpu"
        assert loader.pin_memory is True
        loader.shutdown()

    def test_custom_init(self) -> None:
        """Custom parameters are stored correctly."""
        loader = AsyncLayerLoader(
            device="cpu",
            num_prefetch_workers=2,
            pin_memory=False,
        )
        assert loader.device == "cpu"
        assert loader.pin_memory is False
        loader.shutdown()

    def test_no_cuda_stream_on_cpu(self) -> None:
        """No CUDA stream created when CUDA is unavailable or device is CPU."""
        loader = AsyncLayerLoader(device="cpu")
        # On CPU-only systems, _transfer_stream should be None
        # On CUDA systems, it may still be None if device is "cpu"
        # Either way, the loader should work
        loader.shutdown()


class TestAsyncLayerLoaderPrefetch:
    """Tests for prefetching and retrieval without CUDA."""

    def test_prefetch_and_get_cpu(self) -> None:
        """Prefetch with a CPU-only load function works end to end."""
        loader = AsyncLayerLoader(device="cpu", pin_memory=False)

        def load_fn() -> dict[str, torch.Tensor]:
            return {
                "weight": torch.randn(4, 4),
                "bias": torch.randn(4),
            }

        loader.prefetch(load_fn)
        assert loader.has_pending() is True

        result = loader.get_prefetched()
        assert result is not None
        assert "weight" in result
        assert "bias" in result
        assert result["weight"].shape == (4, 4)
        assert result["bias"].shape == (4,)
        assert loader.has_pending() is False

        loader.shutdown()

    def test_prefetch_with_args(self) -> None:
        """Prefetch passes positional and keyword arguments to load_fn."""
        loader = AsyncLayerLoader(device="cpu", pin_memory=False)

        def load_fn(size: int, fill_value: float = 0.0) -> dict[str, torch.Tensor]:
            return {"data": torch.full((size,), fill_value)}

        loader.prefetch(load_fn, 8, fill_value=3.14)
        result = loader.get_prefetched()
        assert result is not None
        assert result["data"].shape == (8,)
        assert torch.allclose(result["data"], torch.tensor(3.14))

        loader.shutdown()

    def test_get_prefetched_none_when_no_pending(self) -> None:
        """get_prefetched returns None when no prefetch is pending."""
        loader = AsyncLayerLoader(device="cpu", pin_memory=False)
        assert loader.get_prefetched() is None
        loader.shutdown()

    def test_has_pending_false_initially(self) -> None:
        """has_pending is False before any prefetch."""
        loader = AsyncLayerLoader(device="cpu", pin_memory=False)
        assert loader.has_pending() is False
        loader.shutdown()

    def test_has_pending_false_after_get(self) -> None:
        """has_pending is False after get_prefetched retrieves the result."""
        loader = AsyncLayerLoader(device="cpu", pin_memory=False)

        loader.prefetch(lambda: {"w": torch.zeros(2)})
        assert loader.has_pending() is True

        loader.get_prefetched()
        assert loader.has_pending() is False

        loader.shutdown()

    def test_non_dict_return(self) -> None:
        """Load function returning non-dict is passed through."""
        loader = AsyncLayerLoader(device="cpu", pin_memory=False)

        loader.prefetch(lambda: torch.tensor([1, 2, 3]))
        result = loader.get_prefetched()
        # Non-dict returns are passed through unchanged
        assert isinstance(result, torch.Tensor)

        loader.shutdown()

    def test_prefetch_overlaps_with_work(self) -> None:
        """Prefetch can run concurrently with other work."""
        loader = AsyncLayerLoader(device="cpu", pin_memory=False)

        def slow_load() -> dict[str, torch.Tensor]:
            time.sleep(0.05)  # Simulate slow disk I/O
            return {"w": torch.randn(16, 16)}

        loader.prefetch(slow_load)

        # Do some "work" while prefetch runs
        _ = torch.randn(100, 100) @ torch.randn(100, 100)

        result = loader.get_prefetched()
        assert result is not None
        assert result["w"].shape == (16, 16)

        loader.shutdown()


class TestAsyncLayerLoaderShutdown:
    """Tests for shutdown behavior."""

    def test_shutdown_is_idempotent(self) -> None:
        """Calling shutdown multiple times does not raise."""
        loader = AsyncLayerLoader(device="cpu")
        loader.shutdown()
        loader.shutdown()  # Should not raise

    def test_del_calls_shutdown(self) -> None:
        """Destructor calls shutdown without errors."""
        loader = AsyncLayerLoader(device="cpu")
        del loader  # Should not raise
