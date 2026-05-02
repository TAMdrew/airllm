"""Async Layer Loader — Enables prefetching even with quantized weights.

Fixes the long-standing issue where prefetching is disabled when using
quantized (compressed) weights. Uses CUDA streams to overlap:
1. Disk → CPU transfer (via ThreadPoolExecutor)
2. CPU → GPU transfer (via non-blocking CUDA copy on a separate stream)
3. GPU compute (on the default stream)

This enables all three stages to run concurrently, maximizing throughput.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch

logger = logging.getLogger(__name__)


class AsyncLayerLoader:
    """Asynchronous layer loader with CUDA stream overlap.

    Manages a pipeline of disk→CPU→GPU transfers that overlap with
    GPU computation, even when using quantized weights.

    Args:
        device: Target GPU device.
        num_prefetch_workers: Number of background threads for disk I/O.
        pin_memory: Whether to pin CPU memory for faster GPU transfers.

    Example:
        >>> loader = AsyncLayerLoader(device="cuda:0")
        >>> # Start prefetching next layer while current computes
        >>> loader.prefetch(load_fn=load_from_disk, layer_idx=i+1)
        >>> # ... compute current layer ...
        >>> next_weights = loader.get_prefetched()  # Returns when ready
    """

    def __init__(
        self,
        device: str = "cuda:0",
        num_prefetch_workers: int = 1,
        *,
        pin_memory: bool = True,
    ) -> None:
        self.device = device
        self.pin_memory = pin_memory
        self._executor = ThreadPoolExecutor(
            max_workers=num_prefetch_workers,
            thread_name_prefix="airllm-prefetch",
        )
        self._transfer_stream: torch.cuda.Stream | None = None
        self._prefetch_future: Future[dict[str, torch.Tensor]] | None = None

        # Initialize CUDA stream if available
        if torch.cuda.is_available():
            self._transfer_stream = torch.cuda.Stream(device=device)

        logger.debug(
            "AsyncLayerLoader initialized: device=%s, pin_memory=%s, stream=%s",
            device,
            pin_memory,
            self._transfer_stream is not None,
        )

    def prefetch(
        self,
        load_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Start prefetching a layer in the background.

        Args:
            load_fn: Callable that loads layer weights from disk/CPU.
                     Must return a dict of {name: tensor}.
            *args: Positional arguments for load_fn.
            **kwargs: Keyword arguments for load_fn.
        """
        self._prefetch_future = self._executor.submit(
            self._load_and_transfer, load_fn, *args, **kwargs
        )

    def _load_and_transfer(
        self,
        load_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Load from disk and transfer to GPU asynchronously.

        This runs in a background thread. The CPU→GPU transfer
        uses a separate CUDA stream to overlap with computation.

        Args:
            load_fn: Function to load weights from disk.
            *args: Positional arguments for load_fn.
            **kwargs: Keyword arguments for load_fn.

        Returns:
            Dict of tensors on the target GPU device.
        """
        # Step 1: Load from disk to CPU (I/O bound — runs in background)
        cpu_weights = load_fn(*args, **kwargs)

        if not isinstance(cpu_weights, dict):
            return cpu_weights

        gpu_weights: dict[str, torch.Tensor] = {}

        # Step 2: Transfer to GPU on separate stream (overlaps with compute)
        if self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                for name, tensor in cpu_weights.items():
                    if self.pin_memory and not tensor.is_pinned():
                        try:
                            tensor = tensor.pin_memory()
                        except RuntimeError:
                            pass  # pin_memory may fail on some systems
                    gpu_weights[name] = tensor.to(self.device, non_blocking=True)
        else:
            # Fallback: direct transfer (no CUDA available)
            for name, tensor in cpu_weights.items():
                gpu_weights[name] = tensor.to(self.device)

        return gpu_weights

    def get_prefetched(self) -> dict[str, torch.Tensor] | None:
        """Wait for and retrieve prefetched layer weights.

        Synchronizes the transfer stream with the compute stream
        to ensure all data is available on the GPU.

        Returns:
            Dict of layer tensors on GPU, or None if no prefetch pending.
        """
        if self._prefetch_future is None:
            return None

        result = self._prefetch_future.result()
        self._prefetch_future = None

        # Synchronize streams to ensure transfer is complete
        if self._transfer_stream is not None:
            torch.cuda.current_stream().wait_stream(self._transfer_stream)

        return result

    def has_pending(self) -> bool:
        """Check if there's a pending prefetch operation.

        Returns:
            True if a prefetch is in progress.
        """
        return self._prefetch_future is not None

    def shutdown(self) -> None:
        """Shutdown the background executor."""
        self._executor.shutdown(wait=False)
        logger.debug("AsyncLayerLoader shutdown")

    # Context manager support for safe resource cleanup (ARCH-6).
    def __enter__(self) -> AsyncLayerLoader:
        """Enter context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit context manager — shuts down the thread pool."""
        self.shutdown()

    def __del__(self) -> None:
        """Safety-net cleanup on garbage collection."""
        self.shutdown()
