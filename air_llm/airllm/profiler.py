"""Layered profiler for per-layer timing and GPU memory tracking.

Collects timing data across the layer-by-layer forward pass to help
identify bottlenecks in disk I/O, GPU transfer, and compute.
"""

from __future__ import annotations

import logging

import torch

from .constants import BYTES_PER_GB, PROFILER_INITIAL_MIN_FREE_MEM

logger = logging.getLogger(__name__)


class LayeredProfiler:
    """Accumulates per-layer timing measurements and optional GPU memory stats.

    Args:
        print_memory: If ``True``, log free GPU memory after each profiling
            event.
    """

    def __init__(self, print_memory: bool = False) -> None:
        self.profiling_time_dict: dict[str, list[float]] = {}
        self.print_memory = print_memory
        self.min_free_mem: int = PROFILER_INITIAL_MIN_FREE_MEM

    def add_profiling_time(self, item: str, elapsed: float) -> None:
        """Record an elapsed time measurement for *item*.

        Args:
            item: A label identifying the profiling event
                (e.g. ``"load_safe_tensor"``).
            elapsed: Elapsed time in seconds.
        """
        self.profiling_time_dict.setdefault(item, []).append(elapsed)

        if self.print_memory and torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0]
            self.min_free_mem = min(self.min_free_mem, free_mem)
            logger.info(
                "Free VRAM @%s: %.2fGB, min free: %.2fGB",
                item,
                free_mem / BYTES_PER_GB,
                self.min_free_mem / BYTES_PER_GB,
            )

    def clear_profiling_time(self) -> None:
        """Reset all accumulated timing lists (keeps keys)."""
        for item in self.profiling_time_dict:
            self.profiling_time_dict[item] = []

    def print_profiling_time(self) -> None:
        """Log the total accumulated time for each profiling item."""
        for item, times in self.profiling_time_dict.items():
            logger.info("Total time for %s: %.4f", item, sum(times))
