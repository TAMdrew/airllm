"""Test suite for LayeredProfiler.

Tests timing accumulation, memory tracking, and print/clear lifecycle.
"""

from __future__ import annotations

from air_llm.airllm.profiler import LayeredProfiler


def test_profiler_add_profiling_time_accumulates() -> None:
    """Test that add_profiling_time accumulates values for the same key."""
    profiler = LayeredProfiler()
    profiler.add_profiling_time("load", 1.0)
    profiler.add_profiling_time("load", 2.0)
    profiler.add_profiling_time("compute", 0.5)

    assert len(profiler.profiling_time_dict["load"]) == 2
    assert profiler.profiling_time_dict["load"] == [1.0, 2.0]
    assert profiler.profiling_time_dict["compute"] == [0.5]


def test_profiler_clear_resets_all_timings() -> None:
    """Test that clear_profiling_time empties the timing dict."""
    profiler = LayeredProfiler()
    profiler.add_profiling_time("load", 1.0)
    profiler.add_profiling_time("compute", 2.0)

    profiler.clear_profiling_time()

    # clear_profiling_time empties the value lists but keeps keys
    for values in profiler.profiling_time_dict.values():
        assert values == []


def test_profiler_print_does_not_raise() -> None:
    """Test that print_profiling_time runs without errors even when empty."""
    profiler = LayeredProfiler()
    profiler.print_profiling_time()  # Should not raise

    profiler.add_profiling_time("load", 1.5)
    profiler.print_profiling_time()  # Should not raise


def test_profiler_init_defaults() -> None:
    """Test default initialization values."""
    profiler = LayeredProfiler()
    assert profiler.profiling_time_dict == {}
    assert profiler.print_memory is False


def test_profiler_init_with_print_memory() -> None:
    """Test initialization with print_memory=True."""
    profiler = LayeredProfiler(print_memory=True)
    assert profiler.print_memory is True


def test_profiler_multiple_keys_independent() -> None:
    """Test that different keys accumulate independently."""
    profiler = LayeredProfiler()
    profiler.add_profiling_time("disk_io", 0.1)
    profiler.add_profiling_time("gpu_transfer", 0.2)
    profiler.add_profiling_time("disk_io", 0.3)

    assert len(profiler.profiling_time_dict) == 2
    assert sum(profiler.profiling_time_dict["disk_io"]) == 0.4
    assert sum(profiler.profiling_time_dict["gpu_transfer"]) == 0.2
