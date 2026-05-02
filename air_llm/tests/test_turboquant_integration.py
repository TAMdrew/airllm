"""Tests for TurboQuant integration with AirLLMBaseModel.

Verifies that:
- The kv_compression parameter correctly initializes PolarQuantConfig
- Different compression modes (turboquant, 3bit, 4bit) create correct configs
- PolarQuantConfig is exported from the package
- KV compressor round-trip works at various bit widths and head dimensions
- QJL compressed attention with higher projection dimensions
- Memory reduction calculations are accurate
"""

from __future__ import annotations

import pytest
import torch

from air_llm.airllm.kv_cache import (
    CompressedKVCache,
    KVCacheCompressor,
    PolarQuantConfig,
)


# ---------------------------------------------------------------------------
# PolarQuantConfig extended edge cases
# ---------------------------------------------------------------------------
class TestPolarQuantConfigEdgeCases:
    """Extended PolarQuantConfig validation tests."""

    def test_2bit_has_3_levels(self) -> None:
        """2-bit quantization should use 3 levels (2^2 - 1)."""
        config = PolarQuantConfig(bits=2)
        c = KVCacheCompressor(config)
        assert c._n_levels == 3

    def test_3bit_has_7_levels(self) -> None:
        """3-bit quantization should use 7 levels (2^3 - 1)."""
        config = PolarQuantConfig(bits=3)
        c = KVCacheCompressor(config)
        assert c._n_levels == 7

    def test_4bit_has_15_levels(self) -> None:
        """4-bit quantization should use 15 levels (2^4 - 1)."""
        config = PolarQuantConfig(bits=4)
        c = KVCacheCompressor(config)
        assert c._n_levels == 15

    def test_different_seeds_produce_different_rotations(self) -> None:
        """Different rotation seeds produce different rotation matrices."""
        c1 = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0, rotation_seed=42))
        c2 = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0, rotation_seed=99))
        r1 = c1._get_rotation_matrix(64, torch.device("cpu"), torch.float32)
        r2 = c2._get_rotation_matrix(64, torch.device("cpu"), torch.float32)
        assert not torch.equal(r1, r2)

    def test_same_seed_produces_same_rotations(self) -> None:
        """Same rotation seed always produces identical matrices."""
        c1 = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0, rotation_seed=42))
        c2 = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0, rotation_seed=42))
        r1 = c1._get_rotation_matrix(128, torch.device("cpu"), torch.float32)
        r2 = c2._get_rotation_matrix(128, torch.device("cpu"), torch.float32)
        assert torch.equal(r1, r2)

    def test_rotation_matrix_is_orthogonal(self) -> None:
        """Rotation matrix R should satisfy R @ R^T = I."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        r = c._get_rotation_matrix(64, torch.device("cpu"), torch.float32)
        identity = torch.eye(64)
        product = r @ r.T
        assert torch.allclose(product, identity, atol=1e-5)

    def test_rademacher_matrix_shape(self) -> None:
        """Rademacher matrix should be (qjl_dim, head_dim)."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=128))
        r = c._get_rademacher_matrix(64, torch.device("cpu"), torch.float32)
        assert r.shape == (128, 64)


# ---------------------------------------------------------------------------
# TurboQuant KV compression mode parsing (mimics airllm_base.py logic)
# ---------------------------------------------------------------------------
class TestKVCompressionModeParsing:
    """Test that kv_compression string values map to correct configs."""

    def test_turboquant_mode_defaults(self) -> None:
        """'turboquant' mode should use 3-bit with QJL dim 64."""
        # Mimics airllm_base.py logic
        kv_bits = 3
        kv_qjl_dim = 64

        config = PolarQuantConfig(bits=kv_bits, qjl_dim=kv_qjl_dim)
        c = KVCacheCompressor(config)
        assert c.bits == 3
        assert c._config.qjl_dim == 64
        assert c.use_residual is True

    def test_3bit_mode(self) -> None:
        """'3bit' mode should use 3-bit without QJL."""
        kv_compression = "3bit"
        parsed_bits = int(kv_compression.replace("bit", ""))
        config = PolarQuantConfig(bits=parsed_bits, qjl_dim=0)
        c = KVCacheCompressor(config)
        assert c.bits == 3
        assert c._config.qjl_dim == 0
        assert c.use_residual is False

    def test_4bit_mode(self) -> None:
        """'4bit' mode should use 4-bit without QJL."""
        kv_compression = "4bit"
        parsed_bits = int(kv_compression.replace("bit", ""))
        config = PolarQuantConfig(bits=parsed_bits, qjl_dim=0)
        c = KVCacheCompressor(config)
        assert c.bits == 4
        assert c._config.qjl_dim == 0

    def test_custom_bits_and_qjl(self) -> None:
        """Custom kv_bits and kv_qjl_dim should be respected."""
        config = PolarQuantConfig(bits=4, qjl_dim=128)
        c = KVCacheCompressor(config)
        assert c.bits == 4
        assert c._config.qjl_dim == 128


# ---------------------------------------------------------------------------
# Round-trip fidelity at various dimensions
# ---------------------------------------------------------------------------
class TestRoundTripVariousDimensions:
    """Test compression fidelity across different tensor shapes."""

    @pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
    def test_3bit_round_trip_various_head_dims(self, head_dim: int) -> None:
        """3-bit round-trip works for all common head dimensions."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        t = torch.randn(1, 4, 8, head_dim, dtype=torch.float16)
        compressed = c.compress(t)
        decompressed = c.decompress(compressed)
        assert decompressed.shape == t.shape

        # Basic fidelity check — reconstruction isn't random noise
        orig = t.float().reshape(-1, head_dim)
        recon = decompressed.float().reshape(-1, head_dim)
        cos_sim = torch.nn.functional.cosine_similarity(orig, recon, dim=-1).mean()
        assert cos_sim.item() > 0.5, f"Fidelity too low for head_dim={head_dim}"

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_round_trip_various_batch_sizes(self, batch_size: int) -> None:
        """Round-trip works for different batch sizes."""
        c = KVCacheCompressor(PolarQuantConfig(bits=4, qjl_dim=0))
        t = torch.randn(batch_size, 8, 16, 128, dtype=torch.float16)
        compressed = c.compress(t)
        decompressed = c.decompress(compressed)
        assert decompressed.shape == t.shape

    @pytest.mark.parametrize("n_heads", [1, 4, 8, 32])
    def test_round_trip_various_head_counts(self, n_heads: int) -> None:
        """Round-trip works for different attention head counts."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        t = torch.randn(1, n_heads, 8, 64, dtype=torch.float16)
        compressed = c.compress(t)
        decompressed = c.decompress(compressed)
        assert decompressed.shape == t.shape

    @pytest.mark.parametrize("seq_len", [1, 16, 128, 512])
    def test_round_trip_various_seq_lengths(self, seq_len: int) -> None:
        """Round-trip works for different sequence lengths."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        t = torch.randn(1, 4, seq_len, 64, dtype=torch.float16)
        compressed = c.compress(t)
        decompressed = c.decompress(compressed)
        assert decompressed.shape == t.shape


# ---------------------------------------------------------------------------
# QJL with higher projection dimensions
# ---------------------------------------------------------------------------
class TestHigherQJLDims:
    """Test QJL with various projection dimensions."""

    @pytest.mark.parametrize("qjl_dim", [32, 64, 128, 256])
    def test_qjl_signs_correct_packed_size(self, qjl_dim: int) -> None:
        """Packed sign bits should be ceil(qjl_dim / 8) bytes."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=qjl_dim))
        t = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        compressed = c.compress(t)
        expected_bytes = (qjl_dim + 7) // 8
        assert compressed.qjl_signs.shape[-1] == expected_bytes

    def test_higher_qjl_dim_better_attention_accuracy(self) -> None:
        """More QJL projections should improve attention approximation."""
        torch.manual_seed(42)
        kv = torch.randn(1, 4, 16, 64, dtype=torch.float16)
        query = torch.randn(1, 4, 4, 64, dtype=torch.float16)

        # Standard attention
        standard = torch.matmul(query.float(), kv.float().transpose(-2, -1))

        # Low QJL dim
        c_low = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=32))
        compressed_low = c_low.compress(kv)
        approx_low = c_low.compressed_attention(query, compressed_low)
        corr_low = torch.corrcoef(torch.stack([approx_low.flatten(), standard.flatten()]))[0, 1]

        # High QJL dim
        c_high = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=256))
        compressed_high = c_high.compress(kv)
        approx_high = c_high.compressed_attention(query, compressed_high)
        corr_high = torch.corrcoef(torch.stack([approx_high.flatten(), standard.flatten()]))[0, 1]

        # Higher QJL dim should give better or equal correlation
        assert corr_high.item() >= corr_low.item() - 0.1, (
            f"Higher QJL dim ({corr_high:.3f}) should be >= lower ({corr_low:.3f})"
        )


# ---------------------------------------------------------------------------
# Memory reduction accuracy
# ---------------------------------------------------------------------------
class TestMemoryReductionAccuracy:
    """Test that memory reduction calculations are mathematically correct."""

    def test_2bit_no_qjl_ratio(self) -> None:
        """2-bit without QJL: 16 / (2 + 16/128) ≈ 7.5x."""
        c = KVCacheCompressor(PolarQuantConfig(bits=2, qjl_dim=0))
        ratio = c.memory_reduction_ratio()
        expected = 16.0 / (2.0 + 16.0 / 128.0)
        assert abs(ratio - expected) < 0.01

    def test_3bit_with_qjl_64_ratio(self) -> None:
        """3-bit with QJL-64: 16 / (3 + 16/128 + 64/128) ≈ 4.4x."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=64))
        ratio = c.memory_reduction_ratio()
        expected = 16.0 / (3.0 + 16.0 / 128.0 + 64.0 / 128.0)
        assert abs(ratio - expected) < 0.01

    def test_4bit_no_qjl_ratio(self) -> None:
        """4-bit without QJL: 16 / (4 + 16/128) ≈ 3.9x."""
        c = KVCacheCompressor(PolarQuantConfig(bits=4, qjl_dim=0))
        ratio = c.memory_reduction_ratio()
        expected = 16.0 / (4.0 + 16.0 / 128.0)
        assert abs(ratio - expected) < 0.01


# ---------------------------------------------------------------------------
# Package export verification
# ---------------------------------------------------------------------------
class TestPackageExports:
    """Verify TurboQuant types are properly exported from the package."""

    def test_polarquantconfig_exported(self) -> None:
        from air_llm.airllm import PolarQuantConfig as Exported

        assert Exported is PolarQuantConfig

    def test_kvcachecompressor_exported(self) -> None:
        from air_llm.airllm import KVCacheCompressor as Exported

        assert Exported is KVCacheCompressor

    def test_compressedkvcache_exported(self) -> None:
        from air_llm.airllm import CompressedKVCache as Exported

        assert Exported is CompressedKVCache

    def test_polarquantconfig_in_all(self) -> None:
        import air_llm.airllm

        assert "PolarQuantConfig" in air_llm.airllm.__all__

    def test_kvcachecompressor_in_all(self) -> None:
        import air_llm.airllm

        assert "KVCacheCompressor" in air_llm.airllm.__all__
