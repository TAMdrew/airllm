"""Tests for TurboQuant KV cache compression (PolarQuant + QJL).

Validates the faithful ICLR 2026 implementation:
- PolarQuant: rotation → polar decomposition → uniform quantization
- QJL: Rademacher projections → compressed-domain attention
- Round-trip fidelity, edge cases, and memory reduction
"""

import pytest
import torch

from air_llm.airllm.kv_cache import (
    CompressedKVCache,
    KVCacheCompressor,
    PolarQuantConfig,
)


# ---------------------------------------------------------------------------
# PolarQuantConfig tests
# ---------------------------------------------------------------------------
class TestPolarQuantConfig:
    """Tests for the immutable PolarQuantConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config uses 3-bit, 64 QJL dims."""
        config = PolarQuantConfig()
        assert config.bits == 3
        assert config.qjl_dim == 64
        assert config.rotation_seed == 42

    def test_custom_config(self) -> None:
        """Custom config accepts valid parameters."""
        config = PolarQuantConfig(bits=4, qjl_dim=128, rotation_seed=99)
        assert config.bits == 4
        assert config.qjl_dim == 128
        assert config.rotation_seed == 99

    def test_2bit_config(self) -> None:
        """2-bit quantization is valid."""
        config = PolarQuantConfig(bits=2)
        assert config.bits == 2

    def test_invalid_bits_raises(self) -> None:
        """Invalid bit width raises ValueError."""
        with pytest.raises(ValueError, match="bits must be one of"):
            PolarQuantConfig(bits=5)

    def test_invalid_bits_1_raises(self) -> None:
        """1-bit is not supported."""
        with pytest.raises(ValueError, match="bits must be one of"):
            PolarQuantConfig(bits=1)

    def test_negative_qjl_dim_raises(self) -> None:
        """Negative QJL dim raises ValueError."""
        with pytest.raises(ValueError, match="qjl_dim must be non-negative"):
            PolarQuantConfig(qjl_dim=-1)

    def test_zero_qjl_dim_is_valid(self) -> None:
        """QJL dim of 0 disables QJL projections."""
        config = PolarQuantConfig(qjl_dim=0)
        assert config.qjl_dim == 0

    def test_frozen(self) -> None:
        """Config is immutable (frozen dataclass)."""
        config = PolarQuantConfig()
        with pytest.raises(AttributeError):
            config.bits = 4  # type: ignore[misc]


# ---------------------------------------------------------------------------
# KVCacheCompressor initialization tests
# ---------------------------------------------------------------------------
class TestKVCacheCompressorInit:
    """Tests for KVCacheCompressor construction."""

    def test_init_with_config(self) -> None:
        """Initialize with PolarQuantConfig."""
        config = PolarQuantConfig(bits=4, qjl_dim=32)
        c = KVCacheCompressor(config)
        assert c.bits == 4
        assert c._config.qjl_dim == 32

    def test_init_default(self) -> None:
        """Default initialization uses 3-bit with QJL."""
        c = KVCacheCompressor()
        assert c.bits == 3
        assert c.use_residual is True

    def test_init_legacy_bits(self) -> None:
        """Legacy bits parameter works for backward compatibility."""
        c = KVCacheCompressor(bits=4)
        assert c.bits == 4

    def test_init_legacy_no_residual(self) -> None:
        """Legacy use_residual=False disables QJL."""
        c = KVCacheCompressor(use_residual=False)
        assert c.use_residual is False
        assert c._config.qjl_dim == 0

    def test_init_invalid_bits_raises(self) -> None:
        """Invalid bit width raises ValueError."""
        with pytest.raises(ValueError, match="bits must be one of"):
            KVCacheCompressor(bits=5)


# ---------------------------------------------------------------------------
# PolarQuant compress/decompress round-trip tests
# ---------------------------------------------------------------------------
class TestPolarQuantRoundTrip:
    """Tests for PolarQuant compression fidelity."""

    @pytest.fixture()
    def compressor_3bit(self) -> KVCacheCompressor:
        """3-bit compressor without QJL for isolated PolarQuant testing."""
        return KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))

    @pytest.fixture()
    def compressor_4bit(self) -> KVCacheCompressor:
        """4-bit compressor without QJL."""
        return KVCacheCompressor(PolarQuantConfig(bits=4, qjl_dim=0))

    @pytest.fixture()
    def sample_kv(self) -> torch.Tensor:
        """Sample KV cache tensor: (batch=1, heads=4, seq=16, head_dim=128).

        Uses head_dim=128 (standard for most LLMs) for realistic fidelity.
        The JL lemma requires sufficient dimensionality for the rotation
        to fully normalize component distributions.
        """
        torch.manual_seed(42)
        return torch.randn(1, 4, 16, 128, dtype=torch.float16)

    def test_shape_preserved(
        self, compressor_3bit: KVCacheCompressor, sample_kv: torch.Tensor
    ) -> None:
        """Decompressed tensor has same shape as original."""
        compressed = compressor_3bit.compress(sample_kv)
        decompressed = compressor_3bit.decompress(compressed)
        assert decompressed.shape == sample_kv.shape

    def test_dtype_preserved(
        self, compressor_3bit: KVCacheCompressor, sample_kv: torch.Tensor
    ) -> None:
        """Decompressed tensor has requested dtype."""
        compressed = compressor_3bit.compress(sample_kv)
        decompressed = compressor_3bit.decompress(compressed, dtype=torch.float16)
        assert decompressed.dtype == torch.float16

    def test_compressed_fields(
        self, compressor_3bit: KVCacheCompressor, sample_kv: torch.Tensor
    ) -> None:
        """Compressed cache has correct field types."""
        compressed = compressor_3bit.compress(sample_kv)
        assert compressed.magnitudes.dtype == torch.float16
        assert compressed.quantized_directions.dtype == torch.uint8
        assert compressed.shape == sample_kv.shape
        assert compressed.bits == 3
        # No QJL
        assert compressed.qjl_signs is None

    def test_3bit_cosine_similarity(
        self, compressor_3bit: KVCacheCompressor, sample_kv: torch.Tensor
    ) -> None:
        """3-bit PolarQuant achieves meaningful cosine similarity (>0.75).

        For random Gaussian data, 3-bit uniform quantization (7 levels) of
        unit direction vectors yields cos_sim ≈ 0.80-0.85. Real LLM KV
        cache vectors have more structure and achieve higher fidelity.
        The paper's 99.5% claim refers to downstream *attention* fidelity,
        not raw vector cosine similarity.
        """
        compressed = compressor_3bit.compress(sample_kv)
        decompressed = compressor_3bit.decompress(compressed)

        orig = sample_kv.float().reshape(-1, sample_kv.shape[-1])
        recon = decompressed.float().reshape(-1, sample_kv.shape[-1])
        cos_sim = torch.nn.functional.cosine_similarity(orig, recon, dim=-1).mean()
        assert cos_sim.item() > 0.75, f"3-bit cosine similarity too low: {cos_sim.item():.4f}"

    def test_4bit_cosine_similarity(
        self, compressor_4bit: KVCacheCompressor, sample_kv: torch.Tensor
    ) -> None:
        """4-bit PolarQuant achieves high cosine similarity (>0.90).

        For random Gaussian data, 4-bit uniform quantization (15 levels) of
        unit direction vectors yields cos_sim ≈ 0.92-0.96. Real LLM KV
        cache vectors achieve higher fidelity due to non-uniform structure.
        """
        compressed = compressor_4bit.compress(sample_kv)
        decompressed = compressor_4bit.decompress(compressed)

        orig = sample_kv.float().reshape(-1, sample_kv.shape[-1])
        recon = decompressed.float().reshape(-1, sample_kv.shape[-1])
        cos_sim = torch.nn.functional.cosine_similarity(orig, recon, dim=-1).mean()
        assert cos_sim.item() > 0.90, f"4-bit cosine similarity too low: {cos_sim.item():.4f}"

    def test_4bit_better_than_3bit(
        self,
        compressor_3bit: KVCacheCompressor,
        compressor_4bit: KVCacheCompressor,
        sample_kv: torch.Tensor,
    ) -> None:
        """4-bit should achieve better fidelity than 3-bit."""
        report_3 = compressor_3bit.compression_fidelity_report(sample_kv)
        report_4 = compressor_4bit.compression_fidelity_report(sample_kv)
        assert report_4["cosine_similarity"] >= report_3["cosine_similarity"]

    def test_deterministic(
        self, compressor_3bit: KVCacheCompressor, sample_kv: torch.Tensor
    ) -> None:
        """Compression is deterministic (same input → same output)."""
        c1 = compressor_3bit.compress(sample_kv)
        c2 = compressor_3bit.compress(sample_kv)
        assert torch.equal(c1.quantized_directions, c2.quantized_directions)
        assert torch.equal(c1.magnitudes, c2.magnitudes)


# ---------------------------------------------------------------------------
# QJL tests
# ---------------------------------------------------------------------------
class TestQJL:
    """Tests for QJL sign-bit projections."""

    @pytest.fixture()
    def compressor_with_qjl(self) -> KVCacheCompressor:
        """3-bit compressor with QJL enabled."""
        return KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=64))

    @pytest.fixture()
    def sample_kv(self) -> torch.Tensor:
        """Sample KV cache tensor."""
        torch.manual_seed(42)
        return torch.randn(1, 4, 16, 64, dtype=torch.float16)

    def test_qjl_signs_present(
        self, compressor_with_qjl: KVCacheCompressor, sample_kv: torch.Tensor
    ) -> None:
        """QJL sign bits are stored when qjl_dim > 0."""
        compressed = compressor_with_qjl.compress(sample_kv)
        assert compressed.qjl_signs is not None
        # Packed into bytes: 64 bits → 8 bytes
        assert compressed.qjl_signs.shape[-1] == 8

    def test_qjl_signs_dtype(
        self, compressor_with_qjl: KVCacheCompressor, sample_kv: torch.Tensor
    ) -> None:
        """QJL signs are packed as uint8."""
        compressed = compressor_with_qjl.compress(sample_kv)
        assert compressed.qjl_signs.dtype == torch.uint8

    def test_compressed_attention_shape(
        self, compressor_with_qjl: KVCacheCompressor, sample_kv: torch.Tensor
    ) -> None:
        """Compressed attention returns correct shape."""
        compressed_k = compressor_with_qjl.compress(sample_kv)
        query = torch.randn(1, 4, 8, 64, dtype=torch.float16)  # seq_q=8
        scores = compressor_with_qjl.compressed_attention(query, compressed_k)
        # (batch=1, heads=4, seq_q=8, seq_k=16)
        assert scores.shape == (1, 4, 8, 16)

    def test_compressed_attention_without_qjl_raises(self) -> None:
        """compressed_attention raises if QJL signs not available."""
        compressor = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        kv = torch.randn(1, 4, 16, 64, dtype=torch.float16)
        compressed = compressor.compress(kv)
        query = torch.randn(1, 4, 8, 64, dtype=torch.float16)

        with pytest.raises(ValueError, match="QJL projections"):
            compressor.compressed_attention(query, compressed)

    def test_compressed_attention_correlates_with_standard(
        self, compressor_with_qjl: KVCacheCompressor, sample_kv: torch.Tensor
    ) -> None:
        """Compressed attention scores should correlate with standard attention."""
        compressed_k = compressor_with_qjl.compress(sample_kv)
        query = torch.randn(1, 4, 8, 64, dtype=torch.float16)

        # Compressed attention
        approx_scores = compressor_with_qjl.compressed_attention(query, compressed_k)

        # Standard attention (query @ key^T)
        standard_scores = torch.matmul(query.float(), sample_kv.float().transpose(-2, -1))

        # Check correlation (not exact match — QJL is approximate)
        approx_flat = approx_scores.flatten()
        standard_flat = standard_scores.flatten()
        correlation = torch.corrcoef(torch.stack([approx_flat, standard_flat]))[0, 1]

        assert correlation.item() > 0.5, (
            f"Compressed attention poorly correlated with standard: {correlation.item():.4f}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_tensor(self) -> None:
        """Compressing a zero tensor doesn't produce NaN."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        zeros = torch.zeros(1, 2, 4, 32, dtype=torch.float16)
        compressed = c.compress(zeros)
        decompressed = c.decompress(compressed)
        assert not torch.isnan(decompressed).any()
        assert not torch.isinf(decompressed).any()

    def test_single_vector(self) -> None:
        """Compressing a single vector works."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        single = torch.randn(1, 1, 1, 32, dtype=torch.float16)
        compressed = c.compress(single)
        decompressed = c.decompress(compressed)
        assert decompressed.shape == single.shape

    def test_large_values(self) -> None:
        """Large magnitude values are handled correctly."""
        c = KVCacheCompressor(PolarQuantConfig(bits=4, qjl_dim=0))
        large = torch.randn(1, 2, 4, 64, dtype=torch.float16) * 1000
        compressed = c.compress(large)
        decompressed = c.decompress(compressed)
        # Magnitude should be preserved (stored as FP16)
        orig_norms = torch.norm(large.float(), dim=-1)
        recon_norms = torch.norm(decompressed.float(), dim=-1)
        norm_ratio = (recon_norms / orig_norms.clamp(min=1e-10)).mean()
        assert 0.8 < norm_ratio.item() < 1.2

    def test_different_head_dims(self) -> None:
        """Works with various head dimensions."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        for head_dim in [32, 64, 128, 256]:
            tensor = torch.randn(1, 2, 4, head_dim, dtype=torch.float16)
            compressed = c.compress(tensor)
            decompressed = c.decompress(compressed)
            assert decompressed.shape == tensor.shape

    def test_bfloat16_input(self) -> None:
        """Works with bfloat16 input tensors."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        tensor = torch.randn(1, 2, 4, 64).to(torch.bfloat16)
        compressed = c.compress(tensor)
        decompressed = c.decompress(compressed, dtype=torch.bfloat16)
        assert decompressed.dtype == torch.bfloat16
        assert decompressed.shape == tensor.shape


# ---------------------------------------------------------------------------
# CompressedKVCache.cat tests
# ---------------------------------------------------------------------------
class TestCompressedKVCacheCat:
    """Tests for concatenating compressed caches."""

    def test_cat_two_caches(self) -> None:
        """Concatenating two caches along batch dimension."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        t1 = torch.randn(1, 2, 4, 64, dtype=torch.float16)
        t2 = torch.randn(1, 2, 4, 64, dtype=torch.float16)
        c1 = c.compress(t1)
        c2 = c.compress(t2)
        merged = CompressedKVCache.cat([c1, c2], dim=0)
        assert merged.shape[0] == 2

    def test_cat_single_returns_same(self) -> None:
        """Concatenating a single cache returns the same object."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        t = torch.randn(1, 2, 4, 64, dtype=torch.float16)
        compressed = c.compress(t)
        result = CompressedKVCache.cat([compressed], dim=0)
        assert result is compressed

    def test_cat_empty_raises(self) -> None:
        """Concatenating empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot concatenate"):
            CompressedKVCache.cat([], dim=0)

    def test_cat_wrong_dim_raises(self) -> None:
        """Concatenating along non-batch dimension raises."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        t = torch.randn(1, 2, 4, 64, dtype=torch.float16)
        compressed = c.compress(t)
        with pytest.raises(NotImplementedError):
            CompressedKVCache.cat([compressed, compressed], dim=1)

    def test_cat_with_qjl(self) -> None:
        """Concatenation preserves QJL sign bits."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=64))
        t1 = torch.randn(1, 2, 4, 64, dtype=torch.float16)
        t2 = torch.randn(1, 2, 4, 64, dtype=torch.float16)
        c1 = c.compress(t1)
        c2 = c.compress(t2)
        merged = CompressedKVCache.cat([c1, c2], dim=0)
        assert merged.qjl_signs is not None
        assert merged.qjl_signs.shape[0] == 2


# ---------------------------------------------------------------------------
# Memory reduction diagnostics
# ---------------------------------------------------------------------------
class TestDiagnostics:
    """Tests for diagnostic and reporting methods."""

    def test_memory_reduction_3bit(self) -> None:
        """3-bit without QJL should give ~5x reduction."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        ratio = c.memory_reduction_ratio()
        assert 4.0 < ratio < 6.0

    def test_memory_reduction_4bit(self) -> None:
        """4-bit without QJL should give ~4x reduction."""
        c = KVCacheCompressor(PolarQuantConfig(bits=4, qjl_dim=0))
        ratio = c.memory_reduction_ratio()
        assert 3.0 < ratio < 5.0

    def test_memory_reduction_with_qjl(self) -> None:
        """QJL adds some overhead, reducing the ratio."""
        c_no_qjl = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        c_qjl = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=64))
        assert c_no_qjl.memory_reduction_ratio() > c_qjl.memory_reduction_ratio()

    def test_fidelity_report_keys(self) -> None:
        """Fidelity report contains expected keys."""
        c = KVCacheCompressor(PolarQuantConfig(bits=3, qjl_dim=0))
        tensor = torch.randn(1, 2, 4, 64, dtype=torch.float16)
        report = c.compression_fidelity_report(tensor)
        assert "cosine_similarity" in report
        assert "relative_error" in report
        assert "compression_ratio" in report

    def test_fidelity_report_values(self) -> None:
        """Fidelity report values are in valid ranges."""
        c = KVCacheCompressor(PolarQuantConfig(bits=4, qjl_dim=0))
        tensor = torch.randn(1, 4, 16, 128, dtype=torch.float16)
        report = c.compression_fidelity_report(tensor)
        assert 0.0 < report["cosine_similarity"] <= 1.0
        assert report["relative_error"] >= 0.0
        assert report["compression_ratio"] > 1.0
