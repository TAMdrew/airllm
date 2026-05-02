"""Tests for the unified quantization module."""

import json

import pytest

from air_llm.airllm.quantization import (
    QuantizationMethod,
    detect_quantization,
    get_available_methods,
    is_awq_available,
    is_bitsandbytes_available,
    is_gptq_available,
    is_turboquant_available,
    parse_quantization_method,
    validate_quantization_backend,
)


class TestQuantizationMethod:
    """Tests for the :class:`QuantizationMethod` enum."""

    def test_all_values_are_strings(self):
        for method in QuantizationMethod:
            assert isinstance(method.value, str)

    def test_none_method_exists(self):
        assert QuantizationMethod.NONE.value == "none"

    def test_awq_method_exists(self):
        assert QuantizationMethod.AWQ.value == "awq"

    def test_gptq_method_exists(self):
        assert QuantizationMethod.GPTQ.value == "gptq"

    def test_4bit_method_exists(self):
        assert QuantizationMethod.BITSANDBYTES_4BIT.value == "4bit"

    def test_8bit_method_exists(self):
        assert QuantizationMethod.BITSANDBYTES_8BIT.value == "8bit"

    def test_pre_quantized_method_exists(self):
        assert QuantizationMethod.PRE_QUANTIZED.value == "pre_quantized"

    def test_turboquant_kv_method_exists(self):
        assert QuantizationMethod.TURBOQUANT_KV.value == "turboquant"

    def test_enum_has_eight_members(self):
        assert len(QuantizationMethod) == 8


class TestDetectQuantization:
    """Tests for :func:`detect_quantization`."""

    def test_no_config_returns_none(self, tmp_path):
        result = detect_quantization(str(tmp_path))
        assert result == QuantizationMethod.NONE

    def test_awq_config(self, tmp_path):
        config = {"quantization_config": {"quant_method": "awq", "bits": 4}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = detect_quantization(str(tmp_path))
        assert result == QuantizationMethod.AWQ

    def test_gptq_config(self, tmp_path):
        config = {"quantization_config": {"quant_method": "gptq", "bits": 4}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = detect_quantization(str(tmp_path))
        assert result == QuantizationMethod.GPTQ

    def test_exl2_config_maps_to_gptq(self, tmp_path):
        config = {"quantization_config": {"quant_method": "exl2"}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = detect_quantization(str(tmp_path))
        assert result == QuantizationMethod.GPTQ

    def test_bitsandbytes_4bit_config(self, tmp_path):
        config = {"quantization_config": {"quant_method": "bitsandbytes", "bits": 4}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = detect_quantization(str(tmp_path))
        assert result == QuantizationMethod.BITSANDBYTES_4BIT

    def test_bitsandbytes_8bit_config(self, tmp_path):
        config = {"quantization_config": {"quant_method": "bitsandbytes", "bits": 8}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = detect_quantization(str(tmp_path))
        assert result == QuantizationMethod.BITSANDBYTES_8BIT

    def test_plain_config_returns_none(self, tmp_path):
        config = {"model_type": "llama", "hidden_size": 4096}
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = detect_quantization(str(tmp_path))
        assert result == QuantizationMethod.NONE


class TestParseQuantizationMethod:
    """Tests for :func:`parse_quantization_method`."""

    def test_none_string(self):
        assert parse_quantization_method("none") == QuantizationMethod.NONE

    def test_4bit_string(self):
        assert parse_quantization_method("4bit") == QuantizationMethod.BITSANDBYTES_4BIT

    def test_8bit_string(self):
        assert parse_quantization_method("8bit") == QuantizationMethod.BITSANDBYTES_8BIT

    def test_awq_string(self):
        assert parse_quantization_method("awq") == QuantizationMethod.AWQ

    def test_gptq_string(self):
        assert parse_quantization_method("gptq") == QuantizationMethod.GPTQ

    def test_turboquant_string(self):
        assert parse_quantization_method("turboquant") == QuantizationMethod.TURBOQUANT_KV

    def test_turboquant_kv_alias(self):
        assert parse_quantization_method("turboquant_kv") == QuantizationMethod.TURBOQUANT_KV

    def test_exl2_maps_to_gptq(self):
        assert parse_quantization_method("exl2") == QuantizationMethod.GPTQ

    def test_bitsandbytes_alias(self):
        assert parse_quantization_method("bitsandbytes") == QuantizationMethod.BITSANDBYTES_4BIT

    def test_bnb_alias(self):
        assert parse_quantization_method("bnb") == QuantizationMethod.BITSANDBYTES_4BIT

    def test_case_insensitive(self):
        assert parse_quantization_method("AWQ") == QuantizationMethod.AWQ

    def test_whitespace_stripped(self):
        assert parse_quantization_method("  gptq  ") == QuantizationMethod.GPTQ

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown quantization method"):
            parse_quantization_method("invalid_method")


class TestGetAvailableMethods:
    """Tests for :func:`get_available_methods`."""

    def test_returns_list(self):
        result = get_available_methods()
        assert isinstance(result, list)

    def test_always_includes_none(self):
        result = get_available_methods()
        assert "none" in result

    def test_always_includes_pre_quantized(self):
        result = get_available_methods()
        assert "pre_quantized" in result

    def test_result_is_sorted(self):
        result = get_available_methods()
        assert result == sorted(result)


class TestAvailabilityChecks:
    """Tests for backend availability check functions."""

    def test_awq_check_returns_bool(self):
        assert isinstance(is_awq_available(), bool)

    def test_gptq_check_returns_bool(self):
        assert isinstance(is_gptq_available(), bool)

    def test_bitsandbytes_check_returns_bool(self):
        assert isinstance(is_bitsandbytes_available(), bool)

    def test_turboquant_check_returns_bool(self):
        assert isinstance(is_turboquant_available(), bool)


class TestValidateQuantizationBackend:
    """Tests for :func:`validate_quantization_backend`."""

    def test_none_does_not_raise(self):
        validate_quantization_backend(QuantizationMethod.NONE)

    def test_pre_quantized_does_not_raise(self):
        validate_quantization_backend(QuantizationMethod.PRE_QUANTIZED)

    def test_awq_raises_if_not_available(self):
        if not is_awq_available():
            with pytest.raises(ImportError, match="AutoAWQ"):
                validate_quantization_backend(QuantizationMethod.AWQ)

    def test_gptq_raises_if_not_available(self):
        if not is_gptq_available():
            with pytest.raises(ImportError, match="AutoGPTQ"):
                validate_quantization_backend(QuantizationMethod.GPTQ)

    def test_bitsandbytes_raises_if_not_available(self):
        if not is_bitsandbytes_available():
            with pytest.raises(ImportError, match="bitsandbytes"):
                validate_quantization_backend(QuantizationMethod.BITSANDBYTES_4BIT)

    def test_turboquant_raises_if_not_available(self):
        if not is_turboquant_available():
            with pytest.raises(ImportError, match="TurboQuant"):
                validate_quantization_backend(QuantizationMethod.TURBOQUANT_KV)
