"""Tests for the direct model weight downloader."""

import pytest

from air_llm.airllm.io.downloader import (
    _HF_RESOLVE_URL,
    _has_model_files,
    _matches_patterns,
    get_cache_dir,
    resolve_model_path,
)


class TestGetCacheDir:
    """Tests for :func:`get_cache_dir`."""

    def test_returns_string(self):
        result = get_cache_dir()
        assert isinstance(result, str)

    def test_respects_env_var(self, monkeypatch):
        monkeypatch.setenv("AIRLLM_CACHE_DIR", "/tmp/test_cache")
        assert get_cache_dir() == "/tmp/test_cache"

    def test_default_contains_airllm(self, monkeypatch):
        monkeypatch.delenv("AIRLLM_CACHE_DIR", raising=False)
        result = get_cache_dir()
        assert "airllm" in result


class TestResolveModelPath:
    """Tests for :func:`resolve_model_path`."""

    def test_local_path_returned_directly(self, tmp_path):
        result = resolve_model_path(str(tmp_path))
        assert result == str(tmp_path)

    def test_nonexistent_local_path_not_treated_as_local(self, monkeypatch):
        # A non-existent path should not be returned as-is.
        # In offline mode (with hf-hub mocked away) it raises FileNotFoundError.
        monkeypatch.setattr("air_llm.airllm.io.downloader.OFFLINE_MODE", True)
        # Ensure huggingface_hub import fails so we hit the OFFLINE_MODE guard
        import sys

        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(FileNotFoundError):
            resolve_model_path("/nonexistent/path/to/model")


class TestHfResolveUrl:
    """Tests for the HuggingFace resolve URL template."""

    def test_url_format(self):
        url = _HF_RESOLVE_URL.format(
            repo_id="google/gemma-4-12b",
            revision="main",
            filename="config.json",
        )
        assert "google/gemma-4-12b" in url
        assert "config.json" in url
        assert "main" in url

    def test_url_starts_with_https(self):
        url = _HF_RESOLVE_URL.format(
            repo_id="meta/llama-3",
            revision="main",
            filename="model.safetensors",
        )
        assert url.startswith("https://")


class TestMatchesPatterns:
    """Tests for :func:`_matches_patterns`."""

    def test_exact_match(self):
        assert _matches_patterns("config.json", ["config.json"])

    def test_suffix_match(self):
        assert _matches_patterns("model-00001.safetensors", [".safetensors"])

    def test_substring_match(self):
        assert _matches_patterns("tokenizer_config.json", ["tokenizer"])

    def test_no_match(self):
        assert not _matches_patterns("random_file.txt", [".safetensors", "config.json"])


class TestHasModelFiles:
    """Tests for :func:`_has_model_files`."""

    def test_empty_dir_returns_false(self, tmp_path):
        assert not _has_model_files(str(tmp_path))

    def test_dir_with_safetensors_returns_true(self, tmp_path):
        (tmp_path / "model.safetensors").write_text("")
        assert _has_model_files(str(tmp_path))

    def test_dir_with_bin_returns_true(self, tmp_path):
        (tmp_path / "model.bin").write_text("")
        assert _has_model_files(str(tmp_path))

    def test_dir_with_only_json_returns_false(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert not _has_model_files(str(tmp_path))

    def test_nonexistent_dir_returns_false(self):
        assert not _has_model_files("/nonexistent/path")
