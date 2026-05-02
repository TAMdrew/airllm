"""Direct model weight downloader — no huggingface-hub dependency required.

Supports downloading safetensor model weights directly via HTTPS from
HuggingFace Hub (or any HTTP endpoint) using only Python stdlib.

This eliminates the mandatory huggingface-hub dependency for users who
already have model weights locally or want minimal dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ..constants import OFFLINE_MODE

logger = logging.getLogger(__name__)

# HuggingFace Hub download URL template
_HF_RESOLVE_URL = "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
_HF_API_URL = "https://huggingface.co/api/models/{repo_id}"

# Default cache directory (mirrors HF cache structure)
_DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "airllm",
)

# Download chunk size: 8 MB
_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024

# Regex for valid HuggingFace repo IDs (org/model or standalone model names)
_VALID_REPO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_./-]+$")

# Essential file patterns for model download
_ESSENTIAL_FILE_PATTERNS: list[str] = [
    "config.json",
    "tokenizer",
    ".safetensors",
    "generation_config",
]


def get_cache_dir() -> str:
    """Get the model cache directory, respecting ``AIRLLM_CACHE_DIR`` env var.

    Returns:
        Path to the cache directory.
    """
    return os.environ.get("AIRLLM_CACHE_DIR", _DEFAULT_CACHE_DIR)


def _validate_repo_id(repo_id: str) -> None:
    """Validate that a repo_id contains only safe characters.

    Prevents SSRF and token leakage by ensuring repo_id cannot inject
    unexpected URL components.

    Args:
        repo_id: The HuggingFace repository ID to validate.

    Raises:
        ValueError: If repo_id contains invalid characters.
    """
    if not _VALID_REPO_ID_PATTERN.match(repo_id):
        raise ValueError(
            f"Invalid repo_id '{repo_id}'. "
            "Must match pattern ^[a-zA-Z0-9_./-]+$ "
            "(only alphanumeric, dots, underscores, hyphens, and slashes)."
        )


def download_file(
    url: str,
    dest_path: str,
    *,
    token: str | None = None,
    chunk_size: int = _DOWNLOAD_CHUNK_SIZE,
    expected_sha256: str | None = None,
) -> str:
    """Download a file from a URL to a local path with progress logging.

    Args:
        url: The URL to download from.
        dest_path: Local file path to save to.
        token: Optional auth token for gated models.
        chunk_size: Download chunk size in bytes.
        expected_sha256: Optional SHA-256 hash to verify download integrity.

    Returns:
        The local file path.

    Raises:
        HTTPError: If the download fails.
        ConnectionError: If the server is unreachable.
        ValueError: If SHA-256 verification fails.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Skip if file already exists
    if os.path.exists(dest_path):
        logger.info("File already cached: %s", dest_path)
        return dest_path

    temp_path = dest_path + ".tmp"
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    logger.info("Downloading: %s", url)
    request = Request(url, headers=headers)

    sha256_hash = hashlib.sha256()
    try:
        with urlopen(request) as response, open(temp_path, "wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                sha256_hash.update(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = (downloaded / total) * 100
                    logger.info("  %.1f%% (%d / %d bytes)", pct, downloaded, total)
    except Exception:
        # Clean up partial download on any failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    # Verify download integrity if expected hash was provided
    computed_sha = sha256_hash.hexdigest()
    if expected_sha256 is not None:
        if computed_sha != expected_sha256:
            os.remove(temp_path)
            raise ValueError(
                f"SHA-256 mismatch for {url}: "
                f"expected {expected_sha256}, got {computed_sha}"
            )
        logger.info("SHA-256 verified: %s", dest_path)
    else:
        logger.debug(
            "No expected SHA-256 provided for %s — skipping integrity check "
            "(sha256=%s).",
            dest_path,
            computed_sha,
        )

    # Atomic rename to prevent partial-file reads
    shutil.move(temp_path, dest_path)
    logger.info("Downloaded: %s", dest_path)
    return dest_path


def fetch_model_file_list(
    repo_id: str,
    *,
    revision: str = "main",
    token: str | None = None,
) -> list[str]:
    """Fetch the list of files in a HuggingFace model repository.

    Args:
        repo_id: HuggingFace model ID (e.g., ``"google/gemma-4-12b"``).
        revision: Git revision (branch, tag, or commit hash).
        token: Optional auth token for gated models.

    Returns:
        List of filenames in the repository.

    Raises:
        HTTPError: If the API request fails.
    """
    _validate_repo_id(repo_id)
    url = _HF_API_URL.format(repo_id=repo_id)
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    with urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))

    siblings = data.get("siblings", [])
    return [s["rfilename"] for s in siblings]


def _matches_patterns(filename: str, patterns: list[str]) -> bool:
    """Check if a filename matches any of the given patterns.

    Args:
        filename: The filename to check.
        patterns: List of substring/suffix patterns to match against.

    Returns:
        True if the filename matches at least one pattern.
    """
    for pattern in patterns:
        if pattern in filename or filename.endswith(pattern.lstrip("*")):
            return True
    return False


def download_model(
    repo_id: str,
    *,
    revision: str = "main",
    token: str | None = None,
    cache_dir: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Download a model from HuggingFace Hub via direct HTTPS.

    This function does NOT require the ``huggingface-hub`` package.
    It downloads files directly using Python's ``urllib``.

    Args:
        repo_id: HuggingFace model ID (e.g., ``"google/gemma-4-12b"``).
        revision: Git revision (default: ``"main"``).
        token: Auth token for gated models.
        cache_dir: Local directory to cache downloads.
        allow_patterns: List of glob patterns to filter files
            (e.g., ``["*.safetensors", "*.json"]``).

    Returns:
        Local directory path containing the downloaded model files.

    Raises:
        PermissionError: If the model is gated and no valid token is provided.
        HTTPError: If any download request fails.
    """
    _validate_repo_id(repo_id)

    if cache_dir is None:
        cache_dir = get_cache_dir()

    # Create model-specific cache directory
    model_dir = os.path.join(cache_dir, repo_id.replace("/", "--"))
    os.makedirs(model_dir, exist_ok=True)

    # Fetch file list from HF API
    try:
        files = fetch_model_file_list(repo_id, revision=revision, token=token)
    except HTTPError as e:
        if e.code == 401:
            msg = (
                f"Access denied for '{repo_id}'. This is a gated model. "
                f"Provide a token via hf_token parameter or HF_TOKEN env var. "
                f"Get a token at: https://huggingface.co/settings/tokens"
            )
            raise PermissionError(msg) from e
        raise

    # Filter files by patterns
    if allow_patterns is None:
        allow_patterns = _ESSENTIAL_FILE_PATTERNS

    filtered_files = [f for f in files if _matches_patterns(f, allow_patterns)]

    # Download each matching file
    for filename in filtered_files:
        url = _HF_RESOLVE_URL.format(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
        )
        dest = os.path.join(model_dir, filename)
        download_file(url, dest, token=token)

    return model_dir


def resolve_model_path(
    model_id_or_path: str,
    *,
    token: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """Resolve a model ID or local path to a local directory.

    If ``model_id_or_path`` is a local directory, returns it directly.
    If it's a HuggingFace model ID, downloads it (or uses cache).

    Resolution order:
    1. Local path (if exists as directory)
    2. AirLLM cache (``AIRLLM_CACHE_DIR``)
    3. HuggingFace Hub cache (if ``huggingface-hub`` installed)
    4. Direct HTTPS download (no ``huggingface-hub`` needed)

    Args:
        model_id_or_path: Local path or HuggingFace model ID.
        token: Auth token for gated models.
        cache_dir: Cache directory for downloads.

    Returns:
        Local directory path containing model files.

    Raises:
        FileNotFoundError: If model not found and offline mode is enabled.
        PermissionError: If model is gated and no valid token is provided.
    """
    # 1. Check if it's already a local path
    if os.path.isdir(model_id_or_path):
        logger.info("Using local model path: %s", model_id_or_path)
        return model_id_or_path

    # 2. Check AirLLM cache
    if cache_dir is None:
        cache_dir = get_cache_dir()
    cached_path = os.path.join(cache_dir, model_id_or_path.replace("/", "--"))
    if os.path.isdir(cached_path) and _has_model_files(cached_path):
        logger.info("Using cached model: %s", cached_path)
        return cached_path

    # 3. Try huggingface-hub if installed (for HF cache compatibility)
    try:
        from huggingface_hub import snapshot_download

        logger.info("Using huggingface-hub for download: %s", model_id_or_path)
        return snapshot_download(
            model_id_or_path,
            token=token,
            cache_dir=cache_dir,
        )
    except ImportError:
        pass

    # 4. Direct HTTPS download (no huggingface-hub needed)
    if OFFLINE_MODE:
        msg = (
            f"Model '{model_id_or_path}' not found locally and OFFLINE_MODE is enabled. "
            f"Download the model first or provide a local path."
        )
        raise FileNotFoundError(msg)

    logger.info("Downloading via direct HTTPS (no huggingface-hub): %s", model_id_or_path)
    return download_model(model_id_or_path, token=token, cache_dir=cache_dir)


def _has_model_files(directory: str) -> bool:
    """Check if a directory contains model weight files.

    Args:
        directory: Path to check.

    Returns:
        True if the directory contains ``.safetensors`` or ``.bin`` files.
    """
    try:
        return any(f.endswith(".safetensors") or f.endswith(".bin") for f in os.listdir(directory))
    except OSError:
        return False
