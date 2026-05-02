"""Utility functions for model splitting, compression, and memory management.

This module provides helpers to:
- Split large model checkpoints into per-layer safetensor shards.
- Compress / decompress layer weights using bitsandbytes NF4 or 8-bit quantization.
- Manage GPU and system memory.
- Locate or download model checkpoints from HuggingFace Hub (one-time download).

All operations run entirely locally. HuggingFace Hub is used only as a
download source for model weights — no cloud inference APIs are called.
"""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
import shutil
import time
from glob import glob
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from ..constants import (
    BYTES_PER_GB,
    COMPRESSION_RATIO_4BIT,
    COMPRESSION_RATIO_8BIT,
    DEFAULT_COMPRESSION_BLOCK_SIZE_4BIT,
    DEFAULT_COMPRESSION_BLOCK_SIZE_8BIT,
    IS_ON_MAC_OS,
    OFFLINE_MODE,
    PYTORCH_INDEX_FILE,
    SAFETENSORS_INDEX_FILE,
    SPLITTED_MODEL_DIR_NAME,
)
from ..persist import ModelPersister

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: bitsandbytes
# ---------------------------------------------------------------------------
try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
except ImportError:
    bnb = None  # type: ignore[assignment]
    bitsandbytes_installed = False

# ---------------------------------------------------------------------------
# Optional dependency: huggingface-hub
# Falls back to direct HTTPS download via downloader.py if not installed.
# ---------------------------------------------------------------------------
try:
    import huggingface_hub

    _huggingface_hub_installed = True
except ImportError:
    huggingface_hub = None  # type: ignore[assignment]
    _huggingface_hub_installed = False


# ---------------------------------------------------------------------------
# Shard download helper (works with or without huggingface-hub)
# ---------------------------------------------------------------------------
def _download_shard(
    repo_id: str,
    to_load: str,
    *,
    hf_token: str | None = None,
) -> None:
    """Download a single shard file, using huggingface-hub if available.

    Falls back to direct HTTPS download via :mod:`airllm.downloader` when
    ``huggingface-hub`` is not installed.

    Args:
        repo_id: HuggingFace model repo ID.
        to_load: Local path where the shard is expected.
        hf_token: Optional authentication token.
    """
    filename = os.path.basename(to_load)
    if _huggingface_hub_installed:
        huggingface_hub.snapshot_download(
            repo_id,
            allow_patterns=filename,
            token=hf_token,
        )
    else:
        from ..io.downloader import download_file

        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        download_file(url, to_load, token=hf_token)


# ---------------------------------------------------------------------------
# Quant-state serialisation helper
# ---------------------------------------------------------------------------
def save_quant_state_to_dict(
    quant_state: Any,
    packed: bool = True,
) -> dict[str, Any]:
    """Serialize a bitsandbytes ``QuantState`` to a dict of tensors/strings.

    This is a replacement for ``QuantState.as_dict(packed=True)`` that works
    around upstream bugs in earlier bitsandbytes releases.

    Args:
        quant_state: A ``bitsandbytes.functional.QuantState`` instance.
        packed: If ``True`` (default), pack non-tensor metadata into a single
            tensor via ``bnb.utils.pack_dict_to_tensor``.

    Returns:
        A dictionary suitable for persisting alongside the quantised weight.
    """
    qs_dict: dict[str, Any] = {
        "quant_type": quant_state.quant_type,
        "absmax": quant_state.absmax,
        "blocksize": quant_state.blocksize,
        "quant_map": quant_state.code,
        "dtype": str(quant_state.dtype).strip("torch."),
        "shape": tuple(quant_state.shape),
    }
    if quant_state.nested:
        qs_dict.update(
            {
                "nested_absmax": quant_state.state2.absmax,
                "nested_blocksize": quant_state.state2.blocksize,
                "nested_quant_map": quant_state.state2.code,
                "nested_dtype": str(quant_state.state2.dtype).strip("torch."),
                "nested_offset": quant_state.offset.item(),
            }
        )
    if not packed:
        return qs_dict

    qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
    non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
    packed_key = "quant_state." + "bitsandbytes__" + quant_state.quant_type
    qs_packed_dict[packed_key] = bnb.utils.pack_dict_to_tensor(non_tensor_dict)
    return qs_packed_dict


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class NotEnoughSpaceException(Exception):
    """Raised when the target filesystem lacks space for split model shards."""


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------
def clean_memory() -> None:
    """Release unreferenced Python objects and free GPU/system memory.

    On Linux, ``malloc_trim`` is called to return freed heap pages to the OS.
    On other platforms (macOS, Windows) this step is silently skipped.
    """
    gc.collect()

    if not IS_ON_MAC_OS:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            # Expected on non-glibc platforms (macOS, musl-linux, Windows).
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Layer compression / decompression
# ---------------------------------------------------------------------------
def uncompress_layer_state_dict(
    layer_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Decompress a layer state dict that was quantized with 4-bit or 8-bit.

    If the state dict is uncompressed (no ``4bit`` / ``8bit`` keys), it is
    returned unchanged.

    Args:
        layer_state_dict: The (potentially compressed) state dict.

    Returns:
        The decompressed state dict with full-precision weights.
    """
    if any("4bit" in k for k in layer_state_dict):
        uncompressed: dict[str, torch.Tensor] = {}
        for k, v in layer_state_dict.items():
            if "4bit" not in k:
                quant_state_dict = {
                    kk[len(k) :]: kv
                    for kk, kv in layer_state_dict.items()
                    if kk.startswith(k) and k != kk
                }
                quant_state = bnb.functional.QuantState.from_dict(
                    qs_dict=quant_state_dict, device="cuda"
                )
                dqv = bnb.functional.dequantize_nf4(v.cuda(), quant_state)
                uncompressed[k] = dqv
        return uncompressed

    if any("8bit" in k for k in layer_state_dict):
        uncompressed = {}
        for k, v in layer_state_dict.items():
            if "8bit" not in k:
                absmax = layer_state_dict[k + ".8bit.absmax"]
                code = layer_state_dict[k + ".8bit.code"]
                dqv = bnb.functional.dequantize_blockwise(
                    v.cuda(),
                    bnb.functional.QuantState(
                        absmax=absmax.cuda(),
                        code=code.cuda(),
                        blocksize=DEFAULT_COMPRESSION_BLOCK_SIZE_8BIT,
                        dtype=torch.float16,
                    ),
                )
                uncompressed[k] = dqv
        return uncompressed

    return layer_state_dict


def compress_layer_state_dict(
    layer_state_dict: dict[str, torch.Tensor],
    compression: str | None = None,
) -> dict[str, torch.Tensor]:
    """Compress a layer state dict using NF4 or 8-bit block-wise quantization.

    Args:
        layer_state_dict: Full-precision layer state dict.
        compression: ``"4bit"``, ``"8bit"``, or ``None`` (no compression).

    Returns:
        The (potentially compressed) state dict.
    """
    if compression == "4bit":
        compressed: dict[str, torch.Tensor] = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_nf4(
                v.cuda(), blocksize=DEFAULT_COMPRESSION_BLOCK_SIZE_4BIT
            )
            compressed[k] = v_quant
            for qs_k, qs_v in save_quant_state_to_dict(quant_state).items():
                compressed[k + ".4bit." + qs_k] = qs_v
        return compressed

    if compression == "8bit":
        compressed = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_blockwise(
                v.cuda(), blocksize=DEFAULT_COMPRESSION_BLOCK_SIZE_8BIT
            )
            absmax = quant_state.absmax.clone().contiguous()
            code = quant_state.code.clone().contiguous()
            compressed[k] = v_quant
            compressed[k + ".8bit.absmax"] = absmax
            compressed[k + ".8bit.code"] = code
        return compressed

    return layer_state_dict


# ---------------------------------------------------------------------------
# Layer loading
# ---------------------------------------------------------------------------
def load_layer(
    local_path: str | Path,
    layer_name: str,
    profiling: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], float]:
    """Load a single layer from the split model directory.

    Args:
        local_path: Path to the directory containing split layer files.
        layer_name: The layer name (used to resolve the file).
        profiling: If ``True``, return a ``(state_dict, elapsed_time)`` tuple
            where *elapsed_time* is the decompression overhead in seconds.

    Returns:
        The layer state dict, or a tuple of ``(state_dict, elapsed_time)``
        when *profiling* is enabled.
    """
    layer_state_dict = ModelPersister.get_model_persister().load_model(layer_name, local_path)

    if profiling:
        t = time.process_time()

    result = uncompress_layer_state_dict(layer_state_dict)

    if profiling:
        elapsed_time = time.process_time() - t
        return result, elapsed_time

    return result


# ---------------------------------------------------------------------------
# Disk space check
# ---------------------------------------------------------------------------
def check_space(
    checkpoint_path: Path,
    layer_shards_saving_path: str | Path | None = None,
    compression: str | None = None,
    splitted_model_dir_name: str = SPLITTED_MODEL_DIR_NAME,
) -> None:
    """Verify that sufficient disk space is available for split model shards.

    Args:
        checkpoint_path: Path to the original model checkpoint.
        layer_shards_saving_path: Optional alternative destination for shards.
        compression: Compression mode (``"4bit"``, ``"8bit"``, or ``None``).
        splitted_model_dir_name: Name of the shards subdirectory.

    Raises:
        NotEnoughSpaceException: When free space is insufficient.
    """
    total_shard_files_size_bytes = sum(os.path.getsize(f) for f in glob(str(checkpoint_path / "*")))

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(
            str(Path(layer_shards_saving_path) / splitted_model_dir_name / "*")
        ):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    if compression == "4bit":
        total_shard_files_size_bytes = int(total_shard_files_size_bytes / COMPRESSION_RATIO_4BIT)
    elif compression == "8bit":
        total_shard_files_size_bytes = int(total_shard_files_size_bytes / COMPRESSION_RATIO_8BIT)

    target_path = checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path
    _total, _used, free = shutil.disk_usage(target_path)

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        raise NotEnoughSpaceException(
            f"Not enough space. Free space under {target_path}: "
            f"{free / BYTES_PER_GB:.02f}GB. "
            f"Model total size: {total_shard_files_size_bytes / BYTES_PER_GB:.02f}GB. "
            f"Existing splits (reusable): "
            f"{total_saved_split_files_size_bytes / BYTES_PER_GB:.02f}GB."
        )


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def remove_real_and_linked_file(to_delete: str | Path) -> None:
    """Remove a file and its symlink target (if the file is a symlink).

    Args:
        to_delete: Path to the file to remove.
    """
    to_delete = str(to_delete)
    target_path: str | None = None
    if os.path.realpath(to_delete) != to_delete:
        target_path = os.path.realpath(to_delete)

    os.remove(to_delete)
    if target_path is not None:
        os.remove(target_path)


# ---------------------------------------------------------------------------
# Core: split model into per-layer shards
# ---------------------------------------------------------------------------
def split_and_save_layers(
    checkpoint_path: str | Path,
    layer_shards_saving_path: str | Path | None = None,
    splitted_model_dir_name: str = SPLITTED_MODEL_DIR_NAME,
    compression: str | None = None,
    layer_names: dict[str, str] | None = None,
    delete_original: bool = False,
    repo_id: str | None = None,
    hf_token: str | None = None,
) -> str:
    """Split a sharded model checkpoint into per-layer safetensor files.

    Args:
        checkpoint_path: Path to the directory containing the original
            model shards.
        layer_shards_saving_path: Optional alternative directory for the
            per-layer output.  Defaults to a subdirectory of *checkpoint_path*.
        splitted_model_dir_name: Name of the output subdirectory.
        compression: ``"4bit"``, ``"8bit"``, or ``None``.
        layer_names: Optional dict mapping logical layer roles
            (``embed``, ``layer_prefix``, ``norm``, ``lm_head``) to their
            weight-name prefixes.
        delete_original: If ``True``, delete original shard files after
            splitting to reclaim disk space.
        repo_id: HuggingFace repo ID for on-demand shard downloading.
        hf_token: HuggingFace API token.

    Returns:
        The string path to the directory containing the saved layer shards.
    """
    if compression is not None:
        if not bitsandbytes_installed:
            raise ImportError(
                "Compression requires bitsandbytes. Install it with: pip install bitsandbytes"
            )
        splitted_model_dir_name = splitted_model_dir_name + "." + compression

    checkpoint_path = Path(checkpoint_path)

    saving_path = checkpoint_path / splitted_model_dir_name
    if layer_shards_saving_path is not None:
        saving_path = Path(layer_shards_saving_path) / splitted_model_dir_name

    # Detect checkpoint format
    safetensors_format = False
    if os.path.exists(checkpoint_path / PYTORCH_INDEX_FILE):
        with open(checkpoint_path / PYTORCH_INDEX_FILE, "rb") as f:
            index: dict[str, str] = json.load(f)["weight_map"]
    else:
        safetensors_format = True
        index_path = checkpoint_path / SAFETENSORS_INDEX_FILE
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Expected {SAFETENSORS_INDEX_FILE} at {checkpoint_path}")
        with open(index_path, "rb") as f:
            index = json.load(f)["weight_map"]

    # Determine layer count and names
    if layer_names is None:
        n_layers = len({int(k.split(".")[2]) for k in index if "model.layers" in k})
    else:
        n_layers = len(
            {
                int(k[len(layer_names["layer_prefix"]) :].split(".")[1])
                for k in index
                if layer_names["layer_prefix"] in k
            }
        )

    if layer_names is None:
        layers = (
            ["model.embed_tokens."]
            + [f"model.layers.{i}." for i in range(n_layers)]
            + ["model.norm.", "lm_head."]
        )
    else:
        layers = (
            [layer_names["embed"]]
            + [f"{layer_names['layer_prefix']}.{i}" for i in range(n_layers)]
            + [layer_names["norm"], layer_names["lm_head"]]
        )
        if "rotary_pos_emb" in layer_names:
            layers = [layer_names["rotary_pos_emb"], *layers]
        layers = [layer + "." for layer in layers]

    # Check if splits already exist
    if os.path.exists(saving_path):
        found_layers = {
            layer: ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)
            for layer in layers
        }
        logger.info("found_layers: %s", found_layers)
        if all(found_layers.values()):
            logger.info("Saved layers already found in %s", saving_path)
            return str(saving_path)
        logger.info(
            "Some layer splits found, some are not — re-saving all layers "
            "in case there are corruptions."
        )

    if not delete_original:
        check_space(
            checkpoint_path,
            layer_shards_saving_path,
            compression,
            splitted_model_dir_name=splitted_model_dir_name,
        )

    shard = 0
    n_shards = len(set(index.values()))
    state_dict: dict[str, torch.Tensor] = {}

    if not os.path.exists(saving_path):
        saving_path.mkdir(parents=True, exist_ok=True)

    single_modelfile: str | None = None

    for layer in tqdm(layers):
        # Determine which shard(s) contain weights for this layer
        shards = [
            int(v.split("-")[1])
            for k, v in index.items()
            if k.startswith(layer) and "-" in v and len(v.split("-")) > 1
        ]
        if len(shards) > 0:
            if max(shards) > shard:
                # Optionally delete the previous shard file
                if delete_original and shard != 0:
                    if not safetensors_format:
                        to_delete = (
                            checkpoint_path
                            / f"pytorch_model-000{shard:02d}-of-000{n_shards:02d}.bin"
                        )
                    else:
                        to_delete = (
                            checkpoint_path
                            / f"model-000{shard:02d}-of-000{n_shards:02d}.safetensors"
                        )
                    logger.info("Deleting original file: %s", to_delete)
                    remove_real_and_linked_file(to_delete)

                shard += 1
                logger.info("Loading shard %d/%d", shard, n_shards)

                if not safetensors_format:
                    to_load = (
                        checkpoint_path / f"pytorch_model-000{shard:02d}-of-000{n_shards:02d}.bin"
                    )
                else:
                    to_load = (
                        checkpoint_path / f"model-000{shard:02d}-of-000{n_shards:02d}.safetensors"
                    )

                if not os.path.exists(to_load):
                    if repo_id is None:
                        raise FileNotFoundError(
                            f"Shard file {to_load} not found and no repo_id provided "
                            "for automatic download."
                        )
                    _download_shard(repo_id, str(to_load), hf_token=hf_token)

                if not safetensors_format:
                    # SECURITY: weights_only=True prevents arbitrary code
                    # execution via pickle deserialization (RCE vulnerability).
                    state_dict.update(torch.load(to_load, map_location="cpu", weights_only=True))
                else:
                    state_dict.update(load_file(to_load, device="cpu"))
        else:
            shard_files = [v for k, v in index.items() if k.startswith(layer)]
            single_modelfile = shard_files[0]
            to_load = checkpoint_path / single_modelfile

            if not os.path.exists(to_load):
                if repo_id is None:
                    raise FileNotFoundError(
                        f"Model file {to_load} not found and no repo_id provided "
                        "for automatic download."
                    )
                _download_shard(repo_id, str(to_load), hf_token=hf_token)

            if not safetensors_format:
                # SECURITY: weights_only=True prevents arbitrary code
                # execution via pickle deserialization (RCE vulnerability).
                state_dict.update(torch.load(to_load, map_location="cpu", weights_only=True))
            else:
                state_dict.update(load_file(to_load, device="cpu"))

        # Extract and compress layer weights
        layer_state_dict = {k: v for k, v in state_dict.items() if k.startswith(layer)}
        layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)

        # Persist if not already saved
        marker_exists = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)
        if not marker_exists:
            ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)

        # Free memory
        for k in layer_state_dict:
            state_dict.pop(k, None)
        del layer_state_dict
        clean_memory()

    # Delete single model file if applicable
    if delete_original and single_modelfile is not None:
        to_delete_path = checkpoint_path / single_modelfile
        logger.info("Deleting original file: %s", to_delete_path)
        remove_real_and_linked_file(to_delete_path)

    return str(saving_path)


def find_or_create_local_splitted_path(
    model_local_path_or_repo_id: str,
    layer_shards_saving_path: str | Path | None = None,
    compression: str | None = None,
    layer_names: dict[str, str] | None = None,
    hf_token: str | None = None,
    delete_original: bool = False,
) -> tuple[Path, str]:
    """Locate (or download) a model checkpoint and split it into layer shards.

    If *model_local_path_or_repo_id* is a local directory containing an index
    file, the model is split in-place.  Otherwise it is treated as a
    HuggingFace Hub repo ID and downloaded first.

    Args:
        model_local_path_or_repo_id: Local path or HuggingFace repo ID.
        layer_shards_saving_path: Optional alternative save location for shards.
        compression: ``"4bit"``, ``"8bit"``, or ``None``.
        layer_names: Optional layer-name mapping dict.
        hf_token: HuggingFace API token.
        delete_original: If ``True``, delete original shard files after
            splitting.

    Returns:
        A tuple of ``(model_local_path, saved_layer_shards_path)``.
    """
    # Try as a local path first
    if os.path.exists(model_local_path_or_repo_id):
        local_path = Path(model_local_path_or_repo_id)
        has_pytorch_index = os.path.exists(local_path / PYTORCH_INDEX_FILE)
        has_safetensor_index = os.path.exists(local_path / SAFETENSORS_INDEX_FILE)

        if has_pytorch_index or has_safetensor_index:
            logger.info("Found index file in %s", model_local_path_or_repo_id)
            return local_path, split_and_save_layers(
                model_local_path_or_repo_id,
                layer_shards_saving_path,
                compression=compression,
                layer_names=layer_names,
                delete_original=delete_original,
            )

        logger.info(
            "Found local directory %s but no model index file. Trying as a HuggingFace repo ID…",
            model_local_path_or_repo_id,
        )

    # Download from HuggingFace Hub or direct HTTPS (one-time; cached locally)
    if OFFLINE_MODE:
        raise FileNotFoundError(
            f"Model '{model_local_path_or_repo_id}' not found locally and "
            "HF_HUB_OFFLINE=1 is set. Either provide a local model path or "
            "unset HF_HUB_OFFLINE to allow downloading from HuggingFace Hub."
        )

    from ..io.downloader import resolve_model_path

    hf_cache_path = resolve_model_path(
        model_local_path_or_repo_id,
        token=hf_token,
    )

    return Path(hf_cache_path), split_and_save_layers(
        hf_cache_path,
        layer_shards_saving_path,
        compression=compression,
        layer_names=layer_names,
        delete_original=delete_original,
        repo_id=model_local_path_or_repo_id,
        hf_token=hf_token,
    )
