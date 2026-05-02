"""AirLLM Base Model — 100% Local Inference Engine.

``AirLLMBaseModel`` implements the core loop that loads model weights one layer
at a time, runs the forward pass, and frees GPU memory between layers.  All
architecture-specific subclasses inherit from this class.

Uses HuggingFace ``transformers`` library LOCALLY for:
- Model configuration parsing (``AutoConfig``)
- Tokenizer loading (``AutoTokenizer``)
- Generation utilities (``GenerationMixin``)

Supports multiple quantization backends:
- **bitsandbytes** (4-bit/8-bit): Simple but slower (~168 tok/s)
- **AWQ** (4-bit): Activation-aware, 4x faster (~700 tok/s) — ``pip install airllm[awq]``
- **GPTQ** (4-bit): Optimal brain quantization, fast (~712 tok/s) — ``pip install airllm[gptq]``
- **Pre-quantized**: Load pre-quantized safetensor weights directly

All inference runs entirely on your local hardware.
No cloud APIs, no subscriptions, no data leaves your machine.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GenerationMixin,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.quantizers import AutoHfQuantizer

from .constants import DEFAULT_DEVICE, DEFAULT_MAX_SEQ_LEN
from .profiler import LayeredProfiler
from .utils import clean_memory, find_or_create_local_splitted_path, load_layer

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import bitsandbytes as bnb  # noqa: F401

    bitsandbytes_installed = True
    logger.debug("bitsandbytes installed")
except ImportError:
    bitsandbytes_installed = False

try:
    from transformers.cache_utils import Cache, DynamicCache  # noqa: F401

    cache_utils_installed = True
    logger.debug("cache_utils installed")
except ImportError:
    cache_utils_installed = False

# BetterTransformer import (may fail on newer transformers)
try:
    from optimum.bettertransformer import BetterTransformer
except ImportError:
    BetterTransformer = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ModelLayerConfig:
    """Declarative mapping of logical layer roles to weight-name prefixes.

    Each field corresponds to a logical role in the transformer architecture.
    Subclasses can override :pymeth:`AirLLMBaseModel.set_layer_names_dict` to
    supply a custom ``ModelLayerConfig`` for non-standard architectures
    (e.g. ChatGLM, QWen).
    """

    embed_tokens: str = "model.embed_tokens"
    layer_prefix: str = "model.layers"
    norm: str = "model.norm"
    lm_head: str = "lm_head"
    rotary_pos_emb: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Convert to the legacy ``layer_names_dict`` format.

        Returns:
            Dictionary with keys ``embed``, ``layer_prefix``, ``norm``,
            ``lm_head`` and optionally ``rotary_pos_emb``.
        """
        d: dict[str, str] = {
            "embed": self.embed_tokens,
            "layer_prefix": self.layer_prefix,
            "norm": self.norm,
            "lm_head": self.lm_head,
        }
        if self.rotary_pos_emb is not None:
            d["rotary_pos_emb"] = self.rotary_pos_emb
        return d


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
class AirLLMBaseModel(GenerationMixin):
    """Layer-by-layer inference engine for large language models.

    This class splits a HuggingFace-style causal LM into per-layer shards
    and loads each layer into GPU memory one at a time during the forward
    pass.  This allows running models far larger than available GPU VRAM.

    Args:
        model_local_path_or_repo_id: Local directory or HF repo ID.
        device: Target device (default ``"cuda:0"``).
        dtype: Weight precision (default ``torch.float16``).
        max_seq_len: Maximum sequence length (default 512).
        layer_shards_saving_path: Optional alternative shard save directory.
        profiling_mode: Enable per-layer timing instrumentation.
        compression: ``"4bit"``, ``"8bit"``, or ``None``.
        hf_token: HuggingFace API token.
        prefetching: Overlap disk I/O with GPU compute.
        delete_original: Remove original shard files after splitting.

    New in v3.0.0:
        kv_compression: Enable KV cache compression. Options: ``"turboquant"``,
            ``"4bit"``, ``"3bit"``.
        speculative_config: :class:`SpeculativeConfig` or dict for
            self-speculative decoding.
    """

    # ------------------------------------------------------------------
    # Layer name configuration
    # ------------------------------------------------------------------
    def set_layer_names_dict(self) -> None:
        """Populate ``self.layer_names_dict`` with architecture-specific names.

        Subclasses override this to define custom layer name mappings.
        The default mapping corresponds to the standard Llama/Mistral layout.
        """
        self._layer_config = ModelLayerConfig()
        self.layer_names_dict = self._layer_config.to_dict()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        model_local_path_or_repo_id: str | Path,
        device: str = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.float16,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        layer_shards_saving_path: str | Path | None = None,
        profiling_mode: bool = False,
        compression: str | None = None,
        hf_token: str | None = None,
        prefetching: bool = True,
        delete_original: bool = False,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time: float | None = None
        self.total_gpu_loading_time: float | None = None
        self.total_compression_overhead_time: float | None = None
        self._supports_cache_class = False
        self.hf_quantizer: Any | None = None

        # Fail-fast: validate compression parameter (CODE-4).
        valid_compression_values = frozenset({None, "4bit", "8bit"})
        if compression not in valid_compression_values:
            raise ValueError(
                f"Invalid compression={compression!r}. "
                f"Must be one of: {sorted(v for v in valid_compression_values if v)}  or None."
            )

        if compression is not None and not bitsandbytes_installed:
            raise ImportError(
                "Compression requires bitsandbytes. Install it with: pip install bitsandbytes"
            )

        self.compression = compression
        self.hf_token = hf_token
        self.trust_remote_code = trust_remote_code

        # Populate layer name mapping
        self.set_layer_names_dict()

        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            str(model_local_path_or_repo_id),
            layer_shards_saving_path,
            compression=compression,
            layer_names=self.layer_names_dict,
            hf_token=hf_token,
            delete_original=delete_original,
        )

        self.running_device = device
        self.device = torch.device(self.running_device)
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        # Load config
        # SECURITY: trust_remote_code defaults to False to prevent execution
        # of arbitrary code from model repos. Set to True only for models
        # that require custom code (e.g., ChatGLM, Baichuan).
        config_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if hf_token is not None:
            config_kwargs["token"] = hf_token
        self.config = AutoConfig.from_pretrained(self.model_local_path, **config_kwargs)

        self.generation_config = self.get_generation_config()
        self.tokenizer = self.get_tokenizer(hf_token=hf_token)

        self.init_model()

        # Log architecture details for all backends (CODE-9).
        # Subclasses no longer need to override __init__ just for logging.
        self._log_architecture_info()

        # Derive layer count from the model structure
        model_attr: Any = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        layers_count: int = len(model_attr)

        self.layer_names: list[str] = (
            [self.layer_names_dict["embed"]]
            + [f"{self.layer_names_dict['layer_prefix']}.{i}" for i in range(layers_count)]
            + [self.layer_names_dict["norm"], self.layer_names_dict["lm_head"]]
        )

        self.max_seq_len = max_seq_len
        self.main_input_name = "input_ids"

        # ----------------------------------------------------------
        # v3.0 feature integrations (all optional, backward compat)
        # Decomposed into focused helpers for SRP compliance.
        # ----------------------------------------------------------
        self._init_kv_compression(kwargs)
        self._init_speculative_decoding(kwargs)
        self._init_async_loader(prefetching, device)
        self._init_prefetching(prefetching, device)

    # ------------------------------------------------------------------
    # Init helpers (SRP decomposition — each handles one concern)
    # ------------------------------------------------------------------
    def _init_kv_compression(self, kwargs: dict[str, Any]) -> None:
        """Set up TurboQuant KV cache compression if requested."""
        self.kv_compression: str | None = kwargs.get("kv_compression")
        self._kv_compressor: Any | None = None
        if not self.kv_compression:
            return

        from .kv_cache import KVCacheCompressor, PolarQuantConfig

        kv_bits = kwargs.get("kv_bits", 3)
        kv_qjl_dim = kwargs.get("kv_qjl_dim", 64)

        if self.kv_compression == "turboquant":
            config = PolarQuantConfig(bits=kv_bits, qjl_dim=kv_qjl_dim)
        elif self.kv_compression in ("3bit", "4bit", "2bit"):
            parsed_bits = int(self.kv_compression.replace("bit", ""))
            config = PolarQuantConfig(bits=parsed_bits, qjl_dim=0)
        else:
            config = PolarQuantConfig(bits=kv_bits, qjl_dim=0)

        self._kv_compressor = KVCacheCompressor(config)
        logger.info(
            "TurboQuant KV cache compression enabled: %s (%d-bit, qjl_dim=%d, ~%.1fx reduction)",
            self.kv_compression,
            config.bits,
            config.qjl_dim,
            self._kv_compressor.memory_reduction_ratio(),
        )

    def _init_speculative_decoding(self, kwargs: dict[str, Any]) -> None:
        """Set up self-speculative decoding if requested."""
        self.speculative_config: Any | None = kwargs.get("speculative_config")
        if not self.speculative_config:
            return

        from .speculative import SpeculativeConfig

        if isinstance(self.speculative_config, dict):
            self.speculative_config = SpeculativeConfig(**self.speculative_config)
        logger.info(
            "Speculative decoding enabled: exit_ratio=%.2f, speculations=%d",
            self.speculative_config.exit_layer_ratio,
            self.speculative_config.num_speculations,
        )

    def _init_async_loader(self, prefetching: bool, device: str) -> None:
        """Set up async layer loader for CUDA stream overlap."""
        self._async_loader: Any | None = None
        if not (prefetching and torch.cuda.is_available()):
            return

        try:
            from .async_loader import AsyncLayerLoader

            self._async_loader = AsyncLayerLoader(device=str(device))
            logger.info("AsyncLayerLoader enabled for CUDA stream-based prefetching")
        except Exception:
            logger.debug("AsyncLayerLoader not available, using basic prefetching")

    def _init_prefetching(self, prefetching: bool, device: str) -> None:
        """Configure prefetching and CUDA stream."""
        # Only disable basic prefetching when compressed AND no async loader.
        # AsyncLayerLoader works correctly with quantized weights.
        self.prefetching = prefetching
        if self.compression is not None and self._async_loader is None:
            self.prefetching = False
            logger.info(
                "Basic prefetching disabled with compression. Use AsyncLayerLoader for overlap."
            )

        # CUDA stream for async prefetching
        if self.prefetching and device.startswith("cuda"):
            self.stream: torch.cuda.Stream | None = torch.cuda.Stream()
        else:
            self.stream = None

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_local_path}, "
            f"device={self.running_device}, "
            f"dtype={self.running_dtype}, "
            f"compression={self.compression!r})"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} on {self.running_device} "
            f"({len(self.layer_names)} layers, "
            f"max_seq_len={self.max_seq_len})"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Any, ...] | CausalLMOutputWithPast:
        """Alias for :pymeth:`forward`."""
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Generation helpers (override points)
    # ------------------------------------------------------------------
    def get_generation_config(self) -> GenerationConfig:
        """Create the generation configuration.

        Returns a bare ``GenerationConfig`` by default.  Subclasses that
        need pretrained generation parameters (e.g. Llama, QWen2) can
        override this to call ``GenerationConfig.from_pretrained()``.

        Returns:
            A ``GenerationConfig`` instance.
        """
        return GenerationConfig()

    def get_tokenizer(self, hf_token: str | None = None) -> Any:
        """Load the tokenizer for this model.

        Args:
            hf_token: Optional HuggingFace API token.

        Returns:
            A ``PreTrainedTokenizer`` instance.
        """
        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if hf_token is not None:
            kwargs["token"] = hf_token
        return AutoTokenizer.from_pretrained(self.model_local_path, **kwargs)

    def get_use_better_transformer(self) -> bool:
        """Whether to attempt BetterTransformer / SDPA wrapping.

        Returns ``False`` by default since most architectures are
        incompatible with BetterTransformer.  Subclasses that support
        it (e.g. Llama) should override and return ``True``.
        """
        return False

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------
    def init_model(self) -> None:
        """Initialise the empty meta model and configure attention backend.

        This method tries (in order):
        1. BetterTransformer wrapping
        2. ``attn_implementation="sdpa"``
        3. Vanilla model from config
        """
        self.model = None

        if self.get_use_better_transformer() and BetterTransformer is not None:
            try:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(
                        self.config, trust_remote_code=self.trust_remote_code
                    )
                    self.model = BetterTransformer.transform(self.model)
            except ValueError:
                del self.model
                clean_memory()
                self.model = None

            if self.model is None:
                try:
                    logger.info("BetterTransformer unavailable; trying attn_implementation='sdpa'…")
                    self.config.attn_implementation = "sdpa"
                    with init_empty_weights():
                        self.model = AutoModelForCausalLM.from_config(
                            self.config,
                            attn_implementation="sdpa",
                            trust_remote_code=self.trust_remote_code,
                        )
                except TypeError:
                    del self.model
                    clean_memory()
                    self.model = None

        # Fallback
        if self.model is None:
            logger.info("Creating model without attention optimisation.")
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(
                    self.config, trust_remote_code=self.trust_remote_code
                )

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

        self.model.eval()
        self.model.tie_weights()

        self.set_layers_from_layer_names()

        # Move buffers to device
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model,
                buffer_name,
                self.running_device,
                value=buffer,
                dtype=self.running_dtype,
            )

        if "rotary_pos_emb" in self.layer_names_dict:
            self.load_rotary_pos_emb_to_device()

    def set_layers_from_layer_names(self) -> None:
        """Resolve model attributes into an ordered list of layer references."""
        self.layers: list[Any] = []

        # Embedding layer
        model_attr: Any = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        # Transformer blocks
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.extend(list(model_attr))

        # Norm layer
        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        # LM head
        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    def load_rotary_pos_emb_to_device(self) -> None:
        """Load rotary positional embeddings to the running device."""
        state_dict = load_layer(self.checkpoint_path, self.layer_names_dict["rotary_pos_emb"])
        self.move_layer_to_device(state_dict)

    # ------------------------------------------------------------------
    # Layer loading & device movement
    # ------------------------------------------------------------------
    def load_layer_to_cpu(self, layer_name: str) -> dict[str, torch.Tensor]:
        """Load a single layer's weights from disk into CPU memory.

        Args:
            layer_name: Logical layer name.

        Returns:
            The layer state dict with tensors on CPU.
        """
        t = time.time()

        load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time
            self.profiler.add_profiling_time("load_safe_tensor", disk_loading_time)
            self.profiler.add_profiling_time("compression_time", compression_time)
        else:
            state_dict = load_layer_output

        # Pin memory for async transfer
        if self.prefetching:
            t = time.time()
            if torch.cuda.is_available():
                for k in state_dict:
                    state_dict[k].pin_memory()
            else:
                logger.debug("Prefetching enabled but CUDA unavailable; skipping pin_memory.")
            elapsed_time = time.time() - t
            if self.profiling_mode:
                self.profiler.add_profiling_time("pin_memory_to_trigger_load", elapsed_time)

        return state_dict

    def move_layer_to_device(self, state_dict: dict[str, torch.Tensor]) -> list[str]:
        """Transfer layer weights to the running device.

        Args:
            state_dict: The CPU-resident layer state dict.

        Returns:
            List of parameter names that were moved.
        """
        layers: list[str] = []
        for param_name in state_dict:
            if self.hf_quantizer is None:
                layers.append(param_name)
            else:
                if ".weight" in param_name:
                    layer_name = param_name[: param_name.index(".weight") + len(".weight")]
                    if layer_name not in layers:
                        layers.append(layer_name)

        for param_name in layers:
            if self.hf_quantizer is None or not self.hf_quantizer.check_quantized_param(
                self.model, param_value=None, param_name=param_name, state_dict={}
            ):
                set_module_tensor_to_device(
                    self.model,
                    param_name,
                    self.running_device,
                    value=state_dict[param_name],
                    dtype=self.running_dtype,
                )
            else:
                self.hf_quantizer.update_torch_dtype(None)
                self.hf_quantizer.create_quantized_param(
                    self.model,
                    state_dict[param_name],
                    param_name,
                    self.running_device,
                    state_dict,
                )
        return layers

    # ------------------------------------------------------------------
    # GenerationMixin interface
    # ------------------------------------------------------------------
    def can_generate(self) -> bool:
        """Return ``True`` — required by ``GenerationMixin``."""
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare model inputs for a single generation step.

        Follows the HuggingFace ``GenerationMixin`` protocol.
        """
        if past_key_values is not None:
            past_length = self.get_past_key_values_cache_seq_len(past_key_values)

            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs: dict[str, Any] = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    # ------------------------------------------------------------------
    # KV-cache / sequence helpers (override points)
    # ------------------------------------------------------------------
    def get_past_key_values_cache_seq_len(self, past_key_values: Any) -> int:
        """Return cached sequence length from KV cache."""
        if hasattr(past_key_values[0][0], "shape"):
            return past_key_values[0][0].shape[2]
        return past_key_values[0][0].shape[2]

    def get_sequence_len(self, seq: torch.Tensor) -> int:
        """Return the sequence-length dimension of *seq*."""
        return seq.shape[1]

    def get_pos_emb_args(self, len_p: int, len_s: int) -> dict[str, Any]:
        """Return extra keyword arguments for positional embeddings."""
        return {}

    def get_past_key_value_args(
        self, k_cache: torch.Tensor, v_cache: torch.Tensor
    ) -> dict[str, Any]:
        """Return KV-cache keyword arguments for a decoder layer."""
        return {"past_key_value": (k_cache, v_cache)}

    def get_attention_mask_args(
        self,
        full_attention_mask: torch.Tensor,
        len_p: int,
        len_s: int,
    ) -> dict[str, Any]:
        """Return attention mask keyword arguments for a decoder layer."""
        return {"attention_mask": full_attention_mask[:, :, -len_s:, -len_p - len_s :]}

    def get_position_ids_args(
        self,
        full_position_ids: torch.Tensor,
        len_p: int,
        len_s: int,
    ) -> dict[str, Any]:
        """Return position-ID keyword arguments for a decoder layer."""
        return {"position_ids": full_position_ids[:, len_p : len_p + len_s]}

    def run_lm_head(self, layer: Any, seq: torch.Tensor) -> torch.Tensor:
        """Run the language model head and return float logits."""
        return layer(seq).float()

    def run_norm(self, layer: Any, seq: torch.Tensor) -> torch.Tensor:
        """Run the normalisation layer."""
        return layer(seq)

    def _log_architecture_info(self) -> None:
        """Log architecture-specific configuration details.

        Subclasses can override this to log additional architecture-specific
        information. The base implementation logs common config fields.
        Called automatically at the end of ``__init__``.
        """
        num_layers = getattr(self.config, "num_hidden_layers", "unknown")
        hidden_size = getattr(self.config, "hidden_size", "unknown")
        logger.info(
            "%s config: %s layers, hidden_size=%s",
            self.__class__.__name__,
            num_layers,
            hidden_size,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[Any, ...] | CausalLMOutputWithPast:
        """Layer-by-layer forward pass.

        Loads each transformer layer from disk, runs it, then offloads it to
        free GPU memory for the next layer.
        """
        if cache_utils_installed:
            # KV cache not yet supported with new transformers cache API
            use_cache = False

        if self.profiling_mode:
            self.profiler.clear_profiling_time()
            forward_start = time.process_time()
            forward_start_wall = time.time()

        # Reinitialise the empty model scaffold to ensure clean state.
        # This is intentional: the layer-by-layer approach loads weights into
        # the scaffold per-layer during forward(). After forward() completes,
        # the scaffold holds stale weights from the last layer. Recreating it
        # guarantees each forward() starts with a clean meta-device model.
        del self.model
        clean_memory()
        self.init_model()

        batch = [ids.to(self.running_device).unsqueeze(0) for ids in input_ids]

        # Warn if caller-provided attention_mask or position_ids are overridden.
        # The layer-by-layer approach requires a full causal mask spanning
        # max_seq_len, so user-provided values cannot be respected here.
        if attention_mask is not None:
            logger.warning(
                "User-provided attention_mask is overridden by the layer-by-layer "
                "forward pass, which requires a full causal mask of size max_seq_len=%d.",
                self.max_seq_len,
            )
        if position_ids is not None:
            logger.warning(
                "User-provided position_ids is overridden by the layer-by-layer "
                "forward pass, which generates sequential IDs of length max_seq_len=%d.",
                self.max_seq_len,
            )

        # Causal attention mask and position IDs
        attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        attention_mask = attention_mask.to(self.running_device)
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[
            None, :
        ]

        kv_cache_list: list[tuple[list[Any], list[Any]]] | None = (
            [([], []) for _ in self.layers] if use_cache else None
        )
        all_hidden_states: list[Any] | None = (
            [] * len(self.layers) if output_hidden_states else None
        )
        all_self_attns: list[Any] | None = [] * len(self.layers) if output_attentions else None

        with torch.inference_mode(), ThreadPoolExecutor() as executor:
            # Kick off first layer load
            if self.prefetching:
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])

            for i, (layer_name, layer) in tqdm(
                enumerate(zip(self.layer_names, self.layers)),
                desc=f"running layers({self.running_device})",
                total=len(self.layers),
            ):
                if self.prefetching:
                    if self.profiling_mode:
                        t = time.time()
                    state_dict = future.result()
                    if self.profiling_mode:
                        elapsed = time.time() - t
                        self.profiler.add_profiling_time("load_safe_tensor_cpu_wait", elapsed)

                    if self.profiling_mode:
                        t = time.time()
                    moved_layers = self.move_layer_to_device(state_dict)
                    if self.profiling_mode:
                        elapsed = time.time() - t
                        self.profiler.add_profiling_time("create_layer_from_state_dict", elapsed)

                    # Prefetch next layer
                    if (i + 1) < len(self.layer_names):
                        if self.profiling_mode:
                            t = time.time()
                        future = executor.submit(self.load_layer_to_cpu, self.layer_names[i + 1])
                        if self.profiling_mode:
                            elapsed = time.time() - t
                            self.profiler.add_profiling_time("kick_off_load_cpu", elapsed)
                else:
                    state_dict = self.load_layer_to_cpu(layer_name)
                    if self.profiling_mode:
                        t = time.time()
                    moved_layers = self.move_layer_to_device(state_dict)
                    if self.profiling_mode:
                        elapsed = time.time() - t
                        self.profiler.add_profiling_time("create_layer_from_safe_tensor", elapsed)

                # --- Run layer for each batch element ---
                for j, seq in enumerate(batch):
                    if layer_name == self.layer_names_dict["embed"]:
                        batch[j] = layer(seq)
                    elif layer_name == self.layer_names_dict["norm"]:
                        batch[j] = self.run_norm(layer, seq)
                        if output_attentions:
                            all_hidden_states[i].append(batch[j])
                    elif layer_name == self.layer_names_dict["lm_head"]:
                        batch[j] = self.run_lm_head(layer, seq)
                    else:
                        # Decoder layer
                        if past_key_values is not None:
                            k_cache, v_cache = past_key_values[i - 1]
                            if self._kv_compressor is not None:
                                k_cache = self._kv_compressor.decompress(k_cache)
                                v_cache = self._kv_compressor.decompress(v_cache)
                            len_p = self.get_past_key_values_cache_seq_len(past_key_values)
                            len_s = self.get_sequence_len(seq)

                            position_ids_args = self.get_position_ids_args(
                                position_ids, len_p, len_s
                            )
                            attention_mask_args = self.get_attention_mask_args(
                                attention_mask, len_p, len_s
                            )
                            past_key_value_args = self.get_past_key_value_args(k_cache, v_cache)
                            pos_embed_args = self.get_pos_emb_args(len_p, len_s)

                            layer_kwargs: dict[str, Any] = {
                                "use_cache": True,
                                **past_key_value_args,
                                **pos_embed_args,
                                **attention_mask_args,
                                **position_ids_args,
                            }
                            layer_outputs = layer(seq, **layer_kwargs)
                            new_seq = layer_outputs[0]

                            if output_attentions:
                                all_self_attns[i].append(layer_outputs[1])

                            if use_cache:
                                kv = layer_outputs[2 if output_attentions else 1]
                                k_cache, v_cache = kv[0], kv[1]
                                if self._kv_compressor is not None:
                                    k_cache = self._kv_compressor.compress(k_cache)
                                    v_cache = self._kv_compressor.compress(v_cache)
                                kv_cache_list[i][0].append(k_cache)
                                kv_cache_list[i][1].append(v_cache)

                        else:
                            len_seq = self.get_sequence_len(seq)
                            pos_embed_args = self.get_pos_emb_args(0, len_seq)
                            attention_mask_args = self.get_attention_mask_args(
                                attention_mask, 0, len_seq
                            )
                            position_ids_args = self.get_position_ids_args(position_ids, 0, len_seq)

                            if not use_cache:
                                layer_kwargs = {
                                    "use_cache": False,
                                    "attention_mask": attention_mask[:, :, -len_seq:, -len_seq:],
                                    **pos_embed_args,
                                    **attention_mask_args,
                                    **position_ids_args,
                                }
                                new_seq = layer(seq, **layer_kwargs)[0]
                            else:
                                layer_kwargs = {
                                    "use_cache": True,
                                    "attention_mask": attention_mask[:, :, -len_seq:, -len_seq:],
                                    **pos_embed_args,
                                    **attention_mask_args,
                                    **position_ids_args,
                                }
                                layer_out = layer(seq, **layer_kwargs)
                                new_seq, (k_cache, v_cache) = layer_out
                                if self._kv_compressor is not None:
                                    k_cache = self._kv_compressor.compress(k_cache)
                                    v_cache = self._kv_compressor.compress(v_cache)
                                kv_cache_list[i][0].append(k_cache)
                                kv_cache_list[i][1].append(v_cache)

                        batch[j] = new_seq

                if output_hidden_states:
                    all_hidden_states += (torch.cat(batch, 0),)

                # Offload layer from GPU
                if self.hf_quantizer is not None:
                    for param_name in moved_layers:
                        set_module_tensor_to_device(self.model, param_name, "meta")
                else:
                    layer.to("meta")

                layer.to("meta")
                clean_memory()

        logits = torch.cat(batch, 0)

        if use_cache and kv_cache_list is not None:
            kv_cache_list = kv_cache_list[1:-2]
            for i in range(len(kv_cache_list)):
                if self._kv_compressor is not None:
                    from .kv_cache import CompressedKVCache

                    kv_cache_list[i] = (
                        CompressedKVCache.cat(kv_cache_list[i][0], 0),
                        CompressedKVCache.cat(kv_cache_list[i][1], 0),
                    )
                else:
                    kv_cache_list[i] = (
                        torch.cat(kv_cache_list[i][0], 0),
                        torch.cat(kv_cache_list[i][1], 0),
                    )

        if output_attentions and all_self_attns is not None:
            all_self_attns = all_self_attns[:-2]
            for i in range(len(all_self_attns)):
                all_self_attns[i] = torch.cat(all_self_attns[i], 0)

        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states = all_hidden_states[:-2]
            for i in range(len(all_hidden_states)):
                all_hidden_states[i] = torch.cat(all_hidden_states[i], 0)

        if not return_dict:
            return tuple(
                v
                for v in [
                    logits,
                    tuple(kv_cache_list) if kv_cache_list is not None else None,
                    tuple(all_hidden_states) if all_hidden_states is not None else None,
                    tuple(all_self_attns) if all_self_attns is not None else None,
                ]
                if v is not None
            )

        if self.profiling_mode:
            forward_elapsed = time.process_time() - forward_start
            forward_elapsed_wall = time.time() - forward_start_wall
            self.profiler.print_profiling_time()
            logger.info(
                "Total infer process time (incl. GPU compute): %.4f",
                forward_elapsed,
            )
            logger.info(
                "Total infer wall time (incl. GPU compute): %.4f",
                forward_elapsed_wall,
            )
            self.profiler.clear_profiling_time()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=(tuple(kv_cache_list) if kv_cache_list is not None else None),
            hidden_states=(tuple(all_hidden_states) if all_hidden_states is not None else None),
            attentions=(tuple(all_self_attns) if all_self_attns is not None else None),
        )
