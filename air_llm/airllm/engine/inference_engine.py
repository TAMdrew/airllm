import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from accelerate.utils.modeling import set_module_tensor_to_device
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils import clean_memory

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Decoupled orchestrator for the layer-by-layer forward pass."""

    def __init__(
        self,
        model_wrapper: Any,
    ):
        self.wrapper = model_wrapper

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
        **kwargs: Any,
    ) -> tuple[Any, ...] | CausalLMOutputWithPast:

        if self.wrapper.profiling_mode:
            self.wrapper.profiler.clear_profiling_time()
            forward_start = time.process_time()
            forward_start_wall = time.time()

        # Reset layer states to 'meta' instead of full model re-initialization
        for layer in self.wrapper.layers:
            layer.to("meta")
        clean_memory()

        batch = [ids.to(self.wrapper.running_device).unsqueeze(0) for ids in input_ids]

        attention_mask, position_ids = self._build_attention_mask_and_position_ids(
            attention_mask, position_ids
        )

        kv_cache_list: list[tuple[list[Any], list[Any]]] | None = (
            [([], []) for _ in self.wrapper.layers] if use_cache else None
        )
        # BUG-FIX: `[] * N` produces `[]`, not N empty lists.
        # Use list comprehension to create independent empty lists.
        all_hidden_states: list[list[Any]] | None = (
            [[] for _ in range(len(self.wrapper.layers))] if output_hidden_states else None
        )
        all_self_attns: list[list[Any]] | None = (
            [[] for _ in range(len(self.wrapper.layers))] if output_attentions else None
        )

        exit_layer = None
        if self.wrapper.speculative_config is not None and kwargs.get("is_draft", False):
            # Calculate total layers
            total_layers = sum(
                1
                for name in self.wrapper.layer_names
                if name.startswith(self.wrapper.layer_names_dict["layer_prefix"])
            )
            exit_layer = self.wrapper.speculative_config.get_exit_layer(total_layers)

        with torch.inference_mode(), ThreadPoolExecutor() as executor:
            if self.wrapper.prefetching:
                future = executor.submit(
                    self.wrapper.load_layer_to_cpu, self.wrapper.layer_names[0]
                )

            for i, (layer_name, layer) in tqdm(
                enumerate(zip(self.wrapper.layer_names, self.wrapper.layers)),
                desc=f"running layers({self.wrapper.running_device})",
                total=len(self.wrapper.layers),
            ):
                if exit_layer is not None and layer_name.startswith(
                    self.wrapper.layer_names_dict["layer_prefix"]
                ):
                    # BUG-FIX: layer names may end with "." (e.g. "model.layers.0.")
                    # so split(".")[-1] yields "". Strip trailing dots first.
                    layer_idx = int(layer_name.rstrip(".").split(".")[-1])
                    if layer_idx >= exit_layer:
                        continue

                moved_layers = self._load_layer_with_profiling(
                    i, layer_name, executor, future if self.wrapper.prefetching else None
                )

                # Prefetch next layer
                if self.wrapper.prefetching and (i + 1) < len(self.wrapper.layer_names):
                    next_layer_name = self.wrapper.layer_names[i + 1]
                    skip_prefetch = False
                    if exit_layer is not None and next_layer_name.startswith(
                        self.wrapper.layer_names_dict["layer_prefix"]
                    ):
                        next_layer_idx = int(next_layer_name.rstrip(".").split(".")[-1])
                        if next_layer_idx >= exit_layer:
                            skip_prefetch = True

                    if not skip_prefetch:
                        if self.wrapper.profiling_mode:
                            t = time.time()
                        future = executor.submit(self.wrapper.load_layer_to_cpu, next_layer_name)
                        if self.wrapper.profiling_mode:
                            elapsed = time.time() - t
                            self.wrapper.profiler.add_profiling_time("kick_off_load_cpu", elapsed)

                for j, seq in enumerate(batch):
                    batch[j] = self._run_layer(
                        i,
                        layer_name,
                        layer,
                        seq,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        use_cache,
                        output_attentions,
                        all_hidden_states,
                        all_self_attns,
                        kv_cache_list,
                    )

                if output_hidden_states:
                    all_hidden_states += (torch.cat(batch, 0),)

                # Offload layer from GPU
                if self.wrapper.hf_quantizer is not None:
                    for param_name in moved_layers:
                        set_module_tensor_to_device(self.wrapper.model, param_name, "meta")
                else:
                    layer.to("meta")

                clean_memory()

        logits = torch.cat(batch, 0)

        return self._assemble_outputs(
            logits,
            kv_cache_list,
            all_hidden_states,
            all_self_attns,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            forward_start if self.wrapper.profiling_mode else None,
            forward_start_wall if self.wrapper.profiling_mode else None,
        )

    def _build_attention_mask_and_position_ids(self, attention_mask, position_ids):
        if attention_mask is None:
            attention_mask = torch.ones(self.wrapper.max_seq_len, self.wrapper.max_seq_len)
            attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
            attention_mask = attention_mask.to(self.wrapper.running_device)
        else:
            attention_mask = attention_mask.to(self.wrapper.running_device)

        if position_ids is None:
            position_ids = torch.arange(
                self.wrapper.max_seq_len, dtype=torch.long, device=self.wrapper.running_device
            )[None, :]
        else:
            position_ids = position_ids.to(self.wrapper.running_device)

        return attention_mask, position_ids

    def _load_layer_with_profiling(self, i, layer_name, executor, future):
        if self.wrapper.prefetching:
            if self.wrapper.profiling_mode:
                t = time.time()
            state_dict = future.result()
            if self.wrapper.profiling_mode:
                elapsed = time.time() - t
                self.wrapper.profiler.add_profiling_time("load_safe_tensor_cpu_wait", elapsed)

            if self.wrapper.profiling_mode:
                t = time.time()
            moved_layers = self.wrapper.move_layer_to_device(state_dict)
            if self.wrapper.profiling_mode:
                elapsed = time.time() - t
                self.wrapper.profiler.add_profiling_time("create_layer_from_state_dict", elapsed)
        else:
            state_dict = self.wrapper.load_layer_to_cpu(layer_name)
            if self.wrapper.profiling_mode:
                t = time.time()
            moved_layers = self.wrapper.move_layer_to_device(state_dict)
            if self.wrapper.profiling_mode:
                elapsed = time.time() - t
                self.wrapper.profiler.add_profiling_time("create_layer_from_safe_tensor", elapsed)
        return moved_layers

    def _run_layer(
        self,
        i,
        layer_name,
        layer,
        seq,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        output_attentions,
        all_hidden_states,
        all_self_attns,
        kv_cache_list,
    ):
        if layer_name == self.wrapper.layer_names_dict["embed"]:
            return layer(seq)
        elif layer_name == self.wrapper.layer_names_dict["norm"]:
            res = self.wrapper.run_norm(layer, seq)
            if output_attentions:
                all_hidden_states[i].append(res)
            return res
        elif layer_name == self.wrapper.layer_names_dict["lm_head"]:
            return self.wrapper.run_lm_head(layer, seq)
        else:
            return self._run_decoder_layer(
                i,
                layer,
                seq,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                output_attentions,
                all_self_attns,
                kv_cache_list,
            )

    def _run_decoder_layer(
        self,
        i,
        layer,
        seq,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        output_attentions,
        all_self_attns,
        kv_cache_list,
    ):
        if past_key_values is not None:
            k_cache, v_cache = past_key_values[i - 1]
            if self.wrapper._kv_compressor is not None:
                k_cache = self.wrapper._kv_compressor.decompress(k_cache)
                v_cache = self.wrapper._kv_compressor.decompress(v_cache)
            len_p = self.wrapper.get_past_key_values_cache_seq_len(past_key_values)
            len_s = self.wrapper.get_sequence_len(seq)

            position_ids_args = self.wrapper.get_position_ids_args(position_ids, len_p, len_s)
            attention_mask_args = self.wrapper.get_attention_mask_args(attention_mask, len_p, len_s)
            past_key_value_args = self.wrapper.get_past_key_value_args(k_cache, v_cache)
            pos_embed_args = self.wrapper.get_pos_emb_args(len_p, len_s)

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
                if self.wrapper._kv_compressor is not None:
                    k_cache = self.wrapper._kv_compressor.compress(k_cache)
                    v_cache = self.wrapper._kv_compressor.compress(v_cache)
                kv_cache_list[i][0].append(k_cache)
                kv_cache_list[i][1].append(v_cache)

        else:
            len_seq = self.wrapper.get_sequence_len(seq)
            pos_embed_args = self.wrapper.get_pos_emb_args(0, len_seq)
            attention_mask_args = self.wrapper.get_attention_mask_args(attention_mask, 0, len_seq)
            position_ids_args = self.wrapper.get_position_ids_args(position_ids, 0, len_seq)

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
                if self.wrapper._kv_compressor is not None:
                    k_cache = self.wrapper._kv_compressor.compress(k_cache)
                    v_cache = self.wrapper._kv_compressor.compress(v_cache)
                kv_cache_list[i][0].append(k_cache)
                kv_cache_list[i][1].append(v_cache)
        return new_seq

    def _assemble_outputs(
        self,
        logits,
        kv_cache_list,
        all_hidden_states,
        all_self_attns,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
        forward_start,
        forward_start_wall,
    ):
        if use_cache and kv_cache_list is not None:
            kv_cache_list = kv_cache_list[1:-2]
            for i in range(len(kv_cache_list)):
                if self.wrapper._kv_compressor is not None:
                    from ..kv_cache import CompressedKVCache

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

        if self.wrapper.profiling_mode:
            forward_elapsed = time.process_time() - forward_start
            forward_elapsed_wall = time.time() - forward_start_wall
            self.wrapper.profiler.print_profiling_time()
            logger.info(
                "Total infer process time (incl. GPU compute): %.4f",
                forward_elapsed,
            )
            logger.info(
                "Total infer wall time (incl. GPU compute): %.4f",
                forward_elapsed_wall,
            )
            self.wrapper.profiler.clear_profiling_time()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=(tuple(kv_cache_list) if kv_cache_list is not None else None),
            hidden_states=(tuple(all_hidden_states) if all_hidden_states is not None else None),
            attentions=(tuple(all_self_attns) if all_self_attns is not None else None),
        )
