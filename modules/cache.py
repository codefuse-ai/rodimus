# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache


class HybridCache(Cache):
    def __init__(
        self,
        seen_tokens: int = 0,
        has_ssm: bool = True,
        has_attn: bool = False,
    ) -> None:
        assert has_attn or has_ssm, "set `has_attn=True` or `has_ssm=True`"

        self._seen_tokens = seen_tokens

        self.has_ssm = has_ssm
        self.has_attn = has_attn
        self.num_ssm_states = 2 if self.has_ssm else 0

        self.conv_states: List[torch.Tensor] = []
        self.ssm_states: List[torch.Tensor] = []
        self.key_caches: List[torch.Tensor] = []
        self.value_caches: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        if layer_idx < len(self):
            states = ()
            if self.has_ssm:
                states += (self.conv_states[layer_idx], self.ssm_states[layer_idx])
            if self.has_attn:
                states += (self.key_caches[layer_idx], self.value_caches[layer_idx])
            return states
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        assert self.has_ssm
        return self[layer_idx][:self.num_ssm_states]

    def get_attn_states(self, layer_idx: int) -> torch.Tensor:
        assert self.has_attn
        return self[layer_idx][self.num_ssm_states:]

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield self[layer_idx]

    def __len__(self):
        if self.has_ssm:
            return len(self.conv_states)
        else:
            return len(self.key_caches)

    def update(
        self,
        layer_idx: int,
        conv_state: torch.Tensor = None,
        ssm_state: torch.Tensor = None,
        key_cache: torch.Tensor = None,
        value_cache: torch.Tensor = None,
        offset: Optional[int, torch.Tensor] = 1,
        skip_copy: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        is_end = True
        if self.has_ssm:
            if conv_state is not None:
                if len(self.conv_states) <= layer_idx:
                    self.conv_states.append(conv_state)
                    is_end = False
                elif not skip_copy:
                    self.conv_states[layer_idx].copy_(conv_state)

            if ssm_state is not None:
                if len(self.ssm_states) <= layer_idx:
                    self.ssm_states.append(ssm_state)
                    is_end = False
                elif not skip_copy:
                    self.ssm_states[layer_idx].copy_(ssm_state)

        if self.has_attn:
            if key_cache is not None:
                if len(self.key_caches) <= layer_idx:
                    self.key_caches.append(key_cache)
                    is_end = False
                elif not skip_copy:
                    # cache_seq_len = min(key_cache.shape[1], self.key_caches[layer_idx].shape[1])
                    # # self.key_caches[layer_idx] = torch.roll(
                    # #     self.key_caches[layer_idx], shifts=-cache_seq_len, dims=1)  # b l h d
                    # # self.key_caches[layer_idx][:, -cache_seq_len:, :, :].copy_(key_cache[:, -cache_seq_len:, :, :])
                    # self.key_caches[layer_idx] = torch.cat((self.key_caches[layer_idx][:, cache_seq_len:, :, :], key_cache[:, -cache_seq_len:, :, :]), dim=1)

                    if key_cache.shape[0] > self.key_caches[layer_idx].shape[1]:
                        max_cache_len = self.key_caches[layer_idx].shape[1]
                        k_out = key_cache[:, -max_cache_len:, :, :]
                        self.key_caches[layer_idx] += k_out
                    else:
                        k_out = self.key_caches[layer_idx]

                        max_cache_len = self.key_caches[layer_idx].shape[1]
                        input_num_tokens = key_cache.shape[1]

                        slicing = torch.ones(max_cache_len, dtype=torch.long, device=key_cache.device).cumsum(0)
                        cache_position = torch.arange(self._seen_tokens, self._seen_tokens + input_num_tokens, device=key_cache.device).clamp(0, max_cache_len - 1)
                        to_shift = cache_position >= max_cache_len - 1
                        indices = (slicing + to_shift[-1].int() - 1) % max_cache_len

                        k_out = k_out[:, indices, :, :]
                        k_out.index_copy_(1, cache_position, key_cache)

                        self.key_caches[layer_idx].zero_()
                        self.key_caches[layer_idx] += k_out

            if value_cache is not None:
                if len(self.value_caches) <= layer_idx:
                    self.value_caches.append(value_cache)
                    is_end = False
                elif not skip_copy:
                    # cache_seq_len = min(value_cache.shape[1], self.value_caches[layer_idx].shape[1])
                    # # self.value_caches[layer_idx] = torch.roll(
                    # #     self.value_caches[layer_idx], shifts=-cache_seq_len, dims=1)  # b l h d
                    # # self.value_caches[layer_idx][:, -cache_seq_len:, :, :].copy_(value_cache[:, -cache_seq_len:, :, :])
                    # self.value_caches[layer_idx] = torch.cat((self.value_caches[layer_idx][:, cache_seq_len:, :, :], value_cache[:, -cache_seq_len:, :, :]), dim=1)

                    if value_cache.shape[0] > self.value_caches[layer_idx].shape[1]:
                        max_cache_len = self.value_caches[layer_idx].shape[1]
                        v_out = value_cache[:, -max_cache_len:, :, :]
                        self.value_caches[layer_idx] += v_out
                    else:
                        v_out = self.value_caches[layer_idx]

                        max_cache_len = self.value_caches[layer_idx].shape[1]
                        input_num_tokens = value_cache.shape[1]

                        slicing = torch.ones(max_cache_len, dtype=torch.long, device=value_cache.device).cumsum(0)
                        cache_position = torch.arange(self._seen_tokens, self._seen_tokens + input_num_tokens, device=value_cache.device).clamp(0, max_cache_len - 1)
                        to_shift = cache_position >= max_cache_len - 1
                        indices = (slicing + to_shift[-1].int() - 1) % max_cache_len

                        v_out = v_out[:, indices, :, :]
                        v_out.index_copy_(1, cache_position, value_cache)

                        self.value_caches[layer_idx].zero_()
                        self.value_caches[layer_idx] += v_out

        # update the number of seen tokens once we achieve the last layer
        if layer_idx == len(self) - 1 and is_end:
            if self.has_attn and self.has_ssm:
                if value_cache is not None:  # update `offset` once
                    self._seen_tokens += offset
            else:
                self._seen_tokens += offset

        return self[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self)):
            if self.has_ssm:
                device = self.conv_states[layer_idx].device
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
                self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))
            if self.has_attn:
                device = self.key_caches[layer_idx].device
                self.key_caches[layer_idx] = self.key_caches[layer_idx].index_select(0, beam_idx.to(device))
                self.value_caches[layer_idx] = self.value_caches[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[torch.Tensor]:
        legacy_cache = []
        for layer_idx in range(len(self)):
            layer_cache = ()
            if self.has_ssm:
                layer_cache += (self.conv_states[layer_idx], self.ssm_states[layer_idx])
            if self.has_attn:
                layer_cache += (self.key_caches[layer_idx], self.value_caches[layer_idx])
            legacy_cache.append(layer_cache)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        seen_tokens: int = 0,
        has_ssm: bool = True,
        has_attn: bool = False,
        **kwargs,
    ) -> HybridCache:
        """Converts a cache in the legacy cache format into an equivalent `HybridCache`."""
        cache = cls(seen_tokens, has_ssm=has_ssm, has_attn=has_attn, **kwargs)

        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                unpack_states = {}
                if cache.has_ssm:
                    assert len(past_key_values[layer_idx]) >= 2
                    conv_state, ssm_state = past_key_values[layer_idx][:cache.num_ssm_states]
                    unpack_states["conv_state"] = conv_state
                    unpack_states["ssm_state"] = ssm_state
                if cache.has_attn:
                    assert len(past_key_values[layer_idx]) >= 4
                    key_cache, value_cache = past_key_values[layer_idx][cache.num_ssm_states:]
                    unpack_states["key_cache"] = key_cache
                    unpack_states["value_cache"] = value_cache
                cache.update(
                    layer_idx,
                    **unpack_states,
                )
                
        return cache