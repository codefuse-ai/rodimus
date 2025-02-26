import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

from einops import rearrange, repeat

from transformers.cache_utils import Cache
from transformers.utils import is_torchdynamo_compiling

import fla
from ops.layernorm import RMSNorm
from modules.utils import (
    autocast_to_2B,
    safe_eval_number,
    align_multiple
)
from ops.rotary import RotaryEmbedding

USE_FLASH_ATTN = True
try:
    from flash_attn import (
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_func,
        flash_attn_varlen_func,
    )
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    USE_FLASH_ATTN = False

flash_attn_with_kvcache = None  # TODO
if USE_FLASH_ATTN:
    try:
        from flash_attn import flash_attn_with_kvcache
    except ImportError:
        flash_attn_with_kvcache = None


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(
        seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class FlashAttention(nn.Module):
    def __init__(
        self,
        causal=True,
        softmax_scale=None,
        attention_dropout=0.0,
        window_size=(-1, -1),
    ):
        super().__init__()
        assert USE_FLASH_ATTN, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.window_size = window_size

    def __repr__(self):
        return f"{self.drop}\n\twindow_size: {self.window_size}, causal: {self.causal}, softmax_scale: {self.softmax_scale}"

    @amp.autocast(False)
    def forward(
        self,
        qkv,
        kv=None,
        v=None,
        cu_seqlens=None,
        max_seqlen=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
    ):
        dtype = qkv.dtype
        unpadded = cu_seqlens is not None

        if kv is None:
            assert v is None
            assert qkv.dtype in [torch.float16, torch.bfloat16]

            if unpadded:
                assert cu_seqlens.dtype == torch.int32
                assert max_seqlen is not None
                assert isinstance(max_seqlen, int)
                return flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_seqlen,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=self.causal,
                    # [i - window_size[0], i + window_size[1]]
                    window_size=self.window_size,
                ).to(dtype)
            else:
                return flash_attn_qkvpacked_func(
                    qkv,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=self.causal,
                    window_size=self.window_size,
                ).to(dtype)

        else:
            assert qkv.dtype in [torch.float16, torch.bfloat16]
            assert kv.dtype in [torch.float16, torch.bfloat16]
            if v is None:
                q = qkv
                if unpadded:
                    assert cu_seqlens.dtype == torch.int32
                    assert max_seqlen is not None
                    assert isinstance(max_seqlen, int)
                    assert cu_seqlens_k is not None
                    assert cu_seqlens_k.dtype == torch.int32
                    assert max_seqlen_k is not None
                    assert isinstance(max_seqlen, int)
                    return flash_attn_varlen_kvpacked_func(
                        q,
                        kv,
                        cu_seqlens,
                        cu_seqlens_k,
                        max_seqlen,
                        max_seqlen_k,
                        self.drop.p if self.training else 0.0,
                        softmax_scale=self.softmax_scale,
                        causal=self.causal,
                        window_size=self.window_size,
                    ).to(dtype)
                else:
                    batch_size, seqlen_q = q.shape[0], q.shape[1]
                    seqlen_k = kv.shape[1]
                    assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
                    return flash_attn_kvpacked_func(
                        q,
                        kv,
                        self.drop.p if self.training else 0.0,
                        causal=self.causal,
                        softmax_scale=self.softmax_scale,
                        window_size=self.window_size,
                    ).to(dtype)
            else:
                assert v.dtype in [torch.float16, torch.bfloat16]

                q = qkv
                k = kv
                if unpadded:
                    assert cu_seqlens.dtype == torch.int32
                    assert max_seqlen is not None
                    assert isinstance(max_seqlen, int)
                    assert cu_seqlens_k is not None
                    assert cu_seqlens_k.dtype == torch.int32
                    assert max_seqlen_k is not None
                    assert isinstance(max_seqlen, int)
                    return flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens,
                        cu_seqlens_k,
                        max_seqlen,
                        max_seqlen_k,
                        self.drop.p if self.training else 0.0,
                        softmax_scale=self.softmax_scale,
                        causal=self.causal,
                        window_size=self.window_size,
                    ).to(dtype)
                else:
                    batch_size, seqlen_q = q.shape[0], q.shape[1]
                    seqlen_k = kv.shape[1]
                    assert k.shape[3] == q.shape[3]
                    return flash_attn_func(
                        q,
                        k,
                        v,
                        self.drop.p if self.training else 0.0,
                        causal=self.causal,
                        softmax_scale=self.softmax_scale,
                        window_size=self.window_size,
                    ).to(dtype)


class SlideWindowSharedKeyAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_heads_k=None,
        num_heads_v=None,
        window_size=None,
        softmax_scale=None,
        causal=True,
        layer_idx=None,
        rotary_emb_dim=-1,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        dropout=0.,
        activation_dropout=0.,
        attention_dropout=0.,
        max_position_embeddings=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_heads_k = num_heads_k
        self.num_heads_v = num_heads_v

        self.causal = causal
        self.layer_idx = layer_idx

        assert USE_FLASH_ATTN, "pip install flash_attn"
        if window_size is not None and window_size > 0:
            self.window_size = (
                window_size // 2, 0) if self.causal else (window_size // 2, window_size // 2)
        else:
            self.window_size = (-1, 0) if self.causal else (-1, -1)

        self.head_dim = self.dim // self.num_heads
        assert self.head_dim * self.num_heads == self.dim

        assert self.num_heads % self.num_heads_k == 0
        assert self.num_heads % self.num_heads_v == 0

        self.rotary_emb_dim = rotary_emb_dim
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scale_base = rotary_emb_scale_base
        self.rotary_emb_interleaved = rotary_emb_interleaved

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings

        self.dropout_module = nn.Dropout(self.dropout)
        self.activation_dropout_module = nn.Dropout(self.activation_dropout)

        if self.rotary_emb_dim < 0:
            self.rotary_emb_dim = self.head_dim

        if self.rotary_emb_dim > 0:  # 0 -> nope
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
            )
            # self.rotary_emb = None
        else:
            self.rotary_emb = None

        scale = None
        self.softmax_scale = softmax_scale
        if self.softmax_scale is not None:
            if self.softmax_scale == "norm":
                scale = 1.
                self.register_buffer("s", torch.arange(
                    1., 16., step=self.num_heads).unsqueeze(-1))

        self.inner_attn = FlashAttention(
            causal=self.causal,
            softmax_scale=scale,
            attention_dropout=self.attention_dropout,
            window_size=self.window_size,
        )

        self.q_proj = nn.Linear(
            self.dim, self.head_dim * self.num_heads, bias=False)
        self.k_proj = nn.Linear(
            self.dim, self.head_dim * self.num_heads_k, bias=False)
        self.v_proj = nn.Linear(
            self.dim, self.head_dim * self.num_heads_v, bias=False)

        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

    def allocate_inference_cache(
        self,
        batch_size,
    ):
        param = next(self.parameters())
        key_caches = param.new_zeros(
            (batch_size, self.window_size[0], self.num_heads_k, self.head_dim))
        value_caches = param.new_zeros(
            (batch_size, self.window_size[0], self.num_heads_v, self.head_dim))

        if not is_torchdynamo_compiling():
            idx = self.layer_idx

            self.register_buffer(f"key_cache_{idx}", key_caches)
            key_caches = getattr(self, f"key_cache_{idx}")
            torch._dynamo.mark_static_address(key_caches)

            self.register_buffer(f"value_cache_{idx}", value_caches)
            value_caches = getattr(self, f"value_cache_{idx}")
            torch._dynamo.mark_static_address(value_caches)
            
        return key_caches, value_caches

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
            attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            rearrange(key_layer, "b l ... -> (b l) ..."), indices_k
        )
        value_layer = index_first_axis(
            rearrange(value_layer, "b l ... -> (b l) ..."), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                rearrange(query_layer, "b l ... -> (b l) ..."), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            raise NotImplementedError("Not implemented `cross_attention`")

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # pack cache of key and value, with other params
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        # print(attention_mask, attention_mask.all() if attention_mask is not None else None)

        batch_size, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k, v = map(lambda x: rearrange(
            x, "... (h d) -> ... h d", d=self.head_dim), (q, k, v))

        if self.softmax_scale is not None:
            if self.softmax_scale == "norm":
                q = F.normalize(q, dim=-1) * self.s
                k = F.normalize(k, dim=-1)

        if past_key_values is not None:
            assert self.causal
            # assert not self.training, "inference"
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            cached_seqlen = seqlen_offset + seq_len
            rotary_max_seqlen = self.max_position_embeddings if self.max_position_embeddings is not None else 0
            if rotary_max_seqlen < cached_seqlen:
                rotary_max_seqlen = cached_seqlen
        else:
            seqlen_offset = 0
            rotary_max_seqlen = self.max_position_embeddings if self.max_position_embeddings else 0
            if rotary_max_seqlen < seq_len:
                rotary_max_seqlen = seq_len

        if self.rotary_emb is not None:
            # TODO
            q, k = self.rotary_emb(
                q, k, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
            )
        q, k, v = map(lambda x: autocast_to_2B(x), (q, k, v))

        if past_key_values is not None:
            past_key_values.update(
                self.layer_idx,
                key_cache=k,
                value_cache=v,
                offset=seq_len,
            )
            if seqlen_offset > 0:
                assert seq_len == 1, "during inference, length of query should be equal to 1"
                key_caches, value_cahces = past_key_values.get_attn_states(
                    self.layer_idx)
                # k = key_caches[:, -cached_seqlen:, :, :]
                # v = value_cahces[:, -cached_seqlen:, :, :]
                k = key_caches[:, :cached_seqlen, :, :]
                v = value_cahces[:, :cached_seqlen, :, :]
                attention_mask = attention_mask[:, -k.shape[1]
                    :] if attention_mask is not None else None

        if self.num_heads_k != self.num_heads:
            k = repeat(k, "... h d -> ... (n h) d",
                       n=self.num_heads // self.num_heads_k)
        if self.num_heads_v != self.num_heads:
            v = repeat(v, "... h d -> ... (n h) d",
                       n=self.num_heads // self.num_heads_v)

        if attention_mask is not None:
            q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                q, k, v, attention_mask, seq_len,
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
        else:
            cu_seqlens_q, cu_seqlens_k = None, None
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = None, None

        context = self.inner_attn(
            q, k, v,
            cu_seqlens=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k
        )
        if attention_mask is not None:
            context = pad_input(context, indices_q, batch_size, seq_len)

        x_g = rearrange(context, "... h d -> ... (h d)")
        x_g = self.activation_dropout_module(x_g)
        out = self.out_proj(x_g.to(self.out_proj.weight.dtype))

        out = self.dropout_module(out)
        return out, past_key_values
