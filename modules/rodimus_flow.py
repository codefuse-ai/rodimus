import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from functools import partial

from transformers.cache_utils import Cache
from transformers.utils import is_torchdynamo_compiling

from modules.utils import (
    align_multiple,
    safe_eval_number,
    xavier_uniform_,
    reset_parameters_,
)

try:
    from causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
except:
    causal_conv1d_update = None
    causal_conv1d_fn = None

from fla.ops.gla import (
    fused_chunk_gla,
    chunk_gla,
    fused_recurrent_gla
)
from ops.layernorm import RMSNorm
from ops.layernorm_gated import RMSNorm as RMSNormGated


def _unsqueeze(x):
    return x.unsqueeze(1)


def _squeeze(x):
    return x.squeeze(1)


class ShortConv(nn.Module):
    def __init__(
        self,
        dim,
        d_conv=4,
        act="silu",  # silu or None
        use_fast_path=True,
        causal=True,
    ):
        super().__init__()
        self.dim = dim
        self.d_conv = d_conv
        self.use_fast_path = use_fast_path
        self.causal = causal

        if self.use_fast_path:
            assert causal_conv1d_fn is not None

        self.act = act
        self.conv1d = nn.Conv1d(
            in_channels=self.dim,
            out_channels=self.dim,
            bias=True,
            kernel_size=self.d_conv,
            groups=self.dim,
            padding=self.d_conv - 1,
        )

        if not self.causal:
            self.reverse_conv1d = nn.Conv1d(
                in_channels=self.dim,
                out_channels=self.dim,
                bias=True,
                kernel_size=self.d_conv,
                groups=self.dim,
                padding=self.d_conv - 1,
            )

        self._init_weights()

    def _init_weights(self,):
        # self.conv1d.reset_parameters()
        self.conv1d._is_hf_initialized = True
        if not self.causal:
            # self.reverse_conv1d.reset_parameters()
            self.reverse_conv1d.zero_()
            self.reverse_conv1d._is_hf_initialized = True

    def allocate_inference_cache(
        self,
        batch_size,
    ):
        param = next(self.parameters())
        conv_state = param.new_zeros(batch_size, self.dim, self.d_conv)
        return conv_state

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ):
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.)

        seq_len = x.size(1)
        if cache is not None and seq_len == 1:
            return self.step(x, cache)

        re_x = rearrange(x, "b l d -> b d l")

        # Update state (B D W)
        if cache is not None:
            cache.copy_(F.pad(re_x, (self.d_conv - re_x.shape[-1], 0)))

        if self.use_fast_path:
            re_weight = rearrange(self.conv1d.weight, "d 1 w -> d w")
            x = causal_conv1d_fn(
                x=re_x,
                weight=re_weight,
                bias=self.conv1d.bias,
                activation=self.act if self.causal else None,
            )

            if not self.causal:
                re_reverse_weight = rearrange(self.reverse_conv1d.weight, "d 1 w -> d w")
                x = x + causal_conv1d_fn(
                    x=re_x.flip(-1),
                    weight=re_reverse_weight,
                    bias=self.reverse_conv1d.bias,
                    activation=None,
                )
                if self.act is not None:
                    x = F.silu(x)
        else:
            x = self.conv1d(re_x)[..., :seq_len]
            if self.act is not None and self.causal:
                x = F.silu(x)

            if not self.causal:
                x = x + self.reverse_conv1d(re_x.flip(-1))[..., :seq_len]
                if self.act is not None:
                    x = F.silu(x)

        x = rearrange(x, "b d l -> b l d")
        return x

    def step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor
    ):
        assert x.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        x = x.squeeze(1)
        if self.use_fast_path:
            re_weight = rearrange(self.conv1d.weight, "d 1 w -> d w")
            x = causal_conv1d_update(
                x=x,
                conv_state=cache,
                weight=re_weight,
                bias=self.conv1d.bias,
                activation=self.act,
            )
        else:
            dtype = x.dtype
            cache.copy_(torch.roll(cache, shifts=-1, dims=-1))
            cache[:, :, -1] = x
            x = torch.sum(cache * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            if self.act is not None:
                x = F.silu(x)
        return x.unsqueeze(1)


class RodimusFlowInner(nn.Module):
    def __init__(
        self,
        d_inner,
        d_conv=4,
        mem_size=64,
        input_gate_low_rank="auto",
        mode="fused_chunk",
        norm_epsilon=1e-5,
        post_norm_epsilon=None,
        normalize_epsilon=None,
        residual_in_fp32=True,
        use_fast_path=True,
        layer_idx=None,
        causal=True,
    ):
        super().__init__()
        self.d_conv = d_conv
        self.d_inner = d_inner
        self.mem_size = mem_size
        self.input_gate_low_rank = input_gate_low_rank

        self.residual_in_fp32 = residual_in_fp32
        self.use_fast_path = use_fast_path
        self.mode = mode
        self.norm_epsilon = norm_epsilon
        self.post_norm_epsilon = post_norm_epsilon if post_norm_epsilon is not None else norm_epsilon
        self.normalize_epsilon = normalize_epsilon if normalize_epsilon is not None else 1e-12
        self.layer_idx = layer_idx

        self.scale = 1 / math.sqrt(self.mem_size)

        self.short_conv = ShortConv(
            self.d_inner,
            d_conv=self.d_conv,
            use_fast_path=self.use_fast_path,
            causal=True,
        )
        self.residual_weight = nn.Parameter(torch.ones(
            (self.d_inner, ), dtype=torch.float32 if self.residual_in_fp32 else None), requires_grad=True)
        self.residual_weight._no_weight_decay = True

        self.in_proj = nn.Linear(self.d_inner, self.mem_size * 2, bias=False)

        self.ch_gate_proj = nn.Sequential(nn.Linear(self.d_inner, self.input_gate_low_rank, bias=False),)
        self.ch_gate_proj.append(nn.Linear(self.input_gate_low_rank, self.d_inner, bias=True))
        self.ch_gate_proj.append(nn.Sigmoid())

        self.mem_gate_proj = nn.Linear(self.d_inner, self.mem_size * 2, bias=True)

        self._init_weights()

    def allocate_inference_cache(
        self,
        batch_size,
    ):
        param = next(self.parameters())
        conv_state = self.short_conv.allocate_inference_cache(batch_size)
        ssm_state = param.new_zeros(batch_size, 1, self.mem_size, self.d_inner,)

        if not is_torchdynamo_compiling():
            idx = self.layer_idx

            self.register_buffer(f"conv_state_{idx}", conv_state)
            conv_state = getattr(self, f"conv_state_{idx}")
            torch._dynamo.mark_static_address(conv_state)

            self.register_buffer(f"ssm_state_{idx}", ssm_state)
            ssm_state = getattr(self, f"ssm_state_{idx}")
            torch._dynamo.mark_static_address(ssm_state)

        return conv_state, ssm_state

    @torch.no_grad()
    def _init_weights(self):
        xavier_uniform_(self.in_proj.weight)

        sigmoid_bias_max = 0.999
        sigmoid_bias_min = 0.9
        init_floor = 1e-4

        xavier_uniform_(self.ch_gate_proj[0].weight)
        xavier_uniform_(self.ch_gate_proj[1].weight)

        bias = []
        max_ = 1 - sigmoid_bias_min
        min_ = 1 - sigmoid_bias_max
        rt_bias = torch.exp(
            torch.rand(self.mem_size) * (math.log(max_) - math.log(min_))
            + math.log(min_)
        ).clamp(min=init_floor)
        rt_bias = rt_bias + torch.log(-torch.expm1(-rt_bias))

        bias.append(rt_bias)

        tau_bias = torch.empty((self.mem_size,)).uniform_(1/16, 0.9)
        tau_bias = torch.logit(tau_bias.float()).to(tau_bias.dtype)
        bias.append(tau_bias)

        xavier_uniform_(self.mem_gate_proj.weight)

        if self.mem_gate_proj.bias.shape[0] > 0:
            bias = torch.cat([b.to(device=self.mem_gate_proj.bias.device) for b in bias], dim=0)
            self.mem_gate_proj.bias.copy_(bias)
            self.mem_gate_proj.bias._no_reinit = True
        else:
            import warnings
            warnings.warn('mem_gate_proj.bias cannot be initialized using the meta context. Please note that when loading a pre-trained model')

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ):
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode
        
        if past_key_values is not None:
            last_state = past_key_values.get_ssm_states(self.layer_idx)
            attention_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
        else:
            last_state = None

        if last_state is not None:
            conv_state, ssm_state = last_state
        else:
            conv_state, ssm_state = None, None

        if (
            ssm_state is not None
            and hidden_states.dtype != ssm_state.dtype
            and mode == "chunk"
        ):
            mode = "fused_chunk"

        shift_hidden_states = self.short_conv(
            hidden_states,
            mask=attention_mask,
            cache=conv_state,
        )
        residual = shift_hidden_states.float() if self.residual_in_fp32 else shift_hidden_states

        r, k = self.in_proj(shift_hidden_states).chunk(2, -1)

        u = self.ch_gate_proj(hidden_states) * hidden_states

        if attention_mask is not None:
            u = u.masked_fill(~attention_mask.unsqueeze(-1), 0.)

        mem_gates = F.linear(shift_hidden_states, self.mem_gate_proj.weight) + self.mem_gate_proj.bias.float()
        select_gate, tau_gate = mem_gates.chunk(2, -1)

        select_gate = F.softplus(select_gate)
        it_gate = select_gate
        rt_gate_log = -select_gate

        tau_gate = F.sigmoid(tau_gate)
        it_gate = it_gate ** tau_gate
        rt_gate_log = rt_gate_log * tau_gate

        k = F.normalize(k.float(), dim=-1, eps=self.normalize_epsilon)

        r, k, u, rt_gate_log = map(_unsqueeze, (r, k.float() * it_gate, u, rt_gate_log))

        if mode == 'fused_recurrent':
            o, ssm_state = fused_recurrent_gla(r, k, u, rt_gate_log, scale=self.scale,
                                                     initial_state=ssm_state, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            o, ssm_state = fused_chunk_gla(r, k, u, rt_gate_log, scale=self.scale,
                                                 initial_state=ssm_state, output_final_state=use_cache)
        elif mode == 'chunk':
            r, k, rt_gate_log = map(lambda x: x.to(u.dtype), (r, k, rt_gate_log))
            o, ssm_state = chunk_gla(r, k, u, rt_gate_log, scale=self.scale,
                                           initial_state=ssm_state, output_final_state=use_cache)

        if past_key_values is not None:
            past_key_values.update(
                self.layer_idx,
                conv_state=conv_state,
                ssm_state=ssm_state,
                offset=u.shape[-2],
            )
        o = (_squeeze(o) + residual * self.residual_weight).to(o.dtype)  # TODO: fused

        return o, past_key_values


class RodimusFlow(nn.Module):
    def __init__(
        self,
        dim,
        d_conv=4,
        expand_ratio=2,
        mem_size=64,
        dropout=0.,
        activation_dropout=0.,
        input_gate_low_rank="auto",
        mode="fused_chunk",
        norm_epsilon=1e-5,
        post_norm_epsilon=None,
        normalize_epsilon=None,
        residual_in_fp32=True,
        use_fast_path=True,
        layer_idx=None,
        causal=True,
    ):
        super().__init__()
        input_gate_low_rank = safe_eval_number(input_gate_low_rank)

        self.dim = dim
        self.d_conv = d_conv
        self.d_inner = align_multiple(int(dim * expand_ratio), 8)
        self.mem_size = mem_size
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.input_gate_low_rank = max(self.dim // 64, 16) if input_gate_low_rank == "auto" else input_gate_low_rank

        self.residual_in_fp32 = residual_in_fp32
        self.use_fast_path = use_fast_path
        self.mode = mode
        self.norm_epsilon = norm_epsilon
        self.post_norm_epsilon = post_norm_epsilon if post_norm_epsilon is not None else norm_epsilon
        self.normalize_epsilon = normalize_epsilon if normalize_epsilon is not None else 1e-12
        self.layer_idx = layer_idx
        self.causal = causal

        self.act_norm = RMSNormGated(self.d_inner, eps=self.norm_epsilon, norm_before_gate=False)

        self.fc = nn.Linear(self.dim, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.dim, bias=False)

        inner_cls = partial(
            RodimusFlowInner,
            d_inner=self.d_inner,
            d_conv=self.d_conv,
            mem_size=self.mem_size,
            input_gate_low_rank=self.input_gate_low_rank,
            mode=self.mode,
            norm_epsilon=self.norm_epsilon,
            post_norm_epsilon=self.post_norm_epsilon,
            normalize_epsilon=self.normalize_epsilon,
            residual_in_fp32=True,
            use_fast_path=True,
            layer_idx=layer_idx,
        )

        self.inner_mixer = inner_cls()
        if not self.causal:
            self.reverse_inner_mixer = inner_cls()

        self.dropout_module = nn.Dropout(self.dropout)
        self.activation_dropout_module = nn.Dropout(self.activation_dropout)

    def allocate_inference_cache(
        self,
        batch_size,
    ):
        assert not hasattr(self, "reverse_inner_mixer")
        return self.inner_mixer.allocate_inference_cache(batch_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ):
        x, g = self.fc(hidden_states).chunk(2, -1)

        o, past_key_values = self.inner_mixer(
            hidden_states=x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        if not self.causal:
            assert past_key_values is None
            assert use_cache is False
            reverse_o, _ = self.reverse_inner_mixer(
                hidden_states=x.flip(-2),
                attention_mask=attention_mask.flip(-1) if attention_mask is not None else None,
                past_key_values=None,
                use_cache=False,
                output_attentions=output_attentions,
            )
            o = o + reverse_o

        x_g = self.act_norm(o, g)
        x_g = self.activation_dropout_module(x_g)
        y = self.out_proj(x_g)

        y = self.dropout_module(y)

        return y, past_key_values
