import math
import warnings
from typing import List, Optional, Tuple, Union
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    BaseModelOutput,
    MaskedLMOutput
)
from transformers.cache_utils import Cache

from modules.cache import HybridCache
from modules.rodimus_flow import RodimusFlow
from modules.rodimus_attention import SlideWindowSharedKeyAttention
from modules.mlp import GLU

from ops.layernorm import RMSNorm

from configuration_rodimus import RodimusConfig

import logging
logger = logging.getLogger(__name__)

try:
    from fla.modules import FusedCrossEntropyLoss
except ImportError:
    FusedCrossEntropyLoss = None


def _apply_no_weight_decay_on_norm(module):
    from ops.layernorm_gated import RMSNorm as RMSNormWithGate
    from ops.layernorm_gated import LayerNorm as LayerNormWithGate

    if isinstance(module, RMSNorm) or isinstance(module, RMSNormWithGate):
        module.weight._no_weight_decay = True
    elif isinstance(module, nn.LayerNorm) or isinstance(module, LayerNormWithGate):
        module.weight._no_weight_decay = True
        module.bias._no_weight_decay = True


def _apply_no_weight_decay_on_embedding(module, lm_head_param=None):
    if isinstance(module, nn.Embedding):
        if lm_head_param is not None:
            if lm_head_param.weight != module.weight:
                module.weight._no_weight_decay = True
        else:
            logger.warning_once(
                "Unable to find the lm_head, forcibly set embedding's weight decay to 0.0")
            module.weight._no_weight_decay = True


def _set_no_weight_decay(
    module: nn.Module,
    no_weight_decay_on_bias=True,
    no_weight_decay_on_norm=True,
    no_weight_decay_on_embedding=False,
):
    if no_weight_decay_on_bias:
        for n, p in module.named_parameters():
            if n.endswith("bias") and p is not None:
                p._no_weight_decay = True

    if no_weight_decay_on_norm:
        module.apply(_apply_no_weight_decay_on_norm)

    if no_weight_decay_on_embedding:
        lm_head_param = None
        for n, p in module.named_parameters():
            if n.endswith("lm_head"):
                lm_head_param = p
                break
        module.apply(partial(_apply_no_weight_decay_on_embedding,
                     lm_head_param=lm_head_param))


def _init_weights(
    module: nn.Module,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    num_residuals_per_layer: int = 1,
    n_layer: int = 1,
):
    if isinstance(module, nn.Linear):
        if not getattr(module.weight, "_no_reinit", False):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        if not getattr(module.weight, "_no_reinit", False):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight"]:
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * n_layer)


class RodimusTrainedModel(PreTrainedModel):
    config_class = RodimusConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["RodimusBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        if self.config.block_type == "rodimus":
            self.num_residuals_per_layer = 1
        elif self.config.block_type == "rodimus_plus":
            self.num_residuals_per_layer = 3
        else:
            raise NotImplementedError()

    def _init_weights(
        self,
        module: nn.Module,
    ):
        _init_weights(
            module,
            initializer_range=self.config.initializer_range,
            rescale_prenorm_residual=self.config.rescale_prenorm_residual,
            num_residuals_per_layer=self.num_residuals_per_layer,
            n_layer=self.config.n_layer
        )


class RodimusBlock(nn.Module):
    def __init__(
        self,
        block_type,
        d_model,
        max_position_embeddings=None,
        mixer_cfg={},
        attn_cfg={},
        norm_epsilon=1e-5,
        residual_in_fp32=True,
        use_fast_path=True,
        use_fused_swiglu=True,
        layer_idx=None,
        causal=True,
        dropout=0.,
        activation_dropout=0.,
        attention_dropout=0.,
    ):
        super().__init__()
        self.block_type = block_type
        self.d_model = d_model
        self.norm_epsilon = norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.use_fast_path = use_fast_path
        self.use_fused_swiglu = use_fused_swiglu
        self.causal = causal

        attn_cfg = attn_cfg.copy()
        mixer_cfg = mixer_cfg.copy()

        self.mixer_norm = RMSNorm(self.d_model, eps=self.norm_epsilon)
        self.mixer = RodimusFlow(
            d_model, layer_idx=layer_idx, **mixer_cfg,
            use_fast_path=use_fast_path, residual_in_fp32=residual_in_fp32,
            causal=self.causal,
            dropout=dropout,
            activation_dropout=activation_dropout,
            norm_epsilon=self.norm_epsilon,
        )

        if self.block_type == "rodimus_plus":
            attn_cfg["num_heads"] = d_model // 128 if "num_heads" not in attn_cfg or attn_cfg["num_heads"] is None else attn_cfg["num_heads"]
            ffn_expand_ratio = attn_cfg.pop("ffn_expand_ratio", 4/3)

            self.attn_norm = RMSNorm(self.d_model, eps=self.norm_epsilon)
            self.attn = SlideWindowSharedKeyAttention(
                dim=d_model,
                **attn_cfg,
                layer_idx=layer_idx,
                causal=self.causal,
                dropout=dropout,
                activation_dropout=activation_dropout,
                attention_dropout=attention_dropout,
                max_position_embeddings=max_position_embeddings,
            )

            self.ffn_norm = RMSNorm(self.d_model, eps=self.norm_epsilon)
            self.ffn = GLU(
                d_model, ffn_expand_ratio,
                use_fast_path=use_fused_swiglu,
                dropout=dropout,
                activation_dropout=activation_dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        hidden_states, residual = self.mixer_norm(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32
        )
        hidden_states, past_key_values = self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        if self.block_type == "rodimus_plus":
            hidden_states, residual = self.attn_norm(
                hidden_states,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32
            )

            hidden_states, past_key_values = self.attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = self.ffn_norm(
                hidden_states,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32
            )
            hidden_states = self.ffn(hidden_states)

        return hidden_states, residual, past_key_values


class RodimusModel(RodimusTrainedModel):
    def __init__(
        self,
        config: RodimusConfig,
        causal=True,
    ):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        self.n_layer = config.n_layer
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id
        self.norm_epsilon = config.norm_epsilon
        self.residual_in_fp32 = config.residual_in_fp32
        self.use_fast_path = config.use_fast_path
        self.use_fused_swiglu = config.use_fused_swiglu
        self.causal = causal
        self.max_position_embeddings = config.max_position_embeddings

        self.RodimusConfig = config.block_type

        self.embeddings = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=self.padding_idx)

        if self.config.use_scale_embedding:
            mem_size = self.config.mixer_cfg['mem_size'] if 'mem_size' in self.config.mixer_cfg else 64
            self.embed_scale = math.sqrt(mem_size)
        else:
            self.embed_scale = 1.

        if self.config.use_norm_embedding:
            self.embed_norm = RMSNorm(self.d_model, eps=self.norm_epsilon)
        else:
            self.embed_norm = None

        self.layers = nn.ModuleList([])

        for i in range(self.n_layer):
            block = RodimusBlock(
                self.config.block_type,
                self.d_model,
                layer_idx=i,
                max_position_embeddings=self.max_position_embeddings,
                mixer_cfg=self.config.mixer_cfg,
                attn_cfg=self.config.attn_cfg,
                norm_epsilon=self.norm_epsilon,
                residual_in_fp32=self.residual_in_fp32,
                use_fast_path=self.use_fast_path,
                use_fused_swiglu=self.use_fused_swiglu,
                causal=self.causal,
                dropout=self.config.dropout,
                activation_dropout=self.config.activation_dropout,
                attention_dropout=self.config.attention_dropout,
            )

            self.layers.append(block)

        self.norm_f = RMSNorm(self.d_model, eps=self.norm_epsilon)

        self.has_ssm = hasattr(self.layers[0], "mixer")
        self.has_attn = hasattr(self.layers[0], "attn")
        assert self.has_ssm or self.has_attn

        _set_no_weight_decay(
            self,
            no_weight_decay_on_bias=self.config.no_weight_decay_on_bias,
            no_weight_decay_on_norm=self.config.no_weight_decay_on_norm,
            no_weight_decay_on_embedding=False,  # do this at `RodimusForCausalLM`
        )

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        if output_attentions:
            warnings.warn(
                "`Model` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (
            self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask if False in attention_mask else None
            else:
                attention_mask = attention_mask if 0.0 in attention_mask else None
        else:
            attention_mask = None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        inputs_embeds *= self.embed_scale

        if self.embed_norm is not None:
            inputs_embeds = self.embed_norm(inputs_embeds)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            if past_key_values is None:  # init states
                past_key_values = []
                for layer in self.layers:
                    cache = ()
                    if self.has_ssm:
                        cache += layer.mixer.allocate_inference_cache(
                            hidden_states.size(0))
                    if self.has_attn:
                        cache += layer.attn.allocate_inference_cache(
                            hidden_states.size(0))
                    past_key_values.append(cache)
            if not isinstance(past_key_values, HybridCache):
                past_key_values = HybridCache.from_legacy_cache(
                    past_key_values=past_key_values,
                    seen_tokens=0,
                    has_ssm=self.has_ssm,
                    has_attn=self.has_attn,
                )
        else:
            past_key_values = None

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        residual = None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            if self.gradient_checkpointing and self.training:
                hidden_states, residual, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    residual,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                )
            else:
                hidden_states, residual, past_key_values = layer(
                    hidden_states=hidden_states,
                    residual=residual,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

        hidden_states = self.norm_f(
            hidden_states,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32
        )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = past_key_values.to_legacy_cache()
        if not return_dict:
            return tuple(x for x in [hidden_states, next_cache, all_hidden_states, all_attns] if x is not None)

        if self.causal:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_attns
            )
        else:
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attns
            )


class RodimusForCausalLM(RodimusTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: RodimusConfig
    ):
        super().__init__(config)
        self.config = config

        self.model = RodimusModel(config, causal=True)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        _set_no_weight_decay(
            self,
            no_weight_decay_on_bias=False,
            no_weight_decay_on_norm=False,
            no_weight_decay_on_embedding=(
                not self.config.tie_word_embeddings) and self.config.no_weight_decay_on_embedding,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is passed along.
        if past_key_values is not None:
            if not isinstance(past_key_values, HybridCache):
                past_key_values = HybridCache.from_legacy_cache(
                    past_key_values=past_key_values,
                    seen_tokens=input_ids.shape[1] - 1,
                    has_ssm=self.model.has_ssm,
                    has_attn=self.model.has_attn,
                )
            # input_ids, attention_mask = input_ids[:, -1:], attention_mask[:, -1:]
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        })
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = hidden_states
        logits = logits.float()

        loss = None
        if labels is not None and logits is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if FusedCrossEntropyLoss is not None and self.config.use_fused_cross_entropy:
                loss_fct = FusedCrossEntropyLoss(inplace_backward=True)
            else:
                loss_fct = nn.CrossEntropyLoss()

            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

