from transformers.configuration_utils import PretrainedConfig

DEFAULT_MIXER_CFG = {

}
DEFAULT_ATTN_CFG = {

}


class RodimusConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]
    model_type = "rodimus"

    def __init__(
        self,
        block_type="rodimus",
        d_model=2048,
        n_layer=24,
        vocab_size=50277,
        norm_epsilon=1e-5,
        initializer_range=0.02,
        rescale_prenorm_residual=True,
        residual_in_fp32=True,
        use_fast_path=True,
        use_fused_cross_entropy=False,
        use_fused_swiglu=True,
        dropout=0.,
        activation_dropout=0.,
        attention_dropout=0.,
        mixer_cfg=DEFAULT_MIXER_CFG,
        attn_cfg=DEFAULT_ATTN_CFG,
        max_position_embeddings=2048,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=False,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=True,
        use_scale_embedding=False,
        use_norm_embedding=False,
        no_weight_decay_on_bias=True,
        no_weight_decay_on_norm=True,
        no_weight_decay_on_embedding=True,
        **kwargs,
    ):
        assert block_type in ["rodimus", "rodimus_plus"]
        self.block_type = block_type

        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.mixer_cfg = mixer_cfg
        self.attn_cfg = attn_cfg

        self.norm_epsilon = norm_epsilon
        self.initializer_range = initializer_range
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_fast_path = use_fast_path
        self.use_fused_cross_entropy = use_fused_cross_entropy
        self.use_fused_swiglu = use_fused_swiglu

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout

        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_cache = use_cache

        self.use_scale_embedding = use_scale_embedding
        self.use_norm_embedding = use_norm_embedding

        self.no_weight_decay_on_bias = no_weight_decay_on_bias
        self.no_weight_decay_on_norm = no_weight_decay_on_norm
        self.no_weight_decay_on_embedding = no_weight_decay_on_embedding

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
