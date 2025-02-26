import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from modules.utils import align_multiple
from ops.swiglu import swiglu_linear


class GLU(nn.Module):
    def __init__(
        self,
        dim,
        expand_ratio,
        dropout=0.,
        activation_dropout=0.,
        use_fast_path=True,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = align_multiple(int(dim * expand_ratio), 8)
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.use_fast_path = use_fast_path

        if self.use_fast_path:
            assert swiglu_linear is not None

        self.fc = nn.Linear(self.dim, self.ffn_dim * 2, bias=False)
        self.out_proj = nn.Linear(self.ffn_dim, self.dim, bias=False)

        self.dropout_module = nn.Dropout(self.dropout)
        self.activation_dropout_module = nn.Dropout(self.activation_dropout)

    def forward(self, x):
        """
        x: (B L D)
        """
        x, g = self.fc(x).chunk(2, -1)
        if self.use_fast_path and self.activation_dropout == 0.:
            y = swiglu_linear(g, x, self.out_proj.weight, self.out_proj.bias)
        else:
            x_g = F.silu(g) * x
            x_g = self.activation_dropout_module(x_g)
            y = self.out_proj(x_g)

        y = self.dropout_module(y)

        return y
