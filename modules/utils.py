import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


def align_multiple(value, multiple_size=8):
    if value % multiple_size != 0:
        value += multiple_size - (value % multiple_size)
    return value


def safe_eval_number(s):
    if s is None:
        return s
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def autocast_to_2B(x):
    if x.dtype not in {torch.float16, torch.bfloat16}:
        return x.to(dtype=torch.bfloat16)
    else:
        return x


def xavier_uniform_(weight, gain=2 ** -2.5):
    nn.init.xavier_uniform_(weight, gain=2 ** -2.5)
    weight._no_reinit = True


def reset_parameters_(linear_module):
    linear_module.reset_parameters()
    linear_module._is_hf_initialized = True
