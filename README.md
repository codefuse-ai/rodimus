<div align=center>
<img src="assets/rodimus.jpg" width="280px">
</div>

<h2 align="center"> <a href="https://arxiv.org/abs/2410.06577">Rodimus&ast;: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions
</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align=center>

<!-- [![Demo](https://img.shields.io/badge/⚡-Hugging%20Face%20Demo-yellow.svg)](https://huggingface.co/spaces/Chat-UniVi/Chat-UniVi) -->
[![hf](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](TODO)
[![arXiv](https://img.shields.io/badge/Arxiv-2410.11842-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06577)
[![License](https://img.shields.io/badge/Code%20License-Apache2.0-yellow)](TODO)
</h5>

## Overview

We propose Rodimus&ast;, including Rodimus and Rodimus+, which tries to break the accuracy-efficency trade-off existing in Vanilla tranformers by introducing several innovative features. 

**Rodimus:**
* Linear attention-based, purely recurrent model.
* Incorporates Data-Dependent Tempered Selection (DDTS) for semantic compression.
* Reduced memory usage.

**Rodimus+:**
* Hybrid model combining Rodimus with Sliding Window Shared-Key Attention (SW-SKA).
* Enhances semantic, token, and head compression.

<div align=center>
<img src="assets/overview.png" height="" width="800px" style="margin-bottom:px"/> 
</div>

## Highlights

* **Constant memory footprint but better language modeling performance.**
<div align=center>
<img src="assets/memory.jpg" width="600px" style="margin-bottom:px"/> 
</div>

* **Better scaling performance than Transformer.**
<div align=center>
<img src="assets/scaling.jpg" height="" width="600px" style="margin-bottom:px"/> 
</div>

* **A real lite model, without memory complexity O(T) in KV cache.**

## Pretrained Checkpoints

The models enhanced by code and math datasets.

<div align=center>

| Model                    | Contexts | HuggingFace |
| ------------------------ | -------- | ----------- |
| Rodimus+-1.6B-Base     | 4096     |             |
| Rodimus+-1.6B-Instruct | 4096     |             |

</div>

## Quick Starts

### Installation

1. The latest version of <a href="https://github.com/huggingface/transformers">`transformers`</a> is recommended (at least 4.37.0). 
2. We evaluate our models with `python=3.8` and `torch==2.1.2`.
3. If you use Rodimus, you need to install <a href="https://github.com/sustcsonglin/flash-linear-attention">`flash-linear-attention`</a> and <a href="https://github.com/triton-lang/triton">`triton>=2.2.0`</a>. If you use Rodimus+, you need to further install <a href="https://github.com/Dao-AILab/flash-attention">`flash-attention`</a>. 

### Examples

 In `examples/generation_script.py`, we show a code snippet to show you how to use the model to generate:

```python
import os
import torch
from modeling_rodimus import RodimusForCausalLM
from tokenization_rodimus_fast import RodimusTokenizer

# load model
ckpt_dir = "model_path"
tokenizer = RodimusTokenizer.from_pretrained(ckpt_dir)
model = RodimusForCausalLM.from_pretrained(
    ckpt_dir,
    torch_dtype=torch.float16,
    device_map="cuda"
).eval()

# inference
input_prompt = "你好！你是谁？"
model_inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**model_inputs, max_length=32)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(response)
```

In `examples/chat_script.py`, we further show how to chat with Rodimus+:

```python
import os
import torch
from modeling_rodimus import RodimusForCausalLM
from tokenization_rodimus_fast import RodimusTokenizer

# load model
ckpt_dir = "model_path"
tokenizer = RodimusTokenizer.from_pretrained(ckpt_dir)
model = RodimusForCausalLM.from_pretrained(
    ckpt_dir,
    torch_dtype=torch.float16,
    device_map="cuda"
).eval()

# inference
input_prompt = "简单介绍一下大型语言模型。"
messages = [
    {"role": "HUMAN", "content": input_prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    system='You are Rodimus$+$, created by AntGroup. You are a helpful assistant.',
    tokenize=False,
)
print(text)
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**model_inputs, max_length=2048)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(response)
```

## Citation

If you find our work helpful, feel free to give us a cite.

```
@misc{he2024rodimusbreakingaccuracyefficiencytradeoff,
      title={Rodimus*: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions}, 
      author={Zhihao He and Hang Yu and Zi Gong and Shizhan Liu and Jianguo Li and Weiyao Lin},
      year={2024},
      eprint={2410.06577},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.06577}, 
}
```