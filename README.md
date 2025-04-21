<div align=center>
<img src="assets/rodimus.jpg" width="280px">
</div>

<h2 align="center"> <a href="https://openreview.net/forum?id=IIVYiJ1ggK">Rodimus&ast;: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions
</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align=center>

<!-- [![Demo](https://img.shields.io/badge/⚡-Hugging%20Face%20Demo-yellow.svg)](https://huggingface.co/spaces/Chat-UniVi/Chat-UniVi) -->
[![hf](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](https://huggingface.co/)
[![ModelScope](https://img.shields.io/badge/🤖-ModelScope-3771C8.svg)](https://modelscope.cn)
[![ICLR](https://img.shields.io/badge/ICLR-2025-orange?logo=iclryear)](https://openreview.net/forum?id=IIVYiJ1ggK)
[![License](https://img.shields.io/badge/Code%20License-Apache2.0-yellow)](https://choosealicense.com/licenses/apache-2.0/)
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

**Rodimus+Coder:**
* We train and opensource the lightweight Rodimus+-Coder model, available in 1.6B and 4B sizes, achieving performance surpassing SOTA models of similar sizes.

<div align=center>
<img src="assets/rodimus-plus-coder-chat-evaluation.png" height="" width="800px" style="margin-bottom:px"/> 
</div>

You can download the following table to see the various parameters for your use case. If you are located in mainland China, we also provide the model on modelscope.cn to speed up the download process.

<div align="center">

|     **Model**      | **#Total Params** | **Context Length** | **Download** |
| :----------------: | :---------------: | :-------------------: | :----------: |
| Rodimus+-Coder-1.6B-Base |       1.6B       |        4K         |      [🤗 HuggingFace](https://huggingface.co/codefuse-ai/Rodimus-Plus-Coder-1.6B) <br> [🤖 ModelScope](https://modelscope.cn/models/codefuse-ai/Rodimus-Plus-Coder-1.6B) |
| Rodimus+-Coder-1.6B-Chat |       1.6B       |        4K         |      [🤗 HuggingFace](https://huggingface.co/codefuse-ai/Rodimus-Plus-Coder-1.6B-Chat) <br> [🤖 ModelScope](https://modelscope.cn/models/codefuse-ai/Rodimus-Plus-Coder-1.6B-Chat) |
| Rodimus+-Coder-4B-Base |       4B       |        4K         |      [🤗 HuggingFace](https://huggingface.co/codefuse-ai/Rodimus-Plus-Coder-4B) <br> [🤖 ModelScope](https://modelscope.cn/models/codefuse-ai/Rodimus-Plus-Coder-4B) |
| Rodimus+-Coder-4B-Chat |       4B       |        4K         |      [🤗 HuggingFace](https://huggingface.co/codefuse-ai/Rodimus-Plus-Coder-4B-Chat) <br> [🤖 ModelScope](https://modelscope.cn/models/codefuse-ai/Rodimus-Plus-Coder-4B-Chat) |

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

### Benchmark Checkpoints

> This checkpoints completed training before submitting the paper, used to reproduce the benchmarks in the paper. 
> 
> If you want to use the more practical model, we strongly recommand you to download the checkpionts in **Latest Checkpoints**.

<div align=center>

| Model (2024/10/01)                   | Contexts | HuggingFace | ModelScope
| ------------------------ | -------- | ----------- | ----------- |
| Rodimus-1.4B-Base     | 2048     |   <a href="https://huggingface.co/codefuse-admin/rodimus_1B4_base_20241001">link</a>          | <a href="https://www.modelscope.cn/models/codefuse-ai/rodimus_1B4_base_20241001/summary">link</a>          |
| Rodimus+-1.6B-Base | 2048     |   <a href="https://huggingface.co/codefuse-ai/rodimus_plus_1B6_base_20241001">link</a>           | <a href="https://www.modelscope.cn/models/codefuse-ai/rodimus_plus_1B6_base_20241001/summary">link</a>          |
| Rodimus+-Coder-1.6B-Base | 4096     |   <a href="https://huggingface.co/codefuse-ai/rodimus_plus_coder_1B6_base_20241001">link</a>           | <a href="https://www.modelscope.cn/models/codefuse-ai/rodimus_plus_coder_1B6_base_20241001/summary">link</a>          |

</div>

The `Rodimus+-Coder-1.6B-Base` is the model enhanced by multi-stage training with math and code datasets in the paper.

### Latest Checkpoints

> This checkpoints contain the **latest checkpoints** of Rodimus* trained by continuously updated data, for continuous training or actual use.

<div align=center>

| Model                   | Date | HuggingFace | ModelScope
| ------------------------ | -------- | ----------- | ----------- |
| Rodimus+-1.6B-Base | 2025/02/15     |   <a href="https://huggingface.co/codefuse-ai/rodimus_plus_1B6_base_20250215">link</a>           | <a href="https://www.modelscope.cn/models/codefuse-ai/rodimus_plus_1B6_base_20250215/summary">link</a>          |

</div>

## Quick Starts

### Installation

1. The latest version of <a href="https://github.com/huggingface/transformers">`transformers`</a> is recommended (at least 4.42.0). 
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
@inproceedings{
he2025rodimus,
title={Rodimus*: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions},
author={Zhihao He and Hang Yu and Zi Gong and Shizhan Liu and Jianguo Li and Weiyao Lin},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=IIVYiJ1ggK}
}
```
