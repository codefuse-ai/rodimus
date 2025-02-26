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
