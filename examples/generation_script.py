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
