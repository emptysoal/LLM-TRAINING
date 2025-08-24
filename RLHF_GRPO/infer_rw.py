"""
    使用微调后的模型推理
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

# model_path = "Qwen/Qwen2.5-7B-Instruct"
# model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"
lora_path = "./reward_model"

tokenizer = AutoTokenizer.from_pretrained(lora_path, use_fast=False, trust_remote_code=True)
# tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForSequenceClassification.from_pretrained(
    lora_path,
    num_labels=1,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    # quantization_config=bnb_config,
)


def generate(question, answer):
    # prompt = f"<|im_start|>system\n你是一个问题解答助手<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
    prompt = question + answer

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    outputs = model(**inputs).logits.squeeze(-1).cpu().numpy()

    print(outputs)

    # result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # # 提取助手的回复部分
    # response_text = result.split("assistant")[1] if "assistant" in result else result
    # return response_text


# 示例输入
question_text = "天空是什么颜色的？"
answer_text = "蓝色的"
generate(question_text, answer_text)
