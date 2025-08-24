"""
    使用微调后的模型推理
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# base model
# model_path = "Qwen/Qwen2.5-7B-Instruct"
model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"
lora_path = "./sft_model"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    lora_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto"
)

# 加载微调后的模型
# model = PeftModel.from_pretrained(model, lora_path)


def generate_ad(text):
    prompt = f"""<|im_start|>system
你是一个问题解答助手。<|im_end|>
<|im_start|>user
根据以下的提问给出解答
{text}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取助手的回复部分
    response_text = result.split("assistant")[1] if "assistant" in result else result
    return response_text


# 示例输入
test_text = "天空为什么是蓝色的"
print(generate_ad(test_text))
