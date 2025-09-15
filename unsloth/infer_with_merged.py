"""
    使用合并后的模型推理
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./sft_merged"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    dtype="auto"
)


def generate_ad(text):
    prompt = f"""<|im_start|>system
你是一个法律专家，请根据用户的问题给出专业的回答<|im_end|>
<|im_start|>user
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
test_text = "根据以下的提问给出解答:工作时间以外支付工资的标准?"
print(generate_ad(test_text))
