"""
    使用 transformers 库的 Trainer 做指令微调
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

"""加载数据集"""
dataset = load_dataset("json", data_files="./sft_data.json", split="train")


def format_function(example):
    return {
        "instruction": "根据以下的提问给出解答",
        "input": example["query"],
        "output": example["answer"]
    }


dataset = dataset.map(format_function)

# model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

# LoRA配置
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def preprocess_function(sample):
    # 构建提示
    prompt = f"""<|im_start|>system
你是一个问题解答助手。<|im_end|>
<|im_start|>user
{sample['instruction']}
{sample['input']}<|im_end|>
<|im_start|>assistant
"""

    max_length = 512

    # 对提示进行分词
    instruction = tokenizer(prompt, add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    # 对输出进行分词
    response = tokenizer(sample["output"], add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 应用预处理
tokenized_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

# 训练参数
training_args = TrainingArguments(
    output_dir="./sft_output",
    learning_rate=2e-5,
    neftune_noise_alpha=10,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    logging_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    report_to="none",
    fp16=True
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

# 开始训练
print("开始SFT训练...")
trainer.train()

# 保存SFT模型
model.save_pretrained("./sft_model")
print("SFT训练完成，模型已保存至./sft_model")
