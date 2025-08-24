"""
    使用 trl 库的 SFTTrainer 做指令微调

    对原始格式输入的数据集，在本案例中为：{"query": "<prompt text>", "answer": "<ideal generated text>"}
    使用 DataCollatorForCompletionOnlyLM，只对回答部分计算loss
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


# 加载数据集
dataset = load_dataset("json", data_files="./sft_data.json", split="train")

# 加载模型和tokenizer
# model_path = "Qwen/Qwen2.5-7B-Instruct"
model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

sft_config = SFTConfig(
    output_dir="./sft_output",
    learning_rate=2e-5,
    neftune_noise_alpha=10,
    per_device_train_batch_size=2,
    max_seq_length=512,
    num_train_epochs=3,
    logging_steps=10,
    logging_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    report_to="none",
    fp16=True
)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['query'])):
        # 这里是 facebook/opt-350m-chat 的 chat template
        # text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"

        # 把数据格式化为 Qwen/Qwen2.5-7B-Instruct 的 chat template
        text = f"<|im_start|>system\n你是一个问题解答助手<|im_end|>\n<|im_start|>user\n{example['query'][i]}<|im_end|>\n<|im_start|>assistant\n{example['answer'][i]}<|im_end|>"
        output_texts.append(text)
    return output_texts


response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    processing_class=tokenizer
)

# 开始训练
print("开始SFT训练...")
trainer.train()

# 保存SFT模型
model.save_pretrained("./sft_model")
print("SFT训练完成，模型已保存至./sft_model")
