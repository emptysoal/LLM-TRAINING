"""
    LoRA微调训练一个奖励模型，用于 RLHF 强化学习
"""

import os
import torch
from datasets import load_dataset

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from trl import RewardTrainer, RewardConfig

world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", -1))
rank = int(os.environ.get("RANK", 0))
print(f"[INFO] rank={rank}, local_rank={local_rank}, world_size={world_size}")
set_seed(42 + rank)  # 不同进程用不同种子

is_distributed = world_size > 1

# model_path = "Qwen/Qwen2.5-7B-Instruct"
# model_path = "/workspace/Qwen/models/Qwen2.5-7B-Instruct"
model_path = "/workspace/Qwen/models/Qwen2.5-0.5B-Instruct"
# model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"
# model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device_map = "auto" if not is_distributed else {"": local_rank}  # 分布式的话每个进程独占一张 GPU
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch.bfloat16
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.SEQ_CLS,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files="./data/dpo_zh_500.jsonl")
shuffled_train_dataset = dataset["train"].shuffle(seed=42)
split = shuffled_train_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


def process_func(example):
    chosen = f"<|im_start|>system\n你是一个问题解答助手<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n{example['response_chosen']}<|im_end|>"
    rejected = f"<|im_start|>system\n你是一个问题解答助手<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n{example['response_rejected']}<|im_end|>"

    tokenized_chosen = tokenizer(chosen)
    tokenized_rejected = tokenizer(rejected)

    new_example = {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"]
    }
    return new_example


train_dataset = train_dataset.map(process_func, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(process_func, remove_columns=eval_dataset.column_names)

config = RewardConfig(
    output_dir="./reward_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-5,
    warmup_ratio=0.05,
    max_grad_norm=0.5,
    logging_steps=10,
    logging_strategy="steps",
    eval_steps=20,
    eval_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    report_to="tensorboard",
    max_length=2048
)
if is_distributed:
    # torchrun 分布式关键设置：
    config.dataloader_drop_last = True
    config.ddp_find_unused_parameters = False

trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

if rank == 0:
    print("开始训练...")
trainer.train()

if rank == 0:
    trainer.save_model("./reward_model")
    print("奖励模型训练完成，模型已保存至 ./reward_model")
