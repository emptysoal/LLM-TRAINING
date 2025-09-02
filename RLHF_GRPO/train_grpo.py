"""
    RLHF + GRPO with reward model
"""

import os
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    set_seed
)
from trl import GRPOConfig, GRPOTrainer

world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", -1))
rank = int(os.environ.get("RANK", 0))
print(f"[INFO] rank={rank}, local_rank={local_rank}, world_size={world_size}")
set_seed(42 + rank)  # 不同进程用不同种子

is_distributed = world_size > 1

# model_path = "Qwen/Qwen2.5-7B-Instruct"
model_path = "/workspace/Qwen/models/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

device_map = "auto" if not is_distributed else {"": local_rank}  # 分布式的话每个进程独占一张 GPU
# policy model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# 加载奖励模型
rw_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward_model",
    num_labels=1,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config,
)


# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


dataset = load_dataset("json", data_files="./data/dpo_zh_500.jsonl")
shuffled_train_dataset = dataset["train"].shuffle(seed=42)
split = shuffled_train_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


def process_func(example):
    prompts = []
    for text in example["question"]:
        prompt = f"<|im_start|>system\n你是一个问题解答助手<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)

    return {"prompt": prompts}


train_dataset = train_dataset.map(process_func, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(process_func, batched=True, remove_columns=eval_dataset.column_names)

grpo_config = GRPOConfig(
    output_dir="./rl_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    max_grad_norm=0.5,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    save_total_limit=3,
    report_to="tensorboard",
    num_generations=4,
    temperature=0.8,
    top_p=0.9,
    beta=0.1
)
if is_distributed:
    # torchrun 分布式关键设置：
    grpo_config.dataloader_drop_last = True
    grpo_config.ddp_find_unused_parameters = False

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=rw_model,
    args=grpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    reward_processing_classes=tokenizer,
    peft_config=peft_config
)

if rank == 0:
    print("开始训练...")
grpo_trainer.train()

if rank == 0:
    grpo_trainer.save_model("./rl_model")
    print("GRPO训练完成，模型已保存至 ./rl_model")
