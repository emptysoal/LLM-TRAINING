"""
    DPO 直接偏好优化
"""

import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import DPOConfig, DPOTrainer

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

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
model.config.use_cache = False
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)
if rank == 0:
    model.print_trainable_parameters()


# def preprocess_function(example):
#     return {
#         "prompt": [{"role": "user", "content": example['question']}],
#         "chosen": [{"role": "assistant", "content": example['response_chosen']}],
#         "rejected": [{"role": "assistant", "content": example['response_rejected']}]
#     }


def preprocess_function(example):
    return {
        "prompt": f"<|im_start|>system\n你是专业的问题解答助手<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n",
        "chosen": f"{example['response_chosen']}<|im_end|>",
        "rejected": f"{example['response_rejected']}<|im_end|>"
    }


dataset = load_dataset("json", data_files="./data/dpo_zh_500.jsonl")["train"]
dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

dpo_config = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    save_total_limit=3,
    report_to="tensorboard" if rank == 0 else "none",
    remove_unused_columns=False,
    dataloader_drop_last=True,
    deepspeed="ds_config_zero2.json"
)

dpo_trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

if rank == 0:
    print("开始 DPO 训练...")
dpo_trainer.train()

dpo_trainer.save_model("./dpo_model")
if rank == 0:
    print("DPO 训练完成，模型已保存至 ./dpo_model")
