"""
    一个普通的 SFT 微调代码，使用 transformers、trl、peft 库；
    因为安装 unsloth 时，torch和上面的库都安装了最新版本，所以用这个代码在新版本的库上测试普通 SFT 微调；
    测试结果：
        1。下面的LoRA微调测试成功
        2. 在新版本上量化支持的并不好，所以 QLoRA 无法跑通
"""

import os
import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", -1))
rank = int(os.environ.get("RANK", 0))
print(f"[INFO] rank={rank}, local_rank={local_rank}, world_size={world_size}")

is_distributed = world_size > 1

set_seed(42)

model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device_map = "auto" if not is_distributed else {"": local_rank}  # 分布式的话每个进程独占一张 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=device_map,
    dtype=torch.bfloat16
)
# model.config.use_cache = False
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


def format_function(example):
    texts = []
    for q, r in zip(example["input"], example["output"]):
        text = f"<|im_start|>system\n你是一个法律专家，请根据用户的问题给出专业的回答<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>"
        texts.append(text)
    return {"text": texts}


dataset = load_dataset("json", data_files="./data/DISC-Law-SFT-Pair-QA-released.jsonl", split="train").select(
    range(200))
dataset = dataset.map(format_function, batched=True, remove_columns=["id", "input", "output"])
dataset = dataset.train_test_split(test_size=0.2, seed=42)

training_args = SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    max_length=1024,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    report_to="tensorboard",
    # bf16=False,
    # fp16=True
)
if is_distributed:
    # torchrun 分布式关键设置：
    training_args.dataloader_drop_last = True
    training_args.ddp_find_unused_parameters = False

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer
)

if rank == 0:
    print("开始 SFT 训练...")
trainer.train()

if rank == 0:
    trainer.save_model("./sft_model")
    print("SFT 训练完成，模型已保存至 ./sft_model")
