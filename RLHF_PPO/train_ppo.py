"""
    RLHF + PPO
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from trl import PPOConfig, PPOTrainer

# model_path = "Qwen/Qwen2.5-7B-Instruct"
# model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"
model_path = "/workspace/text-generation-webui-customized/models/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

# policy model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    # device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)
model.config.use_cache = False
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 加载奖励模型
rw_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward_model",
    num_labels=1,
    trust_remote_code=True,
    # device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)

value_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward_model",
    num_labels=1,
    trust_remote_code=True,
    # device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

dataset = load_dataset("json", data_files="./data/queries.json", split="train")


def format_function(example):
    outputs = tokenizer(example["query"], padding=False)
    return {"input_ids": outputs["input_ids"]}


queries_dataset = dataset.map(format_function, batched=True, remove_columns=["query"])

ppo_config = PPOConfig(
    output_dir="./rl_output",
    num_ppo_epochs=3,
    num_mini_batches=1,
    learning_rate=3e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4
)
ppo_trainer = PPOTrainer(
    args=ppo_config,
    processing_class=tokenizer,
    model=model,
    ref_model=None,
    reward_model=rw_model,
    value_model=value_model,
    train_dataset=queries_dataset,
    eval_dataset=queries_dataset
)
ppo_trainer.train()

ppo_trainer.save_model("./rl_model")
