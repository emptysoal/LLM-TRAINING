"""
    RLHF + PPO
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from trl import PPOConfig, PPOTrainer

# model_path = "Qwen/Qwen2.5-7B-Instruct"
# model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"
model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-0.5B-Instruct"

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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

# policy model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 加载奖励模型
rw_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward_model",
    num_labels=1,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)
rw_model.config.pad_token_id = tokenizer.pad_token_id
rw_model.config.use_cache = False

value_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward_model",
    num_labels=1,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)
value_model.config.pad_token_id = tokenizer.pad_token_id
value_model.config.use_cache = False
value_model = prepare_model_for_kbit_training(value_model)
value_model = get_peft_model(value_model, peft_config)

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

    outputs = tokenizer(prompts, padding=False)
    return {"input_ids": outputs["input_ids"]}


train_dataset = train_dataset.map(process_func, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(process_func, batched=True, remove_columns=eval_dataset.column_names)

ppo_config = PPOConfig(
    output_dir="./rl_output",
    num_ppo_epochs=2,
    num_mini_batches=1,
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    logging_steps=10,
    eval_steps=30,
    save_steps=100,
    save_total_limit=3,
    report_to="tensorboard"
)
ppo_trainer = PPOTrainer(
    args=ppo_config,
    processing_class=tokenizer,
    model=model,
    ref_model=None,
    reward_model=rw_model,
    value_model=value_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
ppo_trainer.train()

ppo_trainer.save_model("./rl_model")
