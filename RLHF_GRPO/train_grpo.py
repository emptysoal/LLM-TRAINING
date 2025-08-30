"""
    RLHF + GRPO
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from trl import GRPOConfig, GRPOTrainer

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
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# 加载奖励模型
rw_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward_model",
    num_labels=1,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
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
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    num_train_epochs=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    logging_strategy="steps",
    eval_steps=30,
    save_steps=50,
    save_total_limit=3,
    report_to="tensorboard",
    num_generations=4
)

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

print("开始训练...")
grpo_trainer.train()

grpo_trainer.save_model("./rl_model")
