"""
    训练一个奖励模型，用于 RLHF 强化学习
"""

import torch
from datasets import load_dataset

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from trl import RewardTrainer, RewardConfig

# model_path = "Qwen/Qwen2.5-7B-Instruct"
# model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"
model_path = "/workspace/text-generation-webui-customized/models/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
# tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    # quantization_config=bnb_config,
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj"],
    task_type=TaskType.SEQ_CLS,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="./data/preference.json", split="train")


def process_func(example):
    chosen = example["question"] + example["chosen"]
    rejected = example["question"] + example["rejected"]
    # chosen = f"<|im_start|>system\n你是一个问题解答助手<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n{example['chosen']}<|im_end|>"
    # rejected = f"<|im_start|>system\n你是一个问题解答助手<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n{example['rejected']}<|im_end|>"

    tokenized_chosen = tokenizer(chosen, add_special_tokens=False)
    tokenized_rejected = tokenizer(rejected, add_special_tokens=False)

    new_example = {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"]
    }
    return new_example


dataset = dataset.map(process_func, remove_columns=["question", "chosen", "rejected"])
# print(dataset)

config = RewardConfig(
    output_dir="./reward_output",
    per_device_train_batch_size=1,
    # gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=5,
    logging_strategy="steps",
    report_to="none"
)

trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,
    args=config,
    train_dataset=dataset
)

trainer.train()
trainer.save_model("./reward_model")
