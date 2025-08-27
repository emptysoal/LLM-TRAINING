import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", -1))
rank = int(os.environ.get("RANK", 0))
print(f"[INFO] rank={rank}, local_rank={local_rank}, world_size={world_size}")
set_seed(42 + rank)  # 不同进程用不同种子

is_distributed = world_size > 1

dataset = load_dataset("json", data_files="./sft_data.json", split="train")


def format_function(example):
    return {
        "instruction": "根据以下的提问给出解答",
        "input": example["query"],
        "output": example["answer"]
    }


dataset = dataset.map(format_function, remove_columns=["query", "answer"])

# model_path = "Qwen/Qwen2.5-7B-Instruct"
model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"
# model_path = "/workspace/text-generation-webui-customized/models/Qwen2-1.5B-Instruct"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
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
device_map = "auto" if not is_distributed else {"": local_rank}  # 分布式的话每个进程独占一张 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)
model.config.use_cache = False

# LoRA配置
lora_config = LoraConfig(
    r=8,
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules=["q_proj", "k_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
if rank == 0:
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


tokenized_dataset = dataset.map(preprocess_function, remove_columns=["instruction", "input", "output"])

training_args = TrainingArguments(
    output_dir="./sft_output",
    learning_rate=2e-5,
    neftune_noise_alpha=5,
    per_device_train_batch_size=2,
    num_train_epochs=4,
    logging_steps=2,
    logging_strategy="steps",
    save_steps=5,
    save_total_limit=3,
    report_to="none",
    fp16=True
)
if is_distributed:
    # torchrun 分布式关键设置：
    training_args.dataloader_drop_last = True
    training_args.ddp_find_unused_parameters = False

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    processing_class=tokenizer
)

if rank == 0:
    print("开始 SFT 训练...")
trainer.train()

if rank == 0:
    trainer.save_model("./sft_model")
    print("SFT 训练完成，模型已保存至 ./sft_model")
