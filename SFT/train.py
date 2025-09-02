import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", -1))
rank = int(os.environ.get("RANK", 0))
print(f"[INFO] rank={rank}, local_rank={local_rank}, world_size={world_size}")
set_seed(42 + rank)  # 不同进程用不同种子

is_distributed = world_size > 1

# model_path = "Qwen/Qwen2.5-7B-Instruct"
# model_path = "/workspace/Qwen/models/Qwen2.5-7B-Instruct"
# model_path = "/workspace/Qwen/models/Qwen2.5-0.5B-Instruct"
# model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"
model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
device_map = "auto" if not is_distributed else {"": local_rank}  # 分布式的话每个进程独占一张 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config
)
model.config.use_cache = False

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

dataset = load_dataset("json", data_files="./data/DISC-Law-SFT-Pair-QA-released.jsonl", split="train").select(
    range(2000))


def preprocess_function(example):
    prompts = []
    for q, r in zip(example["input"], example["output"]):
        prompt = f"<|im_start|>system\n你是一个法律专家，请根据用户的问题给出专业的回答<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>"
        prompts.append(prompt)
    tokenized = tokenizer(prompts)
    return {"input_ids": tokenized["input_ids"]}


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["id", "input", "output"])
shuffled_dataset = tokenized_dataset.shuffle(seed=42)
split = shuffled_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

response_template_with_context = "<|im_start|>assistant\n"
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

sft_config = SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    neftune_noise_alpha=5,
    max_length=2048,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    report_to="tensorboard"
)
if is_distributed:
    # torchrun 分布式关键设置：
    sft_config.dataloader_drop_last = True
    sft_config.ddp_find_unused_parameters = False

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer
)

if rank == 0:
    print("开始 SFT 训练...")
trainer.train()

if rank == 0:
    trainer.save_model("./sft_model")
    print("SFT 训练完成，模型已保存至 ./sft_model")
