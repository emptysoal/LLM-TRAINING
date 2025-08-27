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
model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"
# model_path = "/workspace/text-generation-webui-customized/models/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
device_map = "auto" if not is_distributed else {"": local_rank}  # 分布式的话每个进程独占一张 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)
model.config.use_cache = False

peft_config = LoraConfig(
    r=8,
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules=["q_proj", "k_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)
if rank == 0:
    model.print_trainable_parameters()

dataset = load_dataset("json", data_files="./sft_data.json", split="train")


def preprocess_function(example):
    # prompt = example["query"] + example["answer"]
    prompt = f"<|im_start|>system\n你是一个问题解答助手<|im_end|>\n<|im_start|>user\n{example['query']}<|im_end|>\n<|im_start|>assistant\n{example['answer']}<|im_end|>"
    tokenized = tokenizer(prompt)
    return {"input_ids": tokenized["input_ids"]}


tokenized_dataset = dataset.map(preprocess_function, remove_columns=["query", "answer"])

response_template_with_context = "<|im_start|>assistant\n"
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

sft_config = SFTConfig(
    output_dir="./sft_output",
    learning_rate=2e-5,
    neftune_noise_alpha=5,
    per_device_train_batch_size=2,
    max_seq_length=512,
    num_train_epochs=4,
    logging_steps=2,
    save_steps=5,
    save_total_limit=3,
    report_to="none",
    fp16=True
)
if is_distributed:
    # torchrun 分布式关键设置：
    sft_config.dataloader_drop_last = True
    sft_config.ddp_find_unused_parameters = False

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=sft_config,
    processing_class=tokenizer,
    data_collator=data_collator
)

if rank == 0:
    print("开始 SFT 训练...")
trainer.train()

if rank == 0:
    trainer.save_model("./sft_model")
    print("SFT 训练完成，模型已保存至 ./sft_model")
