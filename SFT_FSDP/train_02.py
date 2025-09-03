import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", -1))
rank = int(os.environ.get("RANK", 0))
print(f"[INFO] rank={rank}, local_rank={local_rank}, world_size={world_size}")
set_seed(42 + rank)  # 不同进程用不同种子

# model_path = "Qwen/Qwen2.5-7B-Instruct"
# model_path = "/workspace/Qwen/models/Qwen2.5-7B-Instruct"
model_path = "/workspace/Qwen/models/Qwen2.5-0.5B-Instruct"
# model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"
# model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-0.5B-Instruct"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
model.config.use_cache = False

# LoRA配置
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
if rank == 0:
    model.print_trainable_parameters()

dataset = load_dataset("json", data_files="./data/DISC-Law-SFT-Pair-QA-released.jsonl", split="train").select(
    range(2000))


def preprocess_function(sample):
    max_length = 2048

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for q, r in zip(sample["input"], sample["output"]):
        # 对提示进行分词
        prompt = f"<|im_start|>system\n你是一个法律专家，请根据用户的问题给出专业的回答<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        instruction = tokenizer(prompt, add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        # 对输出进行分词
        response = tokenizer(r, add_special_tokens=False)

        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["id", "input", "output"])
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

training_args = TrainingArguments(
    output_dir="./sft_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    fp16=False,
    bf16=True,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    report_to="tensorboard" if rank == 0 else "none",
    remove_unused_columns=False,
    dataloader_drop_last=True,
    # --------- FSDP 关键配置 ---------
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
        "fsdp_backward_prefetch": "backward_pre",
        "fsdp_use_orig_params": True,  # 必须，兼容 peft
        "fsdp_offload_params": False,  # 显存不足可设为 True
        "fsdp_sync_module_states": True,
        "fsdp_state_dict_type": "FULL_STATE_DICT",
    }
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    processing_class=tokenizer
)

if rank == 0:
    print("开始 SFT 训练...")
trainer.train()

trainer.save_model("./sft_model")
if rank == 0:
    print("SFT 训练完成，模型已保存至 ./sft_model")
