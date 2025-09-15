"""
    使用 unsloth 做 SFT 微调
"""

import torch
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

max_length = 1024  # Supports automatic RoPE Scaling, so choose any number

model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-0.5B-Instruct"
# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_length,
    dtype=torch.float16,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Dropout = 0 is currently optimized
    bias="none",  # Bias = "none" is currently optimized
    use_gradient_checkpointing=True,
    random_state=3407,
)

dataset = load_dataset("json", data_files="./data/DISC-Law-SFT-Pair-QA-released.jsonl", split="train").select(
    range(200))
dataset = dataset.train_test_split(test_size=0.2, seed=42)


def formatting_func(example):
    # 根据你的数据格式调整这里
    # 通常格式是: <|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>
    texts = []
    for q, r in zip(example["input"], example["output"]):
        text = f"<|im_start|>system\n你是一个法律专家，请根据用户的问题给出专业的回答<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>"
        texts.append(text)
    return texts


training_args = SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    max_length=max_length,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    report_to="tensorboard",
    bf16=False,
    fp16=True
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    formatting_func=formatting_func,
    processing_class=tokenizer
)

print("开始 SFT 训练...")
trainer.train()

trainer.save_model("./sft_model")
print("SFT 训练完成，模型已保存至 ./sft_model")
