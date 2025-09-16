"""
    使用 unsloth 做 SFT 微调
"""

from unsloth import FastLanguageModel
import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# model_path = "/workspace/Qwen/models/Qwen2.5-7B-Instruct"
model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"
# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=load_in_4bit,  # Use 4bit quantization to reduce memory usage. Can be False
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

dataset = load_dataset("json", data_files="./data/DISC-Law-SFT-Pair-QA-released.jsonl", split="train").select(
    range(2000))
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
    max_length=max_seq_length,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    report_to="tensorboard"
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
