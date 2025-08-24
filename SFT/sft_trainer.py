"""
    使用 trl 库的 SFTTrainer 做指令微调

    对 Standard prompt-completion 格式的数据集，即：{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
    进行 LoRA 监督微调
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

"""原加载数据集方式"""
# items = []
# with open("./sft_data.json", "r", encoding="utf8") as f:
#     for line in f:
#         item = json.loads(line)
#         items.append({"prompt": item["query"], "completion": item["answer"]})
# dataset = Dataset.from_list(items)


"""加载数据集"""
dataset = load_dataset("json", data_files="./sft_data.json", split="train")


def format_function(example):
    return {
        "prompt": example["query"],
        "completion": example["answer"]
    }


# 把原始数据集格式转换为 Standard prompt-completion 格式
dataset = dataset.map(format_function, remove_columns=["query", "answer"])

# 加载模型和tokenizer
# model_path = "Qwen/Qwen2.5-7B-Instruct"
model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

sft_config = SFTConfig(
    output_dir="./sft_output",
    learning_rate=2e-5,
    neftune_noise_alpha=10,
    per_device_train_batch_size=2,
    max_seq_length=512,
    num_train_epochs=3,
    logging_steps=10,
    logging_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    report_to="none",
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer
)

# 开始训练
print("开始SFT训练...")
trainer.train()

# 保存SFT模型
model.save_pretrained("./sft_model")
print("SFT训练完成，模型已保存至./sft_model")
