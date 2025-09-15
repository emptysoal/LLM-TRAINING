import torch
from unsloth import FastLanguageModel
from peft import PeftModel

model_path = "/workspace/text-generation-webui-customized/models/Qwen2.5-0.5B-Instruct"
# 加载基础模型
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=1024,
    dtype=torch.float16,
    load_in_4bit=False
)

# 加载 LoRA 适配器
lora_model = PeftModel.from_pretrained(base_model, "./sft_model")

# 合并模型
merged_model = lora_model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./sft_merged")
tokenizer.save_pretrained("./sft_merged")
