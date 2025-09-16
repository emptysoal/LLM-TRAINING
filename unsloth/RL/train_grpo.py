"""
    RLHF + GRPO with custom reward functions
    使用 dpo_zh_500.jsonl 数据集
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import jieba
from difflib import SequenceMatcher

# 初始化jieba分词
jieba.initialize()

max_length = 2048  # Supports automatic RoPE Scaling, so choose any number

# model_path = "/workspace/Qwen/models/Qwen2.5-0.5B-Instruct"
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


def format_function(example):
    """
        这 2 种数据格式均可以，分别为：standard format 和 conversational format，对应下面的 2 个 prompt
        注意：在使用 conversational format 格式时，把 custom_reward_func 这个函数中的这行：
            completions = [completion[0]["content"] for completion in completions] 的注释解开，
            因为 conversational format 格式数据集返回的 completions 格式为：
            [[{"role": "assistant", "content": "..."}], [...], [...]]
        可参考官方文档：https://huggingface.co/docs/trl/grpo_trainer#using-a-custom-reward-function
    """

    prompts = []
    answers = []  # 保存参考回答
    for q, r in zip(example["question"], example["response_chosen"]):
        # prompt = f"<|im_start|>system\n你是一个问题解答助手<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        prompt = [
            {"role": "system", "content": "你是一个问题解答助手"},
            {"role": "user", "content": q}
        ]

        prompts.append(prompt)
        answers.append(r)
    return {"prompt": prompts, "answer": answers}


dataset = load_dataset("json", data_files="./data/dpo_zh_500.jsonl")["train"]
dataset = dataset.map(format_function, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.2, seed=42)


# ==================== 自定义奖励函数 ====================
def reward_func_jaccard(completions, answers, prompts=None, **kwargs):
    """
    基于Jaccard相似度的奖励函数
    计算生成文本和参考文本之间的词重叠程度
    """
    rewards = []

    for completion, answer in zip(completions, answers):
        # 中文分词
        comp_words = set(jieba.cut(completion))
        ans_words = set(jieba.cut(answer))

        # 计算Jaccard相似度
        if len(comp_words | ans_words) == 0:
            similarity = 0
        else:
            similarity = len(comp_words & ans_words) / len(comp_words | ans_words)

        rewards.append(similarity)

    return rewards


def reward_func_sequence(completions, answers, prompts=None, **kwargs):
    """
    使用SequenceMatcher计算字符串相似度
    考虑字符顺序，更适合短文本匹配
    """
    rewards = []

    for completion, answer in zip(completions, answers):
        similarity = SequenceMatcher(None, completion, answer).ratio()
        rewards.append(similarity)

    return rewards


def reward_func_keywords(completions, answers, prompts=None, **kwargs):
    """
    检查生成文本是否包含参考文本中的关键词
    适合需要确保特定信息点的场景
    """
    rewards = []

    for completion, answer in zip(completions, answers):
        # 从参考回答中提取关键词（长度大于1的词）
        answer_words = set(jieba.cut(answer))
        important_words = [word for word in answer_words if len(word) > 1]

        if not important_words:
            rewards.append(0)
            continue

        # 计算包含的关键词比例
        comp_words = set(jieba.cut(completion))
        matched_count = sum(1 for word in important_words if word in comp_words)
        keyword_ratio = matched_count / len(important_words)

        rewards.append(keyword_ratio)

    return rewards


def reward_func_comprehensive(completions, answers, prompts=None, **kwargs):
    """
    综合多种方法的奖励函数
    weights: (序列相似度权重, 关键词权重, Jaccard权重)
    """
    weights = (0.4, 0.3, 0.3)  # 可调整的权重

    # 计算各种相似度
    seq_rewards = reward_func_sequence(completions, answers)
    key_rewards = reward_func_keywords(completions, answers)
    jac_rewards = reward_func_jaccard(completions, answers)

    # 加权组合
    rewards = []
    for seq, key, jac in zip(seq_rewards, key_rewards, jac_rewards):
        total_reward = (weights[0] * seq +
                        weights[1] * key +
                        weights[2] * jac)
        rewards.append(total_reward)

    return rewards


# ==================== 主奖励函数 ====================
def custom_reward_func(completions, prompts=None, **kwargs):
    """
    主奖励函数，从数据集中获取参考回答并计算奖励
    """
    # 从数据集中获取对应的参考回答
    # 这里假设 prompts 和数据集中的顺序一致
    batch_size = len(completions)

    # 在实际训练中，GRPOTrainer会传递相应的数据
    # 我们需要从数据集中获取对应的answers
    if 'answer' in kwargs:
        answers = kwargs['answer']
    else:
        # 如果没有传入answers，使用默认值（实际训练中应该会有）
        answers = [""] * batch_size

    completions = [completion[0]["content"] for completion in completions]

    # 使用综合奖励函数
    rewards = reward_func_comprehensive(completions, answers, prompts)

    return rewards


grpo_config = GRPOConfig(
    output_dir="./rl_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    max_grad_norm=0.5,
    num_train_epochs=1,
    learning_rate=2e-7,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=4,
    eval_steps=50,
    save_steps=50,
    save_total_limit=3,
    report_to="tensorboard",
    num_generations=8,
    max_prompt_length=1024,
    max_completion_length=512
)

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=[custom_reward_func],
    args=grpo_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer
)

print("开始训练...")
grpo_trainer.train()

grpo_trainer.save_model("./rl_model")
print("GRPO训练完成，模型已保存至 ./rl_model")
