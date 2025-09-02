"""
    RLHF + GRPO with custom reward functions
"""

import os
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)
from trl import GRPOConfig, GRPOTrainer
import jieba
from difflib import SequenceMatcher

# 初始化jieba分词
jieba.initialize()

world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", -1))
rank = int(os.environ.get("RANK", 0))
print(f"[INFO] rank={rank}, local_rank={local_rank}, world_size={world_size}")
set_seed(42 + rank)  # 不同进程用不同种子

is_distributed = world_size > 1

# model_path = "Qwen/Qwen2.5-7B-Instruct"
model_path = "/workspace/Qwen/models/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

device_map = "auto" if not is_distributed else {"": local_rank}  # 分布式的话每个进程独占一张 GPU
# policy model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch.bfloat16
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()


# 加载数据集时保留参考回答
dataset = load_dataset("json", data_files="./data/dpo_zh_500.jsonl")
shuffled_train_dataset = dataset["train"].shuffle(seed=42)
split = shuffled_train_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


def process_func(example):
    prompts = []
    answers = []  # 保存参考回答
    for text, answer in zip(example["question"], example["response_chosen"]):
        prompt = f"<|im_start|>system\n你是一个问题解答助手<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)
        answers.append(answer)

    return {"prompt": prompts, "answer": answers}


train_dataset = train_dataset.map(process_func, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(process_func, batched=True, remove_columns=eval_dataset.column_names)


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
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    save_total_limit=3,
    report_to="tensorboard",
    num_generations=4,
    max_prompt_length=1024,
    max_completion_length=1024
)
if is_distributed:
    # torchrun 分布式关键设置：
    grpo_config.dataloader_drop_last = True
    grpo_config.ddp_find_unused_parameters = False

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=[custom_reward_func],
    args=grpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config
)

if rank == 0:
    print("开始训练...")
grpo_trainer.train()

if rank == 0:
    grpo_trainer.save_model("./rl_model")
    print("GRPO训练完成，模型已保存至 ./rl_model")
