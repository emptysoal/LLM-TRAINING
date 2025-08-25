# 大模型微调

- 本项目以 `Qwen/Qwen2.5-7B-Instruct` 为例

## 文件介绍

```bash
项目目录
   |---- SFT                          # 在自定义数据集上做SFT微调的几种方式
   |       |---- sft_data.json        # 用于微调的数据
   |       # 下面的 3 个是使用 trl 库的 SFTTrainer 做指令微调的不同实现方式
   |       |---- sft_trainer.py       # 对 Standard prompt-completion 格式的数据集做指令微调
   |       |---- sft_trainer_01.py    # 输入原始格式输入的数据集，格式化为 chat template，且只对回答部分计算loss
   |       |---- sft_trainer_02.py    # 输入 conversational format 格式的数据集，且只对回答部分计算loss
   |
   |       # 使用 transformers 库的 Trainer 做指令微调
   |       |---- sft_trainer_03.py
   |
   |       |---- infer.py             # 用于测试，加载SFT后的Lora模型进行推理
   |---- RLHF_PPO                     # 在自定义数据集上，使用 RLHF+PPO 做强化学习
   |       |---- data                 # 存放奖励模型和强化学习所用的数据集
   |       |---- train_reward.py      # 微调奖励模型
   |       |---- train_ppo.py         # 使用奖励模型，以及微调模型、价值模型等做强化学习
   |       |---- infer.py             # 使用强化学习微调后的模型推理
   |---- RLHF_GRPO
           |---- data                 # 存放奖励模型和强化学习所用的数据集
           |---- train_reward.py      # 微调奖励模型
           |---- train_grpo.py        # 使用奖励模型，以及微调模型、价值模型等做强化学习
           |---- infer.py             # 使用强化学习微调后的模型推理
```

## 环境要求

```bash
torch==2.5.0
datasets==4.0.0
transformers==4.48.2
trl==0.16.0
peft==0.14.0
```

## 开始微调

- SFT 指令微调

```bash
cd SFT
python sft_trainer{后缀}.py
```



- RLHF + PPO

```bash
cd RLHF_PPO
# 微调奖励模型
python train_reward.py
# 强化学习微调
python train_ppo.py
```

**备注：**

强化学习最开始应该先进行 SFT 监督微调

然后把微调后的 LoRA 和原 `Qwen2.5-7B-Instruct` 模型合并

把 `train_reward.py`  和 `train_ppo.py` 中的  `model_path` 替换为上面合并 LoRA 后的路径

最后执行上面的训练步骤



- RLHF + GRPO

```bash
cd RLHF_GRPO
# 微调奖励模型
python train_reward.py
# 强化学习微调
python train_grpo.py
```

**备注：**

同上

## 参考链接

**Trainer**

- https://developer.aliyun.com/article/1647599
- https://developer.aliyun.com/article/1641184

**SFTTrainer**

- https://huggingface.co/docs/trl/v0.16.0/en/sft_trainer

**RewardTrainer**

- https://huggingface.co/docs/trl/v0.16.0/en/reward_trainer

**PPOTrainer**

- https://huggingface.co/docs/trl/v0.16.0/en/ppo_trainer
- https://github.com/huggingface/trl/blob/v0.16.0/examples/scripts/ppo/ppo.py

**GRPOTrainer**

- https://huggingface.co/docs/trl/v0.16.0/en/grpo_trainer

- https://github.com/shibing624/MedicalGPT/blob/main/grpo_training.py


