# 大模型微调

- 这里以 `Qwen/Qwen2.5-7B-Instruct` 为例，包括SFT指令微调、RLHF强化学习。

## 环境

```bash
torch==2.5.0
datasets==4.0.0
transformers==4.48.2
trl==0.16.0
peft==0.14.0
bitsandbytes==0.47.0
```

## 运行

- SFT 指令微调

```bash
cd SFT
python train.py

# DDP
./run_sft.sh
```

- RLHF + PPO

```bash
cd RLHF_PPO
python train_reward.py
python train_ppo.py
```

- RLHF + GRPO

```bash
cd RLHF_GRPO
python train_reward.py
python train_grpo.py
```

## 参考

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