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
deepspeed==0.16.2  # 可选
```

## 运行

### SFT

- Single GPU

```bash
cd SFT
python train.py     # 使用 trl 库的 SFTTrainer
python train_02.py  # 使用 transformers 库的 Trainer
```

- DDP

```bash
cd SFT
./run_sft.sh        # 使用 trl 库的 SFTTrainer
./run_sft_02.sh     # 使用 transformers 库的 Trainer
```

- FSDP

```bash
cd SFT_FSDP
./run_sft.sh        # 使用 trl 库的 SFTTrainer
./run_sft_02.sh     # 使用 transformers 库的 Trainer
```

- DeepSpeed

```bash
cd SFT_DeepSpeed
# 若使用 ZeRO-2,确认 train.py 或 train_02.py 中的设置：deepspeed="ds_config_zero2.json"
./run_sft.sh        # 使用 trl 库的 SFTTrainer
./run_sft_02.sh     # 使用 transformers 库的 Trainer

# 若使用 ZeRO-3,确认 train.py 或 train_02.py 中的设置：deepspeed="ds_config_zero3.json"
./run_sft.sh        # 使用 trl 库的 SFTTrainer
./run_sft_02.sh     # 使用 transformers 库的 Trainer
```

### RLHF + PPO

```bash
cd RLHF_PPO
python train_reward.py
python train_ppo.py
```

### RLHF + GRPO

```bash
cd RLHF_GRPO

# 若使用奖励模型获取reward做强化学习
python train_reward.py
python train_grpo.py  # 单进程
./run_grpo.sh         # DDP GRPO

# 若使用自定义奖励函数计算reward
python train_02.py    # 单进程
./run_grpo_02.sh      # DDP GRPO
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

**DeepSpeed**


- https://huggingface.co/docs/transformers/v4.48.2/en/deepspeed





























