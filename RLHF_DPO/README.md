# DPO 直接偏好优化

## 环境

```bash
torch==2.5.0
datasets==4.0.0
transformers==4.48.2
trl==0.16.0
peft==0.14.0
bitsandbytes==0.47.0
deepspeed==0.16.2
```

## 运行

- 单进程

```bash
python train_dpo.py
```

- DDP

```bash
torchrun --nproc_per_node=2 train_dpo.py
```

- FSDP

```bash
torchrun --nproc_per_node=2 train_dpo_fsdp.py
```

- DeepSpeed

```bash
# ZeRO-2，确认 train_dpo_deepspeed.py 中的 deepspeed="ds_config_zero2.json"
torchrun --nproc_per_node=2 train_dpo_deepspeed.py
# 或
deepspeed --num_gpus 2 train_dpo_deepspeed.py



# ZeRO-3，确认 train_dpo_deepspeed.py 中的 deepspeed="ds_config_zero3.json"
torchrun --nproc_per_node=2 train_dpo_deepspeed.py
# 或
deepspeed --num_gpus 2 train_dpo_deepspeed.py
```
