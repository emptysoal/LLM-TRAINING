# 使用 accelerate 启动分布式训练

## 简介

1. 对单进程训练、DDP、DeepSpeed，均使用 `train.py`
2. 对 FSDP，均使用 `train_fsdp.py`
3. FSDP 和 DeepSpeed ZeRO-3 不支持 QLoRA
4. DDP 在使用 QLoRA 时，要把 `train.py` 中的 `device_map = "auto" if not is_distributed else None` 改为： `device_map = "auto" if not is_distributed else {"": local_rank}`

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

**注意：** 以下命令中，和 `yaml` 有关的，记得把 `yaml` 配置文件中的 `num_processes` 参数改为和自己设备 `GPU` 的数量

### LoRA

- 单进程

```bash
python train.py
```

- DDP

```bash
torchrun --nproc_per_node=2 train.py
# 或者
accelerate launch --config_file ./accelerate_configs/multi_gpu.yaml train.py
```

- FSDP

```bash
accelerate launch --config_file ./accelerate_configs/fsdp.yaml train_fsdp.py
```

- DeepSpeed  ZeRO-2

```bash
accelerate launch --config_file ./accelerate_configs/deepspeed_zero2.yaml train.py
```

- DeepSpeed  ZeRO-3

```bash
accelerate launch --config_file ./accelerate_configs/deepspeed_zero3.yaml train.py
```

## QLoRA

把 `train.py` 中的 39 行 ： `quantization_config=bnb_config` 的注释解开

- 单进程

```bash
python train.py
```

- DDP

还要把 `train.py` 中的 `device_map = "auto" if not is_distributed else None` 改为： `device_map = "auto" if not is_distributed else {"": local_rank}`

```bash
torchrun --nproc_per_node=2 train.py
# 或者
accelerate launch --config_file ./accelerate_configs/multi_gpu.yaml train.py
```

- DeepSpeed  ZeRO-2

```bash
accelerate launch --config_file ./accelerate_configs/deepspeed_zero2.yaml train.py
```