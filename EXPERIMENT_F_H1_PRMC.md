# 实验 F + H1-P-RMC：全局位置先验与并行旋转相关航向头

## 1. 与旧 H1-P-RMC 的关系

旧模型 `PARCASGM_v5a_PRMC_H1` 建立在官方 `PARCASGM_v5a` 上，不包含实验 F。
为保留旧实验和 checkpoint，本实验新增类而不覆盖旧类：

```text
PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1
```

模型标识：

```text
phr5_f_h1_prmc
```

## 2. 联合结构

```text
共享 VGG/Non-local 特征
├── 4 RST nl_feat -> GlobalRST -> PSG position prior
│   original CA context + UAV descriptor + prior -> position head
└── original RST/UAV nl_feat -> P-RMC
    36 rotated UAV templates x 81 RST windows -> heading distribution -> heading
```

GlobalRST 只接收位置损失梯度，P-RMC 只接收航向损失梯度；两者从头联合训练，
不先训练 F 100 轮再追加训练。

```text
L = 0.8 * L_position + 0.2 * L_heading
L_heading = SmoothL1(h_pred, h_gt) + 0.1 * L_distribution
```

分布软标签由数据集已有的 `(x_cosa, y_sina)` 生成，不增加数据标签。

## 3. 环境与数据

```bash
cd /data/xuly/cmq/bearinguav
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav:${PYTHONPATH}
export TORCH_HOME=/data/xuly/cmq/bearinguav/.cache/torch
test -f Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

数据集地址：

```text
https://huggingface.co/datasets/HaoyZhou/bearinguav/tree/main
```

公平主实验仅使用 ImageNet VGG-16 初始化，从头训练 100 epoch。官方完整
Bearing-UAV 权重以及实验 F 权重均不作为 `--resume` 或 `--init_checkpoint`。

## 4. 测试

```bash
python -m unittest tests.test_experiment_f_h1_prmc -v
```

## 5. 单轮冒烟训练

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1 \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 \
  --rsi_id 96 \
  --n_sample 100 \
  --num_epochs 1 \
  --factor_bslr 0.5 \
  --device_id 0 \
  --gcth none \
  --checkpoint_interval 1 \
  --flag_ckpt 1 \
  --flag_test 0
```

## 6. 从头联合训练 100 轮

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1 \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 \
  --rsi_id 96 \
  --n_sample 100 \
  --num_epochs 100 \
  --factor_bslr 0.5 \
  --device_id 0 \
  --gcth none \
  --checkpoint_interval 5 \
  --flag_ckpt 1 \
  --flag_test 1
```

参数对应 batch size 16、learning rate `5e-5`、Adam、ReduceLROnPlateau、
weight decay 0、85/5/10 划分和 seed 42。重点比较 UAV/3D 的 MHE、HSR@15，
同时检查实验 F 的 MLE、Recall@1 和 LSR@15 是否保留。
