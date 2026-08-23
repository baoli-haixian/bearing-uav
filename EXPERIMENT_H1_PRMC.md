# 实验 H1-P-RMC：并行旋转边缘化相关航向头

## 1. 实验定义

本实验在官方 `PARCASGM_v5a` 上增加完全独立的 P-RMC 航向分支：

```text
共享 GLUF/SGM
├── 官方 PSG + CA + 位置 Head -> position
└── RST/UAV nl_feat -> P-RMC -> heading
```

P-RMC 不输入位置预测，也不输入官方航向输出。仓库原有的
`PARCASGM_v5a_H1` 保留为旧 H1；本实验使用：

```text
model_class = PARCASGM_v5a_PRMC_H1
model_name  = phr5_h1_prmc_parallel
```

## 2. 模块和损失

四张 RST `nl_feat` 分别投影并池化为 `64 x 8 x 8`，按 `p1 p2 / p3 p4`
拼成 `64 x 16 x 16`。UAV 特征投影为 `64 x 8 x 8`，固定旋转 36 次。
每个旋转模板与 RST mosaic 的 81 个滑动窗口做掩码归一化相关，得到
`36 x 9 x 9` 响应；空间 logsumexp 得到 36-bin 方向分布，圆周期望输出
二维单位航向。

```text
L_heading = SmoothL1(h_pred, h) + 0.1 L_dist
L_total   = 0.8 L_position + 0.2 L_heading
```

`L_dist` 的 von Mises 软标签直接由数据集已有的 `(cos(theta), sin(theta))`
生成，不需要新增标签。主实验 `heading_detach_shared=False`，航向梯度可更新
共享 SGM；`True` 只用于梯度隔离消融。

## 3. 环境与数据

沿用官方环境和 Bearing-UAV-90K：

```text
Bearing_UAV_90K/
└── c4m_254k_96bc_b15_s100_v3d/
    └── metadata/metadata.csv
```

在项目根目录检查：

```bash
conda activate bearing_env
python -m unittest tests.test_experiment_h1_prmc -v
```

## 4. 从头训练 100 轮

```bash
python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_PRMC_H1 \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 \
  --rsi_id 96 \
  --n_sample 100 \
  --num_epochs 100 \
  --factor_bslr 0.5 \
  --learning_rate 5e-5 \
  --optimizer Adam \
  --device_id 0 \
  --gcth none \
  --checkpoint_interval 5 \
  --flag_ckpt 1 \
  --flag_test 1
```

显式设置为 batch size 16、Adam、learning rate `5e-5`、100 epochs、
SmoothL1 位置损失、`SmoothL1 + 0.1 L_dist` 航向损失，以及位置/航向权重
`0.8/0.2`。不使用实验 A 或其他实验 checkpoint，保证与官方 100 轮基线
进行公平的从头训练比较。

## 5. 评估

训练完成后 `--flag_test 1` 会自动加载 best checkpoint 并生成测试 JSON。
重点比较 UAV/3D：

```text
HSR@15：越高越好
MHE：越低越好
```

同时检查 Recall@1、LSR@15 和 MLE，确认航向梯度没有明显损害位置性能。
