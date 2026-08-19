# 实验 H3-D：GPRV-H 几何位姿体航向模块

## 1. 实验目标

H3-D 保留官方 `PARCASGM_v5a` 的位置分支和输出接口，只增加独立航向头：

```text
PARCASGM_v5a_GPRVH
```

模块从官方 SGM 的四张 RST `nl_feat` 与 UAV `nl_feat` 构造
`9 x 9 x 36` 的位置-航向联合概率体，并使用双频相位残差和 180 度反向排序约束。
默认冻结实验 A 基线，所以位置预测与实验 A 保持一致。

理论依据包括 OrienterNet 的离散位姿概率体、Geometry-Guided Cross-View
Transformer 的几何方向对齐、DSM 的循环相关，以及 Phase-Shifting Coder 的
周期角编码。H3-D 不需要新增数据标签。

## 2. 数据与标签

数据目录应为：

```text
Bearing_UAV_90K/
└── c4m_254k_96bc_b15_s100_v3d/
    └── metadata/metadata.csv
```

使用的现有字段：

- `x_norm`, `y_norm`：位置软标签中心；
- `x_cosa`, `y_sina`：航向单位向量；
- `p1_path` 至 `p4_path`, `target_path`：四张 RST 与一张 UAV 图像。

当前代码固定使用 seed 42 和 85/5/10 划分，与本仓库实验 A-H2 保持一致。

## 3. 损失

```text
L_H3D = L_volume + 0.5 L_circle + 0.15 L_phase2 + 0.1 L_opposite
L     = 0.8 L_position + 0.2 L_H3D
```

- `L_volume`：二维高斯位置软标签与 von Mises 航向软标签的联合交叉熵；
- `L_circle`：最终单位航向的圆周余弦损失；
- `L_phase2`：`cos(2 theta), sin(2 theta)` 双频结构轴监督；
- `L_opposite`：真实方向相对 180 度反向方向的 margin ranking loss。

## 4. 测试

```bash
python -m unittest tests.test_experiment_h3d -v
python scripts/smoke_heading_h3d.py \
  --checkpoint /absolute/path/to/experiment_A/best_model.pth
```

## 5. 训练

```bash
cd /data/xuly/cmq/experiment_h3d/bearinguav
export PYTHONPATH=/data/xuly/cmq/experiment_h3d/bearinguav
export TORCH_HOME=/data/xuly/cmq/experiment_f/bearinguav/.cache/torch

BASELINE_A=/data/xuly/cmq/bearinguav/results/c4ma/phr5_d96100_3d_b16_l0.5_e100_gNone_20260812_015315/best_model.pth

CUDA_VISIBLE_DEVICES=1 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GPRVH \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 \
  --rsi_id 96 \
  --n_sample 100 \
  --num_epochs 20 \
  --factor_bslr 0.5 \
  --learning_rate 1e-4 \
  --optimizer AdamW \
  --device_id 0 \
  --gcth none \
  --init_checkpoint ${BASELINE_A} \
  --checkpoint_interval 5 \
  --flag_ckpt 1 \
  --flag_test 1
```

显式训练设置：batch size 16、AdamW、learning rate `1e-4`、weight decay
`1e-4`、20 epochs、8 workers、prefetch factor 2。`CUDA_VISIBLE_DEVICES=1`
后进程内部只看到一张卡，因此 `--device_id 0` 是正确设置。

## 6. 评估重点

H3-D 首先比较 UAV/3D 的 MHE 与 HSR@15，同时报告大于 90 度的反向失败率。
位置分支被冻结，MLE 应与实验 A 基本一致。建议至少运行三个随机种子；只有同一测试划分上
平均 MHE 下降且 HSR@15 提升，才能判断模块有效。
