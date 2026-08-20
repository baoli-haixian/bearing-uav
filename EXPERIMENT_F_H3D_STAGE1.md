# F + H3-D Stage 1 训练教程

## 1. 实验目标

该实验组合两个已经独立验证的模块：

- 实验 F：`RSTGlobalContextFusion` 只增强 PSG 位置先验，CA 保留原始描述子；
- H3-D：`GeometryPoseVolumeHeadingHead` 建立 `81 x 36` 位置-航向联合体积并输出航向。

模型类为：

```text
PARCASGM_v5a_GlobalRST_PosPrior_GPRVH
```

Stage 1 必须使用实验 F 的 `best_model.pth` 初始化。实验 F 的全部参数冻结，
只训练新增的 `gprv_head.*`。位置输出与实验 F 保持一致，航向输出由 H3-D 接管。

## 2. 损失函数

总损失保持 Bearing-UAV 的位置/航向权重：

```text
L = 0.8 * L_pos + 0.2 * L_H3D
```

其中：

```text
L_H3D = L_volume + 0.5 * L_circle
        + 0.15 * L_phase2 + 0.1 * L_opposite
```

Stage 1 中位置分支被冻结，因此 `L_pos` 只用于监控，不会更新实验 F 参数。
所有航向监督都由数据集已有的 `coords` 和 `agl_coords` 构造，不需要新增标签。

## 3. 环境准备

```bash
git clone https://github.com/baoli-haixian/bearing-uav.git
cd bearing-uav

conda create -n bearing_env python=3.9 -y
conda activate bearing_env
pip install -r requirements.txt
```

按照官方 README 下载 Bearing-UAV-90K，并保证项目下存在：

```text
Bearing_UAV_90K/
  c4m_254k_96bc_b15_s100_v3d/
    metadata/metadata.csv
  citya/
  cityb/
  cityc/
  cityd/
  city_rsi/
```

## 4. 准备实验 F 权重

必须提供由以下模型训练得到的权重：

```text
model_class = PARCASGM_v5a_GlobalRST_PosPrior
model_name  = phr5_globalrst_f
```

推荐使用：

```text
/path/to/experiment_f/results/c4ma/phr5_globalrst_f_.../best_model.pth
```

不能使用实验 A、B 或其他实验的权重替代。加载器会严格检查：组合模型只允许缺少
`gprv_head.*`，如果缺少 `rst_global_fusion.*` 会立即报错。

## 5. 训练命令

```bash
python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_GPRVH \
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
  --init_checkpoint /path/to/experiment_f/best_model.pth \
  --checkpoint_interval 5 \
  --flag_ckpt 1 \
  --flag_test 1
```

显式配置：

| 参数 | 数值 |
|---|---:|
| batch size | 16 |
| learning rate | `1e-4` |
| epochs | 20 |
| optimizer | AdamW |
| weight decay | `1e-4` |
| position/heading weight | `0.8 / 0.2` |
| checkpoint interval | 5 |
| trainable module | `gprv_head.*` only |

## 6. 验证冻结范围

训练启动日志应显示实验 F checkpoint 只缺少新增 H3-D 参数。也可以运行：

```bash
python tests/test_experiment_f_h3d_stage1.py
```

测试检查：

- 实验 F 权重能够严格恢复；
- 组合模型的位置输出与实验 F 完全一致；
- 只有 `gprv_head.*` 的 `requires_grad=True`；
- 前向、损失和反向传播均为有限值；
- 冻结的实验 F 参数没有梯度。

## 7. 评估结果

训练结束后会自动使用 `best_model.pth` 测试。主要结果位于：

```text
results/c4ma/phr5_globalrst_f_gprvh_s1_.../
  best_model.pth
  ckpt_latest_model.pth
  gcheck/training_epoch_log.txt
  test_results_.../test_mae.json
```

重点比较实验 F、H3-D 和组合模型的：

- `Recall@1`、`LSR@15`、`MLE`：判断是否保留实验 F 的定位能力；
- `HSR@15`、`MHE`：判断是否保留或超过 H3-D 的航向能力。

Stage 1 完成后再依据测试结果决定是否进行低学习率联合微调，不能仅根据训练损失决定。
