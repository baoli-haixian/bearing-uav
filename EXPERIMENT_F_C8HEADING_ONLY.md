# 实验 F+C8-Heading-Only：仅航向融合消融

## 1. 实验目的

上一版 `F+C8-Pose` 相对实验 F 的航向指标略有改善，但三个位置指标均下降。
因此本实验保留 C8 的航向建模能力，同时彻底移除 C8 位置先验对实验 F 位置路径的
影响。

```text
模型类：PARCASGM_v5a_GlobalRST_PosPrior_C8HeadingOnly
模型标识：phr5_f_c8heading
```

模型仍从头联合训练 100 轮，不加载实验 F 或 F+C8-Pose 的最终权重。

## 2. 架构

```text
4 RST + UAV
    |-- 实验 F 主路径
    |     |-- GlobalRST -> PSG -> F position prior ------------------.
    |     |-- official CA -> shared feature -> position regressor <---'
    |     `-- official heading head --------------------------.
    |                                                        | gated fusion
    `-- C8 orbit encoder -> relative pose volume -> C8 heading'
```

关键约束：

- 最终位置回归器只接收实验 F 的 `f_position_prior`；
- C8 的位置输出仅作为诊断量，不参与 PSG、CA 或位置回归；
- C8 航向与官方航向通过可学习门控融合；
- C8 姿态体仍可在内部对空间位置边缘化，从而估计航向，但不会修改最终位置预测。

## 3. 损失

```text
L = 0.8 Lpos + 0.2 Lheading
Lpos = SmoothL1(p_pred, p_gt)
Lheading = SmoothL1(h_pred, h_gt) + 0.05 Lcircular_dist
```

`Lcircular_dist` 使用数据集已有的 `(cos(theta), sin(theta))` 航向标签生成 C8
von Mises 软标签，不需要新增标注。位置损失和实验 F 保持一致。

## 4. 环境与数据

```bash
cd /data/xuly/cmq/bearinguav
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav:${PYTHONPATH}
export TORCH_HOME=/data/xuly/cmq/bearinguav/.cache/torch
```

数据目录：

```text
/data/xuly/cmq/bearinguav/Bearing_UAV_90K/
  c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

## 5. 代码测试

```bash
python -m unittest tests.test_experiment_f_c8pose -v
```

测试会验证模型注册、前向与反向传播、配置恢复，以及改变 C8 位置输出不会改变最终
位置预测。

## 6. 正式训练

先使用 `nvidia-smi` 确认 GPU 空闲。正式参数与实验 F、F+C8-Pose 一致：

```bash
mkdir -p log/c4ma
tmux new -s f_c8heading

CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_C8HeadingOnly \
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
  --flag_test 1 \
  2>&1 | tee log/c4ma/experiment_f_c8heading_100e.log
```

退出 tmux：`Ctrl-b` 后按 `d`。查看进度：

```bash
tmux attach -t f_c8heading
tail -f log/c4ma/experiment_f_c8heading_100e.log
```

## 7. 对照与判定

至少比较相同数据划分、seed、batch size 和 100 轮训练下的三组：

| 实验 | 模型类 |
|---|---|
| F | `PARCASGM_v5a_GlobalRST_PosPrior` |
| F+C8-Pose | `PARCASGM_v5a_GlobalRST_PosPrior_C8Pose` |
| F+C8-Heading-Only | `PARCASGM_v5a_GlobalRST_PosPrior_C8HeadingOnly` |

主要判断：位置指标是否恢复到实验 F 的水平，同时 `HSR@15` 和 `MHE` 是否保留或
超过 F+C8-Pose 的提升。单次小幅变化不足以证明稳定有效，最终应至少运行 3 个
随机种子并报告 `mean +/- std`。
