# 实验 F + GA-P-RMC：几何自适应旋转匹配

## 模型

模型类：

```text
PARCASGM_v5a_GlobalRST_PosPrior_PRMC_GeometryAdaptive
```

模型标识：

```text
phr5_f_prmc_ga
```

该实验保留实验 F 的 GlobalRST 位置先验。航向分支在原 P-RMC 的36个旋转
模板前预测一个受约束的对称正定伸缩矩阵，用于补偿 UAV 与卫星特征之间的
尺度、方向性压缩和轻微剪切。形变预测器末层为零初始化，因此训练开始时与
F+P-RMC 完全一致。

```text
4 RST + UAV -> shared GLUF
├── GlobalRST -> PSG + official CA -> position
└── RST/UAV maps -> lightweight deformation predictor
                    -> constrained stretch + 36 rotations
                    -> normalized correlation -> heading distribution -> heading
```

损失为：

```text
L = 0.8 * L_position + 0.2 * L_heading
L_heading = L_vector + 0.1 * L_distribution + 0.01 * L_deformation
```

`L_deformation` 是伸缩矩阵对数的平方范数，不需要新增数据标签。

## 环境和数据

```bash
cd /data/xuly/cmq/bearinguav
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav:${PYTHONPATH}
export TORCH_HOME=/data/xuly/cmq/bearinguav/.cache/torch
test -f Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

数据集下载地址：

```text
https://huggingface.co/datasets/HaoyZhou/bearinguav/tree/main
```

## 测试

```bash
python -m unittest tests.test_experiment_f_prmc_geometry_adaptive -v
```

## UAV/3D 从头联合训练100轮

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_PRMC_GeometryAdaptive \
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

这条命令使用 batch size 16、初始学习率 `5e-5`、Adam、
ReduceLROnPlateau、weight decay 0、seed 42 和当前仓库的 85/5/10 划分。
公平主实验从 ImageNet VGG-16 初始化训练100轮，不加载实验 F 或 F+P-RMC
checkpoint。重点比较 MHE、HSR@15，并检查 MLE、Recall@1、LSR@15。
