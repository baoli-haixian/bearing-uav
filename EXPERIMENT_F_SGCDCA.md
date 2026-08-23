# 实验 F + SG-CDCA：位置全局先验与并行方向交叉注意力

## 1. 实验定义

模型类：

```text
PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA
```

模型标识：

```text
phr5_f_sgcdca
```

该实验从头联合训练 100 epoch，不冻结实验 F 后再训练。位置路径保持实验 F：

```text
4 RST nl_feat -> GlobalRST -> PSG position prior -> position head
```

航向路径新增独立的 SG-CDCA：

```text
original RST/UAV nl_feat -> 16-direction polar encoding
                            -> cyclic direction scores [B,4,16]
official CA semantic logits [B,4] --------------------+
                                                       v
                              joint attention [B,4,16]
                                                       |
                         direction context -> gated residual
                                                       |
UAV descriptor + enhanced heading context -> official heading regressor
```

GlobalRST 不进入航向分支，SG-CDCA 不进入位置分支。两个改进共享官方
VGG/Non-local 特征，但在任务头处保持并行。

总损失为：

```text
L = 0.8 * L_pos + 0.2 * L_heading
L_heading = SmoothL1(h_pred, h_gt) + 0.1 * L_circular_distribution
```

`L_circular_distribution` 使用数据集中已有的 `(x_cosa, y_sina)` 生成 von
Mises 圆周软标签，不需要新增人工标注。

## 2. 环境

```bash
cd /data/xuly/cmq/bearinguav
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav:${PYTHONPATH}
export TORCH_HOME=/data/xuly/cmq/bearinguav/.cache/torch
```

首次配置环境时：

```bash
conda create -n bearing_env python=3.9 -y
conda activate bearing_env
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 3. 数据和 VGG 权重

Bearing-UAV-90K：

```text
https://huggingface.co/datasets/HaoyZhou/bearinguav/tree/main
```

3D 训练所需元数据：

```text
Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

检查数据：

```bash
test -f Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
python -c "import os,pandas as pd; p='Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv'; d=pd.read_csv(p); c=['p1_path','p2_path','p3_path','p4_path','target_path']; m=[x for k in c for x in d[k] if not os.path.isfile(x)]; print('rows=',len(d),'missing=',len(m))"
```

与官方模型和实验 F 相同，本实验只使用 ImageNet VGG-16 初始化：

```text
vgg16-397923af.pth
```

作者完整 Bearing-UAV checkpoint 不包含 GlobalRST 和 SG-CDCA 参数，不能作为
公平的 100 epoch 联合训练初始化。它只用于基线评估或模块可行性诊断。

## 4. 代码测试

```bash
python -m unittest tests.test_experiment_f_sgcdca -v
```

测试覆盖模型注册、前后向、概率归一化、辅助损失、配置恢复，以及梯度隔离：

- 位置损失更新 GlobalRST，不更新 SG-CDCA；
- 航向损失更新 SG-CDCA，不更新 GlobalRST。

## 5. 单轮真实数据冒烟测试

先检查一个真实 batch 的前向、反向和 SG-CDCA 梯度：

```bash
python scripts/smoke_f_sgcdca.py \
  --metadata_csv Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv \
  --batch_size 2 \
  --device cuda
```

再运行完整的单 epoch 流程：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA \
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

## 6. 完整训练

```bash
tmux new -s bearing_f_sgcdca
cd /data/xuly/cmq/bearinguav
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav:${PYTHONPATH}
export TORCH_HOME=/data/xuly/cmq/bearinguav/.cache/torch
mkdir -p log/c4ma

CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA \
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
  2>&1 | tee log/c4ma/f_sgcdca_joint100.log
```

显式配置对应：

- batch size：16（`32 * factor_bslr`）
- learning rate：`5e-5`（`1e-4 * factor_bslr`）
- epochs：100
- optimizer：Adam
- scheduler：ReduceLROnPlateau
- weight decay：0
- 数据划分：85/5/10，seed 42
- 位置/航向任务权重：0.8/0.2
- 方向数：16，每 bin 22.5 度
- 航向分布损失权重：0.1

退出 tmux：`Ctrl-b` 后按 `d`。重新查看：

```bash
tmux attach -t bearing_f_sgcdca
```

## 7. 单独评估

训练命令使用 `--flag_test 1` 时会自动评估最佳权重。手动评估时，将训练输出目录
传给测试入口：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.test.cvphr_test \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 \
  --rsi_id 96 \
  --n_sample 100 \
  --bestpth_dir /data/xuly/cmq/bearinguav/results/c4ma/phr5_f_sgcdca_实际目录 \
  --device_id 0
```

最终重点与实验 F 对比 UAV/3D 的 MHE、HSR@15，同时检查 MLE、Recall@1 和
LSR@15 是否保持。SG-CDCA 的主要成功标准是 MHE 下降和 HSR@15 上升。
