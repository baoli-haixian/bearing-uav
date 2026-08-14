# 实验 F：GlobalRST 仅增强位置先验

## 1. 实验目标

实验 F 用于验证实验 B 的位置收益能否在不破坏官方航向路径的情况下保留。

模型类：

```text
PARCASGM_v5a_GlobalRST_PosPrior
```

模型标识：

```text
phr5_globalrst_f
```

损失函数保持官方设置：

\[
L_F=L_{pose}=0.8L_{pos}+0.2L_{heading}
\]

其中位置和航向均使用 Smooth L1。实验 F 不使用 `L_quad`、`L_attn` 或
其他辅助损失。

### 1.1 与实验 B 的区别

实验 B 将全局融合后的 RST 描述子同时送入 PSG 和 CA，因此位置、航向都会受到
GlobalRST 影响。实验 F 改为：

```text
4 RST nl_feat -> GlobalRST -> fused RST descriptors -> PSG -> position prior

4 original RST descriptors + RCE -> official CA -> position head
                                            \-> heading head
```

因此 GlobalRST 参数只接收位置损失梯度，航向路径继续使用官方原始 RST 描述子。

## 2. 获取代码

以下服务器示例严格在 `/data/xuly/cmq/` 下工作：

```bash
ssh xuly@10.6.3.45
ssh 12.12.12.1
cd /data/xuly/cmq/bearinguav
pwd
```

预期输出：

```text
/data/xuly/cmq/bearinguav
```

更新个人仓库代码：

```bash
git remote -v
git remote add baoli https://github.com/baoli-haixian/bearing-uav.git
git fetch baoli
git switch main
git pull --ff-only baoli main
```

如果已经存在 `baoli` remote，跳过 `git remote add`。

检查实验 F 是否存在：

```bash
grep -n "PARCASGM_v5a_GlobalRST_PosPrior" \
  cvphr/models/posaglreg/models.py
```

## 3. 配置环境

### 3.1 使用已有环境

```bash
source ~/.bashrc
conda activate bearing_env
export PROJECT_ROOT=/data/xuly/cmq/bearinguav
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
export TORCH_HOME=${PROJECT_ROOT}/.cache/torch
cd ${PROJECT_ROOT}
```

检查环境：

```bash
python -c "import torch, torchvision, cv2, pandas; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### 3.2 新建环境

```bash
conda create -n bearing_env python=3.9 -y
conda activate bearing_env

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## 4. 下载并组织数据集

Bearing-UAV-90K 官方地址：

```text
https://huggingface.co/datasets/HaoyZhou/bearinguav/tree/main
```

可以使用 Hugging Face CLI 下载：

```bash
pip install -U huggingface_hub
mkdir -p /data/xuly/cmq/Bearing_UAV_90K_download
huggingface-cli download HaoyZhou/bearinguav \
  --repo-type dataset \
  --local-dir /data/xuly/cmq/Bearing_UAV_90K_download
```

按官方压缩包结构解压后，实验 F 的 3D/UAV 训练索引必须能够通过以下路径访问：

```text
/data/xuly/cmq/bearinguav/Bearing_UAV_90K/
  c4m_254k_96bc_b15_s100_v3d/
    metadata/metadata.csv
```

如果数据集已保存在 `/data/xuly/cmq/Bearing_UAV_90K`，不要重复复制，可在仓库内
建立链接：

```bash
cd /data/xuly/cmq/bearinguav
ln -s /data/xuly/cmq/Bearing_UAV_90K Bearing_UAV_90K
```

若 `Bearing_UAV_90K` 已存在，不要再次执行 `ln -s`。

验证 metadata：

```bash
test -f Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
wc -l Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

验证所有图像路径，必须得到 `missing=0`：

```bash
python -c "import os,pandas as pd; p='Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv'; d=pd.read_csv(p); c=['p1_path','p2_path','p3_path','p4_path','target_path']; m=[x for k in c for x in d[k] if not os.path.isfile(x)]; print('rows=',len(d),'missing=',len(m)); print(*m[:20],sep='\n')"
```

检查数据模式：

```bash
grep -n '^rsi_type' config/base_info.py
```

本实验应使用：

```python
rsi_type = "254k"
```

## 5. 预训练权重

实验 F 和官方训练一样使用 ImageNet 预训练 VGG-16。首次构造模型时 torchvision
会下载：

```text
vgg16-397923af.pth
```

设置了上述 `TORCH_HOME` 后，缓存位置是：

```text
/data/xuly/cmq/bearinguav/.cache/torch/hub/checkpoints/
```

如果计算节点不能联网，可在能联网的登录节点下载到该目录：

```bash
mkdir -p /data/xuly/cmq/bearinguav/.cache/torch/hub/checkpoints
cd /data/xuly/cmq/bearinguav/.cache/torch/hub/checkpoints
curl -L -O https://download.pytorch.org/models/vgg16-397923af.pth
```

作者提供的 Bearing-UAV 权重位于：

```text
https://huggingface.co/HaoyZhou/bearinguav/tree/main
```

这些权重适合官方基线测试和导航，但不包含实验 F 新增的 GlobalRST 参数。实验 F
公平消融应从与 A/B/C/D/E 相同的 ImageNet VGG 初始化开始，不要把官方完整模型
权重作为 `--resume` checkpoint。

## 6. 运行代码测试

```bash
cd /data/xuly/cmq/bearinguav
export PYTHONPATH=/data/xuly/cmq/bearinguav:${PYTHONPATH}
python -m unittest discover -s tests -p 'test_experiment_f.py' -v
```

测试验证：

- 模型和配置注册成功；
- 输出位置、航向均为 `[B,2]`；
- 前向结果无 NaN/Inf；
- 航向反向传播时 GlobalRST 梯度为 0；
- 位置反向传播时 GlobalRST 梯度非 0；
- 训练配置可被测试和导航入口恢复。

## 7. 单轮真实数据冒烟测试

先运行 1 个 epoch，确认真实数据加载、前向、反向、验证和保存全部正常：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior \
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

输出目录前缀应为：

```text
results/c4ma/phr5_globalrst_f_d96100_3d_b16_l0.5_e1_gNone_
```

## 8. 完整训练

### 8.1 创建 tmux

```bash
tmux new -s bearing_exp_f
cd /data/xuly/cmq/bearinguav
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav:${PYTHONPATH}
export TORCH_HOME=/data/xuly/cmq/bearinguav/.cache/torch
mkdir -p log/c4ma
```

### 8.2 启动 100 epoch 训练

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior \
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
  2>&1 | tee log/c4ma/experiment_f_100e.log
```

按 `Ctrl-b`，再按 `d` 可退出 tmux 且不中止训练。重新查看：

```bash
tmux attach -t bearing_exp_f
```

其他监控命令：

```bash
tmux ls
tail -f /data/xuly/cmq/bearinguav/log/c4ma/experiment_f_100e.log
watch -n 2 nvidia-smi
```

当前训练入口是单 GPU。`CUDA_VISIBLE_DEVICES=0` 暴露物理 GPU 0，随后
`--device_id 0` 选择当前可见的第 0 张卡。

## 9. 显式训练参数

| 参数 | 数值 |
|---|---:|
| model | `PARCASGM_v5a_GlobalRST_PosPrior` |
| dataset | `RSBlockDatasetPA_v3q` |
| batch size | 16 |
| learning rate | `5e-5` |
| epochs | 100 |
| optimizer | Adam |
| weight decay | 0 |
| scheduler | ReduceLROnPlateau |
| scheduler factor | 0.5 |
| scheduler patience | 3 |
| loss | SmoothL1 |
| position weight | 0.8 |
| heading weight | 0.2 |
| gradient clipping | disabled (`--gcth none`) |
| workers | 8 |
| prefetch factor | 2 |
| split | 85% / 5% / 10% |
| split seed | 42 |
| GlobalRST token grid | 4x4 |
| Transformer heads | 8 |
| Transformer FFN | 512 |
| dropout | 0.1 |

当前代码划分是 85/5/10，而论文正文写的是 70/20/10。实验 F 必须先和使用
相同当前代码、相同划分的实验 A/B 比较，不能只与论文表格直接比较。

## 10. 测试最佳模型

使用 `--flag_test 1` 时，训练结束后会自动测试验证集最优的
`best_model.pth`。完整实验目录前缀为：

```text
results/c4ma/phr5_globalrst_f_d96100_3d_b16_l0.5_e100_gNone_
```

查找结果：

```bash
find results/c4ma \
  -path '*phr5_globalrst_f*' \
  -path '*test_results*' \
  -name 'test_mae.json' -print
```

也可以手动测试：

```bash
export RUN_DIR=/data/xuly/cmq/bearinguav/results/c4ma/你的实验F目录

python -m cvphr.test.cvphr_test \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 \
  --rsi_id 96 \
  --n_sample 100 \
  --bestpth_dir ${RUN_DIR}
```

重点比较：

```text
Recall@1, LSR@15, HSR@15, MLE, MHE
```

实验 F 的目标不是只获得最低 MLE，而是：

```text
保留实验 B 的位置收益，同时让 HSR/MHE 回到实验 A 附近。
```

## 11. 恢复中断训练

```bash
export RUN_DIR=/data/xuly/cmq/bearinguav/results/c4ma/你的实验F目录

CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior \
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
  --resume ${RUN_DIR}/ckpt_latest_model.pth \
  2>&1 | tee -a log/c4ma/experiment_f_100e.log
```

恢复时必须保持模型类、数据集、学习率、batch size 和总 epoch 数与原运行一致。

## 12. 公平消融要求

至少比较：

| 实验 | 模型 | GlobalRST 使用位置 |
|---|---|---|
| A | `PARCASGM_v5a` | 无 |
| B | `PARCASGM_v5a_GlobalRST` | PSG + CA，影响位置和航向 |
| F | `PARCASGM_v5a_GlobalRST_PosPrior` | 仅 PSG 位置先验 |

除 `model_class` 外，其余训练参数必须完全一致。正式结论建议运行至少 3 个随机
种子并报告 `mean +/- std`。如果 F 的 MLE 优于 A，且 HSR/MHE 不再出现 B 的明显
退化，才能支持“位置专用全局 RST 上下文有效”的结论。
