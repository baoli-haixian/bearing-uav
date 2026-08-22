# F + LCPR-1 Joint-100

## 1. 实验定义

模型类：

```text
PARCASGM_v5a_GlobalRST_PosPrior_LCPR1
```

本实验将实验 F 与 LCPR-1 整合为一个模型，并从 ImageNet VGG-16 初始化开始联合
训练 100 轮。不加载实验 F checkpoint，也不冻结实验 F 新增模块。

实验 F 只用四 RST 全局上下文改善 PSG 位置先验；LCPR-1 在实验 F 粗位置附近生成
`3x3` 候选点，使用原始 RST 空间特征和 UAV 空间特征计算局部相似度，再预测门控
位置残差。官方 CA 和航向分支不变。

损失完全保持官方设置：

```text
L = 0.8 * SmoothL1(final_position, target_position)
  + 0.2 * SmoothL1(heading, target_heading)
```

没有 `L_local` 或其他辅助损失。

## 2. 环境

```bash
ssh xuly@10.6.3.45
cd /data/xuly/cmq/experiment_f_lcpr1_joint100/bearinguav
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/experiment_f_lcpr1_joint100/bearinguav:${PYTHONPATH}
export TORCH_HOME=/data/xuly/cmq/experiment_f/bearinguav/.cache/torch
```

如需新建环境，使用 Python 3.9、PyTorch 2.1.2、torchvision 0.16.2，然后安装：

```bash
pip install -r requirements.txt
```

## 3. 数据和 VGG-16 权重

数据集下载地址：

```text
https://huggingface.co/datasets/HaoyZhou/bearinguav/tree/main
```

本服务器复用已有数据，在项目目录中建立指向 `/data/xuly/cmq/Bearing_UAV_90K`
的链接。必须验证：

```bash
test -f Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

VGG-16 ImageNet 权重文件为 `vgg16-397923af.pth`。服务器训练复用：

```text
/data/xuly/cmq/experiment_f/bearinguav/.cache/torch/hub/checkpoints/
```

不要传入 `--init_checkpoint`；否则不再是公平的 Joint-100。

## 4. 测试

```bash
python -m unittest tests.test_experiment_f_lcpr1 -v
```

测试覆盖模块形状、候选概率、坐标边界、前后向梯度、航向梯度隔离、模型注册和
配置恢复。

## 5. 完整训练

先在 `sc1` 用 `nvidia-smi` 确认 GPU 空闲，然后在 `sc0` 的 tmux 中执行：

```bash
cd /data/xuly/cmq/experiment_f_lcpr1_joint100/bearinguav
export TORCH_HOME=/data/xuly/cmq/experiment_f/bearinguav/.cache/torch
bash scripts/run_f_lcpr1_joint100_sc1.sh 0 \
  2>&1 | tee log/c4ma/experiment_f_lcpr1_joint100.log
```

显式训练命令等价于：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_LCPR1 \
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

关键参数：batch size 16、学习率 `5e-5`、Adam、weight decay 0、
ReduceLROnPlateau、100 epochs、8 workers、prefetch factor 2、85/5/10 划分、
seed 42。LCPR 使用 64 维特征、局部半径 0.15、温度 0.1、门控偏置 -2。

## 6. 监控和续训

```bash
tail -f log/c4ma/experiment_f_lcpr1_joint100.log
tmux attach -t f_lcpr1_joint100
```

中断后从原结果目录恢复，必须显式传入：

```bash
--resume /data/xuly/cmq/experiment_f_lcpr1_joint100/bearinguav/results/c4ma/结果目录/ckpt_latest_model.pth
```

## 7. 评估

`--flag_test 1` 会在训练完成后自动用验证集最优权重测试。重点与相同数据划分下的
实验 A 和 F 比较 UAV/3D 的 Recall@1、LSR@15 和 MLE；LCPR-1 的主要目标是降低
MLE，不应期待航向指标因该模块直接提升。
