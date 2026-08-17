# 实验 H1：单位圆约束航向回归

## 1. 实验目标

实验 H1 只修改 Bearing-UAV 的航向预测约束，用于验证训练目标与 MHE 评测几何一致后，
能否降低 3D/UAV 跨视角航向误差。

模型类：

```text
PARCASGM_v5a_H1
```

模型标识：

```text
phr5_h1_circle
```

H1 保持官方端到端联合回归逻辑：

```text
1 UAV + 4 RST -> shared encoder -> SGM -> cross-attention
                                   |-> position (x, y)
                                   \-> heading (cos(theta), sin(theta))
```

H1 不修改 backbone、SGM、RST 坐标编码、跨注意力、位置分支或导航接口。唯一模型改动是
在原航向 MLP 后追加无参数的 L2 归一化，使航向落在单位圆上。

## 2. 原模型问题与 H1 原理

官方模型直接回归两个无约束实数，并对 `(cos(theta), sin(theta))` 分量使用 Smooth L1；
测试 MHE 却使用向量夹角。训练目标和评测几何并不完全一致。

H1 输出：

\[
\hat{u}=\frac{z}{\lVert z\rVert_2+\epsilon}
\]

航向损失：

\[
L_{circle}=1-\hat{u}^{T}u_{gt}=1-\cos(\Delta\theta)
\]

完整多任务损失保持原权重：

\[
L_{H1}=0.8L_{SmoothL1(position)}+0.2L_{circle}
\]

周期角不适合直接作为普通线性变量回归，可参考 CVPR 2023 Phase-Shifting Coder 对角度周期性
和边界不连续的分析：

```text
https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Phase-Shifting_Coder_Predicting_Accurate_Orientation_in_Oriented_Object_Detection_CVPR_2023_paper.pdf
```

H1 没有复制该论文的双频编码，而是先做最小消融：保留 Bearing-UAV 原有一阶
`(cos(theta), sin(theta))` 表示，只增加单位圆投影和圆周损失。

## 3. 获取代码

以下服务器示例在 `/data/xuly/cmq/` 下工作：

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

更新个人仓库：

```bash
git remote -v
git remote add baoli https://github.com/baoli-haixian/bearing-uav.git
git fetch baoli
git switch main
git pull --ff-only baoli main
git log -1 --oneline
```

如果 `baoli` remote 已存在，跳过 `git remote add`。

确认实验 H1 已注册：

```bash
grep -n "PARCASGM_v5a_H1" cvphr/models/posaglreg/models.py
```

## 4. 配置环境

### 4.1 使用服务器现有环境

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
python -c "import torch,torchvision,cv2,pandas; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### 4.2 新建环境

```bash
conda create -n bearing_env python=3.9 -y
conda activate bearing_env
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 5. 检查数据与预训练权重

3D/UAV 训练索引应位于：

```text
Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

检查 metadata 和图像路径：

```bash
test -f Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
wc -l Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
python -c "import os,pandas as pd; p='Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv'; d=pd.read_csv(p); c=['p1_path','p2_path','p3_path','p4_path','target_path']; m=[x for k in c for x in d[k] if not os.path.isfile(x)]; print('rows=',len(d),'missing=',len(m)); print(*m[:20],sep='\n')"
```

只有 `missing=0` 才能开始训练。检查当前数据模式：

```bash
grep -n '^rsi_type' config/base_info.py
```

本实验应使用：

```python
rsi_type = "254k"
```

H1 与官方基线一样使用 ImageNet VGG-16。离线计算节点可提前下载：

```bash
mkdir -p ${TORCH_HOME}/hub/checkpoints
curl -L -o ${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth \
  https://download.pytorch.org/models/vgg16-397923af.pth
```

## 6. 运行代码测试

仓库标准单元测试：

```bash
python -m unittest discover -s tests -p 'test_experiment_h1.py' -v
```

本地/WSL 完整模型 CUDA 冒烟测试：

```bash
python scripts/smoke_heading_h1.py --full-model
```

检查官方 3D checkpoint 是否能严格加载到 H1：

```bash
python scripts/smoke_heading_h1.py \
  --full-model \
  --checkpoint Bearing_UAV/cross_view/best_model.pth
```

测试验证：

- H1 模型和配置注册成功；
- 输出位置、航向形状均为 `[B,2]`；
- 航向模长为 1；
- 对齐、正交、反向航向损失分别为 0、1、2；
- 圆周损失梯度有限且非零；
- 官方 checkpoint 参数键严格兼容；
- 训练和测试入口可以正常导入。

## 7. 单轮真实数据冒烟测试

先运行 1 epoch，确认真实图像加载、前向、圆周损失、反向传播、验证和 checkpoint 保存：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_H1 \
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
results/c4ma/phr5_h1_circle_d96100_3d_b16_l0.5_e1_gNone_
```

检查配置：

```bash
find results/c4ma -path '*phr5_h1_circle*' -name training_configure.json -print
```

其中必须包含：

```json
{
  "model_class": "PARCASGM_v5a_H1",
  "loss_type": "circular",
  "heading_criterion_class": "CircularDirectionLoss"
}
```

## 8. 完整 100 epoch 训练

创建 tmux 会话：

```bash
tmux new -s bearing_exp_h1
cd /data/xuly/cmq/bearinguav
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav:${PYTHONPATH}
export TORCH_HOME=/data/xuly/cmq/bearinguav/.cache/torch
mkdir -p log/c4ma
```

启动训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_H1 \
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
  2>&1 | tee log/c4ma/experiment_h1_100e.log
```

分离会话使用 `Ctrl-b`、`d`。重新连接和监控：

```bash
tmux attach -t bearing_exp_h1
tail -f /data/xuly/cmq/bearinguav/log/c4ma/experiment_h1_100e.log
watch -n 2 nvidia-smi
```

## 9. 实验参数

| 设置 | H1 数值 | 公平基线 |
|---|---:|---:|
| 模型 | `PARCASGM_v5a_H1` | `PARCASGM_v5a` |
| Batch size | 16 | 16 |
| 初始学习率 | `5e-5` | `5e-5` |
| Epoch | 100 | 100 |
| Optimizer | Adam | Adam |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| 位置损失 | Smooth L1 | Smooth L1 |
| 航向损失 | Circular | Smooth L1 |
| 位置/航向权重 | 0.8 / 0.2 | 0.8 / 0.2 |
| 数据划分 | 85/5/10，seed 42 | 相同 |
| 数据增强 | 官方设置 | 相同 |

当前代码划分是 85/5/10，而论文文字报告 70/20/10。H1 必须首先与同一代码和划分下重新训练的
官方基线比较，不能只与论文表格数字直接比较并宣称提升。

## 10. 公平基线训练

只把模型类改回官方 `PARCASGM_v5a`，其他参数完全相同：

```bash
CUDA_VISIBLE_DEVICES=1 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a \
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
  2>&1 | tee log/c4ma/baseline_for_h1_100e.log
```

两个进程可使用不同物理 GPU 并行运行。`CUDA_VISIBLE_DEVICES=1` 后，进程内部该 GPU 的编号仍是 0，
所以命令保留 `--device_id 0`。

## 11. 结果文件与指标

H1 结果目录：

```text
results/c4ma/phr5_h1_circle_d96100_3d_b16_l0.5_e100_gNone_*/
```

关键文件：

```text
training_configure.json
best_model.pth
ckpt_latest_model.pth
training_history.json
tensorboard_logs/
gcheck/training_epoch_log.txt
test_results_*/test_mae.json
test_results_*/test_results*.csv
```

查找主要测试结果：

```bash
find results/c4ma -path '*phr5_h1_circle*' -path '*test_results*' \
  -name test_mae.json -print
```

主要比较：

- `mae_agl` / MHE：越低越好；
- HSR@15：越高越好；
- HSR@5、10、20、25、30；
- 航向误差 median、P90、P95；
- MLE、Recall@1、LSR@15：确认位置性能未明显退化。

代码正确性测试不能证明性能提升。只有完整训练后，在同一 test split 上与公平基线比较，才能判断 H1
是否有效。建议至少运行 3 个随机种子，并对逐样本 `angle_error` 做 paired bootstrap 95% 置信区间。

## 12. 恢复中断训练

查找最新 checkpoint：

```bash
find results/c4ma -maxdepth 2 -name ckpt_latest_model.pth \
  -path '*phr5_h1_circle*' -print
```

恢复命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_H1 \
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
  --resume /data/xuly/cmq/bearinguav/results/c4ma/EXPERIMENT_DIR/ckpt_latest_model.pth \
  2>&1 | tee -a log/c4ma/experiment_h1_100e.log
```

将 `EXPERIMENT_DIR` 替换为真实目录，其他超参数必须与原运行保持一致。

## 13. 已完成的本地验证

WSL2、RTX 4050、`bearinguav_py310` 环境下已完成：

| 检查 | 结果 |
|---|---|
| H1 注册与默认损失 | 通过，`circular` |
| 单位圆最大误差 | `0.0` |
| 对齐/正交/反向损失 | `[0.0, 1.0, 2.0]` |
| 反向梯度 | 有限且非零，范数 `0.4440657` |
| 完整 CUDA 前向 | 通过 |
| 位置/航向输出 | `[1,2]` / `[1,2]` |
| 航向模长 | `1.0` |
| 官方 cross-view checkpoint 严格加载 | 通过 |
| 训练/测试入口导入 | 通过 |

新增可学习参数为 0，推理仅增加二维 L2 归一化。尚未完成 100 epoch 训练，因此本文档不报告新的
MHE/HSR，也不声称 H1 已取得显著性能提升。
