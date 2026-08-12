# 实验 E：全局 RST 融合与双辅助约束

实验 E 在实验 B 的四 RST 全局上下文模块上增加两项训练约束：

```text
L_E = L_pose + lambda_quad * L_quad + lambda_attn * L_attn
L_pose = 0.8 * SmoothL1(position) + 0.2 * SmoothL1(heading)
lambda_quad = 0.05
lambda_attn = 0.10
```

前 5 个 epoch 对两项辅助损失进行线性升权。第 1 至第 5 个 epoch 的
比例依次为 `0.2、0.4、0.6、0.8、1.0`，第 5 个 epoch 后保持完整权重。

验证损失、学习率调度和 `best_model.pth` 的选择仍然只使用
`L_pose`，从而与实验 A、B 保持一致。

## 1. 进入服务器与项目目录

所有操作都必须位于 `/data/xuly/cmq/` 下：

```bash
ssh xuly@10.6.3.45
ssh 12.12.12.1
cd /data/xuly/cmq/bearinguav
pwd
```

`pwd` 必须输出：

```text
/data/xuly/cmq/bearinguav
```

## 2. 获取实验 E 代码

个人仓库地址为：

```text
https://github.com/baoli-haixian/bearing-uav.git
```

如果服务器仓库尚未添加个人远程：

```bash
git remote add baoli https://github.com/baoli-haixian/bearing-uav.git
```

通过本机或服务器可用的 Git 代理拉取最新代码：

```bash
git fetch baoli
git switch main
git pull --ff-only baoli main
git log -3 --oneline
```

确认实验 E 已注册：

```bash
grep -n "PARCASGM_v5a_GlobalRST_Aux" cvphr/models/posaglreg/models.py
```

注意：如果服务器仓库里还有未提交修改，先运行 `git status --short` 检查，
不要直接覆盖已有训练代码。

## 3. 激活环境

```bash
source ~/.bashrc
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

预期可以识别两张 Tesla V100。

## 4. 检查数据集配置

实验 E 使用完整 UAV-Satellite 跨视角数据：

```text
/data/xuly/cmq/bearinguav/Bearing_UAV_90K/
  c4m_254k_96bc_b15_s100_v3d/
    metadata/metadata.csv
```

检查元数据：

```bash
test -f Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
wc -l Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

`config/base_info.py` 中必须启用：

```python
rsi_type = "254k"
```

检查当前值：

```bash
grep -n '^rsi_type' config/base_info.py
```

检查 CSV 引用的全部图片，避免训练到中途才遇到 `FileNotFoundError`：

```bash
python -c "import os,pandas as pd; p='Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv'; d=pd.read_csv(p); c=['p1_path','p2_path','p3_path','p4_path','target_path']; m=[x for k in c for x in d[k] if not os.path.isfile(x)]; print('rows=',len(d),'missing=',len(m)); print(*m[:20],sep='\n')"
```

只有输出 `missing=0` 才开始训练。CSV 中使用相对路径，因此必须从项目根目录
启动程序。

## 5. 运行代码测试

先运行不依赖真实数据的单元测试：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

测试应覆盖：

- 实验 B 两输出接口保持不变；
- 实验 E 推理时仍输出 `(position, heading)`；
- 训练时返回象限上下文和 PSG 注意力；
- 四个角点的几何监督顺序正确；
- 辅助损失有限且可以反向传播；
- 前 5 轮 warm-up 比例正确。

## 6. 运行一轮真实数据测试

建议先使用 GPU 0 跑一个 epoch，并关闭训练后的完整测试：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_Aux \
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

成功标准：

- 能进入 `Epoch 1/1 [Train]` 和验证阶段；
- 没有 NaN、Inf 和图片缺失错误；
- 日志中出现 `train_quad`、`train_attn` 和 `aux_scale=0.200`；
- 生成 `best_model.pth` 和 `ckpt_latest_model.pth`。

结果目录以以下前缀开头：

```text
results/c4ma/phr5_globalrst_e_d96100_3d_b16_l0.5_e1_gNone_
```

## 7. 使用 tmux 完整训练 100 轮

创建会话：

```bash
tmux new -s bearing_exp_e
cd /data/xuly/cmq/bearinguav
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav
mkdir -p log/c4ma
```

启动实验 E：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_Aux \
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
  2>&1 | tee log/c4ma/experiment_e_100e.log
```

按 `Ctrl-b`，再按 `d`，即可退出 tmux 而不终止训练。重新查看：

```bash
tmux attach -t bearing_exp_e
```

其他监控命令：

```bash
tmux ls
tail -f /data/xuly/cmq/bearinguav/log/c4ma/experiment_e_100e.log
watch -n 2 nvidia-smi
```

当前训练器是单 GPU。`CUDA_VISIBLE_DEVICES=0` 只暴露物理 GPU 0，程序内部的
`--device_id 0` 选择该可见设备。不要把 `--device_id` 改成 1。

## 8. 完整训练参数

| 参数 | 实验 E 的值 |
|---|---:|
| 模型 | `PARCASGM_v5a_GlobalRST_Aux` |
| Batch size | 16 |
| Learning rate | `5e-5` |
| Epochs | 100 |
| 优化器 | Adam |
| Weight decay | 0 |
| 调度器 | ReduceLROnPlateau |
| Scheduler patience/factor | `3 / 0.5` |
| 位置/航向权重 | `0.8 / 0.2` |
| `lambda_quad` | 0.05 |
| `lambda_attn` | 0.10 |
| 辅助损失 warm-up | 5 epochs |
| 注意力 temperature | 1.0 |
| DataLoader workers | 8 |
| Prefetch factor | 2 |
| 数据划分 | 85% / 5% / 10%，seed 42 |

模型参数保存在 `training_configure.json`，其中应包含：

```json
{
  "quad_loss_weight": 0.05,
  "attention_loss_weight": 0.1,
  "auxiliary_warmup_epochs": 5,
  "attention_temperature": 1.0
}
```

## 9. 损失日志如何理解

每轮日志包括：

```text
train_loss     = 实际反向传播的总损失
train_pose     = 0.8 * train_pos + 0.2 * train_dir
train_quad     = 未加权的 L_quad
train_attn     = 未加权的 L_attn
weighted_quad  = warmup * 0.05 * L_quad
weighted_attn  = warmup * 0.10 * L_attn
val_loss       = 验证集上的 L_pose，不含辅助损失
aux_scale      = 当前 warm-up 比例
```

TensorBoard 同时记录这些曲线：

```bash
tensorboard --logdir /data/xuly/cmq/bearinguav/results/c4ma --port 6006
```

重点检查前五轮的 `aux_scale` 是否为 `0.2、0.4、0.6、0.8、1.0`，以及
`train_quad`、`train_attn` 是否正常下降。

## 10. 结果与自动评估

100 轮结果目录以以下前缀开头：

```text
results/c4ma/phr5_globalrst_e_d96100_3d_b16_l0.5_e100_gNone_
```

主要文件：

```text
training_configure.json
best_model.pth
ckpt_latest_model.pth
checkpoints/epoch_0005.pth
checkpoints/epoch_0010.pth
...
gcheck/training_epoch_log.txt
gcheck/training_monitor.json
tensorboard_logs/
test_results_*/test_mae.json
test_results_*/test_results*.csv
```

命令使用 `--flag_test 1`，因此训练完成后会自动加载最佳验证权重并运行测试。
查找测试指标：

```bash
find results/c4ma -path '*phr5_globalrst_e*' -path '*test_results*' -name test_mae.json -print
```

重点记录：

- Recall@1；
- LSR@15；
- HSR@15；
- MLE；
- MHE。

然后与已经完成的 A、B 使用相同测试集直接比较。只有 E 相对 B 的差值才能
说明双辅助损失是否有效。

## 11. 断点续训

查找最新检查点：

```bash
find results/c4ma -maxdepth 2 -name ckpt_latest_model.pth -path '*phr5_globalrst_e*' -print
```

恢复命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_Aux \
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
  2>&1 | tee -a log/c4ma/experiment_e_100e.log
```

把 `EXPERIMENT_DIR` 替换成实际目录。恢复时不得更改模型类、batch size、
学习率缩放、损失权重或数据设置。

不要使用实验 B 的 checkpoint 启动实验 E。为了保证消融公平，E 应和 A、B 一样
从相同 ImageNet VGG-16 初始化开始独立训练。

## 12. 最终对比表

| 实验 | 模型 | 损失 |
|---|---|---|
| A | 官方 `PARCASGM_v5a` | `L_pose` |
| B | `PARCASGM_v5a_GlobalRST` | `L_pose` |
| E | `PARCASGM_v5a_GlobalRST_Aux` | `L_pose + 0.05 L_quad + 0.10 L_attn` |

三组实验必须使用相同的 85/5/10 划分、seed 42、训练轮数、增强、batch size
和学习率。当前代码划分与论文文字中的 70/20/10 不同，因此这三组结果属于
当前官方代码口径下的公平消融，不能直接声明为论文原表的严格复现。
