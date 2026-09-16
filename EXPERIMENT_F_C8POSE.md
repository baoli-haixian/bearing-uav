# 实验 F+C8-Pose：并行循环等变姿态体

## 1. 实验目标

本实验建立在实验 F 上，不替换官方 VGG/GLUF/CA，也不删除 GlobalRST。新增一个
轻量并行 C8 分支，显式构造“相对角度 x 二维位置”姿态体：

```text
模型类：PARCASGM_v5a_GlobalRST_PosPrior_C8Pose
模型标识：phr5_f_c8pose
```

核心流程：

```text
4 RST + UAV
    |-- 官方 VGG/GLUF + GlobalRST + PSG + CA --> 实验 F 主路径
    `-- 共享 C8 orbit encoder
            -> RST [B,4,C,8,H,W], UAV [B,C,8,H,W]
            -> relative correlation volume [B,8,Hm,Wm]
            |-- LogSumExp(angle) -> rotation-invariant position prior
            `-- location-conditioned angle distribution -> equivariant heading
```

C8 位置先验与实验 F 的 PSG 先验通过小门控融合；C8 航向与官方航向输出通过另一个
小门控融合。两个门控偏置初始化为 `-2`，使训练初期仍接近实验 F。

## 2. 损失函数

总任务权重保持官方设置：

```text
L = 0.8 Lpos + 0.2 Lheading
Lheading = SmoothL1(h_pred, h_gt) + 0.05 Lcircular_dist
```

`Lcircular_dist` 使用现有 `(cos(theta), sin(theta))` 标签动态生成 C8 von Mises
软标签，不需要新增数据标注。默认 `kappa=4`，避免 45 度方向格上的目标过尖。

## 3. 环境与数据

沿用实验 F 的环境和 Bearing-UAV-90K 目录。服务器操作必须位于：

```bash
cd /data/xuly/cmq/bearinguav
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav:${PYTHONPATH}
export TORCH_HOME=/data/xuly/cmq/bearinguav/.cache/torch
```

3D 数据应存在：

```text
/data/xuly/cmq/bearinguav/Bearing_UAV_90K/
  c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

VGG-16 ImageNet 权重仍由官方主干使用。C8 分支随机初始化；为保证与 A/F 公平，
不要从实验 F 的最终权重继续训练，应按相同初始化策略从头训练 100 轮。

## 4. 代码测试

```bash
python -m unittest tests.test_experiment_f_c8pose -v
```

测试覆盖：

- C8 张量、RST mosaic 和姿态体形状；
- 位置/方向概率归一化；
- 连续航向输出为单位向量；
- 完整模型输出均为 `[B,2]`；
- C8 分支获得非零梯度；
- 训练配置能够恢复模型类。

## 5. 单轮冒烟测试

先检查空闲 GPU，再执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_C8Pose \
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

由于 C8 分支对每张图构造 8 个旋转轨道，它比实验 F 更慢、显存更高。第一次真实
冒烟测试应同时观察 `nvidia-smi`；若 batch 16 OOM，必须记录实际 batch size，且
与基线重新做等 batch 或等有效 batch 的公平对照。

## 6. 正式训练 100 轮

```bash
mkdir -p log/c4ma
tmux new -s f_c8pose

CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_C8Pose \
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
  2>&1 | tee log/c4ma/experiment_f_c8pose_100e.log
```

退出 tmux 而不中止训练：`Ctrl-b`，再按 `d`。重新查看：

```bash
tmux attach -t f_c8pose
tail -f log/c4ma/experiment_f_c8pose_100e.log
```

## 7. 默认显式参数

| 参数 | 数值 |
|---|---:|
| batch size | 16（由 `factor_bslr=0.5`） |
| learning rate | `5e-5` |
| epochs | 100 |
| optimizer | Adam |
| scheduler | ReduceLROnPlateau |
| position/heading weight | `0.8 / 0.2` |
| C8 directions | 8 |
| per-direction channels | 32 |
| group feature grid | `8 x 8` |
| position temperature | 0.2 |
| orientation temperature | 0.1 |
| heading soft-label kappa | 4.0 |
| heading distribution weight | 0.05 |
| position/heading gate bias | -2.0 |

## 8. 评估与公平对照

训练结束后 `--flag_test 1` 自动测试最佳验证 checkpoint。至少比较：

| 对照 | 模型类 |
|---|---|
| A | `PARCASGM_v5a` |
| F | `PARCASGM_v5a_GlobalRST_PosPrior` |
| F+C8-Pose | `PARCASGM_v5a_GlobalRST_PosPrior_C8Pose` |

重点报告 `Recall@1、LSR@15、HSR@15、MLE、MHE`，并额外记录训练时间、峰值显存和
推理耗时。只有同数据划分、同 seed、同训练轮数、同 batch/有效 batch 下的结果才
能用于公平消融。正式结论建议至少 3 个随机种子并报告 `mean +/- std`。

本实现采用离散图像轨道和共享轻量编码器实现近似 C8 等变；由于双线性旋转插值和
原始图像边界裁剪，它不是连续任意角度下的严格数学等变网络。C8 只严格覆盖 45 度
离散旋转，连续航向由圆周期望和格内残差头补偿。
