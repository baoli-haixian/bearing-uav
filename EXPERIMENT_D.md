# 实验 D：四 RST 全局融合 + 几何注意力损失

## 1. 实验定义

实验 D 在实验 B 的四 RST 全局上下文模块上，只增加几何注意力损失：

```text
L_D = L_pose + warmup(epoch) * 0.10 * L_attn
L_pose = 0.8 * SmoothL1(position) + 0.2 * SmoothL1(heading)
L_attn = KL(geometry_target || PSG_attention)
```

`geometry_target` 根据 UAV 的真实归一化坐标，对四个 RST 象限生成双线性空间
权重。前 5 个 epoch 的辅助损失比例依次为 `0.2, 0.4, 0.6, 0.8, 1.0`。
实验 D 的 `quad_loss_weight` 固定为 `0`，不使用 `L_quad`。验证损失、学习率
调度和 `best_model.pth` 选择仍只使用 `L_pose`。

模型类：`PARCASGM_v5a_GlobalRST_Attn`

## 2. 环境与路径

以下命令以 Linux 服务器 `/data/xuly/cmq/bearinguav` 为例，并确保缓存也位于
`/data/xuly/cmq/` 下：

```bash
cd /data/xuly/cmq/bearinguav
export PROJECT_ROOT=/data/xuly/cmq/bearinguav
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$PROJECT_ROOT/_downloads"
```

首次创建环境：

```bash
conda create -n bearing_env python=3.9 -y
conda activate bearing_env

python -m pip install torch==2.1.2 torchvision==0.16.2 \
  torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install "huggingface_hub[cli]"
```

检查 CUDA：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu 0:', torch.cuda.get_device_name(0))
PY
```

如需代理，必须使用服务器可访问的代理地址：

```bash
export http_proxy=http://PROXY_HOST:7890
export https_proxy=http://PROXY_HOST:7890
```

## 3. 获取代码

首次部署官方仓库：

```bash
cd /data/xuly/cmq
git clone https://github.com/liukejia121/bearinguav.git
cd /data/xuly/cmq/bearinguav
```

实验 D 需要包含改进类的分支。检查：

```bash
grep -n "PARCASGM_v5a_GlobalRST_Attn" \
  cvphr/models/posaglreg/models.py
```

输出为空表示当前代码没有实验 D，不能直接使用实验 E 代替。

## 4. 下载和组织数据集

官方数据集：

```text
https://huggingface.co/datasets/HaoyZhou/bearinguav/tree/main
```

下载并解压：

```bash
cd "$PROJECT_ROOT"
hf download HaoyZhou/bearinguav Bearing_UAV_90K.zip \
  --repo-type dataset \
  --local-dir "$PROJECT_ROOT/_downloads"
unzip -q "$PROJECT_ROOT/_downloads/Bearing_UAV_90K.zip" \
  -d "$PROJECT_ROOT"
```

目标结构：

```text
/data/xuly/cmq/bearinguav/Bearing_UAV_90K/
├── city_rsi/
├── citya/
├── cityb/
├── cityc/
├── cityd/
├── c4m_254k_96bc_b15_s100/
└── c4m_254k_96bc_b15_s100_v3d/
```

实验 D 的 UAV/3D 训练元数据为：

```text
Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

检查数据完整性：

```bash
test -f Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv

python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path('/data/xuly/cmq/bearinguav')
csv_path = root / 'Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv'
df = pd.read_csv(csv_path)
cols = [c for c in ['target_path', 'p1_path', 'p2_path', 'p3_path', 'p4_path'] if c in df]
missing = []
for col in cols:
    for value in df[col].dropna().astype(str):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            missing.append((col, str(path)))
            if len(missing) >= 20:
                break
    if len(missing) >= 20:
        break
print('rows:', len(df), 'checked columns:', cols, 'missing shown:', len(missing))
for item in missing:
    print(item)
raise SystemExit(1 if missing else 0)
PY
```

如有缺失文件，重新下载或解压。不要跳过缺失样本，否则消融实验使用的数据不同。

## 5. 下载官方权重

官方权重：

```text
https://huggingface.co/HaoyZhou/bearinguav/tree/main
```

```bash
cd "$PROJECT_ROOT"
hf download HaoyZhou/bearinguav Bearing_UAV.zip \
  --local-dir "$PROJECT_ROOT/_downloads"
unzip -q "$PROJECT_ROOT/_downloads/Bearing_UAV.zip" -d "$PROJECT_ROOT"

find "$PROJECT_ROOT/Bearing_UAV" -maxdepth 2 \
  -type f \( -name best_model.pth -o -name training_configure.json \) -print
```

官方压缩包中的权重是实验 A 基线权重，不是实验 D 权重。D 为公平消融应从头
训练，不传 `--resume`。VGG-16 的 ImageNet 权重会由 torchvision 自动下载到
`$TORCH_HOME`；新增全局模块保持随机初始化。

## 6. 训练前检查

```bash
cd "$PROJECT_ROOT"
python -m unittest discover -s tests -p 'test_experiments_cd.py' -v
python -m cvphr.train.cvphr_train --help
```

当前 A/B/E 结果使用代码中的 `85%/5%/10%, seed=42`，因此 D 也保持该设置。
论文描述的 `70%/20%/10%` 属于另一个复现协议，不能只修改 D。

## 7. 启动完整训练

可以让 D 使用第二张 V100，与 C 并行：

```bash
tmux new -s exp_d
```

在 tmux 中运行：

```bash
cd /data/xuly/cmq/bearinguav
export PYTHONPATH=/data/xuly/cmq/bearinguav:${PYTHONPATH}
export TORCH_HOME=/data/xuly/cmq/bearinguav/.cache/torch
mkdir -p log/c4ma

CUDA_VISIBLE_DEVICES=1 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_Attn \
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
  2>&1 | tee log/c4ma/experiment_d_3d.log
```

这里 `CUDA_VISIBLE_DEVICES=1` 只暴露物理 GPU 1，因此进程内它编号为 0，
`--device_id 0` 是正确写法。

显式配置对应：

| 项目 | 值 |
|---|---:|
| batch size | 16 |
| learning rate | `5e-5` |
| epochs | 100 |
| optimizer | Adam |
| scheduler | ReduceLROnPlateau，patience 3，factor 0.5 |
| weight decay | 0 |
| pose loss | Smooth L1，位置/航向 `0.8/0.2` |
| `lambda_quad` | 0 |
| `lambda_attn` | 0.10 |
| attention temperature | 1.0 |
| auxiliary warmup | 5 epochs |

`factor_bslr=0.5` 对应 `batch_size=16`、`learning_rate=5e-5`。

按 `Ctrl+B`、`D` 返回终端且不停止训练。查看进度：

```bash
tmux attach -t exp_d
tail -f /data/xuly/cmq/bearinguav/log/c4ma/experiment_d_3d.log
```

## 8. 结果和单独评估

输出目录前缀：

```text
results/c4ma/phr5_globalrst_d_...
```

检查保存配置：

```bash
grep -E 'model_class|quad_loss_weight|attention_loss_weight' \
  results/c4ma/phr5_globalrst_d_*/training_configure.json
```

应看到 D 类、`quad_loss_weight=0.0`、`attention_loss_weight=0.10`。
`--flag_test 1` 会在训练结束后自动评估，也可以单独运行：

```bash
CUDA_VISIBLE_DEVICES=1 python -u -m cvphr.test.cvphr_test \
  --rsi_id 96 \
  --n_sample 100 \
  --is_3d 1 \
  --model_class PARCASGM_v5a_GlobalRST_Attn \
  --dataset_class RSBlockDatasetPA_v3q \
  --device_id 0 \
  --bestpth_dir /data/xuly/cmq/bearinguav/results/c4ma/你的实验D目录
```

记录 Recall@1、LSR@15、HSR@15、MLE、MHE。若 D 相比 B 明显退化，说明当前
几何目标或 `0.10` 权重不适合；若 D 接近 B 而 C 退化，则主要问题来自
`L_quad`；若 C、D 单独正常但 E 退化，则应继续检查两项辅助损失的梯度冲突。
