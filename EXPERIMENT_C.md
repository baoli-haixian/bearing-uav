# 实验 C：四 RST 全局融合 + 象限一致性损失

## 1. 实验定义

实验 C 在实验 B 的四 RST 全局上下文模块上，只增加象限一致性损失：

```text
L_C = L_pose + warmup(epoch) * 0.05 * L_quad
L_pose = 0.8 * SmoothL1(position) + 0.2 * SmoothL1(heading)
L_quad = mean(1 - cosine(context_descriptor, original_descriptor.detach()))
```

前 5 个 epoch 的辅助损失比例依次为 `0.2, 0.4, 0.6, 0.8, 1.0`。
实验 C 的 `attention_loss_weight` 固定为 `0`，不使用 `L_attn`。验证损失、
学习率调度和 `best_model.pth` 选择仍只使用 `L_pose`，便于和 A、B、D、E
公平比较。

模型类：`PARCASGM_v5a_GlobalRST_Quad`

## 2. 环境与路径

以下命令以 Linux 服务器目录 `/data/xuly/cmq/bearinguav` 为例，所有缓存和
下载文件也放在 `/data/xuly/cmq/` 下：

```bash
cd /data/xuly/cmq/bearinguav
export PROJECT_ROOT=/data/xuly/cmq/bearinguav
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$PROJECT_ROOT/_downloads"
```

新建环境：

```bash
conda create -n bearing_env python=3.9 -y
conda activate bearing_env

python -m pip install torch==2.1.2 torchvision==0.16.2 \
  torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install "huggingface_hub[cli]"
```

检查环境：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu 0:", torch.cuda.get_device_name(0))
PY
```

如果服务器使用代理，只有在代理地址对服务器本身可达时才设置，例如：

```bash
export http_proxy=http://PROXY_HOST:7890
export https_proxy=http://PROXY_HOST:7890
```

不要直接假设服务器的 `127.0.0.1:7890` 是本地电脑上的代理。

## 3. 获取代码

首次部署官方代码：

```bash
cd /data/xuly/cmq
git clone https://github.com/liukejia121/bearinguav.git
cd /data/xuly/cmq/bearinguav
```

实验 C 必须使用包含以下类的改进代码：

```bash
grep -n "PARCASGM_v5a_GlobalRST_Quad" \
  cvphr/models/posaglreg/models.py
```

若输出为空，当前分支还没有实验 C，不能用实验 B/E 的类名代替。

## 4. 下载和组织数据集

官方数据集地址：

```text
https://huggingface.co/datasets/HaoyZhou/bearinguav/tree/main
```

命令行下载：

```bash
cd "$PROJECT_ROOT"
hf download HaoyZhou/bearinguav Bearing_UAV_90K.zip \
  --repo-type dataset \
  --local-dir "$PROJECT_ROOT/_downloads"

unzip -q "$PROJECT_ROOT/_downloads/Bearing_UAV_90K.zip" \
  -d "$PROJECT_ROOT"
```

最终目录必须是：

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

训练 UAV/3D 实验 C 使用：

```text
Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

检查关键文件和元数据中的图片缺失情况：

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

出现缺失文件时应重新下载或解压数据集，不要在 Dataset 中静默跳过样本，否则
A/B/C/D/E 的测试集合会发生变化。

## 5. 下载官方权重

官方权重地址：

```text
https://huggingface.co/HaoyZhou/bearinguav/tree/main
```

下载和解压：

```bash
cd "$PROJECT_ROOT"
hf download HaoyZhou/bearinguav Bearing_UAV.zip \
  --local-dir "$PROJECT_ROOT/_downloads"
unzip -q "$PROJECT_ROOT/_downloads/Bearing_UAV.zip" -d "$PROJECT_ROOT"
```

检查：

```bash
find "$PROJECT_ROOT/Bearing_UAV" -maxdepth 2 \
  -type f \( -name best_model.pth -o -name training_configure.json \) -print
```

这些是官方 A 基线的最终权重，只用于复核官方结果或导航。实验 C 应从随机初始化
的新增模块开始训练；不要通过 `--resume` 加载 A/B/E 权重，否则不再是公平消融。
VGG-16 的 ImageNet 预训练权重由 torchvision 首次训练时自动下载到 `$TORCH_HOME`。

## 6. 训练前检查

```bash
cd "$PROJECT_ROOT"
python -m unittest discover -s tests -p 'test_experiments_cd.py' -v
python -m cvphr.train.cvphr_train --help
```

当前改进仓库为了和已经完成的 A/B/E 保持公平，数据划分仍为
`85%/5%/10%, seed=42`。论文正文写的是 `70%/20%/10%`；不要只为 C 修改
划分，否则结果不可直接比较。

## 7. 启动完整训练

创建 tmux：

```bash
tmux new -s exp_c
```

在 tmux 中执行：

```bash
cd /data/xuly/cmq/bearinguav
export PYTHONPATH=/data/xuly/cmq/bearinguav:${PYTHONPATH}
export TORCH_HOME=/data/xuly/cmq/bearinguav/.cache/torch
mkdir -p log/c4ma

CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_Quad \
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
  2>&1 | tee log/c4ma/experiment_c_3d.log
```

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
| `lambda_quad` | 0.05 |
| `lambda_attn` | 0 |
| auxiliary warmup | 5 epochs |

`factor_bslr=0.5` 在训练代码中同时得到 `batch_size=32*0.5=16` 和
`learning_rate=1e-4*0.5=5e-5`。

按 `Ctrl+B` 再按 `D` 退出 tmux 但保持训练。重新查看：

```bash
tmux attach -t exp_c
tail -f /data/xuly/cmq/bearinguav/log/c4ma/experiment_c_3d.log
```

## 8. 结果和单独评估

训练结果目录前缀为：

```text
results/c4ma/phr5_globalrst_c_...
```

目录中应包含 `best_model.pth` 和 `training_configure.json`。确认配置：

```bash
grep -E 'model_class|quad_loss_weight|attention_loss_weight' \
  results/c4ma/phr5_globalrst_c_*/training_configure.json
```

应看到 C 类、`0.05` 和 `0.0`。若训练时已经使用 `--flag_test 1`，训练结束会
自动测试。也可单独执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.test.cvphr_test \
  --rsi_id 96 \
  --n_sample 100 \
  --is_3d 1 \
  --model_class PARCASGM_v5a_GlobalRST_Quad \
  --dataset_class RSBlockDatasetPA_v3q \
  --device_id 0 \
  --bestpth_dir /data/xuly/cmq/bearinguav/results/c4ma/你的实验C目录
```

最终记录 Recall@1、LSR@15、HSR@15、MLE、MHE，并与相同数据划分和随机种子
下的 A、B、D、E 比较。
