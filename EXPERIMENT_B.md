# Experiment B: Global Context Fusion for Four RSTs

This guide runs the `PARCASGM_v5a_GlobalRST` ablation on Bearing-UAV-90K.
Experiment B changes only the model architecture. Its objective remains:

```text
L = 0.8 * SmoothL1(position) + 0.2 * SmoothL1(heading)
```

No quadrant-consistency or geometry-attention auxiliary loss is enabled.

## 1. Connect to the training machine

The following paths stay under `/data/xuly/cmq/`:

```bash
ssh xuly@10.6.3.45
ssh 12.12.12.1
cd /data/xuly/cmq/bearinguav
```

Check the location before every operation:

```bash
pwd
```

The expected output is:

```text
/data/xuly/cmq/bearinguav
```

## 2. Update the code

The experiment is in commit `afcdeb8` of the personal repository:

```bash
git remote -v
git remote add baoli https://github.com/baoli-haixian/bearing-uav.git
git fetch baoli
git switch main
git pull --ff-only baoli main
git log -1 --oneline
```

If the `baoli` remote already exists, skip `git remote add`. The final command
must show `afcdeb8` or a later commit containing it.

Confirm that the model is registered:

```bash
grep -n "PARCASGM_v5a_GlobalRST" cvphr/models/posaglreg/models.py
```

## 3. Activate the environment

Use the existing environment on the server:

```bash
source ~/.bashrc
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

For a new environment, follow `README.md` and install PyTorch 2.1.2 plus
`requirements.txt`.

## 4. Verify Bearing-UAV-90K

The complete cross-view index used by this experiment must exist at:

```text
/data/xuly/cmq/bearinguav/Bearing_UAV_90K/
  c4m_254k_96bc_b15_s100_v3d/
    metadata/metadata.csv
```

Run these checks from the repository root:

```bash
test -f Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
wc -l Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
python -c "import pandas as pd; p='Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv'; d=pd.read_csv(p); print(d.shape); print(d[['p1_path','p2_path','p3_path','p4_path','target_path']].head(1).T)"
```

The CSV image paths are relative to the repository root, so training must be
launched from `/data/xuly/cmq/bearinguav`.

The released code currently may use the AirSim debug setting. In
`config/base_info.py`, ensure that the active assignment is:

```python
rsi_type = "254k"
```

and that `rsi_type = "40_1024"` is commented out. Verify it with:

```bash
grep -n '^rsi_type' config/base_info.py
```

Before a long run, check that every CSV path exists:

```bash
python -c "import os,pandas as pd; p='Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv'; d=pd.read_csv(p); c=['p1_path','p2_path','p3_path','p4_path','target_path']; m=[x for k in c for x in d[k] if not os.path.isfile(x)]; print('rows=',len(d),'missing=',len(m)); print(*m[:20],sep='\n')"
```

Do not start training unless `missing=0`.

## 5. Run a one-epoch data smoke test

Use GPU 0 and disable automatic testing for this short run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST \
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

This verifies real image decoding, augmentation, eight DataLoader workers,
forward propagation, loss, backward propagation, validation, and checkpoint
writing. A successful run creates a directory beginning with:

```text
results/c4ma/phr5_globalrst_b_d96100_3d_b16_l0.5_e1_gNone_
```

## 6. Start the complete 100-epoch run in tmux

Create a dedicated tmux session:

```bash
tmux new -s bearing_exp_b
cd /data/xuly/cmq/bearinguav
conda activate bearing_env
export PYTHONPATH=/data/xuly/cmq/bearinguav
mkdir -p log/c4ma
```

Start training and keep a separate console log:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST \
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
  2>&1 | tee log/c4ma/experiment_b_100e.log
```

Detach without stopping training with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t bearing_exp_b
```

Useful monitoring commands:

```bash
tmux ls
tail -f /data/xuly/cmq/bearinguav/log/c4ma/experiment_b_100e.log
watch -n 2 nvidia-smi
```

The current trainer is single-GPU. `CUDA_VISIBLE_DEVICES=0` exposes one V100,
and `--device_id 0` selects that visible device. GPU 1 can run the official
baseline as a separate process, but it is not automatically used by Experiment B.

## 7. Parameters used by Experiment B

| Setting | Value | Source |
|---|---:|---|
| Model | `PARCASGM_v5a_GlobalRST` | Experiment B |
| Batch size | 16 | `32 * factor_bslr`, factor `0.5` |
| Learning rate | `5e-5` | `1e-4 * factor_bslr` |
| Epochs | 100 | command line |
| Optimizer | Adam | trainer |
| Weight decay | 0 | Adam default |
| Scheduler | ReduceLROnPlateau | trainer |
| Scheduler factor/patience | `0.5 / 3` | trainer |
| Position/heading weights | `0.8 / 0.2` | `config/base_info.py` |
| Loss | SmoothL1 | trainer |
| Backbone | ImageNet VGG-16 | model registry |
| DataLoader workers | 8 | dataset loader |
| Prefetch factor | 2 | dataset loader |
| Split | 85% / 5% / 10% | current code, seed 42 |

The code split is 85/5/10, while the paper text reports 70/20/10. Experiment B
must first be compared with a baseline trained using the same current code split.
Do not claim an exact paper reproduction from this run alone.

## 8. Results and evaluation

The full result directory starts with:

```text
results/c4ma/phr5_globalrst_b_d96100_3d_b16_l0.5_e100_gNone_
```

Important outputs are:

```text
training_configure.json
best_model.pth
ckpt_latest_model.pth
checkpoints/epoch_0005.pth
checkpoints/epoch_0010.pth
...
tensorboard_logs/
gcheck/training_epoch_log.txt
test_results_*/test_mae.json
test_results_*/test_results*.csv
```

Because `--flag_test 1` is used, the best validation checkpoint is evaluated
automatically after epoch 100. Read the primary metrics with:

```bash
find results/c4ma -path '*phr5_globalrst_b*' -path '*test_results*' -name test_mae.json -print
```

The JSON contains MLE, MHE, Recall@1, LSR and HSR values. Navigation SR, SPL,
and NE require a subsequent Bearing-Naver run using this result directory.

TensorBoard can be started on the server with:

```bash
tensorboard --logdir /data/xuly/cmq/bearinguav/results/c4ma --port 6006
```

Then create an SSH tunnel from the local machine if required.

## 9. Resume an interrupted run

Find the experiment directory and resume from its latest checkpoint:

```bash
find results/c4ma -maxdepth 2 -name ckpt_latest_model.pth -path '*phr5_globalrst_b*' -print
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST \
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
  --resume /data/xuly/cmq/bearinguav/results/c4ma/EXPERIMENT_DIR/ckpt_latest_model.pth
```

Replace `EXPERIMENT_DIR` with the actual directory name. Keep all other
parameters identical to the original run.

## 10. Fair baseline comparison

Run the official architecture on GPU 1 with exactly the same settings, changing
only the model class:

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
  2>&1 | tee log/c4ma/baseline_a_100e.log
```

Compare at least Recall@1, LSR@15, HSR@15, MLE, and MHE. The architecture is
the only intended difference between baseline A and Experiment B.
