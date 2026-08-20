# Experiment P5: MS-RDCP dense position refinement

P5 keeps experiment F unchanged and adds a multi-scale, rotation-marginalized
dense correlation position head. Stage 1 strictly loads an experiment F
checkpoint, freezes every F parameter, and trains only `dense_position_head`.

## Why initialize from F

The VGG/non-local encoder, RST global context fusion, PSG, cross-attention, and
the original position/heading regressors are identical to experiment F. The
checkpoint loader accepts missing keys only under `dense_position_head.*`; any
other missing or unexpected key aborts training.

## Environment

Use the same environment and dataset as experiment F:

```bash
cd /data/xuly/cmq/experiment_p5_msrdcp/bearinguav
source /home/xuly/.conda/envs/bearing_env/bin/activate
```

The project expects:

```text
Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
Bearing_UAV_90K/citya/...
Bearing_UAV_90K/cityb/...
Bearing_UAV_90K/cityc/...
Bearing_UAV_90K/cityd/...
```

On the existing server, the dataset may be linked read-only from
`/data/xuly/cmq/Bearing_UAV_90K` into the experiment directory.

## Stage-1 training

```bash
CUDA_VISIBLE_DEVICES=0 python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_MSRDCP_P5 \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 \
  --rsi_id 96 \
  --n_sample 100 \
  --num_epochs 20 \
  --factor_bslr 0.5 \
  --learning_rate 1e-4 \
  --optimizer AdamW \
  --device_id 0 \
  --gcth none \
  --checkpoint_interval 5 \
  --flag_ckpt 1 \
  --flag_test 1 \
  --init_checkpoint /path/to/experiment_f/best_model.pth
```

Explicit settings are batch size 16, learning rate `1e-4`, 20 epochs, AdamW
with weight decay `1e-4`, and the existing 0.8 position / 0.2 heading task
weights. The position objective is:

```text
L_position = SmoothL1(p_final, p_gt)
           + 0.30 KL(Q_heat || P_heat)
           + 0.20 SmoothL1(p_coarse, p_gt)
           + 0.10 SmoothL1(p_dense, p_gt)
```

`Q_heat` is generated online from the existing normalized `(x, y)` target;
P5 requires no additional dataset labels.

## Evaluation and decision

Training with `--flag_test 1` evaluates `best_model.pth` automatically. Compare
P5 against the same experiment F split using Recall@1, LSR@15, HSR@15, and MLE.
Continue to joint fine-tuning only if P5 improves MLE or Recall@1 without a
material LSR/HSR regression. Joint fine-tuning should start from the P5 best
checkpoint with a lower learning rate (`1e-5`); it is a separate experiment.
