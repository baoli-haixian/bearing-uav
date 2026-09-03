# F+P-RMC Task Adapters

## Purpose

This experiment keeps the shared GLUF, experiment F position path, official CA,
and the parallel 36-bin P-RMC heading head. It adds small residual adapters at
the shared `nl_feat` branch point:

```text
Shared GLUF nl_feat [B,256,16,16]
  |-- Position Adapter (shared by four RSTs)
  |     -> GlobalRST -> PSG -> position head
  |-- Heading RST Adapter (shared by four RSTs) --|
  |-- Heading UAV Adapter ------------------------|-> P-RMC -> heading

Original descriptors -> official CA -> position head
```

Each adapter is:

```text
1x1 Conv 256->64 -> GN -> GELU
-> depthwise 3x3 Conv -> GN -> GELU
-> 1x1 Conv 64->256
-> alpha * residual + input
```

All `alpha` values start at zero, so loading the F+P-RMC checkpoint initially
reproduces the old feature paths exactly. No detach and no new loss are used.

## Loss

The objective is unchanged from F+P-RMC:

```text
L = 0.8 * L_pos + 0.2 * (L_vec + 0.1 * L_dist)
```

`L_pos` is SmoothL1. `L_vec` is SmoothL1 on the normalized heading vector.
`L_dist` is the 36-bin von Mises soft-target cross entropy.

## Preliminary 3D screening run

Use the completed F+P-RMC 3D `best_model.pth` only as initialization. This is a
15-epoch low-learning-rate feasibility screen, not the final fair paper result.

```bash
cd /data/xuly/cmq/experiment_f_prmc_taskadapters/bearinguav

python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_PRMC_TaskAdapters \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 \
  --rsi_id 96 \
  --n_sample 100 \
  --num_epochs 15 \
  --factor_bslr 0.5 \
  --learning_rate 1e-5 \
  --device_id GPU_ID \
  --gcth none \
  --checkpoint_interval 5 \
  --flag_ckpt 1 \
  --flag_test 1 \
  --init_checkpoint /data/xuly/cmq/experiment_f_h1_prmc/bearinguav/results/c4ma/phr5_f_h1_prmc_d96100_3d_b16_l0.5_e100_gNone_20260823_234309/best_model.pth
```

Explicit screening settings:

- input: UAV/3D, four RSTs plus one UAV image;
- batch size: 16 (`factor_bslr=0.5`);
- epochs: 15;
- learning rate: `1e-5`;
- optimizer: Adam;
- scheduler: ReduceLROnPlateau;
- split and seed: unchanged repository defaults (85/5/10, seed 42);
- initialization: F+P-RMC best checkpoint (best validation epoch 96/100);
- trainable path: all parameters that were trainable in F+P-RMC plus adapters;
- VGG backbone remains frozen, matching the baseline configuration.

The epoch log and TensorBoard contain:

```text
adapter_alpha_position
adapter_alpha_heading_rst
adapter_alpha_heading_uav
adapter_ratio_position
adapter_ratio_heading_rst
adapter_ratio_heading_uav
```

## Evaluation criteria

Compare the 15-epoch result with the fixed completed F+P-RMC 3D test result:

| Metric | F+P-RMC 3D baseline |
|---|---:|
| Recall@1 | 86.48% |
| LSR@15 | 92.89% |
| HSR@15 | 85.01% |
| MLE | 7.46 m |
| MHE | 10.29 degrees |

The screen is promising if heading improves without materially degrading MLE,
or position improves while MHE/HSR remain stable. Also verify that at least one
adapter alpha and residual ratio move away from zero.

This comparison is not publication-fair because it adds 15 fine-tuning epochs
after a 100-epoch baseline. If promising, train the baseline and adapter model
from the same initialization for 100 epochs and repeat with three seeds.

## Resume

Use `--resume` only to continue an interrupted adapter run:

```bash
python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_PRMC_TaskAdapters \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 --rsi_id 96 --n_sample 100 \
  --num_epochs 15 --factor_bslr 0.5 --learning_rate 1e-5 \
  --device_id GPU_ID --gcth none --checkpoint_interval 5 \
  --flag_ckpt 1 --flag_test 1 \
  --resume /absolute/path/to/ckpt_latest_model.pth
```

Do not pass `--init_checkpoint` and `--resume` together.
