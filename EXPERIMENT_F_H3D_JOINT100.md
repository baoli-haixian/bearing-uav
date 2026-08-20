# F + H3-D Joint-100

This experiment integrates experiment F localization and H3-D heading into one
model and trains the combined model for exactly 100 epochs. It does not load an
experiment F or experiment H3-D checkpoint. VGG-16 uses the same ImageNet
initialization and freezing policy as the official baseline.

## Model

`PARCASGM_v5a_GlobalRST_PosPrior_GPRVH_Joint100` keeps the experiment F path:

- four RST `nl_feat` maps are fused by `RSTGlobalContextFusion`;
- global RST descriptors affect only `SimilarityPositionPrior`;
- official RST descriptors remain on the CA path.

H3-D consumes the four original RST maps and UAV map. Its correlation center is
the predicted F position with stopped coordinate gradient. Shared trainable
non-backbone features still receive gradients from both tasks.

The objective is:

```text
L = 0.8 * SmoothL1(position)
  + 0.2 * heading_scale(epoch) * L_H3D
```

`heading_scale` increases linearly from `0.1` to `1.0` in epochs 1-10 and stays
at `1.0` for epochs 11-100. Validation always uses the final 0.8/0.2 objective.

## Server training

Expected data:

```text
/data/xuly/cmq/Bearing_UAV_90K/
  c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
```

Run on an idle GPU only:

```bash
cd /data/xuly/cmq/experiment_f_h3d_joint100/bearinguav
bash scripts/run_f_h3d_joint100_sc1.sh 0
```

The explicit training command is equivalent to:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_GPRVH_Joint100 \
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

This gives batch size 16, ReduceLROnPlateau, no Adam weight decay, the existing
85/5/10 split, seed 42, `num_workers=8`, and `prefetch_factor=2`.

Do not pass `--init_checkpoint`: doing so changes this into a pre-trained
adaptation experiment and no longer represents Joint-100.

