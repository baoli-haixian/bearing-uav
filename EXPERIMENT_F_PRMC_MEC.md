# F+P-RMC Matching Evidence Calibration (MEC)

This experiment reuses the spatial response already produced by P-RMC to
calibrate the official four-RST cross-attention used by the position head.
The heading prediction and the loss remain unchanged.

## 1. Variants

| Variant | Model class | Position gradient through P-RMC evidence |
| --- | --- | --- |
| M1 | `PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECDetach` | No |
| M2 | `PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECJoint` | Yes |
| M3 | `PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECShuffle` | No; evidence is batch-shuffled |

M3 is a falsification control. It has the same new parameter count as M1 but
does not preserve the correspondence between an input and its P-RMC evidence.

The training objective is identical to F+P-RMC:

```text
L = 0.8 * L_pos + 0.2 * (L_vec + 0.1 * L_dist)
```

## 2. Environment and data

Follow `envset.md` to create the official environment. The 3D dataset expected
by the commands below is:

```text
Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/
└── metadata/metadata.csv
```

For screening, use the completed 100-epoch F+P-RMC `best_model.pth` as the
initial checkpoint. MEC adds only `matching_evidence_calibration.alpha`, which
is initialized to zero. Loading the checkpoint therefore exactly reproduces
the baseline before optimization starts.

## 3. Local diagnostic

The diagnostic compares the learned P-RMC spatial peak with the F position
prediction and reports retrieval, error correlation, and oracle
complementarity.

```bash
python scripts/diagnose_prmc_spatial_evidence.py \
  --code-root /path/to/bearinguav \
  --data-root /path/to/bearinguav-data-root \
  --metadata /path/to/Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv \
  --checkpoint /path/to/f_prmc/best_model.pth \
  --output results/prmc_spatial_evidence_4500.json \
  --limit 4500 \
  --batch-size 8 \
  --temperature 0.05
```

## 4. Fifteen-epoch screening

Run all variants from the same F+P-RMC checkpoint. Keep the split, seed,
optimizer, batch size, and learning rate identical.

### M1: detached evidence

```bash
python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECDetach \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 --rsi_id 96 --n_sample 100 \
  --num_epochs 15 --factor_bslr 0.5 --device_id 0 \
  --gcth none --checkpoint_interval 5 --flag_ckpt 1 --flag_test 1 \
  --init_checkpoint /path/to/f_prmc/best_model.pth
```

### M2: joint evidence

```bash
python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECJoint \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 --rsi_id 96 --n_sample 100 \
  --num_epochs 15 --factor_bslr 0.5 --device_id 0 \
  --gcth none --checkpoint_interval 5 --flag_ckpt 1 --flag_test 1 \
  --init_checkpoint /path/to/f_prmc/best_model.pth
```

### M3: shuffled-evidence control

```bash
python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECShuffle \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 --rsi_id 96 --n_sample 100 \
  --num_epochs 15 --factor_bslr 0.5 --device_id 0 \
  --gcth none --checkpoint_interval 5 --flag_ckpt 1 --flag_test 1 \
  --init_checkpoint /path/to/f_prmc/best_model.pth
```

These commands explicitly use batch size 16 and learning rate `5e-5` through
`factor_bslr=0.5`. The optimizer remains Adam, weight decay remains zero, and
the scheduler remains ReduceLROnPlateau.

## 5. Fair 100-epoch training

Only the best screening variant should be retrained from the official
ImageNet-pretrained VGG-16 initialization. Do not pass `--init_checkpoint`:

```bash
python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECJoint \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 --rsi_id 96 --n_sample 100 \
  --num_epochs 100 --factor_bslr 0.5 --device_id 0 \
  --gcth none --checkpoint_interval 5 --flag_ckpt 1 --flag_test 1
```

Replace `MECJoint` with `MECDetach` if M1 wins screening. Repeat the final run
with three declared random seeds before reporting a paper result.

## 6. Decision rule

Proceed to fair training only if a non-shuffled variant satisfies all of:

- MLE below `7.30 m`;
- Recall@1 improves by at least `0.3` percentage points;
- LSR@15 does not decrease;
- MHE degrades by no more than `0.2` degrees;
- M3 does not reproduce the gain.
