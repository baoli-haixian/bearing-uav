# F+CDM contribution ablation

The two variants split the completed F+CDM experiment into its two changes:

| Class | CA direction residual | CDM auxiliary heading loss |
| --- | --- | --- |
| `PARCASGM_v5a_GlobalRST_PosPrior_CDMResidualOnly` | yes | no |
| `PARCASGM_v5a_GlobalRST_PosPrior_CDMAuxOnly` | no | yes |

The existing F, F+CDM, and F+CSME classes retain their behavior. Both variants
retain the shared VGG/GLUF, F GlobalRST-to-PSG path, RCE, official CA and pose
heads. The residual-only total loss is `0.8 L_pos + 0.2 L_heading`, where both
terms are SmoothL1. The auxiliary-only loss is
`0.8 L_pos + 0.2 (L_heading + 0.1 L_CDM_aux)`.

Residual-only still computes an auxiliary prediction in forward for checkpoint
compatibility, but skips its loss entirely; the auxiliary head receives no
gradient. Auxiliary-only computes a residual but does not add it to CA; its
residual scale receives no gradient. Its main predictions initially and
throughout the forward pass use exactly the F architecture, but auxiliary
heading gradients can update shared GLUF. Full F+CDM combines both changes.

## Controlled training

Use the same dataset, VGG-16 ImageNet initialization, split 85/5/10 with seed
42, batch size 16 (`factor_bslr=0.5`), learning rate 5e-5, Adam, 100 epochs,
five-epoch checkpoints, and test entry as the completed F+CDM. Do not load
finished F or F+CDM pose checkpoints. Results are selected by existing
validation behavior; compare test metrics and validation curves separately.
The paper states a different 70/20/10 split, so this is a repository-controlled
comparison, not an exact paper-table replication.

From the repository root, a single 3D command is:

```bash
python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_CDMResidualOnly \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 --rsi_id 96 --n_sample 100 \
  --num_epochs 100 --factor_bslr 0.5 \
  --learning_rate 5e-5 --optimizer Adam \
  --device_id 0 --gcth none \
  --checkpoint_interval 5 --flag_ckpt 1 --flag_test 1
```

Replace the model class with `PARCASGM_v5a_GlobalRST_PosPrior_CDMAuxOnly` for
the auxiliary-only run. Set `--is_3d 0` for 2D. To run both views per variant
on the two V100s from sc0, use the scoped scripts:

```bash
cd /data/xuly/cmq/experiment_f_cdm_ablation/bearinguav
/bin/bash scripts/launch_f_cdm_ablation_sc0.sh residual 0
/bin/bash scripts/launch_f_cdm_ablation_sc0.sh auxiliary 1
```

Each script runs 3D first, then checks its own physical GPU again and runs 2D.
If the GPU is occupied at either start, that script stops without touching the
occupying process. sc0 tmux maintains the connection to sc1; the socket and
all experiment outputs remain under `/data/xuly/cmq/experiment_f_cdm_ablation`.
Inspect via:

```bash
cd /data/xuly/cmq/experiment_f_cdm_ablation
/usr/bin/tmux -S runtime/tmux/ablation.sock list-sessions
tail -n 20 runtime/residual/3d/logs/train.log
tail -n 20 runtime/auxiliary/3d/logs/train.log
```

Results are created in the checkout's `results/c4ma/`. Compare Recall@1,
LSR@15, HSR@15, MLE, and MHE against completed F and full F+CDM for each view.
The ablations identify contribution within one setup; repeated seeds are still
needed before making a statistical claim.

## Minimal checks

```bash
python -m unittest tests.test_experiment_f_cdm_ablation tests.test_experiment_f_cdm
```

Synthetic tests verify residual-only does not train the auxiliary head and
auxiliary-only leaves the F prediction path unchanged while training CDM.
